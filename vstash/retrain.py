"""
retrain.py -- Self-supervised embedding fine-tuning from hybrid retrieval disagreement.

Generates (query, positive) pairs from vector/FTS signal disagreement
in the user's own corpus, then fine-tunes the embedding model using
MultipleNegativesRankingLoss (MNRL). The resulting model produces
embeddings that better distinguish "semantically close" from "actually
relevant" for the user's specific data.

Includes an honest eval-gated training entry (`retrain`) that reindexes
the relevant + noise chunks with each candidate model (baseline and
fine-tuned) and reports NDCG@10 deltas before saving.

Requires: pip install sentence-transformers torch
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .embed import embed_query
from .store import VstashStore

if False:  # TYPE_CHECKING only; avoids import cycle at runtime
    from .config import VstashConfig  # noqa: F401

logger = logging.getLogger(__name__)

TOP_K = 10
_EVAL_TOP_K = 10
_EVAL_NDCG_SHALLOW_K = 3
_EVAL_RECALL_K = 100
_EVAL_MAX_QUERIES = 200
_EVAL_MIN_QUERIES = 20
_EVAL_NOISE_DEFAULT = 1000


def adaptive_rrf_weights(word_count: int) -> tuple[float, float, float, float]:
    """Return ``(vec_hi, fts_hi, vec_lo, fts_lo)`` weights for RRF fusion.

    Short queries (<= 10 words) weight FTS higher because exact-term
    match is informative; long queries (> 50 words) lean hard on vector.
    The mid band (11-50 words) uses a steep ladder for sharper
    disagreement mining. Shared by ``generate_triples`` and the batched
    miner so both signals use byte-identical weights.
    """
    if word_count <= 10:
        return 0.70, 0.30, 0.30, 0.70
    if word_count <= 50:
        return 0.85, 0.15, 0.15, 0.85
    return 0.95, 0.05, 0.50, 0.50


def sample_training_chunks(
    store: VstashStore,
    max_queries: int,
    seed: int = 42,
    exclude_chunk_ids: set[int] | None = None,
) -> list[dict]:
    """Deterministically sample chunks to use as pseudo-queries.

    Returns ``[{id, text, path}, ...]`` in the shuffled training order.
    Extracted so LLM query synthesis can share the exact chunk set
    with ``generate_triples``.
    """
    rng = random.Random(seed)
    excluded = exclude_chunk_ids or set()

    id_rows = store._conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    if not id_rows:
        return []

    ids = [r["id"] for r in id_rows]
    if excluded:
        ids = [i for i in ids if i not in excluded]
    rng.shuffle(ids)
    sample_size = min(max_queries, len(ids))
    sample_ids = ids[:sample_size]
    if not sample_ids:
        return []

    placeholders = ",".join("?" * len(sample_ids))
    fetched = store._conn.execute(
        f"SELECT c.id, c.text, d.path FROM chunks c "
        f"JOIN documents d ON d.id = c.doc_id "
        f"WHERE c.id IN ({placeholders})",
        sample_ids,
    ).fetchall()
    by_id = {r["id"]: {"id": r["id"], "text": r["text"], "path": r["path"]} for r in fetched}
    return [by_id[i] for i in sample_ids if i in by_id]


def generate_triples(
    store: VstashStore,
    model_name: str,
    max_queries: int = 5000,
    seed: int = 42,
    exclude_chunk_ids: set[int] | None = None,
    synthesized_queries: dict[int, list[str]] | None = None,
    pre_sampled_chunks: list[dict] | None = None,
) -> list[dict]:
    """Generate training triples from RRF signal disagreement.

    For each document chunk, uses it as a pseudo-query against the store.
    Identifies cases where vector-heavy and FTS-heavy search disagree on
    the top results, and builds (query, positive) pairs for MNRL training.

    Args:
        store: VstashStore with ingested documents.
        model_name: Embedding model to use for queries.
        max_queries: Maximum number of pseudo-queries to generate.
        seed: Random seed for reproducibility.
        exclude_chunk_ids: Chunk IDs reserved for evaluation; skipped so
            training and eval never overlap.
        synthesized_queries: Optional mapping ``{chunk_id: [query, ...]}``
            produced by ``retrain_synth.synthesize_queries``. When a
            chunk appears in this mapping, its synthesized queries are
            used instead of the first-200-chars prefix, and one triplet
            is emitted per synthesized query. Chunks missing from the
            map fall back to the prefix behavior.

    Returns:
        List of dicts with 'query' and 'positive' keys.
    """
    if pre_sampled_chunks is not None:
        rows = pre_sampled_chunks
    else:
        rows = sample_training_chunks(
            store,
            max_queries=max_queries,
            seed=seed,
            exclude_chunk_ids=exclude_chunk_ids,
        )
    if not rows:
        return []

    synth_map = synthesized_queries or {}
    pairs = []
    disagreements = 0
    total_query_iterations = 0
    synth_used = 0
    prefix_used = 0

    for row in rows:
        doc_path = row["path"]
        # Build the list of queries for this chunk. Synthesized queries
        # take precedence when available; otherwise fall back to the
        # first-200-char prefix (legacy behavior).
        queries_for_row: list[str]
        if row["id"] in synth_map and synth_map[row["id"]]:
            queries_for_row = list(synth_map[row["id"]])
            synth_used += len(queries_for_row)
        else:
            queries_for_row = [row["text"][:200]]
            prefix_used += 1

        # The positive is always the chunk's own text. Fixed per chunk,
        # independent of how many queries map to it.
        own_chunk_text = row["text"]

        for query_text in queries_for_row:
            total_query_iterations += 1
            if not query_text:
                continue

            emb = embed_query(query_text, model_name)

            # Adaptive weights based on query length. Centralised in
            # adaptive_rrf_weights() so the batched miner uses the
            # same ladder.
            vec_hi, fts_hi, vec_lo, fts_lo = adaptive_rrf_weights(len(query_text.split()))

            # Vector-heavy search
            try:
                vec_results = store.search(
                    query_embedding=emb,
                    query_text=query_text,
                    top_k=TOP_K,
                    vec_weight=vec_hi,
                    fts_weight=fts_hi,
                    adaptive_rrf=False,
                )
            except Exception:
                continue

            # FTS-heavy search
            try:
                fts_results = store.search(
                    query_embedding=emb,
                    query_text=query_text,
                    top_k=TOP_K,
                    vec_weight=vec_lo,
                    fts_weight=fts_lo,
                    adaptive_rrf=False,
                )
            except Exception:
                continue

            vec_paths = {r.path for r in vec_results[:5]}
            fts_paths = {r.path for r in fts_results[:5]}

            if vec_paths != fts_paths:
                disagreements += 1

            # Build text lookup from all results
            result_texts: dict[str, str] = {}
            for r in vec_results + fts_results:
                result_texts[r.path] = r.text

            # The document's own chunk is the positive. Prefer the
            # result-side copy (its path is in the index) but fall back
            # to the raw chunk text so synthesized queries whose source
            # chunk does not surface in retrieval still emit a pair.
            positive_text = result_texts.get(doc_path) or own_chunk_text

            if not positive_text or positive_text == query_text:
                continue

            # Hard negatives: chunks in one signal's top-5 but not the other's
            hard_neg_text = None
            for r in vec_results[:5]:
                if r.path not in fts_paths and r.path != doc_path:
                    hard_neg_text = r.text
                    break
            if hard_neg_text is None:
                for r in fts_results[:5]:
                    if r.path not in vec_paths and r.path != doc_path:
                        hard_neg_text = r.text
                        break

            pairs.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negative": hard_neg_text,
                }
            )

    source_desc = (
        f"{synth_used} synthesized + {prefix_used} prefix"
        if synth_used
        else f"{prefix_used} prefix"
    )
    logger.info(
        "Generated %d pairs from %d chunks / %d query iterations (%s; %d disagreements, %.0f%%)",
        len(pairs),
        len(rows),
        total_query_iterations,
        source_desc,
        disagreements,
        disagreements / total_query_iterations * 100 if total_query_iterations else 0,
    )
    return pairs


def train_mnrl(
    pairs: list[dict],
    base_model: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "~/.vstash/models/retrained",
    epochs: int = 2,
    lr: float = 3e-6,
    batch_size: int = 64,
    use_amp: bool = True,
    max_seq_length: int | None = None,
    seed: int = 42,
) -> str:
    """Fine-tune an embedding model using MNRL on disagreement pairs.

    When pairs include a 'negative' key, MNRL uses it as an explicit
    hard negative in addition to the in-batch negatives. This produces
    better embeddings than in-batch negatives alone (+1-2% NDCG).

    Args:
        pairs: List of dicts with 'query', 'positive', and optionally 'negative' keys.
        base_model: HuggingFace model to fine-tune from.
        output_path: Where to save the fine-tuned model.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        use_amp: Enable automatic mixed precision. Roughly halves GPU
            memory and is near-free on modern accelerators; required on
            T4-class GPUs when training bge-small with hard negatives at
            batch_size >= 32. Ignored when running on CPU.
        max_seq_length: Cap the encoder's max sequence length before
            training. ``None`` keeps the model's default (512 for
            bge-small). Lower values (256 or 128) further reduce memory
            when most chunks are short.
        seed: Controls training-time randomness. Seeds
            ``torch.manual_seed`` + ``torch.cuda.manual_seed_all``
            globally (the dropout + optimizer init path), pre-shuffles
            ``examples`` with a ``random.Random(seed)`` so the initial
            order is reproducible, and passes a seeded
            ``torch.Generator`` to ``DataLoader`` so per-epoch
            reshuffles are also reproducible. Two back-to-back
            ``train_mnrl`` calls with the same inputs and seed produce
            identical gradient steps.

    Returns:
        Path to the saved model.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "sentence-transformers, torch, and accelerate are required for "
            "vstash retrain. Install with: "
            "pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'"
        )

    output = str(Path(output_path).expanduser())
    Path(output).mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    examples = []
    for p in pairs:
        if p.get("negative"):
            examples.append(InputExample(texts=[p["query"], p["positive"], p["negative"]]))
        else:
            examples.append(InputExample(texts=[p["query"], p["positive"]]))

    # Reproducibility: seed every RNG training touches. Pre-shuffle the
    # examples with Python's RNG so even the initial order is stable,
    # then hand DataLoader a seeded torch.Generator so per-epoch
    # reshuffles are reproducible too. Global torch.manual_seed covers
    # dropout + optimizer init inside model.fit.
    rng = random.Random(seed)
    rng.shuffle(examples)

    import torch as _torch

    _torch.manual_seed(seed)
    cuda = getattr(_torch, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", lambda: False)():
        cuda.manual_seed_all(seed)

    loader_kwargs: dict = {"batch_size": batch_size, "shuffle": True}
    generator_factory = getattr(_torch, "Generator", None)
    if generator_factory is not None:
        gen = generator_factory()
        # Some generators (CPU/CUDA variants) expose manual_seed; guard
        # defensively so test stubs without the method do not crash.
        seed_method = getattr(gen, "manual_seed", None)
        if callable(seed_method):
            seed_method(seed)
        loader_kwargs["generator"] = gen
    loader = DataLoader(examples, **loader_kwargs)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = min(50, len(loader) // 5)
    logger.info(
        "Training: %d pairs, %d epochs, batch=%d, lr=%s, amp=%s, max_seq_length=%s, seed=%d",
        len(pairs),
        epochs,
        batch_size,
        lr,
        use_amp,
        max_seq_length,
        seed,
    )

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output,
        show_progress_bar=True,
        use_amp=use_amp,
    )
    elapsed = time.perf_counter() - t0

    model.save(output)

    # Save training metadata
    meta = {
        "base_model": base_model,
        "n_pairs": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "use_amp": use_amp,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "training_time_s": round(elapsed, 1),
    }
    (Path(output) / "training_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s (%.0fs)", output, elapsed)
    return output


def _release_gpu_memory() -> None:
    """Free PyTorch CUDA cache + run GC. No-op when torch is not loaded.

    Training + evaluation both encode large corpus slices; on a 15 GB
    T4 the lingering cache can push training over the edge. Call this
    between phases (e.g., after baseline eval, before training) to
    keep peak memory down without touching the library's internal
    references. Defensive against stubbed torch modules in tests.
    """
    import gc

    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def _derive_seed(seed: int, name: str) -> int:
    """Stable, reproducible per-dataset seed from a shared base seed.

    Uses SHA-256 so distinct dataset names produce uncorrelated
    permutations, while the same (seed, name) pair always yields the
    same int. Keeps retrain_multi reproducibility guarantees when
    users rename datasets.
    """
    import hashlib

    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _promote_candidate(candidate_path: Path, final_path: Path) -> None:
    """Move a trained candidate model into its final location atomically.

    The current model (if any) is renamed to ``<name>.old`` first so a
    crash between operations never leaves the user without a working
    model. On success the backup is removed; on failure it is rolled
    back into place.
    """
    backup_path = final_path.with_name(final_path.name + ".old")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        shutil.rmtree(backup_path)
    try:
        if final_path.exists():
            final_path.rename(backup_path)
        shutil.move(str(candidate_path), str(final_path))
    except Exception:
        if backup_path.exists() and not final_path.exists():
            backup_path.rename(final_path)
        raise
    else:
        if backup_path.exists():
            shutil.rmtree(backup_path)


# ---------------------------------------------------------------------- #
# Eval-gated retraining                                                   #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalMetrics:
    """Honest retrieval metrics on a held-out query set.

    ``ndcg_at_10`` stays the headline number driving the retrain gate.
    ``ndcg_at_3`` tracks strict head quality (first three positions)
    and is more sensitive to reranker wins. ``recall_at_100`` measures
    candidate-set health, which is the input a downstream reranker
    (T2.4) actually sees. The two new fields default to ``0.0`` so
    existing positional call sites (``EvalMetrics(0.0, 0.0, 0.0, 0)``)
    keep working unchanged.
    """

    ndcg_at_10: float
    mrr: float
    hit_at_10: float
    n_queries: int
    ndcg_at_3: float = 0.0
    recall_at_100: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrainResult:
    """Outcome of a full eval-gated retrain."""

    output_path: str | None
    baseline: EvalMetrics | None
    final: EvalMetrics | None
    n_pairs: int
    gated_out: bool
    min_gain: float

    @property
    def delta_ndcg(self) -> float:
        if self.baseline is None or self.final is None:
            return 0.0
        return self.final.ndcg_at_10 - self.baseline.ndcg_at_10


def split_corpus_for_eval(
    store: VstashStore,
    eval_fraction: float = 0.15,
    min_queries: int = _EVAL_MIN_QUERIES,
    max_queries: int = _EVAL_MAX_QUERIES,
    seed: int = 42,
) -> tuple[set[int], list[dict]]:
    """Reserve a subset of chunks as held-out pseudo-queries for evaluation.

    Each held-out chunk becomes an eval query: first 200 chars as query
    text, the chunk's own document path as the single relevant target.

    This is a fallback for stores without labeled queries. On diverse
    corpora it saturates at NDCG=1.0 because the first 200 chars are a
    near-perfect cue for their own doc. Prefer passing external
    labeled queries (BEIR qrels, user-annotated pairs) via
    ``retrain(..., eval_queries=...)`` when available.

    Args:
        store: VstashStore with ingested documents.
        eval_fraction: Fraction of chunks to reserve.
        min_queries: Minimum eval set size. Returns empty if the corpus
            cannot provide at least this many queries.
        max_queries: Cap on eval size. Large corpora are capped so eval
            cost stays predictable.
        seed: Deterministic split.

    Returns:
        Tuple of (reserved_chunk_ids, eval_queries). Each eval query is
        a dict with 'query', 'relevant_paths' (list with one entry),
        'source_chunk_id'. ``relevant_path`` (singular) is still
        populated for legacy callers.
    """
    # Fetch ids + paths first in stable order (ints + short strings);
    # shuffle with a seeded Python RNG so the split is deterministic
    # across SQLite versions. We only pay the full-text read for the
    # final held-out set, not the whole corpus.
    id_rows = store._conn.execute(
        "SELECT c.id AS id, d.path AS path FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "ORDER BY c.id",
    ).fetchall()
    if len(id_rows) < min_queries:
        return set(), []

    rng = random.Random(seed)
    shuffled = list(id_rows)
    rng.shuffle(shuffled)

    target = max(min_queries, min(max_queries, int(len(shuffled) * eval_fraction)))
    target = min(target, len(shuffled))

    reserved = shuffled[:target]
    reserved_ids = {r["id"] for r in reserved}

    placeholders = ",".join("?" * len(reserved))
    text_rows = store._conn.execute(
        f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
        list(reserved_ids),
    ).fetchall()
    text_by_id = {r["id"]: r["text"] for r in text_rows}

    eval_queries = [
        {
            "query": text_by_id.get(r["id"], "")[:200],
            "relevant_paths": [r["path"]],
            "relevant_path": r["path"],  # legacy single-relevant field
            "source_chunk_id": r["id"],
        }
        for r in reserved
    ]
    return reserved_ids, eval_queries


def qrels_to_eval_queries(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int | float]],
    path_for_doc_id: "callable | None" = None,
    min_relevance: float = 1.0,
) -> list[dict]:
    """Convert BEIR-style (queries, qrels) to the eval_queries format
    expected by ``retrain(..., eval_queries=...)`` / ``evaluate_model``.

    Args:
        queries: Mapping query_id -> query text.
        qrels: Mapping query_id -> {doc_id: relevance}. Any doc with
            ``relevance >= min_relevance`` is treated as relevant.
        path_for_doc_id: Optional function mapping a BEIR doc_id to the
            ``path`` used when the doc was ingested into the vstash
            store. Defaults to ``lambda d: d`` -- use a matching format
            at ingest time (e.g., ``path=f"scifact://{doc_id}"``) so
            the two sides line up.
        min_relevance: Threshold for treating a qrel entry as relevant
            (binary). Defaults to 1.0 (BEIR convention).

    Returns:
        List of ``{query, relevant_paths, query_id}`` dicts. Queries
        whose qrels have no docs above threshold are dropped.
    """
    if path_for_doc_id is None:

        def path_for_doc_id(doc_id: str) -> str:
            return doc_id

    out: list[dict] = []
    for qid, text in queries.items():
        rel = qrels.get(qid) or {}
        relevant_paths = [
            path_for_doc_id(doc_id) for doc_id, score in rel.items() if score >= min_relevance
        ]
        if not relevant_paths:
            continue
        out.append(
            {
                "query": text,
                "relevant_paths": relevant_paths,
                "query_id": qid,
            }
        )
    return out


def _sample_noise_chunks(
    store: VstashStore,
    relevant_paths: set[str],
    n: int,
    seed: int,
) -> list[dict]:
    """Sample chunks from documents that are NOT relevant to any eval query.

    These distractor chunks make NDCG meaningful: without real competition
    a 10-query search on 10 chunks is trivially perfect.

    Strategy: push the NOT IN filter to SQL so we never materialise the
    whole chunks table in Python. Then fetch ids only, shuffle with a
    seeded RNG for determinism, and pull the full rows for just the
    sample.
    """
    if n <= 0:
        return []

    if relevant_paths:
        placeholders = ",".join("?" * len(relevant_paths))
        id_rows = store._conn.execute(
            f"SELECT c.id FROM chunks c "
            f"JOIN documents d ON d.id = c.doc_id "
            f"WHERE d.path NOT IN ({placeholders}) "
            f"ORDER BY c.id",
            list(relevant_paths),
        ).fetchall()
    else:
        id_rows = store._conn.execute(
            "SELECT id FROM chunks ORDER BY id",
        ).fetchall()

    if not id_rows:
        return []

    ids = [r["id"] for r in id_rows]
    rng = random.Random(seed)
    rng.shuffle(ids)
    sample_ids = ids[:n]
    if not sample_ids:
        return []

    ph = ",".join("?" * len(sample_ids))
    rows = store._conn.execute(
        f"SELECT c.id, c.text, d.path, d.title FROM chunks c "
        f"JOIN documents d ON d.id = c.doc_id "
        f"WHERE c.id IN ({ph})",
        sample_ids,
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in sample_ids if i in by_id]


def _relevant_chunks(store: VstashStore, relevant_paths: set[str]) -> list[dict]:
    """Return every chunk from the documents that are eval-relevant.

    We include the full document, not just the originating chunk, so the
    eval index mirrors how retrieval works in production: any chunk of the
    relevant doc counts as a hit.
    """
    if not relevant_paths:
        return []
    placeholders = ",".join("?" * len(relevant_paths))
    rows = store._conn.execute(
        f"SELECT c.id, c.text, d.path, d.title FROM chunks c "
        f"JOIN documents d ON d.id = c.doc_id "
        f"WHERE d.path IN ({placeholders})",
        list(relevant_paths),
    ).fetchall()
    return [dict(r) for r in rows]


def _ndcg_at_k(rank: int | None, k: int = _EVAL_TOP_K) -> float:
    """Binary NDCG@k for a single relevant doc.

    Returns 1 / log2(rank + 1) when the relevant doc is at 1 <= rank <= k,
    else 0.0. IDCG = 1.0 (one relevant doc at rank 1).
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _ndcg_from_ranks(
    ranks_in_topk: list[int],
    num_relevant: int,
    k: int = _EVAL_TOP_K,
) -> float:
    """Binary NDCG@k for queries with any number of relevant docs.

    Args:
        ranks_in_topk: 1-indexed ranks (<= k) at which relevant docs
            actually appear in the result list. Unordered.
        num_relevant: Total number of relevant docs for this query.
            Used for IDCG (the denominator counts perfect placements
            only up to min(num_relevant, k)).
        k: Cutoff.

    Returns:
        NDCG@k in [0, 1].
    """
    if num_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks_in_topk if 1 <= r <= k)
    ideal_hits = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _recall_at_k(
    ranks_in_topk: list[int],
    num_relevant: int,
    k: int = _EVAL_RECALL_K,
) -> float:
    """Fraction of relevant docs that appear in the top-k ranked list.

    Args:
        ranks_in_topk: 1-indexed ranks at which relevant docs appeared
            in the full ranked list. Ranks beyond ``k`` are ignored.
        num_relevant: Total number of relevant docs for this query. The
            denominator so per-query recall stays comparable across
            queries with different relevant-doc counts.
        k: Cutoff.

    Returns:
        Recall@k in [0, 1].
    """
    if num_relevant <= 0:
        return 0.0
    # Clamp by ``num_relevant`` so callers that pass duplicate ranks
    # (e.g., chunk-level lists where the same doc appears twice) cannot
    # push recall above 1.0. The doc-level callers in this module
    # already dedupe, but _recall_at_k is a public helper and defensive
    # clamping is cheap.
    hits = min(num_relevant, sum(1 for r in ranks_in_topk if 1 <= r <= k))
    return hits / num_relevant


def _load_sentence_transformer(model_name: str):
    """Import + load a SentenceTransformer. Raises ImportError with install hint."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for eval. Install with: "
            "pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'"
        ) from exc
    return SentenceTransformer(model_name)


def evaluate_model(
    base_store: VstashStore,
    model_name_or_path: str,
    eval_queries: list[dict],
    noise_sample_size: int = _EVAL_NOISE_DEFAULT,
    seed: int = 42,
    tmp_dir: Path | None = None,
) -> EvalMetrics:
    """Build a temp index with ``model_name_or_path`` and score it on
    ``eval_queries``. Re-embeds every relevant + noise chunk with the
    target model, so baseline and fine-tuned numbers are directly
    comparable.

    Args:
        base_store: The original VstashStore (source of chunks).
        model_name_or_path: HF hub name or local path to a sentence-
            transformers model.
        eval_queries: Output of ``split_corpus_for_eval``.
        noise_sample_size: Non-relevant chunks added as distractors.
        seed: Deterministic noise sample.
        tmp_dir: Where to create the temp db. Defaults to system temp.

    Returns:
        EvalMetrics with NDCG@10, NDCG@3, MRR, Hit@10, Recall@100.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    if not eval_queries:
        return EvalMetrics(ndcg_at_10=0.0, mrr=0.0, hit_at_10=0.0, n_queries=0)

    # Normalize each query's relevant_paths into a list[str]. Accept the
    # legacy 'relevant_path' (str) as a single-element list.
    normalized_queries: list[dict] = []
    for q in eval_queries:
        paths = q.get("relevant_paths")
        if paths is None:
            legacy = q.get("relevant_path")
            paths = [legacy] if legacy is not None else []
        normalized_queries.append({**q, "relevant_paths": list(paths)})

    model = _load_sentence_transformer(model_name_or_path)
    # sentence-transformers renamed get_sentence_embedding_dimension ->
    # get_embedding_dimension in v5.x. Support both for compatibility.
    if hasattr(model, "get_embedding_dimension"):
        dim = int(model.get_embedding_dimension())
    else:
        dim = int(model.get_sentence_embedding_dimension())

    relevant_paths: set[str] = set()
    for q in normalized_queries:
        relevant_paths.update(q["relevant_paths"])
    relevant_rows = _relevant_chunks(base_store, relevant_paths)
    noise_rows = _sample_noise_chunks(base_store, relevant_paths, noise_sample_size, seed)
    all_rows = relevant_rows + noise_rows

    if not relevant_rows:
        return EvalMetrics(ndcg_at_10=0.0, mrr=0.0, hit_at_10=0.0, n_queries=0)

    # Embed all corpus chunks with the target model.
    texts = [r["text"] for r in all_rows]
    t0 = time.perf_counter()
    corpus_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info(
        "Embedded %d eval corpus chunks with %s (%.1fs)",
        len(texts),
        model_name_or_path,
        time.perf_counter() - t0,
    )

    # Build a fresh temp store with the matching dim. Initialise the
    # cleanup targets up front so the finally block cannot raise
    # UnboundLocalError if the constructor or path math throws.
    eval_store: VstashStore | None = None
    tmp_db: Path | None = None
    try:
        tmp_parent = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_db = (
            tmp_parent / f"retrain_eval_{int(time.time() * 1000)}_{random.randint(0, 1 << 30)}.db"
        )

        eval_store = VstashStore(str(tmp_db), embedding_dim=dim)

        # Group chunks by document path, then batch-ingest all docs in a
        # single transaction via add_documents_batch (much faster than a
        # per-doc loop, which commits once per document).
        by_path: dict[str, dict] = {}
        idx_by_path: dict[str, list[int]] = {}
        for i, row in enumerate(all_rows):
            path = row["path"]
            if path not in by_path:
                by_path[path] = {"title": row.get("title") or path}
                idx_by_path[path] = []
            idx_by_path[path].append(i)

        docs_to_add = []
        for path, doc_meta in by_path.items():
            idxs = idx_by_path[path]
            docs_to_add.append(
                {
                    "path": path,
                    "title": doc_meta["title"],
                    "chunks": [all_rows[i]["text"] for i in idxs],
                    "embeddings": [list(map(float, corpus_vecs[i])) for i in idxs],
                    "source_type": "file",
                }
            )
        eval_store.add_documents_batch(docs_to_add)

        # Batch-encode every eval query in one shot instead of per-query.
        query_texts = [q["query"] for q in normalized_queries]
        q_vecs = model.encode(
            query_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Score each query via production search (RRF with adaptive
        # weights). Pull top-100 so we can compute Recall@100 in the
        # same pass; NDCG@10 / NDCG@3 / MRR / Hit@10 truncate internally
        # via the rank cutoff in their helpers.
        #
        # NDCG and Recall are *document-level* metrics, but
        # ``eval_store.search`` returns chunk-level SearchResults. Long
        # relevant documents can legitimately produce multiple chunks in
        # the top-k (MMR penalises intra-doc duplication but does not
        # guarantee one-per-path). Counting the same relevant doc
        # twice would inflate NDCG above 1.0 and break Recall
        # calibration. Collapse the ranking to unique paths in
        # first-occurrence order before scoring, matching the
        # doc-level dedup in ``evaluate_model_batched``.
        ndcgs_10: list[float] = []
        ndcgs_3: list[float] = []
        mrrs: list[float] = []
        hits: list[float] = []
        recalls_100: list[float] = []
        for q, q_vec in zip(normalized_queries, q_vecs):
            results = eval_store.search(
                query_embedding=list(map(float, q_vec)),
                query_text=q["query"],
                top_k=_EVAL_RECALL_K,
            )
            relevant_set = set(q["relevant_paths"])

            unique_paths: list[str] = []
            seen_paths: set[str] = set()
            for r in results:
                if r.path in seen_paths:
                    continue
                seen_paths.add(r.path)
                unique_paths.append(r.path)

            ranks_hit: list[int] = []
            first_rank: int | None = None
            for rank_i, path in enumerate(unique_paths, start=1):
                if path in relevant_set:
                    ranks_hit.append(rank_i)
                    if first_rank is None and rank_i <= _EVAL_TOP_K:
                        first_rank = rank_i
            num_rel = len(relevant_set)
            ndcgs_10.append(_ndcg_from_ranks(ranks_hit, num_relevant=num_rel, k=_EVAL_TOP_K))
            ndcgs_3.append(
                _ndcg_from_ranks(ranks_hit, num_relevant=num_rel, k=_EVAL_NDCG_SHALLOW_K)
            )
            mrrs.append(1.0 / first_rank if first_rank is not None else 0.0)
            hits.append(1.0 if first_rank is not None else 0.0)
            recalls_100.append(_recall_at_k(ranks_hit, num_relevant=num_rel, k=_EVAL_RECALL_K))

        n = len(normalized_queries)
        return EvalMetrics(
            ndcg_at_10=sum(ndcgs_10) / n,
            mrr=sum(mrrs) / n,
            hit_at_10=sum(hits) / n,
            n_queries=n,
            ndcg_at_3=sum(ndcgs_3) / n,
            recall_at_100=sum(recalls_100) / n,
        )
    finally:
        if eval_store is not None:
            try:
                eval_store.close()
            except Exception:
                pass
        if tmp_db is not None:
            for suffix in ("", "-wal", "-shm"):
                try:
                    (tmp_db.parent / (tmp_db.name + suffix)).unlink(missing_ok=True)
                except Exception:
                    pass


def retrain(
    store: VstashStore,
    base_model: str,
    output_path: str = "~/.vstash/models/retrained",
    max_queries: int = 5000,
    epochs: int = 2,
    lr: float = 3e-6,
    batch_size: int = 64,
    use_amp: bool = True,
    max_seq_length: int | None = None,
    eval_fraction: float = 0.15,
    eval_noise_size: int = _EVAL_NOISE_DEFAULT,
    min_gain: float = 0.0,
    skip_eval: bool = False,
    seed: int = 42,
    eval_queries: list[dict] | None = None,
    synthesize_queries: bool = False,
    synth_n: int = 2,
    synth_cache: str | Path | None = None,
    synth_model: str | None = None,
    cfg: "VstashConfig | None" = None,
) -> RetrainResult:
    """Full eval-gated retrain pipeline.

    Splits the corpus into train and held-out eval sets, measures the
    base model's NDCG@10 on eval, generates training triples from the
    train set, fine-tunes, measures the fine-tuned model, and only
    commits the model to ``output_path`` if the NDCG delta meets
    ``min_gain``.

    Args:
        store: The corpus to train on.
        base_model: HF model name to fine-tune from.
        output_path: Final save location.
        max_queries: Max pseudo-queries for triple generation.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        eval_fraction: Fraction of corpus reserved for held-out eval.
            Ignored when ``eval_queries`` is provided.
        eval_noise_size: Distractor chunks added to the eval index.
        min_gain: Required NDCG@10 improvement (0.0 = no regression).
            Pass a negative value to always save.
        skip_eval: If True, skip eval entirely and save unconditionally.
        seed: Deterministic split + sampling.
        eval_queries: Optional externally provided eval set (e.g., BEIR
            qrels converted via ``qrels_to_eval_queries``). When passed,
            the internal pseudo-query split is skipped and these queries
            are used for both baseline and final eval. Training
            pseudo-queries are still derived from the full store (no
            exclusion), which is correct because external eval docs
            typically have no chunk overlap with training.

    Returns:
        RetrainResult with baseline, final, delta, and final path.
        ``output_path`` is None when training was gated out or skipped.
    """
    final_path_str = str(Path(output_path).expanduser())

    if skip_eval:
        training_chunks = sample_training_chunks(store, max_queries=max_queries, seed=seed)
        synth_map: dict[int, list[str]] = {}
        if synthesize_queries and training_chunks:
            if cfg is None:
                raise ValueError("synthesize_queries=True requires the ``cfg`` argument.")
            from .retrain_synth import synthesize_queries as _synth_queries

            synth_map = _synth_queries(
                training_chunks,
                cfg=cfg,
                n_per_chunk=synth_n,
                cache_path=synth_cache,
                model=synth_model,
            )
        pairs = generate_triples(
            store,
            base_model,
            max_queries=max_queries,
            seed=seed,
            synthesized_queries=synth_map or None,
            pre_sampled_chunks=training_chunks,
        )
        if not pairs:
            return RetrainResult(
                output_path=None,
                baseline=None,
                final=None,
                n_pairs=0,
                gated_out=False,
                min_gain=min_gain,
            )
        train_mnrl(
            pairs,
            base_model=base_model,
            output_path=final_path_str,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            use_amp=use_amp,
            max_seq_length=max_seq_length,
            seed=seed,
        )
        return RetrainResult(
            output_path=final_path_str,
            baseline=None,
            final=None,
            n_pairs=len(pairs),
            gated_out=False,
            min_gain=min_gain,
        )

    if eval_queries is not None:
        # Externally labeled queries: skip the internal split. Training
        # excludes no chunks -- labeled queries live outside the corpus.
        effective_queries = eval_queries
        reserved_ids: set[int] = set()
    else:
        reserved_ids, effective_queries = split_corpus_for_eval(
            store, eval_fraction=eval_fraction, seed=seed
        )
    if len(effective_queries) < _EVAL_MIN_QUERIES:
        logger.warning(
            "Only %d eval queries available (need >= %d). Running without eval gate.",
            len(effective_queries),
            _EVAL_MIN_QUERIES,
        )
        return retrain(
            store,
            base_model=base_model,
            output_path=output_path,
            max_queries=max_queries,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            eval_fraction=eval_fraction,
            eval_noise_size=eval_noise_size,
            min_gain=min_gain,
            skip_eval=True,
            seed=seed,
            synthesize_queries=synthesize_queries,
            synth_n=synth_n,
            synth_cache=synth_cache,
            synth_model=synth_model,
            cfg=cfg,
        )

    logger.info("Running baseline eval on %d held-out queries ...", len(effective_queries))
    baseline = evaluate_model(
        store,
        model_name_or_path=base_model,
        eval_queries=effective_queries,
        noise_sample_size=eval_noise_size,
        seed=seed,
    )

    # Sample training chunks once, then optionally synthesize queries on
    # that exact set so generate_triples and the LLM operate on the same
    # chunk population (no wasted LLM calls).
    training_chunks = sample_training_chunks(
        store,
        max_queries=max_queries,
        seed=seed,
        exclude_chunk_ids=reserved_ids,
    )

    synth_map: dict[int, list[str]] = {}
    if synthesize_queries and training_chunks:
        if cfg is None:
            raise ValueError(
                "synthesize_queries=True requires the ``cfg`` argument "
                "(a VstashConfig) so the LLM backend can be resolved."
            )
        from .retrain_synth import synthesize_queries as _synth_queries

        logger.info(
            "Synthesizing %d queries per chunk across %d chunks ...",
            synth_n,
            len(training_chunks),
        )
        synth_map = _synth_queries(
            training_chunks,
            cfg=cfg,
            n_per_chunk=synth_n,
            cache_path=synth_cache,
            model=synth_model,
        )
        logger.info(
            "Synthesis done: %d of %d chunks produced queries.",
            len(synth_map),
            len(training_chunks),
        )

    pairs = generate_triples(
        store,
        base_model,
        max_queries=max_queries,
        seed=seed,
        exclude_chunk_ids=reserved_ids,
        synthesized_queries=synth_map or None,
        pre_sampled_chunks=training_chunks,
    )
    if len(pairs) < 10:
        logger.warning(
            "Only %d training pairs generated (need >= 10). Skipping training.", len(pairs)
        )
        # Treat as gated_out so callers (CLI, scripts) don't mistake a
        # skipped training for a successful save.
        return RetrainResult(
            output_path=None,
            baseline=baseline,
            final=None,
            n_pairs=len(pairs),
            gated_out=True,
            min_gain=min_gain,
        )

    # Train to a .candidate path; promote to output_path only if gate passes.
    candidate_path = Path(final_path_str + ".candidate")
    if candidate_path.exists():
        shutil.rmtree(candidate_path)

    # Free the baseline-eval encoder's GPU state before loading the
    # training model; on a 15 GB T4 this prevents peak-memory overlap.
    _release_gpu_memory()

    train_mnrl(
        pairs,
        base_model=base_model,
        output_path=str(candidate_path),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_amp=use_amp,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    _release_gpu_memory()

    logger.info("Running final eval on fine-tuned candidate ...")
    final = evaluate_model(
        store,
        model_name_or_path=str(candidate_path),
        eval_queries=effective_queries,
        noise_sample_size=eval_noise_size,
        seed=seed,
    )

    delta = final.ndcg_at_10 - baseline.ndcg_at_10
    gated_out = delta < min_gain

    # Persist eval numbers into training_meta.json regardless of gate
    # outcome, so the candidate directory is useful for debugging.
    meta_path = candidate_path / "training_meta.json"
    try:
        existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:
        existing_meta = {}
    existing_meta["eval"] = {
        "baseline": baseline.as_dict(),
        "final": final.as_dict(),
        "delta_ndcg_at_10": round(delta, 5),
        "min_gain": min_gain,
        "gated_out": gated_out,
        "seed": seed,
        "eval_noise_size": eval_noise_size,
    }
    meta_path.write_text(json.dumps(existing_meta, indent=2))

    if gated_out:
        logger.warning(
            "Candidate gated out: delta NDCG@10 = %+.4f < min_gain=%+.4f. "
            "Candidate left at %s for inspection.",
            delta,
            min_gain,
            candidate_path,
        )
        return RetrainResult(
            output_path=None,
            baseline=baseline,
            final=final,
            n_pairs=len(pairs),
            gated_out=True,
            min_gain=min_gain,
        )

    # Atomic promotion with last-known-good backup (keeps the user's
    # previous model available if anything below fails).
    final_path = Path(final_path_str)
    _promote_candidate(candidate_path, final_path)

    logger.info(
        "Retrain saved: delta NDCG@10 = %+.4f (baseline=%.4f, final=%.4f)",
        delta,
        baseline.ndcg_at_10,
        final.ndcg_at_10,
    )
    return RetrainResult(
        output_path=str(final_path),
        baseline=baseline,
        final=final,
        n_pairs=len(pairs),
        gated_out=False,
        min_gain=min_gain,
    )


# ---------------------------------------------------------------------- #
# Multi-corpus retraining (T1.4)                                          #
# ---------------------------------------------------------------------- #


SamplingStrategy = Literal["uniform", "proportional", "temperature"]
TrainingPairSource = Literal["auto", "labeled", "prefix"]


def compute_triple_budget(
    sizes: dict[str, int],
    total_triples: int,
    strategy: SamplingStrategy = "temperature",
    temperature: float = 0.5,
) -> dict[str, int]:
    """Allocate ``total_triples`` across datasets by the given strategy.

    Strategies (``|D_d|`` is the chunk count of dataset ``d``):

    - ``uniform``: every dataset gets the same share (weights=1).
    - ``proportional``: weights=``|D_d|``, ie naive concatenation.
    - ``temperature``: weights=``|D_d|^temperature``. ``temperature=0``
      collapses to uniform; ``temperature=1`` to proportional;
      ``temperature=0.5`` (the default) is a common middle ground that
      favours smaller corpora without abandoning size signal.

    The returned budget always sums exactly to ``total_triples``. The
    rounding remainder is distributed using the largest-remainder
    method, so per-dataset budgets never differ from the ideal
    fractional allocation by more than 1.
    """
    if not sizes:
        return {}
    if total_triples <= 0:
        return {name: 0 for name in sizes}

    if strategy == "uniform":
        weights = {name: 1.0 for name in sizes}
    elif strategy == "proportional":
        weights = {name: float(n) for name, n in sizes.items()}
    elif strategy == "temperature":
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature!r}")
        weights = {name: float(n) ** temperature for name, n in sizes.items()}
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy!r}. "
            "Expected one of: uniform, proportional, temperature."
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {name: 0 for name in sizes}

    fractions = {name: weights[name] / total_weight * total_triples for name in sizes}
    floors = {name: int(fractions[name]) for name in sizes}
    remainder = total_triples - sum(floors.values())
    if remainder > 0:
        # Largest-remainder: datasets whose fractional share is furthest
        # above their floor get the extra triples first.
        leftovers = sorted(
            sizes.keys(),
            key=lambda name: (fractions[name] - floors[name], name),
            reverse=True,
        )
        for i in range(remainder):
            floors[leftovers[i % len(leftovers)]] += 1
    return floors


def _count_chunks(store: VstashStore) -> int:
    row = store._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
    return int(row["n"]) if row is not None else 0


@dataclass(frozen=True)
class MultiRetrainResult:
    """Outcome of a full multi-corpus eval-gated retrain."""

    output_path: str | None
    per_dataset_baseline: dict[str, EvalMetrics] = field(default_factory=dict)
    per_dataset_final: dict[str, EvalMetrics] = field(default_factory=dict)
    macro_baseline_ndcg: float = 0.0
    macro_final_ndcg: float = 0.0
    per_dataset_pairs: dict[str, int] = field(default_factory=dict)
    per_dataset_budget: dict[str, int] = field(default_factory=dict)
    total_pairs: int = 0
    gated_out: bool = False
    min_gain: float = 0.0
    sampling: SamplingStrategy = "temperature"
    temperature: float = 0.5
    per_dataset_gate: bool = False

    @property
    def macro_delta_ndcg(self) -> float:
        return self.macro_final_ndcg - self.macro_baseline_ndcg

    @property
    def per_dataset_delta(self) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for name, base in self.per_dataset_baseline.items():
            final = self.per_dataset_final.get(name)
            if final is None:
                continue
            deltas[name] = final.ndcg_at_10 - base.ndcg_at_10
        return deltas


def _normalize_stores(
    stores: dict[str, VstashStore] | list[VstashStore],
) -> dict[str, VstashStore]:
    if isinstance(stores, dict):
        if not stores:
            raise ValueError("retrain_multi requires at least one store.")
        return dict(stores)
    if not stores:
        raise ValueError("retrain_multi requires at least one store.")
    # Preserve order while giving each store a predictable alias so
    # per-dataset metrics round-trip through JSON meta.
    return {f"corpus_{i}": s for i, s in enumerate(stores)}


def retrain_multi(
    stores: dict[str, VstashStore] | list[VstashStore],
    base_model: str,
    output_path: str = "~/.vstash/models/retrained-multi",
    sampling: SamplingStrategy = "temperature",
    temperature: float = 0.5,
    total_triples: int = 10000,
    epochs: int = 2,
    lr: float = 3e-6,
    batch_size: int = 32,
    use_amp: bool = True,
    max_seq_length: int | None = None,
    eval_fraction: float = 0.15,
    eval_noise_size: int = _EVAL_NOISE_DEFAULT,
    min_gain: float = 0.0,
    per_dataset_gate: bool = False,
    skip_eval: bool = False,
    seed: int = 42,
    eval_queries_by_dataset: dict[str, list[dict]] | None = None,
    synthesize_queries: bool = False,
    synth_n: int = 2,
    synth_cache: str | Path | None = None,
    synth_model: str | None = None,
    bulk_mine: bool = False,
    bulk_mine_device: str | None = None,
    bulk_eval: bool = False,
    training_queries_by_dataset: dict[str, list[dict]] | None = None,
    training_pair_source: TrainingPairSource = "auto",
    cfg: "VstashConfig | None" = None,
) -> MultiRetrainResult:
    """Fine-tune an embedding model over N corpora with balanced sampling.

    Extracts the v5-notebook training flow that produced
    ``Stffens/bge-small-rrf-v2`` into a first-class API: compute a
    per-dataset triple budget via ``compute_triple_budget``, generate
    triples from each store with shared disagreement mining, globally
    shuffle, train one model on the union, and evaluate per-dataset.

    Args:
        stores: Mapping ``{alias: VstashStore}`` or a list (aliases
            auto-assigned as ``corpus_0`` etc).
        base_model: HF model name / local path to fine-tune.
        output_path: Final save location. ``<path>.candidate`` is used
            during training; ``<path>.old`` is the last-known-good
            rollback target during atomic promote.
        sampling: ``"uniform" | "proportional" | "temperature"``.
        temperature: Exponent for the temperature strategy
            (``|D_d|^alpha``). ``0`` = uniform, ``1`` = proportional.
        total_triples: Target total pseudo-query budget across all
            datasets. Actual triple count may be lower when individual
            corpora are smaller than their per-dataset share or when
            generate_triples drops degenerate pairs.
        batch_size: Training batch size. Default 32 (vs 64 for the
            single-store retrain) keeps peak memory inside a 15 GB T4
            when three corpora are live during eval.
        use_amp: Automatic mixed precision. Default on; halves GPU
            memory on T4 and higher and is near-free.
        max_seq_length: Optional encoder sequence cap. Lower values
            reduce memory further when most chunks are short.
        bulk_mine: Route per-dataset triple generation through
            ``retrain_batch.generate_triples_batched``. Batch-encodes
            queries + corpus on GPU and replaces the per-query
            sqlite-vec full scan with a single ``Q @ D.T`` matmul. On
            a FiQA-sized corpus (~57k chunks) this is 20-50x faster
            than the default per-query path, making the published
            regression target reproducible on a Colab T4.
        bulk_mine_device: ``"cuda" | "cpu" | None``. ``None`` lets
            the batched miner auto-detect. Ignored when
            ``bulk_mine=False``.
        training_queries_by_dataset: Optional per-dataset labeled
            training queries, shape ``{alias: [{query, relevant_paths},
            ...]}``. Explicit override: when set for a dataset, the
            batched miner uses these real queries (and their gold
            docs as positives) instead of the chunk-prefix
            pseudo-queries. Implies ``bulk_mine=True`` because the
            labeled-query path only exists in ``retrain_batch``.
            Datasets missing from the map follow ``training_pair_source``.
        training_pair_source: Resolution policy when
            ``training_queries_by_dataset`` does not cover a dataset:
            - ``"auto"`` (default, H-R1 ships 2026-04-21): reuse
              ``eval_queries_by_dataset[dataset]`` as training queries
              when present. Only USER-SUPPLIED eval queries are
              promoted; queries produced internally by
              ``split_corpus_for_eval`` never are (those would just
              reintroduce the chunk-prefix signal under a new name).
              Falls back to chunk-prefix when no eval labels exist.
              This removes the 2026-04-18 footgun where a user passed
              eval qrels but silently trained on chunk-prefixes
              (-5.15% macro run).
            - ``"labeled"``: require labels (either explicit training
              queries or eval queries) for every dataset. Raises if a
              dataset has neither.
            - ``"prefix"``: force the legacy chunk-prefix / synth path
              for every dataset that does not have explicit training
              queries, even when eval labels exist. Opt-in for users
              who intentionally want chunk-prefix training on a
              labeled corpus.
        bulk_eval: Route baseline + final eval through
            ``retrain_batch.evaluate_model_batched`` which replaces
            the per-query ``eval_store.search`` scan with one GPU
            matmul over all eval queries. Biggest win when
            ``eval_noise_size`` is large (e.g. >= 10k): baseline +
            final eval for a 3-corpus run drops from ~60 min to
            ~2 min on Colab T4. See the module docstring for the
            approximation trade-offs.
        eval_queries_by_dataset: Optional per-dataset labeled eval
            sets (e.g., BEIR qrels converted via
            ``qrels_to_eval_queries``). Datasets missing from the map
            fall back to ``split_corpus_for_eval``.
        min_gain: Required NDCG@10 improvement. Applied to the
            macro-average unless ``per_dataset_gate=True``.
        per_dataset_gate: Every dataset must individually satisfy
            ``min_gain``. Off by default because single-dataset
            regressions are common in multi-corpus training.
        skip_eval: Train + save unconditionally, no baseline/final
            eval. Useful for smoke tests.

    Returns:
        ``MultiRetrainResult``. ``output_path`` is ``None`` when
        training was gated out or produced too few pairs.
    """
    stores_dict = _normalize_stores(stores)

    sizes = {name: _count_chunks(store) for name, store in stores_dict.items()}
    if sum(sizes.values()) == 0:
        raise ValueError("All provided stores are empty; nothing to train on.")

    budget = compute_triple_budget(
        sizes,
        total_triples=total_triples,
        strategy=sampling,
        temperature=temperature,
    )
    logger.info(
        "retrain_multi budget (strategy=%s, temperature=%s, total=%d): %s",
        sampling,
        temperature,
        total_triples,
        {name: {"chunks": sizes[name], "budget": budget[name]} for name in stores_dict},
    )

    # Per-dataset seeds derived from (seed, name) so datasets with
    # identical chunk counts do not share a permutation prefix. This
    # matters when a user runs the same seed across renamed datasets
    # or mirrored corpora.
    per_dataset_seed = {name: _derive_seed(seed, name) for name in stores_dict}

    # Per-dataset eval prep: reserved_ids exclude train chunks from
    # the matching dataset only; stores do not share chunk ids.
    per_dataset_eval: dict[str, list[dict]] = {}
    per_dataset_reserved: dict[str, set[int]] = {}
    for name, store in stores_dict.items():
        if skip_eval:
            per_dataset_eval[name] = []
            per_dataset_reserved[name] = set()
            continue
        if eval_queries_by_dataset and name in eval_queries_by_dataset:
            per_dataset_eval[name] = list(eval_queries_by_dataset[name])
            per_dataset_reserved[name] = set()
        else:
            reserved, queries = split_corpus_for_eval(
                store, eval_fraction=eval_fraction, seed=per_dataset_seed[name]
            )
            per_dataset_eval[name] = queries
            per_dataset_reserved[name] = reserved

    # Baseline eval per dataset (run once, reused for all gate math).
    if bulk_eval:
        from .retrain_batch import evaluate_model_batched as _eval_fn
    else:
        _eval_fn = evaluate_model

    def _run_eval(store: VstashStore, model_path: str, queries: list[dict], ds_seed: int):
        kwargs = {
            "model_name_or_path": model_path,
            "eval_queries": queries,
            "noise_sample_size": eval_noise_size,
            "seed": ds_seed,
        }
        if bulk_eval:
            kwargs["device"] = bulk_mine_device
        return _eval_fn(store, **kwargs)

    per_dataset_baseline: dict[str, EvalMetrics] = {}
    if not skip_eval:
        for name, store in stores_dict.items():
            queries = per_dataset_eval[name]
            if not queries:
                per_dataset_baseline[name] = EvalMetrics(0.0, 0.0, 0.0, 0)
                continue
            logger.info("Baseline eval on '%s' (%d queries)", name, len(queries))
            per_dataset_baseline[name] = _run_eval(
                store, base_model, queries, per_dataset_seed[name]
            )

    # Triple generation per dataset. Datasets with labeled training
    # queries route through the v5-faithful ``generate_labeled_triples_batched``
    # path; the rest fall back to chunk-prefix / synth.
    #
    # H-R1 default (2026-04-21): when ``training_pair_source="auto"`` and
    # a dataset has no explicit training queries, promote its eval queries
    # (if present) to training queries automatically. This removes the
    # 2026-04-18 footgun where BEIR qrels were silently ignored for
    # training and the corpus trained on chunk-prefixes (-5.15% macro run).
    all_pairs: list[dict] = []
    per_dataset_pairs: dict[str, int] = {}
    explicit_training = training_queries_by_dataset or {}
    # Only USER-SUPPLIED eval queries are eligible for auto-promotion.
    # ``eval_training_source[name]`` can be ``None`` or ``[]`` if a caller
    # passes a partial map; the ``and`` chain below handles both.
    eval_training_source = eval_queries_by_dataset or {}
    training_by_ds: dict[str, list[dict]] = {}
    pair_source_by_ds: dict[str, str] = {}
    for name in stores_dict:
        if name in explicit_training and explicit_training[name]:
            training_by_ds[name] = explicit_training[name]
            pair_source_by_ds[name] = "explicit"
        elif training_pair_source == "auto" and name in eval_training_source and eval_training_source[name]:
            training_by_ds[name] = list(eval_training_source[name])
            pair_source_by_ds[name] = "auto-from-eval"
        elif training_pair_source == "labeled":
            raise ValueError(
                f"training_pair_source='labeled' requires training or eval "
                f"queries for dataset '{name}'; neither was provided."
            )
        else:
            pair_source_by_ds[name] = "prefix"
    # Emit a one-shot warning when auto fires for at least one dataset,
    # so users upgrading past the H-R1 ship can see that their eval
    # queries are now flowing into training too. Escalated from info to
    # warning because this is a semantic default change (2026-04-18
    # footgun).
    promoted_ds = [n for n, src in pair_source_by_ds.items() if src == "auto-from-eval"]
    if promoted_ds:
        logger.warning(
            "retrain_multi (H-R1, shipped 2026-04-21): auto-promoted "
            "eval_queries_by_dataset to training queries for datasets %s. "
            "This replaces the chunk-prefix training signal and fixes the "
            "2026-04-18 -5.15%% footgun. Pass training_pair_source='prefix' "
            "to keep the legacy chunk-prefix training path.",
            promoted_ds,
        )
    logger.info(
        "retrain_multi pair source resolution (training_pair_source=%s): %s",
        training_pair_source,
        pair_source_by_ds,
    )
    for name, store in stores_dict.items():
        n_queries = budget.get(name, 0)
        if n_queries <= 0:
            per_dataset_pairs[name] = 0
            continue

        # Labeled-query path: real queries + gold positives. Matches
        # the v5 recipe that produced Stffens/bge-small-rrf-v2.
        if name in training_by_ds and training_by_ds[name]:
            from .retrain_batch import generate_labeled_triples_batched

            dataset_pairs = generate_labeled_triples_batched(
                store,
                base_model,
                labeled_queries=training_by_ds[name],
                max_queries=n_queries,
                device=bulk_mine_device,
            )
            # The labeled miner emits one triple per (gold, hard_neg)
            # pair so output can vastly exceed the per-dataset budget.
            # Downsample deterministically so total_triples stays
            # meaningful and per-dataset balance is preserved.
            if len(dataset_pairs) > n_queries:
                generated = len(dataset_pairs)
                rng = random.Random(per_dataset_seed[name])
                keep_indices = sorted(rng.sample(range(generated), n_queries))
                dataset_pairs = [dataset_pairs[i] for i in keep_indices]
                logger.info(
                    "Dataset '%s' (labeled-query path): downsampled %d -> %d "
                    "pairs to respect per-dataset budget.",
                    name,
                    generated,
                    n_queries,
                )
            per_dataset_pairs[name] = len(dataset_pairs)
            all_pairs.extend(dataset_pairs)
            logger.info(
                "Dataset '%s' (labeled-query path): %d training pairs generated.",
                name,
                len(dataset_pairs),
            )
            continue

        training_chunks = sample_training_chunks(
            store,
            max_queries=n_queries,
            seed=per_dataset_seed[name],
            exclude_chunk_ids=per_dataset_reserved.get(name) or None,
        )
        if not training_chunks:
            per_dataset_pairs[name] = 0
            continue

        synth_map: dict[int, list[str]] = {}
        if synthesize_queries:
            if cfg is None:
                raise ValueError(
                    "synthesize_queries=True requires the ``cfg`` argument "
                    "(a VstashConfig) so the LLM backend can be resolved."
                )
            from .retrain_synth import synthesize_queries as _synth_queries

            logger.info(
                "Synthesizing %d queries per chunk across %d chunks in '%s' ...",
                synth_n,
                len(training_chunks),
                name,
            )
            synth_map = _synth_queries(
                training_chunks,
                cfg=cfg,
                n_per_chunk=synth_n,
                cache_path=synth_cache,
                model=synth_model,
            )

        if bulk_mine:
            from .retrain_batch import generate_triples_batched

            dataset_pairs = generate_triples_batched(
                store,
                base_model,
                max_queries=n_queries,
                seed=per_dataset_seed[name],
                exclude_chunk_ids=per_dataset_reserved.get(name) or None,
                synthesized_queries=synth_map or None,
                pre_sampled_chunks=training_chunks,
                device=bulk_mine_device,
            )
        else:
            dataset_pairs = generate_triples(
                store,
                base_model,
                max_queries=n_queries,
                seed=per_dataset_seed[name],
                exclude_chunk_ids=per_dataset_reserved.get(name) or None,
                synthesized_queries=synth_map or None,
                pre_sampled_chunks=training_chunks,
            )
        per_dataset_pairs[name] = len(dataset_pairs)
        all_pairs.extend(dataset_pairs)
        logger.info("Dataset '%s': %d training pairs generated.", name, len(dataset_pairs))

    total_pairs = len(all_pairs)

    def _macro(metrics: dict[str, EvalMetrics]) -> float:
        evaluated = [m for m in metrics.values() if m.n_queries > 0]
        if not evaluated:
            return 0.0
        return sum(m.ndcg_at_10 for m in evaluated) / len(evaluated)

    macro_baseline = _macro(per_dataset_baseline)

    if total_pairs < 10:
        logger.warning(
            "Only %d training pairs generated across all corpora (need >= 10). Skipping training.",
            total_pairs,
        )
        return MultiRetrainResult(
            output_path=None,
            per_dataset_baseline=per_dataset_baseline,
            per_dataset_final={},
            macro_baseline_ndcg=macro_baseline,
            macro_final_ndcg=0.0,
            per_dataset_pairs=per_dataset_pairs,
            per_dataset_budget=budget,
            total_pairs=total_pairs,
            gated_out=True,
            min_gain=min_gain,
            sampling=sampling,
            temperature=temperature,
            per_dataset_gate=per_dataset_gate,
        )

    # Global shuffle so the DataLoader mixes datasets inside each batch.
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    final_path_str = str(Path(output_path).expanduser())
    candidate_path = Path(final_path_str + ".candidate")
    if candidate_path.exists():
        shutil.rmtree(candidate_path)

    # Release GPU cache held by the baseline-eval encoders. Without
    # this, the 3-corpus eval leaves enough fragmented memory to push
    # bge-small + hard-neg training past the 15 GB T4 limit.
    _release_gpu_memory()

    train_mnrl(
        all_pairs,
        base_model=base_model,
        output_path=str(candidate_path),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_amp=use_amp,
        max_seq_length=max_seq_length,
        seed=seed,
    )

    _release_gpu_memory()

    per_dataset_final: dict[str, EvalMetrics] = {}
    if not skip_eval:
        for name, store in stores_dict.items():
            queries = per_dataset_eval[name]
            if not queries:
                per_dataset_final[name] = EvalMetrics(0.0, 0.0, 0.0, 0)
                continue
            logger.info("Final eval on '%s' (%d queries)", name, len(queries))
            per_dataset_final[name] = _run_eval(
                store, str(candidate_path), queries, per_dataset_seed[name]
            )

    macro_final = _macro(per_dataset_final) if not skip_eval else 0.0

    # Gate: macro-average by default, or every dataset individually
    # when per_dataset_gate is on. Skip when eval is off.
    if skip_eval:
        gated_out = False
    elif per_dataset_gate:
        gated_out = False
        for name, base_metrics in per_dataset_baseline.items():
            final_metrics = per_dataset_final.get(name)
            if final_metrics is None or base_metrics.n_queries == 0:
                continue
            if (final_metrics.ndcg_at_10 - base_metrics.ndcg_at_10) < min_gain:
                gated_out = True
                break
    else:
        gated_out = (macro_final - macro_baseline) < min_gain

    # Persist eval metadata on the candidate directory regardless of
    # gate outcome, so debugging/inspection still works. ``train_mnrl``
    # normally creates this directory; the ``mkdir`` below is defensive
    # so tests that mock training still get meta on disk.
    candidate_path.mkdir(parents=True, exist_ok=True)
    meta_path = candidate_path / "training_meta.json"
    try:
        existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:
        existing_meta = {}
    existing_meta["multi_eval"] = {
        "per_dataset_baseline": {n: m.as_dict() for n, m in per_dataset_baseline.items()},
        "per_dataset_final": {n: m.as_dict() for n, m in per_dataset_final.items()},
        "per_dataset_pairs": per_dataset_pairs,
        "per_dataset_budget": budget,
        "per_dataset_sizes": sizes,
        "macro_baseline_ndcg": round(macro_baseline, 5),
        "macro_final_ndcg": round(macro_final, 5),
        "macro_delta_ndcg": round(macro_final - macro_baseline, 5),
        "sampling": sampling,
        "temperature": temperature,
        "total_triples_target": total_triples,
        "gated_out": gated_out,
        "per_dataset_gate": per_dataset_gate,
        "min_gain": min_gain,
        "seed": seed,
    }
    meta_path.write_text(json.dumps(existing_meta, indent=2))

    if gated_out:
        logger.warning(
            "Candidate gated out: macro delta NDCG@10 = %+.4f, per_dataset_gate=%s, "
            "min_gain=%+.4f. Candidate left at %s for inspection.",
            macro_final - macro_baseline,
            per_dataset_gate,
            min_gain,
            candidate_path,
        )
        return MultiRetrainResult(
            output_path=None,
            per_dataset_baseline=per_dataset_baseline,
            per_dataset_final=per_dataset_final,
            macro_baseline_ndcg=macro_baseline,
            macro_final_ndcg=macro_final,
            per_dataset_pairs=per_dataset_pairs,
            per_dataset_budget=budget,
            total_pairs=total_pairs,
            gated_out=True,
            min_gain=min_gain,
            sampling=sampling,
            temperature=temperature,
            per_dataset_gate=per_dataset_gate,
        )

    final_path = Path(final_path_str)
    _promote_candidate(candidate_path, final_path)

    logger.info(
        "retrain_multi saved: macro delta NDCG@10 = %+.4f "
        "(macro baseline=%.4f, macro final=%.4f, %d pairs across %d corpora)",
        macro_final - macro_baseline,
        macro_baseline,
        macro_final,
        total_pairs,
        len(stores_dict),
    )
    return MultiRetrainResult(
        output_path=str(final_path),
        per_dataset_baseline=per_dataset_baseline,
        per_dataset_final=per_dataset_final,
        macro_baseline_ndcg=macro_baseline,
        macro_final_ndcg=macro_final,
        per_dataset_pairs=per_dataset_pairs,
        per_dataset_budget=budget,
        total_pairs=total_pairs,
        gated_out=False,
        min_gain=min_gain,
        sampling=sampling,
        temperature=temperature,
        per_dataset_gate=per_dataset_gate,
    )
