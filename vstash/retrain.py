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
from dataclasses import asdict, dataclass
from pathlib import Path

from .embed import embed_query
from .store import VstashStore

logger = logging.getLogger(__name__)

TOP_K = 10
_EVAL_TOP_K = 10
_EVAL_MAX_QUERIES = 200
_EVAL_MIN_QUERIES = 20
_EVAL_NOISE_DEFAULT = 1000


def generate_triples(
    store: VstashStore,
    model_name: str,
    max_queries: int = 5000,
    seed: int = 42,
    exclude_chunk_ids: set[int] | None = None,
    triplets_per_query: int = 1,
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
        triplets_per_query: Up to N triplets emitted per pseudo-query,
            each sharing the same (query, positive) but paired with a
            different hard negative drawn from the top-5 disagreement
            set. Default 1 preserves legacy behavior. Higher values
            (e.g., 5) yield 3-5x more training signal from the same
            disagreement data.

    Returns:
        List of dicts with 'query', 'positive', and 'negative' keys.
    """
    rng = random.Random(seed)
    excluded = exclude_chunk_ids or set()

    # Sample chunks as pseudo-queries. Determinism: fetch IDs in stable
    # order, shuffle with a seeded Python RNG, slice to sample_size, then
    # fetch the chunk text/path only for those IDs. Keeps memory linear
    # in the sample size (not in the whole corpus) and actually honors
    # ``seed`` across runs and SQLite versions.
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
    by_id = {r["id"]: r for r in fetched}
    rows = [by_id[i] for i in sample_ids if i in by_id]

    pairs: list[dict] = []
    disagreements = 0
    k_per_query = max(1, int(triplets_per_query))

    for row in rows:
        query_text = row["text"][:200]  # use first 200 chars as pseudo-query
        doc_path = row["path"]

        emb = embed_query(query_text, model_name)

        # Adaptive weights based on query length: short queries have
        # more FTS value (exact term matching matters), long queries
        # lean harder on vector (semantic matching dominates).
        word_count = len(query_text.split())
        if word_count <= 10:
            vec_hi, fts_hi = 0.70, 0.30
            vec_lo, fts_lo = 0.30, 0.70
        elif word_count <= 50:
            vec_hi, fts_hi = 0.85, 0.15
            vec_lo, fts_lo = 0.15, 0.85
        else:
            vec_hi, fts_hi = 0.95, 0.05
            vec_lo, fts_lo = 0.50, 0.50

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

        # The document's own chunk is the positive
        positive_text = result_texts.get(doc_path)

        if not positive_text or positive_text == query_text:
            continue

        # Hard negatives: chunks in one signal's top-5 but not the other's.
        # Collect up to k_per_query distinct negatives in rank order,
        # preferring vec-side disagreements first (consistent with the
        # legacy single-negative preference), then fts-side. Dedup by
        # negative text (same text from different paths is still a
        # duplicate signal to the loss).
        hard_negs: list[str] = []
        seen_neg_texts: set[str] = set()

        def _try_add(candidate_text: str, candidate_path: str, other_paths: set[str]) -> None:
            if candidate_path == doc_path:
                return
            if candidate_path in other_paths:
                return
            if not candidate_text or candidate_text in seen_neg_texts:
                return
            hard_negs.append(candidate_text)
            seen_neg_texts.add(candidate_text)

        # Disagreement set definition stays at top-5 (vec_paths/fts_paths
        # above). For candidate gathering we iterate the whole top_k
        # window so K > 5 has enough depth to find K distinct negatives.
        for r in vec_results:
            if len(hard_negs) >= k_per_query:
                break
            _try_add(r.text, r.path, fts_paths)
        for r in fts_results:
            if len(hard_negs) >= k_per_query:
                break
            _try_add(r.text, r.path, vec_paths)

        if not hard_negs:
            # No disagreement-based negative found. Emit a single triplet
            # with negative=None so MNRL falls back to in-batch negatives
            # (legacy behavior when the signals agreed).
            pairs.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negative": None,
                }
            )
        else:
            for neg_text in hard_negs:
                pairs.append(
                    {
                        "query": query_text,
                        "positive": positive_text,
                        "negative": neg_text,
                    }
                )

    avg_triplets = len(pairs) / len(rows) if rows else 0.0
    logger.info(
        "Generated %d triplets from %d queries "
        "(%d disagreements, %.0f%%, avg %.2f triplets/query, target k=%d)",
        len(pairs),
        len(rows),
        disagreements,
        disagreements / len(rows) * 100 if rows else 0,
        avg_triplets,
        k_per_query,
    )
    return pairs


def train_mnrl(
    pairs: list[dict],
    base_model: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "~/.vstash/models/retrained",
    epochs: int = 2,
    lr: float = 3e-6,
    batch_size: int = 64,
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

    examples = []
    for p in pairs:
        if p.get("negative"):
            examples.append(InputExample(texts=[p["query"], p["positive"], p["negative"]]))
        else:
            examples.append(InputExample(texts=[p["query"], p["positive"]]))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = min(50, len(loader) // 5)
    logger.info(
        "Training: %d pairs, %d epochs, batch=%d, lr=%s",
        len(pairs),
        epochs,
        batch_size,
        lr,
    )

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output,
        show_progress_bar=True,
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
        "training_time_s": round(elapsed, 1),
    }
    (Path(output) / "training_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s (%.0fs)", output, elapsed)
    return output


# ---------------------------------------------------------------------- #
# Eval-gated retraining                                                   #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalMetrics:
    """Honest retrieval metrics on a held-out query set."""

    ndcg_at_10: float
    mrr: float
    hit_at_10: float
    n_queries: int

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
        EvalMetrics with NDCG@10, MRR, Hit@10.

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

        # Score each query via production search (RRF with adaptive weights).
        ndcgs: list[float] = []
        mrrs: list[float] = []
        hits: list[float] = []
        for q, q_vec in zip(normalized_queries, q_vecs):
            results = eval_store.search(
                query_embedding=list(map(float, q_vec)),
                query_text=q["query"],
                top_k=_EVAL_TOP_K,
            )
            relevant_set = set(q["relevant_paths"])
            ranks_hit: list[int] = []
            first_rank: int | None = None
            for i, r in enumerate(results, start=1):
                if r.path in relevant_set:
                    ranks_hit.append(i)
                    if first_rank is None:
                        first_rank = i
            ndcgs.append(_ndcg_from_ranks(ranks_hit, num_relevant=len(relevant_set)))
            mrrs.append(1.0 / first_rank if first_rank is not None else 0.0)
            hits.append(1.0 if first_rank is not None else 0.0)

        return EvalMetrics(
            ndcg_at_10=sum(ndcgs) / len(ndcgs),
            mrr=sum(mrrs) / len(mrrs),
            hit_at_10=sum(hits) / len(hits),
            n_queries=len(normalized_queries),
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
    eval_fraction: float = 0.15,
    eval_noise_size: int = _EVAL_NOISE_DEFAULT,
    min_gain: float = 0.0,
    skip_eval: bool = False,
    seed: int = 42,
    eval_queries: list[dict] | None = None,
    triplets_per_query: int = 1,
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
        pairs = generate_triples(
            store,
            base_model,
            max_queries=max_queries,
            seed=seed,
            triplets_per_query=triplets_per_query,
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
            triplets_per_query=triplets_per_query,
        )

    logger.info("Running baseline eval on %d held-out queries ...", len(effective_queries))
    baseline = evaluate_model(
        store,
        model_name_or_path=base_model,
        eval_queries=effective_queries,
        noise_sample_size=eval_noise_size,
        seed=seed,
    )

    pairs = generate_triples(
        store,
        base_model,
        max_queries=max_queries,
        seed=seed,
        exclude_chunk_ids=reserved_ids,
        triplets_per_query=triplets_per_query,
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

    train_mnrl(
        pairs,
        base_model=base_model,
        output_path=str(candidate_path),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

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

    # Promote candidate to final path with a last-known-good backup.
    # If we rmtree(final_path) before the move, a crash between those
    # two steps leaves the user with no model. Instead: rename the
    # existing model aside to `.old`, move the candidate into place,
    # remove the backup only once the new model is committed. On any
    # failure, roll the backup back so the user never ends up empty.
    final_path = Path(final_path_str)
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
