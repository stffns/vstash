"""retrain_batch.py -- GPU-batched triple mining + eval for big corpora.

Two fast-approximation replacements for the per-query
``store.search`` path used during retraining:

``generate_triples_batched`` (T1.4b)
    The default ``vstash.retrain.generate_triples`` calls
    ``embed_query`` once per pseudo-query and runs two
    ``store.search`` invocations per query (vec-heavy + fts-heavy).
    On a FiQA-sized corpus (~57k chunks) the per-query sqlite-vec
    full scan dominates at ~500 ms on a Colab CPU path. For 19k
    queries that is ~5 hours before any training happens. The
    batched miner replaces the vec search with one
    ``query_vecs @ corpus_vecs.T`` matmul on GPU.

``evaluate_model_batched`` (T1.4c)
    ``vstash.retrain.evaluate_model`` ingests the relevant + noise
    chunks into a temp store and calls ``eval_store.search()`` per
    query. At ``EVAL_NOISE = 57638`` and ~600 qrels per dataset
    that is ~10 min per eval pass (baseline + final per dataset = 60
    min total on a 3-corpus run). ``evaluate_model_batched`` keeps
    the temp store for FTS5 but skips the vec scan by computing a
    single GPU matmul over all eval queries.

Both are **fast approximations** of their legacy counterparts, not
drop-in reproductions. Trade-offs:

- Vec ranking via one matmul, embedded with
  ``SentenceTransformer(base_model)`` directly. Bypasses
  ``embed_query``'s ONNX path and the ``vec_chunks`` table, so the
  resulting query/doc vectors may not match what the production
  index produces bit-for-bit (e.g. if the store was reindexed with
  a different model between ingest and retrain).
- No ``distance_cutoff`` filter: always keeps the top-10 cosine
  hits. ``store.search`` drops results past a default cutoff of 1.15.
- FTS5 ``LIMIT 10`` per query. ``store.search`` uses a larger
  candidate pool (``min(K*10, max(K*3, total//3))``) before RRF.
- Raw RRF over the top-10 vec + top-10 FTS, no MMR dedup. Multiple
  chunks from the same doc can occupy the result set.

For mining this is fine -- the training signal is similar in
expectation but a different sample of disagreement. For eval, the
absolute NDCG numbers will differ from the legacy path by a few
points. The macro-averaged baseline-vs-final delta is
*approximately* preserved because both sides of the comparison
use the same batched eval, but per-query sign can flip:

- Adaptive RRF in production up-weights FTS on rare query terms
  via IDF. The batched path skips IDF, so a rare-term query that
  legacy eval scores on the FTS side will score on vec here.
  Fine-tuning changes vec distances but not IDF, so the two
  paths can disagree on whether the fine-tune helped that query.
- Distance cutoff (1.15 in production): baseline can have a
  relevant doc just under the cutoff and fine-tune pushes it
  over (or vice versa). Batched eval has no cutoff, so it keeps
  top-10 unconditionally.

If you change the RRF weights used by ``evaluate_model_batched``,
re-run the batched-vs-legacy delta parity check on a real BEIR
slice -- not just the unit tests -- before shipping. Unit tests
verify plumbing, not semantic equivalence.

When faithful reproduction of production search matters (paper
numbers, published eval tables), use the non-batched defaults.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .retrain import TOP_K, adaptive_rrf_weights, sample_training_chunks
from .store import RRF_K, VstashStore

if TYPE_CHECKING:
    from .retrain import EvalMetrics

logger = logging.getLogger(__name__)

_DISAGREEMENT_TOP = 5


def _fetch_corpus_rows(store: VstashStore) -> list[dict]:
    """Load every chunk's id + text + path + seq for batched embedding.

    Ordered by chunk id so the returned list aligns with any later
    index-based lookups. ``seq`` is included so callers that care
    about "first chunk of doc" can use the explicit per-doc ordering
    rather than relying on autoincrement id ordering.
    """
    rows = store._conn.execute(
        "SELECT c.id, c.text, c.seq, d.path FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "ORDER BY c.id"
    ).fetchall()
    return [dict(r) for r in rows]


def _fts_top_k(
    conn: sqlite3.Connection,
    query_text: str,
    top_k: int,
) -> list[int]:
    """Return the chunk ids of the FTS5 top-k matches for ``query_text``.

    Uses the same sanitization as ``VstashStore.search`` so keyword
    queries with quotes or operators do not break FTS5 parsing.
    """
    safe_query, _ = VstashStore._build_fts_match_query(query_text)
    try:
        rows = conn.execute(
            "SELECT c.id FROM fts_chunks f "
            "JOIN chunks c ON c.id = f.rowid "
            "WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            [safe_query, top_k],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # FTS5 rejected the sanitized query (e.g. only stop-words left).
        # Drop to empty result so mining continues, but warn so the
        # user can diagnose abnormally low pair counts.
        logger.warning(
            "FTS5 match failed for query %r (sanitized=%r): %s",
            query_text[:60],
            safe_query[:60],
            exc,
        )
        return []
    return [int(r["id"]) for r in rows]


def _rrf_fuse(
    vec_chunk_ids: list[int],
    fts_chunk_ids: list[int],
    vec_weight: float,
    fts_weight: float,
    chunk_id_to_path: dict[int, str],
    top_n: int = _DISAGREEMENT_TOP,
) -> list[int]:
    """Reciprocal-rank-fuse vec and FTS rankings, return top-N chunk ids.

    Matches the RRF formula used by ``VstashStore._fuse_rrf_scores``:
    ``w * 1 / (RRF_K + rank)`` with 0-indexed rank. ``top_n`` defaults
    to the disagreement-mining top-5 but callers that need top-10
    (evaluate_model_batched) pass it explicitly.
    """
    scores: dict[int, float] = {}
    for rank, cid in enumerate(vec_chunk_ids):
        scores[cid] = scores.get(cid, 0.0) + vec_weight * (1.0 / (RRF_K + rank))
    for rank, cid in enumerate(fts_chunk_ids):
        scores[cid] = scores.get(cid, 0.0) + fts_weight * (1.0 / (RRF_K + rank))

    # Sort by descending score; break ties by chunk id for determinism.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # Keep only chunks we know a path for (all of them, since both
    # input lists come from chunk_id_to_path).
    return [cid for cid, _ in ordered if cid in chunk_id_to_path][:top_n]


def generate_triples_batched(
    store: VstashStore,
    base_model: str,
    max_queries: int = 5000,
    seed: int = 42,
    exclude_chunk_ids: set[int] | None = None,
    synthesized_queries: dict[int, list[str]] | None = None,
    pre_sampled_chunks: list[dict] | None = None,
    device: str | None = None,
    encode_batch_size: int = 256,
    matmul_batch_queries: int = 512,
) -> list[dict]:
    """GPU-batched equivalent of ``generate_triples`` for big corpora.

    Args:
        store: Source corpus.
        base_model: Sentence-transformers model to use for both query
            and corpus embedding. Using the same model for both sides
            keeps the two vector spaces aligned.
        max_queries: Max pseudo-queries to sample.
        seed: Deterministic sampling / ordering.
        exclude_chunk_ids: Chunk ids to leave out of training (eval
            hold-out).
        synthesized_queries: Optional ``{chunk_id: [query, ...]}`` map
            identical to the one consumed by ``generate_triples``.
        pre_sampled_chunks: Optional pre-sampled training chunks (from
            ``sample_training_chunks``) so synth and batched mining
            operate on the same chunk set.
        device: ``"cuda" | "cpu" | None`` (None auto-detects).
        encode_batch_size: Sentence-transformers encode batch size.
        matmul_batch_queries: Query chunk size for the ``Q @ D.T``
            matmul. Caps GPU memory when N is big (e.g. 57k corpus +
            19k queries would be ~4 GB fp32 for the full matrix; 512
            at a time keeps peak well under 1 GB).

    Returns:
        List of ``{query, positive, negative}`` dicts. Shape and keys
        match ``generate_triples`` so callers are drop-in compatible.

    Raises:
        ImportError: If sentence-transformers / torch are not installed.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers + torch are required for batched "
            "mining. Install with: pip install 'sentence-transformers>=3' "
            "torch 'accelerate>=1.1.0'"
        ) from exc

    # 1. Sample training chunks (reuse single-store helper for parity).
    if pre_sampled_chunks is not None:
        training_rows = pre_sampled_chunks
    else:
        training_rows = sample_training_chunks(
            store,
            max_queries=max_queries,
            seed=seed,
            exclude_chunk_ids=exclude_chunk_ids,
        )
    if not training_rows:
        return []

    # 2. Fetch full corpus so we can build doc_vecs once.
    corpus_rows = _fetch_corpus_rows(store)
    if not corpus_rows:
        return []
    corpus_ids = [int(r["id"]) for r in corpus_rows]
    corpus_texts = [r["text"] for r in corpus_rows]
    corpus_paths = [r["path"] for r in corpus_rows]
    chunk_id_to_path = {cid: corpus_paths[i] for i, cid in enumerate(corpus_ids)}
    chunk_id_to_text = {cid: corpus_texts[i] for i, cid in enumerate(corpus_ids)}

    # 3. Model + device.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Loading base model '%s' on device=%s for batched mining ...",
        base_model,
        device,
    )
    model = SentenceTransformer(base_model, device=device)

    # 4. Build the full (query_text, source_row) iteration list, keeping
    # the mapping back to the source chunk so we emit one triple per
    # query regardless of whether prefix or synth is used.
    synth_map = synthesized_queries or {}
    query_texts: list[str] = []
    query_sources: list[dict] = []
    synth_used = 0
    prefix_used = 0
    for row in training_rows:
        if row["id"] in synth_map and synth_map[row["id"]]:
            for q in synth_map[row["id"]]:
                if not q:
                    continue
                query_texts.append(q)
                query_sources.append(row)
                synth_used += 1
        else:
            q = row["text"][:200]
            if not q:
                continue
            query_texts.append(q)
            query_sources.append(row)
            prefix_used += 1

    if not query_texts:
        return []

    # 5. Batch-encode corpus + queries once.
    t0 = time.perf_counter()
    corpus_vecs = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    logger.info("Encoded %d corpus chunks in %.1fs", len(corpus_texts), time.perf_counter() - t0)
    t0 = time.perf_counter()
    query_vecs = model.encode(
        query_texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    logger.info(
        "Encoded %d training queries in %.1fs",
        len(query_texts),
        time.perf_counter() - t0,
    )

    # 6. Vec top-K for every query in one big matmul (batched to bound memory).
    corpus_vecs = corpus_vecs.to(device)
    query_vecs = query_vecs.to(device)
    all_vec_topk: list[list[int]] = []  # per query, list of chunk_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, query_vecs.size(0), matmul_batch_queries):
            batch = query_vecs[start : start + matmul_batch_queries]
            sims = batch @ corpus_vecs.T  # (B, N)
            _top_scores, top_idx = sims.topk(min(TOP_K, sims.size(1)), dim=1)
            idx_cpu = top_idx.cpu().tolist()
            for row in idx_cpu:
                all_vec_topk.append([corpus_ids[i] for i in row])
    logger.info(
        "Vec similarity for %d queries in %.1fs (batched matmul)",
        len(query_texts),
        time.perf_counter() - t0,
    )

    # 7. Per-query FTS + RRF disagreement mining. This is the only loop
    # that is not batched. FTS5 calls are cheap (~5-50 ms) so the loop
    # finishes in minutes even for 10k+ queries.
    pairs: list[dict] = []
    disagreements = 0
    t0 = time.perf_counter()
    for q_idx, (query_text, source) in enumerate(zip(query_texts, query_sources, strict=True)):
        vec_top = all_vec_topk[q_idx]
        fts_top = _fts_top_k(store._conn, query_text, TOP_K)
        # A chunk id can appear in fts_top but not in corpus slice only
        # if the chunk was deleted after our initial fetch; guard against
        # that by filtering.
        fts_top = [cid for cid in fts_top if cid in chunk_id_to_path]

        # Adaptive RRF weights: shared with generate_triples so the
        # two signals use byte-identical ladders.
        vec_hi, fts_hi, vec_lo, fts_lo = adaptive_rrf_weights(len(query_text.split()))

        vec_heavy_top5 = _rrf_fuse(vec_top, fts_top, vec_hi, fts_hi, chunk_id_to_path)
        fts_heavy_top5 = _rrf_fuse(vec_top, fts_top, vec_lo, fts_lo, chunk_id_to_path)

        vec_paths = {chunk_id_to_path[cid] for cid in vec_heavy_top5}
        fts_paths = {chunk_id_to_path[cid] for cid in fts_heavy_top5}

        if vec_paths != fts_paths:
            disagreements += 1

        source_path = source["path"]
        own_chunk_text = source["text"]

        # Positive lookup mirrors generate_triples exactly: path -> text
        # across the union of vec + fts results, last-seen wins. This
        # lets multi-chunk docs produce a sibling-chunk positive when
        # the pseudo-query is short enough to equal its own chunk text.
        result_texts: dict[str, str] = {}
        for cid in vec_top + fts_top:
            p = chunk_id_to_path.get(cid)
            t = chunk_id_to_text.get(cid)
            if p is not None and t is not None:
                result_texts[p] = t
        positive_text = result_texts.get(source_path) or own_chunk_text
        if not positive_text or positive_text == query_text:
            continue

        # Hard negative: first chunk in vec_heavy top-5 whose path is
        # not in fts_heavy top-5 and is not the source doc. Fall back
        # to the mirrored side if vec side has no unique candidate.
        hard_neg_text = None
        for cid in vec_heavy_top5:
            p = chunk_id_to_path[cid]
            if p not in fts_paths and p != source_path:
                hard_neg_text = chunk_id_to_text.get(cid)
                if hard_neg_text:
                    break
        if hard_neg_text is None:
            for cid in fts_heavy_top5:
                p = chunk_id_to_path[cid]
                if p not in vec_paths and p != source_path:
                    hard_neg_text = chunk_id_to_text.get(cid)
                    if hard_neg_text:
                        break

        pairs.append(
            {
                "query": query_text,
                "positive": positive_text,
                "negative": hard_neg_text,
            }
        )

    logger.info(
        "Mined %d pairs from %d queries in %.1fs (%d synth + %d prefix, %d disagreements, %.0f%%)",
        len(pairs),
        len(query_texts),
        time.perf_counter() - t0,
        synth_used,
        prefix_used,
        disagreements,
        disagreements / len(query_texts) * 100 if query_texts else 0,
    )

    # Release the big GPU tensors before returning so the next dataset
    # in retrain_multi's loop does not see doubled peak memory.
    del corpus_vecs, query_vecs
    from .retrain import _release_gpu_memory

    _release_gpu_memory()

    return pairs


# ---------------------------------------------------------------------- #
# evaluate_model_batched (T1.4c)                                          #
# ---------------------------------------------------------------------- #


def _normalise_eval_queries(eval_queries: list[dict]) -> list[dict]:
    """Coerce ``relevant_path`` (legacy str) into ``relevant_paths`` (list)."""
    out: list[dict] = []
    for q in eval_queries:
        paths = q.get("relevant_paths")
        if paths is None:
            legacy = q.get("relevant_path")
            paths = [legacy] if legacy is not None else []
        out.append({**q, "relevant_paths": list(paths)})
    return out


def evaluate_model_batched(
    base_store: VstashStore,
    model_name_or_path: str,
    eval_queries: list[dict],
    noise_sample_size: int = 1000,
    seed: int = 42,
    tmp_dir: Path | None = None,
    device: str | None = None,
    encode_batch_size: int = 256,
    matmul_batch_queries: int = 512,
) -> EvalMetrics:
    """GPU-batched equivalent of ``vstash.retrain.evaluate_model``.

    Skips the per-query ``store.search`` scan (the bottleneck when
    ``noise_sample_size`` is large) by:

    1. Batch-encoding every eval query with
       ``SentenceTransformer(model_name_or_path)`` on GPU.
    2. Computing ``query_vecs @ corpus_vecs.T`` once (chunked by
       ``matmul_batch_queries`` to bound peak memory) to obtain per-
       query vec top-K.
    3. Running FTS5 per query against a temp store (same as
       ``evaluate_model``; FTS5 is already fast at 5-50 ms).
    4. Fusing vec + FTS rankings with adaptive RRF weights and
       scoring the result set for NDCG@10 / MRR / Hit@10.

    The absolute NDCG numbers may differ from ``evaluate_model`` by a
    few percent because this path skips the production pipeline's
    MMR dedup, IDF weighting, and distance cutoff. The
    baseline-vs-final delta (which the retrain gate uses) is
    preserved because both sides of the comparison use the same
    batched eval path.

    Args mirror ``evaluate_model`` plus ``device`` / batch-size
    overrides. Returns an ``EvalMetrics`` with NDCG@10, MRR, Hit@10.
    """
    import random as _random
    import tempfile
    import time as _time

    from .retrain import (
        _EVAL_NDCG_SHALLOW_K,
        _EVAL_RECALL_K,
        _EVAL_TOP_K,
        _ndcg_from_ranks,
        _recall_at_k,
        _relevant_chunks,
        _release_gpu_memory,
        _sample_noise_chunks,
        EvalMetrics,
    )

    if not eval_queries:
        return EvalMetrics(ndcg_at_10=0.0, mrr=0.0, hit_at_10=0.0, n_queries=0)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers + torch are required for batched eval. "
            "Install with: pip install 'sentence-transformers>=3' torch "
            "'accelerate>=1.1.0'"
        ) from exc

    normalized_queries = _normalise_eval_queries(eval_queries)
    relevant_paths: set[str] = set()
    for q in normalized_queries:
        relevant_paths.update(q["relevant_paths"])

    relevant_rows = _relevant_chunks(base_store, relevant_paths)
    noise_rows = _sample_noise_chunks(base_store, relevant_paths, noise_sample_size, seed)
    all_rows = relevant_rows + noise_rows
    if not relevant_rows:
        return EvalMetrics(ndcg_at_10=0.0, mrr=0.0, hit_at_10=0.0, n_queries=0)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "evaluate_model_batched: loading %s on device=%s for %d corpus + %d queries",
        model_name_or_path,
        device,
        len(all_rows),
        len(normalized_queries),
    )
    model = SentenceTransformer(model_name_or_path, device=device)
    if hasattr(model, "get_embedding_dimension"):
        dim = int(model.get_embedding_dimension())
    else:
        dim = int(model.get_sentence_embedding_dimension())

    # Encode corpus + queries once, keep on GPU for the matmul.
    texts = [r["text"] for r in all_rows]
    corpus_vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    query_texts = [q["query"] for q in normalized_queries]
    query_vecs = model.encode(
        query_texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    corpus_vecs = corpus_vecs.to(device)
    query_vecs = query_vecs.to(device)

    # Build a temp store: we only need its FTS5 index. Ingest via
    # add_documents_batch so the 60k-chunk case still loads in under a
    # minute.
    eval_store: VstashStore | None = None
    tmp_db: Path | None = None
    ndcgs_10: list[float] = []
    ndcgs_3: list[float] = []
    mrrs: list[float] = []
    hits: list[float] = []
    recalls_100: list[float] = []
    try:
        tmp_parent = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_db = (
            tmp_parent
            / f"retrain_eval_batched_{int(_time.time() * 1000)}_{_random.randint(0, 1 << 30)}.db"
        )
        # Throwaway eval-only DB; sqlite-vec defaults are intentional.
        # See vstash/retrain.py for rationale.
        eval_store = VstashStore(str(tmp_db), embedding_dim=dim)

        # Ingest so FTS5 is populated. We feed zero-vectors for the
        # vec index since we never query it -- saves memory and
        # disk. Passing the real embeddings would be wasted work.
        zero_vec = [0.0] * dim
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
                    "embeddings": [zero_vec for _ in idxs],
                    "source_type": "file",
                }
            )
        eval_store.add_documents_batch(docs_to_add)

        # The corpus chunks are ordered by their position in all_rows;
        # that ordering is exactly what add_documents_batch assigned
        # to the FTS5 rowids (1-indexed, dense).  Map matmul indices
        # (0-indexed into all_rows) -> chunk_id by querying the temp
        # store for the final ordering.
        chunk_rows = eval_store._conn.execute(
            "SELECT c.id, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
        ).fetchall()
        # Re-order all_rows's index map so chunk_id_to_idx maps FTS rowid -> matmul index.
        # Simpler: build chunk_id -> path map and chunk_id -> matmul row index. To do the
        # latter we need to know which of all_rows each chunk came from; since paths can
        # have multiple chunks we cannot invert by path alone. The simpler fix is to
        # recompute a chunk_id -> embedding_row_index by matching on (path, position).
        by_path_positions: dict[str, list[int]] = {}
        for i, row in enumerate(all_rows):
            by_path_positions.setdefault(row["path"], []).append(i)
        chunk_id_to_matmul_idx: dict[int, int] = {}
        chunk_id_to_path: dict[int, str] = {}
        path_cursor: dict[str, int] = {}
        for r in chunk_rows:
            cid = int(r["id"])
            p = r["path"]
            chunk_id_to_path[cid] = p
            cursor = path_cursor.get(p, 0)
            positions = by_path_positions.get(p, [])
            if cursor >= len(positions):
                continue
            chunk_id_to_matmul_idx[cid] = positions[cursor]
            path_cursor[p] = cursor + 1

        # Reverse lookup: matmul-matrix index -> chunk_id. Built once
        # from the post-ingest rowid ordering. Every row in all_rows
        # maps to exactly one chunk so the inverse is well-defined.
        matmul_idx_to_chunk_id = {v: k for k, v in chunk_id_to_matmul_idx.items()}

        # Vec top-K per query via batched matmul on GPU. Pull RECALL_K
        # candidates so the same ranking supports NDCG@10 / NDCG@3 /
        # Recall@100 in one pass. After doc-level deduplication the
        # fused list is still almost always >= 10 unique paths, so
        # NDCG@10 is unaffected; the extra candidates only feed
        # Recall@100.
        all_vec_topk: list[list[int]] = []
        with torch.no_grad():
            for start in range(0, query_vecs.size(0), matmul_batch_queries):
                batch = query_vecs[start : start + matmul_batch_queries]
                sims = batch @ corpus_vecs.T
                _top_scores, top_idx = sims.topk(min(_EVAL_RECALL_K, sims.size(1)), dim=1)
                idx_cpu = top_idx.cpu().tolist()
                for matmul_indices in idx_cpu:
                    all_vec_topk.append(
                        [
                            matmul_idx_to_chunk_id[mi]
                            for mi in matmul_indices
                            if mi in matmul_idx_to_chunk_id
                        ]
                    )

        # Per-query FTS + RRF merge. Multiple chunks from the same
        # doc can land in the fused list (no MMR dedup in this path),
        # but NDCG is a per-document metric, so we collapse the
        # ranking to unique paths in first-occurrence order before
        # scoring. Without this, a relevant doc surfacing twice in
        # top-10 would be counted twice and push NDCG above 1.0.
        for q, vec_top in zip(normalized_queries, all_vec_topk, strict=True):
            fts_top = _fts_top_k(eval_store._conn, q["query"], _EVAL_RECALL_K)
            fts_top = [cid for cid in fts_top if cid in chunk_id_to_path]

            # Adaptive RRF weights, vec-heavy side. Faithful enough to
            # store.search's default for short/medium queries.
            vec_hi, fts_hi, _vec_lo, _fts_lo = adaptive_rrf_weights(len(q["query"].split()))
            fused = _rrf_fuse(
                vec_top, fts_top, vec_hi, fts_hi, chunk_id_to_path, top_n=_EVAL_RECALL_K
            )

            # Collapse chunk-level ranking to doc-level (unique paths).
            unique_paths: list[str] = []
            seen_paths: set[str] = set()
            for cid in fused:
                p = chunk_id_to_path[cid]
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                unique_paths.append(p)

            relevant_set = set(q["relevant_paths"])
            ranks_hit: list[int] = []
            first_rank: int | None = None
            for rank_i, p in enumerate(unique_paths, start=1):
                if p in relevant_set:
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
        # Release GPU tensors here so exceptions during the per-query
        # loop do not leave the corpus + query matrices live for the
        # next dataset in retrain_multi's loop.
        try:
            del corpus_vecs, query_vecs
        except NameError:
            pass
        _release_gpu_memory()

    n = len(normalized_queries)
    return EvalMetrics(
        ndcg_at_10=sum(ndcgs_10) / len(ndcgs_10) if ndcgs_10 else 0.0,
        mrr=sum(mrrs) / len(mrrs) if mrrs else 0.0,
        hit_at_10=sum(hits) / len(hits) if hits else 0.0,
        n_queries=n,
        ndcg_at_3=sum(ndcgs_3) / len(ndcgs_3) if ndcgs_3 else 0.0,
        recall_at_100=sum(recalls_100) / len(recalls_100) if recalls_100 else 0.0,
    )


# ---------------------------------------------------------------------- #
# T1.5: labeled-query training pair mining                                 #
# ---------------------------------------------------------------------- #
#
# The default ``generate_triples`` / ``generate_triples_batched`` use
# ``chunk[:200]`` as a pseudo-query. That is a statement, not a question,
# and MNRL trained on statement-as-query damages question-based
# retrieval (SciFact/FiQA cratered -5/-9% in the first T1.4 Colab run,
# NFCorpus was flat -- the two that regressed are question datasets).
#
# ``generate_labeled_triples_batched`` reproduces the v5 training recipe
# from ``experiments/rrf_training_pairs.py``:
#
# - Training query text comes from a ``labeled_queries`` list (typically
#   BEIR ``queries.jsonl`` filtered to queries with qrels).
# - Positive = first chunk of the gold document (from ``relevant_paths``).
# - Fixed RRF weights 0.95/0.05 for both searches (vec-heavy and
#   fts-heavy), matching v5's ``rrf_training_pairs``.
# - Emits multiple triples per query: one per (gold_path, hard_neg_path)
#   pair, where hard_neg_path is drawn from both vec-only and fts-only
#   disagreement candidates.
#
# Output shape is still ``{query, positive, negative}`` so train_mnrl
# is drop-in compatible.


_LABELED_TRAIN_WEIGHTS_VEC_HEAVY = (0.95, 0.05)
_LABELED_TRAIN_WEIGHTS_FTS_HEAVY = (0.05, 0.95)


def generate_labeled_triples_batched(
    store: VstashStore,
    base_model: str,
    labeled_queries: list[dict],
    max_queries: int | None = None,
    device: str | None = None,
    encode_batch_size: int = 256,
    matmul_batch_queries: int = 512,
) -> list[dict]:
    """Mine training triples using real labeled queries (T1.5).

    Reproduces the v5 flow from ``experiments/rrf_training_pairs.py``:

    - Training query text = ``q["query"]`` (typically a BEIR
      ``queries.jsonl`` entry).
    - Positive = first chunk of any gold doc in ``q["relevant_paths"]``.
    - Hard negatives mined from vec-heavy / fts-heavy top-K disagreement
      using fixed weights 0.95/0.05 (matching v5).
    - Emits ``|gold| * (|vec_only_neg| + |fts_only_neg|)`` triples per
      query. Triple count can be >> number of queries.

    Args:
        store: Source corpus.
        base_model: HF model name for batched encode.
        labeled_queries: List of ``{query, relevant_paths}`` dicts. The
            ``relevant_paths`` list must contain paths that exist in
            the store. Queries whose gold paths are all absent are
            skipped.
        max_queries: Optional cap on queries processed. ``None`` uses
            all entries. Useful for smoke runs.
        device: ``"cuda" | "cpu" | None``.
        encode_batch_size: SentenceTransformer encode batch.
        matmul_batch_queries: Per-matmul query chunk size.

    Returns:
        List of ``{query, positive, negative}`` dicts, drop-in
        compatible with ``train_mnrl``.

    Raises:
        ImportError: If sentence-transformers / torch are not installed.
        ValueError: If ``labeled_queries`` is empty.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers + torch are required for labeled "
            "triple mining. Install with: "
            "pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'"
        ) from exc

    if not labeled_queries:
        raise ValueError("labeled_queries must contain at least one query.")

    # Normalize + cap.
    queries_normalized: list[dict] = []
    for q in labeled_queries:
        paths = q.get("relevant_paths")
        if paths is None:
            legacy = q.get("relevant_path")
            paths = [legacy] if legacy is not None else []
        if not q.get("query") or not paths:
            continue
        queries_normalized.append({**q, "relevant_paths": list(paths)})

    if max_queries is not None:
        queries_normalized = queries_normalized[:max_queries]
    if not queries_normalized:
        return []

    # Fetch corpus once.
    corpus_rows = _fetch_corpus_rows(store)
    if not corpus_rows:
        return []
    corpus_ids = [int(r["id"]) for r in corpus_rows]
    corpus_texts = [r["text"] for r in corpus_rows]
    corpus_paths = [r["path"] for r in corpus_rows]
    chunk_id_to_path = {cid: corpus_paths[i] for i, cid in enumerate(corpus_ids)}
    chunk_id_to_text = {cid: corpus_texts[i] for i, cid in enumerate(corpus_ids)}

    # path -> first chunk_id selected by explicit (seq, id) ordering,
    # not just autoincrement id. ``chunks`` has an explicit ``seq``
    # column so "first chunk of doc" means seq=0 in the general case.
    # v5 uses ``doc_texts[path][:512]`` for both positive and negative;
    # we use the first chunk's text (truncated to 512 chars below).
    path_to_first_chunk_id: dict[str, int] = {}
    path_to_first_chunk_order: dict[str, tuple[int, int]] = {}
    for row in corpus_rows:
        cid = int(row["id"])
        p = row["path"]
        seq = int(row.get("seq") or 0)
        current = path_to_first_chunk_order.get(p)
        if current is None or (seq, cid) < current:
            path_to_first_chunk_order[p] = (seq, cid)
            path_to_first_chunk_id[p] = cid

    # Drop queries whose gold paths are all absent from the store.
    filtered: list[dict] = []
    for q in queries_normalized:
        valid_golds = [p for p in q["relevant_paths"] if p in path_to_first_chunk_id]
        if valid_golds:
            filtered.append({**q, "relevant_paths": valid_golds})
    if not filtered:
        return []

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "generate_labeled_triples_batched: %s on device=%s, %d queries over %d corpus chunks",
        base_model,
        device,
        len(filtered),
        len(corpus_texts),
    )
    model = SentenceTransformer(base_model, device=device)

    # Encode corpus + queries once.
    t0 = time.perf_counter()
    corpus_vecs = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    logger.info("Encoded %d corpus chunks in %.1fs", len(corpus_texts), time.perf_counter() - t0)
    t0 = time.perf_counter()
    query_vecs = model.encode(
        [q["query"] for q in filtered],
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    logger.info("Encoded %d labeled queries in %.1fs", len(filtered), time.perf_counter() - t0)
    corpus_vecs = corpus_vecs.to(device)
    query_vecs = query_vecs.to(device)

    # Vec top-K per query via batched matmul.
    all_vec_topk: list[list[int]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, query_vecs.size(0), matmul_batch_queries):
            batch = query_vecs[start : start + matmul_batch_queries]
            sims = batch @ corpus_vecs.T
            _top_scores, top_idx = sims.topk(min(TOP_K, sims.size(1)), dim=1)
            idx_cpu = top_idx.cpu().tolist()
            for matmul_indices in idx_cpu:
                all_vec_topk.append([corpus_ids[i] for i in matmul_indices])
    logger.info("Matmul top-K for %d queries in %.1fs", len(filtered), time.perf_counter() - t0)

    # Per-query FTS + triple mining faithful to the v5 recipe:
    # - Produce TWO RRF-fused rankings from the raw vec/FTS top-K.
    #   vec-heavy fuses with weights (0.95, 0.05); fts-heavy with
    #   (0.05, 0.95). Matches store.search() with adaptive_rrf=False
    #   in experiments/rrf_training_pairs.py.
    # - Disagreement = paths in vec-heavy top-K but not in fts-heavy
    #   (vec_only) and vice versa (fts_only).
    # - Emit one triple per (gold_path, hard_neg) pair.
    # - Cap positive/negative text at 512 chars (v5 does the same
    #   via doc_texts[path][:512]).
    vec_hi, fts_hi = _LABELED_TRAIN_WEIGHTS_VEC_HEAVY
    vec_lo, fts_lo = _LABELED_TRAIN_WEIGHTS_FTS_HEAVY
    pairs: list[dict] = []
    total_disagreements = 0
    try:
        for q, vec_top in zip(filtered, all_vec_topk, strict=True):
            qtext = q["query"]
            gold_paths = set(q["relevant_paths"])

            # FTS top-K over the main corpus (not a temp store).
            fts_top = _fts_top_k(store._conn, qtext, TOP_K)
            fts_top = [cid for cid in fts_top if cid in chunk_id_to_path]

            # RRF-fuse the raw vec + FTS rankings with the two v5
            # weight configs to produce vec-heavy and fts-heavy
            # hybrid top-K chunk id lists.
            vec_heavy_fused = _rrf_fuse(
                vec_top, fts_top, vec_hi, fts_hi, chunk_id_to_path, top_n=TOP_K
            )
            fts_heavy_fused = _rrf_fuse(
                vec_top, fts_top, vec_lo, fts_lo, chunk_id_to_path, top_n=TOP_K
            )

            # Collapse to unique paths (first occurrence wins) on each
            # fused ranking before disagreement detection.
            def _unique_paths(chunk_ids: list[int]) -> list[str]:
                seen: set[str] = set()
                out: list[str] = []
                for cid in chunk_ids:
                    p = chunk_id_to_path[cid]
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                return out

            vec_paths_ordered = _unique_paths(vec_heavy_fused)
            fts_paths_ordered = _unique_paths(fts_heavy_fused)

            vec_set = set(vec_paths_ordered)
            fts_set = set(fts_paths_ordered)
            if vec_set != fts_set:
                total_disagreements += 1

            # Hard negs from each fused side, excluding gold paths.
            vec_only_neg = [
                p for p in vec_paths_ordered if p not in fts_set and p not in gold_paths
            ]
            fts_only_neg = [
                p for p in fts_paths_ordered if p not in vec_set and p not in gold_paths
            ]

            for gold_path in gold_paths:
                gold_cid = path_to_first_chunk_id.get(gold_path)
                if gold_cid is None:
                    continue
                # v5-parity: cap positive text at 512 chars.
                positive_text = (chunk_id_to_text[gold_cid] or "")[:512]
                if not positive_text or positive_text == qtext:
                    continue

                for neg_path in vec_only_neg:
                    neg_cid = path_to_first_chunk_id.get(neg_path)
                    if neg_cid is None:
                        continue
                    neg_text = (chunk_id_to_text[neg_cid] or "")[:512]
                    if not neg_text:
                        continue
                    pairs.append(
                        {
                            "query": qtext,
                            "positive": positive_text,
                            "negative": neg_text,
                        }
                    )

                for neg_path in fts_only_neg:
                    neg_cid = path_to_first_chunk_id.get(neg_path)
                    if neg_cid is None:
                        continue
                    neg_text = (chunk_id_to_text[neg_cid] or "")[:512]
                    if not neg_text:
                        continue
                    pairs.append(
                        {
                            "query": qtext,
                            "positive": positive_text,
                            "negative": neg_text,
                        }
                    )

        logger.info(
            "Mined %d labeled triples from %d queries (%d disagreements, %.0f%%)",
            len(pairs),
            len(filtered),
            total_disagreements,
            total_disagreements / len(filtered) * 100 if filtered else 0,
        )
    finally:
        try:
            del corpus_vecs, query_vecs
        except NameError:
            pass
        from .retrain import _release_gpu_memory

        _release_gpu_memory()

    return pairs
