"""retrain_batch.py -- GPU-batched triple mining for large corpora.

The default ``vstash.retrain.generate_triples`` calls ``embed_query``
once per pseudo-query and runs two ``store.search`` invocations per
query (vec-heavy + fts-heavy). On a FiQA-sized corpus (~57k chunks)
the per-query sqlite-vec full scan dominates at ~500 ms per search on
a Colab CPU path. For ~19k queries that is ~5 hours before any
training happens.

``generate_triples_batched`` is a **fast approximation** of that
disagreement signal, not a drop-in reproduction. It trades a few
pipeline features for a 20-50x speedup:

- Vec ranking via one ``query_vecs @ corpus_vecs.T`` matmul on GPU,
  embedded with ``SentenceTransformer(base_model)`` directly. This
  bypasses ``embed_query``'s ONNX path and the ``vec_chunks`` table,
  so the resulting query/doc vectors may not match what the
  production index would produce bit-for-bit (e.g. if the store was
  reindexed with a different model between ingest and retrain).
- No ``distance_cutoff`` filter: the batched path always keeps the
  top-10 cosine hits. ``store.search`` drops results past a default
  cutoff of 1.15.
- FTS5 ``LIMIT 10`` per query. ``store.search`` uses a larger
  candidate pool (``min(K*10, max(K*3, total//3))``) before RRF.
- Raw RRF over the top-10 vec + top-10 FTS, no MMR dedup. Multiple
  chunks from the same doc can occupy the disagreement set.

The output dict shape (``{query, positive, negative}``) and the RRF
fusion math match ``generate_triples`` exactly, so ``train_mnrl`` /
``retrain`` / ``retrain_multi`` are drop-in compatible. The training
signal is similar in expectation but is a different sample of
disagreement. Use it when wall-time is the bottleneck on a big
corpus; use ``generate_triples`` when faithful reproduction of the
production search pipeline matters.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING

from .retrain import TOP_K, adaptive_rrf_weights, sample_training_chunks
from .store import RRF_K, VstashStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DISAGREEMENT_TOP = 5


def _fetch_corpus_rows(store: VstashStore) -> list[dict]:
    """Load every chunk's id + text + path for batched embedding.

    Ordered by chunk id so the returned list aligns with any later
    index-based lookups.
    """
    rows = store._conn.execute(
        "SELECT c.id, c.text, d.path FROM chunks c "
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


def _rrf_top5_paths(
    vec_chunk_ids: list[int],
    fts_chunk_ids: list[int],
    vec_weight: float,
    fts_weight: float,
    chunk_id_to_path: dict[int, str],
    top_n: int = _DISAGREEMENT_TOP,
) -> list[int]:
    """Reciprocal-rank-fuse vec and FTS rankings, return top-N chunk ids.

    Matches the RRF formula used by ``VstashStore._fuse_rrf_scores``:
    ``w * 1 / (RRF_K + rank)`` with 0-indexed rank.
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
            top_scores, top_idx = sims.topk(min(TOP_K, sims.size(1)), dim=1)
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
    for q_idx, (query_text, source) in enumerate(zip(query_texts, query_sources)):
        vec_top = all_vec_topk[q_idx]
        fts_top = _fts_top_k(store._conn, query_text, TOP_K)
        # A chunk id can appear in fts_top but not in corpus slice only
        # if the chunk was deleted after our initial fetch; guard against
        # that by filtering.
        fts_top = [cid for cid in fts_top if cid in chunk_id_to_path]

        # Adaptive RRF weights: shared with generate_triples so the
        # two signals use byte-identical ladders.
        vec_hi, fts_hi, vec_lo, fts_lo = adaptive_rrf_weights(len(query_text.split()))

        vec_heavy_top5 = _rrf_top5_paths(vec_top, fts_top, vec_hi, fts_hi, chunk_id_to_path)
        fts_heavy_top5 = _rrf_top5_paths(vec_top, fts_top, vec_lo, fts_lo, chunk_id_to_path)

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
