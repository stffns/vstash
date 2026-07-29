"""Hybrid search pipeline for ``vstash.store``: RRF fusion, MMR, miss analysis.

The largest concern of the #280 split, extracted as ``_SearchEngineMixin``. It
is a mixin: ``VstashStore`` inherits it, and the methods rely on instance state
and sibling helpers created on the concrete store (``self._conn``, the cache /
IDF state, ``self._snap``, ``self.last_best_distance``, plus ``_stem_terms`` /
``_build_idf_cache`` / ``_compute_adaptive_rrf_params`` which resolve via the
MRO), so the class is never instantiated on its own.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from ..models import (
    ExplainInfo,
    MissAnalysis,
    MissAnalysisActualResult,
    SearchResult,
    StageVerdict,
)
from ..validation import validate_search_input
from ._common import (
    _ADAPTIVE_RRF_LONG_QUERY,
    _LONG_QUERY_DISTANCE_CUTOFF,
    _PipelineTracer,
    _SQLITE_PARAM_BATCH,
    _canonicalize_added_filter,
    _compile_filter_tree,
    _cosine_sim,
    _deserialize,
    _normalize_tags,
    _serialize,
    RRF_K,
)

logger = logging.getLogger(__name__)


class _SearchEngineMixin:
    """Vector + FTS5 retrieval, adaptive RRF fusion, MMR dedup, and miss analysis."""

    # Attributes supplied by the host class (``VstashStore.__init__``).
    _conn: sqlite3.Connection
    _snap: Any

    @staticmethod
    def _compute_search_cache_key(
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        vec_weight: float | None,
        fts_weight: float | None,
        distance_cutoff: float,
        collection: str | None,
        project: str | None,
        layer: str | None,
        adaptive_rrf: bool,
        recency_boost: float,
        added_after: str | None,
        added_before: str | None,
        tags: str | list[str] | None,
        mmr_lambda: float,
        retrieval_mode: str,
        cache_epoch: int,
    ) -> int:
        """Build the search cache key from the full set of query parameters.

        ``cache_epoch`` is mixed in so a write invalidates every cached
        entry without having to scan the LRU. ``retrieval_mode`` is
        one of ``"hybrid" | "vec_only" | "fts_only"`` so queries that
        short-circuit one branch do not collide with hybrid queries of
        the same text. ``tags`` is normalized to a tuple so the same
        filter expressed as ``"a,b"`` and ``["a", "b"]`` shares a key.
        """
        # Hash the embedding as a float tuple rather than allocating a numpy
        # array + bytes blob on every search. embed_query is deterministic, so
        # the same query yields the same list and thus the same key; query_text
        # is also in the key, so distinct queries never collide.
        return hash(
            (
                tuple(query_embedding),
                query_text,
                top_k,
                vec_weight,
                fts_weight,
                distance_cutoff,
                collection,
                project,
                layer,
                adaptive_rrf,
                recency_boost,
                added_after,
                added_before,
                # Sort so that ``tags=["alpha", "beta"]`` and
                # ``tags=["beta", "alpha"]`` share the same cache entry
                # -- the SQL ``OR`` is commutative, so semantically
                # equivalent queries must hit the same cache slot.
                tuple(sorted(_normalize_tags(tags))),
                mmr_lambda,
                retrieval_mode,
                cache_epoch,
            )
        )

    @staticmethod
    def _resolve_retrieval_mode(
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None,
    ) -> Literal["hybrid", "vec_only", "fts_only"]:
        """Normalize ``retrieval_mode`` to the canonical enum.

        ``None`` resolves to ``"hybrid"`` (the default pipeline).  Any
        other value must be one of the three known modes; everything
        else raises ``ValueError`` with the expected set.

        The legacy ``fts_only=True`` bool parameter was deprecated in
        v0.33.0 and removed in v0.35.0 (#281).  Callers still passing
        it will hit a ``TypeError`` from Python's argument binder
        instead of a soft deprecation warning -- an intentional hard
        break after two releases of advance notice.
        """
        if retrieval_mode is None:
            return "hybrid"
        if retrieval_mode not in ("hybrid", "vec_only", "fts_only"):
            raise ValueError(
                f"retrieval_mode must be one of 'hybrid', 'vec_only', 'fts_only'; "
                f"got {retrieval_mode!r}"
            )
        return retrieval_mode

    @staticmethod
    def _resolve_rrf_weights(
        vec_weight: float | None, fts_weight: float | None
    ) -> tuple[float, float]:
        """Fill in missing RRF weights with the default 0.6/0.4 split.

        If only one side is provided, the other is set so the pair sums
        to 1.0. If both are ``None``, return the historical defaults.
        """
        if vec_weight is None and fts_weight is None:
            return 0.6, 0.4
        if vec_weight is None:
            assert fts_weight is not None
            fts_w = float(fts_weight)
            return 1.0 - fts_w, fts_w
        if fts_weight is None:
            return float(vec_weight), 1.0 - float(vec_weight)
        return float(vec_weight), float(fts_weight)

    @staticmethod
    def _build_fts_match_query(
        query_text: str, words: list[str] | None = None
    ) -> tuple[str, list[str]]:
        """Build an injection-safe FTS5 MATCH string from raw query text.

        Returns ``(match_string, quoted_words)``. Words of length 1 are
        dropped (FTS5 tokenizer usually strips them anyway) and each
        token is double-quoted so FTS5 cannot interpret operators like
        NEAR/NOT/OR from user input. If nothing survives, the entire
        query is quoted as a single phrase.

        ``words`` may be passed by callers that already split the query
        (e.g. ``search()``) so we do not tokenize twice on the hot path.
        """
        if words is None:
            words = query_text.split()
        quoted_words = ['"' + w.replace('"', '""') + '"' for w in words if len(w) > 1]
        if quoted_words:
            return " OR ".join(quoted_words), quoted_words
        return '"' + query_text.replace('"', '""') + '"', quoted_words

    def _fuse_rrf_scores(
        self,
        vec_rows: list,
        fts_rows: list,
        vec_weight: float,
        fts_weight: float,
        relevant_chunk_ids: set[int],
        effective_k: int,
        fts_only: bool,
        explain: bool,
    ) -> tuple[
        dict[int, dict[str, str | int | float]],
        dict[int, float],
        dict[int, float],
        dict[int, int],
    ]:
        """Run Reciprocal Rank Fusion over vector and FTS result sets.

        Returns ``(scores, explain_rrf_vec, explain_rrf_fts, explain_fts_rank)``.
        The explain maps are populated only when ``explain=True``, otherwise
        they are empty.

        FTS rows that neither pass the vector distance filter nor land in
        the top ``effective_k * 2`` FTS hits are dropped, to keep weak
        keyword noise out of the score pool when vector search has a
        strong opinion. In ``fts_only`` mode this gate is disabled.
        """
        scores: dict[int, dict[str, str | int | float]] = {}
        explain_rrf_vec: dict[int, float] = {}
        explain_rrf_fts: dict[int, float] = {}
        explain_fts_rank: dict[int, int] = {}

        for rank, row in enumerate(vec_rows):
            chunk_id: int = row["id"]
            vec_contrib = vec_weight * (1.0 / (RRF_K + rank))
            scores[chunk_id] = {
                "id": chunk_id,
                "text": row["text"],
                "title": row["title"],
                "path": row["path"],
                "chunk": row["seq"],
                "rrf": vec_contrib,
                "added_at": row["added_at"],
                "collection": row["collection"],
                "tags": row["tags"],
                "layer": row["layer"],
            }
            if explain:
                explain_rrf_vec[chunk_id] = vec_contrib

        for rank, row in enumerate(fts_rows):
            chunk_id = row["id"]
            # is_fts_top is the gate that lets FTS-only hits into the
            # score pool when they are not also in the vector candidate
            # set. In fts_only mode the vector signal is absent, so we
            # let every FTS row through up to the candidate pool.
            is_fts_top = True if fts_only else rank < effective_k * 2
            fts_contribution = fts_weight * (1.0 / (RRF_K + rank))
            if chunk_id in scores:
                scores[chunk_id]["rrf"] = float(scores[chunk_id]["rrf"]) + fts_contribution
                if explain:
                    explain_rrf_fts[chunk_id] = fts_contribution
                    explain_fts_rank[chunk_id] = rank
            elif chunk_id in relevant_chunk_ids or is_fts_top:
                scores[chunk_id] = {
                    "id": chunk_id,
                    "text": row["text"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk": row["seq"],
                    "rrf": fts_contribution,
                    "added_at": row["added_at"],
                    "collection": row["collection"],
                    "tags": row["tags"],
                    "layer": row["layer"],
                }
                if explain:
                    explain_rrf_fts[chunk_id] = fts_contribution
                    explain_fts_rank[chunk_id] = rank

        return scores, explain_rrf_vec, explain_rrf_fts, explain_fts_rank

    def _apply_recency_boost(
        self,
        ranked: list[dict[str, str | int | float]],
        recency_boost: float,
    ) -> list[dict[str, str | int | float]]:
        """Multiply each chunk's RRF score by an exponential decay factor.

        ``recency_boost=0`` disables the boost. Chunks whose ``created_at``
        is unparseable are left untouched. Returns a newly sorted list
        so callers always see post-boost ordering.
        """
        if recency_boost <= 0 or not ranked:
            return ranked

        now = datetime.now(timezone.utc)
        chunk_ids = [int(r["id"]) for r in ranked]
        # Batch the IN clause so large top_k / candidate pools don't trip
        # SQLite's default SQLITE_LIMIT_VARIABLE_NUMBER (999 on most builds).
        created_map: dict[int, datetime] = {}
        for start in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
            batch = chunk_ids[start : start + _SQLITE_PARAM_BATCH]
            placeholders = ",".join("?" * len(batch))
            for row in self._conn.execute(
                f"SELECT id, created_at FROM chunks WHERE id IN ({placeholders})",
                batch,
            ).fetchall():
                try:
                    created_map[row["id"]] = datetime.fromisoformat(row["created_at"])
                except (TypeError, ValueError):
                    pass

        for r in ranked:
            cid = int(r["id"])
            if cid in created_map:
                days_ago = max(0.0, (now - created_map[cid]).total_seconds() / 86400)
                decay = math.exp(-0.05 * days_ago)
                r["rrf"] = float(r["rrf"]) * (1.0 + recency_boost * decay)

        return sorted(ranked, key=lambda x: float(x["rrf"]), reverse=True)

    @staticmethod
    def _build_search_results(
        ranked: list[dict[str, str | int | float]],
        *,
        explain: bool,
        explain_vec: dict[int, tuple[int, float]],
        explain_rrf_vec: dict[int, float],
        explain_rrf_fts: dict[int, float],
        explain_fts_rank: dict[int, int],
        quoted_words: list[str],
        vec_weight: float,
        fts_weight: float,
    ) -> list[SearchResult]:
        """Materialize ``SearchResult`` instances with optional ExplainInfo.

        All rounding is applied here so callers see stable floating-point
        values (scores to 6 decimals, distances to 4).
        """
        explain_map: dict[int, ExplainInfo] = {}
        if explain:
            fts_terms = [w.strip('"') for w in quoted_words] if quoted_words else []
            for r in ranked:
                cid = int(r["id"])
                vec_info = explain_vec.get(cid)
                rrf_vec_val = round(explain_rrf_vec.get(cid, 0.0), 6)
                rrf_fts_val = round(explain_rrf_fts.get(cid, 0.0), 6)
                explain_map[cid] = ExplainInfo(
                    vec_rank=vec_info[0] if vec_info else None,
                    vec_distance=round(vec_info[1], 4) if vec_info else None,
                    fts_rank=explain_fts_rank.get(cid),
                    rrf_vec=rrf_vec_val,
                    rrf_fts=rrf_fts_val,
                    rrf_total=round(rrf_vec_val + rrf_fts_val, 6),
                    mmr_penalty=round(float(r.get("_mmr_penalty", 0.0)), 4),
                    fts_terms=fts_terms,
                    rrf_vec_weight=vec_weight,
                    rrf_fts_weight=fts_weight,
                )

        return [
            SearchResult(
                chunk_id=int(r["id"]),
                text=str(r["text"]),
                title=str(r["title"]),
                path=str(r["path"]),
                chunk=int(r["chunk"]),
                score=round(float(r["rrf"]), 6),
                explain=explain_map.get(int(r["id"])) if explain else None,
                added_at=r.get("added_at"),
                collection=r.get("collection"),
                tags=r.get("tags"),
                layer=r.get("layer"),
            )
            for r in ranked
        ]

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        distance_cutoff: float = 1.3225,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        explain: bool = False,
        adaptive_rrf: bool = True,
        recency_boost: float = 0.0,
        added_after: str | None = None,
        added_before: str | None = None,
        tags: str | list[str] | None = None,
        filters: dict | None = None,
        mmr_lambda: float = 0.5,
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
        exact_match: str | None = None,
        exact_match_case_sensitive: bool = False,
        _tracer: _PipelineTracer | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: vector (semantic) + FTS5 (keyword) combined with RRF.

        RRF score = vec_weight * 1/(k+rank_vec) + fts_weight * 1/(k+rank_fts)

        Results are filtered by vector distance — chunks whose distance from
        the query is more than ``distance_cutoff`` times the best (closest)
        distance are discarded before RRF scoring.  This prevents irrelevant
        noise (e.g. Art of War appearing in deep learning queries).

        Args:
            query_embedding: Query vector from the embedding model.
            query_text: Raw query text for FTS5 keyword matching.
            top_k: Number of results to return.
            vec_weight: Weight for vector search contribution.
            fts_weight: Weight for keyword search contribution.
            distance_cutoff: Maximum allowed ratio of distance to best distance.
                Chunks with distance > best_distance * distance_cutoff are dropped.
                Default 1.3225 = 1.15^2; under BGE unit-normalized
                embeddings the cosine-distance ratio relates to the
                legacy v1 L2-distance ratio by a square (#272), so this
                value preserves v1 cutoff semantics.
            collection: If set, restrict search to documents in this collection.
            project: If set, restrict search to documents with this project tag.
            layer: If set, restrict search to documents with this layer tag.
            exact_match: Optional substring that each returned chunk's
                ``text`` must contain. Applied as a post-filter after
                the full pipeline so the upstream candidate pool does
                NOT know about this constraint -- callers who need a
                guaranteed top_k under a selective substring should
                request a larger ``top_k`` and accept that fewer
                results may come back. Bypasses FTS5 tokenization, so
                literal strings with punctuation / casing / code
                identifiers survive (unlike the FTS5 keyword path
                which stems and lowercases). #106, 2026-04-21.
            exact_match_case_sensitive: Toggle for the above. Default
                ``False`` does a casefold comparison which matches
                typical retrieval-filter UX expectations.
            retrieval_mode: Which search branches to run (default
                ``"hybrid"``). Three values:

                - ``"hybrid"`` (default): vector ANN + FTS5 + adaptive
                  RRF + distance cutoff + MMR. This is the pipeline the
                  paper and README benchmarks are measured against.
                - ``"fts_only"`` (#152): skip the vector ANN scan,
                  distance cutoff, and adaptive RRF. FTS5 hits still
                  flow through RRF scoring (``vec_weight=0.0``,
                  ``fts_weight=1.0``), MMR, recency boost, and context
                  expansion — not a raw BM25 dump. Useful for queries
                  with diffuse vector representations (cross-lingual,
                  highly technical) or as a fallback when the vector
                  pool is expected to be empty.
                - ``"vec_only"`` (#275): symmetric to ``"fts_only"``.
                  Skip the FTS5 keyword search; force
                  ``vec_weight=1.0``, ``fts_weight=0.0``. Useful when
                  the corpus has no meaningful keyword signal (tabular
                  data, code where identifiers are noise, cross-lingual
                  corpora where tokenization disagrees with the query)
                  or for benchmarking / ranking debug.

        Returns:
            Ranked list of SearchResult ordered by descending score.
        """
        _mode = self._resolve_retrieval_mode(retrieval_mode)
        _fts_only = _mode == "fts_only"
        _vec_only = _mode == "vec_only"
        # Fail-safe validation at the public API boundary (#133).  Runs
        # before any work — if the caller passed a 50k-token query or a
        # negative top_k, we reject it here instead of crashing inside
        # SQLite or the embedding model with a cryptic error.  Cost is
        # a handful of comparisons.
        validate_search_input(
            query_text=query_text,
            top_k=top_k,
            distance_cutoff=distance_cutoff,
            recency_boost=recency_boost,
            limits=self._limits,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
        )

        # --- Query cache key ---
        # Skip the cache when an exact_match filter is ACTIVE (non-empty).
        # An empty string and None are no-ops against the post-filter and
        # stay cacheable. Including the substring in the cache key would
        # be correct but blow up key cardinality and reduce reuse; the
        # filter is rare enough that the simpler skip is preferable.
        _cache_key: int | None = None
        _cache_max = self._cache_config.query_cache_size
        if _cache_max > 0 and _tracer is None and not explain and not exact_match:
            _cache_key = self._compute_search_cache_key(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
                distance_cutoff=distance_cutoff,
                collection=collection,
                project=project,
                layer=layer,
                adaptive_rrf=adaptive_rrf,
                recency_boost=recency_boost,
                added_after=added_after,
                added_before=added_before,
                tags=tags,
                mmr_lambda=mmr_lambda,
                retrieval_mode=_mode,
                cache_epoch=self._cache_epoch,
            )

        # Per-search miss-analysis tracker (#108).  The tracer is owned
        # by the caller (miss_analysis) and passed in; this keeps the
        # tracking state thread-local to the caller and zero-cost on
        # the regular search hot path (tracer is None).
        #
        # Internal: _tracer is a hook for miss_analysis() only.  Power
        # users of VstashStore should not pass it directly.
        track_target: int | None = _tracer.target if _tracer is not None else None

        # Observability: we wrap the entire body in try/finally so that
        # latency and slow_queries_total are recorded even when the body
        # raises.  An earlier version used Timer.__enter__/__exit__ with
        # no finally — exceptions silently dropped the histogram write,
        # causing searches_total to drift ahead of the histogram count
        # (exactly the metrics skew you do NOT want when debugging a
        # production incident).
        from ..metrics import registry

        _search_start = time.perf_counter()
        _result_count = 0
        # vec_weight/fts_weight may still be None here — initialize to
        # 0.0 so the slow-query log in finally doesn't crash if we fail
        # before adaptive RRF has resolved them.
        _vec_weight_observed: float = 0.0
        _fts_weight_observed: float = 0.0
        registry.counter_inc("searches_total")
        try:
            # --- Cache hit (inside try/finally so observability is recorded) ---
            if _cache_key is not None:
                cached: list[SearchResult] | None = None
                with self._cache_lock:
                    if _cache_key in self._query_cache:
                        self._query_cache.move_to_end(_cache_key)
                        cached = list(self._query_cache[_cache_key])
                if cached is not None:
                    _result_count = len(cached)
                    registry.counter_inc("query_cache_hits_total")
                    return cached

            # Branch short-circuits for non-hybrid modes.
            #
            # FTS-only (#152): bypass vector search entirely. Force
            # weights to (0.0, 1.0) and disable adaptive RRF so the
            # pipeline cannot silently re-enable the vector path. The
            # vector search, distance cutoff, and relevance filter
            # blocks below are guarded on ``not _fts_only`` or on
            # ``vec_rows`` being non-empty, so downstream MMR / scoring
            # runs unchanged.
            #
            # Vec-only (#275): symmetric. Bypass the FTS5 query. Force
            # weights to (1.0, 0.0) and disable adaptive RRF so the
            # FTS path cannot re-enable itself via IDF signals. The
            # ``fts_rows`` block below is skipped when ``_vec_only``
            # so no FTS hits ever enter the RRF fusion.
            if _fts_only:
                vec_weight = 0.0
                fts_weight = 1.0
                adaptive_rrf = False
            elif _vec_only:
                vec_weight = 1.0
                fts_weight = 0.0
                # Long-query cutoff relaxation must still fire here.
                # _compute_adaptive_rrf_params owns the cutoff for
                # >50-word queries, but it's gated on adaptive_rrf which
                # we're about to turn off. Without this, ArguAna-style
                # long queries collapse to ~0 NDCG in vec_only mode
                # (best_dist ~0.2 in cosine, default cutoff 1.3225 yields
                # threshold ~0.265, rejecting nearly every candidate
                # past rank 0). Mirror the hybrid long-query relaxation
                # so vec_only ablations are honest, and gate it on
                # adaptive_rrf so an explicit adaptive_rrf=False from
                # the caller still opts out (parity with hybrid).
                if adaptive_rrf and len(query_text.split()) > _ADAPTIVE_RRF_LONG_QUERY:
                    distance_cutoff = _LONG_QUERY_DISTANCE_CUTOFF
                adaptive_rrf = False

            # Adaptive RRF: compute weights from query characteristics (IDF + length)
            # Skip if caller provided explicit weights
            if adaptive_rrf and vec_weight is None and fts_weight is None:
                vec_weight, fts_weight, distance_cutoff = self._compute_adaptive_rrf_params(
                    query_text, default_cutoff=distance_cutoff
                )
            vec_weight, fts_weight = self._resolve_rrf_weights(vec_weight, fts_weight)

            # Adaptive candidate pool — avoid pulling half the corpus on small DBs
            effective_k = top_k
            total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            candidate_pool = min(effective_k * 10, max(effective_k * 3, total_chunks // 3))

            # --- Build metadata filter ---
            vec_clause, col_clause, filter_params = self._build_doc_filter(
                collection=collection,
                project=project,
                layer=layer,
                tags=tags,
                added_after=added_after,
                added_before=added_before,
                filters=filters,
            )

            # --- Vector search ---
            vec_rows: list = []
            if _fts_only:
                # #152: skip vector ANN entirely. No candidates, no
                # distances, no distance-cutoff work downstream. Set
                # last_best_distance to the worst possible so the
                # distance-based relevance tier renders "low" — the
                # caller opted out of the vector signal.
                self.last_best_distance = 2.0
                if track_target is not None and _tracer is not None:
                    # Record vector_search as *passed* (not failed) so
                    # the downstream "both generators missed → invisible
                    # to RRF" logic in miss_analysis does not
                    # misattribute the drop to rrf_fusion.  The detail
                    # string makes the intent explicit: the stage was
                    # intentionally skipped by the caller, not that it
                    # ran and found nothing.
                    _tracer.record(
                        "vector_search",
                        passed=True,
                        detail="retrieval_mode='fts_only': vector search intentionally skipped by caller",
                    )
                    _tracer.record(
                        "distance_cutoff",
                        passed=True,
                        detail="retrieval_mode='fts_only': no distance cutoff applied",
                    )
            elif self._snap is not None and len(self._snap) > 0:
                # The two snapvec backends disagree on their return
                # semantics and this shared call site has to reconcile
                # them to the cosine-distance contract the rest of
                # the pipeline assumes (#289).
                #
                # - Flat ``snapvec`` (``SnapIndex``) returns
                #   ``(id, similarity)`` in ``[-1, 1]`` sorted
                #   descending and needs the ``1 - similarity``
                #   conversion here.
                # - ``snapvec-ivfpq`` (``IVFPQBackend``) already
                #   applies ``1 - similarity`` inside its own
                #   ``search()`` (see ``vectorbackend/snapvec_ivfpq.py``) and
                #   must not be double-inverted.
                #
                # ``min / max`` clamps keep the invariant that
                # ``last_best_distance`` and ``relevance_tier`` both
                # assume (``[0, 2]``), even if an implementation
                # quirk yields a value slightly outside ``[-1, 1]``.
                snap_results = self._snap.search(
                    np.array(query_embedding, dtype=np.float32), k=candidate_pool
                )
                snap_ids = [int(r[0]) for r in snap_results]
                if self._vector_backend == "snapvec":
                    snap_dists = {
                        int(r[0]): min(2.0, max(0.0, 1.0 - float(r[1]))) for r in snap_results
                    }
                else:
                    snap_dists = {int(r[0]): min(2.0, max(0.0, float(r[1]))) for r in snap_results}

                if snap_ids:
                    placeholders = ",".join("?" * len(snap_ids))
                    # Build filter clause adapted for snapvec (no v.rowid)
                    snap_filter = vec_clause.replace("v.rowid", "c.id") if vec_clause else ""
                    rows = self._conn.execute(
                        f"""
                        SELECT c.id, c.text, d.title, d.path, c.seq, d.added_at, d.collection, d.tags, d.layer
                        FROM chunks c
                        JOIN documents d ON d.id = c.doc_id
                        WHERE c.id IN ({placeholders})
                          {snap_filter}
                        """,
                        [*snap_ids, *filter_params],
                    ).fetchall()
                    # Build dict for fast lookup, attach distance, preserve snap order
                    row_by_id = {r["id"]: dict(r) for r in rows}
                    vec_rows = []
                    for sid in snap_ids:
                        if sid in row_by_id:
                            entry = row_by_id[sid]
                            entry["distance"] = snap_dists.get(sid, 2.0)
                            vec_rows.append(entry)
                else:
                    vec_rows = []
            else:
                vec_rows = self._conn.execute(
                    f"""
                    SELECT c.id, c.text, d.title, d.path, c.seq, v.distance, d.added_at, d.collection, d.tags, d.layer
                    FROM vec_chunks v
                    JOIN chunks c ON c.id = v.rowid
                    JOIN documents d ON d.id = c.doc_id
                    WHERE v.embedding MATCH ?
                      AND k = ?
                      {vec_clause}
                    ORDER BY v.distance
                    """,
                    [_serialize(query_embedding), candidate_pool, *filter_params],
                ).fetchall()

            # --- Track: vector search stage ---
            # In fts_only mode the "vector_search" and "distance_cutoff"
            # stages were already recorded as skipped above; don't
            # overwrite them with a generic "not found" verdict.
            if track_target is not None and not _fts_only:
                target_vec_rank: int | None = None
                target_vec_distance: float | None = None
                for i, row in enumerate(vec_rows):
                    if int(row["id"]) == track_target:
                        target_vec_rank = i
                        target_vec_distance = float(row["distance"])
                        break
                if target_vec_rank is not None:
                    _tracer.record(
                        "vector_search",
                        passed=True,
                        rank=target_vec_rank,
                        score=target_vec_distance,
                        detail=(
                            f"Found in vector candidate pool at rank {target_vec_rank + 1}/{len(vec_rows)} "
                            f"with distance {target_vec_distance:.4f}"
                        ),
                    )
                else:
                    _tracer.record(
                        "vector_search",
                        passed=False,
                        rank=None,
                        score=None,
                        detail=(
                            f"Not in vector candidate pool of size {candidate_pool}. "
                            "The chunk's embedding is too far from the query embedding."
                        ),
                        counterfactual="Would need candidate_pool > current rank to appear",
                    )

            # --- Filter by vector distance gap ---
            # The best (closest) result has the smallest distance.
            # Remove results that are semantically too far from the ideal match.
            if vec_rows:
                best_distance = float(vec_rows[0]["distance"])
                self.last_best_distance = best_distance
                # Capture target distance BEFORE filtering for tracking
                target_dist_before: float | None = None
                if track_target is not None:
                    for r in vec_rows:
                        if int(r["id"]) == track_target:
                            target_dist_before = float(r["distance"])
                            break
                threshold = best_distance * distance_cutoff
                cutoff_applied = best_distance > 0
                if cutoff_applied:
                    vec_rows = [r for r in vec_rows if float(r["distance"]) <= threshold]
                # Track distance cutoff verdict whenever we have a before-distance
                # (i.e., the target was in vec_rows pre-filter)
                if track_target is not None and target_dist_before is not None:
                    target_in_after = any(int(r["id"]) == track_target for r in vec_rows)
                    if not cutoff_applied:
                        # best_distance == 0: cutoff logic is skipped entirely.
                        # Surface the target's absolute distance so users know
                        # whether the bypass is a lucky rescue or a "carried by
                        # the loophole" situation (high-distance chunks are
                        # getting through because a perfect match exists).
                        _tracer.record(
                            "distance_cutoff",
                            passed=True,
                            score=target_dist_before,
                            detail=(
                                f"Distance cutoff bypassed (best_distance=0, perfect "
                                f"match exists). Target distance={target_dist_before:.4f}; "
                                f"this would normally require cutoff ratio > "
                                f"{target_dist_before * 100:.1f} to pass."
                            ),
                        )
                    elif target_in_after:
                        _tracer.record(
                            "distance_cutoff",
                            passed=True,
                            score=target_dist_before,
                            detail=(
                                f"Distance {target_dist_before:.4f} ≤ threshold "
                                f"{threshold:.4f} (best={best_distance:.4f} × cutoff={distance_cutoff:.2f})"
                            ),
                        )
                    else:
                        needed_cutoff = (
                            target_dist_before / best_distance
                            if best_distance > 0
                            else float("inf")
                        )
                        _tracer.record(
                            "distance_cutoff",
                            passed=False,
                            score=target_dist_before,
                            detail=(
                                f"Distance {target_dist_before:.4f} > threshold "
                                f"{threshold:.4f} (best={best_distance:.4f} × cutoff={distance_cutoff:.2f})"
                            ),
                            counterfactual=(
                                f"Would have passed with distance_cutoff ≥ {needed_cutoff:.2f}"
                            ),
                        )
            else:
                self.last_best_distance = 2.0  # max cosine distance = worst case

            # Track which chunk IDs passed the vector distance filter
            relevant_chunk_ids: set[int] = {row["id"] for row in vec_rows}

            # --- Explain: capture per-chunk vector rank and distance ---
            _explain_vec: dict[int, tuple[int, float]] = {}  # chunk_id -> (rank, distance)
            if explain:
                for rank, row in enumerate(vec_rows):
                    _explain_vec[int(row["id"])] = (rank, float(row["distance"]))

            # --- FTS5 search ---
            words = query_text.split()
            safe_query, quoted_words = self._build_fts_match_query(query_text, words)
            if _vec_only:
                # #275: caller asked to bypass keyword search entirely.
                # Produce no FTS rows; downstream RRF scoring uses
                # weights (1.0, 0.0) so the absence cannot re-weight
                # anything. Record the stage in the tracer as an
                # intentional skip (not a miss) for parity with the
                # fts_only path above.
                fts_rows = []
                if track_target is not None and _tracer is not None:
                    _tracer.record(
                        "fts_search",
                        passed=True,
                        detail="retrieval_mode='vec_only': FTS5 search intentionally skipped by caller",
                    )
            else:
                try:
                    fts_rows = self._conn.execute(
                        f"""
                        SELECT c.id, c.text, d.title, d.path, c.seq,
                               rank as fts_rank, d.added_at, d.collection, d.tags, d.layer
                        FROM fts_chunks f
                        JOIN chunks c ON c.id = f.rowid
                        JOIN documents d ON d.id = c.doc_id
                        WHERE fts_chunks MATCH ?
                          {col_clause}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        [safe_query, *filter_params, candidate_pool],
                    ).fetchall()
                except sqlite3.OperationalError:
                    # FTS5 query syntax error (e.g. single char) — fall back to no FTS
                    fts_rows = []

                # --- Track: FTS search stage ---
                if track_target is not None:
                    target_fts_rank: int | None = None
                    for i, row in enumerate(fts_rows):
                        if int(row["id"]) == track_target:
                            target_fts_rank = i
                            break
                    stemmed_terms = self._stem_terms(words) if words else []
                    if target_fts_rank is not None:
                        _tracer.record(
                            "fts_search",
                            passed=True,
                            rank=target_fts_rank,
                            detail=(
                                f"Matched FTS at rank {target_fts_rank + 1}/{len(fts_rows)}. "
                                f"Stemmed query terms: {stemmed_terms}"
                            ),
                        )
                    else:
                        _tracer.record(
                            "fts_search",
                            passed=False,
                            detail=(
                                f"Did not match FTS5. Stemmed query terms: {stemmed_terms}. "
                                "The chunk text does not contain any of these stems "
                                "(after porter stemming)."
                            ),
                            counterfactual="Would need a query containing words from the chunk's vocabulary",
                        )

            # --- Adaptive fallback: vector pool empty (#156) ---
            # If the vector pool is empty — either the initial ANN scan
            # returned nothing, the metadata filter eliminated everything,
            # or the distance cutoff was so tight that no chunk survived
            # — the pipeline would otherwise fuse an empty vec list with
            # the FTS list using the default or adaptive RRF weights
            # (e.g. 0.6/0.4). The FTS rows would then be scored with
            # only ``fts_weight * 1/(k+rank)``, which degrades the
            # absolute RRF score unnecessarily — a literal-match FTS
            # hit at rank 0 gets ~0.0067 instead of the 0.0167 it would
            # earn under pure FTS weighting.  Detect the condition and
            # collapse to FTS-only scoring explicitly, so downstream
            # consumers of ``SearchResult.score`` see a meaningful value
            # and the relevance-tier signal correctly reports "low"
            # instead of carrying a stale high-confidence distance.
            #
            # First, always reset ``last_best_distance`` when vec_rows
            # is empty — whether or not there are FTS results. Without
            # this, a query where both pools are empty (e.g. corpus
            # doesn't contain the query at all) would report whatever
            # ``last_best_distance`` held from a previous query,
            # lying about the current query's confidence.  Flagged by
            # Gemini review on #156.
            if not vec_rows:
                self.last_best_distance = 2.0

            # Then, if there are FTS results to boost, actually apply
            # the weight override and fire the observability signal.
            if not vec_rows and fts_rows:
                registry.counter_inc("adaptive_rrf_vector_empty_fallback_total")
                vec_weight = 0.0
                fts_weight = 1.0
                if track_target is not None and _tracer is not None:
                    _tracer.record(
                        "adaptive_fallback",
                        passed=True,
                        detail=(
                            "vector pool empty after distance cutoff: collapsed "
                            "to FTS-only scoring (vec_weight=0.0, fts_weight=1.0)"
                        ),
                    )

            # --- Reciprocal Rank Fusion ---
            scores, _explain_rrf_vec, _explain_rrf_fts, _explain_fts_rank = self._fuse_rrf_scores(
                vec_rows=vec_rows,
                fts_rows=fts_rows,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
                relevant_chunk_ids=relevant_chunk_ids,
                effective_k=effective_k,
                fts_only=_fts_only,
                explain=explain,
            )

            # Sort by RRF score descending
            ranked = sorted(scores.values(), key=lambda x: float(x["rrf"]), reverse=True)

            # --- Track: RRF fusion stage ---
            if track_target is not None:
                target_rrf_rank: int | None = None
                target_rrf_score: float | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_rrf_rank = i
                        target_rrf_score = float(r["rrf"])
                        break
                if target_rrf_rank is not None:
                    _tracer.record(
                        "rrf_fusion",
                        passed=True,
                        rank=target_rrf_rank,
                        score=target_rrf_score,
                        detail=(
                            f"After RRF fusion: rank {target_rrf_rank + 1}/{len(ranked)}, "
                            f"combined score {target_rrf_score:.6f}"
                        ),
                    )
                else:
                    _tracer.record(
                        "rrf_fusion",
                        passed=False,
                        detail=(
                            "Eliminated before RRF fusion (failed both vector and FTS, "
                            "or filtered by metadata)"
                        ),
                    )

            # --- Recency boost (temporal decay) ---
            # Biases scores toward recent content, useful for agentic
            # memory where latest context tends to be most relevant.
            ranked = self._apply_recency_boost(ranked, recency_boost)

            # --- Track: recency boost stage ---
            if track_target is not None and recency_boost > 0:
                target_after_boost: int | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_after_boost = i
                        break
                if target_after_boost is not None:
                    _tracer.record(
                        "recency_boost",
                        passed=True,
                        rank=target_after_boost,
                        detail=(
                            f"After recency_boost={recency_boost}: rank {target_after_boost + 1}"
                        ),
                    )

            # --- Pre-MMR ranks (for tracking what MMR removes) ---
            pre_mmr_rank_of_target: int | None = None
            if track_target is not None:
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        pre_mmr_rank_of_target = i
                        break

            # Intra-document MMR deduplication: allow multiple chunks from the
            # same document only when they are semantically diverse (e.g. different
            # chapters of a book).  Chunks from different documents compete purely
            # on score — no cross-document penalty.
            ranked = self._mmr_dedup(ranked, top_k, mmr_lambda=mmr_lambda, _explain=explain)

            # --- Track: MMR + top_k cutoff ---
            if track_target is not None:
                target_final_rank: int | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_final_rank = i
                        break
                # MMR verdict
                if pre_mmr_rank_of_target is not None and target_final_rank is None:
                    _tracer.record(
                        "mmr_dedup",
                        passed=False,
                        rank=pre_mmr_rank_of_target,
                        detail=(
                            f"Was rank {pre_mmr_rank_of_target + 1} pre-MMR but eliminated by "
                            "intra-document MMR deduplication (another chunk from the same "
                            f"document was already selected and they're too similar). "
                            f"Current mmr_lambda={mmr_lambda:.2f}."
                        ),
                        counterfactual=(
                            f"Try a higher mmr_lambda (>{mmr_lambda:.2f}) for less "
                            "aggressive dedup, or mmr_lambda=1.0 to disable MMR entirely."
                        ),
                    )
                elif target_final_rank is not None:
                    _tracer.record(
                        "mmr_dedup",
                        passed=True,
                        rank=target_final_rank,
                        detail=f"Survived MMR dedup at rank {target_final_rank + 1}",
                    )
                # top_k cutoff verdict
                if target_final_rank is not None:
                    if target_final_rank < top_k:
                        _tracer.record(
                            "top_k_cutoff",
                            passed=True,
                            rank=target_final_rank,
                            detail=f"Final rank {target_final_rank + 1} ≤ top_k={top_k}",
                        )
                    else:
                        _tracer.record(
                            "top_k_cutoff",
                            passed=False,
                            rank=target_final_rank,
                            detail=f"Final rank {target_final_rank + 1} > top_k={top_k}",
                            counterfactual=f"Would appear with top_k ≥ {target_final_rank + 1}",
                        )

            results = self._build_search_results(
                ranked,
                explain=explain,
                explain_vec=_explain_vec,
                explain_rrf_vec=_explain_rrf_vec,
                explain_rrf_fts=_explain_rrf_fts,
                explain_fts_rank=_explain_fts_rank,
                quoted_words=quoted_words,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
            )

            # Exact-match text filter (#106). Applied post-pipeline on
            # the final ranked list: a chunk that otherwise would have
            # made the top-k is dropped unless its ``text`` contains
            # ``exact_match`` as a substring. Bypasses FTS5 tokenization
            # so identifiers / phrases that stem or tokenize differently
            # than the user typed survive -- e.g. ``exact_match="rate-
            # limit"`` requires the literal string, not ``rate limit``.
            #
            # Case-sensitive opt-in via ``exact_match_case_sensitive``.
            # Default is case-insensitive which is the usual UX
            # expectation for retrieval filters.
            #
            # Post-filter means the candidate pool upstream was sized
            # without knowing about this filter, so callers who need a
            # guaranteed top_k under a selective substring should pass a
            # larger ``top_k`` (e.g. 3x) or accept a smaller result set.
            if exact_match:
                if exact_match_case_sensitive:
                    results = [r for r in results if exact_match in r.text]
                else:
                    needle = exact_match.casefold()
                    results = [r for r in results if needle in r.text.casefold()]

            # Stash values the finally block needs to log slow queries
            # with accurate data.
            _result_count = len(results)
            _vec_weight_observed = float(vec_weight)
            _fts_weight_observed = float(fts_weight)

            # _tracer.verdicts has been populated in-place by the record()
            # calls above when tracking is enabled.  No store-level state.

            if _cache_key is not None:
                with self._cache_lock:
                    self._query_cache[_cache_key] = list(results)
                    while len(self._query_cache) > _cache_max:
                        self._query_cache.popitem(last=False)

            return results
        finally:
            # Record latency and slow-query telemetry for every search,
            # including those that raised.  _result_count is 0 if we
            # failed before reaching the results-building block.
            elapsed_ms = (time.perf_counter() - _search_start) * 1000.0
            registry.histogram_observe("search_latency_ms", elapsed_ms)
            if elapsed_ms >= self._observability.slow_query_ms:
                registry.counter_inc("slow_queries_total")
                # Sanitize query preview: strip control characters and
                # non-printable codepoints so log shippers don't choke
                # on embedded NULs or escape sequences.
                preview = "".join(c if c.isprintable() else "?" for c in query_text[:60])
                if len(query_text) > 60:
                    preview += "…"
                logger.warning(
                    "slow query: %.1fms query=%s results=%d vec_w=%.2f fts_w=%.2f",
                    elapsed_ms,
                    preview,
                    _result_count,
                    _vec_weight_observed,
                    _fts_weight_observed,
                )

    # ------------------------------------------------------------------ #
    # Miss analysis (#108) — explain why a doc did NOT appear in top-k     #
    # ------------------------------------------------------------------ #

    def miss_analysis(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        expected_path: str | None = None,
        expected_chunk_id: int | None = None,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> MissAnalysis:
        """Diagnose why an expected document did not appear in search results.

        Runs the same search pipeline as ``search()`` but with per-stage
        tracking enabled, then builds a structured ``MissAnalysis`` with
        the trace, the actual top-k for context, and rule-based
        suggestions.

        Args:
            query_embedding: Query vector (same as ``search()``).
            query_text: Raw query text (same as ``search()``).
            expected_path: Path of the document the caller expected to see.
                Either this or ``expected_chunk_id`` must be provided.
            expected_chunk_id: Specific chunk id to track instead of
                resolving from a path.  Useful when the caller already has
                a chunk id from a previous search.
            top_k: Number of results to evaluate (same as ``search()``).
            collection/project/layer: Same metadata filters as ``search()``.

        Returns:
            ``MissAnalysis`` with stage verdicts, actual top-k, and
            suggestions.

        Raises:
            ValueError: if neither ``expected_path`` nor ``expected_chunk_id``
                is given, or the path/id resolves to nothing.
        """
        if expected_path is None and expected_chunk_id is None:
            raise ValueError("Provide either expected_path or expected_chunk_id")

        # Resolve target chunk id, tracking how the choice was made so the
        # caller knows whether the trace represents the whole document or
        # just its best-matching chunk.
        target_chunk_id: int
        resolved_path: str | None = expected_path
        target_resolution: str
        total_chunks_in_doc = 1

        if expected_chunk_id is not None:
            row = self._conn.execute(
                "SELECT c.id, d.path, d.id AS doc_id "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
                [int(expected_chunk_id)],
            ).fetchone()
            if row is None:
                raise ValueError(f"Chunk id {expected_chunk_id} not found")
            target_chunk_id = int(row["id"])
            resolved_path = str(row["path"])
            target_resolution = "explicit_id"
            # Count siblings in the same document for caller transparency
            sibling_count = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", [row["doc_id"]]
            ).fetchone()[0]
            total_chunks_in_doc = int(sibling_count)
        else:
            # Apply the same metadata filters used by the search pipeline
            # so that miss_analysis("rate limits", expected_path="/x.md",
            # collection="docs") resolves to the chunk(s) of /x.md that
            # belong to collection="docs", not a cross-collection copy.
            filter_conditions, filter_params = self._get_filter_conditions(
                "d",
                collection=collection,
                project=project,
                layer=layer,
            )
            where_extras = ""
            if filter_conditions:
                where_extras = " AND " + " AND ".join(filter_conditions)
            doc_chunks = self._conn.execute(
                f"""
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.path = ?
                  {where_extras}
                """,
                [expected_path, *filter_params],
            ).fetchall()
            if not doc_chunks:
                # Be helpful: differentiate "path exists but filtered out"
                # from "path truly not in the store".
                exists_elsewhere = self._conn.execute(
                    "SELECT 1 FROM documents WHERE path = ? LIMIT 1", [expected_path]
                ).fetchone()
                if exists_elsewhere is not None:
                    raise ValueError(
                        f"Path {expected_path!r} exists but is excluded by the "
                        "collection/project/layer filters passed to miss_analysis(). "
                        "Re-run without filters to diagnose."
                    )
                raise ValueError(f"No chunks found for path: {expected_path}")
            chunk_ids = [int(r["id"]) for r in doc_chunks]
            total_chunks_in_doc = len(chunk_ids)

            if total_chunks_in_doc == 1:
                target_chunk_id = chunk_ids[0]
                target_resolution = "only_chunk"
            else:
                # Multi-chunk document: pick the chunk with the smallest
                # distance to the query.  This biases the trace toward
                # the best-case chunk; target_resolution reflects that.
                #
                # Batch the IN clause to respect SQLITE_LIMIT_VARIABLE_NUMBER
                # (default 999).  Books / long manuals can have 1000+ chunks,
                # which would otherwise blow the limit.
                best_rowid: int | None = None
                best_dist: float = float("inf")
                q_ser = _serialize(query_embedding)
                try:
                    for start in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
                        batch = chunk_ids[start : start + _SQLITE_PARAM_BATCH]
                        placeholders = ",".join("?" * len(batch))
                        rows = self._conn.execute(
                            f"""
                            SELECT v.rowid, v.distance
                            FROM vec_chunks v
                            WHERE v.embedding MATCH ?
                              AND v.rowid IN ({placeholders})
                              AND k = ?
                            ORDER BY v.distance
                            LIMIT 1
                            """,
                            [q_ser, *batch, len(batch)],
                        ).fetchall()
                        if rows and float(rows[0]["distance"]) < best_dist:
                            best_dist = float(rows[0]["distance"])
                            best_rowid = int(rows[0]["rowid"])
                except sqlite3.Error:
                    best_rowid = None
                if best_rowid is not None:
                    target_chunk_id = best_rowid
                else:
                    logging.getLogger(__name__).warning(
                        "miss_analysis: vec lookup failed for multi-chunk doc "
                        "'%s'; falling back to first chunk id",
                        expected_path,
                    )
                    target_chunk_id = chunk_ids[0]
                target_resolution = "best_of_n"

        # Run search with a caller-owned tracer — the buffer is local
        # to this call, so concurrent miss_analysis() on the same store
        # cannot corrupt each other.
        tracer = _PipelineTracer(target_chunk_id)
        results = self.search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            _tracer=tracer,
        )

        # Determine if the expected doc appeared in results
        appeared = False
        final_rank: int | None = None
        for i, r in enumerate(results):
            if r.chunk_id == target_chunk_id or (
                resolved_path is not None and r.path == resolved_path
            ):
                appeared = True
                final_rank = i
                break

        # Build StageVerdict list from this caller's verdicts
        stage_verdicts = [
            StageVerdict(
                stage=v["stage"],  # type: ignore[arg-type]
                passed=bool(v["passed"]),
                rank=v["rank"],  # type: ignore[arg-type]
                score=v["score"],  # type: ignore[arg-type]
                detail=str(v["detail"]),
                counterfactual=v["counterfactual"],  # type: ignore[arg-type]
            )
            for v in tracer.verdicts
        ]

        # Find the stage that actually eliminated the chunk from the
        # pipeline.  Tricky: vector_search and fts_search are INDEPENDENT
        # candidate generators — failing one does NOT drop the chunk,
        # because the other modality can still surface it into RRF.
        # Only the gate stages (distance_cutoff, rrf_fusion, mmr_dedup,
        # top_k_cutoff) actually remove a chunk from the pipeline.
        #
        # Special case: if BOTH vector_search and fts_search failed, the
        # chunk never reached RRF and the functional drop_at is "rrf_fusion"
        # (it's invisible to the fusion layer).
        _GATE_STAGES = {"distance_cutoff", "rrf_fusion", "mmr_dedup", "top_k_cutoff"}
        dropped_at: str | None = None
        by_stage = {v.stage: v for v in stage_verdicts}
        vec_failed = "vector_search" in by_stage and not by_stage["vector_search"].passed
        fts_failed = "fts_search" in by_stage and not by_stage["fts_search"].passed
        if vec_failed and fts_failed:
            # Both generators missed — chunk is invisible to RRF.
            dropped_at = "rrf_fusion"
        else:
            # Otherwise, the first gate stage that failed is the drop point.
            for v in stage_verdicts:
                if v.stage in _GATE_STAGES and not v.passed:
                    dropped_at = v.stage
                    break

        # Actual top-k for context
        actual_top_k = [
            MissAnalysisActualResult(
                rank=i,
                chunk_id=r.chunk_id,
                path=r.path,
                title=r.title,
                score=r.score,
            )
            for i, r in enumerate(results)
        ]

        # Generate rule-based suggestions
        metadata_filtered = bool(collection or project or layer)
        suggestions = self._build_miss_suggestions(
            stage_verdicts,
            dropped_at,
            appeared,
            final_rank=final_rank,
            top_k=top_k,
            metadata_filtered=metadata_filtered,
        )

        return MissAnalysis(
            query=query_text,
            expected_path=resolved_path,
            expected_chunk_id=target_chunk_id,
            target_resolution=target_resolution,  # type: ignore[arg-type]
            total_chunks_in_doc=total_chunks_in_doc,
            top_k_requested=top_k,
            appeared_in_results=appeared,
            final_rank=final_rank,
            dropped_at=dropped_at,
            stage_verdicts=stage_verdicts,
            actual_top_k=actual_top_k,
            suggestions=suggestions,
        )

    @staticmethod
    def _build_miss_suggestions(
        stage_verdicts: list[StageVerdict],
        dropped_at: str | None,
        appeared: bool,
        final_rank: int | None = None,
        top_k: int = 5,
        metadata_filtered: bool = False,
    ) -> list[str]:
        """Map stage failure modes to actionable suggestion strings.

        Pure rule-based, no LLM calls.  Each rule corresponds to a
        specific reason a chunk could fall out of the pipeline.
        Note: the ``recency_boost`` stage is intentionally absent from
        this map because it cannot drop a chunk — it only re-ranks.
        """
        suggestions: list[str] = []
        by_stage = {v.stage: v for v in stage_verdicts}

        if appeared:
            # Near-miss: the doc appeared but in the bottom of top-k.
            # Users passing --miss for a doc at rank 4/5 are asking
            # "why is this not where I expected?", not "is it there?".
            if final_rank is not None and top_k > 0 and final_rank >= top_k * 0.6:
                return [
                    f"The expected document IS in top-k but at rank {final_rank + 1}/{top_k} "
                    "(near-miss tier). It's competitive but being outranked by other chunks. "
                    "Consider a more specific query, or raise top_k to get more headroom."
                ]
            return ["The expected document IS in the top-k. No miss to analyze."]

        if dropped_at == "vector_search":
            suggestions.append(
                "The chunk's embedding is semantically far from the query. "
                "Try reformulating using vocabulary that appears in the target document, "
                "or increase the candidate pool size."
            )
            if "fts_search" in by_stage and by_stage["fts_search"].passed:
                suggestions.append(
                    "FTS5 keyword search DID find this chunk — consider raising fts_weight "
                    "or relying on keyword matching for this query type."
                )

        if dropped_at == "distance_cutoff":
            cf = by_stage["distance_cutoff"].counterfactual
            cf_clause = f" {cf}." if cf else ""
            suggestions.append(
                "Distance cutoff dropped the chunk just past threshold."
                f"{cf_clause} Pass distance_cutoff=2.0 (or higher) to relax the cutoff."
            )
            # Rescue hint: if FTS already matched, raising fts_weight
            # works even without touching the cutoff.
            if "fts_search" in by_stage and by_stage["fts_search"].passed:
                suggestions.append(
                    "FTS5 already matched this chunk — raising fts_weight would rescue it "
                    "without needing to relax the distance cutoff."
                )

        if dropped_at == "fts_search":
            v = by_stage["fts_search"]
            suggestions.append(
                f"FTS5 keyword search did not match. {v.detail} "
                "Either reformulate the query with words that appear in the chunk, "
                "or rely entirely on vector search by setting fts_weight=0."
            )

        if dropped_at == "rrf_fusion":
            vec_failed = "vector_search" in by_stage and not by_stage["vector_search"].passed
            fts_failed = "fts_search" in by_stage and not by_stage["fts_search"].passed
            if vec_failed and fts_failed:
                if metadata_filtered:
                    suggestions.append(
                        "Most likely the metadata filter (collection/project/layer) excluded "
                        "this document.  Re-run miss_analysis without filters to confirm."
                    )
                else:
                    suggestions.append(
                        "The chunk is invisible to BOTH vector and FTS search. Possible causes: "
                        "(1) the document was indexed with a different embedding model than the "
                        "current one — try `vstash reindex`; (2) the chunk text is empty or "
                        "contains only stopwords; (3) the query is semantically and lexically "
                        "unrelated to the chunk."
                    )
            else:
                suggestions.append(
                    "The chunk was eliminated before RRF fusion.  "
                    "Inspect the vector_search and fts_search verdicts for detail."
                )

        if dropped_at == "mmr_dedup":
            cf = by_stage["mmr_dedup"].counterfactual
            cf_clause = f" {cf}" if cf else ""
            suggestions.append(
                "MMR deduplication eliminated this chunk because another chunk from the "
                f"same document was already selected and they're too similar.{cf_clause}"
            )

        if dropped_at == "top_k_cutoff":
            cf = by_stage["top_k_cutoff"].counterfactual
            cf_clause = f" {cf}" if cf else ""
            suggestions.append(
                f"The chunk survived all stages but was just below the top_k cutoff.{cf_clause}"
            )

        if not suggestions:
            suggestions.append(
                "Unable to localize the failure to a single stage. "
                "Inspect stage_verdicts manually for details."
            )

        return suggestions

    # ------------------------------------------------------------------ #
    # MMR intra-document deduplication                                      #
    # ------------------------------------------------------------------ #

    def _mmr_dedup(
        self,
        ranked: list[dict[str, str | int | float]],
        top_k: int,
        mmr_lambda: float,
        _explain: bool = False,
    ) -> list[dict[str, str | int | float]]:
        """Select top-k results using intra-document MMR diversity.

        Chunks from *different* documents compete purely on score.  When
        multiple chunks from the *same* document appear, the second (and
        subsequent) chunks are penalised by their cosine similarity to the
        already-selected chunk(s) from that document.

        This allows two distant chapters of a book to both appear in results,
        while still preventing near-duplicate chunks from flooding top-k.

        When ``mmr_lambda = 1.0`` this degrades to hard per-document dedup
        (at most one chunk per document, highest-scoring wins).  The method
        also falls back to hard dedup if embedding lookup fails.
        """
        if not ranked:
            return []

        # Fast path: if mmr_lambda == 1.0, no diversity penalty — just dedup.
        # Also fast-path when there are no same-doc duplicates.
        from collections import Counter

        doc_counts = Counter(str(r["path"]) for r in ranked)
        has_duplicates = any(c > 1 for c in doc_counts.values())

        if mmr_lambda >= 1.0 or not has_duplicates:
            # Hard dedup: keep first (highest-scoring) chunk per document.
            seen: set[str] = set()
            deduped: list[dict[str, str | int | float]] = []
            for r in ranked:
                doc_key = str(r["path"])
                if doc_key not in seen:
                    seen.add(doc_key)
                    deduped.append(r)
            return deduped[:top_k]

        # --- Fetch embeddings for chunks with same-doc duplicates ---
        dup_doc_paths = {p for p, c in doc_counts.items() if c > 1}
        dup_ids = [int(r["id"]) for r in ranked if str(r["path"]) in dup_doc_paths]

        embeddings: dict[int, list[float]] = {}
        if dup_ids:
            placeholders = ",".join("?" * len(dup_ids))
            try:
                rows = self._conn.execute(
                    f"SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({placeholders})",
                    dup_ids,
                ).fetchall()
                for row in rows:
                    embeddings[row["rowid"]] = _deserialize(row["embedding"])
            except sqlite3.Error:
                logging.getLogger(__name__).warning(
                    "MMR embedding fetch failed — falling back to hard dedup. "
                    "Results may be less diverse.",
                    exc_info=True,
                )
                # Fallback: hard dedup
                seen_fb: set[str] = set()
                deduped_fb: list[dict[str, str | int | float]] = []
                for r in ranked:
                    doc_key = str(r["path"])
                    if doc_key not in seen_fb:
                        seen_fb.add(doc_key)
                        deduped_fb.append(r)
                return deduped_fb[:top_k]

        # --- Greedy MMR selection ---
        # Normalise scores to [0, 1] for MMR balancing.
        #
        # All current RRF / fusion paths store the score under the
        # "rrf" key. The old scoring pipeline used "final_score" and
        # was removed in v0.18.0 (#109); the dead fallback it left
        # behind here was a maintenance liability that made this loop
        # harder to reason about during the #153 audit. Removed.
        scores = [float(r["rrf"]) for r in ranked]
        s_min, s_max = min(scores), max(scores)
        s_range = s_max - s_min if s_max > s_min else 1.0

        selected: list[dict[str, str | int | float]] = []
        remaining = list(range(len(ranked)))
        # Boolean mask mirrors ``remaining`` membership so the sibling-penalty
        # loop can iterate a pre-grouped index list and skip candidates that
        # have already been selected, without an O(N) scan of ``remaining``.
        in_remaining = [True] * len(ranked)

        # Pre-compute normalized scores once (#167).  The value of
        # ``(score - s_min) / s_range`` is invariant across the outer
        # top_k loop — it depends only on each candidate's static score
        # — so recomputing it inside the inner ``for idx in remaining``
        # loop costs O(N * top_k) pure-Python ops for no benefit.
        # Hoisting it cuts the inner loop complexity to O(N) and
        # reuses the ``scores`` list already computed above, avoiding
        # the extra dict lookup and float conversion that a naive
        # hoist would do (flagged in the #167 review).
        norm_scores = [(s - s_min) / s_range for s in scores]

        # Precompute invariant MMR relevance terms to avoid O(K * N) recalculations
        relevance_terms = [mmr_lambda * ns for ns in norm_scores]
        penalty_multiplier = 1.0 - mmr_lambda

        # Extract values into fast lists for index-based access
        doc_keys = [str(r["path"]) for r in ranked]
        chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]

        # Lazy compute L2 norms for cosine similarity to avoid O(N) recomputation
        # when most chunks will never be compared against a sibling.
        chunk_norms = [None] * len(ranked)

        # Pre-group ranked indices by document key so the sibling-penalty
        # update walks O(S) (siblings only) instead of O(N) (all remaining).
        # Together with the in_remaining mask this turns the dedup loop from
        # O(K * N) into O(K * S_avg).
        doc_to_indices: dict[str, list[int]] = {}
        for i, doc_key in enumerate(doc_keys):
            doc_to_indices.setdefault(doc_key, []).append(i)

        # Track the maximum similarity to any selected chunk from the *same document*.
        # Replaces O(N * S) recomputation with O(1) lookup + O(N) update.
        max_sims = [0.0] * len(ranked)

        for _ in range(min(top_k, len(ranked))):
            best_idx = -1
            best_mmr = -float("inf")
            best_rem_idx = -1

            for rem_idx, idx in enumerate(remaining):
                max_sim = max_sims[idx]

                mmr_score = relevance_terms[idx] - penalty_multiplier * max_sim
                # Prefer the smaller original ``idx`` on exact ties so the
                # selected set matches the pre-rewrite ordering, where
                # ``remaining`` was kept sorted ascending and ``>`` (strict)
                # let the first-seen candidate win. The swap-with-last
                # removal scrambles the order of ``remaining`` so we have
                # to add the tie-break explicitly here.
                if mmr_score > best_mmr or (mmr_score == best_mmr and idx < best_idx):
                    best_mmr = mmr_score
                    best_idx = idx
                    best_rem_idx = rem_idx

            if best_idx < 0 or best_mmr < 0:
                # Stop when the best remaining candidate has negative MMR,
                # meaning its redundancy penalty exceeds its relevance.
                break

            chosen = ranked[best_idx]
            if _explain:
                chosen["_mmr_penalty"] = (1 - mmr_lambda) * max_sims[best_idx]
            selected.append(chosen)

            # O(1) swap-with-last removal: the order of ``remaining`` is not
            # significant — only the set of eligible candidates is — so we can
            # overwrite the popped slot with the tail and shrink in place
            # instead of paying O(N) for ``list.remove``.
            remaining[best_rem_idx] = remaining[-1]
            remaining.pop()
            in_remaining[best_idx] = False

            # Update max_sims for remaining chunks from the same document
            # by comparing against the newly selected embedding.
            new_doc_key = doc_keys[best_idx]
            doc_indices = doc_to_indices[new_doc_key]

            # Skip similarity penalty updates entirely if this is the only chunk from this doc
            if len(doc_indices) <= 1:
                continue

            new_emb = chunk_embs[best_idx]
            if new_emb is not None:
                if chunk_norms[best_idx] is None:
                    chunk_norms[best_idx] = math.hypot(*new_emb)
                new_norm = chunk_norms[best_idx]

                for idx in doc_indices:
                    if in_remaining[idx]:
                        idx_emb = chunk_embs[idx]
                        if idx_emb is not None:
                            if chunk_norms[idx] is None:
                                chunk_norms[idx] = math.hypot(*idx_emb)
                            idx_norm = chunk_norms[idx]

                            sim = _cosine_sim(idx_emb, new_emb, norm_a=idx_norm, norm_b=new_norm)
                            if sim > max_sims[idx]:
                                max_sims[idx] = sim

        return selected

    # ------------------------------------------------------------------ #
    # Filter builder                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_filter_conditions(
        alias: str = "",
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | list[str] | None = None,
        added_after: str | None = None,
        added_before: str | None = None,
        filters: dict | None = None,
    ) -> tuple[list[str], list[str]]:
        """Build filter conditions for document metadata.

        Args:
            alias: Table alias prefix (e.g. ``'d.'``, ``'d2.'``, or ``''``).
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by one or more tags. Pass a comma-separated string
                (``"alpha,beta"``) or a list (``["alpha", "beta"]``);
                whitespace and empty entries are stripped, then matched OR
                with comma-anchored LIKE (``",alpha,"``) so ``alpha`` does
                not false-match ``alphabet``.
            added_after: ISO date — only documents added on or after this date.
            added_before: ISO date — only documents added before this date.

        Returns:
            Tuple of (condition strings, bind parameters).
        """
        prefix = f"{alias}." if alias else ""
        conditions: list[str] = []
        params: list[str] = []
        if collection:
            conditions.append(f"{prefix}collection = ?")
            params.append(collection)
        if project:
            conditions.append(f"{prefix}project = ?")
            params.append(project)
        if layer:
            conditions.append(f"{prefix}layer = ?")
            params.append(layer)
        tag_list = _normalize_tags(tags)
        if tag_list:
            # Anchor the LIKE pattern with commas so a tag does not
            # false-match a longer tag that contains it as a substring
            # (e.g. searching ``alpha`` must NOT hit a doc tagged
            # ``alphabet``). The stored ``tags`` column is a
            # comma-separated string; wrapping both sides in commas
            # turns substring containment into exact membership.
            wrapped = f"','||{prefix}tags||','"
            ors = [f"{wrapped} LIKE ?" for _ in tag_list]
            conditions.append("(" + " OR ".join(ors) + ")")
            params.extend(f"%,{t},%" for t in tag_list)
        if added_after:
            conditions.append(f"{prefix}added_at >= ?")
            params.append(_canonicalize_added_filter(added_after, label="added_after"))
        if added_before:
            conditions.append(f"{prefix}added_at < ?")
            params.append(_canonicalize_added_filter(added_before, label="added_before"))
        if filters is not None:
            # Boolean cross-field filter tree (#106), AND-combined with the flat
            # filters above. Field names are validated; values are bound as params.
            tree_sql, tree_params = _compile_filter_tree(filters, prefix)
            if tree_sql is not None:
                conditions.append(tree_sql)
                params.extend(tree_params)
        return conditions, params

    @staticmethod
    def _build_doc_filter(
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
        added_after: str | None = None,
        added_before: str | None = None,
        filters: dict | None = None,
    ) -> tuple[str, str, list[str]]:
        """Build SQL filter clauses for document metadata.

        Returns three items:
        - **vec_clause**: ``AND v.rowid IN (…)`` for vec0 pre-filtering
        - **col_clause**: ``AND d.collection = ? …`` for JOINed queries (FTS)
        - **params**: bind-parameter values (same order for both clauses)

        Args:
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by tag (LIKE match).
            added_after: ISO date — only documents added on or after this date.
            added_before: ISO date — only documents added before this date.
        """
        conditions_d2, params = _SearchEngineMixin._get_filter_conditions(
            "d2",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            added_after=added_after,
            added_before=added_before,
            filters=filters,
        )

        if not conditions_d2:
            return "", "", []

        where = " AND ".join(conditions_d2)
        vec_clause = f"""
            AND v.rowid IN (
                SELECT c2.id
                FROM chunks c2
                JOIN documents d2 ON d2.id = c2.doc_id
                WHERE {where}
            )
        """
        # For JOINed queries where documents is aliased as 'd'
        conditions_d, _ = _SearchEngineMixin._get_filter_conditions(
            "d",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            added_after=added_after,
            added_before=added_before,
            filters=filters,
        )
        col_clause = "AND " + " AND ".join(conditions_d)
        return vec_clause, col_clause, params
