"""vstash.services.search -- the validate -> embed -> search -> expand triplet.

A pure function that the adapters (CLI, MCP, web, SDK) all call to
turn a user query into a list of relevant chunks. Replaces the
hand-rolled triplet that was duplicated 13+ times across the
codebase before v0.37.

The function validates first (so adapters cannot bypass
:func:`vstash.validation.validate_search_input`), then embeds the
query, then runs the hybrid retrieval pipeline, then optionally
expands the context window. The split ``expand_window=0`` exists
because the miss-analysis path wants raw search results without
context expansion, and one of the daemon endpoints used to take
the window from config.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..embed import embed_query
from ..models import SearchResults
from ..validation import LimitError, validate_search_input

if TYPE_CHECKING:
    from ..config import VstashConfig
    from ..models import SearchResult
    from ..store import VstashStore

logger = logging.getLogger(__name__)

# The literal Protocol of valid retrieval modes. Kept as a module
# constant so the runtime check in `search_with_embedding` does not
# have to maintain a parallel list to the Literal type alias above.
_VALID_RETRIEVAL_MODES: frozenset[str] = frozenset({"hybrid", "vec_only", "fts_only"})


def search_with_embedding(
    *,
    cfg: VstashConfig,
    store: VstashStore,
    query: str,
    top_k: int = 5,
    expand_window: int = 1,
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
    recency_boost: float = 0.0,
    added_after: str | None = None,
    added_before: str | None = None,
    tags: str | list[str] | None = None,
    filters: dict | None = None,
    mmr_lambda: float = 0.5,
    dedup_threshold: float | None = None,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
    retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
    distance_cutoff: float | None = None,
    exact_match: str | None = None,
    exact_match_case_sensitive: bool = False,
    log_usage: bool = True,
    usage_prior: bool | None = None,
) -> SearchResults:
    """Embed ``query`` then hybrid-search ``store`` then expand context.

    The single replacement for the
    ``embed_query -> store.search -> store.expand_context`` triplet
    that adapters used to inline. Validates the public inputs against
    the configured limits before doing any work, so a 200_000-char
    query or a ``top_k`` of 9999 is rejected at the boundary instead
    of crashing inside SQLite.

    Args:
        cfg: Active vstash config (used for the embedding model and
            the validation limits).
        store: Open ``VstashStore``. Caller owns its lifecycle.
        query: Natural language search query.
        top_k: Maximum results to return.
        expand_window: How many sibling chunks per side to include in
            context expansion. ``0`` skips expansion (returns the raw
            search results, useful for miss analysis where the
            unexpanded chunk is what the caller is debugging).
        collection: Optional collection filter.
        project: Optional project filter.
        layer: Optional layer filter.
        recency_boost: Temporal decay multiplier (0.0 = off).
        added_after: ISO date filter on document add timestamp.
        added_before: ISO date filter on document add timestamp.
        tags: Restrict to documents tagged with one or more of these tags
            (OR semantics). Accepts a comma-separated string
            (``"alpha,beta"``) or a list (``["alpha", "beta"]``).
            Comma-anchored LIKE match so ``alpha`` does not false-match
            ``alphabet``.
        mmr_lambda: MMR diversity coefficient (0.0 to 1.0).
        dedup_threshold: Opt-in cross-document near-duplicate collapse.
            ``None`` (default) leaves ranking untouched. When set, a chunk
            is dropped if its cosine similarity to an already kept,
            higher-ranked chunk is ``>= dedup_threshold``, even across
            different documents -- which ``mmr_lambda`` never does, since
            MMR only penalises same-document siblings. Range ``(0.0, 1.0]``;
            try ``0.95`` when one answer is mirrored across many documents.
        vec_weight: Pin RRF vector weight; None keeps adaptive RRF.
        fts_weight: Pin RRF FTS weight; None keeps adaptive RRF.
        retrieval_mode: ``"hybrid"`` (default), ``"vec_only"``, or
            ``"fts_only"``. None resolves to ``"hybrid"``.
        distance_cutoff: Override the store's default distance cutoff
            for this call. ``None`` (default) lets the store apply
            its built-in default. Validated against
            ``cfg.limits.max_distance_cutoff`` when provided.
        exact_match: Literal substring required in the result text.
        exact_match_case_sensitive: Whether ``exact_match`` is
            case-sensitive.
        log_usage: Record this search in the usage log (``search_events``
            + one ``search_results`` row per returned chunk) when
            ``cfg.observability.log_search_results`` is on. Internal
            callers that are not a user-visible search (benchmarks,
            federated fan-out) pass ``False``. Query-cache hits ARE
            logged: the caller still saw those results, so they are as
            much a usage signal as a cold search.
        usage_prior: Apply the query-conditioned usage prior
            (``[usage_prior]``, see :class:`vstash.config.UsagePriorConfig`).
            ``None`` (default) follows ``cfg.usage_prior.enabled``. When it
            runs and a similar past search exists, ``candidate_pool``
            candidates are re-scored and cut to ``top_k``; each returned
            result carries ``usage_boost``. Without a similar past search
            the ranking is exactly the prior-off ranking.

    Returns:
        ``SearchResults`` (a ``list[SearchResult]`` ranked by relevance,
        possibly empty) whose ``event_id`` names the usage-log event to
        pass to :func:`mark_used`; ``None`` when logging is off.

    Raises:
        LimitError: Any input exceeded the configured limit.
    """
    # Validate the cheap-to-check public knobs BEFORE embed_query so
    # an invalid expand_window or retrieval_mode does not pay for a
    # daemon round-trip just to crash inside store.search later.
    # These raise LimitError (not ValueError) so they honour the
    # function's documented "Raises: LimitError" contract and so
    # web/MCP handlers that catch LimitError specifically (api_search,
    # api_chat) map them to HTTP 400 / MCP error rather than letting
    # them propagate as 500 / unhandled exception.
    if not isinstance(expand_window, int) or isinstance(expand_window, bool) or expand_window < 0:
        raise LimitError(f"expand_window must be a non-negative int, got {expand_window!r}")
    if retrieval_mode is not None and retrieval_mode not in _VALID_RETRIEVAL_MODES:
        raise LimitError(
            f"retrieval_mode must be one of "
            f"{sorted(_VALID_RETRIEVAL_MODES)} or None, got {retrieval_mode!r}"
        )

    # Resolve the mode and drop caller weights BEFORE validation when
    # the mode is non-hybrid. The store forces (1.0, 0.0) or (0.0,
    # 1.0) on the short-circuit branch, so any caller-supplied
    # numeric here would either no-op or trip an unrelated validation
    # error. Memory.search (and the matching MCP / CLI paths) have
    # always treated `fts_only`/`vec_only` as "ignore weights" --
    # validating them first would be a behavior change.
    mode = retrieval_mode or "hybrid"
    if mode != "hybrid":
        vec_weight = None
        fts_weight = None

    # If the caller pinned a cutoff, validate THAT value against
    # the configured ceiling; otherwise validate the ceiling against
    # itself (a noop) so the rest of validate_search_input can
    # still run with a populated arg.
    effective_cutoff = (
        distance_cutoff if distance_cutoff is not None else cfg.limits.max_distance_cutoff
    )
    validate_search_input(
        query_text=query,
        top_k=top_k,
        distance_cutoff=effective_cutoff,
        recency_boost=recency_boost,
        limits=cfg.limits,
        vec_weight=vec_weight,
        fts_weight=fts_weight,
    )

    q_embedding = embed_query(query, cfg.embeddings.model)

    # Only forward distance_cutoff when the caller pinned one;
    # otherwise let store.search apply its own default. Avoids
    # hardcoding the store's default value here (currently 1.3225).
    extra_kwargs: dict = {}
    if distance_cutoff is not None:
        extra_kwargs["distance_cutoff"] = distance_cutoff

    prior_weights = usage_prior_weights(
        cfg=cfg, store=store, query_embedding=q_embedding, enabled=usage_prior
    )

    results = store.search(
        query_embedding=q_embedding,
        query_text=query,
        top_k=usage_prior_pool(cfg, top_k, prior_weights),
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        retrieval_mode=mode,
        collection=collection,
        project=project,
        layer=layer,
        recency_boost=recency_boost,
        added_after=added_after,
        added_before=added_before,
        tags=tags,
        filters=filters,
        mmr_lambda=mmr_lambda,
        dedup_threshold=dedup_threshold,
        exact_match=exact_match,
        exact_match_case_sensitive=exact_match_case_sensitive,
        **extra_kwargs,
    )
    if prior_weights:
        results = apply_usage_prior(
            store=store,
            results=results,
            weights=prior_weights,
            beta=cfg.usage_prior.beta,
            top_k=top_k,
        )

    # Log the ranked list BEFORE context expansion: expansion only widens
    # each result's text, the chunk ids / ranks the caller acts on are the
    # ones ranking produced.
    event_id: int | None = None
    if log_usage and cfg.observability.log_search_results:
        event_id = _log_search(store, query, results, q_embedding)

    if expand_window > 0 and results:
        results = store.expand_context(results, window=expand_window)

    return SearchResults(results, event_id=event_id)


def _log_search(
    store: VstashStore, query: str, results: list[SearchResult], query_embedding=None
) -> int | None:
    """Write the usage-log event for one search; never fails the search.

    A logging failure (locked DB past the busy timeout, a chunk deleted by
    another process between ranking and logging tripping the FK) must not
    turn a successful search into an error, so it degrades to "not logged".
    The catch is deliberately broad: this is telemetry on the read path.
    """
    from ..store import relevance_tier

    try:
        best_distance = store.last_best_distance
        return store.record_search_event(
            query=query,
            best_distance=best_distance,
            relevance_tier=relevance_tier(best_distance),
            result_count=len(results),
            results=results,
            query_embedding=query_embedding,
        )
    except Exception:
        logger.warning("usage log: could not record search event", exc_info=True)
        return None


def usage_prior_weights(
    *,
    cfg: VstashConfig,
    store: VstashStore,
    query_embedding,
    enabled: bool | None = None,
) -> dict[str, float]:
    """Per-document weights of the query-conditioned usage prior.

    ``{}`` when the prior is off (``enabled`` falls back to
    ``cfg.usage_prior.enabled``), ``beta`` is 0, no logged search is
    similar enough, or the lookup fails -- a broken prior must never break
    the search, so errors degrade to "no prior".
    """
    prior = cfg.usage_prior
    on = prior.enabled if enabled is None else enabled
    # ``is not True``: only a real opt-in turns the prior on (a mocked or
    # mis-typed config must not silently enable a ranking change).
    if on is not True or not prior.beta > 0.0:
        return {}
    try:
        return store.usage_prior_doc_weights(
            query_embedding, tau=prior.tau, max_neighbors=prior.max_neighbors
        )
    except Exception:
        logger.warning("usage prior: lookup failed; searching without it", exc_info=True)
        return {}


def usage_prior_pool(cfg: VstashConfig, top_k: int, weights: dict[str, float]) -> int:
    """``top_k`` to ask the store for: the candidate pool when the prior will run."""
    if not weights:
        return top_k
    return max(top_k, min(cfg.usage_prior.candidate_pool, cfg.limits.max_top_k))


def apply_usage_prior(
    *,
    store: VstashStore,
    results: list[SearchResult],
    weights: dict[str, float],
    beta: float,
    top_k: int,
) -> list[SearchResult]:
    """Re-score ``results`` with ``score * (1 + beta * log1p(uses))`` and cut to ``top_k``.

    ``uses`` is the document's similarity-weighted use count from
    :func:`usage_prior_weights`. Ties keep the pipeline's order (stable
    sort). Every returned result carries the multiplier in
    ``usage_boost`` (1.0 for candidates the prior did not touch).
    """
    doc_of = store.chunk_doc_ids([r.chunk_id for r in results])
    rescored: list[SearchResult] = []
    for r in results:
        mult = 1.0 + beta * math.log1p(weights.get(doc_of.get(r.chunk_id, ""), 0.0))
        rescored.append(r.model_copy(update={"score": r.score * mult, "usage_boost": mult}))
    rescored.sort(key=lambda r: -r.score)
    return rescored[:top_k]


def mark_used(*, store: VstashStore, event_id: int, chunk_ids: int | Sequence[int]) -> int:
    """Report that the caller used one or more results of a logged search.

    Shared entry point for the SDK (``Memory.mark_used``), MCP
    (``vstash_mark_used``) and web (``POST /api/search/used``).

    Args:
        store: Open ``VstashStore`` the search ran against.
        event_id: ``SearchResults.event_id`` of that search.
        chunk_ids: ``SearchResult.chunk_id`` of the result(s) used.

    Returns:
        Number of results newly marked. ``0`` when the event was pruned,
        the chunk was not in that result list, or it was already marked.

    Raises:
        LimitError: ``event_id`` / ``chunk_ids`` are not integers.
    """
    ids = [chunk_ids] if isinstance(chunk_ids, int) else list(chunk_ids)
    for v in (event_id, *ids):
        if not isinstance(v, int) or isinstance(v, bool):
            raise LimitError(f"event_id / chunk_ids must be ints, got {v!r}")
    return store.mark_result_used(event_id, ids)
