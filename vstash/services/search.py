"""vstash.services.search -- the embed -> search -> expand triplet.

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

from typing import TYPE_CHECKING, Literal

from ..embed import embed_query
from ..validation import LimitError, validate_search_input

if TYPE_CHECKING:
    from ..config import VstashConfig
    from ..models import SearchResult
    from ..store import VstashStore

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
    mmr_lambda: float = 0.5,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
    retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
    distance_cutoff: float | None = None,
    exact_match: str | None = None,
    exact_match_case_sensitive: bool = False,
) -> list[SearchResult]:
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
        mmr_lambda: MMR diversity coefficient (0.0 to 1.0).
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

    Returns:
        List of ``SearchResult`` ranked by relevance. May be empty.

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

    results = store.search(
        query_embedding=q_embedding,
        query_text=query,
        top_k=top_k,
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        retrieval_mode=mode,
        collection=collection,
        project=project,
        layer=layer,
        recency_boost=recency_boost,
        added_after=added_after,
        added_before=added_before,
        mmr_lambda=mmr_lambda,
        exact_match=exact_match,
        exact_match_case_sensitive=exact_match_case_sensitive,
        **extra_kwargs,
    )

    if expand_window > 0 and results:
        results = store.expand_context(results, window=expand_window)

    return results
