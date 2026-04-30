"""vstash.services.ask -- retrieve context, then send to LLM.

Wraps :func:`vstash.services.search.search_with_embedding` and
:func:`vstash.chat.ask_full` so adapters do not have to remember to
do retrieval before generation, or to feed the result chunks into
the right shape for the LLM call.

Returns :class:`AskResult`, which carries content + reasoning +
usage + resolved backend / model. Adapters that only need the
text reply can read ``.content``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..chat import ask_full as _chat_ask_full
from .search import search_with_embedding

if TYPE_CHECKING:
    from ..config import VstashConfig
    from ..models import AskResult, SearchResult
    from ..store import VstashStore


def ask_with_context(
    *,
    cfg: VstashConfig,
    store: VstashStore,
    query: str,
    top_k: int = 5,
    expand_window: int = 1,
    history: list[dict[str, str]] | None = None,
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
) -> AskResult:
    """Retrieve context for ``query`` and send it to the LLM backend.

    The single replacement for the
    ``embed_query -> store.search -> store.expand_context -> chat.ask_full``
    chain that the SDK and the (legacy) adapter handlers used to
    inline. Validates retrieval inputs via
    :func:`vstash.services.search.search_with_embedding`, then runs
    :func:`vstash.chat.ask_full` against the resolved backend
    (Cerebras, Ollama, OpenAI, or auto-detected local).

    Exposes the full search-side knob set (recency, date filters,
    MMR diversity, RRF weights, distance cutoff, exact-match) so
    callers do not have to choose between "use the service" and
    "use the adapter". A user who wants recency-boosted retrieval
    in their LLM answer should not have to re-implement the
    triplet just to pass one parameter.

    Args:
        cfg: Active vstash config.
        store: Open ``VstashStore``.
        query: User question.
        top_k: How many context chunks to retrieve.
        expand_window: Sibling chunks per side to include in
            context expansion. Pass ``0`` to skip expansion.
        history: Prior conversation turns (multi-turn chat).
        collection: Optional collection filter for retrieval.
        project: Optional project filter for retrieval.
        layer: Optional layer filter for retrieval.
        recency_boost: Temporal decay multiplier on retrieval (0.0
            = off).
        added_after: ISO date filter on document add timestamp.
        added_before: ISO date filter on document add timestamp.
        mmr_lambda: MMR diversity coefficient applied to context
            chunks before they reach the LLM.
        vec_weight: Pin RRF vector weight on the retrieval call.
        fts_weight: Pin RRF FTS weight on the retrieval call.
        retrieval_mode: ``"hybrid"`` (default), ``"vec_only"``, or
            ``"fts_only"``.
        distance_cutoff: Override the store's default distance
            cutoff for the retrieval call.
        exact_match: Literal substring required in retrieval results.
        exact_match_case_sensitive: Whether ``exact_match`` is
            case-sensitive.

    Returns:
        ``AskResult`` carrying content + (when the backend surfaces
        it) reasoning + usage + resolved backend / model.

    Raises:
        LimitError: Retrieval input rejected at the boundary.
        ValueError: Unknown inference backend in ``cfg.inference``.
        ConnectionError: Inference API call failed.
    """
    chunks: list[SearchResult] = search_with_embedding(
        cfg=cfg,
        store=store,
        query=query,
        top_k=top_k,
        expand_window=expand_window,
        collection=collection,
        project=project,
        layer=layer,
        recency_boost=recency_boost,
        added_after=added_after,
        added_before=added_before,
        mmr_lambda=mmr_lambda,
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        retrieval_mode=retrieval_mode,
        distance_cutoff=distance_cutoff,
        exact_match=exact_match,
        exact_match_case_sensitive=exact_match_case_sensitive,
    )
    return _chat_ask_full(query, chunks, cfg, history)
