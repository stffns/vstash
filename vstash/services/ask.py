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
    vec_weight: float | None = None,
    fts_weight: float | None = None,
    retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
) -> AskResult:
    """Retrieve context for ``query`` and send it to the LLM backend.

    The single replacement for the
    ``embed_query -> store.search -> store.expand_context -> chat.ask_full``
    chain that the SDK and the (legacy) adapter handlers used to
    inline. Validates retrieval inputs via
    :func:`vstash.services.search.search_with_embedding`, then runs
    :func:`vstash.chat.ask_full` against the resolved backend
    (Cerebras, Ollama, OpenAI, or auto-detected local).

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
        vec_weight: Pin RRF vector weight on the retrieval call.
        fts_weight: Pin RRF FTS weight on the retrieval call.
        retrieval_mode: ``"hybrid"`` (default), ``"vec_only"``, or
            ``"fts_only"``.

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
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        retrieval_mode=retrieval_mode,
    )
    return _chat_ask_full(query, chunks, cfg, history)
