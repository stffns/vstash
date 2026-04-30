"""vstash.services -- shared business-logic helpers for adapters.

The CLI, MCP server, HTTP API, and SDK all need to perform the same
high-level operations (embed query + search + expand context, then
optionally feed the chunks to an LLM). Historically each adapter
re-implemented this triplet inline (mcp.py:546, web.py:98,
cli.py:438, etc.) and the implementations drifted from each other
in subtle ways: one passed ``window=1`` to ``expand_context``, the
next passed ``window=2``; one applied ``recency_boost`` from config,
another forced it to 0; one validated inputs, the next didn't.

This package centralises the shared steps as pure functions that
take ``(cfg, store, ...)`` explicitly. They:

- validate user input via :mod:`vstash.validation` (single source of
  truth for limits)
- embed the query via :func:`vstash.embed.embed_query`
- call ``store.search`` / ``store.miss_analysis`` / ``chat.ask_full``
- optionally expand context via ``store.expand_context``

Adapters (``mcp.py``, ``web.py``, ``cli.py``, ``memory.py``) become
thin presentation layers that handle their protocol-specific
concerns (MCP tool result shape, HTTP response format, Rich CLI
output, SDK return types) and delegate the actual retrieval logic
here.

These functions are stateless: they do not hold a store, a config,
or any session state. The caller passes whatever ``VstashStore``
and ``VstashConfig`` they already have. This keeps the services
testable in isolation and re-usable across the four adapters.
"""

from __future__ import annotations

from .ask import ask_with_context
from .search import search_with_embedding

__all__ = [
    "ask_with_context",
    "search_with_embedding",
]
