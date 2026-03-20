"""
mcp.py — MCP server for vstash.

Exposes vstash document memory as MCP tools so that Claude Desktop
(or any MCP-compatible client) can use vstash as persistent memory
between sessions.

Run with:
    python -m vstash.mcp
    vstash-mcp          (entry point)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats
from .store import VstashStore

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Server instance                                                      #
# ------------------------------------------------------------------ #

mcp_server = FastMCP(
    "vstash",
    instructions=(
        "vstash — local document memory with hybrid semantic search. "
        "Ingest files, URLs, or directories and query them with natural language."
    ),
)

# ------------------------------------------------------------------ #
# Lazy singletons                                                      #
# ------------------------------------------------------------------ #

_config: VstashConfig | None = None
_store: VstashStore | None = None


def _get_config() -> VstashConfig:
    """Load configuration lazily (same resolution as CLI).

    Returns:
        Cached VstashConfig instance.
    """
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_config()
    return _config


def _get_store() -> VstashStore:
    """Open the vector store lazily, reusing across tool calls.

    Returns:
        Cached VstashStore instance.

    Raises:
        FileNotFoundError: If the database directory cannot be created.
    """
    global _store  # noqa: PLW0603
    if _store is None:
        cfg = _get_config()
        dim = get_embedding_dim(cfg.embeddings.model)
        _store = VstashStore(cfg.db_path, embedding_dim=dim)
    return _store


def _ok(data: Any) -> str:
    """Wrap a successful result as JSON.

    Args:
        data: Serializable payload (dict, list, Pydantic model, etc.).

    Returns:
        JSON string.
    """
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        payload = [item.model_dump() for item in data]
    else:
        payload = data
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error(message: str) -> str:
    """Wrap an error message as JSON.

    Args:
        message: Human-readable error description.

    Returns:
        JSON string with ``{"error": "..."}`` structure.
    """
    return json.dumps({"error": message}, ensure_ascii=False)


# ------------------------------------------------------------------ #
# Tools                                                                #
# ------------------------------------------------------------------ #


@mcp_server.tool()
def vstash_add(path: str) -> str:
    """Ingest a file, directory, or URL into vstash memory.

    Supports PDF, DOCX, PPTX, XLSX, Markdown, plain text, code files,
    and HTTP(S) URLs. Directories are ingested recursively.

    Args:
        path: Absolute file path, directory path, or HTTP(S) URL to ingest.

    Returns:
        JSON string with ingestion result (status, chunks, timing).
    """
    try:
        from .ingest import ingest, ingest_directory

        cfg = _get_config()
        store = _get_store()

        resolved = Path(path).expanduser().resolve()

        # Directory → recursive ingestion
        if resolved.is_dir():
            results: list[IngestResult] = ingest_directory(
                str(resolved), cfg, store, force=False,
            )
            summary = {
                "status": "ok",
                "path": str(resolved),
                "files_processed": len(results),
                "results": [r.model_dump() for r in results],
            }
            return _ok(summary)

        # Single file or URL
        result: IngestResult = ingest(path, cfg, store, force=False)
        return _ok(result)

    except FileNotFoundError:
        return _error(f"Path not found: {path}")
    except PermissionError:
        return _error(f"Permission denied: {path}")
    except Exception as exc:
        logger.exception("vstash_add failed")
        return _error(f"Ingestion failed: {exc}")


@mcp_server.tool()
def vstash_ask(query: str, top_k: int = 5) -> str:
    """Query vstash memory and get an LLM-generated answer with sources.

    Performs hybrid semantic + keyword search, then sends the top chunks
    to the configured LLM backend for answer generation.

    Args:
        query: Natural language question to answer from memory.
        top_k: Number of context chunks to retrieve (default: 5).

    Returns:
        JSON string with answer text and source documents.
    """
    try:
        from .chat import ask

        cfg = _get_config()
        store = _get_store()

        # Embed query and search
        query_embedding = embed_query(query, cfg.embeddings.model)
        chunks: list[SearchResult] = store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
        )

        if not chunks:
            return _ok({
                "answer": "No relevant documents found in memory.",
                "sources": [],
            })

        # Get LLM answer
        answer = ask(query, chunks, cfg)

        # Build source list
        sources = [
            {"title": c.title, "path": c.path, "score": c.score}
            for c in chunks
        ]
        # Deduplicate sources by path
        seen: set[str] = set()
        unique_sources: list[dict[str, Any]] = []
        for src in sources:
            if src["path"] not in seen:
                seen.add(src["path"])
                unique_sources.append(src)

        return _ok({"answer": answer, "sources": unique_sources})

    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except ConnectionError as exc:
        return _error(f"LLM backend error: {exc}")
    except Exception as exc:
        logger.exception("vstash_ask failed")
        return _error(f"Query failed: {exc}")


@mcp_server.tool()
def vstash_search(query: str, top_k: int = 5) -> str:
    """Search vstash memory without LLM — returns raw chunks with scores.

    Performs hybrid semantic + keyword search using Reciprocal Rank Fusion.
    Useful for retrieving context without consuming LLM tokens.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default: 5).

    Returns:
        JSON array of matching chunks with text, title, path, and score.
    """
    try:
        cfg = _get_config()
        store = _get_store()

        query_embedding = embed_query(query, cfg.embeddings.model)
        chunks: list[SearchResult] = store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
        )

        return _ok(chunks)

    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_search failed")
        return _error(f"Search failed: {exc}")


@mcp_server.tool()
def vstash_list() -> str:
    """List all documents currently stored in vstash memory.

    Returns:
        JSON array of documents with path, title, type, chunk count,
        character count, and ingestion timestamp.
    """
    try:
        store = _get_store()
        docs: list[DocumentInfo] = store.list_documents()
        return _ok(docs)

    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_list failed")
        return _error(f"Failed to list documents: {exc}")


@mcp_server.tool()
def vstash_stats() -> str:
    """Get aggregate statistics about the vstash memory store.

    Returns:
        JSON object with document count, chunk count, database size
        in MB, and database file path.
    """
    try:
        store = _get_store()
        stats: StoreStats = store.stats()
        return _ok(stats)

    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_stats failed")
        return _error(f"Failed to get stats: {exc}")


@mcp_server.tool()
def vstash_forget(source: str) -> str:
    """Remove a document from vstash memory by its source path or URL.

    The source must match exactly the path used during ingestion.
    Use vstash_list() to see currently stored document paths.

    Args:
        source: Exact file path or URL of the document to remove.

    Returns:
        JSON object with deletion status.
    """
    try:
        store = _get_store()
        deleted = store.delete_document(source)

        if deleted:
            return _ok({"status": "deleted", "source": source})
        return _ok({"status": "not_found", "source": source})

    except FileNotFoundError:
        return _error("vstash database not found.")
    except Exception as exc:
        logger.exception("vstash_forget failed")
        return _error(f"Failed to delete document: {exc}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #


def main() -> None:
    """Run the vstash MCP server over stdio.

    This is the entry point registered as ``vstash-mcp`` in pyproject.toml.
    """
    # Suppress Rich output that would interfere with MCP stdio transport
    os.environ.setdefault("TERM", "dumb")

    mcp_server.run()


if __name__ == "__main__":
    main()
