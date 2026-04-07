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

import atexit
import json
import logging
import threading
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _exc:
    raise ImportError(
        "MCP server requires the mcp package. Install it with: pip install vstash[mcp]"
    ) from _exc

from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim, warmup
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats
from .store import VstashStore, relevance_tier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Server instance                                                      #
# ------------------------------------------------------------------ #

mcp_server = FastMCP(
    "vstash",
    instructions=(
        "vstash — local document memory with hybrid semantic search. "
        "Ingest files, URLs, or directories and query them with natural language.\n\n"
        "WHEN TO USE vstash_search/vstash_ask:\n"
        "- User asks a knowledge question about their documents (how, what, why, explain)\n"
        "- User needs context from specs, docs, notes, or past decisions\n"
        "- User references information they previously added to memory\n\n"
        "WHEN NOT TO USE vstash_search/vstash_ask:\n"
        "- User gives an action command (commit, merge, push, deploy, build, test)\n"
        "- User gives a short confirmation (yes, no, ok, do it)\n"
        "- User asks about current code or git state (use other tools instead)\n"
        "- The query is not related to stored documents"
    ),
)

# ------------------------------------------------------------------ #
# Lazy singletons                                                      #
# ------------------------------------------------------------------ #

_config: VstashConfig | None = None
_store: VstashStore | None = None
_lock = threading.RLock()

# ------------------------------------------------------------------ #
# Background job tracking for async directory ingestion                #
# ------------------------------------------------------------------ #

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _get_config() -> VstashConfig:
    """Load configuration lazily (same resolution as CLI).

    Thread-safe via double-checked locking pattern.

    Returns:
        Cached VstashConfig instance.
    """
    global _config  # noqa: PLW0603
    if _config is None:
        with _lock:
            if _config is None:
                _config = load_config()
    return _config


def _get_store() -> VstashStore:
    """Open the vector store lazily, reusing across tool calls.

    Uses the same DB resolution chain as the CLI (resolve_db_path) so that
    MCP and CLI always operate on the same database given the same config.

    Thread-safe via double-checked locking pattern.

    Returns:
        Cached VstashStore instance.

    Raises:
        OSError: If the database directory cannot be created (e.g. PermissionError).
    """
    global _store  # noqa: PLW0603
    if _store is None:
        with _lock:
            if _store is None:
                from .profile import resolve_db_path

                cfg = _get_config()
                dim = get_embedding_dim(cfg.embeddings.model)
                db_path = str(resolve_db_path(config_db_path=cfg.storage.db_path))
                _store = VstashStore(
                    db_path,
                    embedding_dim=dim,
                    vector_backend=cfg.storage.vector_backend,
                    snapvec_bits=cfg.storage.snapvec_bits,
                    observability=cfg.observability,
                )
                atexit.register(_store.close)
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
    elif isinstance(data, list):
        payload = [item.model_dump() if hasattr(item, "model_dump") else item for item in data]
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


_MAX_JOBS = 100  # Cap job store to prevent unbounded growth


def _cleanup_jobs() -> None:
    """Remove oldest completed/errored jobs when store exceeds _MAX_JOBS."""
    with _jobs_lock:
        if len(_jobs) <= _MAX_JOBS:
            return
        # Remove completed/errored jobs first (oldest first by insertion order)
        to_remove = []
        for jid, job in _jobs.items():
            if job["status"] in ("completed", "error"):
                to_remove.append(jid)
        for jid in to_remove[: len(_jobs) - _MAX_JOBS]:
            del _jobs[jid]


def _run_directory_job(
    job_id: str,
    directory: str,
    cfg: VstashConfig,
    *,
    force: bool,
    collection: str,
    meta: dict[str, Any],
) -> None:
    """Background worker for directory ingestion.

    Opens its own VstashStore connection to avoid sharing SQLite across threads.
    """
    try:
        from .embed import get_embedding_dim
        from .ingest import ingest_directory

        from .profile import resolve_db_path

        dim = get_embedding_dim(cfg.embeddings.model)
        db_path = str(resolve_db_path(config_db_path=cfg.storage.db_path))
        store = VstashStore(
            db_path,
            embedding_dim=dim,
            vector_backend=cfg.storage.vector_backend,
            snapvec_bits=cfg.storage.snapvec_bits,
        )
        try:
            results = ingest_directory(
                directory,
                cfg,
                store,
                force=force,
                collection=collection,
                **meta,
            )
        finally:
            store.close()
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "status": "completed",
                    "files_processed": len(results),
                    "results": [r.model_dump() for r in results],
                }
            )
        _cleanup_jobs()
    except Exception as exc:
        logger.exception("Background ingestion failed for job %s", job_id)
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


@mcp_server.tool()
def vstash_add(
    path: str,
    force: bool = False,
    collection: str = "default",
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
) -> str:
    """Ingest a file, directory, or URL into vstash memory.

    Supports PDF, DOCX, PPTX, XLSX, Markdown, plain text, code files,
    and HTTP(S) URLs.

    For directories: ingestion runs in background to avoid timeouts.
    Returns a job_id — poll with vstash_job(job_id) to check progress.
    Excludes __pycache__, node_modules, .venv, .git, and basic patterns from
    a top-level .gitignore (no nested files or negation rules).

    YAML frontmatter (---/--- block) is auto-parsed for project, layer, tags.
    Explicit params override frontmatter values.

    Args:
        path: Absolute file path, directory path, or HTTP(S) URL to ingest.
        force: If True, re-ingest even if the document already exists.
        collection: Named collection to group this document (default: 'default').
        project: Project tag for this document (overrides frontmatter).
        layer: Layer/category for this document (overrides frontmatter).
        tags: Comma-separated tags (overrides frontmatter).

    Returns:
        JSON string with ingestion result, or job_id for directories.
    """
    try:
        from .ingest import ingest

        force = bool(force)
        cfg = _get_config()
        store = _get_store()
        meta = {"project": project, "layer": layer, "tags": tags}

        # URL → skip path resolution, go straight to ingest
        if path.startswith(("http://", "https://")):
            result: IngestResult = ingest(
                path,
                cfg,
                store,
                force=force,
                collection=collection,
                **meta,
            )
            return _ok(result)

        resolved = Path(path).expanduser().resolve()

        # Directory → launch background job
        if resolved.is_dir():
            import uuid

            job_id = uuid.uuid4().hex[:12]
            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "running",
                    "path": str(resolved),
                    "collection": collection,
                }
            threading.Thread(
                target=_run_directory_job,
                args=(job_id, str(resolved), cfg),
                kwargs={
                    "force": force,
                    "collection": collection,
                    "meta": meta,
                },
                daemon=True,
            ).start()
            return _ok(
                {
                    "status": "accepted",
                    "job_id": job_id,
                    "path": str(resolved),
                    "message": "Directory ingestion started in background. "
                    "Use vstash_job(job_id) to check progress.",
                }
            )

        # Single file
        result = ingest(
            str(resolved),
            cfg,
            store,
            force=force,
            collection=collection,
            **meta,
        )
        return _ok(result)

    except FileNotFoundError:
        return _error(f"Path not found: {path}")
    except PermissionError:
        return _error(f"Permission denied: {path}")
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.exception("vstash_add failed")
        return _error(f"Ingestion failed: {exc}")


@mcp_server.tool()
def vstash_remember(
    text: str,
    title: str | None = None,
    collection: str = "default",
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
) -> str:
    """Ingest text directly into vstash memory — no file needed.

    Agent-friendly alternative to vstash_add. Pass text content directly
    instead of writing to a temp file first. Ideal for storing notes,
    decisions, code snippets, or any text an agent wants to remember.

    Args:
        text: The content to ingest and remember.
        title: Human-readable title for the document (used in search results).
            When not provided, a descriptive title is auto-generated from
            the text content plus a UTC timestamp.
        collection: Named collection to group this document (default: 'default').
        project: Project tag for this document.
        layer: Layer/category for this document.
        tags: Comma-separated tags.

    Returns:
        JSON string with ingestion result (chunks created, timing, etc.).
    """
    try:
        from .ingest import ingest_text

        cfg = _get_config()
        store = _get_store()

        result = ingest_text(
            text,
            cfg,
            store,
            title=title,
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
        )
        return _ok(result)

    except Exception as exc:
        logger.exception("vstash_remember failed")
        return _error(f"Ingestion failed: {exc}")


@mcp_server.tool()
def vstash_job(job_id: str) -> str:
    """Check the status of a background ingestion job.

    Directory ingestion runs in the background to avoid timeouts.
    Use this tool to poll for completion.

    Args:
        job_id: The job ID returned by vstash_add for directory ingestion.

    Returns:
        JSON object with job status (running, completed, or error).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return _error(f"Job not found: {job_id}")
    return _ok(job)


@mcp_server.tool()
def vstash_ask(
    query: str,
    top_k: int = 5,
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
    added_after: str | None = None,
    added_before: str | None = None,
) -> str:
    """Query vstash memory and get an LLM-generated answer with sources.

    Performs hybrid semantic + keyword search, then sends the top chunks
    to the configured LLM backend for answer generation.

    Args:
        query: Natural language question to answer from memory.
        top_k: Number of context chunks to retrieve (default: 5).
        collection: If set, restrict search to this collection only.
        project: If set, restrict search to documents with this project tag.
        layer: If set, restrict search to documents with this layer tag.
        added_after: ISO date (e.g. '2024-01-15') — only documents added on or after.
        added_before: ISO date (e.g. '2024-06-01') — only documents added before.

    Returns:
        JSON string with answer text and source documents.
    """
    try:
        from .chat import ask

        top_k = int(top_k)
        cfg = _get_config()
        store = _get_store()

        # Embed query and search
        query_embedding = embed_query(query, cfg.embeddings.model)
        chunks: list[SearchResult] = store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            added_after=added_after,
            added_before=added_before,
        )

        if not chunks:
            return _ok(
                {
                    "answer": "No relevant documents found in memory.",
                    "sources": [],
                }
            )

        # Record search telemetry (same as vstash_search)
        best_distance = store.last_best_distance
        tier = relevance_tier(best_distance)
        store.record_search_event(
            query=query,
            best_distance=best_distance,
            relevance_tier=tier,
            result_count=len(chunks),
        )

        # Expand context with adjacent chunks for richer LLM answers
        chunks = store.expand_context(chunks, window=1)

        # Get LLM answer
        answer = ask(query, chunks, cfg)

        # Build source list
        sources = [{"title": c.title, "path": c.path, "score": c.score} for c in chunks]
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
def vstash_search(
    query: str,
    top_k: int = 5,
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
    recency_boost: float = 0.0,
    added_after: str | None = None,
    added_before: str | None = None,
    mmr_lambda: float = 0.5,
) -> str:
    """Search vstash memory without LLM — returns raw chunks with scores.

    Performs hybrid semantic + keyword search using Reciprocal Rank Fusion.
    Useful for retrieving context without consuming LLM tokens.

    Only use this for knowledge questions about stored documents.
    Do NOT call this for action commands (commit, merge, push, test, deploy).

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default: 5).
        collection: If set, restrict search to this collection only.
        project: If set, restrict search to documents with this project tag.
        layer: If set, restrict search to documents with this layer tag.
        recency_boost: Temporal decay multiplier (0.0 = off). When > 0,
            recent chunks score higher. Use 0.5 for mild bias, 1.0 for strong.
        added_after: ISO date (e.g. '2024-01-15') — only documents added on or after.
        added_before: ISO date (e.g. '2024-06-01') — only documents added before.

    Returns:
        JSON object with chunks array and relevance hint.
    """
    try:
        top_k = int(top_k)
        cfg = _get_config()
        store = _get_store()

        query_embedding = embed_query(query, cfg.embeddings.model)
        chunks: list[SearchResult] = store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            recency_boost=float(recency_boost),
            added_after=added_after,
            added_before=added_before,
            mmr_lambda=float(mmr_lambda),
        )

        if not chunks:
            return _ok({"chunks": [], "relevance": "none"})

        # Expand context: include adjacent chunks (±1) for richer LLM context.
        # This is critical for MCP/Claude Code where the LLM consumes these
        # chunks directly — more context yields better answers.
        chunks = store.expand_context(chunks, window=1)

        # Tiered relevance signal based on vector distance.
        best_distance = store.last_best_distance
        relevance = relevance_tier(best_distance)
        _hints = {
            "high": "Results appear relevant to the query.",
            "medium": "Results may be tangential — consider refining your query.",
            "low": "Results may not be relevant — best match is semantically distant.",
        }
        hint = _hints[relevance]

        store.record_search_event(
            query=query,
            best_distance=best_distance,
            relevance_tier=relevance,
            result_count=len(chunks),
        )

        return _ok(
            {
                "chunks": [c.model_dump(exclude_none=True) for c in chunks],
                "relevance": relevance,
                "hint": hint,
                "best_distance": round(best_distance, 4),
            }
        )

    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_search failed")
        return _error(f"Search failed: {exc}")


@mcp_server.tool()
def vstash_miss_analysis(
    query: str,
    expected_path: str | None = None,
    expected_chunk_id: int | None = None,
    top_k: int = 5,
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
) -> str:
    """Diagnose why an expected document did NOT appear in search results.

    Use this when you (or the user) ran a search expecting a specific
    document and it didn't show up in the top-k.  This tool runs the
    full search pipeline with per-stage tracking and returns a structured
    explanation of which stage eliminated the expected document, plus
    rule-based suggestions to fix the query.

    Args:
        query: The search query to diagnose.
        expected_path: Path of the document the user expected to see.
            Either this or expected_chunk_id must be provided.
        expected_chunk_id: Chunk id to track instead of resolving by path.
        top_k: Number of results to evaluate (default: 5).
        collection: If set, restrict to this collection only.
        project: If set, restrict to documents with this project tag.
        layer: If set, restrict to documents with this layer tag.

    Returns:
        JSON object with stage_verdicts, actual_top_k, dropped_at,
        suggestions, and the original query/expected_path/chunk_id.
    """
    try:
        if expected_path is None and expected_chunk_id is None:
            return _error("Provide either expected_path or expected_chunk_id")
        cfg = _get_config()
        store = _get_store()
        # Normalize path the same way add() does
        if expected_path is not None and not expected_path.startswith(
            ("http://", "https://", "text://")
        ):
            expected_path = str(Path(expected_path).resolve())
        query_embedding = embed_query(query, cfg.embeddings.model)
        analysis = store.miss_analysis(
            query_embedding=query_embedding,
            query_text=query,
            expected_path=expected_path,
            expected_chunk_id=int(expected_chunk_id) if expected_chunk_id is not None else None,
            top_k=int(top_k),
            collection=collection,
            project=project,
            layer=layer,
        )
        return _ok(analysis)
    except ValueError as exc:
        return _error(str(exc))
    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_miss_analysis failed")
        return _error(f"Miss analysis failed: {exc}")


@mcp_server.tool()
def vstash_get_document_chunks(path: str, collection: str | None = None) -> str:
    """Get all chunk texts for a document by its path.

    Use this to retrieve the full content of a previously ingested document,
    reconstructed from its chunks in order.

    Args:
        path: Document path as stored in the database.
        collection: Optional collection filter. If the same path exists in
            multiple collections and no collection is given, returns chunks
            from the most recently added document.

    Returns:
        JSON object with the list of chunk texts, or error if not found.
    """
    try:
        path_str = path
        if not path_str.startswith(("http://", "https://", "text://")):
            path_str = str(Path(path_str).resolve())

        store = _get_store()
        chunks = store.get_document_chunks(path_str, collection=collection)
        if not chunks:
            return _error(f"No chunks found for document: {path_str}")
        return _ok({"path": path_str, "chunk_count": len(chunks), "chunks": chunks})
    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_get_document_chunks failed")
        return _error(f"Get document chunks failed: {exc}")


@mcp_server.tool()
def vstash_get_chunk(chunk_id: int) -> str:
    """Retrieve a single chunk by its database row ID.

    Use this when you already have a chunk_id (e.g., from a previous search result)
    and need to fetch the full text without running a new search.

    Args:
        chunk_id: The integer primary key of the chunk.

    Returns:
        JSON object with chunk text and document metadata, or error if not found.
    """
    try:
        cid = int(chunk_id)
    except (TypeError, ValueError):
        return _error(f"chunk_id must be an integer, got: {chunk_id!r}")
    try:
        store = _get_store()
        chunk = store.get_chunk(cid)
        if chunk is None:
            return _error(f"Chunk {cid} not found.")
        return _ok(chunk.model_dump())
    except FileNotFoundError:
        return _error("vstash database not found. Ingest documents first with vstash_add.")
    except Exception as exc:
        logger.exception("vstash_get_chunk failed")
        return _error(f"Get chunk failed: {exc}")


@mcp_server.tool()
def vstash_list(
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
) -> str:
    """List all documents currently stored in vstash memory.

    Args:
        collection: If set, filter to this collection only.
        project: If set, filter to documents with this project tag.
        layer: If set, filter to documents with this layer tag.

    Returns:
        JSON array of documents with path, title, type, collection,
        project, layer, tags, chunk count, character count, and ingestion timestamp.
    """
    try:
        store = _get_store()
        docs: list[DocumentInfo] = store.list_documents(
            collection=collection,
            project=project,
            layer=layer,
        )
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
def vstash_forget(source: str, collection: str | None = None) -> str:
    """Remove a document from vstash memory by its source path or URL.

    Accepts exact paths or partial matches (filename, title). If an exact
    match is not found, a fuzzy search by filename/title is attempted.
    Use vstash_list() to see currently stored document paths.

    Args:
        source: File path, URL, or partial filename/title of the document to remove.
        collection: If set, restrict fuzzy matching to this collection.

    Returns:
        JSON object with deletion status.
    """
    try:
        store = _get_store()

        # Try exact match first
        deleted = store.delete_document(source)
        if deleted:
            return _ok({"status": "deleted", "source": source})

        # Fallback: fuzzy match by partial path/title
        match = store.find_document(source, collection=collection)
        if match:
            store.delete_document(match)
            return _ok({"status": "deleted", "source": match, "matched_from": source})

        return _ok({"status": "not_found", "source": source})

    except FileNotFoundError:
        return _error("vstash database not found.")
    except Exception as exc:
        logger.exception("vstash_forget failed")
        return _error(f"Failed to delete document: {exc}")


@mcp_server.tool()
def vstash_collections() -> str:
    """List all available collections in vstash memory.

    Returns:
        JSON array of collection names.
    """
    try:
        store = _get_store()
        return _ok(store.list_collections())
    except FileNotFoundError:
        return _error("vstash database not found.")
    except Exception as exc:
        logger.exception("vstash_collections failed")
        return _error(f"Failed to list collections: {exc}")


@mcp_server.tool()
def vstash_export(
    collection: str | None = None,
    project: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
    include_embeddings: bool = False,
) -> str:
    """Export chunks with document metadata for training data curation.

    Returns a JSON object containing all chunks matching the filters, with
    their document metadata (title, path, project, layer, tags, collection).
    Useful for building fine-tuning datasets or importing into other vector
    stores.

    Args:
        collection: Filter by collection name.
        project: Filter by project tag.
        layer: Filter by layer tag.
        tags: Filter by tag substring.
        include_embeddings: Include embedding vectors (large output).

    Returns:
        JSON object with 'chunks' array and 'count'.
    """
    try:
        store = _get_store()
        chunks = store.export_chunks(
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            include_embeddings=include_embeddings,
        )
        return _ok({"chunks": chunks, "count": len(chunks)})
    except FileNotFoundError:
        return _error("vstash database not found.")
    except Exception as exc:
        logger.exception("vstash_export failed")
        return _error(f"Failed to export chunks: {exc}")


# ------------------------------------------------------------------ #
# Journal tools                                                        #
# ------------------------------------------------------------------ #


@mcp_server.tool()
def vstash_journal_save(
    text: str,
    title: str | None = None,
    project: str | None = None,
    tags: str | None = None,
    source: str | None = None,
) -> str:
    """Save a journal entry for cross-session memory.

    Stores text in a dedicated journal profile with auto-timestamped titles.
    Use this to persist decisions, context, and findings across sessions.

    Args:
        text: Content to save in the journal.
        title: Optional title (auto-generated with timestamp if omitted).
        project: Project tag (auto-detected from cwd if omitted).
        tags: Comma-separated tags (journal tag added automatically).
        source: Source identifier (e.g. 'agent', 'session', 'mcp').

    Returns:
        JSON with entry metadata (title, chunks, tags, added_at).
    """
    try:
        from .journal import journal_save

        result = journal_save(text, title=title, project=project, tags=tags, source=source or "mcp")
        return _ok(result)
    except Exception as exc:
        logger.exception("vstash_journal_save failed")
        return _error(f"Journal save failed: {exc}")


@mcp_server.tool()
def vstash_journal_recall(
    query: str | None = None,
    top_k: int = 5,
    project: str | None = None,
) -> str:
    """Recall relevant journal entries from past sessions.

    When query is omitted, returns the most recent entries.
    When query is provided, performs semantic search over the journal.

    Use this at the start of a session to recover context, or when
    the user references prior work.

    Args:
        query: Search query (omit for most recent entries).
        top_k: Number of entries to return (default: 5).
        project: Filter by project tag.

    Returns:
        JSON array of journal entries with text, title, and score/date.
    """
    try:
        from .journal import journal_recall

        entries = journal_recall(query=query, top_k=top_k, project=project)
        return _ok(entries)
    except Exception as exc:
        logger.exception("vstash_journal_recall failed")
        return _error(f"Journal recall failed: {exc}")


@mcp_server.tool()
def vstash_journal_log(
    limit: int = 20,
    recent: str | None = None,
    project: str | None = None,
) -> str:
    """List journal entries chronologically (newest first).

    Shows entry titles, projects, tags, and dates — like git log
    for your memory journal.

    Args:
        limit: Max entries to return (default: 20).
        recent: Time window filter (e.g. '7d', '24h', '2w'). Only entries
            within this window are returned.
        project: Filter by project tag.

    Returns:
        JSON array of entry summaries.
    """
    try:
        from .journal import journal_log

        entries = journal_log(limit=limit, recent=recent, project=project)
        return _ok(entries)
    except Exception as exc:
        logger.exception("vstash_journal_log failed")
        return _error(f"Journal log failed: {exc}")


@mcp_server.tool()
def vstash_journal_prune(
    age: str,
    project: str | None = None,
    dry_run: bool = False,
) -> str:
    """Remove old journal entries to keep the journal lean.

    Deletes entries older than the specified age threshold.

    Args:
        age: Age threshold like '30d' (days), '2w' (weeks), '24h' (hours).
        project: Only prune entries for this project.
        dry_run: If true, report what would be deleted without deleting.

    Returns:
        JSON with count of deleted entries and their titles.
    """
    try:
        from .journal import journal_prune

        result = journal_prune(age, project=project, dry_run=dry_run)
        return _ok(result)
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:
        logger.exception("vstash_journal_prune failed")
        return _error(f"Journal prune failed: {exc}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #


def main() -> None:
    """Run the vstash MCP server over stdio.

    This is the entry point registered as ``vstash-mcp`` in pyproject.toml.
    MCP uses stdio for transport, so any stdout writes from Rich progress
    bars would corrupt the JSON-RPC stream.  We redirect the module-level
    Console instances in ``ingest`` (and any future modules) to stderr.

    The embedding model is warmed up in a background thread to avoid
    blocking the first tool call with model download / JIT compilation.
    """
    import sys

    from rich.console import Console

    stderr_console = Console(file=sys.stderr, force_terminal=False)

    # Patch ingest.py's module-level console so Progress bars go to stderr
    from . import ingest

    ingest.console = stderr_console

    # Pre-load embedding model in background to avoid cold-start latency
    try:
        cfg = _get_config()
        threading.Thread(
            target=warmup,
            args=(cfg.embeddings.model,),
            daemon=True,
        ).start()
        logger.info("Embedding warmup started in background thread")
    except Exception:
        logger.warning("Background warmup failed — will load on first query", exc_info=True)

    mcp_server.run()


if __name__ == "__main__":
    main()
