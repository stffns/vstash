"""
web.py — Minimal web interface for vstash.

    vstash serve          # localhost:8585
    vstash serve --port 9000

A pocket memory agent: chat with your documents, search, upload — all
from a browser.  Uses Starlette (already installed) + uvicorn.  No
FastAPI, no React, no build step.  One HTML page served inline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from ._store_open import open_store_for_config
from .chat import stream as chat_stream
from .config import VstashConfig, load_config
from .embed import embed_query
from .ingest import ingest
from .models import SearchResult
from .store import VstashStore, relevance_tier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Lazy singletons (same pattern as mcp.py)                            #
# ------------------------------------------------------------------ #

_config: VstashConfig | None = None
_store: VstashStore | None = None


def _get_config() -> VstashConfig:
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_config()
    return _config


def _get_store() -> VstashStore:
    global _store  # noqa: PLW0603
    if _store is None:
        cfg = _get_config()
        _store = open_store_for_config(cfg)
        # Detect embedding model drift on non-empty stores.
        if _store.stats().chunks > 0:
            drift_msg = _store.check_embedding_drift(cfg.embeddings.model)
            if drift_msg:
                logger.warning(drift_msg)
    return _store


def _json(data: Any, status: int = 200) -> JSONResponse:
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    elif isinstance(data, list):
        payload = [item.model_dump() if hasattr(item, "model_dump") else item for item in data]
    else:
        payload = data
    return JSONResponse(payload, status_code=status)


async def _run_sync(fn, *args):
    """Run a blocking function in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ------------------------------------------------------------------ #
# Chat history (per-session, in-memory)                                #
# ------------------------------------------------------------------ #

_chat_history: list[dict[str, str]] = []


# ------------------------------------------------------------------ #
# Sync helpers (run in thread pool via _run_sync)                      #
# ------------------------------------------------------------------ #


def _do_search(query: str, top_k: int, recency_boost: float) -> dict:
    cfg = _get_config()
    store = _get_store()
    q_emb = embed_query(query, cfg.embeddings.model)
    results = store.search(
        query_embedding=q_emb,
        query_text=query,
        top_k=top_k,
        recency_boost=recency_boost,
    )
    results = store.expand_context(results, window=1)
    best_distance = store.last_best_distance
    relevance = relevance_tier(best_distance)
    return {
        "results": [r.model_dump(exclude_none=True) for r in results],
        "relevance": relevance,
        "best_distance": round(best_distance, 4),
    }


def _do_documents() -> list:
    return _get_store().list_documents()


def _do_stats() -> Any:
    return _get_store().stats()


_UPLOADS_DIR = Path("~/.vstash/uploads").expanduser()


def _do_upload(file_bytes: bytes, suffix: str, original_name: str | None = None) -> Any:
    """Persist an uploaded blob and ingest it.

    The earlier implementation wrote to ``tempfile.NamedTemporaryFile``
    and ``unlink``ed the path right after ``ingest()`` returned. The
    document then pointed at a deleted ``/var/folders/.../tmpXXX``
    path with the temp basename as title (#295). Persisting under
    ``~/.vstash/uploads/<uuid>-<safe-name>`` gives the stored document
    a stable source the user can later re-open or re-ingest.
    """
    cfg = _get_config()
    store = _get_store()
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Strip any path fragments a client may have included (POSIX or
    # Windows separators) before sanitization so we never persist
    # ``a/b/leaked.md`` as ``a_b_leaked.md``.
    raw_name = original_name or "upload"
    src_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1] or "upload"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in src_name)
    # Cap the FINAL basename (incl. suffix) so an adversarial filename or
    # suffix can't blow past ENAMETOOLONG (255 on most filesystems). The
    # uuid12 prefix + dash add 13 chars to the on-disk name; keep margin.
    _MAX_BASENAME = 128
    safe_suffix = "".join(c if c.isalnum() or c in "._-" else "_" for c in (suffix or ""))
    if safe_suffix and not safe_suffix.startswith("."):
        safe_suffix = "." + safe_suffix
    # If the suffix alone exceeds the cap, truncate it; never let it eat
    # the whole budget.
    if len(safe_suffix) >= _MAX_BASENAME:
        safe_suffix = safe_suffix[: _MAX_BASENAME - 1]
    # Strip an existing copy of the suffix so we always re-append it
    # exactly once at the cap boundary.
    if safe_suffix and safe_name.endswith(safe_suffix):
        safe_name = safe_name[: -len(safe_suffix)]
    if safe_suffix:
        keep = max(1, _MAX_BASENAME - len(safe_suffix))
        safe_name = safe_name[:keep] + safe_suffix
    elif len(safe_name) > _MAX_BASENAME:
        safe_name = safe_name[:_MAX_BASENAME]
    target = _UPLOADS_DIR / f"{uuid.uuid4().hex[:12]}-{safe_name}"
    target.write_bytes(file_bytes)
    return ingest(str(target), cfg, store, force=True)


def _do_chat_retrieve(query: str, top_k: int) -> tuple[list[SearchResult], list[dict]]:
    cfg = _get_config()
    store = _get_store()
    q_emb = embed_query(query, cfg.embeddings.model)
    chunks = store.search(query_embedding=q_emb, query_text=query, top_k=top_k)
    chunks = store.expand_context(chunks, window=1)
    sources = list({c.path: {"title": c.title, "path": c.path} for c in chunks}.values())
    return chunks, sources


def _do_embed(texts: list[str]) -> dict:
    from .embed import embed_texts, get_embedding_dim

    cfg = _get_config()
    model = cfg.embeddings.model
    vectors = embed_texts(texts, model)
    return {
        "embeddings": vectors,
        "model": model,
        "dim": get_embedding_dim(model),
    }


def _do_embed_query(text: str) -> dict:
    cfg = _get_config()
    model = cfg.embeddings.model
    vec = embed_query(text, model)
    return {
        "embedding": vec,
        "model": model,
    }


def _do_miss_analysis(
    query: str,
    expected_path: str | None,
    expected_chunk_id: int | None,
    top_k: int,
    collection: str | None,
    project: str | None,
    layer: str | None,
) -> dict:
    """Run ``VstashStore.miss_analysis`` from the debug endpoint.
    Returns the dumped MissAnalysis as a dict ready for JSONResponse."""
    cfg = _get_config()
    store = _get_store()
    q_emb = embed_query(query, cfg.embeddings.model)
    analysis = store.miss_analysis(
        query_embedding=q_emb,
        query_text=query,
        expected_path=expected_path,
        expected_chunk_id=expected_chunk_id,
        top_k=top_k or cfg.chunking.top_k,
        collection=collection,
        project=project,
        layer=layer,
    )
    return analysis.model_dump(exclude_none=True)


# ------------------------------------------------------------------ #
# API endpoints (all async, blocking work via _run_sync)               #
# ------------------------------------------------------------------ #


async def api_embed(request: Request) -> JSONResponse:
    """POST /api/embed -- embed texts via the warm daemon.

    Request body:
        {"texts": ["hello", "world"]}        -- batch embed
        {"text": "hello"}                     -- single query embed

    Response:
        {"embeddings": [[...], [...]], "model": "...", "dim": 384}
        {"embedding": [...], "model": "..."}
    """
    body = await request.json()
    texts = body.get("texts")
    text = body.get("text")

    _MAX_BATCH = 256

    if texts is not None:
        if (
            not isinstance(texts, list)
            or not texts
            or not all(isinstance(t, str) and t.strip() for t in texts)
        ):
            return _json({"error": "texts must be a non-empty list of non-empty strings"}, 400)
        if len(texts) > _MAX_BATCH:
            return _json({"error": f"batch size {len(texts)} exceeds limit of {_MAX_BATCH}"}, 400)
        data = await _run_sync(_do_embed, texts)
        return _json(data)

    if text is not None:
        if not isinstance(text, str) or not text.strip():
            return _json({"error": "text must be a non-empty string"}, 400)
        data = await _run_sync(_do_embed_query, text)
        return _json(data)

    return _json({"error": "provide 'text' (single) or 'texts' (batch)"}, 400)


async def api_search(request: Request) -> JSONResponse:
    """POST /api/search — hybrid search."""
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    recency_boost = float(body.get("recency_boost", 0.0))

    if not query.strip():
        return _json({"error": "query is required"}, 400)

    data = await _run_sync(_do_search, query, top_k, recency_boost)
    return _json(data)


async def api_chat(request: Request) -> Any:
    """POST /api/chat — streaming chat with memory context (SSE)."""
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))

    if not query.strip():
        return _json({"error": "query is required"}, 400)

    chunks, sources = await _run_sync(_do_chat_retrieve, query, top_k)

    if not chunks:

        async def no_results():
            yield f"data: {json.dumps({'token': 'No relevant documents found in memory.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

        return StreamingResponse(no_results(), media_type="text/event-stream")

    cfg = _get_config()
    _chat_history.append({"role": "user", "content": query})

    async def generate():
        full_response = ""
        try:
            loop = asyncio.get_event_loop()
            # Run the blocking stream iterator in a thread
            q = asyncio.Queue()

            def _stream_worker():
                try:
                    for token in chat_stream(query, chunks, cfg, history=_chat_history[:-1]):
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token))
                    loop.call_soon_threadsafe(q.put_nowait, ("done", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

            thread = threading.Thread(target=_stream_worker, daemon=True)
            thread.start()

            while True:
                kind, value = await q.get()
                if kind == "token":
                    full_response += value
                    yield f"data: {json.dumps({'token': value})}\n\n"
                elif kind == "done":
                    _chat_history.append({"role": "assistant", "content": full_response})
                    yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
                    break
                elif kind == "error":
                    msg = value
                    if "Connection refused" in msg or "connect" in msg.lower():
                        msg = f"LLM backend not available. Check your inference config (current: {cfg.inference.backend}). Error: {msg}"
                    yield f"data: {json.dumps({'error': msg})}\n\n"
                    break
        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def api_chat_reset(request: Request) -> JSONResponse:
    """POST /api/chat/reset — clear chat history."""
    _chat_history.clear()
    return _json({"status": "ok"})


async def api_documents(request: Request) -> JSONResponse:
    """GET /api/documents — list all documents."""
    docs = await _run_sync(_do_documents)
    return _json(docs)


async def api_stats(request: Request) -> JSONResponse:
    """GET /api/stats — memory statistics."""
    stats = await _run_sync(_do_stats)
    return _json(stats)


async def api_upload(request: Request) -> JSONResponse:
    """POST /api/upload — add a file to memory."""
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return _json({"error": "No file provided"}, 400)

    suffix = Path(upload.filename).suffix if upload.filename else ".txt"
    content = await upload.read()
    result = await _run_sync(_do_upload, content, suffix, upload.filename)
    return _json(result)


# ------------------------------------------------------------------ #
# HTML frontend (served inline)                                        #
# ------------------------------------------------------------------ #


async def index(request: Request) -> HTMLResponse:
    """GET / — serve the single-page app."""
    return HTMLResponse(_HTML)


# ------------------------------------------------------------------ #
# Observability endpoints (#132)                                        #
# ------------------------------------------------------------------ #


def _do_health_check() -> None:
    """Lightweight liveness probe: verify the SQLite connection is alive
    by running ``SELECT 1``.  Cheap enough to run on every probe without
    fighting a busy ingest lock."""
    store = _get_store()
    store._conn.execute("SELECT 1").fetchone()


async def api_health(request: Request) -> JSONResponse:
    """GET /health — returns 200 if the store connection is reachable.

    Designed for load balancers, Docker health probes, and uptime
    monitoring.  Runs a trivial ``SELECT 1`` — not ``stats()`` — so a
    busy ingest doesn't cause k8s to restart healthy pods.

    Returns a minimal payload with version and uptime for dashboards
    that want a single round-trip for identity + liveness.
    """
    from . import __version__
    from .metrics import registry

    try:
        await _run_sync(_do_health_check)
    except Exception:
        # Log the full exception server-side but return a generic error
        # to the client.  /health is typically exposed without auth so
        # we must not leak file paths, stack traces, or library
        # internals to whoever is probing.
        logger.exception("health check failed")
        return _json({"status": "error"}, status=503)

    snap = registry.snapshot()
    return _json(
        {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": snap["uptime_seconds"],
        },
        status=200,
    )


async def api_debug_why(request: Request) -> JSONResponse:
    """GET /debug/why?q=...&expect=... -- miss analysis as JSON.

    Issue #157 part 2. Behind the ``--debug`` flag on ``vstash serve``
    because the endpoint echoes arbitrary query text + path back to
    the caller and is intended for local diagnostic use, not public
    consumption. When ``--debug`` is not set, this route is NOT
    mounted and the URL returns 404.

    Query parameters:
        q (required): the search query that missed.
        expect (optional): expected document path.
        expect_chunk_id (optional): specific chunk id.
          One of ``expect`` / ``expect_chunk_id`` is required.
        top_k (optional, int, default from config): top-k window.
        collection / project / layer (optional): metadata filters.

    Response: the MissAnalysis model as JSON, ``exclude_none=True``.

    Errors return 400 with ``{"error": "..."}`` for input problems,
    500 for unexpected internals.
    """
    qp = request.query_params
    query = (qp.get("q") or "").strip()
    # Normalize ``expect`` the same way the CLI and Memory SDK do:
    # http(s)/text URIs pass through verbatim, everything else resolves
    # to an absolute path (ingest stores ``Path(source).resolve()``,
    # so a caller passing a relative path would otherwise reliably get
    # "No chunks found" even when the doc exists). ``.strip()`` treats
    # whitespace-only values as missing.
    expect_raw = (qp.get("expect") or "").strip()
    if not expect_raw:
        expect = None
    elif expect_raw.startswith(("http://", "https://", "text://")):
        expect = expect_raw
    else:
        expect = str(Path(expect_raw).resolve(strict=False))
    expect_chunk_id_raw = qp.get("expect_chunk_id")
    top_k_raw = qp.get("top_k")
    collection = qp.get("collection") or None
    project = qp.get("project") or None
    layer = qp.get("layer") or None

    if not query:
        return _json({"error": "q (query) is required"}, 400)
    if expect is None and expect_chunk_id_raw is None:
        return _json({"error": "one of expect=<path> or expect_chunk_id=<id> is required"}, 400)

    expect_chunk_id: int | None = None
    if expect_chunk_id_raw is not None:
        try:
            expect_chunk_id = int(expect_chunk_id_raw)
        except ValueError:
            return _json({"error": "expect_chunk_id must be an integer"}, 400)

    top_k = 0
    if top_k_raw is not None:
        try:
            top_k = int(top_k_raw)
        except ValueError:
            return _json({"error": "top_k must be an integer"}, 400)

    try:
        data = await _run_sync(
            _do_miss_analysis,
            query,
            expect,
            expect_chunk_id,
            top_k,
            collection,
            project,
            layer,
        )
    except ValueError as exc:
        # Expected-path / chunk-id resolution failures come through here.
        return _json({"error": str(exc)}, 400)
    except Exception:
        logger.exception("debug/why failed")
        return _json({"error": "internal error"}, 500)

    return _json(data)


async def api_metrics(request: Request) -> JSONResponse:
    """GET /metrics — snapshot of the metrics registry as JSON.

    Intentionally not Prometheus text format — keeps dependencies light
    and the JSON is trivial to scrape.  Operators running Prometheus
    can stand up a small translator in their agent.

    Note: gauges are refreshed by ``stats()`` calls (via the CLI or
    programmatic use), not by this endpoint.  Scrapers polling every
    few seconds don't need fresh-to-the-millisecond values, and forcing
    a ``SELECT COUNT(*)`` here would serialize all scrapes behind the
    single SQLite connection.
    """
    from .metrics import registry

    return _json(registry.snapshot())


# ------------------------------------------------------------------ #
# App factory                                                          #
# ------------------------------------------------------------------ #


@asynccontextmanager
async def _lifespan(app: Starlette):  # noqa: ARG001
    """Close the store on shutdown.

    Without an explicit close(), the sqlite3 connection and the
    sqlite-vec C extension tear down in unpredictable order during
    interpreter shutdown, producing a malloc double-free crash on
    Ctrl+C. Closing deterministically while the event loop is still
    alive avoids that race.
    """
    yield
    global _store  # noqa: PLW0603
    if _store is not None:
        try:
            _store.close()
        except Exception:  # noqa: BLE001
            logger.warning("store.close() failed during shutdown", exc_info=True)
        _store = None


def create_app(debug: bool = False) -> Starlette:
    """Create the Starlette app with all routes.

    Args:
        debug: When True, also mounts ``/debug/why`` (miss analysis as
            JSON). Issue #157 part 2. Off by default because the
            endpoint echoes arbitrary query text back and is intended
            for local diagnostic use, not production.
    """
    # Eagerly initialize store to avoid lazy-init deadlocks with asyncio
    _get_store()
    routes = [
        Route("/", index),
        Route("/api/embed", api_embed, methods=["POST"]),
        Route("/api/search", api_search, methods=["POST"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Route("/api/chat/reset", api_chat_reset, methods=["POST"]),
        Route("/api/documents", api_documents),
        Route("/api/stats", api_stats),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/health", api_health),
        Route("/metrics", api_metrics),
    ]
    if debug:
        routes.append(Route("/debug/why", api_debug_why))
    return Starlette(lifespan=_lifespan, routes=routes)


# ------------------------------------------------------------------ #
# Inline HTML — the entire frontend                                    #
# ------------------------------------------------------------------ #

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vstash</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
  --accent-dim: #1f6feb; --green: #3fb950; --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); height: 100vh;
  display: flex; overflow: hidden;
}
.sidebar {
  width: 280px; background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; flex-shrink: 0; min-height: 0;
}
.sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); }
.sidebar-header h1 { font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.sidebar-header h1 span { color: var(--accent); }
.stats {
  padding: 12px 20px; font-size: 12px; color: var(--text-dim);
  border-bottom: 1px solid var(--border); display: flex; gap: 16px;
}
.stat-val { color: var(--text); font-weight: 600; }
.doc-list { flex: 1; overflow-y: auto; padding: 8px; }
.doc-item { padding: 8px 12px; border-radius: var(--radius); font-size: 13px; cursor: default; margin-bottom: 2px; }
.doc-item:hover { background: var(--border); }
.doc-title { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { color: var(--text-dim); font-size: 11px; margin-top: 2px; }
.sidebar-footer { padding: 12px; border-top: 1px solid var(--border); }
.upload-zone {
  border: 2px dashed var(--border); border-radius: var(--radius);
  padding: 16px; text-align: center; font-size: 13px; color: var(--text-dim);
  cursor: pointer; transition: border-color 0.2s;
}
.upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); color: var(--accent); }
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); flex-shrink: 0; }
.tab {
  padding: 10px 20px; font-size: 13px; color: var(--text-dim);
  cursor: pointer; border-bottom: 2px solid transparent;
  background: none; border-top: none; border-left: none; border-right: none;
}
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; flex: 1; flex-direction: column; min-height: 0; }
.tab-content.active { display: flex; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.msg {
  max-width: 85%; padding: 12px 16px; border-radius: var(--radius); font-size: 14px;
  line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;
}
.msg.user { align-self: flex-end; background: var(--accent-dim); color: white; }
.msg.assistant { align-self: flex-start; background: var(--surface); border: 1px solid var(--border); }
.msg .sources {
  margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
  font-size: 12px; color: var(--text-dim);
}
.msg .sources a { color: var(--accent); text-decoration: none; }
.input-bar { padding: 16px 20px; border-top: 1px solid var(--border); display: flex; gap: 8px; flex-shrink: 0; }
.input-bar input {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 10px 14px; color: var(--text); font-size: 14px; outline: none;
}
.input-bar input:focus { border-color: var(--accent); }
.input-bar button {
  background: var(--accent-dim); color: white; border: none;
  border-radius: var(--radius); padding: 10px 18px; font-size: 14px; cursor: pointer;
}
.input-bar button:hover { background: var(--accent); }
.input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
.search-bar { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; flex-shrink: 0; }
.search-bar input {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 10px 14px; color: var(--text); font-size: 14px; outline: none;
}
.search-bar input:focus { border-color: var(--accent); }
.search-results { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.result-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.result-title { font-weight: 600; font-size: 14px; }
.result-score { font-size: 12px; color: var(--green); background: #0d1117; padding: 2px 8px; border-radius: 12px; }
.result-text { font-size: 13px; line-height: 1.5; color: var(--text-dim); max-height: 120px; overflow: hidden; }
.result-path { font-size: 11px; color: var(--text-dim); margin-top: 8px; }
.welcome { flex: 1; display: flex; align-items: center; justify-content: center; flex-direction: column; color: var(--text-dim); gap: 8px; }
.welcome h2 { color: var(--text); font-size: 20px; }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header"><h1><span>v</span>stash</h1></div>
  <div class="stats" id="stats">
    <div><span class="stat-val" id="stat-docs">-</span> docs</div>
    <div><span class="stat-val" id="stat-chunks">-</span> chunks</div>
    <div><span class="stat-val" id="stat-size">-</span> MB</div>
  </div>
  <div class="doc-list" id="doc-list"></div>
  <div class="sidebar-footer">
    <div class="upload-zone" id="upload-zone">Drop files here or click to upload
      <input type="file" id="file-input" hidden multiple>
    </div>
  </div>
</div>
<div class="main">
  <div class="tabs">
    <button class="tab active" data-tab="chat">Chat</button>
    <button class="tab" data-tab="search">Search</button>
  </div>
  <div class="tab-content active" id="tab-chat">
    <div class="messages" id="messages">
      <div class="welcome"><h2>Ask your memory anything</h2>
      <p>Chat with your documents. Answers are generated from your local memory.</p></div>
    </div>
    <div class="input-bar">
      <input type="text" id="chat-input" placeholder="Ask a question..." autofocus>
      <button id="chat-send">Send</button>
    </div>
  </div>
  <div class="tab-content" id="tab-search">
    <div class="search-bar"><input type="text" id="search-input" placeholder="Search your documents..."></div>
    <div class="search-results" id="search-results">
      <div class="welcome"><h2>Hybrid search</h2><p>Vector similarity + keyword matching. No LLM needed.</p></div>
    </div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const NL = String.fromCharCode(10);

$$('.tab').forEach(t => t.addEventListener('click', () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.tab-content').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('tab-' + t.dataset.tab).classList.add('active');
}));

async function loadSidebar() {
  try {
    const [stats, docs] = await Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/documents').then(r => r.json()),
    ]);
    $('#stat-docs').textContent = stats.documents;
    $('#stat-chunks').textContent = stats.chunks;
    $('#stat-size').textContent = stats.db_size_mb.toFixed(1);
    const list = $('#doc-list');
    list.innerHTML = docs.map(d =>
      '<div class="doc-item"><div class="doc-title">' + esc(d.title) +
      '</div><div class="doc-meta">' + d.chunk_count + ' chunks</div></div>'
    ).join('');
  } catch(e) { console.error('sidebar', e); }
}
loadSidebar();

const msgBox = $('#messages');
const chatInput = $('#chat-input');
const chatSend = $('#chat-send');

chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) sendChat(); });
chatSend.addEventListener('click', sendChat);

async function sendChat() {
  const query = chatInput.value.trim();
  if (!query) return;
  chatInput.value = '';
  chatSend.disabled = true;
  const welcome = msgBox.querySelector('.welcome');
  if (welcome) welcome.remove();
  const userMsg = document.createElement('div');
  userMsg.className = 'msg user';
  userMsg.textContent = query;
  msgBox.appendChild(userMsg);
  const aMsg = document.createElement('div');
  aMsg.className = 'msg assistant';
  aMsg.textContent = '';
  msgBox.appendChild(aMsg);
  msgBox.scrollTop = msgBox.scrollHeight;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: query}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({error: 'Server error'}));
      aMsg.textContent = err.error || 'Request failed';
      chatSend.disabled = false;
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const rd = await reader.read();
      if (rd.done) break;
      buf += decoder.decode(rd.value, {stream: true});
      const parts = buf.split(NL + NL);
      buf = parts.pop();
      for (const part of parts) {
        const dataLine = part.split(NL).find(l => l.startsWith('data: '));
        if (!dataLine) continue;
        try {
          const evt = JSON.parse(dataLine.slice(6));
          if (evt.token) { aMsg.textContent += evt.token; msgBox.scrollTop = msgBox.scrollHeight; }
          if (evt.done && evt.sources) {
            const srcDiv = document.createElement('div');
            srcDiv.className = 'sources';
            srcDiv.innerHTML = 'Sources: ' + evt.sources.map(s =>
              '<a title="' + esc(s.path) + '">' + esc(s.title) + '</a>'
            ).join(', ');
            aMsg.appendChild(srcDiv);
          }
          if (evt.error) { aMsg.textContent = evt.error; aMsg.style.color = '#f85149'; }
        } catch(e) {}
      }
    }
  } catch(e) { aMsg.textContent = 'Connection error: ' + e.message; aMsg.style.color = '#f85149'; }
  chatSend.disabled = false;
  chatInput.focus();
}

const searchInput = $('#search-input');
let searchTimeout;
searchInput.addEventListener('input', () => { clearTimeout(searchTimeout); searchTimeout = setTimeout(doSearch, 300); });
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
  const query = searchInput.value.trim();
  const box = $('#search-results');
  if (!query) { box.innerHTML = '<div class="welcome"><h2>Hybrid search</h2><p>Vector similarity + keyword matching.</p></div>'; return; }
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: query, top_k: 10}),
    });
    const data = await res.json();
    if (!data.results || data.results.length === 0) {
      box.innerHTML = '<div class="welcome"><p>No results found.</p></div>';
      return;
    }
    box.innerHTML = data.results.map(r =>
      '<div class="result-card">' +
      '<div class="result-header"><span class="result-title">' + esc(r.title) + '</span>' +
      '<span class="result-score">' + r.score.toFixed(4) + '</span></div>' +
      '<div class="result-text">' + esc(r.text) + '</div>' +
      '<div class="result-path">' + esc(r.path) + '</div></div>'
    ).join('');
  } catch(e) { box.innerHTML = '<div class="welcome"><p>Error: ' + esc(e.message) + '</p></div>'; }
}

const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); uploadFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

async function uploadFiles(files) {
  for (const file of files) {
    uploadZone.textContent = 'Uploading ' + file.name + '...';
    const form = new FormData();
    form.append('file', file);
    try { await fetch('/api/upload', {method: 'POST', body: form}); } catch(e) { console.error('upload', e); }
  }
  uploadZone.textContent = 'Drop files here or click to upload';
  loadSidebar();
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
</script>
</body>
</html>
"""
