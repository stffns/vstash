"""
web.py — Minimal web interface for vstash.

    vstash serve          # localhost:8585
    vstash serve --port 9000

A pocket memory agent: chat with your documents, search, upload — all
from a browser.  Uses Starlette (already installed) + uvicorn.  No
FastAPI, no React, no build step.  One HTML page served inline.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .chat import stream as chat_stream
from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .ingest import ingest
from .models import SearchResult
from .store import VstashStore, relevance_tier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Lazy singletons (same pattern as mcp.py)                            #
# ------------------------------------------------------------------ #

_config: VstashConfig | None = None
_store: VstashStore | None = None
_lock = threading.Lock()


def _get_config() -> VstashConfig:
    global _config  # noqa: PLW0603
    if _config is None:
        with _lock:
            if _config is None:
                _config = load_config()
    return _config


def _get_store() -> VstashStore:
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
                )
    return _store


def _json(data: Any, status: int = 200) -> JSONResponse:
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    elif isinstance(data, list):
        payload = [item.model_dump() if hasattr(item, "model_dump") else item for item in data]
    else:
        payload = data
    return JSONResponse(payload, status_code=status)


# ------------------------------------------------------------------ #
# Chat history (per-session, in-memory)                                #
# ------------------------------------------------------------------ #

_chat_history: list[dict[str, str]] = []

# ------------------------------------------------------------------ #
# API endpoints                                                        #
# ------------------------------------------------------------------ #


async def api_search(request: Request) -> JSONResponse:
    """POST /api/search — hybrid search."""
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    recency_boost = float(body.get("recency_boost", 0.0))

    if not query.strip():
        return _json({"error": "query is required"}, 400)

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

    return _json(
        {
            "results": [r.model_dump(exclude_none=True) for r in results],
            "relevance": relevance,
            "best_distance": round(best_distance, 4),
        }
    )


async def api_chat(request: Request) -> Any:
    """POST /api/chat — streaming chat with memory context (SSE)."""
    from starlette.responses import StreamingResponse

    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))

    if not query.strip():
        return _json({"error": "query is required"}, 400)

    cfg = _get_config()
    store = _get_store()

    # Retrieve context
    q_emb = embed_query(query, cfg.embeddings.model)
    chunks: list[SearchResult] = store.search(query_embedding=q_emb, query_text=query, top_k=top_k)
    chunks = store.expand_context(chunks, window=1)
    sources = list({c.path: {"title": c.title, "path": c.path} for c in chunks}.values())

    if not chunks:

        async def no_results():
            yield f"data: {json.dumps({'token': 'No relevant documents found in memory.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

        return StreamingResponse(no_results(), media_type="text/event-stream")

    # Build history
    _chat_history.append({"role": "user", "content": query})

    async def generate():
        full_response = ""
        try:
            for token in chat_stream(query, chunks, cfg, history=_chat_history[:-1]):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            _chat_history.append({"role": "assistant", "content": full_response})
            yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
        except Exception as exc:
            logger.exception("Chat stream error")
            msg = str(exc)
            if "Connection refused" in msg or "connect" in msg.lower():
                msg = f"LLM backend not available. Check your inference config (current: {cfg.inference.backend}). Error: {msg}"
            yield f"data: {json.dumps({'error': msg})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def api_chat_reset(request: Request) -> JSONResponse:
    """POST /api/chat/reset — clear chat history."""
    _chat_history.clear()
    return _json({"status": "ok"})


async def api_documents(request: Request) -> JSONResponse:
    """GET /api/documents — list all documents."""
    store = _get_store()
    docs = store.list_documents()
    return _json(docs)


async def api_stats(request: Request) -> JSONResponse:
    """GET /api/stats — memory statistics."""
    store = _get_store()
    return _json(store.stats())


async def api_upload(request: Request) -> JSONResponse:
    """POST /api/upload — add a file to memory."""
    import tempfile

    cfg = _get_config()
    store = _get_store()

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return _json({"error": "No file provided"}, 400)

    # Save to temp file, ingest, clean up
    suffix = Path(upload.filename).suffix if upload.filename else ".txt"
    fd_temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await upload.read()
        fd_temp.write(content)
        fd_temp.close()
        result = ingest(fd_temp.name, cfg, store, force=True)
        return _json(result)
    finally:
        Path(fd_temp.name).unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# HTML frontend (served inline)                                        #
# ------------------------------------------------------------------ #


async def index(request: Request) -> HTMLResponse:
    """GET / — serve the single-page app."""
    return HTMLResponse(_HTML)


# ------------------------------------------------------------------ #
# App factory                                                          #
# ------------------------------------------------------------------ #


def create_app() -> Starlette:
    """Create the Starlette app with all routes."""
    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/search", api_search, methods=["POST"]),
            Route("/api/chat", api_chat, methods=["POST"]),
            Route("/api/chat/reset", api_chat_reset, methods=["POST"]),
            Route("/api/documents", api_documents),
            Route("/api/stats", api_stats),
            Route("/api/upload", api_upload, methods=["POST"]),
        ],
    )


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

/* Sidebar */
.sidebar {
  width: 280px; background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; flex-shrink: 0;
}
.sidebar-header {
  padding: 20px; border-bottom: 1px solid var(--border);
}
.sidebar-header h1 {
  font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px;
}
.sidebar-header h1 span { color: var(--accent); }
.stats {
  padding: 12px 20px; font-size: 12px; color: var(--text-dim);
  border-bottom: 1px solid var(--border); display: flex; gap: 16px;
}
.stat-val { color: var(--text); font-weight: 600; }
.doc-list {
  flex: 1; overflow-y: auto; padding: 8px;
}
.doc-item {
  padding: 8px 12px; border-radius: var(--radius); font-size: 13px;
  cursor: default; margin-bottom: 2px;
}
.doc-item:hover { background: var(--border); }
.doc-title { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { color: var(--text-dim); font-size: 11px; margin-top: 2px; }
.sidebar-footer {
  padding: 12px; border-top: 1px solid var(--border);
}
.upload-zone {
  border: 2px dashed var(--border); border-radius: var(--radius);
  padding: 16px; text-align: center; font-size: 13px; color: var(--text-dim);
  cursor: pointer; transition: border-color 0.2s;
}
.upload-zone:hover, .upload-zone.dragover {
  border-color: var(--accent); color: var(--accent);
}

/* Main area */
.main { flex: 1; display: flex; flex-direction: column; }

/* Tabs */
.tabs {
  display: flex; border-bottom: 1px solid var(--border); background: var(--surface);
}
.tab {
  padding: 10px 20px; font-size: 13px; color: var(--text-dim);
  cursor: pointer; border-bottom: 2px solid transparent;
  background: none; border-top: none; border-left: none; border-right: none;
}
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; flex: 1; flex-direction: column; }
.tab-content.active { display: flex; }

/* Chat */
.messages {
  flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px;
}
.msg {
  max-width: 85%; padding: 12px 16px; border-radius: var(--radius); font-size: 14px;
  line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;
}
.msg.user {
  align-self: flex-end; background: var(--accent-dim); color: white;
}
.msg.assistant {
  align-self: flex-start; background: var(--surface); border: 1px solid var(--border);
}
.msg .sources {
  margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
  font-size: 12px; color: var(--text-dim);
}
.msg .sources a { color: var(--accent); text-decoration: none; }
.input-bar {
  padding: 16px 20px; border-top: 1px solid var(--border); display: flex; gap: 8px;
}
.input-bar input {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 10px 14px; color: var(--text);
  font-size: 14px; outline: none;
}
.input-bar input:focus { border-color: var(--accent); }
.input-bar button {
  background: var(--accent-dim); color: white; border: none;
  border-radius: var(--radius); padding: 10px 18px; font-size: 14px;
  cursor: pointer;
}
.input-bar button:hover { background: var(--accent); }
.input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Search */
.search-container { flex: 1; display: flex; flex-direction: column; }
.search-bar {
  padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; gap: 8px;
}
.search-bar input {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 10px 14px; color: var(--text);
  font-size: 14px; outline: none;
}
.search-bar input:focus { border-color: var(--accent); }
.search-results {
  flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px;
}
.result-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px;
}
.result-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 8px;
}
.result-title { font-weight: 600; font-size: 14px; }
.result-score {
  font-size: 12px; color: var(--green); background: #0d1117;
  padding: 2px 8px; border-radius: 12px;
}
.result-text {
  font-size: 13px; line-height: 1.5; color: var(--text-dim);
  max-height: 120px; overflow: hidden;
}
.result-path { font-size: 11px; color: var(--text-dim); margin-top: 8px; }

/* Welcome */
.welcome {
  flex: 1; display: flex; align-items: center; justify-content: center;
  flex-direction: column; color: var(--text-dim); gap: 8px;
}
.welcome h2 { color: var(--text); font-size: 20px; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">
    <h1><span>v</span>stash</h1>
  </div>
  <div class="stats" id="stats">
    <div><span class="stat-val" id="stat-docs">-</span> docs</div>
    <div><span class="stat-val" id="stat-chunks">-</span> chunks</div>
    <div><span class="stat-val" id="stat-size">-</span> MB</div>
  </div>
  <div class="doc-list" id="doc-list"></div>
  <div class="sidebar-footer">
    <div class="upload-zone" id="upload-zone">
      Drop files here or click to upload
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
      <div class="welcome">
        <h2>Ask your memory anything</h2>
        <p>Chat with your documents. Answers are generated from your local memory.</p>
      </div>
    </div>
    <div class="input-bar">
      <input type="text" id="chat-input" placeholder="Ask a question..." autofocus>
      <button id="chat-send">Send</button>
    </div>
  </div>

  <div class="tab-content" id="tab-search">
    <div class="search-container">
      <div class="search-bar">
        <input type="text" id="search-input" placeholder="Search your documents...">
      </div>
      <div class="search-results" id="search-results">
        <div class="welcome">
          <h2>Hybrid search</h2>
          <p>Vector similarity + keyword matching. No LLM needed.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// Tabs
$$('.tab').forEach(t => t.addEventListener('click', () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.tab-content').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $(`#tab-${t.dataset.tab}`).classList.add('active');
}));

// Stats + docs
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
    list.innerHTML = docs.map(d => `
      <div class="doc-item">
        <div class="doc-title">${esc(d.title)}</div>
        <div class="doc-meta">${d.chunk_count} chunks &middot; ${d.source_type}</div>
      </div>
    `).join('');
  } catch(e) { console.error('sidebar', e); }
}
loadSidebar();

// Chat
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

  // Clear welcome
  const welcome = msgBox.querySelector('.welcome');
  if (welcome) welcome.remove();

  // User bubble
  const userMsg = document.createElement('div');
  userMsg.className = 'msg user';
  userMsg.textContent = query;
  msgBox.appendChild(userMsg);

  // Assistant bubble
  const aMsg = document.createElement('div');
  aMsg.className = 'msg assistant';
  aMsg.textContent = '';
  msgBox.appendChild(aMsg);
  msgBox.scrollTop = msgBox.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query}),
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
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const parts = buf.split(String.fromCharCode(10) + String.fromCharCode(10));
      buf = parts.pop();
      for (const part of parts) {
        const dataLine = part.split(String.fromCharCode(10)).find(l => l.startsWith("data: "));
        if (!dataLine) continue;
        try {
          const evt = JSON.parse(dataLine.slice(6));
          if (evt.token) {
            aMsg.textContent += evt.token;
            msgBox.scrollTop = msgBox.scrollHeight;
          }
          if (evt.done && evt.sources) {
            const srcDiv = document.createElement('div');
            srcDiv.className = 'sources';
            srcDiv.innerHTML = 'Sources: ' + evt.sources.map(s =>
              `<a title="${esc(s.path)}">${esc(s.title)}</a>`
            ).join(', ');
            aMsg.appendChild(srcDiv);
          }
          if (evt.error) {
            aMsg.textContent = evt.error;
            aMsg.style.color = '#f85149';
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    aMsg.textContent = 'Connection error: ' + e.message;
    aMsg.style.color = '#f85149';
  }
  chatSend.disabled = false;
  chatInput.focus();
}

// Search
const searchInput = $('#search-input');
let searchTimeout;
searchInput.addEventListener('input', () => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(doSearch, 300);
});
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
  const query = searchInput.value.trim();
  const box = $('#search-results');
  if (!query) { box.innerHTML = '<div class="welcome"><h2>Hybrid search</h2><p>Vector similarity + keyword matching.</p></div>'; return; }
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, top_k: 10}),
    });
    const data = await res.json();
    if (!data.results || data.results.length === 0) {
      box.innerHTML = '<div class="welcome"><p>No results found.</p></div>';
      return;
    }
    box.innerHTML = data.results.map(r => `
      <div class="result-card">
        <div class="result-header">
          <span class="result-title">${esc(r.title)}</span>
          <span class="result-score">${r.score.toFixed(4)}</span>
        </div>
        <div class="result-text">${esc(r.text)}</div>
        <div class="result-path">${esc(r.path)}</div>
      </div>
    `).join('');
  } catch(e) { box.innerHTML = '<div class="welcome"><p>Error: '+esc(e.message)+'</p></div>'; }
}

// Upload
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

async function uploadFiles(files) {
  for (const file of files) {
    uploadZone.textContent = `Uploading ${file.name}...`;
    const form = new FormData();
    form.append('file', file);
    try {
      await fetch('/api/upload', {method: 'POST', body: form});
    } catch(e) { console.error('upload', e); }
  }
  uploadZone.textContent = 'Drop files here or click to upload';
  loadSidebar();
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
</script>
</body>
</html>
"""
