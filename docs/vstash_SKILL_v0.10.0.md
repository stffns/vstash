---
name: vstash
description: >
  Persistent memory system for Jay using vstash MCP tools. USE THIS SKILL at the
  start of EVERY session to load context from previous sessions. Also use it
  whenever Jay mentions past work, projects, decisions, or asks about something
  that might have been discussed before — even casually (like "remember when we...",
  "what did we decide about...", "how does X work again?").
  Always proactively ingest important new information during sessions — architectural
  decisions, project names, benchmarks, key outcomes, action items. Clean up
  irrelevant or outdated data when it's no longer useful.
  vstash is Jay's long-term memory. Treat it as the source of truth for context
  across sessions.
---

# vstash Memory System

Jay uses vstash as persistent memory between Cowork sessions. The MCP tools
connect directly to his Mac. Use them without asking for permission.

## Session Start Protocol

At the start of every session, before answering questions that might involve
past context, run:

```
vstash_ask("current projects, decisions, and context for Jay")
```

Use the result silently to inform your responses. Don't announce "I queried
vstash" unless it's directly relevant to the user's question.

---

## Saving to vstash

Always use `vstash_remember(text="...", title="...")`. The `title` parameter
makes each note identifiable in search results — **always provide one**.

If `title` is omitted, vstash auto-generates a slug from the first words +
UTC timestamp (e.g. `oauth2-uses-pkce-for-public-20260330-143052474102`).
This is useful but less readable — prefer explicit titles.

**Naming convention for title:** `YYYY-MM-DD — Topic`
Examples: `2026-03-31 — Meetings`, `2026-03-30 — Daily Outlook`, `2026-03-30 — Decision: Auth Flow`

**Example:**
```python
vstash_remember(
    text="- 11:30 OKR Monthly Review\n- 13:00 Daily standup...",
    title="2026-03-31 — Meetings"
)
```

### vstash_add (for files and URLs only)

vstash runs on Jay's Mac. The MCP server cannot see VM paths like /sessions/...

| VM path (write files here) | Mac path (use in vstash_add) |
|---|---|
| /sessions/[session]/mnt/Personal/Projects/vstash-memory/ | /Users/jaysonsteffens/Desktop/Personal/Projects/vstash-memory/ |

Never call vstash_add with a /sessions/... path — it will always fail.

---

## When to Ingest

Save information to vstash when:
- An important architectural or technical decision is made
- A project reaches a milestone (shipped, published, merged, etc.)
- Key benchmarks or data points are established
- Action items or next steps are agreed upon
- Personal context about Jay's work or preferences is shared

Keep entries concise. Use headers: ## Project, ## Decision, ## Context, ## Next Steps

## When to Forget

Remove documents from vstash when:
- Information is outdated and replaced by newer context
- A project was abandoned or renamed
- Temporary test data is no longer needed

Use the **full path** with `vstash_forget`:
- Files: `vstash_forget("<Mac file path>")`
- URLs: `vstash_forget("<url>")`
- Text ingested via `vstash_remember`: `vstash_forget("text://<title>")`

Examples:
```
vstash_forget("/Users/jaysonsteffens/Desktop/notes.md")
vstash_forget("https://example.com/article")
vstash_forget("text://2026-03-31 — Meetings")
```

To find the correct path, use `vstash_list()` or `vstash_search()` first.

---

## Key Context About Jay

- Works in QA at Pluxee, experiments with dev tools personally
- Hardware: Apple M4 Pro 24GB
- GitHub: stffns | PyPI: vstash published at pypi.org/project/vstash
- Cerebras API key stored in ~/.vstash/vstash.toml
- vstash project directory: ~/Desktop/Personal/Projects/vex/
- Memory folder: ~/Desktop/Personal/Projects/vstash-memory/
- vstash MCP server configured in Claude Desktop

## Current Version: 0.10.0 (March 2026)

### v0.10.0 — Hybrid Code Splitting
- **3-tier code splitting**: tree-sitter AST → parso AST → regex fallback
- **25+ languages** via optional `tree-sitter-language-pack`
- **parso** added as base dependency for Python AST splitting
- **UTF-8 safe** byte-offset handling in tree-sitter backend
- New `vstash/code_split.py` module — 356 tests passing

### v0.9.0 — Auto-Generated Titles & SDK Fix
- **Auto-generated descriptive titles for `vstash remember`:** When no `title`
  is provided, vstash generates a slug from the first 5 words + UTC timestamp
  with microsecond precision (e.g. `oauth2-uses-pkce-20260330-143052474102`).
- **`Memory.remove()` fix:** `text://` paths are no longer mangled by
  `Path.resolve()`, so `forget` works correctly from the Python SDK.
- **`ingest_text()` signature change:** `title` is now keyword-only,
  `cfg` and `store` are required positional params.
- **Test suite fix:** Added missing `tests/__init__.py` — 326/326 tests passing.

### v0.8.0–0.8.1 — Direct Text Ingestion, Code Chunking & MMR
- **`vstash remember` command:** Ingest text directly without writing files.
  Agent-friendly alternative to `vstash add`.
- **MMR deduplication:** Replaced hard per-document dedup with intra-document
  Maximal Marginal Relevance. Multiple chunks from the same document can appear
  if semantically diverse. Configurable via `mmr_lambda`.
- **Code-aware chunking:** Regex-based splitting at column-0 definitions
  for Python, JS/TS, Go, Rust, Java. 3-tier fallback: regex → paragraph → fixed-window.
- **Multilingual embeddings:** Search in any language via `vstash reindex`.
- **Python SDK (`from vstash import Memory`):** Programmatic access to
  add, search, ask, remember, remove, list, stats.
- **LangChain integration:** `VstashRetriever` for LangChain pipelines.
- **MCP server:** Full tool suite for Claude Desktop integration.

### v0.7.0 — Adaptive Scoring & Cold Start Fix
- **Adaptive scoring with maturity gate (γ):** suppresses frequency+decay signals
  until access patterns show genuine signal.
- **Zero-cost cold start:** when γ = 0, scoring overhead is eliminated entirely.

### v0.6.0 — Confidence & Deduplication
- **Distance-based confidence signals** for relevance filtering (F1=0.952)
- **Document deduplication** — diversity improved to 5.0 unique docs per top-5
- **Context expansion** with adjacent chunks (±1) for 2.64× richer LLM context
- **LLM grounding** with anti-hallucination system prompt rules

### Key Capabilities
- Hybrid ranking via Reciprocal Rank Fusion (semantic + keyword)
- FastEmbed for embeddings (~700 chunks/s, fully local ONNX)
- SQLite-vec vector store in single `.db` file
- Supports: PDF, DOCX, PPTX, XLSX, Markdown, HTML, CSV, code files, URLs
- Hybrid code-aware chunking: tree-sitter (25+ langs) → parso (Python) → regex (6 langs)
- Privacy: search is 100% local; only inference hits external APIs
- Ollama option for fully private inference

## Tools Available (MCP)

| Tool | Use for |
|------|---------|
| vstash_ask(query) | Query memory with LLM answer |
| vstash_search(query) | Raw retrieval without LLM (free, fast) |
| vstash_remember(text, title) | Ingest text content directly — always pass title |
| vstash_add(path, collection, project, layer, tags) | Ingest file or URL (Mac paths only) |
| vstash_forget(source) | Remove document (use `text://<title>` for remembered text) |
| vstash_list() | List all ingested documents |
| vstash_stats() | Check DB size and doc count |
| vstash_collections() | List all collections |
| vstash_export() | Export data as JSONL for curation |
| vstash_job(job_id) | Check status of background ingestion (directories) |

## CLI Commands (on Mac)

| Command | Use for |
|---------|---------|
| `vstash add` | Ingest documents |
| `vstash remember` | Ingest text directly (with optional `--title`) |
| `vstash search` | Local semantic search (free) |
| `vstash ask` | LLM-powered Q&A |
| `vstash chat` | Interactive sessions |
| `vstash list` / `vstash stats` / `vstash forget` | Management |
| `vstash watch` | File change monitoring |
| `vstash export` | JSONL data curation |
| `vstash config` | Display configuration |
