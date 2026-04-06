# Future Improvements

Consolidated from past reviews, plans, and brainstorms. Updated as of v0.19.0.

---

## Engineering — High Impact

### ~~API Resilience (retry with backoff)~~ ✅ Done (v0.11.0)

Implemented in `chat.py` via `_retry_call()` with exponential backoff and transient error detection (429, 503, timeout). Covered by `test_retry_e2e.py`.

### Domain-Specific Exceptions

All errors surface as generic `ValueError`, `ImportError`, or `Exception`. Callers can't distinguish a database corruption from a missing API key.

**Fix:** Add a small exception hierarchy:
- `VstashDatabaseError` — SQLite / sqlite-vec failures
- `VstashInferenceError` — LLM backend timeouts, auth failures
- `VstashIngestError` — file parsing or encoding failures

**Impact:** Better error messages, easier debugging, SDK consumers can catch specific errors.

### Direct Chunk Access API (`get_chunk(id)`)

The SDK and MCP server have no way to retrieve a single chunk by ID. Recall needs this to fetch card text at runtime via `vstash_chunk_id` without a full search query.

**Fix:** Expose `get_chunk(chunk_id) → ChunkResult` in `VstashStore`, `Memory` SDK, and MCP server.

**Impact:** Enables Recall (and any app that stores `chunk_id` references) to do O(1) lookups instead of search-based workarounds.

---

## Engineering — Nice to Have

### CLI Input Validation (Pydantic)

Config is validated through Pydantic, but CLI arguments flow directly to business logic without validation. Mapping CLI args to Pydantic models at the edge would guarantee type safety.

### Property-Based Testing (hypothesis)

Chunking logic handles varied inputs (empty strings, huge files, mixed encodings). Hypothesis would catch edge cases that manual tests miss.

---

## Recency & Temporal — v2

### Per-Collection Recency Defaults

One global `recency_boost` for the entire store. Users with mixed collections (reference docs vs active project notes) might want different defaults. Allow overrides in `[recency.collections.<name>]`.

### Recency by Last Modified (not just created)

Current recency boost uses `created_at` (ingestion time). For watched directories where files are updated in place, boost based on file modification time would be more accurate. Requires tracking `modified_at` on documents.

### Query-Intent Detection

Auto-detect whether a query is retrieval ("how does OAuth2 work?") vs temporal ("what was I working on?") and apply recency boost only when temporal intent is detected. Could use simple heuristics (temporal keywords) or a lightweight classifier.

---

## Ingestion — v2

### ~~Real Titles for URL Documents~~ ✅ Done (v0.11.0)

Implemented in `ingest.py` via `_extract_title_from_content()`. Extracts real titles from HTML content during ingestion. Covered by `test_url_titles_e2e.py`.

---

## Performance — v3

### Native SQLite Scoring Extension

For corpus >1M chunks, move hot-path scoring to a C/Rust SQLite extension. Eliminates Python serialization overhead and enables `ORDER BY final_score` directly in the query.

---

## Application Ideas

Potential projects built on top of vstash:

- **Recall** — Adaptive learning platform using vstash as knowledge store + FSRS scheduling + LLM Judge. In progress.
- **Web UI** — Frontend talking directly to MCP server (FastAPI bridge ~50 lines)
- **Multi-Agent Shared Memory** — vstash as persistence layer for LangGraph/Agno agent swarms (WAL mode supports concurrency)
- **Research Assistant** — Papers + notes + transcripts with semantic search
- **Architecture Decision Records** — Teams store decisions, query months later

---

*Consolidated: March 2026 — Updated: April 2026 — Sources: REVIEW.md, NEXT_STEPS.md, brainstorm.md, frequency_decay_plan.md*
