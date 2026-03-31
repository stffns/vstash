# Future Improvements

Consolidated from past reviews, plans, and brainstorms. Only items that remain relevant as of v0.10.0.

---

## Engineering — High Impact

### API Resilience (retry with backoff)

API calls to Cerebras/OpenAI/Ollama are single-attempt with generic `try/except`. A transient timeout or rate limit crashes the CLI.

**Fix:** Add `tenacity` with exponential backoff for remote inference calls in `chat.py`. Catch specific HTTP exceptions (429, 503, timeout) instead of bare `Exception`.

**Impact:** Prevents CLI crashes on flaky connections. Essential for MCP server reliability.

### Domain-Specific Exceptions

All errors surface as generic `ValueError`, `ImportError`, or `Exception`. Callers can't distinguish a database corruption from a missing API key.

**Fix:** Add a small exception hierarchy:
- `VstashDatabaseError` — SQLite / sqlite-vec failures
- `VstashInferenceError` — LLM backend timeouts, auth failures
- `VstashIngestError` — file parsing or encoding failures

**Impact:** Better error messages, easier debugging, SDK consumers can catch specific errors.

---

## Engineering — Nice to Have

### CLI Input Validation (Pydantic)

Config is validated through Pydantic, but CLI arguments flow directly to business logic without validation. Mapping CLI args to Pydantic models at the edge would guarantee type safety.

### Property-Based Testing (hypothesis)

Chunking logic handles varied inputs (empty strings, huge files, mixed encodings). Hypothesis would catch edge cases that manual tests miss.

---

## Scoring — v2

### Importance Scoring (weighted access types)

All accesses count equally. A casual list view weighs the same as a deep `ask` citation. Weighted access types (`search=1.0`, `ask_context=2.0`, `bookmark=5.0`) would improve ranking quality. Requires schema change: access log table instead of simple counter.

### Per-Collection Scoring Params

One α/β/λ for the entire store. Users with mixed collections (reference docs vs research notes) can't tune per use case. Allow overrides in `[scoring.collections.<name>]`.

### Decay Reset on Re-ingestion

When a document is re-ingested (`force=True`), new chunks lose all access history. Transfer `access_count` from the most similar old chunk (by embedding distance) to preserve memory across updates.

### Adaptive α/β via Feedback Loop

Auto-tune scoring weights based on implicit signals: if LLM cites memory-boosted chunks, increase β; if it prefers pure-semantic matches, increase α. Complex — depends on inference backend instrumentation.

---

## Ingestion — v2

### Real Titles for URL Documents

URLs are stored with the URL as title. Extract `<title>` from HTML or metadata from PDFs during ingestion for better readability and semantic matching.

---

## Performance — v3

### Native SQLite Scoring Extension

For corpus >1M chunks, move `rerank_with_decay()` to a C/Rust SQLite extension. Eliminates Python serialization overhead and enables `ORDER BY final_score` directly in the query.

---

## Application Ideas

Potential projects built on top of vstash:

- **Web UI** — Frontend talking directly to MCP server (FastAPI bridge ~50 lines)
- **Multi-Agent Shared Memory** — vstash as persistence layer for LangGraph/Agno agent swarms (WAL mode supports concurrency)
- **Research Assistant** — Papers + notes + transcripts with semantic search
- **Architecture Decision Records** — Teams store decisions, query months later

---

*Consolidated: March 2026 — Sources: REVIEW.md, NEXT_STEPS.md, brainstorm.md, frequency_decay_plan.md*
