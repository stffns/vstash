## Project: vstash SDK (Phase 4)
Date: 2026-03-23

## Decision
Design and implement `from vstash import Memory` — a high-level Python SDK wrapping the existing vstash core.

## Architecture
- New file: `vstash/memory.py` — `Memory` class
- Modified: `vstash/__init__.py` — exports `Memory` + public models
- Zero breaking changes to CLI or MCP server

## Public API
```python
mem = Memory(project="my_agent")
mem.add("doc.pdf")
chunks = mem.search("query", top_k=5)   # → list[SearchResult], no LLM
answer = mem.ask("question")             # → str, with LLM
mem.remove("doc.pdf")
mem.list()   # → list[DocumentInfo]
mem.stats()  # → StoreStats
```

## Key Design Decisions
- Sync-first (async deferred to v0.3.2 — no real need yet)
- Memory is NOT a singleton — WAL mode supports multiple instances on same .db
- Filters (project, collection, layer) are optional — not forced
- `add()` returns IngestResult instead of None
- Only Memory + model types exported publicly

## Roadmap
- v0.3.0: Memory sync + 6 new tests
- v0.3.1: mem.stream() token generator
- v0.3.2: AsyncMemory for event-loop agents
- v0.4.0: LangGraph demo using vstash as agent memory

## Next Steps
Implement vstash/memory.py following SDK_PLAN.md in the vex project.
