# vstash roadmap

vstash targets the sweet spot of local hybrid retrieval: **1K to 50K chunks,
sub-second latency, a single SQLite file, no services**. This roadmap
orients feature work around that niche and the path to it becoming
the obvious default for local retrieval.

The guiding parallel is Richard Hipp's framing of SQLite:

> SQLite does not compete with client/server databases.
> SQLite competes with fopen().

The vstash equivalent: **vstash competes with `pickle.dump(embeddings)`**,
not with Pinecone or Weaviate.

## Phase 0 - Foundation

Nothing below scales without these being excellent.

- [x] **Paper** on arXiv with a recognized endorsement from the retrieval research community.
- [ ] **Docs site** on its own domain (MkDocs Material). Sections: Quick Start, Guides, API Reference, Benchmarks, FAQ.
- [ ] **API 1.0 freeze**: explicit stability surface (`Memory` SDK, CLI, MCP). Strict SemVer beyond that point.
- [ ] **README pass**: Hipp-style pitch line, a 15-second install-to-first-query demo, mobile-readable in under 10 seconds.

## Phase 1 - Distribution

The product exists, now it needs to reach people.

- [x] **Anchor integrations**: MCP server, LangChain retriever, Claude Desktop.
- [ ] **LlamaIndex and Haystack retrievers** (covers the three dominant Python agent frameworks).
- [ ] **Five starter repos**, each under ~200 lines:
  - `vstash-chatbot` (local chat with a corpus)
  - `vstash-code-search` (semantic grep over a repo)
  - `vstash-research-assistant` (paper summarization + citations)
  - `vstash-mcp-memory-server` (MCP endpoint for Claude Desktop)
  - `vstash-agent-long-memory` (persistent memory for autonomous agents via merken)
- [ ] **Technical blog posts** (one per month, no hype):
  - Why hybrid search beats dense-only
  - IVFPQ + fp16 rerank for 100K local retrieval
  - Fine-tuning your embedding model from your own disagreement signal
  - Integer-math MMR for selection under quantization (follow-up paper material)
- [ ] **Conference or meetup talk** (lightning talk scope, PyData / PyCon / local Python meetup).

## Phase 2 - Ecosystem

When traffic exists, open the surface for others to build.

- [ ] **Minimal plugin system**: Python protocols for custom embedders, chunkers, and re-rankers. No plugin loader machinery, just typed interfaces.
- [ ] **JS/TS client** (via `vstash serve` HTTP or a Tauri/Neon binding). Covers the ~50% of agent frameworks that live in JS (Langchain.js, Mastra, Vercel AI SDK).
- [ ] **Published case studies**: three to five real deployments with permission to cite.
- [ ] **CI regression benchmarks**: BEIR NDCG + pipeline latency run per PR; block merges that regress >1% NDCG or >10% p50 latency.

## Phase 3 - Standard

When these become true, vstash is the niche default:

- When someone asks "how do I add local memory to my agent?" on Reddit, Discord, or HN, the first unsolicited answer is "use vstash."
- When books and courses on AI agents mention it as a default (O'Reilly, DeepLearning.AI).
- When companies adopt it without opening a support issue: they find it, install it, ship.
- When it appears in "local AI stack" slides alongside Ollama, llama.cpp, MCP.

## Optimization backlog

Concrete engineering items that compound into the narrative above, ranked by ROI
for the 1K-50K niche (not the 100K+ ceiling).

### Tier 1 - high impact, 1-3 days each

1. **Persistent embedder daemon**. First-query cold start drops from ~2s to <50ms by keeping ONNX warm in `vstash serve --warm`. Largest felt UX improvement.
2. **Query LRU cache**. Agents in loops repeat queries; cached hits drop from ~20ms to <1ms. Small code surface.
3. **Batch ingest + deferred FTS indexing**. Current ingest is ~4 ms/chunk; batching commits and deferring FTS5 rebuild can reach ~0.5-1 ms/chunk. Makes 10K docs ingest in under 30s feel "instant."

### Tier 2 - medium impact, ~1 week each

4. **fp16 embeddings in `vec_chunks`**. sqlite-vec supports it natively. Halves vec table disk with no recall loss.
5. **Denormalize `chunks.title` and `chunks.path`**. Eliminates two joins in the fetch phase; saves ~10-15 ms per search at 50K. Destructive migration, one-time.
6. **Async search API**. `async def Memory.search()` so concurrent agents don't fight the GIL on a single interpreter.

### Tier 3 - real differentiators, 2-4 weeks each

7. **Context-aware code chunking**. Tree-sitter is in today but the chunker only uses syntactic units (functions, classes). A pass over imports and references gives semantically complete chunks.
8. **Incremental re-embedding**. Swapping models today forces a full reindex. A per-document content hash enables delta re-embed, making it cheap to evaluate new embedding models on a production corpus.
9. **Per-stage observability**. p50/p95 metrics broken out by vector, FTS, RRF, MMR, fetch. Advanced users need this to tune; today only `slow query` exists.

### Explicitly out of scope

- Rewriting the hot path in Rust or Cython. Python is not the bottleneck yet.
- Additional storage backends beyond `sqlite-vec` and `snapvec-ivfpq`. Two covers the niche with a ceiling.
- Multi-user server features, replication, sharding, distributed transactions. These are Pinecone's problem, not vstash's.
- GPU dependencies in the core. Embeddings can use MLX or CUDA at the embedder level, but the store stays CPU-only.

## Non-goals

vstash intentionally does not pursue:

- Competing with managed vector databases on scale axis alone.
- Serving as a general-purpose analytics or feature store.
- Multi-tenant cloud deployment primitives.
- Becoming a retrieval service instead of a library.

A clear "no" is as important as a clear "yes" for staying in the niche.
