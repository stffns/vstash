# vstash — Project Constitution
> *An engram is the biological unit of memory in the human brain.*
> *This project is its digital equivalent — local, instant, yours.*

---

## Vision

**vstash** is a local-first document memory system with instant semantic search.

Drop any document. Ask anything. Get an answer in under a second.

Your database lives on your machine. Your embeddings are generated locally.
The only data that leaves is the context sent to your chosen inference backend — and that choice is yours.

---

## The Problem

Today's tools for working with documents are broken in one of two ways:

- **Cloud-dependent** — your documents, your knowledge, goes to someone else's server
- **Slow** — RAG pipelines with LLMs take 5–15 seconds per query, which kills the flow

Nobody has combined *truly local* storage + *instant* semantic search + *configurable* inference in a single, simple, open-source tool.

**vstash does exactly that.**

---

## Core Principles

1. **Local first** — storage and embeddings never leave your machine. Inference is configurable.
2. **Single file** — the entire memory is one `.db` file you can copy, backup, or delete
3. **No server required** — no Docker, no Postgres, no cloud accounts to set up
4. **Speed as a feature** — sub-second responses are not a nice-to-have, they are the product
5. **Honest about tradeoffs** — we document every privacy boundary, not hide it
6. **Simple over complete** — a tool you actually use beats a framework you admire

---

## Privacy Model — Be Explicit

vstash is transparent about what stays local and what doesn't.

| Operation | Where it runs | Data involved |
|-----------|--------------|---------------|
| Document parsing | Local | Full document |
| Embedding generation | Local (MLX GPU or ONNX CPU) | Full document chunks |
| Vector storage | Local (sqlite-vec) | Embeddings + text chunks |
| Keyword index | Local (FTS5) | Text chunks |
| Semantic search | Local | Query vector only |
| **Inference (Cerebras)** | **Remote API** | **Top-k relevant chunks** |
| **Inference (OpenAI)** | **Remote API** | **Top-k relevant chunks** |
| **Inference (Ollama)** | **Local** | **Top-k relevant chunks** |

**Bottom line:** with Cerebras or OpenAI, the relevant text chunks travel to their API for generation.
With a local model via Ollama, nothing leaves your machine at any step.

### Inference backends (user's choice)

```toml
# vstash.toml
[inference]
backend = "cerebras"    # fastest — chunks sent to Cerebras API
# backend = "ollama"   # fully local — nothing leaves your machine
# backend = "openai"   # OpenAI API or any compatible endpoint

model = "gpt-oss-120b"  # or any model supported by your backend

[embeddings]
model   = "BAAI/bge-small-en-v1.5"
backend = "auto"   # "onnx" | "mlx" | "auto" (detects Apple Silicon)
```

**Default:** Cerebras for the demo experience. Swap to Ollama for absolute privacy.
The codebase treats all three identically — one config line changes the backend.

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Vector store** | `sqlite-vec` | Single file, no server, fast enough for 100K+ vectors |
| **Keyword search** | `FTS5` (SQLite) | Exact matches, porter stemming, built into SQLite |
| **Hybrid ranking** | RRF + distance cutoff | Semantic + keyword with noise filtering |
| **Embeddings (Apple Silicon)** | `mlx-embeddings` (MLX) | Apple GPU, ~1,200 chunks/s, 2.5ms per query |
| **Embeddings (portable)** | `FastEmbed` (ONNX) | CPU runtime, ~950 chunks/s, 5ms per query |
| **Embedding model** | `BAAI/bge-small-en-v1.5` | 384 dims, fastest quality/speed ratio |
| **Inference (fast)** | Cerebras API | ~2,000 tokens/second — the fastest available API |
| **Inference (flexible)** | OpenAI API | Compatible with any OpenAI-compatible endpoint |
| **Inference (private)** | Ollama (any model) | 100% local fallback, no data leaves the machine |
| **Document parsing** | `markitdown` | Universal: PDF, DOCX, PPTX, HTML, code, URLs |
| **Chunking** | Semantic-first pipeline | Headers → paragraphs → fixed-window → merge small |
| **Python SDK** | `Memory` class | 6-method API for agents and pipelines |
| **MCP Server** | `vstash-mcp` | Claude Desktop integration via Model Context Protocol |
| **Configuration** | Pydantic v2 | Type-safe config with validation and defaults |
| **CLI** | `Typer` + `Rich` | Clean, beautiful terminal interface with progress feedback |
| **Language** | Python 3.10+ | Ecosystem, speed of development, accessibility |

### Speed is the philosophy — every layer chosen for it

**Dual embedding backends — auto-detected:**
On Apple Silicon, MLX uses the GPU directly for ~1,200 chunks/s and 2.5ms queries.
On other platforms, FastEmbed uses ONNX Runtime for ~950 chunks/s.
Auto mode detects Apple Silicon and picks the fastest backend.

```
Ollama:    HTTP request → server → PyTorch → embedding    (~150 chunks/s)
FastEmbed: in-process   → ONNX runtime  → embedding       (~950 chunks/s CPU)
MLX:       in-process   → Apple GPU     → embedding       (~1,200 chunks/s GPU)
```

**Cold start elimination:**
The CLI pre-loads the embedding model (ONNX or MLX) during initialization via `warmup()`,
ensuring the first query is as fast as subsequent ones (~3-8ms vs ~450ms without warmup).

**Cerebras as default inference:**
At 2,000 tok/s, the response latency disappears. The demo GIF speaks for itself.
Swap to Ollama in one config line for absolute privacy — same interface, different backend.

**sqlite-vec over pgvector/Chroma:**
No server. One file. Works on any machine without setup.
Cosine search on 100K vectors: ~9ms. On 500K vectors: ~48ms.

**Hybrid RRF with distance cutoff:**
FTS5 provides exact keyword matching (BM25). sqlite-vec provides semantic similarity.
Reciprocal Rank Fusion (k=60, vec_weight=0.6, fts_weight=0.4) combines both rankings.
A vector distance cutoff (1.15× best distance) filters out noise before RRF scoring,
eliminating false positives that plagued early versions.
Adaptive candidate pool sizing adjusts to corpus size to prevent noise injection.

**URL ingestion with User-Agent:**
URLs are pre-downloaded with a proper User-Agent header before parsing,
avoiding 403 errors from sites like Wikipedia.

**markitdown over custom parsers:**
Universal document parsing — one library handles everything. Don't reinvent this.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     vstash.db                          │
│                   (sqlite-vec)                      │
│                                                     │
│   documents    →  chunks    →  embeddings (vectors) │
│   (metadata)      (text)       (384-dim float32)    │
│                      ↓                              │
│               fts_chunks (FTS5)                     │
│               (keyword index)                       │
└─────────────────────────────────────────────────────┘
         ↑                              ↓
    [ingest]                       [search]
         ↑                              ↓
┌─────────────┐              ┌──────────────────────┐
│ markitdown  │              │  MLX (Apple GPU)     │
│ PDF/DOCX/   │  [LOCAL]     │    or                │
│ URL/code    │              │  FastEmbed (ONNX)    │
└─────────────┘              │  embed query → vec   │
                             │  cosine similarity   │
                             │         +            │
                             │  FTS5 → BM25 rank   │
                             │         ↓            │
                             │  distance cutoff     │
                             │  RRF merge rankings  │
                             └──────────────────────┘
                                        ↓
                             ┌──────────────────────────────┐
                             │   Inference backend           │
                             │                               │
                             │   Cerebras  → chunks + query │  ← fast, remote
                             │   OpenAI    → chunks + query │  ← flexible, remote
                             │   Ollama    → chunks + query │  ← private, local
                             └──────────────────────────────┘
```

### Data flow — ingest (semantic chunking)

```
file/URL
  → markitdown → raw text
    → _split_by_headers (Markdown sections)       ← preserve document structure
      → _split_by_paragraphs (paragraph breaks)   ← respect natural boundaries
        → _fixed_window (oversized paragraphs)     ← token-bounded fallback with overlap
          → _merge_small (< 80 tokens)             ← bidirectional merge, avoid low-quality embeddings
            → MLX/ONNX → vector (384 dim)          ← always local, ~1,200 chunks/s (MLX)
              → store in sqlite-vec + FTS5
                → Rich progress bar shown throughout
```

### Data flow — query (Hybrid RRF)

```
user question
  → warmup (if first call)                     ← pre-load model, JIT compile
  → MLX/ONNX → query vector                    ← local, ~3-8ms
    → sqlite-vec cosine search → adaptive candidate pool
    → FTS5 BM25 keyword search → adaptive candidate pool
      → distance cutoff (1.15× best distance)  ← filter noise
      → FTS gating (vector-relevant only)
      → Reciprocal Rank Fusion → combined ranking
        → top-k results (default: 5)
          → build prompt (context + question + history)
            → inference backend → response     ← Cerebras / OpenAI / Ollama
              → display with sources cited
```

### Chunking strategy — semantic-first

The chunking pipeline prioritizes document structure over arbitrary token windows:

1. **Headers** — split at Markdown `#` boundaries; each section becomes a candidate chunk
2. **Paragraphs** — within each section, split at `\n\n` paragraph breaks
3. **Fixed-window** — oversized paragraphs are split at token boundaries with configurable overlap
4. **Merge small** — chunks under 80 tokens are merged bidirectionally with their neighbors
5. **Token counting** — includes the `\n\n` separator cost between chunks to avoid exceeding limits

Default chunk size: 1,024 tokens with 128 token overlap (fixed-window fallback only).
Overlap is clamped to `chunk_size - 1` to prevent infinite loops.

---

## Scalability

Benchmarked on Apple Silicon (M-series) with hybrid search (vector + FTS5 + RRF merge):

| Chunks | ≈ Documents | DB Vectors | Hybrid Search |
|--------|------------|-----------|---------------|
| 1,000 | ~50 docs | 1.5 MB | **0.6ms** |
| 10,000 | ~500 docs | 15 MB | **5.7ms** |
| 50,000 | ~2,500 docs | 73 MB | **24ms** |
| 100,000 | ~5,000 docs | 147 MB | **52ms** |
| 500,000 | ~25,000 docs | 732 MB | **286ms** |

FTS5 is the bottleneck at scale (~5× slower than vector search alone). Up to **100K chunks
(~5,000 documents) the hybrid search stays under 52ms** — imperceptible against the ~1s LLM latency.

---

## Project Structure

```
vstash/
├── vstash/
│   ├── __init__.py       # Package metadata + SDK exports (Memory, models)
│   ├── memory.py         # Python SDK — Memory class with 6-method public API
│   ├── cli.py            # Typer CLI entry point (with warmup)
│   ├── mcp.py            # MCP server for Claude Desktop integration
│   ├── config.py         # Pydantic v2 config loader (vstash.toml)
│   ├── models.py         # Typed result models (IngestResult, SearchResult, DocumentInfo, StoreStats)
│   ├── ingest.py         # Semantic chunking pipeline (headers → paragraphs → merge)
│   ├── chat.py           # Inference backend abstraction (Cerebras/Ollama/OpenAI)
│   ├── embed.py          # Dual MLX/ONNX embedding backend with auto-detection
│   ├── store.py          # sqlite-vec + FTS5 hybrid store with RRF + distance cutoff
│   └── watch.py          # File watcher for auto-ingestion
├── benchmark/
│   ├── benchmark.py          # Semantic search vs grep comparison
│   ├── benchmark_chunking.py # Semantic vs fixed-window A/B comparison
│   ├── benchmark_sdk.py      # Memory SDK overhead measurement
│   └── e2e_test.py           # End-to-end retrieval + LLM benchmark
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py        # Config validation tests
│   ├── test_ingest.py        # Semantic chunking + source detection tests
│   ├── test_store.py         # Store CRUD + search tests
│   ├── test_embed.py         # Embedding dimension tests
│   ├── test_chat.py          # Prompt building + dispatch tests
│   ├── test_memory.py        # Memory SDK tests (init, add, search, remove, scoping)
│   ├── test_export.py        # Export pipeline tests
│   └── test_frontmatter.py   # Frontmatter parsing + metadata filtering tests
├── vstash.toml.example       # Config template
├── pyproject.toml
├── README.md
├── SDK_PLAN.md               # SDK design document
└── VSTASH_CONSTITUTION.md    # This file
```

---

## CLI Interface

```bash
# Add documents to memory
vstash add report.pdf
vstash add https://arxiv.org/abs/2310.06825
vstash add ./src/                          # entire directory
vstash add paper.pdf --force               # re-ingest even if already stored
vstash add notes.md --collection research --project ml-survey --tags "attention"

# Search (free, no API key — 100% local)
vstash search "what is the proposed method?"
vstash search "auth flow" --project backend

# Ask questions (requires inference backend)
vstash ask "What were the main conclusions of the report?"
vstash ask "Which files handle authentication?" --top-k 10

# Interactive chat mode (keeps conversation context)
vstash chat

# Inspect memory
vstash list                 # show all ingested documents
vstash list --project work  # filter by project
vstash stats                # memory size, chunk count, inference backend
vstash forget report.pdf    # remove from memory

# Auto-ingestion
vstash watch ./folder       # auto-ingest on file changes

# Export
vstash export               # export chunks as JSONL for training data curation
vstash export --project ml-survey --format jsonl

# Configuration
vstash config               # show current settings

# MCP Server
vstash-mcp                  # start MCP server for Claude Desktop
```

---

## Python SDK — `from vstash import Memory`

*Added in v0.3.0.* A minimal, sync-first API for embedding vstash into agents and pipelines.

```python
from vstash import Memory

# Scoped to a project — separate namespace
mem = Memory(project="my_agent")

# Ingest
mem.add("docs/spec.pdf")
mem.add("https://example.com/api-docs")

# Search (free, no LLM)
results = mem.search("deployment strategy", top_k=5)
for r in results:
    print(r.text, r.score)

# Ask (requires inference backend)
answer = mem.ask("What are the system requirements?")

# Management
docs = mem.list()          # → list[DocumentInfo]
info = mem.stats()         # → StoreStats
mem.remove("docs/old.pdf")
mem.close()
```

### Design decisions

- **Sync-first** — no async until a real need arises (YAGNI)
- **Not a singleton** — multiple `Memory()` instances can coexist (WAL mode)
- **Optional filters** — `project`, `collection`, `layer` can be set at init or per-call
- **Zero new dependencies** — wraps existing store, ingest, embed, chat modules
- **Context manager** — `with Memory() as mem:` for automatic cleanup

---

## Roadmap

### Phase 1 — MVP ✅ Done
The core loop working end-to-end, honest about privacy.

- [x] `sqlite-vec` store with CRUD
- [x] FastEmbed embedding integration (ONNX, in-process) — always local
- [x] FTS5 keyword index + Reciprocal Rank Fusion (hybrid search)
- [x] markitdown ingestion: PDF, DOCX, PPTX, plain text, code, URLs
- [x] Cerebras inference integration
- [x] Ollama inference integration (same interface, different backend)
- [x] OpenAI inference integration (or any compatible endpoint)
- [x] `vstash.toml` config with Pydantic v2 validation
- [x] CLI: `add`, `ask`, `chat`, `list`, `stats`, `forget`, `config`
- [x] `--force` flag for re-ingestion
- [x] Conversation memory in `vstash chat` mode
- [x] Rich progress bars on ingest
- [x] Context manager for safe resource cleanup
- [x] pytest test suite
- [x] README with honest privacy table

**Definition of done:** drop a PDF, ask a question, get an answer in < 1 second (Cerebras) or < 10 seconds (Ollama local).

### Phase 1.5 — Performance & Precision ✅ Done
Eliminate noise, maximize speed, prove it with benchmarks.

- [x] RRF distance cutoff (1.15× best distance) — eliminates false positives
- [x] Adaptive candidate pool sizing — adjusts to corpus size
- [x] FTS gating — keyword results filtered by vector relevance
- [x] MLX embedding backend — Apple Silicon GPU, ~1,200 chunks/s
- [x] Auto-detection — picks MLX on Apple Silicon, ONNX elsewhere
- [x] ONNX model warm-up — eliminates cold start on first query
- [x] MLX multi-pass warm-up — JIT kernel pre-compilation
- [x] URL ingestion User-Agent fix — resolves 403 errors (Wikipedia, etc.)
- [x] End-to-end benchmark suite with timing breakdown
- [x] Semantic search vs grep comparison benchmark

### Phase 2 — Usability ✅ Done
Make it something people actually use daily.

- [x] Semantic chunking (split by headers → paragraphs → merge small)
- [x] `vstash search` — local semantic search without LLM (free)
- [x] `vstash watch ./folder` — auto-ingest on file changes
- [x] `vstash export` — export chunks as JSONL for training data curation
- [x] Collections and namespaces (`--collection`, `--project`)
- [x] Hierarchical frontmatter metadata (project, layer, tags)
- [x] Filtered retrieval across all commands
- [x] MCP server for Claude Desktop integration (`vstash-mcp`)
- [x] Benchmark: semantic vs fixed-window chunking A/B comparison

### Phase 3 — Python SDK ✅ Done (v0.3.0)
vstash as a building block for agents and pipelines.

- [x] `from vstash import Memory` — 6-method public API
- [x] Project/collection scoping at init or per-call
- [x] Context manager support (`with Memory() as mem:`)
- [x] WAL mode for concurrent Memory instances
- [x] Typed return models: `DocumentInfo`, `IngestResult`, `SearchResult`, `StoreStats`
- [x] SDK design document (`SDK_PLAN.md`)
- [x] SDK benchmark suite
- [x] 147 pytest tests across 9 test modules

### Phase 4 — Profiles & Agent Memory ✅ Done (v0.11.0–v0.13.0)
Multi-profile support, cross-session journal, and direct chunk access.

- [x] Multiple memory profiles: `vstash --profile work ask "..."` (v0.11.0)
- [x] Federated search across profiles (v0.11.0)
- [x] Cross-session journal: save/recall/log/prune (v0.12.0)
- [x] Transcript parsing for Claude Code sessions (v0.12.0)
- [x] Direct chunk access: `get_chunk(id)` / `get_chunks(ids)` (v0.13.0)
- [x] `ChunkInfo` model for typed chunk lookups (v0.13.0)
- [x] 556 pytest tests across 14 test modules

### Phase 5 — Sync & Integrations
Share memory across machines and expose to non-Python tools.

- [ ] `cr-sqlite` integration — CRDT-based SQLite sync (peer-to-peer)
- [ ] `vstash sync` — merge two `.db` files intelligently
- [ ] REST API mode (opt-in, local only) for non-Python integrations
- [ ] Web UI (optional, lightweight, localhost only)

---

## What makes this GitHub-worthy

- **Speed at every layer** — MLX GPU or ONNX + sqlite-vec + Cerebras. No compromises.
- **Dual embedding backends** — MLX for Apple Silicon (1,200 chunks/s), ONNX for portability
- **Hybrid search with noise filtering** — RRF + distance cutoff for precision
- **Semantic chunking** — preserves document structure instead of arbitrary token windows
- **Python SDK** — `from vstash import Memory` for agents and pipelines
- **MCP server** — Claude Desktop integration out of the box
- **Honest privacy model** — engineers respect projects that don't hide tradeoffs
- **sqlite-vec** is new and underused — devs will want to see it in production
- **Backend agnostic** — Cerebras for speed, OpenAI for flexibility, Ollama for privacy
- **Dead simple** — `pip install vstash`, one config file, running in 5 minutes
- **Typed and tested** — Pydantic v2 models + 556 pytest tests across 14 modules
- **Benchmarked** — E2E timing, grep comparison, chunking A/B, SDK overhead, scalability to 500K chunks

---

## Non-goals

- Not a RAG framework (too generic, too abstract)
- Not a chatbot (not the point)
- Not a cloud product (explicitly rejected)
- Not multi-user (single user, local machine — team use is a consequence of sync, not a feature)
- Not a replacement for a real vector database at scale
- Not opinionated about your LLM provider

---

## Contributing

This project values:
- **Simplicity** over features — every addition must justify its complexity
- **Speed** — never regress on latency
- **Honesty** — document every privacy boundary, never market past them
- **Good defaults** — works out of the box without configuration
- **Type safety** — Pydantic v2 models, strict type hints, pytest coverage

---

*"The best tool is the one you use."*
