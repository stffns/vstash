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
| Embedding generation | Local (FastEmbed ONNX) | Full document chunks |
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

model = "llama-3.3-70b"  # or any model supported by your backend
```

**Default:** Cerebras for the demo experience. Swap to Ollama for absolute privacy.
The codebase treats all three identically — one config line changes the backend.

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Vector store** | `sqlite-vec` | Single file, no server, fast enough for millions of vectors |
| **Keyword search** | `FTS5` (SQLite) | Exact matches, porter stemming, built into SQLite |
| **Hybrid ranking** | Reciprocal Rank Fusion | Best of both: semantic + keyword, no training needed |
| **Embeddings** | `FastEmbed` (Qdrant ONNX) | ONNX runtime, ~700 chunks/s on CPU, no server, `pip install` only |
| **Embedding model** | `BAAI/bge-small-en-v1.5` | 384 dims, fastest quality/speed ratio |
| **Inference (fast)** | Cerebras API | ~2,000 tokens/second — the fastest available API |
| **Inference (flexible)** | OpenAI API | Compatible with any OpenAI-compatible endpoint |
| **Inference (private)** | Ollama (any model) | 100% local fallback, no data leaves the machine |
| **Document parsing** | `markitdown` | Universal: PDF, DOCX, PPTX, HTML, code, URLs |
| **Configuration** | Pydantic v2 | Type-safe config with validation and defaults |
| **CLI** | `Typer` + `Rich` | Clean, beautiful terminal interface with progress feedback |
| **Language** | Python 3.10+ | Ecosystem, speed of development, accessibility |

### Speed is the philosophy — every layer chosen for it

**FastEmbed over Ollama for embeddings:**
Ollama is a server. FastEmbed is a library. ONNX Runtime vs PyTorch.
The result: ~700 chunks/second on CPU vs ~150 with Ollama. A 500-page PDF goes from 10 minutes to under 2.
No background process, no port conflicts, no setup friction. Just `pip install fastembed`.

```
Ollama:    HTTP request → server process → PyTorch → embedding   (~150 chunks/s CPU)
FastEmbed: in-process   → ONNX runtime  → embedding             (~700 chunks/s CPU)
```

**Cerebras as default inference:**
At 2,000 tok/s, the response latency disappears. The demo GIF speaks for itself.
Swap to Ollama in one config line for absolute privacy — same interface, different backend.

**sqlite-vec over pgvector/Chroma:**
No server. One file. Works on any machine without setup. Fast enough for millions of vectors.
Cosine search on 100k vectors: sub-millisecond.

**Hybrid RRF over pure vector search:**
FTS5 provides exact keyword matching (BM25). sqlite-vec provides semantic similarity.
Reciprocal Rank Fusion (k=60, vec_weight=0.6, fts_weight=0.4) combines both rankings,
ensuring that neither semantic drift nor keyword misses lose relevant results.

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
│ markitdown  │              │  FastEmbed (ONNX)    │
│ PDF/DOCX/   │  [LOCAL]     │  embed query → vec   │
│ URL/code    │              │  cosine similarity   │
└─────────────┘              │         +            │
                             │  FTS5 → BM25 rank   │
                             │         ↓            │
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

### Data flow — ingest

```
file/URL
  → markitdown → raw text
    → chunk (1024 tokens, 128 overlap)       ← larger chunks = more context for LLM
      → FastEmbed ONNX → vector (384 dim)    ← always local, ~700 chunks/s
        → store in sqlite-vec + FTS5
          → Rich progress bar shown throughout
```

### Data flow — query (Hybrid RRF)

```
user question
  → FastEmbed ONNX → query vector              ← local
    → sqlite-vec cosine search → top-k×10 candidates
    → FTS5 BM25 keyword search → top-k×10 candidates
      → Reciprocal Rank Fusion → combined ranking
        → top-k results (default: 5)
          → build prompt (context + question + history)
            → inference backend → response   ← Cerebras / OpenAI / Ollama
              → display with sources cited
```

### Chunking strategy

- **Default chunk size:** 1,024 tokens with 128 token overlap
- Larger chunks give the LLM more context per retrieved segment
- Overlap prevents losing meaning at chunk boundaries
- Chunks under 20 characters are filtered out

---

## Project Structure

```
vstash/
├── vstash/
│   ├── __init__.py       # Package metadata
│   ├── cli.py            # Typer CLI entry point
│   ├── config.py         # Pydantic v2 config loader (vstash.toml)
│   ├── models.py         # Typed result models (IngestResult, SearchResult, etc.)
│   ├── ingest.py         # Document ingestion pipeline (with Rich progress)
│   ├── chat.py           # Inference backend abstraction (Cerebras/Ollama/OpenAI)
│   ├── embed.py          # FastEmbed ONNX wrapper (always local)
│   └── store.py          # sqlite-vec + FTS5 hybrid store with RRF
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_config.py    # Config validation tests
│   ├── test_ingest.py    # Chunking + source detection tests
│   ├── test_store.py     # Store CRUD + search tests
│   ├── test_embed.py     # Embedding dimension tests
│   └── test_chat.py      # Prompt building + dispatch tests
├── vstash.toml.example      # Config template
├── pyproject.toml
├── README.md
└── VSTASH_CONSTITUTION.md   # This file
```

---

## CLI Interface

```bash
# Add documents to memory
vstash add report.pdf
vstash add https://arxiv.org/abs/2310.06825
vstash add ./src/                          # entire directory
vstash add paper.pdf --force               # re-ingest even if already stored

# Ask questions
vstash ask "What were the main conclusions of the report?"
vstash ask "Which files handle authentication?" --top-k 10

# Interactive chat mode (keeps conversation context)
vstash chat

# Inspect memory
vstash list                 # show all ingested documents
vstash stats                # memory size, chunk count, inference backend
vstash forget report.pdf    # remove from memory

# Configuration
vstash config               # show current settings
```

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
- [x] pytest test suite (72 tests)
- [x] README with honest privacy table

**Definition of done:** drop a PDF, ask a question, get an answer in < 1 second (Cerebras) or < 10 seconds (Ollama local).

### Phase 2 — Usability
Make it something people actually use daily.

- [ ] Demo GIF for README
- [ ] Semantic chunking (split at paragraph/section boundaries)
- [ ] `vstash watch ./folder` — auto-ingest on file changes
- [ ] `vstash export` / `vstash import` — database portability
- [ ] Rich source citations with file + page references

### Phase 3 — Sync without a server
Share memory across machines without compromising the no-server principle.

- [ ] `cr-sqlite` integration — CRDT-based SQLite sync (peer-to-peer)
- [ ] `vstash sync` — merge two `.db` files intelligently
- [ ] Multiple memory profiles: `vstash --profile work ask "..."`
- [ ] Export memory as structured JSON / Markdown

### Phase 4 — Agent integration
Vex as memory layer for other tools and agents.

- [ ] Python SDK — `from vstash import Memory` for external programs
- [ ] REST API mode (opt-in, local only) for non-Python integrations
- [ ] Web UI (optional, lightweight, localhost only)

---

## What makes this GitHub-worthy

- **Speed at every layer** — FastEmbed (ONNX) + sqlite-vec + Cerebras. No compromises.
- **Hybrid search** — RRF combines vector and keyword search for best results.
- **Honest privacy model** — engineers respect projects that don't hide tradeoffs
- **FastEmbed** closes the ingestion gap — 700 chunks/s on CPU, no server
- **sqlite-vec** is new and underused — devs will want to see it in production
- **Backend agnostic** — Cerebras for speed, OpenAI for flexibility, Ollama for privacy
- **Dead simple** — `pip install vstash`, one config file, running in 5 minutes
- **Typed and tested** — Pydantic v2 models + 72 pytest tests

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
