# vstash

**Local document memory with instant semantic search.**

![vstash demo](demo.gif)

Drop any file. Ask anything. Get an answer fast.

```
pip install vstash
vstash add paper.pdf notes.md https://example.com/article
vstash search "what's the main argument about X?"
```

---

## Why vstash?

Most RAG tools are slow, cloud-dependent, or require a running server. vstash is none of those things.

| Layer | Technology | Why |
|---|---|---|
| Embeddings | FastEmbed (ONNX Runtime) | ~700 chunks/s, fully local, no server |
| Vector store | sqlite-vec | Single `.db` file, cosine similarity, zero deps |
| Keyword search | FTS5 (SQLite) | Exact matches, porter stemming, built into SQLite |
| Hybrid ranking | Reciprocal Rank Fusion | Best of both: semantic + keyword, no training needed |
| Inference | Cerebras / Ollama / OpenAI | ~2,000 tok/s via Cerebras, or 100% local via Ollama |
| Parsing | markitdown | PDF, DOCX, PPTX, XLSX, HTML, Markdown, URLs |

**Zero cloud required for search. Inference is optional.**

### What's new in v0.13

- **Direct chunk retrieval** — `get_chunk(id)` and `get_chunks(ids)` for O(1) access to specific chunks by database ID. Enables downstream apps (spaced repetition, pinned references) to retrieve knowledge atoms without re-running search.
- **579 tests** across 14 test modules, all passing on Python 3.10–3.12.

### What's new in v0.12

- **Cross-session journal** — `vstash journal save/recall/log/prune` for lightweight agent memory across sessions. Append-only entries with semantic recall, project tags, and time-window filtering.
- **Transcript parsing** — automatically extract structured journal entries from conversation logs.

### What's new in v0.11

- **Multi-profile support** — isolated databases per profile with `vstash profile create/list/delete/active`.
- **Federated search** — query across all profiles simultaneously with cross-profile deduplication.
- **Profile resolution chain** — `--profile` flag → `VSTASH_PROFILE` env → `default`.

### What's new in v0.10

- **Hybrid code splitting** — 3-tier backend: tree-sitter AST → parso AST → regex fallback. Each backend gracefully degrades to the next.
- **25+ languages** — tree-sitter support for C, C++, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, C#, Bash, Zig, Elixir, Erlang, Haskell, OCaml, Dart, Vue, Svelte (plus all previously supported).
- **Optional install** — `pip install vstash[treesitter]` for tree-sitter, or use parso (Python) + regex (6 languages) by default.

### What's new in v0.9

- **Auto-generated titles** — `vstash remember` generates descriptive slugs when no `--title` is provided.
- **Forget remembered text** — `vstash forget "text://<title>"` removes text ingested via `remember`.

### What's new in v0.8

- **Multilingual embeddings** — search in any language. Cross-lingual similarity improves ~40%.
- **`vstash reindex`** — switch embedding models without re-ingesting.
- **Intra-document MMR dedup** — replaces hard per-document dedup. Semantically diverse sections from the same long document now surface in results.

### Earlier versions

- **v0.7** — Adaptive scoring maturity gate (γ), zero-cost cold start.
- **v0.6** — Distance-based relevance signal (F1=0.952), document dedup, context expansion (±1 chunks).

---

## Install

```bash
pip install vstash
```

Or from source:

```bash
git clone https://github.com/stffns/vstash
cd vstash
pip install -e .
```

---

## Quick Start

### Search (free, no API key needed)

Semantic search works 100% locally — no inference backend required:

```bash
vstash add report.pdf
vstash add ~/docs/notes.md
vstash add https://arxiv.org/abs/2310.06825
vstash search "what is the proposed method?"
```

### Ask (requires an LLM backend)

To get natural language answers, configure an inference backend:

```bash
# Option A: Fully local with Ollama (free, private)
ollama pull llama3.2

# Option B: Fast with Cerebras (free tier available)
export CEREBRAS_API_KEY=your_key_here

# Option C: OpenAI or any compatible API
export OPENAI_API_KEY=your_key_here
```

Then:

```bash
vstash ask "summarize the key findings"
vstash chat   # interactive Q&A session
```

---

## Python SDK

Use vstash as a building block in your own agents and pipelines:

```python
from vstash import Memory

mem = Memory(project="my_agent")
mem.add("docs/spec.pdf")
mem.remember("OAuth uses PKCE for public clients", title="Auth Decision")

# Semantic search — free, no LLM
chunks = mem.search("deployment strategy", top_k=5)
for c in chunks:
    print(c.text, c.score, c.chunk_id)

# Direct chunk access by ID (O(1) lookup)
chunk = mem.get_chunk(chunks[0].chunk_id)

# Full document reconstruction from chunks
all_chunks = mem.get_document_chunks("docs/spec.pdf")

# Search + LLM answer
answer = mem.ask("What are the system requirements?")

# Cross-session journal
mem.journal_save("Decided to use FastAPI for the gateway")
entries = mem.journal_recall("architecture decisions")

# Management
mem.list()                # → list[DocumentInfo]
mem.stats()               # → StoreStats
mem.remove("docs/old.pdf")
```

---

## Commands

```
vstash add <file/dir/url>   Add documents to memory
vstash remember "<text>"    Ingest text directly (no file needed)
vstash ask "<question>"     Answer a question from your documents
vstash search "<query>"     Semantic search without LLM (free, local)
vstash chat                 Interactive Q&A session
vstash list                 Show all documents in memory
vstash stats                Memory statistics (docs, chunks, DB size)
vstash forget <file>        Remove a document from memory
vstash reindex              Re-embed all chunks with a new model
vstash watch <dir>          Auto-ingest on file changes
vstash export               Export chunks as JSONL for training data curation
vstash config               Show current configuration
vstash profile <cmd>        Manage named profiles (create, list, delete, active)
vstash journal <cmd>        Cross-session memory (save, recall, log, prune)
vstash-mcp                  Start MCP server (for Claude Desktop integration)
```

### Filtering with metadata

```bash
vstash add notes.md --collection research --project ml-survey --tags "attention,transformers"
vstash list --project ml-survey
vstash ask "what architectures were compared?" --project ml-survey
vstash export --project ml-survey --format jsonl
```

Documents with YAML frontmatter are parsed automatically:

```markdown
---
project: ml-survey
layer: literature-review
tags: [attention, transformers]
---

# My Research Notes
...
```

---

## Configuration

vstash looks for `vstash.toml` in your current directory, then `~/.vstash/vstash.toml`, then falls back to sensible defaults. Run `vstash config` to see your active settings.

See the [Configuration Reference](docs/configuration.md) for all options.

---

## Privacy

| Component | Data leaves machine? |
|---|---|
| Embeddings (FastEmbed) | Never — fully local ONNX |
| Vector store (sqlite-vec) | Never — local `.db` file (+ `.snpv` sidecar if snapvec enabled) |
| Semantic search | Never — local embeddings + SQLite |
| Inference (Cerebras/OpenAI) | Yes — query + retrieved chunks sent to API |
| Inference (Ollama) | Never — fully local |

For full privacy, use `backend = "ollama"` or skip inference entirely and use `vstash search` instead of `vstash ask`.

---

## Supported File Types

PDF, DOCX, PPTX, XLSX, Markdown, TXT, HTML, CSV — and any URL.

**Code files (25+ languages with tree-sitter):** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, C#, Bash, Zig, Elixir, Erlang, Haskell, OCaml, Dart, Vue, Svelte.

---

## Experiments

vstash retrieval quality has been validated at Kaggle scale:

| Experiment | Corpus | Best P@5 | Best MRR | Command |
|---|---|---|---|---|
| [ArXiv Retrieval Bench](experiments/arxiv_retrieval_bench.py) | 1,000 ML papers, 10 topics | 0.703 | 0.895 | `python -m experiments.arxiv_retrieval_bench` |
| [Dataset Discovery](experiments/dataset_discovery.py) | 954 HuggingFace datasets, 10 task categories | 0.629 | 0.777 | `python -m experiments.dataset_discovery` |

The dataset discovery engine also has an interactive mode — describe what you need, get the right dataset:

```bash
python -m experiments.dataset_discovery --interactive
> time series forecasting for retail sales
1. walmart-sales-dataset (time-series-forecasting) — 0.87
```

Run all experiments: `python -m experiments.run_all`

---

## Documentation

| Guide | Description |
|---|---|
| [Configuration](docs/configuration.md) | Full TOML reference — all sections and options |
| [How It Works](docs/how-it-works.md) | Ingestion pipeline, search pipeline, chunking strategies, RRF |
| [Memory Scoring](docs/scoring.md) | Frequency + decay re-ranking — formula, tuning, disabling |
| [MCP Server](docs/mcp-server.md) | Claude Desktop integration (15 tools) |
| [Claude Integration](docs/claude-integration.md) | Claude Code hook + Claude Desktop setup |
| [LangChain](docs/langchain.md) | VstashRetriever for chains and agents |
| [Embedding Models](docs/embedding-models.md) | Model comparison and backend selection |
| [Experiments](docs/experiments.md) | Retrieval benchmarks — hypotheses, results, conclusions |

---

## Roadmap

- **Phase 1 ✅:** Core — ingest, embed, hybrid search, answer
- **Phase 2 ✅:** Usability — MCP server, collections, watch mode, metadata, export
- **Phase 3 ✅:** Python SDK — `from vstash import Memory`
- **Phase 4 ✅:** LangChain integration — `VstashRetriever`
- **Phase 5 ✅:** Memory scoring — frequency + temporal decay re-ranking
- **Phase 6 ✅:** Retrieval quality — distance-based relevance signal, document dedup, context expansion
- **Phase 7 ✅:** Multilingual — cross-lingual embeddings, `vstash reindex`, MMR dedup
- **Phase 8 ✅:** Hybrid code splitting — tree-sitter + parso + regex, 25+ languages
- **Phase 9 ✅:** Multi-profile — isolated databases, federated search, profile management
- **Phase 10 ✅:** Cross-session journal — save, recall, log, prune for agent memory
- **Phase 11 ✅:** Direct chunk API — `get_chunk`/`get_chunks` for O(1) retrieval by ID

---

## Easter Egg

> In a 2018 Cornell paper *"Local Homology of Word Embeddings"*, researchers used the variable v_stash (p. 11) to refer to the "vector of the word stash" — making this the first documented use of the exact term in the context of AI/embeddings.

---

## License

MIT
