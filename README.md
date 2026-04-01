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

### What's new in v0.10.3

- **API retry with backoff** — transient errors (429, 503, timeout) are retried automatically across all inference backends.
- **Watch mode deletion** — `vstash watch` now removes documents from memory when files are deleted.
- **12 robustness fixes** — cross-collection isolation, reindex safety, scoring edge cases, and more.

### What's new in v0.10.1

- **Optional snapvec backend** — compressed ANN vector search via `pip install vstash[snapvec]`. Opt-in with `storage.vector_backend = "snapvec"` in `vstash.toml`. sqlite-vec remains the default for most users.

### What's new in v0.10

- **Hybrid code splitting** — 3-tier backend: tree-sitter AST → parso AST → regex fallback. Each backend gracefully degrades to the next.
- **25+ languages** — tree-sitter support for C, C++, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, C#, Bash, Zig, Elixir, Erlang, Haskell, OCaml, Dart, Vue, Svelte (plus all previously supported).
- **Optional install** — `pip install vstash[treesitter]` for tree-sitter, or use parso (Python) + regex (6 languages) by default.

### What's new in v0.9

- **Auto-generated titles** — `vstash remember` generates descriptive slugs when no `--title` is provided.
- **Forget remembered text** — `vstash forget "text://<title>"` removes text ingested via `remember`.

### What's new in v0.8

- **Multilingual embeddings** — search in any language. Queries in English and Spanish return the same results. Cross-lingual similarity improves ~40%.
- **`vstash reindex`** — switch embedding models without re-ingesting. Re-embeds all chunks in-place with a progress bar.
- **Intra-document MMR dedup** — replaces hard per-document dedup. Semantically diverse sections from the same long document now surface in results (3-5× more for cross-section queries).

### What's new in v0.7

- **Adaptive scoring** — maturity gate (γ) suppresses frequency+decay until access patterns show genuine signal (max/mean ≥ 8×). Scoring is now safe to enable by default.
- **Zero-cost cold start** — when γ = 0, scoring is completely short-circuited. Pure RRF with zero overhead.

### What's new in v0.6

- **Relevance signal** — distance-based confidence (F1=0.952) warns when results may not match your query.
- **Document deduplication** — improving diversity from ~3.2 to 5.0 unique docs per top-5.
- **Context expansion** — adjacent chunks (±1) automatically included for LLM answers, 2.64× richer context.
- **Tiered feedback** — high (silent), medium (`?` indicator), low (full warning) in CLI and MCP.
- **Discard telemetry** — search events tracked for real-world relevance signal validation.

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

# Semantic search — free, no LLM
chunks = mem.search("deployment strategy", top_k=5)
for c in chunks:
    print(c.text, c.score)

# Search + LLM answer
answer = mem.ask("What are the system requirements?")

# Management
mem.list()                # → list[DocumentInfo]
mem.stats()               # → StoreStats
mem.remove("docs/old.pdf")
```

---

## Commands

```
vstash add <file/dir/url>   Add documents to memory
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
| Vector store (sqlite-vec) | Never — local `.db` file |
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
| [MCP Server](docs/mcp-server.md) | Claude Desktop integration setup |
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
- **Phase 9:** Sync — cr-sqlite CRDT peer-to-peer sync, multiple profiles

---

## Easter Egg

> In a 2018 Cornell paper *"Local Homology of Word Embeddings"*, researchers used the variable v_stash (p. 11) to refer to the "vector of the word stash" — making this the first documented use of the exact term in the context of AI/embeddings.

---

## License

MIT
