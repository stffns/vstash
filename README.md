# vstash

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vstash)](https://pypi.org/project/vstash/)
[![python](https://img.shields.io/badge/python-3.10+-blue)]()
[![tests](https://img.shields.io/badge/tests-591_passing-brightgreen)]()
[![BEIR SciFact](https://img.shields.io/badge/BEIR_SciFact-NDCG@10_0.726-brightgreen)]()
[![MCP](https://img.shields.io/badge/MCP-16_tools-blue)]()
[![latency](https://img.shields.io/badge/latency-<25ms_@10K_chunks-brightgreen)]()

**Local hybrid retrieval engine that beats ColBERTv2 on BEIR SciFact. Beats BM25 on all 5 BEIR datasets.**

Single SQLite file. Zero cloud dependencies. Sub-25ms at 10K chunks.

```
pip install vstash
vstash add paper.pdf notes.md https://example.com/article
vstash search "what's the main argument about X?"
```

---

## Retrieval Quality

Evaluated on the [BEIR benchmark](https://github.com/beir-cellar/beir) — the standard for comparing retrieval systems:

| Dataset | vstash (NDCG@10) | ColBERTv2 | BM25 | Dense-only | 
|---------|:---:|:---:|:---:|:---:|
| SciFact (5K docs) | **0.726** | 0.693 (+4.8%) | 0.665 (+9.2%) | 0.653 (+11.2%) |
| NFCorpus (3.6K docs) | **0.359** | 0.344 (+4.4%) | 0.325 (+10.5%) | 0.338 (+6.2%) |
| SciDocs (25K docs) | **0.194** | 0.154 (+26.2%) | 0.158 (+23.0%) | 0.163 (+19.2%) |
| FiQA (57K docs) | **0.392** | 0.356 (+10.0%) | 0.236 (+65.8%) | **0.402** (−2.5%) |
| ArguAna (8.7K docs) | 0.437 | **0.463** (−5.6%) | 0.315 (+38.7%) | **0.584** (−25.2%) |

*Same embedding model (BGE-small 384d) across all comparisons. Adaptive RRF improves all 5 datasets vs fixed weights. Results reproducible via `python -m experiments.beir_benchmark`.*

---

## Why vstash?

| Layer | Technology | Why |
|---|---|---|
| Embeddings | FastEmbed (ONNX Runtime) | ~700 chunks/s, fully local, no server |
| Vector store | sqlite-vec | Single `.db` file, cosine similarity, zero deps |
| Keyword search | FTS5 (SQLite) | Exact matches, built into SQLite |
| Hybrid ranking | Reciprocal Rank Fusion | Semantic + keyword fusion — beats both alone |
| Recency | Optional temporal boost | Recent content ranks higher for agentic memory (off by default) |
| Dedup | Intra-document MMR | Diverse sections from long docs, not redundant chunks |
| Inference | Local auto-detect / Cloud | Ollama, LM Studio, Cerebras, OpenAI — all optional |

**Zero cloud required for search. Inference is optional.**

### What's new in v0.19

- **Recency boost** — `recency_boost` parameter on `search()` applies temporal decay favoring recent chunks. Designed for agentic memory. Off by default so pure retrieval is unaffected.
- **Temporal filters** — `added_after`/`added_before` ISO date parameters for hard time boundaries on all search surfaces.
- **`RecencyConfig`** — new `[recency]` section in `vstash.toml`.
- **591 tests** across 26 test modules.

### What's new in v0.18

- **Batch IDF cache** — `store.batch_mode()` context manager defers cache invalidation during bulk ingest (50x → 1x invalidation).
- **Scoring pipeline removed** — frequency+decay, history recall, and cross-encoder reranking all evaluated and removed after failing to improve NDCG on BEIR datasets. Replaced by the simpler recency boost in v0.19.

### What's new in v0.17

- **Dynamic chunk_size** — `Memory(chunk_size=2048)` or `vstash add --chunk-size 2048`. Per-document override without modifying config. Validation: overlap < chunk_size.
- **Adaptive RRF** — IDF-based weight adjustment per query. Rare terms boost keyword search, common terms boost vector search. Long queries relax distance cutoff. Improves all 5 BEIR datasets.

### What's new in v0.16

- **Local-first LLM auto-detect** — New default backend `"local"` probes for Ollama, LM Studio, or any OpenAI-compatible server. Zero config needed — just start a local server and `vstash ask` works.
- **Search --explain** — Diagnostic flag showing why each chunk ranked where it did: vector distance, FTS rank, RRF breakdown, frequency/decay scoring, and MMR penalty.
- **612 tests** across 27 test modules, all passing on Python 3.10–3.12.

### What's new in v0.15

- **Unified DB resolution** — CLI, MCP server, SDK, and reindex all share the same 6-level database resolution chain. Fixes bugs where different entry points could silently operate on different databases.
- **Federated context expansion** — `--all-profiles` now expands adjacent chunks per-store before merging, matching single-profile answer quality.
- **592 tests** across 27 test modules, all passing on Python 3.10–3.12.

### What's new in v0.14

- **Document reconstruction** — `get_document_chunks(path)` retrieves all chunks for a document in order. Available in Python SDK and as MCP tool.

### What's new in v0.13

- **Direct chunk retrieval** — `get_chunk(id)` and `get_chunks(ids)` for O(1) access to specific chunks by database ID. Enables downstream apps (spaced repetition, pinned references) to retrieve knowledge atoms without re-running search.

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

The minimum install gives you the SDK and the search/recall path:

```bash
pip install vstash
```

For the full CLI experience (`vstash add`, `vstash serve`) install the
relevant extras — vstash keeps heavy parsers and the web stack optional
so a programmatic-only consumer doesn't pay for them:

```bash
# CLI ingestion (PDF, DOCX, PPTX, XLSX, HTML, …)
pip install 'vstash[ingest]'

# Web UI + /health + /metrics endpoints (vstash serve)
pip install 'vstash[serve]'

# Everything at once (CLI + serve + LLM backends + watch + …)
pip install 'vstash[all]'
```

Or from source:

```bash
git clone https://github.com/stffns/vstash
cd vstash
pip install -e '.[all]'
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

To get natural language answers, start any local LLM server — vstash auto-detects it:

```bash
# Option A: Ollama (auto-detected on port 11434)
ollama pull qwen3.5:9b

# Option B: LM Studio (auto-detected on port 1234 or 8080)
# Just load a model in the GUI

# Option C: Cloud backends (set in vstash.toml)
# inference.backend = "cerebras" + inference.model = "llama3.1-8b" + CEREBRAS_API_KEY env
# inference.backend = "openai"   + OPENAI_API_KEY env
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

Search is always private. For fully private answers, use a local LLM (default) or skip inference entirely with `vstash search`.

---

## Supported File Types

PDF, DOCX, PPTX, XLSX, Markdown, TXT, HTML, CSV — and any URL.

**Code files (25+ languages with tree-sitter):** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, C#, Bash, Zig, Elixir, Erlang, Haskell, OCaml, Dart, Vue, Svelte.

---

## Experiments

| Experiment | Corpus | Key Result | Command |
|---|---|---|---|
| [BEIR Benchmark](experiments/beir_benchmark.py) | 5 BEIR datasets, up to 57K docs | Beats BM25 5/5, ColBERTv2 4/5; NDCG@10=0.726 on SciFact | `python -m experiments.beir_benchmark` |
| [ArXiv Retrieval](experiments/arxiv_retrieval_bench.py) | 1,000 ML papers, 3 models | P@5=0.703, MRR=0.895 | `python -m experiments.arxiv_retrieval_bench` |
| [Dataset Discovery](experiments/dataset_discovery.py) | 954 HuggingFace datasets | 91.4% discovery rate | `python -m experiments.dataset_discovery` |
| [Answer Relevance](experiments/answer_relevance.py) | SciFact, NFCorpus | +8.3% answer quality vs Chroma (LLM judge) | `python -m experiments.answer_relevance` |

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
| [Recency & Temporal Filters](docs/scoring.md) | Recency boost, temporal date filters, MMR dedup |
| [MCP Server](docs/mcp-server.md) | MCP integration — 16 tools for any MCP-compatible client |
| [Agent Integration](docs/claude-integration.md) | Claude Code, Claude Desktop, and other LLM agents |
| [LangChain](docs/langchain.md) | VstashRetriever for chains and agents |
| [Embedding Models](docs/embedding-models.md) | Model comparison and backend selection |
| [Experiments](docs/experiments.md) | Retrieval benchmarks — hypotheses, results, conclusions |

---

## Roadmap

- **Phase 1 ✅:** Core — ingest, embed, hybrid search, answer
- **Phase 2 ✅:** Usability — MCP server, collections, watch mode, metadata, export
- **Phase 3 ✅:** Python SDK — `from vstash import Memory`
- **Phase 4 ✅:** LangChain integration — `VstashRetriever`
- **Phase 5 ✅:** Memory scoring — recency boost + temporal filters (v0.19)
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
