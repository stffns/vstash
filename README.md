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
| Embeddings | FastEmbed (ONNX Runtime) | ~700 chunks/s, fully in-process, no server |
| Vector store | sqlite-vec | Single `.db` file, cosine similarity, zero deps |
| Keyword search | FTS5 (SQLite) | Exact matches, porter stemming, built into SQLite |
| Hybrid ranking | Reciprocal Rank Fusion | Best of both: semantic + keyword, no training needed |
| Chunking | Semantic-first | Markdown headers & paragraphs, with token-bounded fallback |
| Inference | Cerebras / Ollama / OpenAI | ~2,000 tok/s via Cerebras, or 100% local via Ollama |
| Parsing | markitdown | PDF, DOCX, PPTX, XLSX, HTML, Markdown, URLs |

**Philosophy: extreme speed at every layer. Zero cloud required for search.**

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

## Quick start

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
# Install Ollama: https://ollama.com
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

*New in v0.3.0.* Use vstash as a building block in your own agents and pipelines:

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

The `Memory` class supports project/collection scoping, context managers, and works with any configured inference backend. See the full API in `vstash/memory.py`.

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
vstash watch <dir>          Auto-ingest on file changes
vstash export               Export chunks as JSONL for training data curation
vstash config               Show current configuration
vstash-mcp                  Start MCP server (for Claude Desktop integration)
```

### Filtering with metadata

vstash supports hierarchical metadata via frontmatter or CLI flags:

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

## MCP Server — Claude Desktop Integration

vstash includes a built-in [MCP](https://modelcontextprotocol.io/) server that gives Claude Desktop persistent document memory across sessions.

### Setup

**1. Install vstash:**

```bash
pip install vstash
```

**2. Add to Claude Desktop config** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vstash": {
      "command": "vstash-mcp"
    }
  }
}
```

> If using `pyenv`, use the full path: `"command": "/path/to/.pyenv/versions/3.x.x/bin/vstash-mcp"`

**3. Restart Claude Desktop.**

### Available MCP Tools

| Tool | Description |
|---|---|
| `vstash_add(path)` | Ingest a file, directory, or URL into memory |
| `vstash_ask(query, top_k)` | Semantic search + LLM-generated answer with sources |
| `vstash_search(query, top_k)` | Raw retrieval without LLM — returns chunks with scores |
| `vstash_list()` | List all ingested documents |
| `vstash_stats()` | Database statistics (doc count, chunks, size) |
| `vstash_forget(source)` | Remove a document from memory |

> Make sure your `~/.vstash/vstash.toml` includes the API key under `[cerebras]` (or your chosen backend), since MCP servers don't inherit shell environment variables.

---

## Configuration

vstash looks for `vstash.toml` in your current directory, then `~/.vstash/vstash.toml`, then falls back to defaults.

```toml
[inference]
backend = "cerebras"       # "cerebras" | "ollama" | "openai"
model   = "llama3.1-8b"

[cerebras]
api_key = ""               # or set CEREBRAS_API_KEY env var

[ollama]
host  = "http://localhost:11434"
model = "llama3.2"

[embeddings]
model = "BAAI/bge-small-en-v1.5"   # 384 dims, ~700 chunks/s

[chunking]
size    = 1024    # max tokens per chunk
overlap = 128     # token overlap (used in fixed-window fallback)
top_k   = 5       # chunks retrieved per query

[storage]
db_path = "~/.vstash/memory.db"
```

### Embedding models

| Model | Dims | Speed | Quality |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~700 chunks/s | Great |
| `BAAI/bge-base-en-v1.5` | 768 | ~300 chunks/s | Excellent |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~300 chunks/s | Excellent |

> Changing the embedding model requires re-ingesting all documents (dimensions must match).

---

## How it works

### Ingestion pipeline

```
file/URL
  → markitdown         (parse to plain text)
  → _split_by_headers  (Markdown sections)
  → _split_by_paragraphs (paragraph boundaries)
  → _fixed_window      (fallback for oversized paragraphs)
  → _merge_small       (merge tiny chunks < 80 tokens)
  → FastEmbed ONNX     (embed each chunk, ~700 chunks/s)
  → sqlite-vec         (store vectors)
  → FTS5               (index text for keyword search)
```

Semantic chunking preserves document structure: Markdown headers stay with their body content, paragraphs aren't torn mid-sentence, and tiny fragments are merged to avoid low-quality embeddings.

### Search pipeline

```
query
  → FastEmbed ONNX     (embed query)
  → sqlite-vec         (top-k×10 vector candidates by cosine similarity)
  → FTS5               (top-k×10 keyword candidates by BM25)
  → RRF                (merge rankings: score = Σ 1/(60+rank))
  → top-k results      (default: 5 chunks)
  → LLM                (optional: Cerebras, Ollama, or OpenAI)
```

**Reciprocal Rank Fusion** (k=60, vec_weight=0.6, fts_weight=0.4) ensures that semantic queries find conceptually related chunks while exact keyword queries are never missed.

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

## Supported file types

PDF, DOCX, PPTX, XLSX, Markdown, TXT, HTML, CSV, Python, JavaScript, TypeScript, Go, Rust, Java — and any URL.

---

## Roadmap

- **Phase 1 ✅:** Core — ingest, embed, hybrid search, answer
- **Phase 2 ✅:** Usability — MCP server, collections/namespaces, watch mode, frontmatter metadata, export, semantic chunking
- **Phase 3 ✅:** Python SDK — `from vstash import Memory`
- **Phase 4:** Sync — cr-sqlite CRDT peer-to-peer sync, multiple profiles, REST API

---

## Easter Egg

> In a 2018 Cornell paper *"Local Homology of Word Embeddings"*, researchers used the variable v_stash (p. 11) to refer to the "vector of the word stash" — making this the first documented use of the exact term in the context of AI/embeddings.

---

## License

MIT
