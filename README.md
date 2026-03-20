# vstash

**Local document memory with instant semantic search.**

![vstash demo](demo.gif)

Drop any file. Ask anything. Get an answer fast.

```
vstash add paper.pdf notes.md https://example.com/article
vstash ask "what's the main argument about X?"
vstash chat
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
| Inference | Cerebras API / Ollama / OpenAI | ~2,000 tok/s via Cerebras, or 100% local via Ollama |
| Parsing | markitdown | PDF, DOCX, PPTX, XLSX, HTML, Markdown, URLs |

**Philosophy: extreme speed at every layer.**

---

## Install

```bash
pip install vstash
```

Or from source:

```bash
git clone https://github.com/yourusername/vstash
cd vstash
pip install -e .
```

---

## Quick start

**1. Configure your backend** (copy the example):

```bash
cp vstash.toml.example vstash.toml
# Edit: set your Cerebras API key, or switch to ollama
```

Or just set the env var:

```bash
export CEREBRAS_API_KEY=your_key_here
```

**2. Add documents:**

```bash
vstash add report.pdf
vstash add ~/docs/notes.md
vstash add https://arxiv.org/abs/2310.06825
vstash add ./my-project/          # entire directory, recursive
```

**3. Ask questions:**

```bash
vstash ask "what is the proposed method?"
vstash ask "summarize the key findings"
vstash ask "what are the limitations?"
```

**4. Chat interactively:**

```bash
vstash chat
```

---

## Commands

```
vstash add <file/dir/url>   Add documents to memory (supports PDF, DOCX, PPTX, MD, TXT, code, URLs)
vstash ask "<question>"     Answer a question from your documents
vstash chat                 Interactive Q&A session
vstash list                 Show all documents in memory
vstash stats                Memory statistics (docs, chunks, DB size)
vstash forget <file>        Remove a document from memory
vstash config               Show current configuration
```

Options for `vstash ask`:
```
--top-k INT             Number of chunks to retrieve (default: from config)
--sources/--no-sources  Show source citations (default: show)
--stream/--no-stream    Stream the response token by token (default: stream)
```

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
size    = 1024    # tokens per chunk
overlap = 128     # token overlap between chunks
top_k   = 5       # chunks retrieved per query

[storage]
db_path = "~/.vstash/memory.db"
```

### Embedding models

| Model | Dims | Speed | Quality |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~700 chunks/s | ★★★★☆ |
| `BAAI/bge-base-en-v1.5` | 768 | ~300 chunks/s | ★★★★★ |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~300 chunks/s | ★★★★★ |

> ⚠️ Changing the embedding model requires re-ingesting all documents (dimensions must match).

---

## How it works

### Ingestion pipeline

```
file/URL
  → markitdown         (parse to plain text)
  → tiktoken           (count tokens)
  → chunk_text()       (1024 tok / 128 overlap)
  → FastEmbed ONNX     (embed each chunk, ~700 chunks/s)
  → sqlite-vec         (store vectors)
  → FTS5               (index text for keyword search)
```

### Search pipeline

```
query
  → FastEmbed ONNX     (embed query)
  → sqlite-vec         (top-k×10 vector candidates by cosine similarity)
  → FTS5               (top-k×10 keyword candidates by BM25)
  → RRF                (merge rankings: score = Σ 1/(60+rank))
  → top-k results      (default: 5 chunks)
  → LLM                (Cerebras, Ollama, or OpenAI)
```

**Reciprocal Rank Fusion** (k=60, vec_weight=0.6, fts_weight=0.4) ensures that:
- Semantic queries ("fast inference approach") find conceptually related chunks
- Exact keyword queries ("Cerebras API") are never missed due to embedding drift

---

## Privacy

| Component | Data leaves machine? |
|---|---|
| Embeddings (FastEmbed) | Never — fully local ONNX |
| Vector store (sqlite-vec) | Never — local `.db` file |
| Inference (Cerebras/OpenAI) | Yes — query + retrieved chunks sent to API |
| Inference (Ollama) | Never — fully local |

If privacy is paramount, use `backend = "ollama"` in your config.

---

## Supported file types

PDF, DOCX, PPTX, XLSX, Markdown, TXT, HTML, CSV, Python, JavaScript, TypeScript, Go, Rust, Java — and any URL.

---

## Roadmap

- **Phase 1 (now):** Core pipeline — ingest, embed, search, answer
- **Phase 2:** Watch mode (auto-ingest on file change), `vstash export`, JSON output for scripting
- **Phase 3:** Multi-agent sync via cr-sqlite (CRDT peer-to-peer, no server required)

---

## Easter Egg 🥚

> In a 2018 Cornell paper *"Local Homology of Word Embeddings"*, researchers used the variable $v_{stash}$ (p. 11) to refer to the "vector of the word stash" — making this the first documented use of the exact term in the context of AI/embeddings.

---

## License

MIT
