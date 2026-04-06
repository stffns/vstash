# Configuration Reference

vstash loads configuration from `vstash.toml` in the following order:

1. `VSTASH_CONFIG` environment variable (if set)
2. `./vstash.toml` in the current directory
3. `~/.vstash/vstash.toml` (global)
4. Built-in defaults

Run `vstash config` to see your active settings.

---

## `[inference]`

Controls which LLM backend is used for `vstash ask` and `vstash chat`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `"local"` | `"local"` (auto-detect, default), `"ollama"`, `"cerebras"`, or `"openai"` |
| `model` | string | `"auto"` | Model name. Only used by `"cerebras"` backend; ignored when `"local"`, `"ollama"`, or `"openai"` (which use their own `model` setting) |

```toml
[inference]
backend = "local"  # auto-detects Ollama, LM Studio, or any local OpenAI-compatible server
model = "auto"

# To use a specific backend instead:
# backend = "cerebras"
# model = "llama3.1-8b"
```

> **Note:** Inference is only needed for `ask` and `chat`. Search works 100% locally with no LLM. When `backend = "local"`, vstash probes Ollama (port 11434), LM Studio (ports 1234, 8080), and LocalAI (port 8081) in order, using the first that responds.

---

## `[cerebras]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_key` | string | `""` | Cerebras API key |

```toml
[cerebras]
api_key = ""  # prefer CEREBRAS_API_KEY env var
```

> **Security:** Prefer the `CEREBRAS_API_KEY` environment variable over storing the key in `vstash.toml`, which may be committed to version control.

---

## `[ollama]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `"http://localhost:11434"` | Ollama server URL |
| `model` | string | `"qwen3.5:9b"` | Ollama model name |

```toml
[ollama]
host = "http://localhost:11434"
model = "qwen3.5:9b"
```

---

## `[openai]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_key` | string | `""` | OpenAI API key |
| `model` | string | `"gpt-4o-mini"` | OpenAI model name |
| `base_url` | string | `null` | Custom base URL for OpenAI-compatible APIs |

```toml
[openai]
api_key = ""  # prefer OPENAI_API_KEY env var
model = "gpt-4o-mini"
# base_url = "https://my-proxy.example.com/v1"  # optional
```

> **Security:** Prefer the `OPENAI_API_KEY` environment variable.

---

## `[embeddings]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `"BAAI/bge-small-en-v1.5"` | FastEmbed model name |
| `backend` | string | `"auto"` | `"onnx"` (portable), `"mlx"` (Apple Silicon GPU), or `"auto"` |

```toml
[embeddings]
model = "BAAI/bge-small-en-v1.5"
backend = "auto"
```

See [Embedding Models](embedding-models.md) for model comparison and notes on changing models.

---

## `[chunking]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `size` | int | `1024` | Max tokens per chunk |
| `overlap` | int | `128` | Token overlap between chunks (fixed-window fallback) |
| `top_k` | int | `5` | Chunks retrieved per query |
| `code_aware` | bool | `true` | Split code files at function/class boundaries instead of fixed windows |

```toml
[chunking]
size = 1024
overlap = 128
top_k = 5
code_aware = true
```

When `code_aware` is enabled, source code files are split at top-level function and class definitions using a 3-tier backend: tree-sitter AST (25+ languages, requires `pip install vstash[treesitter]`) → parso AST (Python, included by default) → regex (Python, JS/TS, Go, Rust, Java). Non-code files use semantic chunking (Markdown headers → paragraphs → fixed-window fallback).

**Per-document override:** Chunk size can be overridden per document via the SDK or CLI without changing the config file:

```python
# SDK — larger chunks for medical/legal documents
mem = Memory(project="medical", chunk_size=2048, chunk_overlap=256)
mem.add("protocol.pdf")                        # uses 2048
mem.add("code.py", chunk_size=512)             # per-file override
```

```bash
# CLI
vstash add paper.pdf --chunk-size 2048 --chunk-overlap 256
```

Validation: `chunk_size` must be positive, `chunk_overlap` must be non-negative and less than `chunk_size`.

---

## `[recency]`

*Added in v0.19.0.* Temporal recency boost for agentic memory. See [Recency Boost & Temporal Filters](scoring.md) for details.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `boost` | float | `0.0` | Default recency boost multiplier. 0.0 = off, 0.5 = mild, 1.0 = strong. Applied as `score *= (1 + boost * exp(-0.05 * days_ago))` |

```toml
[recency]
boost = 0.0   # off by default — pure retrieval unaffected
# boost = 0.5 # mild recency bias for agentic memory
# boost = 1.0 # strong recency bias
```

The `recency_boost` parameter can also be set per-call in `store.search()`, `Memory.search()`, and MCP tools, overriding the config default.

### Temporal filters

`added_after` and `added_before` parameters are available on all search surfaces (store, SDK, MCP). These are per-call only — no config setting.

```python
mem.search("meeting notes", added_after="2024-06-01", added_before="2024-12-31")
```

---

## `[scoring]` (deprecated)

The `[scoring]` section from v0.5–v0.17 is still parsed for backward compatibility — existing `vstash.toml` files won't error. However, all scoring parameters are ignored. The frequency+decay scoring pipeline was removed in v0.18.0 and replaced by the simpler `[recency]` boost in v0.19.0.

---

## `[storage]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | string | `"~/.vstash/memory.db"` | Path to SQLite database file |
| `vector_backend` | string | `"sqlite-vec"` | `"sqlite-vec"` (default) or `"snapvec"` (compressed ANN) |
| `snapvec_bits` | int | `4` | Quantization bits for snapvec (2, 3, or 4) |

```toml
[storage]
db_path = "~/.vstash/memory.db"
vector_backend = "sqlite-vec"  # or "snapvec" for compressed ANN
# snapvec_bits = 4             # only used when vector_backend = "snapvec"
```

> **snapvec** uses PolarQuant compression for ~8x smaller vector storage at slightly reduced recall. Install with `pip install vstash[snapvec]`. sqlite-vec is the correct default for most users.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CEREBRAS_API_KEY` | Cerebras API key (preferred over config file) |
| `OPENAI_API_KEY` | OpenAI API key (preferred over config file) |
| `VSTASH_CONFIG` | Path to a specific `vstash.toml` file |
