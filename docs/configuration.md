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
| `model` | string | `"auto"` | Model name (`"auto"` = use first available local model) |

```toml
[inference]
backend = "local"  # auto-detects Ollama, LM Studio, or any local OpenAI-compatible server
model = "auto"
```

> **Note:** Inference is only needed for `ask` and `chat`. Search works 100% locally with no LLM.

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

---

## `[scoring]`

Frequency + temporal decay re-ranking. See [Memory Scoring](scoring.md) for a full explanation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable frequency + decay re-ranking |
| `alpha` | float | `0.8` | Weight for semantic similarity (RRF) |
| `beta` | float | `0.2` | Weight for access history |
| `decay_lambda` | float | `0.05` | Temporal decay rate (higher = faster forgetting) |
| `over_fetch` | int | `50` | Candidates to retrieve before re-ranking |
| `track_access` | bool | `true` | Record access counts on each search |
| `mmr_lambda` | float | `0.5` | MMR diversity for intra-document dedup. 1.0 = hard dedup (one chunk per doc), 0.0 = maximum diversity |

```toml
[scoring]
enabled = true
alpha = 0.8
beta = 0.2
decay_lambda = 0.05
over_fetch = 50
track_access = true
mmr_lambda = 0.5
```

Set `enabled = false` to revert to pure RRF ranking. Set `mmr_lambda = 1.0` to restore the pre-v0.8 hard dedup behavior (at most one chunk per document).

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
