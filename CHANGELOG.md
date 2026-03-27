# Changelog

All notable changes to vstash are documented here.

## [0.5.1] — 2026-03-27

### Added
- **Code-aware chunking** — source code files now split at function/class boundaries instead of markdown headers
  - Regex-based splitting for Python, JavaScript/TypeScript, Go, Rust, Java (zero new deps)
  - Code files read as raw text, bypassing markitdown which destroyed code structure
  - Decorator/annotation post-processing keeps `@decorator` attached to its function/class
  - Configurable via `code_aware = true/false` in `[chunking]` config
  - Added `.tsx`/`.jsx` support to ingestion pipeline
- `_MIN_CHUNK_CHARS` constant replaces magic number across chunking functions

### Fixed
- Ruff lint and formatting cleanup from v0.5.0

---

## [0.5.0] — 2026-03-27

### Added
- **Frequency + temporal decay scoring** — post-RRF re-ranker that surfaces frequently-accessed, recent chunks
  - Formula: `final_score = α · normalized_rrf + β · log(1 + access_count · e^(−λ · days_ago))`
  - Enabled by default (α=0.8, β=0.2, λ=0.05, over_fetch=50)
  - Configurable via `[scoring]` section in `vstash.toml`
- Schema migration adds `access_count`, `last_accessed_at`, `created_at` columns to chunks table
  - Automatic backfill on existing databases (created_at from document's added_at)
  - Cold start: new chunks get `access_count = 1` (ingestion counts as first access)
- `rerank_with_decay()` method on `VstashStore` with min-max normalization of RRF scores
- `track_access()` records access frequency and recency on each search
- `ScoringConfig` with Pydantic validation in `vstash.toml`
- Per-stage latency benchmark (`benchmark/benchmark_scoring_latency.py`)
- Scoring grid search experiment (`experiments/scoring_grid.py`) — 16 configs × 5 scenarios × 10 queries

### Performance
- Scoring overhead: **0.12ms absolute** (~17% relative) on a 0.7ms total pipeline
- ANN lookup dominates at 71% — scoring is negligible
- All stages remain sub-millisecond at P99

---

## [0.4.1] — 2026-03-20

### Fixed
- Robust directory ingestion — proper exclusions and safety limits for async MCP
- Atomic delete for document removal
- SSRF redirect protection on URL ingestion
- Lazy tiktoken loading to avoid import-time overhead
- Clarified `vstash_add` docstring — top-level `.gitignore` only

---

## [0.4.0] — 2026-03-15

### Added
- **LangChain integration** — `VstashRetriever` for use in chains and agents
  - `pip install vstash[langchain]`
  - Returns standard LangChain `Document` objects with metadata
  - Supports project/collection/layer filtering
  - Compatible with LangSmith tracing

---

## [0.3.1] — 2026-03-10

### Fixed
- Comprehensive tech debt cleanup (#7)
- PyPI trusted publisher workflow fix

---

## [0.3.0] — 2026-03-08

### Added
- **Python SDK** — `from vstash import Memory`
  - Project/collection scoping, context managers
  - `memory.add()`, `memory.search()`, `memory.ask()`, `memory.list()`, `memory.stats()`
- **Semantic chunking** — split by Markdown headers and paragraphs instead of fixed windows
- **Export command** — `vstash export` for training data curation (JSONL format)

---

## [0.2.4] — 2026-03-01

### Added
- Hierarchical frontmatter + filtered retrieval
  - YAML frontmatter parsing for project, layer, tags
  - CLI flags: `--project`, `--collection`, `--tags`

---

## [0.2.3] — 2026-02-25

### Added
- Collections and namespaces
- Watch mode — `vstash watch <dir>` for auto-ingestion on file changes

---

## [0.2.2] — 2026-02-20

### Fixed
- 15 reliability, performance, and polish improvements
- MLX embedding backend for Apple Silicon GPU
- ONNX model warm-up to eliminate cold start
- RRF false positive elimination
- URL ingestion User-Agent fix (Wikipedia 403)

---

## [0.2.1] — 2026-02-15

### Fixed
- Defensive int coercion for `top_k` (MCP clients may send strings)
- RLock to prevent reentrant deadlock in MCP singletons

---

## [0.2.0] — 2026-02-10

### Added
- **MCP server** — `vstash-mcp` for Claude Desktop integration
  - 6 tools: add, ask, search, list, stats, forget
  - Thread-safe locking for concurrent access

---

## [0.1.0] — 2026-01-15

### Added
- Initial release
- Ingestion: PDF, DOCX, PPTX, XLSX, Markdown, TXT, HTML, CSV, code files, URLs
- Embeddings: FastEmbed (ONNX Runtime), ~700 chunks/s
- Vector store: sqlite-vec with cosine similarity
- Keyword search: FTS5 with porter stemming
- Hybrid ranking: Reciprocal Rank Fusion (k=60)
- Inference: Cerebras, Ollama, OpenAI backends
- CLI: add, ask, search, chat, list, stats, forget
