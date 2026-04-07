# Changelog

All notable changes to vstash are documented here.

## [0.25.1] — 2026-04-07

### Fixed
- **CLI hardening** — `rich`-escape exception messages so broken paths or user-supplied strings can no longer break CLI rendering.
- **Clearer install docs** — dedicated `vstash[serve]` extra for the web interface; install-path guidance rewritten for PyPI users.
- **E2E from PyPI install** — hotfix caught by end-to-end verification against a fresh PyPI install (#148).

---

## [0.25.0] — 2026-04-07

### Added
- **Explicit contracts and schema versioning** (#135, #144) — a good substrate makes its contracts explicit.
  - `SCHEMA_VERSION` constant and `KNOWN_SCHEMA_VERSIONS` set in `vstash/store.py`.
  - `SchemaVersionError` raised on `open()` when the database declares a version this build does not recognize.
  - Fresh databases are stamped with the current version; legacy unstamped databases are re-stamped as `v1`; the recorded vstash version is refreshed on every open.
  - `VstashConfig` now allows **unknown top-level keys** with a one-time WARNING (forward-compatible config); nested sections keep strict validation.
  - `SearchResult.score` docstring documents typical range, comparability **within** a query, and the explicit rule that scores are **NOT comparable across queries**.

### Fixed
- **Concurrent fresh-open race** (#145) — schema version stamping now uses `INSERT OR IGNORE`, surviving concurrent first-open across threads and processes.

---

## [0.24.1] — 2026-04-07

### Fixed
- **Integrity hotfix** (#134, #142) — `integrity_repair` is now **collection-scoped**: operating on one collection cannot clobber data in another. Earlier global scope was too aggressive.
- **FTS5 parity check** — corrected the invariant query so `integrity_check` no longer reports false positives on healthy stores.

---

## [0.24.0] — 2026-04-07

### Added
- **Integrity & recovery** (#134, #140) — a good substrate is honest about what survived a crash.
  - `VstashStore.doc_completeness(path)` classifies a path as **missing / partial / complete** (chunk_count parity + vec_chunks parity).
  - `ingest()` is now **idempotent**: complete docs are skipped, partial docs are dropped and re-ingested fresh, missing docs ingest from scratch.
  - `VstashStore.integrity_check()` runs five invariants: chunk_count parity, vec/snapvec parity, fts_chunks parity, orphan chunks, and SQLite `PRAGMA integrity_check`.
  - `VstashStore.integrity_repair()` recomputes chunk_count, rebuilds `fts_chunks` via FTS5 `rebuild`, and deletes orphan chunks (with their `vec_chunks` companions).
  - New `IntegrityCheck` and `IntegrityRepair` Pydantic models in `vstash/models.py`.
  - New `vstash check [--repair] [--json]` CLI command with rich table output.

---

## [0.23.0] — 2026-04-07

### Added
- **Explicit limits at public API boundaries** (#133, #138) — new `vstash/validation.py` module that rejects pathological inputs at the `VstashStore` and `Memory` boundaries before they reach SQLite, sqlite-vec, or the embedding model.
  - `LimitsConfig` (new `[limits]` section in `vstash.toml`) with seven knobs: `max_query_chars`, `max_top_k`, `max_distance_cutoff`, `max_recency_boost`, `max_path_chars`, `max_chunks_per_document`, `max_chunk_chars`.
  - `LimitError(ValueError)` hierarchy with one subclass per category so callers can catch a single bucket or the whole family.
  - Malformed inputs now produce typed Python exceptions at the API boundary instead of opaque SQLite / ONNX failures deep in the stack.

---

## [0.22.0] — 2026-04-07

### Added
- **Operational observability** (#132, #136) — transparent internal state that upper-layer memory frameworks (Mem0, Zep, LangChain memory) cannot expose.
  - In-process **metrics registry** with per-stage latency histograms across ingest and search pipelines.
  - **Slow query log** capturing query text, stage breakdown (vector ANN, FTS5, RRF fusion, MMR, context expansion), and result count for any search exceeding a configurable threshold.
  - Accessible via the Python SDK and MCP tools — operators running `vstash serve` or the MCP server are no longer flying blind.

---

## [0.21.0] — 2026-04-07

### Added
- **Ranking miss analysis** (#108, #130) — `VstashStore.miss_analysis(query, expected_doc)` diagnoses *why* an expected document did not appear in a result set. Returns a structured trace identifying where the chunk was eliminated in the pipeline (vector ANN cutoff, FTS5 Porter-stem mismatch, RRF rank dropout, MMR redundancy penalty, post-fusion distance cutoff) plus rule-based suggestions. Exposed via SDK, CLI, and MCP — transparent retrieval debugging without LLM dependencies.

---

## [0.20.2] — 2026-04-06

### Changed
- **Threading hardening** (#128) — the assumption that the underlying `libsqlite` is built with `SQLITE_THREADSAFE=1` is now *explicit*: checked at `open()` and surfaced as a clear error rather than manifesting as sporadic corruption.

---

## [0.20.1] — 2026-04-06

### Fixed
- **Close STEM connections from any thread** (#125, #127) — fixes an asyncio/threading deadlock in the MCP server path where embedding connections could only be closed from the thread that opened them.

---

## [0.20.0] — 2026-04-06

### Added
- **`vstash serve`** (#121) — pocket memory agent web interface, a lightweight HTTP/SSE server that exposes search, ask, and journal over HTTP for local agents and browser-based tools.

### Fixed
- **SQLite resource leaks + parent-child negative result evidence** (#124).

---

## [0.19.0] — 2026-04-06

### Added
- **Recency boost** — `recency_boost` parameter on `store.search()`, `Memory.search()`, and MCP `vstash_search`. Applies temporal decay to RRF scores, favoring recently created chunks. Off by default (0.0) so pure retrieval is unaffected.
- **Temporal filters** — `added_after`/`added_before` ISO date parameters for hard time boundaries on all search surfaces (store, SDK, MCP search, MCP ask).
- **`RecencyConfig`** — new `[recency]` config section in `vstash.toml` with configurable `boost` default.
- 7 new tests for recency boost and temporal filters.

### Changed
- 591 tests (up from 584).

---

## [0.18.2] — 2026-04-06

### Added
- **Batch IDF cache invalidation** — `store.batch_mode()` context manager defers IDF cache invalidation during bulk operations. `ingest_directory` now triggers 1 invalidation instead of N.
- 8 new tests for batch_mode: deferral, nesting, exception safety, deletes, search correctness.

### Fixed
- Insecure `tempfile.mktemp()` replaced with `mkstemp()` in ablation experiment.
- Guaranteed cleanup with `try/finally` in ablation experiment.

### Changed
- 584 tests (up from 576).

---

## [0.18.1] — 2026-04-05

### Added
- **Multi-dataset ablation experiment** — pipeline lift measured across SciFact, NFCorpus, SciDocs, FiQA, ArguAna.
- **Pipeline ablation on BEIR SciFact** — vector-only → +FTS/RRF → +adaptive IDF+MMR.

---

## [0.18.0] — 2026-04-05

### Removed
- **Frequency+decay scoring pipeline** — `rerank_with_decay()`, `scoring_maturity()`, `track_access()`, `total_access_count()` removed after failing to improve NDCG on any benchmark (SciFact: -1.6%, scoring grid: 0%, cross-encoder: -0.3% to -3.1%).
- Over-fetch logic, scoring parameters on `search()`, scoring fields in `ExplainInfo`.
- `test_scoring.py` and `test_scoring_e2e.py` (850+ lines).

### Kept
- `access_count`, `last_accessed_at`, `created_at` columns on chunks (backward compat, zero cost).
- `ScoringConfig` class in `config.py` (backward compat for existing `vstash.toml` files).
- All scoring experiment files (historical evidence).

### Changed
- Pipeline simplified: vector + FTS5 → adaptive RRF → MMR dedup.
- 576 tests (down from ~580 due to removed scoring tests, up from additions).

---

## [0.17.5] — 2026-04-04

### Added
- **Dynamic chunk_size** — `Memory(chunk_size=2048)` or `vstash add --chunk-size 2048`. Per-document override without modifying config.
- **Adaptive RRF** — IDF-based weight adjustment per query. Rare terms boost FTS weight; common terms boost vector weight. Long queries relax distance cutoff.
- 6 benchmark regression tests for BEIR NDCG@10 thresholds.

---

## [0.10.4] — 2026-04-01

### Added
- **`delete_by_path_prefix` empty-prefix guard** — raises `ValueError` on empty prefix to prevent accidental full wipe
- **4 tests for `delete_by_path_prefix`** — basic prefix match, zero-match returns 0, SQL LIKE wildcard escaping (%, _), empty-prefix ValueError

---

## [0.10.3] — 2026-04-01

### Added
- **Watch mode file deletion** — `on_deleted` handler automatically removes deleted files from the store
- **Stream interruption warning** — shows "Stream interrupted after N tokens" on mid-stream errors
- **Frontmatter validation warnings** — warns when `project`/`layer` is a dict/list instead of silently dropping
- **URL title extraction** — URLs now get real titles from parsed content instead of raw URL
- **API retry with exponential backoff** — retries transient errors (429, 503, timeout) for all inference backends

### Fixed
- **expand_context cross-collection isolation** — resolves doc_id via chunk text match to prevent leaking chunks across collections
- **Reindex dim safety** — `embedding_dim` only updates after successful commit; rollback restores correct state
- **Watch shutdown cleanup** — `stop_event` + queue drain for clean exit without orphaned threads or DB locks
- **Scoring maturity gate** — guards against division-by-zero when access mean ≈ 0
- **Scoring bias for new chunks** — new chunks get `freq_normalized=0.0` instead of artificial frequency boost
- **Unknown embedding model** — raises `ValueError` with list of known models instead of silent 384-dim default
- **MMR fallback visibility** — upgraded from debug to warning when falling back to hard dedup
- **SnapIndex failure messages** — actionable "run vstash reindex" hint when `.snpv` file is corrupt
- **Shared `relevance_tier()` helper** — deduplicated from cli.py, mcp.py into store.py

### Changed
- 447 tests (up from 368), including 9 watch e2e integration tests and 13 robustness tests

---

## [0.10.2] — 2026-04-01

### Added
- **`openai.extra_body` config** — pass arbitrary JSON fields to OpenAI-compatible chat completions (e.g., `chat_template_kwargs` for Qwen thinking mode, vLLM sampling params)
- 2 new config tests for `extra_body` loading

---

## [0.10.1] — 2026-03-31

### Added
- **Optional snapvec vector backend** — compressed ANN search via [snapvec](https://pypi.org/project/snapvec/) (PolarQuant). Opt-in with `storage.vector_backend = "snapvec"` in `vstash.toml`. sqlite-vec remains the default.
- New config fields: `storage.vector_backend` (`"sqlite-vec"` | `"snapvec"`) and `storage.snapvec_bits` (2–4)
- Optional dependency: `pip install vstash[snapvec]`
- 12 new tests for snapvec backend (add, search, delete, persistence, reindex, dim mismatch)

### Changed
- 368 tests (up from 356)

---

## [0.10.0] — 2026-03-31

### Added
- **Hybrid code splitting** — 3-tier backend with graceful degradation:
  1. **tree-sitter** (AST-level, 25+ languages) via optional `tree-sitter-language-pack`
  2. **parso** (AST-level, Python only) — now a base dependency
  3. **regex** (pattern-based, 6 languages) — original fallback
- New `vstash/code_split.py` module with clean separation from `ingest.py`
- **25+ language support** via tree-sitter: Python, JS/TS, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, Lua, R, C#, Bash, Zig, Elixir, Erlang, Haskell, OCaml, Dart, Vue, Svelte
- **Backend-forcing tests** — each splitting tier tested independently via monkeypatching
- **Unicode safety** — tree-sitter byte-offset handling correctly handles multi-byte characters
- Optional dependency: `pip install vstash[treesitter]` for tree-sitter support

### Fixed
- UTF-8 byte vs char offset bug in tree-sitter backend (multi-byte characters safe)
- Data loss between definitions — full source preserved by slicing between definition boundaries
- C/C++ `declaration` node type added for proper function prototype recognition

### Changed
- `parso>=0.8.0` moved to base dependencies (was not included before)
- Code splitting logic extracted from `ingest.py` into dedicated `code_split.py` module
- 356 tests (up from 326)

---

## [0.9.0] — 2026-03-31

### Added
- **Auto-generated titles for `vstash remember`** — when no `--title` is provided, generates a slug from the first 5 words + UTC timestamp with microsecond precision (e.g. `oauth2-uses-pkce-20260330-143052474102`)
- **`vstash forget` support for remembered text** — use `text://<title>` path prefix

### Fixed
- `Memory.remove()` no longer mangles `text://` synthetic paths via `Path.resolve()`
- `ingest_text()` signature: `title` is now keyword-only, `cfg` and `store` are required positional params
- Added missing `tests/__init__.py` — all tests now collect correctly

---

## [0.8.0] — 2026-03-29

### Added
- **Multilingual embedding support** — new models in registry:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, 50+ languages)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dims, 50+ languages)
  - `intfloat/multilingual-e5-large` (1024 dims, 100+ languages)
- **`vstash reindex` command** — re-embed all chunks with a new model without re-ingesting. Supports `--model`, `--batch-size`, `--yes` flags. Progress bar via Rich.
- **Intra-document MMR deduplication** — replaces hard per-document dedup. Greedy MMR selection penalizes same-document chunks by cosine similarity, allowing diverse sections from long documents to surface. Configurable via `scoring.mmr_lambda` (default 0.5).
- **Negative MMR cutoff** — stops selecting when best remaining candidate has MMR < 0 (redundancy exceeds relevance).
- **`_cosine_sim()` helper** for MMR similarity computation.
- **4 reindex tests** and **4 MMR dedup tests** (312 total).
- **ArXiv retrieval benchmark** (`experiments/arxiv_retrieval_bench.py`) — 1,000 ML papers from HuggingFace, 10 topic clusters, 3 models × 5 configs. BGE-base (768d) P@5=0.703, MRR=0.895. Validates hybrid RRF, scoring, and model comparison.
- **Dataset discovery engine** (`experiments/dataset_discovery.py`) — 954 HuggingFace Hub datasets, 10 task categories. P@5=0.629, MRR=0.777, 91.4% discovery rate. Interactive REPL mode with `--interactive`.

### Changed
- `_mmr_dedup()` replaces the hard dedup block in search pipeline. `mmr_lambda=1.0` degrades to hard dedup for backwards compatibility.
- Narrower exception handling: `sqlite3.Error` in MMR fallback, `URLError/JSONDecodeError` in experiment fetch (PR review feedback).

### Paper
- §3.4: Rewritten as "Intra-Document MMR Deduplication" with formula and comparison table.
- §6: Code-aware chunking regex justification and Tree-sitter tradeoff analysis.
- §8.6: Updated with real Wikipedia experiment data (120 articles, 919 chunks).
- §9: MMR λ design rationale with Carbonell & Goldstein 1998 reference.
- §10: Updated conclusions with real Wikipedia and MMR results.

---

## [0.7.0] — 2026-03-28

### Added
- **Adaptive scoring maturity gate (γ)** — suppresses frequency+decay scoring until access patterns show genuine outlier signal (max/mean ≥ 8×). Linear ramp between 8× and 15×.
- **Zero-cost cold start** — when γ = 0, scoring is short-circuited entirely: no metadata lookups, no decay computation.
- **Cold start experiment** — 120 real Wikipedia articles across 12 CS topic clusters (919 chunks), 30 rounds, Zipf-weighted queries. Adaptive γ maintains 0.0% degradation vs fixed β which degrades in 6/30 rounds.
- Experiment scripts and cached Wikipedia corpus in `experiments/`.

### Changed
- Scoring is now safe to enable by default — γ eliminates the -8.6% cold start degradation from fixed β.

---

## [0.6.0] — 2026-03-27

### Added
- **Relevance signal** — distance-based confidence tier (F1=0.952) using cosine distance of best vector match. Tiers: high (≤0.95), medium (0.95-0.98), low (>0.98).
- **Document deduplication** — one result per document in search, improving diversity from ~3.2 to 5.0 unique docs per top-5.
- **Context expansion** — adjacent chunks (±1 window) automatically included for LLM answers. 2.64× richer context at +0.12ms.
- **Tiered ghost warning** — high (silent), medium (`?` indicator), low (full `⚠` warning) in CLI and MCP.
- **LLM grounding** — system prompt rules enforce source citation, passing 9/9 anti-hallucination trap tests.
- **Discard telemetry** — `search_events` table tracks query, distance, tier, result count. Chat mode marks dismissed events.

---

## [0.5.3] — 2026-03-27

### Added
- **Relevance signal** — search results now include a `relevance` field (`high`, `low`, `none`) based on score spread
  - CLI: shows `⚠ Results may not be relevant` warning when spread < 0.15
  - CLI `--json`: includes `relevance` field in output
  - MCP: `vstash_search` returns `relevance` + `hint` so LLM clients can filter noise
- **MCP server instructions** — explicit guidance for LLM clients on when to use/skip vstash tools
- **Claude Code integration** — hook, skills, and setup guide
  - `vstash-context.sh` hook: auto-injects document context on knowledge questions
  - `/memory` and `/remember` slash commands
  - `docs/claude-integration.md` — setup guide for Claude Code (hook) and Claude Desktop (MCP)

---

## [0.5.2] — 2026-03-27

### Added
- **`vstash search` CLI command** — semantic search without LLM, free and fully local
  - Table output with normalized scores, source, and text preview
  - `--json` flag for programmatic output
  - Supports `--collection`, `--project`, `--layer` filters
- **PyPI metadata** — project.urls, classifiers, sdist exclusions (~1MB → 70KB)
- **docs/ directory** — 9 standalone guides (configuration, scoring, MCP, LangChain, how-it-works, embedding models, future improvements)
- Demo GIF re-recorded with full flow (add → search → add URL → ask → stats)

### Fixed
- **CLI scoring passthrough** — `search`, `ask`, and `chat` now pass `scoring=cfg.scoring` to `store.search()` (was silently disabled for all CLI users)
- **access_count default 0** — ingestion is not an access; chunks start at 0 instead of 1
- **Capped frequency score** — normalized to [0,1] via `log1p(freq) / log1p(100)` to prevent heavily-accessed chunks from dominating semantic relevance
- **Type safety** — `scoring` param typed as `ScoringConfig | None` instead of `object`
- **Config validation** — `model_validator` enforcing `alpha + beta <= 1.0`
- **track_access logging** — failures now log at DEBUG instead of silent `pass`
- **last_accessed_at initialized** on chunk insert to avoid NULL propagation

---

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
  - Cold start: new chunks get `access_count = 0` (fixed in v0.5.2 — ingestion is not an access)
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
