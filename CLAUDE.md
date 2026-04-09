# CLAUDE.md — vstash project guide

## What is vstash?

Local-first document memory with instant semantic search. Single SQLite file, zero cloud dependencies for search. Hybrid retrieval (vector + FTS5 + adaptive RRF + MMR dedup).

## Quick reference

```bash
# Run tests
python -m pytest tests/ -x -q

# Lint + format
ruff check . && ruff format --check .

# Install locally
pip install -e .

# Verify
vstash --version
vstash stats
```

## Project structure

```
vstash/
  __init__.py       # version (__version__ = "0.27.0")
  cli.py            # typer CLI — add, search, ask, chat, list, stats, forget, reindex, watch, config, export, remember, profile, journal
  profile.py        # Multi-profile management: resolution chain, CRUD, federated search
  journal.py        # Cross-session memory: save, recall, log, prune, transcript parsing
  store.py          # VstashStore — SQLite + sqlite-vec + FTS5, RRF, scoring, MMR dedup, reindex
  ingest.py         # parse → chunk → embed pipeline
  code_split.py     # hybrid code splitting: tree-sitter → parso → regex (25+ languages)
  embed.py          # FastEmbed ONNX + MLX backends, model registry (English + multilingual)
  config.py         # Pydantic v2 config from vstash.toml, all defaults
  chat.py           # LLM chat/ask with retrieval context
  mcp.py            # MCP server for Claude Desktop
  memory.py         # Python SDK (from vstash import Memory)
  langchain.py      # VstashRetriever for LangChain
  models.py         # Pydantic models (SearchResult, DocumentInfo, StoreStats, IngestResult)
  watch.py          # File watcher for auto-ingestion and deletion handling

tests/
  test_store.py     # Store CRUD, search, MMR dedup, reindex, scoring, context expansion
  test_ingest.py    # Ingestion, chunking
  test_code_split.py  # Hybrid code splitting backends (tree-sitter / parso / regex)
  test_code_chunking.py # Code-aware chunking integration
  test_cli_commands.py  # CLI command tests
  test_scoring_e2e.py   # End-to-end scoring scenarios
  test_snapvec_backend.py # Optional snapvec backend tests
  test_robustness.py  # Expand context isolation, reindex safety, scoring edge cases
  test_watch_e2e.py   # Watch mode e2e: create, modify, delete, debounce, shutdown
  test_retry_e2e.py   # Retry with backoff e2e tests
  test_url_titles_e2e.py # URL title extraction e2e tests
  test_profile.py   # Multi-profile resolution, management, federated search, CLI, SDK
  test_journal.py   # Journal save/recall/log/prune, CLI, SDK, MCP, transcript parsing
  test_get_chunk.py # Direct chunk retrieval by ID (store, SDK, MCP, edge cases)
  test_store_helpers.py # Store helper methods (get_document_chunks, added_at, batching)
  conftest.py       # Fixtures (tmp_db_path, sample_store)

experiments/        # Research experiment scripts + results
paper/              # Academic paper (vstash-paper.md)
docs/               # User-facing documentation
```

## Key architecture decisions

- **Hybrid search**: Vector (sqlite-vec cosine) + FTS5 keyword, combined via Reciprocal Rank Fusion (k=60).
- **Adaptive RRF**: IDF-based weight adjustment per query. Rare terms boost FTS; common terms boost vector. Long queries (>50 words) relax distance cutoff. Cached via fts5vocab.
- **Pipeline**: vector search → FTS5 keyword → adaptive RRF fusion (IDF-weighted) → optional recency boost → MMR dedup. Frequency+decay scoring was evaluated and removed in v0.18.0. Replaced by opt-in recency boost in v0.19.0.
- **MMR dedup**: Intra-document Maximal Marginal Relevance replaces hard per-document dedup. `mmr_lambda=0.5` (fixed).
- **Embeddings**: FastEmbed (ONNX) or MLX (Apple Silicon). Default `BAAI/bge-small-en-v1.5` (384 dims). Multilingual models available via `vstash reindex`.
- **Code-aware chunking**: Hybrid 3-tier splitting — tree-sitter AST (25+ languages, optional) → parso AST (Python) → regex (6 languages). Graceful degradation. See `code_split.py`.
- **Single SQLite file**: WAL mode, foreign keys, all data in one `.db`.
- **Optional snapvec backend**: Compressed ANN via PolarQuant. Opt-in with `storage.vector_backend = "snapvec"`. sqlite-vec stays default.
- **Local-first LLM**: Default backend `"local"` auto-detects Ollama, LM Studio, or any OpenAI-compatible local server.
- **BEIR benchmark results**: NDCG@10=0.7263 on SciFact with adaptive RRF (surpasses ColBERTv2 0.693). Wins 5/5 BEIR datasets vs BM25, 4/5 vs ColBERTv2.
- **Integrity & recovery (v0.24)**: `doc_completeness(path, collection)` → idempotent ingest; `integrity_check()` runs 5 invariants (chunk_count parity, vec/snapvec parity, FTS5 built-in `integrity-check`, orphans, PRAGMA) and returns a `list[IntegrityCheck]`; `integrity_repair()` is profile-scoped (rebuilds `fts_chunks`, recomputes chunk_count, deletes orphans). v0.24.1 made the **partial-ingest recovery path** (via `delete_document(path, collection=...)`) collection-scoped so repairing one collection cannot wipe a sibling collection's copy of the same path. Exposed as `vstash check [--repair] [--json]`.
- **Explicit contracts & schema versioning (v0.25)**: `SCHEMA_VERSION` + `KNOWN_SCHEMA_VERSIONS` stamped in the `store_meta` table; `SchemaVersionError` on unknown versions; `INSERT OR IGNORE` for concurrent fresh-open; forward-compatible top-level config keys (warn-on-unknown). `SearchResult.score` is the RRF score with `k=60`, range `[0, ~0.033]`, comparable within a query but **not across queries**.
- **Operational observability (v0.21–v0.22)**: in-process metrics registry, slow query log, `miss_analysis(query_embedding, query_text, *, expected_path=...)` API for ranking debugging (traces where a chunk was eliminated + rule-based suggestions).
- **Explicit limits (v0.23)**: `vstash/validation.py` + `[limits]` section with 7 knobs and a `LimitError(ValueError)` hierarchy. Rejects pathological inputs at `VstashStore`/`Memory` boundaries before they hit SQLite/sqlite-vec/ONNX.
- **Threading hardening (v0.20)**: `sqlite3.threadsafety > 0` asserted at module import time; STEM (FTS5 Porter stemming) connections can be closed from any thread.

## Conventions

- **Python 3.10+** with `from __future__ import annotations`
- **Pydantic v2** for all config and data models (frozen=True)
- **Type hints** on all public functions
- **ruff** for linting and formatting (enforced in CI)
- **pytest** for testing (~750 tests + 6 benchmark regression tests as of v0.27.0)
- **Conventional commits** with emoji prefixes (feat, fix, docs, chore, perf)
- **No AI co-author lines** — do NOT add `Co-Authored-By` or any AI attribution to commit messages

## Database schema

```sql
documents (id TEXT PK, path, title, source_type, collection, project, layer, tags, char_count, chunk_count, added_at)
chunks (id INTEGER PK, doc_id FK, seq, text, access_count, created_at, last_accessed_at)
vec_chunks USING vec0(embedding float[384])   -- virtual table, sqlite-vec
fts_chunks USING fts5(text, content=chunks)   -- virtual table, FTS5
search_stats (id, spread, created_at)
search_events (id, query, best_distance, relevance_tier, result_count, dismissed, created_at)
```

## Config file

`vstash.toml` — resolution order: `$VSTASH_CONFIG` → `./vstash.toml` → `~/.vstash/vstash.toml` → defaults.

Key sections: `[inference]`, `[cerebras]`, `[ollama]`, `[openai]`, `[embeddings]`, `[chunking]`, `[scoring]`, `[storage]`.

## Common tasks

- **Add a new CLI command**: Add `@app.command()` function in `cli.py`, update docstring at top of file.
- **Add a new embedding model**: Add to `KNOWN_DIMS` and optionally `_MLX_MODEL_MAP` in `embed.py`.
- **Change search behavior**: Modify `VstashStore.search()` in `store.py`. The pipeline is: vector search → FTS search → RRF merge → scoring → MMR dedup → context expansion.
- **Add a config field**: Add to the relevant Pydantic model in `config.py`, update `docs/configuration.md`.
- **Run experiments**: `python -m experiments.<name>` (e.g., `experiments.scoring_grid`).
- **Run Kaggle-scale benchmarks**: `python -m experiments.arxiv_retrieval_bench` (1K papers, 3 models) or `python -m experiments.dataset_discovery` (954 HuggingFace datasets, interactive mode with `--interactive`).
- **Run all experiments**: `python -m experiments.run_all`.
- **Run BEIR benchmarks**: `python -m experiments.beir_benchmark` (5 BEIR datasets, vstash vs Chroma). Datasets auto-download on first run (~330MB). Use `--datasets scifact nfcorpus` for quick runs, `--no-chroma` to skip Chroma comparison.

## Branching strategy

```
feature/* ──→ develop ──→ main (via release PR)
   │              │            │
   │              │            └── protected, = PyPI published version
   │              └── integration branch, all features merge here
   └── short-lived, one feature per branch
```

- **Feature branches**: branch from `develop`, PR back to `develop`.
- **develop**: integration branch. All feature PRs target `develop`.
- **main**: protected. Only updated via release PRs from `develop`. Always matches the latest PyPI version.
- **Release flow**: when `develop` is ready, create a version bump PR from `develop` → `main`, then publish to PyPI and create a GitHub release.
- **NEVER merge features directly to `main`** — this causes `develop` to fall behind and creates conflicts.

## CI

GitHub Actions: `lint` (ruff check + format) + `test` (pytest on Python 3.10, 3.11, 3.12). All must pass before merge.

## Publishing

```bash
# 1. Bump version on develop
# 2. Create PR: develop → main
# 3. Merge PR, then from main:
python -m build
python -m twine upload dist/vstash-<version>*
gh release create v<version> --title "v<version>" --notes "..."
```
