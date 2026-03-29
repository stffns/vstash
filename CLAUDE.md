# CLAUDE.md — vstash project guide

## What is vstash?

Local-first document memory with instant semantic search. Single SQLite file, zero cloud dependencies for search. Hybrid retrieval (vector + FTS5 + RRF) with frequency/decay scoring.

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
  __init__.py       # version (__version__ = "0.8.0")
  cli.py            # typer CLI — add, search, ask, chat, list, stats, forget, reindex, watch, config, export
  store.py          # VstashStore — SQLite + sqlite-vec + FTS5, RRF, scoring, MMR dedup, reindex
  ingest.py         # parse → chunk → embed pipeline, code-aware chunking (6 languages)
  embed.py          # FastEmbed ONNX + MLX backends, model registry (English + multilingual)
  config.py         # Pydantic v2 config from vstash.toml, all defaults
  chat.py           # LLM chat/ask with retrieval context
  mcp.py            # MCP server for Claude Desktop
  memory.py         # Python SDK (from vstash import Memory)
  langchain.py      # VstashRetriever for LangChain
  models.py         # Pydantic models (SearchResult, DocumentInfo, StoreStats, IngestResult)
  watch.py          # File watcher for auto-ingestion

tests/
  test_store.py     # Store CRUD, search, MMR dedup, reindex, scoring, context expansion
  test_ingest.py    # Ingestion, chunking, code-aware splitting
  test_cli_commands.py  # CLI command tests
  test_scoring_e2e.py   # End-to-end scoring scenarios
  conftest.py       # Fixtures (tmp_db_path, sample_store)

experiments/        # Research experiment scripts + results
paper/              # Academic paper (vstash-paper.md)
docs/               # User-facing documentation
```

## Key architecture decisions

- **Hybrid search**: Vector (sqlite-vec cosine) + FTS5 keyword, combined via Reciprocal Rank Fusion (k=60).
- **Scoring**: Post-RRF re-ranking with frequency + temporal decay. Adaptive maturity gate (γ) suppresses scoring until access patterns show outlier signal.
- **MMR dedup**: Intra-document Maximal Marginal Relevance replaces hard per-document dedup. `mmr_lambda=0.5` default, configurable.
- **Embeddings**: FastEmbed (ONNX) or MLX (Apple Silicon). Default `BAAI/bge-small-en-v1.5` (384 dims). Multilingual models available via `vstash reindex`.
- **Code-aware chunking**: Regex-based splitting at column-0 definitions for Python, JS/TS, Go, Rust, Java. 3-tier fallback: regex → paragraph → fixed-window.
- **Single SQLite file**: WAL mode, foreign keys, all data in one `.db`.

## Conventions

- **Python 3.10+** with `from __future__ import annotations`
- **Pydantic v2** for all config and data models (frozen=True)
- **Type hints** on all public functions
- **ruff** for linting and formatting (enforced in CI)
- **pytest** for testing (312 tests as of v0.8.0)
- **Conventional commits** with emoji prefixes (feat, fix, docs, chore, perf)

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

## CI

GitHub Actions: `lint` (ruff check + format) + `test` (pytest on Python 3.10, 3.11, 3.12). All must pass before merge.

## Publishing

```bash
python -m build
python -m twine upload dist/vstash-<version>*
gh release create v<version> --title "v<version>" --notes "..."
```
