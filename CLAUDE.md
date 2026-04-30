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
  __init__.py       # version (__version__ = "0.36.0")
  cli.py            # typer CLI — add, search, ask, chat, list, stats, forget, reindex, watch, config, export, remember, profile, journal, retrain, serve, check, snapvec
  retrain.py        # Eval-gated self-supervised embedding fine-tuning. Composes split_corpus_for_eval, evaluate_model, generate_triples, train_mnrl. Refuses to save a model that regresses on held-out NDCG@10.
  retrain_synth.py  # LLM query synthesis for retrain training pairs (OpenAI-compat, Ollama). JSONL cache keyed by (chunk_id, prompt_hash, model).
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
  test_retrain.py   # generate_triples + evaluate_model + retrain() composer (eval gate, atomic promote, synth-queries integration)
  test_retrain_synth.py # LLM query synthesis: prompt builder, parser edge cases, JSONL cache, LLM-failure resilience
  conftest.py       # Fixtures (tmp_db_path, sample_store, populated_store -- the workhorse for most new tests)

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
- **Optional snapvec backends**: Compressed ANN via PolarQuant. Two variants: `snapvec` (flat quantized) and `snapvec-ivfpq` (IVFPQ with fp16 rerank, pareto-dominant over sqlite-vec at N >= 50K). Opt-in with `storage.vector_backend`; sqlite-vec stays default. Fit via `vstash snapvec fit` before use.
- **Local-first LLM**: Default backend `"local"` auto-detects Ollama, LM Studio, or any OpenAI-compatible local server.
- **BEIR benchmark results**: Two tracks worth keeping straight. Baseline BGE-small + adaptive RRF hits NDCG@10=0.7263 on SciFact, winning 5/5 BEIR vs BM25 and 4/5 vs ColBERTv2. Tuned fine-tunes (all 33M params, 384d, published on HF): `Stffens/bge-small-rrf-v2` (+5% SciFact / +18% NFCorpus) and the newer `Stffens/bge-small-rrf-v3` (2026-04-19, H-R9 winning config `temperature=0.5 + total_triples=60000`; +5.35% macro NDCG@10 on SciFact+NFCorpus+FiQA vs base). Recommend v3 as default; v2 still valid. The bench script lives at `experiments/beir_benchmark.py` with `--no-chroma` as the recommended default.
- **Query LRU cache (v0.31)**: Opt-in via `[cache] query_cache_size`. ~700x speedup on cache hits for repeated queries. Automatically invalidated on any write. Skipped for `explain=True` and `miss_analysis()`.
- **Batched ingest + deferred FTS (v0.31)**: `add_documents_batch` + `batch_mode(defer_fts=True)` collect FTS5 inserts in memory and flush in one bulk pass on exit. 5x speedup on `ingest_directory` at 500 docs.
- **Embedder daemon (v0.32)**: `vstash serve --warm` pre-loads the embedding model and exposes `/api/embed` on localhost:8585. CLI and SDK clients auto-detect a running daemon and delegate. Drops cold start from ~2 s to ~5 ms. Fallback is transparent local embedding. Override via `VSTASH_EMBED_URL`.
- **`retrieval_mode` enum (v0.33)**: `Literal["hybrid", "vec_only", "fts_only"] = "hybrid"` on `VstashStore.search`, `Memory.search`, `Memory.ask`, MCP tools, and `VstashRetriever`. `vec_only` is the new symmetric branch to `fts_only` (skip FTS5, force `(1.0, 0.0)` weights). Default stays `hybrid` -- paper / README / v3 model numbers all measured against it. Legacy `fts_only=True` bool is deprecated in v0.33.0 with a `DeprecationWarning`.
- **`chat.ask_full()` returning `AskResult` (v0.36)** (#303/#310): Public API that surfaces the reasoning channel and token usage that `ask()` discards. `_ask_cerebras` / `_ask_ollama` / `_ask_openai` now return `AskResult` internally; `ask()` is a thin wrapper returning `.content` so the existing `-> str` contract is preserved with zero call-site changes. Cerebras `gpt-oss-120b` populates `message.reasoning`; Ollama qwen3 thinking-mode uses `message.thinking`; OpenAI-compat servers (vLLM, DeepSeek, Together, xAI Grok, OpenAI o1/o3) read `message.reasoning_content`. Shared helpers `_extract_reasoning` (accepts both field names) and `_normalize_usage` (returns complete dict or `None`, never partial). `Memory.ask_full()` mirrors `Memory.ask()`; `Memory.ask` itself routes through `ask_full(...).content` so retrieval / LLM plumbing live in a single load-bearing path. Drives Merken Phase 2 distillation (`Q -> reasoning_trace -> A` shape).
- **Centralized store construction (v0.36)** (#297/#306): `vstash._store_open.open_store_for_config(cfg)` is the single entry point used by CLI, MCP, web, SDK, journal, and `federated_search`. Replaces the previous per-surface `VstashStore(...)` wiring that silently dropped IVFPQ tuning fields on some paths.
- **`vec_only` long-query distance cutoff (v0.36)** (#304): `retrieval_mode="vec_only"` now applies the same long-query distance-cutoff relaxation as `hybrid`. Previously it forced `adaptive_rrf=False` and skipped the relaxation; ArguAna `vec_only` had been collapsing to NDCG@10 = 0.0013 (1403/1406 zero) and is now 0.4250. Hybrid mode and all paper / model-card numbers untouched.
- **`vstash why` miss analysis (v0.33)**: CLI + `/debug/why` HTTP route + auto-logged `miss_hint` on empty / low-relevance searches. `vstash why "<q>" --expect <path>` traces where a target chunk was eliminated in the pipeline (vector pool, distance cutoff, FTS match, RRF fusion, MMR, context expansion) and suggests the parameter that would have surfaced it. `vstash why --recent` lists recent misses from the auto-log.
- **Eval-gated retrain (v0.33 complete: T1.1 / T1.3 / T1.4 / T1.5 / H-R1 / H-R5 / H-R7 / H-R8)**: `retrain()` composes `split_corpus_for_eval` + `evaluate_model` + `generate_triples` + `train_mnrl` + atomic `.candidate`/`.old` promote. Refuses to save a candidate whose held-out NDCG@10 is worse than the baseline. `qrels_to_eval_queries` converts BEIR-style labels. Multi-corpus training with temperature sampling (T1.4), GPU-batched mining + eval (T1.4b/c), labeled-query mining from BEIR qrels (T1.5), auto-promoted eval queries (H-R1), bulk mining on single-corpus (H-R8), full eval observability + seed reproducibility (H-R5/H-R7). Validated in Colab at +5.00% macro NDCG@10. v3 model on HF as `Stffens/bge-small-rrf-v3`.
- **HF ONNX fallback (v0.32+)**: `_init_hf_onnx` wraps ort init in a broad except; on failure the model gets routed through `SentenceTransformer` for safetensors loading. Added because `Stffens/bge-small-rrf-v2` shipped an ONNX stub referencing an external data file that was never uploaded to the repo. The fallback is cached per-model so we only take the ONNX-path hit once.
- **O(N^2) audit closed in v0.33** (issue #252 Tier 1): three ingest-path quadratics fixed in one cycle. Flat snapvec `_save_snapvec` now deferred to `close()` (#250/#251), `_rebuild_snapvec_from_vec_chunks` rewritten with keyset pagination + coalesced `add_batch` (#264, 10.3x at N=100k), `store.reindex` shape fix (#267, 3.05x -> 2.28x at N=200k), CLI snapvec leak closed via atexit (#269), watch burst ingest 4-5x via drain-window batching (#274). Regression witnesses in `experiments/perf_watch_burst.py` and related probes.
- **`vec_chunks` cosine metric + schema v2 (v0.34)** (#271/#272/#286): sqlite-vec defaults to L2, but vstash labelled the value "cosine distance" everywhere. Worked accidentally for BGE unit-normalized embeddings; broke for `paraphrase-multilingual` and any model where L2 exceeds 2.0. DDL now includes `distance_metric=cosine`; v1 DBs migrate in place on open via atomic (`BEGIN IMMEDIATE`) + idempotent (`sqlite_master` guard) + streaming-in-SQL (`TEMP` backup, no Python materialisation) rebuild. `relevance_tier` thresholds rescaled to cosine (0.4513/0.4802) and `distance_cutoff` defaults squared (`1.15 -> 1.3225`, long-query `5.0 -> 25.0`) so BGE keeps identical behaviour. BEIR 5-dataset regression gate passed. Validation probes in `experiments/probe_272_*.py` + `experiments/beir_272_cosine_validation_colab.ipynb`.
- **Custom encoder resolver hook (v0.34)** (#278/#287/#288): `register_encoder_resolver(fn)` / `unregister_encoder_resolver` + an `Encoder` `Protocol` let callers plug in LoRA-adapted, locally fine-tuned, or otherwise-unnamable encoders. Consulted before every built-in path (daemon, Gemma, HF ONNX, MLX, FastEmbed); identity-based registration; shape + protocol validation on resolver output; `ValueError` with offending index when a custom encoder returns the wrong batch size or dim. Docstring carries a SentenceTransformer adapter recipe (ST uses `get_sentence_embedding_dimension()`).
- **Flat snapvec similarity-to-distance fix (v0.34)** (#289/#290): `SnapIndex.search` returns similarity in `[-1, 1]` but the store was feeding it straight into `distance_cutoff` / `relevance_tier` / `last_best_distance`. Ranking worked by accident, but the cutoff was effectively a no-op and perfect matches reported as `"low"` relevance. Per-backend conversion at the call site; `[0, 2]` clamp; sibling `snapvec-ivfpq` backend already did the right thing internally and is not double-inverted.
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
- **pytest** for testing (900+ tests as of v0.35.0, benchmark regression tests marker-gated off by default)
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
- **Run BEIR benchmarks**: `python -m experiments.beir_benchmark --no-chroma` (recommended default, 5 BEIR datasets, vstash only). Drop `--no-chroma` to add a vstash-vs-Chroma comparison. Datasets auto-download on first run (~330 MB). `--datasets scifact nfcorpus` for a quick subset. Pipeline latency bench: `python -m experiments.vstash_pipeline_ivfpq_bench --n 100000` compares sqlite-vec vs snapvec-ivfpq end-to-end with multi-dataset padding.

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

PyPI publication is automated via OIDC trusted publisher in
`.github/workflows/publish.yml`. The workflow runs on release
publication; do not run `twine` manually. End-to-end flow:

```bash
# 1. Bump __version__ on develop (chore/0.X-housekeeping PR usually
#    bundles version bump + README highlights + CLAUDE.md updates).
# 2. Create release PR: develop -> main, merge with --merge (not
#    squash) so individual feature commits stay visible in main's log.
# 3. From main, publish a GitHub release. This triggers publish.yml,
#    which runs `python -m build` and uploads to PyPI via OIDC:
gh release create v<version> --target main \
    --title "v<version>" --notes "..."
# 4. Open a sync PR `sync/main-to-develop-v<version>` to mirror main
#    back into develop. Merge with --admin if branch protection is
#    sticky from old commits.
```

The `pypi` environment in `publish.yml` is wired to a PyPI trusted
publisher; no API tokens are stored in the repo. If you ever need to
fall back to a manual upload, install `build` + `twine` locally and
read the password from `~/.pypirc`, but this should be a break-glass
path.
