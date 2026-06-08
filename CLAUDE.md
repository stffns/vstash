# CLAUDE.md -- vstash project guide

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
  __init__.py        # version (__version__ = "0.38.1")
  cli/               # typer CLI as a package (#284 split cli.py into command groups). __init__.py is a ~60-LOC facade re-exporting `app` (entry point vstash.cli:app) + main(); _app.py holds the shared Typer app/callback/helpers; commands live in _ingest/_search/_inspect/_manage/_retrain + the _profile/_snapvec/_journal sub-typers
  _store_open.py     # open_store_for_config(cfg): single VstashStore construction entry point used by every adapter (CLI/MCP/web/SDK/journal/federated)
  services/          # Adapter-agnostic service layer. All four adapters (web/MCP/CLI/SDK) route through here.
    search.py        # validate -> embed -> search -> expand triplet shared by every search caller
    ask.py           # ask/ask_full orchestration with retrieval context
  vectorbackend/     # Vector backend Protocol + concrete impls
    base.py          # VectorBackend Protocol contract (runtime_checkable)
    snapvec_ivfpq.py # IVFPQ backend extracted from legacy _ivfpq_backend.py
  _ivfpq_backend.py  # Deprecated re-export shim, scheduled removal v0.40
  errors.py          # VstashError ancestor + LimitError / SchemaVersionError. Multi-inherits ValueError/RuntimeError so legacy except callers keep working.
  validation.py      # API-boundary input validation, [limits] section knobs, LimitError hierarchy
  metrics.py         # In-process metrics registry, slow query log
  store/             # VstashStore as a package (#280 split the 5100-LOC store.py into mixins). __init__.py = facade (VstashStore + re-export hub) + CRUD; _common.py (constants/pure fns/_PipelineTracer), _schema.py, _index.py (snapvec/IVFPQ), _search.py (search/RRF/MMR/IDF/cache), _integrity.py
  ingest.py          # parse -> chunk -> embed pipeline
  code_split.py      # hybrid code splitting: tree-sitter -> parso -> regex (25+ languages)
  embed.py           # FastEmbed ONNX + MLX backends + built-in model registry (English + multilingual) + custom encoder resolver hook
  config.py          # Pydantic v2 config from vstash.toml, all defaults
  chat.py            # LLM chat/ask/ask_full with retrieval context. Returns AskResult (content + reasoning + usage + backend + model).
  mcp.py             # MCP server for Claude Desktop (vstash_search/vstash_ask route through services/)
  web.py             # FastAPI server: search, ask, upload, /debug/why
  memory.py          # Python SDK (from vstash import Memory). search/ask/ask_full route through services/.
  langchain.py       # VstashRetriever for LangChain
  models.py          # Pydantic models (SearchResult, AskResult, DocumentInfo, StoreStats, IngestResult)
  profile.py         # Multi-profile management: resolution chain, CRUD, federated search
  journal.py         # Cross-session memory: save, recall, log, prune, transcript parsing
  retrain.py         # Eval-gated self-supervised embedding fine-tuning. Composes split_corpus_for_eval, evaluate_model, generate_triples, train_mnrl. Refuses to save a model that regresses on held-out NDCG@10.
  retrain_batch.py   # Bulk training-pair mining helpers
  retrain_synth.py   # LLM query synthesis for retrain training pairs (OpenAI-compat, Ollama). JSONL cache keyed by (chunk_id, prompt_hash, model).
  watch.py           # File watcher for auto-ingestion and deletion handling

tests/               # 1200+ tests across 50+ files. Highlights below; see tests/ for the full set.
  conftest.py                # Fixtures (tmp_db_path, sample_store, populated_store: workhorse for most new tests)
  test_store.py              # Store CRUD, search, MMR dedup, reindex, scoring, context expansion
  test_store_helpers.py      # Store helper methods (get_document_chunks, added_at, batching)
  test_store_open.py         # open_store_for_config end-to-end
  test_services.py           # services.search / services.ask shared validation + plumbing
  test_vectorbackend.py      # VectorBackend Protocol conformance
  test_snapvec_backend.py    # Flat snapvec backend
  test_snapvec_ivfpq_backend.py # IVFPQ backend, fp16 rerank, stale-load detection
  test_builtin_backend_registry.py # Embed-side built-in model registry
  test_encoder_resolver.py   # Custom encoder resolver hook
  test_embed.py / test_embed_daemon.py / test_embed_mlx.py
  test_retrieval_mode.py     # hybrid / vec_only / fts_only enum
  test_query_cache.py        # LRU query cache invalidation
  test_deferred_fts.py / test_ingest_batch.py # Batched ingest + deferred FTS
  test_integrity.py / test_schema_versioning.py
  test_validation.py / test_errors.py / test_metrics.py
  test_chat.py / test_memory.py / test_mcp.py / test_web.py / test_langchain.py
  test_cli_commands.py / test_cli_error_messages.py / test_cli_get_store_lifecycle.py / test_cli_retrain.py
  test_retrain.py / test_retrain_batch.py / test_retrain_multi.py / test_retrain_synth.py
  test_code_split.py / test_code_chunking.py
  test_watch.py / test_watch_e2e.py / test_watch_burst.py
  test_ingest.py / test_export.py / test_frontmatter.py / test_remember.py / test_get_chunk.py
  test_profile.py / test_journal.py / test_config.py
  test_miss_analysis.py / test_robustness.py / test_stem_conn_cleanup.py
  test_retry_e2e.py / test_url_titles_e2e.py / test_beir_regression.py

experiments/        # Research experiment scripts + results (BEIR, scoring grids, perf probes)
paper/              # Academic paper (v1 in paper/, v2 in paper/v2/, arXiv build in paper/arxiv/)
docs/               # User-facing documentation
```

## Key architecture decisions

- **Hybrid search**: Vector (sqlite-vec cosine) + FTS5 keyword, combined via Reciprocal Rank Fusion (k=60).
- **Adaptive RRF**: IDF-based weight adjustment per query. Rare terms boost FTS; common terms boost vector. Long queries (>50 words) relax distance cutoff. Cached via fts5vocab.
- **Pipeline**: vector search -> FTS5 keyword -> adaptive RRF fusion (IDF-weighted) -> optional recency boost -> MMR dedup. Frequency+decay scoring was evaluated and removed in v0.18.0. Replaced by opt-in recency boost in v0.19.0.
- **MMR dedup**: Intra-document Maximal Marginal Relevance replaces hard per-document dedup. `mmr_lambda=0.5` (fixed).
- **Embeddings**: FastEmbed (ONNX) or MLX (Apple Silicon). Default `BAAI/bge-small-en-v1.5` (384 dims). Multilingual models available via `vstash reindex`.
- **Code-aware chunking**: Hybrid 3-tier splitting: tree-sitter AST (25+ languages, optional) -> parso AST (Python) -> regex (6 languages). Graceful degradation. See `code_split.py`.
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
- **Services layer (post-v0.36)** (#327/#334/#335/#336): `vstash/services/{search,ask}.py` centralises the `validate -> embed -> search -> expand` triplet that the four adapters (web, MCP, CLI, SDK) all duplicated. Every adapter now routes through `services/`, so validation runs at the API boundary before the daemon round-trip and there is one load-bearing path for retrieval/LLM plumbing. When changing search behaviour, edit `services/search.py` first; `VstashStore.search` remains the underlying primitive.
- **Domain error tree (post-v0.36)** (#326): `vstash/errors.py` defines `VstashError` as the ancestor for `LimitError` (boundary validation), `SchemaVersionError`, and friends. Each leaf multi-inherits the historical `ValueError` / `RuntimeError` it used to raise, so existing `except ValueError` callers keep working without changes.
- **VectorBackend Protocol (post-v0.36)** (#328): `vstash/vectorbackend/base.py` defines the `VectorBackend` `Protocol` (runtime-checkable). `snapvec_ivfpq.py` is the first concrete impl extracted from legacy `_ivfpq_backend.py`; the legacy module is a deprecated re-export shim slated for removal in v0.40. Use the Protocol contract when adding a new backend.
- **Built-in embed model registry (post-v0.36)** (#330): replaces the if/elif dispatch in `embed_texts` / `embed_query` / `warmup`. New built-in models register themselves in the registry; the custom encoder resolver hook (#278) still wins over built-ins for non-stock models.
- **`vec_only` long-query distance cutoff (v0.36)** (#304): `retrieval_mode="vec_only"` now applies the same long-query distance-cutoff relaxation as `hybrid`. Previously it forced `adaptive_rrf=False` and skipped the relaxation; ArguAna `vec_only` had been collapsing to NDCG@10 = 0.0013 (1403/1406 zero) and is now 0.4250. Hybrid mode and all paper / model-card numbers untouched.
- **MMR dedup swap-pop + pre-grouped siblings (v0.37)** (#363, supersedes #351): `_mmr_dedup`'s greedy loop now removes selected candidates via swap-with-last + `in_remaining` mask (O(1) per pick) and walks the new selection's same-doc siblings via a pre-built `doc_to_indices` map (O(S_avg) per pick) instead of scanning all `remaining` and filtering on `doc_keys[idx]` (O(N)). Honest end-to-end speedup on `store.search()` with real BGE-small embeddings is 1.15x-1.19x across docs=200..1000, top_k=10..100. The isolated synthetic micro-probe shows ~1.00x because CPython overhead (`enumerate` tuples, Python-level swap-pop vs C-level `list.remove`) eats the algorithmic save when the penalty loop is benchmarked alone; the gain only materialises end-to-end. Both probes (`experiments/perf_mmr_dedup.py`, `experiments/perf_mmr_dedup_real.py`) are kept so future perf claims for this hot path have to defend against both. Tie-break on smaller original `idx` preserves the pre-rewrite selection ordering.
- **Tag filters in search + journal_recall (v0.37)** (#106 partial, #364): `tags: str | list[str] | None` exposed across `VstashStore.search`, `services/search.search_with_embedding`, `Memory.search`, `Memory.journal_recall`, `journal.journal_recall`, `federated_search`, MCP `vstash_search`/`vstash_journal_recall`, and CLI `vstash search`/`vstash journal recall` (repeatable `--tag` flag plus `--after`/`--before` on the journal side). Multiple tags use OR semantics. Matching is **comma-anchored** (`','||tags||',' LIKE '%,foo,%'`) so `tag='alpha'` does NOT false-match `alphabet`. The module-level `vstash.store._normalize_tags` helper accepts a comma-separated string OR a list (each list element also split on commas, e.g. `["alpha,beta"]` -> `["alpha", "beta"]`), dedupes preserving order, and is reused on the write path (`add_document`, `add_documents_batch`, `update_metadata`) so stored tags are always canonical `"a,b,c"` with no whitespace. The cache key sorts tags before hashing so `["a","b"]` and `["b","a"]` share a slot.
- **`Memory.update()` in-place document mutation (v0.37)** (#365): explicit update API across SDK / CLI / MCP. Two modes picked from kwargs: **metadata-only** (`title`/`tags`) runs a single atomic SQL `UPDATE` via the new `VstashStore.update_metadata(path, *, title=, tags=, collection=)` primitive -- no re-chunking, no embed-pipeline invocation; **content** (`text`) chunks + embeds the caller-supplied text and replaces every chunk while preserving `source_type`/`collection`/`project`/`layer`/non-overridden metadata. Content path queries the existing rows via direct `SELECT ... WHERE path = ? AND collection = ?` (O(1) on `idx_documents_path`) and re-adds chunks into *every* matching collection so a `collection=None` content refresh does not silently collapse a multi-collection doc into one. Empty call raises `ValueError`; not-found and noop return structured dicts. CLI surfaces are `vstash update <path> [--text|--title|--tags]` (`--text -` for stdin) and MCP `vstash_update`.
- **`Memory.prune()` + `Memory.compact()` + `vstash compact` CLI (v0.37)** (#366): three new housekeeping primitives in `VstashStore` -- `prune_documents(*, before_iso=, collection=, project=, layer=, tags=, dry_run=)` (deletes everything matching the filter; requires at least one filter and rejects an unfiltered call with `ValueError` so the "wipe everything" foot-gun is opt-in; SELECT runs inside `BEGIN IMMEDIATE` so the reported `paths`/`deleted` count is atomic with the delete), `vacuum()` (SQLite VACUUM outside any transaction), `optimize_fts()` (FTS5 `'optimize'`). The SDK exposes `Memory.prune(*, before=...)` and `Memory.compact(*, before=None, vacuum=True, optimize_fts=True, dry_run=False)`; `before` accepts an age string (`"30d"`/`"2w"`/`"24h"`, shared parser with `vstash journal prune`) or an ISO date/timestamp. ISO inputs are validated via `datetime.fromisoformat` and **canonicalised to UTC** before reaching the store -- a non-UTC offset would lexically diverge from the chronological order of `added_at` (stored as `+00:00`) and silently delete the wrong rows around timezone boundaries. CLI is `vstash compact [--before AGE_OR_ISO] [--collection ...] [--project ...] [--layer ...] [--no-vacuum] [--no-optimize-fts] [--dry-run] [--json]`; warnings route to stderr when `--json` is set so the stdout payload stays pure JSON. MCP exposes `vstash_compact`.
- **`store.py` + `cli.py` decomposed into packages (v0.38)** (#280, #284): both god-modules split into focused mixin/command-group modules with byte-identical move-only PRs. `store.py` (5104 LOC) -> `vstash/store/` (facade `__init__.py` re-export hub + `_common`/`_schema`/`_index`/`_search`/`_integrity` mixins). `cli.py` (3140 LOC) -> `vstash/cli/` (57-LOC facade re-exporting `app`/`main` + `_app` shared core + `_ingest`/`_search`/`_inspect`/`_manage`/`_retrain` command groups + `_profile`/`_snapvec`/`_journal` sub-typers). Public surface preserved (`from vstash.store import X`, `vstash.cli:app`, `vstash.cli._get_store`). CLI tests that mock `_get_store`/`embed_query` route through `tests/_cli_helpers.patch_cli_attr` (per-module binding after the split).
- **v0.38 correctness / perf / cleanup cluster**: (#403) `reindex` wraps DROP/CREATE/INSERT in `BEGIN IMMEDIATE` (a failed reindex no longer wipes `vec_chunks`); `prune_documents` UTC-canonicalises `before_iso` (the store primitive is now self-protecting, not just the SDK); `_build_idf_cache` narrows `except Exception` -> `sqlite3.OperationalError` so DB corruption surfaces instead of silently disabling adaptive RRF. (#404) index `documents.added_at` (the dominant ORDER BY/range column) + the query-cache key hashes the embedding as a tuple (no per-search numpy alloc). (#405) removed the dead `_save_snapvec` no-op + the deprecated `ScoringConfig` (kept `BackendError`/`RecencyConfig` -- both intentional). (#406) MCP `vstash_ask` routes through `ask_full` so the reasoning channel + token usage reach the client. (#407) flat-snapvec rollback recovery re-runs `_init_snapvec()` (staleness check -> rebuild from `vec_chunks`) instead of restoring a stale `.snpv`, and `reindex` checkpoints snapvec immediately (best-effort, guarded).
- **CI flake hardening (v0.38)**: (#376) cache + retrying warm-up for the HuggingFace BGE-small download (was a ~60-min hang on flake). (#402) the test step retries last-failed tests in fresh `pytest` processes for the per-process tree-sitter native-lib load flake that intermittently degrades `TestExtendedLanguages` on the 3.10/3.11 runners.
- **v0.38.1 correctness cluster** (audit-driven, each verified to fail without the fix): (#411) `retrain._promote_candidate` survives a partial cross-device `shutil.move` -- tracks `backed_up`/`final_existed` and removes a partial `final_path` before restoring the backup, so a failed promotion no longer strands the prior model in `.old`; the rollback is itself guarded so it cannot mask the original error. (#412) the embedder-daemon client caches a mismatched `(url, model)` pair so a non-default model skips the full-batch HTTP round-trip instead of re-paying it every call. (#413) `ingest._sep_tokens()` makes `_split_by_paragraphs` / `_merge_small_chunks` self-contained (they no longer rely on a prior `_token_count` having set `_SEPARATOR_TOKENS` as a side effect). (#414) `vstash search` validates at the CLI boundary (`validate_search_input`) -> clean `LimitError` (JSON under `--json`) instead of a SQLite crash, matching the other adapters. (#415) the file watcher cancels a path's pending create/modify debounce on delete (`_DebounceTimer.cancel` for a file, `cancel_prefix` for a deleted directory) so a stale timer cannot re-enqueue a just-removed file; also corrected the `vstash_journal_save` docstring's false "auto-detected from cwd" claim.
- **`vstash why` miss analysis (v0.33)**: CLI + `/debug/why` HTTP route + auto-logged `miss_hint` on empty / low-relevance searches. `vstash why "<q>" --expect <path>` traces where a target chunk was eliminated in the pipeline (vector pool, distance cutoff, FTS match, RRF fusion, MMR, context expansion) and suggests the parameter that would have surfaced it. `vstash why --recent` lists recent misses from the auto-log.
- **Eval-gated retrain (v0.33 complete: T1.1 / T1.3 / T1.4 / T1.5 / H-R1 / H-R5 / H-R7 / H-R8)**: `retrain()` composes `split_corpus_for_eval` + `evaluate_model` + `generate_triples` + `train_mnrl` + atomic `.candidate`/`.old` promote. Refuses to save a candidate whose held-out NDCG@10 is worse than the baseline. `qrels_to_eval_queries` converts BEIR-style labels. Multi-corpus training with temperature sampling (T1.4), GPU-batched mining + eval (T1.4b/c), labeled-query mining from BEIR qrels (T1.5), auto-promoted eval queries (H-R1), bulk mining on single-corpus (H-R8), full eval observability + seed reproducibility (H-R5/H-R7). Validated in Colab at +5.00% macro NDCG@10. v3 model on HF as `Stffens/bge-small-rrf-v3`.
- **HF ONNX fallback (v0.32+)**: `_init_hf_onnx` wraps ort init in a broad except; on failure the model gets routed through `SentenceTransformer` for safetensors loading. Added because `Stffens/bge-small-rrf-v2` shipped an ONNX stub referencing an external data file that was never uploaded to the repo. The fallback is cached per-model so we only take the ONNX-path hit once.
- **O(N^2) audit closed in v0.33** (issue #252 Tier 1): three ingest-path quadratics fixed in one cycle. Flat snapvec `_save_snapvec` now deferred to `close()` (#250/#251), `_rebuild_snapvec_from_vec_chunks` rewritten with keyset pagination + coalesced `add_batch` (#264, 10.3x at N=100k), `store.reindex` shape fix (#267, 3.05x -> 2.28x at N=200k), CLI snapvec leak closed via atexit (#269), watch burst ingest 4-5x via drain-window batching (#274). Regression witnesses in `experiments/perf_watch_burst.py` and related probes.
- **`vec_chunks` cosine metric + schema v2 (v0.34)** (#271/#272/#286): sqlite-vec defaults to L2, but vstash labelled the value "cosine distance" everywhere. Worked accidentally for BGE unit-normalized embeddings; broke for `paraphrase-multilingual` and any model where L2 exceeds 2.0. DDL now includes `distance_metric=cosine`; v1 DBs migrate in place on open via atomic (`BEGIN IMMEDIATE`) + idempotent (`sqlite_master` guard) + streaming-in-SQL (`TEMP` backup, no Python materialisation) rebuild. `relevance_tier` thresholds rescaled to cosine (0.4513/0.4802) and `distance_cutoff` defaults squared (`1.15 -> 1.3225`, long-query `5.0 -> 25.0`) so BGE keeps identical behaviour. BEIR 5-dataset regression gate passed. Validation probes in `experiments/probe_272_*.py` + `experiments/beir_272_cosine_validation_colab.ipynb`.
- **Custom encoder resolver hook (v0.34)** (#278/#287/#288): `register_encoder_resolver(fn)` / `unregister_encoder_resolver` + an `Encoder` `Protocol` let callers plug in LoRA-adapted, locally fine-tuned, or otherwise-unnamable encoders. Consulted before every built-in path (daemon, Gemma, HF ONNX, MLX, FastEmbed); identity-based registration; shape + protocol validation on resolver output; `ValueError` with offending index when a custom encoder returns the wrong batch size or dim. Docstring carries a SentenceTransformer adapter recipe (ST uses `get_sentence_embedding_dimension()`).
- **Flat snapvec similarity-to-distance fix (v0.34)** (#289/#290): `SnapIndex.search` returns similarity in `[-1, 1]` but the store was feeding it straight into `distance_cutoff` / `relevance_tier` / `last_best_distance`. Ranking worked by accident, but the cutoff was effectively a no-op and perfect matches reported as `"low"` relevance. Per-backend conversion at the call site; `[0, 2]` clamp; sibling `snapvec-ivfpq` backend already did the right thing internally and is not double-inverted.
- **Integrity & recovery (v0.24)**: `doc_completeness(path, collection)` -> idempotent ingest; `integrity_check()` runs 5 invariants (chunk_count parity, vec/snapvec parity, FTS5 built-in `integrity-check`, orphans, PRAGMA) and returns a `list[IntegrityCheck]`; `integrity_repair()` is profile-scoped (rebuilds `fts_chunks`, recomputes chunk_count, deletes orphans). v0.24.1 made the **partial-ingest recovery path** (via `delete_document(path, collection=...)`) collection-scoped so repairing one collection cannot wipe a sibling collection's copy of the same path. Exposed as `vstash check [--repair] [--json]`.
- **Explicit contracts & schema versioning (v0.25)**: `SCHEMA_VERSION` + `KNOWN_SCHEMA_VERSIONS` stamped in the `store_meta` table; `SchemaVersionError` on unknown versions; `INSERT OR IGNORE` for concurrent fresh-open; forward-compatible top-level config keys (warn-on-unknown). `SearchResult.score` is the RRF score with `k=60`, range `[0, ~0.033]`, comparable within a query but **not across queries**.
- **Operational observability (v0.21-v0.22)**: in-process metrics registry, slow query log, `miss_analysis(query_embedding, query_text, *, expected_path=...)` API for ranking debugging (traces where a chunk was eliminated + rule-based suggestions).
- **Explicit limits (v0.23)**: `vstash/validation.py` + `[limits]` section with 7 knobs and a `LimitError(ValueError)` hierarchy. Rejects pathological inputs at `VstashStore`/`Memory` boundaries before they hit SQLite/sqlite-vec/ONNX.
- **Threading hardening (v0.20)**: `sqlite3.threadsafety > 0` asserted at module import time; STEM (FTS5 Porter stemming) connections can be closed from any thread.

## Conventions

- **Python 3.10+** with `from __future__ import annotations`
- **Pydantic v2** for all config and data models (frozen=True)
- **Type hints** on all public functions
- **ruff** for linting and formatting (enforced in CI)
- **pytest** for testing (1300+ tests as of v0.38.0, benchmark regression tests marker-gated off by default)
- **Conventional commits** with emoji prefixes (feat, fix, docs, chore, perf)
- **No AI co-author lines**: do NOT add `Co-Authored-By` or any AI attribution to commit messages

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

`vstash.toml` resolution order: `$VSTASH_CONFIG` -> `./vstash.toml` -> `~/.vstash/vstash.toml` -> defaults.

Key sections: `[inference]`, `[cerebras]`, `[ollama]`, `[openai]`, `[embeddings]`, `[chunking]`, `[recency]`, `[storage]`.

## Common tasks

- **Add a new CLI command**: Add `@app.command()` function in `cli.py`, update docstring at top of file. (#284 will split this into `vstash/cli/<command>.py`; until merged, cli.py stays monolithic.)
- **Add a new embedding model**: Register in the built-in registry in `embed.py` (and optionally `_MLX_MODEL_MAP` for MLX). For non-stock fine-tunes, prefer `register_encoder_resolver(fn)` over editing the registry.
- **Change search behavior**: Edit `vstash/services/search.py` first if it is adapter-shared (validation, embedding, context expansion via `store.expand_context()`). Edit `VstashStore.search()` in `store.py` for the underlying retrieval pipeline: vector search -> FTS search -> RRF merge -> scoring -> MMR dedup.
- **Add a new vector backend**: Implement the `VectorBackend` Protocol in `vstash/vectorbackend/base.py`. See `snapvec_ivfpq.py` as the reference impl. Add a runtime-checkable conformance test mirroring `test_vectorbackend.py`.
- **Open a `VstashStore` from a new surface**: call `vstash._store_open.open_store_for_config(cfg)`. Never instantiate `VstashStore(...)` directly from an adapter.
- **Add a config field**: Add to the relevant Pydantic model in `config.py`, update `docs/configuration.md`.
- **Mutate an existing document**: `Memory.update(path, *, text=, title=, tags=)` (v0.37). Metadata-only paths (`title`/`tags`) skip the embed pipeline entirely; passing `text` re-chunks + re-embeds via `chunk_text` + `embed_texts` and preserves every other metadata field. The underlying primitive `VstashStore.update_metadata` is collection-scoped by default; pass `collection=None` to widen to every collection holding the path.
- **Prune or compact the store**: `Memory.prune(*, before=, collection=, ...)` for date/metadata-scoped deletion; `Memory.compact(*, before=, vacuum=True, optimize_fts=True)` for the full housekeeping pass. `Memory.prune` rejects fully-unfiltered calls with `ValueError`; `Memory.compact(before=None)` skips the prune phase entirely and runs only the VACUUM + FTS-optimize legs (no error). `before` accepts `"30d"`/`"2w"`/`"24h"` or ISO date/timestamp (canonicalised to UTC via `astimezone` before reaching SQLite, so lexical `added_at < ?` agrees with chronological order). CLI is `vstash compact`; MCP is `vstash_compact`.
- **Run experiments**: `python -m experiments.<name>` (e.g., `experiments.scoring_grid`).
- **Run Kaggle-scale benchmarks**: `python -m experiments.arxiv_retrieval_bench` (1K papers, 3 models) or `python -m experiments.dataset_discovery` (954 HuggingFace datasets, interactive mode with `--interactive`).
- **Run all experiments**: `python -m experiments.run_all`.
- **Run BEIR benchmarks**: `python -m experiments.beir_benchmark --no-chroma` (recommended default, 5 BEIR datasets, vstash only). Drop `--no-chroma` to add a vstash-vs-Chroma comparison. Datasets auto-download on first run (~330 MB). `--datasets scifact nfcorpus` for a quick subset. Pipeline latency bench: `python -m experiments.vstash_pipeline_ivfpq_bench --n 100000` compares sqlite-vec vs snapvec-ivfpq end-to-end with multi-dataset padding.

## Branching strategy

```
feature/* --> develop --> main (via release PR)
   |             |            |
   |             |            +-- protected, = PyPI published version
   |             +-- integration branch, all features merge here
   +-- short-lived, one feature per branch
```

- **Feature branches**: branch from `develop`, PR back to `develop`.
- **develop**: integration branch. All feature PRs target `develop`.
- **main**: protected. Only updated via release PRs from `develop`. Always matches the latest PyPI version.
- **Release flow**: when `develop` is ready, create a version bump PR from `develop` -> `main`, then publish to PyPI and create a GitHub release.
- **NEVER merge features directly to `main`**: this causes `develop` to fall behind and creates conflicts.

## CI

GitHub Actions: `lint` (ruff check + format) + `test` (pytest on Python 3.10, 3.11, 3.12). All must pass before merge.

## Publishing

PyPI publication is automated via OIDC trusted publisher in
`.github/workflows/publish.yml`. The workflow runs on release
publication; do not run `twine` manually. End-to-end flow:

```bash
# 1. Bump __version__ on develop (chore/0.X-housekeeping PR usually
#    bundles version bump + README highlights + CLAUDE.md updates).
# 2. Create release PR develop -> main and merge with --merge (not
#    squash) so individual feature commits stay visible in main's log:
gh pr create --base main --head develop --title "release: v<version>"
gh pr merge --merge
# 3. From main, publish a GitHub release. --generate-notes auto-fills
#    the body from merged PR titles. This triggers publish.yml,
#    which runs `python -m build` and uploads to PyPI via OIDC:
gh release create v<version> --target main \
    --title "v<version>" --generate-notes
# 4. Sync main back to develop. --admin lets the merge bypass sticky
#    branch protections that the release commit may carry forward:
gh pr create --base develop --head main \
    --title "sync: main to develop v<version>"
gh pr merge --merge --admin
```

The `pypi` environment in `publish.yml` is wired to a PyPI trusted
publisher; no API tokens are stored in the repo. **Do not run `twine`
manually under normal circumstances.** The only sanctioned exception is
a break-glass scenario where the OIDC workflow is unrecoverable (e.g.
PyPI trusted-publisher outage); in that case, install `build` + `twine`
locally, upload from a clean checkout of the tagged commit, and document
the manual upload in the release notes so the next release does not
inherit the workaround.
