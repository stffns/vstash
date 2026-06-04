# Changelog

All notable changes to vstash are documented here.

## [Unreleased]

## [0.37.0] - 2026-05-27

### Added

- **`Memory.update()` — in-place document mutation** (#365).  Explicit update API across SDK / CLI / MCP.  Metadata-only (`title` / `tags`) runs a single atomic SQL `UPDATE` via the new `VstashStore.update_metadata(path, *, title=, tags=, collection=)` primitive — no re-chunking, no embed pipeline.  Content (`text`) re-chunks + re-embeds and replaces every chunk while preserving `source_type` / `collection` / `project` / `layer` / non-overridden metadata; a `collection=None` content refresh re-adds into *every* matching collection so a multi-collection doc is not silently collapsed.  Empty call raises `ValueError`; not-found / noop return structured dicts.  CLI `vstash update <path> [--text|--title|--tags]` (`--text -` reads stdin); MCP `vstash_update`.
- **`Memory.prune()` + `Memory.compact()` + `vstash compact` CLI** (#366, #369).  Three housekeeping primitives in `VstashStore`: `prune_documents(*, before_iso=, collection=, project=, layer=, tags=, dry_run=)` (requires at least one filter — an unfiltered call raises `ValueError`, so the "wipe everything" foot-gun is opt-in; the SELECT runs inside `BEGIN IMMEDIATE` so the reported `paths` / `deleted` count is atomic with the delete), `vacuum()` (VACUUM outside any transaction), and `optimize_fts()` (FTS5 `'optimize'`).  The SDK exposes `Memory.prune(*, before=, tags=, ...)` and `Memory.compact(*, before=None, vacuum=True, optimize_fts=True, dry_run=False)`; `before` accepts an age string (`"30d"` / `"2w"` / `"24h"`, shared parser with `vstash journal prune`) or an ISO date / timestamp, canonicalised to UTC before reaching SQLite so lexical `added_at < ?` agrees with chronological order.  `Memory.compact(before=None)` skips the prune phase and runs only the VACUUM + FTS-optimize legs.  CLI `vstash compact [--before AGE_OR_ISO] [--collection ...] [--project ...] [--layer ...] [--no-vacuum] [--no-optimize-fts] [--dry-run] [--json]`; MCP `vstash_compact`.
- **Tag filters in search + date / tag filters in `journal_recall`** (#106 partial, #364).  `tags: str | list[str] | None` exposed across `VstashStore.search`, `services/search.search_with_embedding`, `Memory.search`, `Memory.journal_recall`, `journal.journal_recall`, `federated_search`, MCP `vstash_search` / `vstash_journal_recall`, and CLI `vstash search` / `vstash journal recall` (repeatable `--tag`, plus `--after` / `--before` on the journal side).  Multiple tags use OR semantics.  Matching is comma-anchored (`',' || tags || ',' LIKE '%,foo,%'`) so `tag='alpha'` does not false-match `alphabet`.  The new `vstash.store._normalize_tags` helper (accepts a comma-separated string or a list, dedupes preserving order) is reused on the write path (`add_document`, `add_documents_batch`, `update_metadata`) so stored tags are always canonical `"a,b,c"`.

### Changed

- **Services layer + Sprint 2 architecture pass** (#326, #327, #328, #330, #334, #335, #336).  `vstash/services/{search,ask}.py` centralises the `validate -> embed -> search -> expand` triplet that the web / MCP / CLI / SDK adapters each duplicated; every adapter now routes through `services/`, so validation runs at the API boundary and there is one load-bearing retrieval path.  Supporting refactors landed the same cycle: a `VstashError` domain error tree (#326) whose leaves multi-inherit the historical `ValueError` / `RuntimeError` (existing `except` callers unaffected); a runtime-checkable `VectorBackend` Protocol with IVFPQ extracted to `vstash/vectorbackend/snapvec_ivfpq.py` (#328, legacy `_ivfpq_backend.py` is now a deprecation shim slated for removal in v0.40); and a built-in embed-model registry replacing the if/elif dispatch in `embed_texts` / `embed_query` / `warmup` (#330).

### Performance

- **MMR dedup swap-pop + pre-grouped siblings** (#363, supersedes #351).  `_mmr_dedup`'s greedy loop removes selected candidates via swap-with-last + an `in_remaining` mask (O(1) per pick) and walks the new selection's same-doc siblings via a pre-built `doc_to_indices` map (O(S_avg) per pick) instead of scanning all `remaining` (O(N)).  Honest end-to-end speedup on `store.search()` with real BGE-small embeddings is 1.15x–1.19x across docs=200..1000, top_k=10..100.  Tie-break on smaller original `idx` preserves pre-rewrite selection ordering.  Both probes (`experiments/perf_mmr_dedup.py`, `perf_mmr_dedup_real.py`) are kept so future perf claims for this hot path must defend against both.

### Fixed

- **IVFPQ stale-index detection** (#329, #332).  On load the IVFPQ backend compares the `.snpi` sidecar against `vec_chunks` and downgrades to unfitted when they diverge, instead of serving a stale index.

### Infrastructure

- LICENSE file + SPDX expression (PEP 639) (#314); SECURITY policy, code of conduct, and issue / PR templates (#316); Dependabot + CodeQL security workflows (#319); coverage gate + Codecov upload + badge (#318); expanded ruff lint rule set with safe autofixes (#320); `tree-sitter-language-pack` pinned `<1.8` (#344, #345); `langchain-core` requirement update (#325); CI action bumps (#321–#324); README 50K latency claim refined to measured numbers (#333).

## [0.36.0] - 2026-04-30

### Added

- **`chat.ask_full()` returning `AskResult`** (#303, #310).  Public API that surfaces the reasoning channel and token usage `ask()` discards.  `_ask_cerebras` / `_ask_ollama` / `_ask_openai` now return `AskResult` internally; `ask()` is a thin wrapper returning `.content`, so the existing `-> str` contract is preserved with zero call-site changes.  Cerebras `gpt-oss-120b` populates `message.reasoning`; Ollama qwen3 thinking-mode uses `message.thinking`; OpenAI-compat servers (vLLM, DeepSeek, Together, xAI Grok, OpenAI o1/o3) read `message.reasoning_content`.  Shared helpers `_extract_reasoning` (accepts both field names) and `_normalize_usage` (returns a complete dict or `None`, never partial).  `Memory.ask_full()` mirrors `Memory.ask()`; `Memory.ask` itself routes through `ask_full(...).content` so retrieval / LLM plumbing live in a single path.  Drives Merken Phase 2 distillation.
- **Centralized store construction** (#297, #306).  `vstash._store_open.open_store_for_config(cfg)` is the single entry point used by CLI, MCP, web, SDK, journal, and `federated_search`, replacing the per-surface `VstashStore(...)` wiring that silently dropped IVFPQ tuning fields on some paths.

### Fixed

- **`vec_only` long-query distance cutoff** (#304).  `retrieval_mode="vec_only"` now applies the same long-query distance-cutoff relaxation as `hybrid`.  Previously it forced `adaptive_rrf=False` and skipped the relaxation; ArguAna `vec_only` had collapsed to NDCG@10 = 0.0013 (1403/1406 zero) and is now 0.4250.  Hybrid mode and all paper / model-card numbers untouched.
- **`Memory.add(collection=None)`** (#296) falls back to the schema default instead of crashing on the `NOT NULL` constraint.
- **`vstash retrain --synthesize-queries`** (#294) no longer crashes on Ollama / Cerebras backends; `retrain_synth` backend config realigned with the current schema.
- **Web uploads** (#295) now persist under `~/.vstash/uploads/<uuid>-<safe-name>` instead of pointing at deleted temp paths.

### Research artifacts (this release)

- **v4 retrain validation** (#305) — post-v0.34 cosine-fix evidence plus a multi-seed reproducibility toolkit.

## [0.35.0] - 2026-04-27

### Removed (breaking)

- **``fts_only=True`` bool parameter on ``search`` / ``ask``** (#281).  Deprecated in v0.33.0 with a ``DeprecationWarning`` and retained through v0.34.0; removed from every public surface in v0.35.0:

  - ``VstashStore.search(..., fts_only=...)``
  - ``Memory.search(..., fts_only=...)`` and ``Memory.ask(..., fts_only=...)``
  - MCP tools ``vstash_search`` and ``vstash_ask``

  Callers still passing it hit a ``TypeError`` from Python's argument binder.  The migration is a straight rename to ``retrieval_mode="fts_only"``.  Everything else about the three-mode enum (``"hybrid"`` | ``"vec_only"`` | ``"fts_only"``) is unchanged.

### Added

- **`vstash retrain --eval-queries` flag** (#299).  The retrain CLI now accepts a held-out labeled JSONL (`{"query": str, "relevant_paths": [str, ...]}` per line) for the eval gate, in addition to the training JSONL.  Drives a refuse-to-save policy: a candidate is promoted to `--output` only if its NDCG@10 on the holdout exceeds the active model by `--min-gain` (default 0.0).  Failed candidates are retained at `output.candidate/` for inspection but never replace the user's deployed model.  This is the deployable surface of the **gated domain-adaptation loop** documented in the v2 paper draft.

### Changed

- ``VstashStore._resolve_retrieval_mode`` is now single-argument (only the ``retrieval_mode`` string).  The removal of the legacy bool also removed the ``"conflicts with fts_only=True"`` ``ValueError`` branch since that combination is no longer expressible.
- ``_coerce_bool`` helper in ``vstash.mcp`` removed; no remaining callers.

### Research artifacts (this release)

- **Paper v2 draft** -- `paper/v2/vstash-paper-v2.md`, full markdown draft of "vstash v2: Eval-Gated Domain Adaptation for Local-First LLM Memory" (1749 lines), with bootstrap CIs on the LongMemEval holdout, Mermaid figures, and the gated-domain-adaptation thesis as central contribution.  LaTeX conversion to `paper/arxiv/vstash-v2.tex` for arXiv submission is the next milestone.
- **`bge-small-rrf-lme-v1` model** -- chat-memory specialist embedding (33M params, 384-dim) trained via the new `--eval-queries` gate on 398 LongMemEval-s questions.  Holdout NDCG@10 0.6878 vs vanilla BGE-small 0.6143 (+0.0735 absolute, +11.85% relative); R@5 +3.79pp [95% CI +1.72, +6.19] paired bootstrap on n=102.  Published at https://huggingface.co/Stffens/bge-small-rrf-lme-v1.
- **LongMemEval head-to-head harness** -- `experiments/longmemeval_retrieval.py`, `experiments/longmemeval_colbert.py`, `experiments/colbert_minimal.py` (faithful HF ColBERTv2 re-implementation; raw `BertModel` + 768->128 linear projection + manual MaxSim einsum), and `experiments/h2h_combine.py` for 5-arm result aggregation.  Calibration band derived from 5-dataset BEIR run (`experiments/beir_colbert.py`), ArguAna outlier disclosed.
- **Eval-gated retrain reproducibility** -- `experiments/lme_prepare_retrain.py` (deterministic 80/20 stratified split), `experiments/lme_holdout_bootstrap.py` (paired bootstrap CIs, B=1000, seed 42), `experiments/results/lme_eval.jsonl` + `lme_train.jsonl` + `lme_retrain_meta.json` (eval-gate proof), `experiments/results/lme_full_500_*.json` (5 arms).
- **Cross-domain transfer failure evidence** -- `Stffens/bge-small-rrf-v3` (BEIR-tuned, v0.34.0 release) regresses -2.45pp NDCG@10 on the LongMemEval holdout vs vanilla BGE-small.  This is the v2 paper's core empirical claim and the motivation for per-domain retrain.

## [0.34.0] - 2026-04-24

### Fixed

- **`vec_chunks` now uses cosine metric, not L2** (#271, #272, #286). `sqlite-vec`'s `vec0(embedding float[N])` defaults to L2 distance, but every comment, threshold, and telemetry field in vstash labelled the value "cosine distance". Worked accidentally for BGE-small unit-normalized embeddings; broke for non-normalized models like `paraphrase-multilingual` where L2 exceeds 2.0 on large-magnitude vectors. `SCHEMA_VERSION` bumped to `"2"`; v1 DBs rebuild `vec_chunks` in place on open via an atomic (`BEGIN IMMEDIATE`) + idempotent (`sqlite_master` guard) + streaming-in-SQL (`TEMP` backup, never materializes embeddings in Python) migration. No re-embedding required. `relevance_tier` thresholds recalibrated from 0.95/0.98 (L2-on-unit-vec) to 0.4513/0.4802 (cosine equivalents) so BGE-small keeps identical tier assignments. `distance_cutoff` defaults squared (`1.15 -> 1.3225`, long-query `5.0 -> 25.0`) because cosine ratios relate to L2 ratios by a square on unit vectors -- without this, NFCorpus / FiQA / ArguAna regressed past the 0.005 BEIR tolerance. BEIR non-regression gate green on all 5 datasets post-fix (SciFact 0.7251, NFCorpus 0.3591, FiQA 0.3917, SciDocs 0.1945, ArguAna 0.4367).
- **Flat `snapvec` backend treated similarity as distance** (#289, #290). `snapvec.SnapIndex.search` returns `(id, similarity)` in `[-1, 1]`, but the store was feeding that directly into `distance_cutoff`, `relevance_tier`, and `last_best_distance`, all of which assume cosine-distance space `[0, 2]`. Ranking worked by accident (descending similarity = ascending distance monotonically), but `distance_cutoff` was effectively disabled and a perfect match classified as `"low"` relevance. The sibling `snapvec-ivfpq` backend already converted internally -- flat just never got the same treatment. Fix branches on `self._vector_backend` so IVFPQ is not double-inverted; both branches clamp to `[0, 2]`.

### Added

- **Custom encoder resolver hook** (#278, #287, #288). `register_encoder_resolver(fn)` / `unregister_encoder_resolver` plus an `Encoder` `Protocol` let callers plug in LoRA-adapted, locally fine-tuned, or otherwise-unnamable encoders without monkey-patching `vstash.embed`. `embed_texts` / `embed_query` / `get_embedding_dim` consult the registry before every built-in path (daemon delegation, Gemma, HF ONNX, MLX, FastEmbed). Identity-based registration (so `__eq__`-defining callables don't collapse), runtime protocol validation (malformed encoders skipped with a warning instead of cascading into `AttributeError`), and shape validation (`ValueError` with index on row-count mismatch, wrong per-row dim, or non-sequence row). Resolvers are process-global; the CLI / MCP / SDK all share the same registry today. Docstring includes a SentenceTransformer adapter recipe (ST uses `get_sentence_embedding_dimension()`, not an `embedding_dim` attribute).

### Validation artifacts (this release)

- `experiments/probe_vec0_cosine.py` -- sqlite-vec cosine distance probe.
- `experiments/probe_272_migration_parity.py` -- v1->v2 top-10 ranking parity (7/7 queries identical on BGE).
- `experiments/probe_272_concurrent_migration.py` -- 4 concurrent processes migrating the same v1 DB, no data loss.
- `experiments/probe_272_reindex.py` -- cosine DDL preserved across `reindex()`.
- `experiments/beir_272_cosine_validation_colab.ipynb` -- GPU-accelerated BEIR regression gate (monkey-patches `embed_texts` onto CUDA + batch 256; drops wall time from ~30-45 min CPU to ~5 min T4).

### Migration notes

- Existing v1 stores are migrated automatically on first open with v0.34. The migration is single-connection safe and streams in SQL, so a store with 5+ million embeddings will not OOM the Python process. `search_stats` rolling window is cleared (L2 spreads are not comparable to cosine spreads; the window repopulates from v2 searches). No user action required. `vstash check` is a good post-upgrade sanity pass -- all five invariants still hold post-migration.
- Callers that pass an explicit `distance_cutoff=1.15` keep that cutoff verbatim (now tighter in cosine space than in v1 L2 space). To preserve v1 selectivity set `distance_cutoff=1.3225`. The default already matches v1 behaviour.

## [0.33.0] - 2026-04-23

### Added

- **`retrieval_mode` enum on search** (#275, #276). New `retrieval_mode` parameter on `VstashStore.search`, `Memory.search`, `Memory.ask`, the MCP `vstash_search` / `vstash_ask` tools, and `VstashRetriever`. Three values:
  - `"hybrid"` (default, unchanged): vector ANN + FTS5 + adaptive RRF.
  - `"vec_only"`: skip the FTS5 branch; force `(vec_weight, fts_weight) = (1.0, 0.0)`. Useful when the corpus has no meaningful keyword signal (tabular, code, cross-lingual).
  - `"fts_only"`: existing #152 short-circuit, now named consistently.
- **`vstash why` CLI command** (#260, #261, #262). First-class miss-analysis interface: `vstash why "<query>" --expect <path>` traces where a target chunk was eliminated in the pipeline and suggests the parameter that would have surfaced it. `vstash why --recent` lists the latest logged misses. The same trace is exposed over HTTP at `/debug/why` on `vstash serve` and auto-logged on empty / low-relevance searches for later inspection.
- **`exact_match` substring filter on `search()`** (#263, #106). Post-filter bypasses FTS5 tokenization so identifiers and punctuation survive (unlike the FTS keyword path which stems and lowercases).
- **Eval-gated retrain pipeline** (#230, #232, #238, #239, #241, #242, #243, #257, #259). `vstash retrain` now composes: corpus split, held-out NDCG@10 eval, triple mining, MNRL training, atomic candidate/old promote. Refuses to save a candidate whose held-out NDCG@10 regresses versus the baseline. Supports LLM query synthesis (T1.3), multi-corpus training with temperature sampling (T1.4), GPU-batched eval and mining (T1.4b/c), labeled-query pair mining from BEIR qrels (T1.5), auto-promotion of eval queries (H-R1), bulk mining on single-corpus (H-R8), and full eval observability + seed reproducibility (H-R5/H-R7). Validated in Colab at +5.00% macro NDCG@10 across SciFact + NFCorpus + FiQA. The new `bge-small-rrf-v3` model ships on HuggingFace as `Stffens/bge-small-rrf-v3`.

### Changed

- **`ingest_directory` routes through `ingest_batch`** (#274). Internal refactor; no behaviour change. The new `ingest_batch(paths, ...)` helper is also consumed by the watch worker.

### Fixed

- **Two O(N^2) regressions in the ingest path** (#250, #251, #264, #267). Flat snapvec now mirrors the ivfpq deferred-save pattern (#250) and `SnapIndex.add_batch` is coalesced across all docs in a batch instead of per-doc (#251). `_rebuild_snapvec_from_vec_chunks` (#264) and `store.reindex` (#267) rewritten with keyset pagination + coalesced `add_batch`. At N=100k: rebuild dropped 41.6 s -> 4.0 s (10.3x); reindex shape improved 3.05x -> 2.28x at N=200k.
- **Watch worker burst ingest is now 4-5x faster** (#274). Previously processed one file per queue iteration; now drains a small burst window (default 64 files / 250 ms) and routes through `ingest_batch` so the whole burst shares one SQLite transaction, one FTS5 flush, and one snapvec vstack. Probe results: 1000-file burst on snapvec flat drops from 201 ms to 46 ms. (Note: this is a constant-factor optimization, not the O(N^2) fix originally framed in #266 -- that was already addressed by #250/#251.)
- **CLI leaked `VstashStore` on every exit** (#269, #268). `cli.py::_get_store` now registers `store.close` via `atexit`, so snapvec-backed DBs flush `.snpv` on process exit and the next open does NOT trigger a full `_rebuild_snapvec_from_vec_chunks`. Post-#264 this saves ~4 s per CLI command at N=100k.
- **ONNX embedding init** (#234, #235). When a shipped model contains only the `model.onnx` stub without its external data file, `_init_hf_onnx` now downloads the external data alongside and falls back to `SentenceTransformer` for safetensors loading on ONNX failure. Added because `Stffens/bge-small-rrf-v2` shipped an ONNX stub referencing an external data file that was never uploaded.

### Deprecated

- **`fts_only=True` bool on `search` / `ask`** (#275, #276). Use `retrieval_mode="fts_only"` instead. Still honoured for this release with a `DeprecationWarning`. Combining `fts_only=True` with a contradictory `retrieval_mode` raises `ValueError` rather than silently ignoring one of them.

### Performance

- MMR greedy selection: invariant precomputed once per query (#256).
- Context expansion CTE: VALUES batch lookup (#228).

### CI / Repo hygiene

- CI now runs on PRs targeting `develop`, not only `main` (#270). Release PRs to `main` had been silently merging red since v0.30.0 because feature PRs never hit the lint + test matrix.
- `ruff format .` pass across the repo; `E402` silenced for `.ipynb` (#270).
- Added `docs/professionalization-roadmap.md` with a prioritized P0-P3 plan for the next hygiene upgrades (#273).

## [0.32.0] - 2026-04-16

### Added

- **Persistent embedder daemon**. `vstash serve` now pre-loads the embedding model at startup (`--warm`, on by default) and exposes a `/api/embed` HTTP endpoint. CLI and SDK clients automatically detect a running daemon on localhost:8585 and delegate embedding to it, eliminating ~2s cold start on every invocation. Falls back to local embedding transparently if no daemon is running. Override with `VSTASH_EMBED_URL` env var.

## [0.31.0] - 2026-04-16

### Added

- **Query LRU cache** for `VstashStore.search()`. Opt-in via `[cache] query_cache_size` in `vstash.toml` (default 0 = disabled). Repeated identical queries return from an in-memory LRU cache. Automatically invalidated on any write (add, delete, reindex). Skipped for `explain=True` and `miss_analysis()`. Benchmark shows ~700x speedup on cache hits for repeated queries.
- **Deferred FTS indexing** in `batch_mode(defer_fts=True)`. FTS5 inserts are collected in memory during batch operations and flushed in a single bulk pass on exit.

### Changed

- **`ingest_directory` now uses batched store writes.** Files are prepared (parse + chunk + embed) individually, then stored via `add_documents_batch` in a single transaction with deferred FTS. Combined speedup: **5x** at 500 docs versus the old per-file `add_document` loop.

## [0.30.0] - 2026-04-15

### Added

- **`snapvec-ivfpq` vector backend** (#209). New `vector_backend` option that routes searches through `snapvec.IVFPQSnapIndex` with `keep_full_precision=True` and `rerank_candidates=100`. Benchmark at N=100K (BGE-small, SciFact padded with FIQA): 23x faster than sqlite-vec (1.04 ms vs 23.8 ms p50) with -0.4% recall (0.994 vs 0.998) and 43% less disk (85 MB vs 149 MB). Pareto-dominant over sqlite-vec at N>=50K.
- **`vstash snapvec fit` CLI command** (#209). Trains and persists the IVFPQ index from the current corpus: reads every embedding out of `vec_chunks`, samples up to `--training-sample` for codebook training, indexes all rows, and saves the `.snpi` next to the database. Until this runs, sqlite-vec stays authoritative.
- **New IVFPQ tuning knobs under `[storage]`**: `ivfpq_nlist`, `ivfpq_M`, `ivfpq_K`, `ivfpq_rerank_candidates`, `ivfpq_nprobe`.
- **`experiments/snapvec_backends_bench.py`** (#207). Standalone benchmark comparing sqlite-vec vs snapvec flat/pq/residual/ivfpq (+ rerank) at 10K/50K/100K on BGE-small embeddings, measuring recall@10 vs exact brute-force.

### Changed

- **Minimum `snapvec` version raised from `>=0.1.0` to `>=0.7.1`** (#207). Picks up upstream `delete()` O(1) via swap-with-last, the fp16 rerank cache that halved the `keep_full_precision` footprint, and CRC32 trailers on all four file formats.
- Removed the `SnapIndex.delete_batch` monkey-patch in `vstash/store.py`; snapvec ships O(1) per-id delete upstream, and the loop over `.delete()` is the intended API.

## [0.27.0] — 2026-04-09

### ⚠️ Breaking change

- **`Memory.search()` now honors the instance collection when it is the literal string `"default"`** (#165, PR #166). Prior to v0.27, `Memory(collection="default").search("x")` silently searched **every collection** in the database because `_resolve_collection` had a `!= "default"` shortcut that treated `"default"` as a sentinel for "no filter". This created a read/write asymmetry — writes were scoped to `"default"` while reads leaked across collections. The shortcut is removed. Callers who relied on the old "search everywhere" behavior must now pass `collection=None` explicitly:

    ```python
    # old (implicit):
    mem = Memory(db=...)
    mem.search("x")

    # new (explicit, preserves old behavior):
    mem = Memory(db=...)
    mem.search("x", collection=None)
    ```

    The fix also applies to `list()`, `get_document_chunks()`, and `miss_analysis()` — every method that calls `_resolve_collection`. Downstream consumers that maintain a second collection in the same store (engram's audit log, medlocal-style multi-profile setups) will now get correct isolation without needing the explicit-collection workaround.

### Added

- **MCP RRF passthrough** (#159, PR #168) — the MCP server tools `vstash_search` and `vstash_ask` now expose `vec_weight`, `fts_weight`, and `fts_only` parameters, mirroring the SDK surface added in v0.26.0. Claude Desktop and any other MCP client can now pin RRF weights or force FTS-only retrieval on a per-call basis, making the paraphrase-multilingual clinical-domain mitigation (documented in `docs/embedding-models.md`) reachable from outside Python.
- **Defensive type coercion** for MCP-supplied RRF parameters — MCP clients inconsistently send JSON strings where floats/bools are expected (e.g. `"0.5"` or `"true"`). New `_coerce_optional_float` and `_coerce_bool` helpers in `vstash/mcp.py` handle this permissively, rejecting NaN and ±Inf explicitly so non-finite weights cannot propagate into RRF scoring. Unparseable values surface as structured MCP errors naming the offending field, not server crashes.
- **`fts_only` precedence rule** for MCP tools — when `fts_only=true`, `vec_weight` and `fts_weight` are dropped before coercion and validation. A caller can safely pass `fts_only=true` with an invalid or out-of-range weight and still get a successful FTS-only query.

### Performance

- **Pure-Python `_cosine_sim` is 5–11× faster** (#149, PR #149). Version-gated dispatch: `math.sumprod` on Python 3.12+ (3× faster than `sum(map(operator.mul, ...))`) with a fallback to the map-based path for 3.10/3.11. `math.hypot(*vec)` replaces the generator-expression norm on all versions. Real-world impact: `_cosine_sim` is called inside the triple-nested loop of `_mmr_dedup`, saving ~20–25 ms per search on corpora where MMR dedup fires with multi-chunk documents. Zero impact on single-chunk-per-doc corpora (the MMR path short-circuits). Credit to @google-labs-jules for surfacing the opportunity; the `math.sumprod` refinement was added on top.

### Fixed

- **`_resolve_collection` review follow-ups** (#166 review): trimmed verbose docstring (historical bug context moved to commit message), and added a regression tripwire test covering `list()` to catch any future refactor that accidentally re-introduces the `"default"` shortcut — the fix applies to four methods, not just `search()`.
- **#168 review feedback** (7 items, all applied): NaN/Inf float rejection, `fts_only` precedence short-circuit, `ValueError` split from broad except (no stack traces for user-input errors), `"none"`/`"null"` recognized as `_coerce_bool` false values, docs correction on weight alone-or-together semantics, two new regression tests.

### Documentation

- **`docs/mcp-server.md`** — new "Per-call RRF controls" section documenting the three new parameters, when to use each, type coercion behavior, and the precedence rule.

---

## [0.26.0] — 2026-04-08

### Added
- **Per-call RRF weight overrides on `Memory.search()`** (#151, PR #158) — `vec_weight` and `fts_weight` parameters let callers pin the hybrid-search weights for a single query without reaching into `Memory._store.search`. `None` (default) preserves adaptive per-query RRF; explicit values override it. New `RRFWeightOutOfRangeError` validates the `[0.0, 1.0]` range at the API boundary. `Memory.ask()` forwards the kwargs too.
- **First-class `fts_only` mode on `Memory.search()` / `.ask()`** (#152, PR #163) — `fts_only=True` short-circuits the pipeline to FTS5 keyword matching only: no vector ANN scan, no distance cutoff, no adaptive RRF. Useful for debugging ranking, cross-lingual / highly technical queries with diffuse embeddings, and as a deliberate fallback when the vector pool is expected to be empty. The FTS hits still flow through MMR dedup, recency boost, and context expansion. `is_fts_top` cap is bypassed in `fts_only` mode so all FTS candidates can compete.
- **Adaptive vector-empty fallback** (#156, PR #162) — when the vector candidate pool is empty after the distance cutoff (embedding mismatch, tight cutoff, sparse metadata filter) and FTS5 has results, the pipeline now automatically collapses to FTS-only scoring with `vec_weight=0.0, fts_weight=1.0`. A literal-match FTS hit at rank 0 now scores ~0.0167 (full FTS weight) instead of the degraded ~0.0067 it would have earned under the previous fused-with-empty-vec behavior. Increments a new `adaptive_rrf_vector_empty_fallback_total` metric and records an `adaptive_fallback` stage in the `miss_analysis` tracer.
- **`adaptive_rrf_vector_empty_fallback_total` counter** in the metrics registry, with documentation in `docs/observability.md` covering common causes, diagnostic drill-down, and a Prometheus alerting recipe.

### Fixed
- **`last_best_distance` always resets when vector pool is empty** — even when both `vec_rows` AND `fts_rows` are empty (a query that matches nothing). Previously the pipeline would carry the stale distance from a prior query, lying about the current query's confidence. Flagged in PR #162 review.
- **`miss_analysis` tracer no longer misattributes `fts_only` skips** — the `vector_search` stage in `fts_only` mode is now recorded with `passed=True` and an explicit "intentionally skipped by caller" detail, so the downstream "both generators missed → invisible to RRF" heuristic does not flag an intentional skip as a modality failure.
- **`fts_only` strips conflicting weight arguments before validation** — passing `fts_only=True` together with out-of-range `vec_weight=1.5` no longer raises `RRFWeightOutOfRangeError`, since the weights are dropped before reaching the validator.

### Documentation
- **`paraphrase-multilingual-MiniLM-L12-v2` clinical-domain weakness** (#155, PR #161) — new section in `docs/embedding-models.md` documenting the failure mode observed in production on a medical-document corpus, with five mitigations ordered by effort (`fts_only=True`, relax distance cutoff, pin RRF weights toward FTS, switch model, reindex), diagnostic signal via `miss_analysis()`, and the auto-detection that #156 now provides. Also flagged in `paper/vstash-paper.md` §8.8.

### Internal
- Silenced 3 pre-existing `sentencepiece` SWIG `DeprecationWarning` in `pyproject.toml [tool.pytest.ini_options].filterwarnings`. Cannot be fixed upstream; filter is narrow (matched by exact message text).

---

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
- **`doc_completeness` now takes `collection`** (#134, #142) — the classification ignored collection in v0.24, so a partial copy of a path in collection A could mask collection B's complete copy and the recovery delete could wipe B too. Now collection-scoped.
- **`delete_document` gains optional `collection` kwarg** — default unchanged (delete every copy of the path) so existing callers in `watch.py` / `mcp.py` / `cli.py` / `journal.py` keep working; `ingest()`'s partial-recovery path passes the current collection explicitly.
- **FTS5 parity invariant** — `fts_index_parity` used `COUNT(*) FROM fts_chunks` on a `content=chunks` virtual table, which reads from the underlying `chunks` table and could never detect FTS5 index drift. Replaced with the canonical FTS5 `integrity-check` command, which actually scans the index.
- **Implicit transaction** — the FTS5 `integrity-check` is DML, so the stdlib `sqlite3` driver opens an implicit transaction for it; `integrity_repair()` now commits the pending state before `BEGIN IMMEDIATE` so the explicit transaction isn't preempted.
- **CLI cleanup** — `vstash check --repair` now reads the post-repair state from the same connection instead of re-opening the store.

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
- **Threading hardening** (#128) — `vstash/store.py` now asserts `sqlite3.threadsafety > 0` at **module import time**, surfacing the requirement loudly on exotic single-threaded libsqlite builds instead of letting it manifest as sporadic corruption at runtime. Most Python builds use `sqlite3.threadsafety = 3` (fully serialized) and pass this check transparently.

---

## [0.20.1] — 2026-04-06

### Fixed
- **Close STEM (stemming) connections from any thread** (#125, #127) — fixes an asyncio/threading deadlock in the MCP server path. The per-thread FTS5 Porter stemming connections in `VstashStore._stem_conns` could previously only be `close()`d from the thread that opened them; after that thread exited, the connection leaked file descriptors and hung process shutdown. Connections are still only *used* from their owning thread (guaranteed by dict-by-tid lookup), but `check_same_thread=False` now lets the main thread release them at shutdown.

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
