# Retrain module roadmap

Living design doc for extending `vstash/retrain.py`. Tiered by ROI / effort.
Owner: Jay + Claude. Not shipped (under `experiments/`, excluded from sdist).

## Current state (v0.32)

- 247 LOC self-supervised fine-tuning via MNRL.
- Signal: first 200 chars of each chunk becomes a pseudo-query, the chunk itself
  is the positive, first vec/FTS top-5 disagreement provides 1 hard negative.
- Validated on 5 BEIR datasets: +4.5% NDCG avg, +18.3% NFCorpus, +5% SciFact.
  Released as `stffens/bge-small-rrf-v1` and `v2` on HuggingFace.
- CLI: `vstash retrain` with `--quick`, `--max-queries`, `--epochs`, `--lr`,
  `--batch-size`, `--base-model`, `--output`.
- Tests: `tests/test_retrain.py` covers triple generation and training wrapper
  with mocked `sentence_transformers`.

## Limitations driving this roadmap

1. Queries do not look like real queries (chunk prefixes, not short human text).
2. Only 1 triplet emitted per pseudo-query (top-5 disagreement has more signal).
3. Positives lack diversity (always the source chunk itself).
4. Hard negatives are not quality-filtered (first disagreement wins).
5. Training is blind: no hold-out, no NDCG delta, no regression guard.
6. Only MNRL loss. No `CachedMNRL`, no `GISTEmbedLoss`, no MarginMSE.
7. No reranker training path (stack is bi-encoder only).
8. No continual / incremental retrain on delta corpus.
9. Lineage gap: `training_meta.json` records hyperparameters but not corpus
   fingerprint, disagreement rate, or eval delta.
10. No HF Hub push helper, no integration with embedder daemon hot-swap.

---

## Tier 1 (high ROI, ~1-2 days each)

### T1.1 Eval-gated training

[SHIPPED on feat/retrain-tier1] Added `split_corpus_for_eval`,
`evaluate_model`, and the composed `retrain()` with a `.candidate`
promote/reject gate. See commits on the branch.

**T1.1b follow-up: external labeled queries + multi-relevant NDCG.**

Colab smoke on a SciFact 1k subset showed baseline + final NDCG both
saturating at 1.0. Root cause: our internal pseudo-queries (first
200 chars of a chunk) are too strong a cue for their own doc on
diverse corpora. The eval then has no discriminating power.

Fix shipped in the same branch:

- New multi-relevant `_ndcg_from_ranks(ranks, num_relevant, k)` math
  that handles the BEIR case where a query has more than one relevant
  doc.
- New public helper `qrels_to_eval_queries(queries, qrels,
  path_for_doc_id)` converting BEIR-style labels into our eval
  format. Supports a configurable binary-relevance threshold.
- `evaluate_model` accepts the new `relevant_paths: list[str]` shape
  and still honors the legacy single-path form.
- `retrain(..., eval_queries=...)` override so a caller can plug in
  human-labeled queries. The internal pseudo-query split becomes the
  fallback when no labels are available (typical user stores).
- Colab notebook switched to real SciFact qrels across the full 5k
  corpus for an honest eval.

**Goal:** never save a model that is worse than the base on the user's corpus.

**Design:**
- New function `split_corpus_for_eval(store, eval_fraction=0.15, seed=42)`
  returns `(train_chunk_ids, held_out_rows)` where each held-out row is
  `{query: str, relevant_path: str}`.
- New function `evaluate_model(model_name, store, held_out, top_k=10)` loads
  the model, embeds each held-out query, runs `store.search(top_k)`, computes
  binary NDCG@10 against the single relevant path. Returns
  `{"ndcg@10": float, "mrr": float, "hit@10": float, "n": int}`.
- `train_mnrl` gains a `baseline_metrics` and `final_metrics` block in
  `training_meta.json`.
- Top-level `retrain()` helper composes: split -> baseline eval -> generate
  -> train -> final eval -> gate on `min_gain`.
- CLI flags: `--min-gain 0.0`, `--no-eval`, `--eval-fraction 0.15`.
- If `final - baseline < min_gain`: print delta table, do NOT save to
  `output_path`, exit 1. `--min-gain -1` acts as "always save".

**Files:**
- `vstash/retrain.py`: add `split_corpus_for_eval`, `evaluate_model`, wrap
  composition into a new `retrain()` entry function (used by CLI).
- `vstash/cli.py`: wire flags, print delta table.
- `tests/test_retrain.py`: tests for split determinism, ndcg computation,
  gate pass/fail, `--no-eval` path.

**Risk:** reindexing temporarily with a new model to evaluate is expensive on
large stores. Mitigation: evaluate by re-embedding held-out queries only and
comparing search results against the existing vector index. That is not
strictly fair (base vs fine-tuned produce different query embeddings), but
captures the delta at the decision point: "will this model improve retrieval
as a drop-in query encoder?".

**Exit criteria:**
- `vstash retrain` on empty / trivial store exits cleanly with "not enough
  data for eval".
- On a BEIR SciFact-sized corpus, runs end-to-end, prints delta, saves model
  only when delta >= `--min-gain`.
- Tests cover split determinism, NDCG@10 math, gate behavior.

---

### T1.2 Multi-triplet emission

[PARKED on feat/retrain-t12-multi-triplet, PR #231 left open as WIP]
Mechanics shipped (tests green), but the Colab smoke on SciFact
showed the approach regresses hard when the underlying query signal
is poor: delta NDCG@10 -4.97% with 10k pairs (vs +0.24% with 2k pairs
at K=1). The T1.1 gate correctly rejected it.

**Finding**: multiplying training signal only helps when the signal
itself is quality. With chunk-prefix pseudo-queries, K>1 amplifies
the distribution mismatch between training queries and real queries
instead of improving the model. Revisit once T1.3 (LLM query
synthesis) lands so the base signal is clean.

**Goal:** extract 3-5x more training signal from the same disagreement data.

**Design:**
- `generate_triples` gains `triplets_per_query: int = 1` (default preserves
  current behavior).
- When > 1: iterate over top-K disagreements; each disagreeing path produces
  one triplet (query, positive, hard_neg_i). Skip duplicates on negative text.
- Log disagreement histogram so users see "your corpus produces 2.3 triplets
  per query on average".

**Files:**
- `vstash/retrain.py`: refactor the hard-negative loop to yield multiple.
- `vstash/cli.py`: add `--triplets-per-query` (default 5 in retrain command).
- `tests/test_retrain.py`: existing tests stay green with default=1; new test
  asserts K triplets emitted when K disagreements exist.

**Risk:** larger training sets increase epoch time. Benchmark and document.

**Exit criteria:**
- Existing tests untouched.
- New test verifying 5 triplets emitted from a crafted 5-way disagreement.
- Run on SciFact shows > 1.5 avg triplets/query.

---

### T1.3 LLM query synthesis [SHIPPED in PR #232]

Module `vstash/retrain_synth.py` with `synthesize_queries()` + JSONL
cache. `generate_triples(..., synthesized_queries=..., pre_sampled_chunks=...)`.
`retrain(..., synthesize_queries=True, synth_n, synth_cache, synth_model, cfg)`.
CLI flags: `--synthesize-queries`, `--synth-n`, `--synth-cache`, `--synth-model`.

**Empirical result on two BEIR datasets (Gemini 2.5 Flash, synth_n=2,
max_queries=1000):**

| Dataset  | Baseline NDCG@10 | Final NDCG@10 | Delta     |
|----------|------------------|---------------|-----------|
| SciFact  | 0.7251           | 0.7272        | +0.21%    |
| NFCorpus | 0.3570           | 0.3573        | +0.03%    |

Both deltas are within noise. Synth queries DO NOT beat chunk prefix
at this scale on single-domain corpora. The published +5/+18%
numbers (bge-small-rrf-v2 card) come from **multi-dataset training**
with ~100k triples combining scifact + nfcorpus + fiqa, not from
better queries on a single domain.

T1.3 mechanics are sound; the quality lift is a different problem
handled by T1.4.

---

### T1.4 Multi-corpus training harness with temperature sampling

[LANDED on feat/retrain-t14-multi-corpus] `retrain_multi()` + CLI
`vstash retrain-multi` shipped with `compute_triple_budget` (uniform /
proportional / temperature with largest-remainder allocation), atomic
`.candidate` / `.old` promotion, per-dataset + macro NDCG@10 eval
gate, and optional `--per-dataset-gate` for stricter regressions.
Reference Colab: `experiments/retrain_t1_4_multi_beir.ipynb`. 23 new
unit tests under `tests/test_retrain_multi.py`. Next action: run the
notebook on T4 to confirm per-dataset deltas match v5 within noise
(target: NFCorpus > +10%, SciFact > +3%).

**T4 memory follow-up (same branch):** `use_amp=True`, `batch_size=32`
default for multi-corpus, optional `max_seq_length`, plus a
`_release_gpu_memory()` hook between baseline eval and training.
Drops the T4 OOM observed on the first overnight Colab run.

### T1.5 Labeled-query training pair mining

[LANDED on feat/retrain-t15-labeled-training-queries] First Colab run
of T1.4+b+c on 2026-04-18 gated out at macro NDCG@10 -5.15% (SciFact
-5.8%, NFCorpus -0.19%, FiQA -9.47%). Root cause: chunk-prefix pseudo-
queries are statements, not questions, and MNRL trained on statement-
as-query damages question-based retrieval. The two datasets that
regressed are both question-based (SciFact, FiQA); NFCorpus (keyword)
was flat.

Comparison vs v5 notebook (`experiments/retrain_v5_hard_neg.ipynb` +
`experiments/rrf_training_pairs.py`) showed v5 uses real BEIR queries
as training text, gold doc first chunks as positives, multiple hard
negs per (query, gold) pair, and fixed 0.95/0.05 RRF weights.

T1.5 adds `generate_labeled_triples_batched()` that reproduces the
v5 recipe. `retrain_multi(..., training_queries_by_dataset=...)`
routes datasets with labeled queries to the new path; datasets
missing from the map fall back to chunk-prefix. 7 new tests under
`tests/test_retrain_batch.py`. Notebook Cell 5 now passes
`training_queries_by_dataset=eval_queries_by_dataset` so the same
qrels feed both training and eval, matching v5.

Next regression run with T1.5: target NFCorpus > +10%, SciFact > +2%,
macro positive. If those land, the paper claims reproduce and
chunk-prefix can be deprecated or documented as a weaker fallback for
users without labeled queries.

**T1.5 Colab validation (2026-04-19).** First run after the recipe
fix came in at:

| Dataset  | Baseline | Final  | Delta    |
|----------|----------|--------|----------|
| SciFact  | 0.7261   | 0.7802 | +5.41%   |
| NFCorpus | 0.3449   | 0.3670 | +2.22%   |
| FiQA     | 0.3776   | 0.4513 | +7.37%   |
| Macro    | 0.4828   | 0.5328 | +5.00%   |

SciFact and FiQA reproduce or beat the v5 published numbers
(+5%, +5-6%). NFCorpus is well short of v5's published +18.3%; the
gap is the open question that motivated H-R3 (hard-neg margin
filter) and will likely motivate a H-T14.5 (per-query triple cap /
temperature re-tuning) if H-R3 does not close it.

**Second validation (2026-04-19, post PR #243 merge).** Re-run with
H-R5 (NDCG@3 / Recall@100 / wider top-K in batched eval) + H-R7
(seed determinism) came in at macro +4.14% (SciFact +4.53%,
NFCorpus +1.39%, FiQA +6.51%). **Final absolute NDCG@10 was
equivalent or a touch better than the first run on every dataset**
(SciFact 0.7802 vs 0.7786, NFCorpus 0.3670 vs 0.3677, FiQA 0.4513
vs 0.4568). Delta% dropped only because the widened eval pipeline
raised baseline NDCG@10 in parallel. Corollary: when comparing our
numbers to v5's published deltas, prefer comparing absolute final
NDCG@10, not percentage deltas (see
`project_t15_colab_validation.md` + `feedback_eval_pipeline_shift.md`
in memory).

### T1.4c Batched GPU eval

[LANDED on feat/retrain-t14c-batched-eval] Follow-up to T1.4b.
Once triple mining became fast, `evaluate_model`'s per-query
`eval_store.search()` scan became the new bottleneck: at
`eval_noise_size=57638` and ~600 qrels per dataset, baseline + final
eval on a 3-corpus run took ~60 minutes on Colab T4.
`evaluate_model_batched()` in `retrain_batch.py` keeps the temp
store for FTS5 but replaces the vec scan with one
`query_vecs @ corpus_vecs.T` matmul over all eval queries, dropping
total eval to ~2 minutes.
`retrain_multi(..., bulk_eval=True)` + CLI `--bulk-eval` route the
baseline + final eval calls to the batched evaluator. 9 new tests
under `tests/test_retrain_batch.py`. Typical end-to-end runtime on
the 3-corpus BEIR notebook with `bulk_mine=True + bulk_eval=True`:
~20-25 min (was ~3 hours pre-T1.4b, then ~90 min post-T1.4b).

### T1.4b Batched GPU triple mining

[LANDED on feat/retrain-t14b-batched-mining] New module
`vstash/retrain_batch.py` with `generate_triples_batched()`. Replaces
the per-query `store.search` call with a single
`query_vecs @ corpus_vecs.T` matmul on GPU plus per-query FTS5
(already cheap). Output shape is byte-for-byte compatible with
`generate_triples`, so `train_mnrl`/`retrain_multi` are unchanged.
`retrain_multi(..., bulk_mine=True, bulk_mine_device=...)` routes
through the batched miner; CLI exposes `--bulk-mine` and
`--bulk-mine-device`. 13 new unit tests under
`tests/test_retrain_batch.py` cover the RRF math, FTS sanitizer,
synth-query override, `exclude_chunk_ids`, ImportError handling, and
the `retrain_multi` wiring. Colab runtime on the 3-corpus
scifact+nfcorpus+fiqa mix drops from ~3 hours to ~20-30 min.

**Goal**: productize the training path that produced
`Stffens/bge-small-rrf-v2` (+5% SciFact, +18.3% NFCorpus, validated
in `experiments/retrain_v5_hard_neg.ipynb`) into a first-class
`retrain_multi()` API. The numbers are real -- the v5 notebook
proves them -- but today reproducing them requires running two
ad-hoc scripts (`experiments.rrf_training_pairs` + a bespoke
training cell). T1.4 brings that flow into `vstash.retrain` so
users can get the same win with one `retrain_multi(stores=...)`
call + the T1.1 eval gate in front.

**Why naive concatenation fails**: with NFCorpus=3.6k, SciFact=5k,
FiQA=57k, raw concat produces batches that are ~87% FiQA. The
smallest corpus (NFCorpus) receives almost no gradient signal, so
its NDCG gain evaporates. This is a classic multi-task sampling
problem, standard fix is temperature sampling. The v5 notebook
generated a fixed number of triples per dataset which implicitly
balanced it; T1.4 makes the strategy explicit and tunable.

**Design:**

Add a new entry point that accepts N stores (or a single store with
a `collection` filter list) plus a sampling strategy:

```python
def retrain_multi(
    stores: list[VstashStore] | dict[str, VstashStore],
    base_model: str,
    sampling: Literal["uniform", "proportional", "temperature"] = "temperature",
    temperature: float = 0.5,
    total_triples: int = 10000,
    output_path: str = "~/.vstash/models/retrained-multi",
    synthesize_queries: bool = False,
    eval_queries_by_dataset: dict[str, list[dict]] | None = None,
    ...
) -> RetrainResult:
```

Sampling math (for dataset `d` with `|D_d|` eligible chunks):
- **uniform**: `p(d) = 1/N` -- every dataset contributes equally
  regardless of size.
- **proportional**: `p(d) = |D_d| / Sum(|D_i|)` -- naive concat,
  included for comparison only.
- **temperature** (default): `p(d) = |D_d|^alpha / Sum(|D_i|^alpha)`
  where `alpha = temperature`. `alpha=0` -> uniform, `alpha=1` ->
  proportional. `alpha=0.5` is a common middle ground (favours
  smaller datasets somewhat but still rewards size).

Per-dataset triple budget = `round(p(d) * total_triples)`.

**Batch construction**: triples from all datasets get shuffled
together before going into the DataLoader. The sampling-ratio work
is done at triple generation, not per-batch. This is simpler than a
custom BatchSampler and gives the same expected per-epoch ratio.

**Eval**: accept a mapping `{dataset_name: eval_queries}` so the
gate reports per-dataset NDCG@10 plus a macro-average. Save all
slices into `training_meta.json`. `min_gain` applies to the
macro-average by default; optional `--per-dataset-gate` makes the
gate require each dataset individually to hold.

**Reference implementation**: `experiments/retrain_v5_hard_neg.ipynb`
+ `experiments/rrf_training_pairs.py`. T1.4 is effectively
extracting that flow into library code + adding explicit sampling
control + wiring the T1.1 eval gate around it.

**Files:**
- `vstash/retrain.py`: add `retrain_multi()` composer. Reuse
  `sample_training_chunks`, `generate_triples`, `train_mnrl`,
  `evaluate_model`. Per-dataset triple generation iterates the
  stores; global shuffle before DataLoader. The generator logic
  already exists in `experiments/rrf_training_pairs.py`, migrate
  the core loop into `retrain.py`.
- `vstash/cli.py`: new `vstash retrain-multi` subcommand OR extend
  `retrain` with `--extra-store path=alias path=alias` +
  `--sampling-temperature 0.5`. Probably a new subcommand is clearer.
- `experiments/retrain_t1_4_multi_beir.ipynb`: Colab notebook that
  ingests scifact+nfcorpus+fiqa into three vstash stores, runs
  `retrain_multi` with `temperature=0.5`, reports per-dataset +
  macro NDCG delta. Regression target: match the v5 notebook's
  NFCorpus delta of +18.3% within noise. If we do not match, the
  delta between v5's flow and T1.4's implementation is a bug we
  need to hunt.
- `tests/test_retrain_multi.py`: triple budget math across 3 sizes
  under each strategy, eval-per-dataset persistence, gate behaviour
  on mixed deltas.

**Exit criteria:**
- With `sampling="uniform"` and 3 BEIR datasets, each dataset
  contributes `total_triples / 3` triples exactly.
- With `sampling="temperature", temperature=0.5`, ratios follow the
  formula within +-1 triple rounding.
- On Colab smoke with the same dataset mix as v5
  (scifact+nfcorpus+fiqa), the per-dataset NDCG deltas match v5's
  numbers within a reasonable noise band (v5 ran at a specific
  seed, rerunning will vary). Concretely: NFCorpus delta > +10%,
  SciFact delta > +3%.
- T1.1 gate still works exactly as before on single-store calls.

**Risk**: scaling to 30k-100k triples with synthesize_queries=True
is LLM-bill-heavy (30k+ calls). Default should be prefix on this
harness and synth opt-in per dataset. Cache helps on re-runs.

**Estimate**: 1-2 weeks. Biggest unknown is whether naive-shuffled
batches suffice or whether we actually need a domain-balanced
BatchSampler. Start simple; upgrade only if metrics demand it.

[IN REVIEW on feat/retrain-t13-llm-synth] New module
`vstash/retrain_synth.py` with `synthesize_queries()` + JSONL cache
keyed by (chunk_id, prompt_hash, model). `generate_triples` gained
`synthesized_queries` + `pre_sampled_chunks` parameters so the LLM
and triple-generation paths operate on the same chunk set. `retrain()`
exposes `synthesize_queries: bool`, `synth_n`, `synth_cache`,
`synth_model`, `cfg`. CLI flags: `--synthesize-queries`, `--synth-n`,
`--synth-cache`, `--synth-model`. Tests: prompt + parser edge cases,
cache hit/miss, LLM failure resilience, progress callback, and a
generate_triples integration test proving synth queries replace the
prefix and emit one triplet per synthesized query. Colab smoke
notebook: `experiments/retrain_t1_3_llm_synth.ipynb` (separate from
the T1.1 notebook so history stays clean).

**Goal:** replace chunk-prefix pseudo-queries with short, realistic queries
produced by a local LLM. Closes the query-chunk distribution gap that limits
the current approach.

**Design:**
- New module boundary: `vstash/retrain_synth.py` (keeps the loss-free query
  generation separate from training).
- Function `synthesize_queries(chunks, backend="local", n_per_chunk=2,
  cache_path=None)` -> list of `{chunk_id, query}` dicts. Uses the existing
  `chat.py` local LLM backend (Ollama / LM Studio / OpenAI-compatible).
- Prompt (InPars-style, short):
  > "Given the following passage, write 2 short queries (5 to 15 words each)
  > that a user would type to find it. Output JSON: [\"q1\", \"q2\"]."
- JSONL cache keyed by `(chunk_id, prompt_hash, model)` so re-runs are free.
- `generate_triples` gains `query_source="prefix" | "synthesized" | "both"`.
  When "synthesized": replace pseudo-query with LLM output; retain the
  disagreement mining logic unchanged.
- CLI: `--synthesize-queries`, `--synth-model <name>`, `--synth-n 2`.

**Files:**
- `vstash/retrain_synth.py` (new).
- `vstash/retrain.py`: thread `query_source` parameter.
- `vstash/cli.py`: wire flags.
- `tests/test_retrain_synth.py` (new): mocked LLM backend; cache hit/miss;
  malformed JSON resilience.

**Risk:** LLM failures / rate limits on large corpora. Mitigations: JSONL
cache, graceful skip on parse failure, timeout per call.

**Exit criteria:**
- Generates realistic short queries for a sample chunk (manual sanity check).
- Cached runs re-use cache.
- End-to-end `vstash retrain --synthesize-queries` works on SciFact and
  produces a model with NDCG@10 >= baseline.

---

## Post-T1.5 hypotheses (tracked in `experiments/hypotheses.md`)

The full living backlog lives in `experiments/hypotheses.md`.
Retrain-side highlights below so this doc still works as the
standalone roadmap:

### H-R3. Hard-negative margin filter [INFRA LANDED, HYPOTHESIS NEGATIVE]

Infrastructure shipped in PR #245 (`margin_min` / `margin_max` on
`generate_labeled_triples_batched`, threaded through
`retrain_multi` + CLI). Default-off, opt-in, zero cost when unused.

**Ablation outcome (2026-04-19).** arm_a (`margin_min=0.05`,
no upper bound) macro **+1.65%** vs baseline **+4.14%** --
regression of -2.49pp. 53% of pairs dropped, training starved.
Pivot away from H-R3 as the NFCorpus-gap solution.

**Diagnosis.** On the base bge-small, cos(q, gold) typically sits
at 0.85-0.92 and cos(q, hard_neg) at 0.80-0.88, so margins are
intrinsically small (0.02-0.08). `margin_min=0.05` cuts the
actual hard-neg signal, not noise. Filter assumption holds only
for an already-trained model where cos(q, gold) saturates near
1.0. Keep the infra for continual-retrain (T2.6) and reranker
(T2.4) use cases where it applies cleanly.

**Next lever for NFCorpus gap**: H-R9 in `experiments/hypotheses.md`
(lower sampling temperature and/or larger total_triples). 0.5-day
Colab sweep, no code changes required.

### H-R7. Seed determinism [LANDED in PR #243]

`train_mnrl` pre-shuffles examples with `random.Random(seed)`,
calls `torch.manual_seed` + `torch.cuda.manual_seed_all`, and hands
a seeded `torch.Generator` to `DataLoader`. Threaded through
`retrain`, `retrain_multi`, and CLI `--seed`. `training_meta.json`
records the seed. Two back-to-back runs with the same seed produce
identical gradient steps.

### H-R8. Labeled queries + margin filter on single-corpus `retrain` [BACKLOG]

Today both levers live only on `retrain-multi`. A user with qrels
over a single private store has to wrap it in a one-element dict
to reach the labeled miner. Add `training_queries` / `margin_min` /
`margin_max` / `bulk_mine` to `vstash.retrain.retrain(...)` and
expose the matching CLI flags. 1 day, purely additive.

---

## Tier 2 (high impact, ~1 week each)

### T2.4 Cross-encoder reranker training

`vstash retrain-reranker` using the same disagreement data. Small cross
(~10 MB) on top of the bi-encoder typically adds 3-8 NDCG points. Requires
a new `rerank()` call in the search pipeline (opt-in) and a separate model
file under `~/.vstash/models/reranker/`.

### T2.5 GISTEmbedLoss option

Teacher-guided hard negative filtering; 2024 SOTA for small embedding
models. Add `--loss {mnrl, gist}` flag, teacher model defaults to
`sentence-transformers/all-mpnet-base-v2`. Drop-in replacement for MNRL in
`train_mnrl`.

### T2.6 Continual retrain

`--continue-from <model_path>` plus "retrain on delta corpus since last run".
Turns retrain into a workflow rather than a one-off. Needs a timestamp
watermark in `training_meta.json` and a corpus diff query.

---

## Tier 3 (strategic)

### T3.7 Model lineage / registry

New `models` table in the store: `(id, path, base_model, corpus_fingerprint,
disagreement_rate, ndcg_baseline, ndcg_final, trained_at)`. CLI:
`vstash models list`, `vstash models use <id>`. Answers "which model is best
for profile X?".

### T3.8 HF Hub push helper

`vstash retrain --push-to-hub stffens/bge-small-rrf-v3` plus auto model card
(training pairs, base model, evals, corpus description). Formalizes what is
already happening manually.

### T3.9 Distillation mode

Train small model to match a big model's top-K rankings on the user's
corpus. Different goal (speed under constant recall, not calibrated gains).
Loss: MarginMSE from cross-encoder scores.

---

## Execution order

1. **T1.1 first.** Without eval, every other change is blind. Foundation.
2. **T1.2 second.** Cheap, independent, multiplies data for every later run.
3. **T1.3 third.** Biggest quality lever, but only meaningful once T1.1 can
   prove gains.
4. Tier 2 gated on Tier 1 completion + a clean baseline eval on SciFact.
5. Tier 3 reopens after Tier 2 lands.

## Tracking

Tier 1 items tracked as in-session tasks. Check off in this doc as each
lands on `develop`.
