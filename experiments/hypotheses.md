# vstash improvement hypotheses

Living backlog of concrete, testable hypotheses to raise vstash retrieval
quality or reduce cost. Spun off from `retrain_roadmap.md` after the T1.4
first-Colab regression (-5.15% macro NDCG@10) and the T1.5 fix landing.

Each hypothesis is:

- **Statement**: what we believe will happen.
- **Test**: how we measure it, with a clear success bar.
- **Files**: where the change lives.
- **Effort**: rough day count.
- **Risk**: what breaks if the hypothesis is wrong.

Ordered by expected ROI / day. Strike through when shipped. Link the PR.

## Methodological note: eval pipeline shifts absolute baselines

Between the 2026-04-18 and 2026-04-19 retrain runs, PR #243 merged
changes to `evaluate_model` + `evaluate_model_batched` (widened
top-K pool from 10 to 100 to support Recall@100, plus reinforced
doc-level dedup). These are semantically correct but they raise
absolute baseline NDCG@10 by ~0.01-0.02 per dataset because the
base model now has a wider candidate pool to surface relevant docs
from.

Consequence for ablation reading:

- **Delta %** between baseline and final depends on the eval
  pipeline. Not comparable across PR-#243 pre/post or across
  independent evaluators.
- **Final absolute NDCG@10** is a genuine model-quality signal.
  Prefer this when comparing arms or quoting published numbers.
- When putting our numbers next to v5's published
  +5% SciFact / +18.3% NFCorpus, make the pipeline caveat
  explicit. Our final absolute NDCG@10 on SciFact/FiQA already
  matches or beats v5; the remaining NFCorpus gap is a
  model-quality question (H-R9), not a measurement artefact.

Keep this caveat in mind when reading the T1.5 / H-R9 tables
further down this document and in
`experiments/retrain_roadmap.md`.

---

## Retrain (signal quality + training)

### H-R1. Labeled queries should be the default path when qrels exist

**Statement.** `retrain_multi(..., training_queries_by_dataset=...)` currently
opts in to the v5 recipe. If the user passes eval_queries with BEIR-style
qrels, we can reuse them for training automatically. Chunk-prefix fallback
should only fire when no labels exist. Removes the footgun that caused the
2026-04-18 -5.15% run.

**Test.** Re-run `experiments/retrain_t1_4_multi_beir.ipynb` with the new
default path (no explicit flag). Target: macro NDCG@10 delta >= the explicit
T1.5 run on the same seed.

**Files.** `vstash/retrain.py` (retrain_multi entrypoint), `vstash/cli.py`
(`vstash retrain-multi` wiring).

**Effort.** 1 day.

**Risk.** Users who relied on chunk-prefix mining on a labeled corpus get
different training data silently. Mitigate with a `training_pair_source`
parameter that defaults to `"auto"` and logs the choice.

---

### H-R2. Multi-triplet emission on labeled queries (T1.2 revisited)

**Statement.** `generate_labeled_triples_batched` already emits
`|gold| * |hard_neg|` triples per query. The legacy unlabeled path
(`generate_triples`, `generate_triples_batched`) still emits 1 triplet per
query. When T1.5 validates, porting multi-triplet to the prefix path should
multiply signal 2-3x without new infrastructure.

**Test.** Ablation on SciFact + NFCorpus: chunk-prefix T=1 vs T=3 vs T=5
hard negs at constant `total_triples`. Track NDCG@10 gate outcome and
training wall-clock. Target: T=3 wins or matches T=1 on macro NDCG@10,
within noise (+/- 0.5%).

**Files.** `vstash/retrain.py:226-244`, `vstash/retrain_batch.py:360-404`.

**Effort.** 2 days.

**Risk.** T1.2 was parked because multi-triplet on chunk-prefix queries
amplified the distribution mismatch. Only revisit after T1.5 validates.

---

### H-R3. Hard-negative margin filter [TRIED, NEGATIVE; PR #245 CLOSED]

**Original statement.** Filter `(gold, hard_neg)` pairs by cosine
margin to remove ambiguous-close and too-easy tails; expected
+0.5-1% macro NDCG@10.

**Outcome (2026-04-19).** Infra was built on PR #245 (`margin_min`
/ `margin_max` on `generate_labeled_triples_batched`, threaded
through `retrain_multi` + CLI, 5 unit tests, ablation notebook).
Ablation arm_a on Colab (`margin_min=0.05`, no upper bound)
regressed macro -2.49pp (+1.65% vs +4.14%) with 53% of pairs
filtered out. Arms b/c aborted (kernel disconnect + strong prior
of further regression).

**Diagnosis.** On the base bge-small the margin distribution
`cos(q, gold) - cos(q, hard_neg)` concentrates at 0.02-0.08,
because both are high-but-not-saturated (gold ~0.85-0.92, neg
~0.80-0.88). Cutting below 0.05 removes hard-neg signal, not
noise. The filter's mental model ("margin < 0.05 = neg is
ambiguous / probably relevant") holds only for an already-trained
model where `cos(q, gold)` saturates near 1.0.

**Decision.** PR #245 closed without merge. Infra stays in git
history on `feat/retrain-hr3-margin-filter` and can be
cherry-picked when a trained-model path needs it (continual
retrain T2.6; cross-encoder hard-neg filtering T2.4). Pivot
NFCorpus gap to H-R9 below.

**Files.** Git log on closed branch
`feat/retrain-hr3-margin-filter` (not on develop).

---

### H-R4. Hyperparameters as CLI knobs + small grid on Colab (deprioritized)

**Statement.** Originally framed as "v5 used lr=2e-5 / epochs=1, ours is
3e-6 / epochs=2, that is the gap". Verified against
`experiments/retrain_v5_hard_neg.ipynb`: v5 actually uses
`lr=3e-6, epochs=2, warmup_steps=50, batch=64` -- the same defaults we
ship. The regression source is elsewhere (signal quality, T1.5 recipe
fix). Warmup is the only genuinely hardcoded knob left
(`min(50, len(loader) // 5)`) and matches v5 at the T1.4 scale
(len(loader)/5 >= 50 for any run over ~250 pairs/batch).

**Keep around for:** small-corpus users (< 250 pairs, where warmup would
fire early) and future ablation experiments. Not urgent.

**Files.** `vstash/retrain.py:267-347` (`train_mnrl`), `vstash/cli.py`.

**Effort.** 1 day code.

**Risk.** Defaults already validated; changes invite regressions.

---

### H-R5. Eval surface: add NDCG@3, Recall@100 [IMPLEMENTED, not yet merged]

**Statement.** `EvalMetrics` now tracks NDCG@10, NDCG@3, MRR, Hit@10,
Recall@100 (N.B. per-dataset std across `--eval-seed` values is still on
the todo list -- see follow-up below). Both new fields default to 0.0 for
backward compatibility with positional construction.

**Status.** Implemented in `vstash/retrain.py` (EvalMetrics + evaluate_model)
and `vstash/retrain_batch.py` (evaluate_model_batched uses top-100 matmul
and doc-level dedup up to 100). CLI retrain + retrain-multi print the new
metrics. 9 new tests cover _recall_at_k math, NDCG@3 truncation,
EvalMetrics shape, and end-to-end perfect-retrieval invariants.
Full suite: 965 tests passing.

**Follow-up.** Per-dataset std across multiple `--eval-seed` values is
useful for catching noisy wins; deferred to H-R5b.

**Files.** `vstash/retrain.py`, `vstash/retrain_batch.py`, `vstash/cli.py`,
`tests/test_retrain.py`.

---

### H-R6. Dedup `generate_triples` and `generate_triples_batched`

**Statement.** The non-batched and batched miners are 90% the same logic
with subtly different FTS pooling, RRF weights, and synth-query handling.
Divergence is a latent bug source (T1.4 review caught one; there may be
more). Refactor to a single template with a device-selectable backend.

**Test.** Port both paths to a shared core. Re-run full retrain test suite
(115 tests). Byte-for-byte compatibility of emitted triples on a fixed
seed + corpus fixture.

**Files.** `vstash/retrain.py:100-260`, `vstash/retrain_batch.py:162-404`.

**Effort.** 2-3 days.

**Risk.** Medium. Keep both paths callable during migration; delete the
legacy one only after 1 release cycle.

---

### H-R7. Seed audit + global `--seed` flag [IMPLEMENTED, not yet merged]

**Statement.** Every RNG that retrain training touches is now seeded from
a single user-controllable root. `train_mnrl` pre-shuffles examples with
`random.Random(seed)` (stable initial order), calls `torch.manual_seed` +
`torch.cuda.manual_seed_all` (covers dropout + optimizer init), and
passes a seeded `torch.Generator` to `DataLoader` (reproducible
per-epoch reshuffles). `retrain()` and `retrain_multi()` thread the
seed into `train_mnrl` on every call site. CLI `--seed` flag on both
`retrain` and `retrain-multi`. `training_meta.json` records the seed.

**Status.** Implemented. 3 new tests cover `torch.manual_seed` call,
`Generator` threading into `DataLoader`, same-seed deterministic
example ordering, and `training_meta.json` round-trip. Test stubs
extended in `test_retrain.py` and `test_retrain_multi.py`. 965+ tests
still passing.

**Files.** `vstash/retrain.py`, `vstash/cli.py`, `tests/test_retrain.py`,
`tests/test_retrain_multi.py`.

---

### H-R8. Expose labeled queries + margin filter on single-corpus `retrain`

**Statement.** Today the two most powerful levers of the retrain
stack (labeled-query training pairs, hard-neg margin filter) live
only on `retrain-multi`. A single-corpus user with qrels has to
wrap their one store in a one-element dict to reach the batched
labeled miner. API-shaped gap.

**Design.** Add `training_queries: list[dict] | None = None`,
`margin_min`, `margin_max`, `bulk_mine`, `bulk_mine_device` to
`vstash.retrain.retrain(...)`. When `training_queries` is set,
route through `generate_labeled_triples_batched`. CLI: add
`--training-queries path/to/qrels.jsonl`, `--margin-min`,
`--margin-max`, `--bulk-mine`.

**Test.** Re-run the retrain suite; add a unit test that feeds a
labeled set into `retrain()` and asserts the labeled miner was
called.

**Files.** `vstash/retrain.py`, `vstash/cli.py`,
`tests/test_retrain.py`.

**Effort.** 1 day.

**Risk.** Low. Purely additive to an existing entrypoint; defaults
preserve the current single-corpus flow byte-for-byte.

---

### H-R9. Close NFCorpus gap via corpus balance (temperature + triples)

**Statement.** After H-R3 ruled out hard-neg quality as the cause
of the NFCorpus gap, the two remaining suspects are corpus balance
and training volume. v5 trained 76k pairs total; we trained 28k.
At `temperature=0.5`, NFCorpus received 4856 pairs (17% of our
budget); at `temperature=0.3` it would get ~26%, and at
`sampling=uniform` it would get 33%. Doubling `total_triples` to
60k scales every dataset's contribution proportionally.

**Test (three-arm Colab ablation, pure CLI).**

- arm_t03: `--sampling-temperature 0.3`, `--total-triples 30000`.
  Same volume as T1.5 baseline, different balance. Isolates
  temperature effect.
- arm_uniform: `--sampling-strategy uniform`,
  `--total-triples 30000`. Maximum NFCorpus share; upper bound on
  the balance lever.
- arm_vol: `--sampling-temperature 0.5`, `--total-triples 60000`.
  Same balance as baseline, more volume. Isolates volume effect.

Target: NFCorpus final absolute NDCG@10 > 0.38 on at least one
arm, without regressing SciFact (stay above 0.775) or FiQA
(stay above 0.45).

**Files.** No code change. Notebook at
`experiments/retrain_t1_5_hr9_balance_ablation.ipynb`.

**Effort.** 0.5 day notebook + ~1.5h Colab.

**Risk.** Low. Pure hyperparameter sweep; no code to regress.

**Follow-ups if this closes the gap.**

- Publish `Stffens/bge-small-rrf-v3` with the winning config.
- Update README + paper with new numbers.
- Document temperature recommendation per corpus size in the
  `retrain-multi` help.

---

## Store / search

### H-S1. Persist IDF cache to a `store_idf` table

**Statement.** `_build_idf_cache()` runs on every `search(adaptive_rrf=True)`
call with an empty cache. For 100k+ chunks the fts5vocab scan is not free.
Persist the cache in SQLite, invalidate on any write (same trigger as FTS5
rebuild), skip the scan on the read path.

**Test.** Benchmark 1000 queries on a 100k-chunk store (BEIR FiQA). Target:
>= 10% wall-clock reduction at p50, no regression at p99.

**Files.** `vstash/store.py:3741-3805`
(`_build_idf_cache`, `_compute_adaptive_rrf_params`).

**Effort.** 2 days.

**Risk.** Stale cache after an ingest. Mitigate by keying on `chunk_count`
and a cheap FTS5 `integrity-check` watermark.

---

### H-S2. Adaptive MMR lambda

**Statement.** `mmr_lambda=0.5` is fixed. When the candidate set is almost
all one document, we want aggressive diversity (lambda closer to 0.3). When
docs are already distinct, we want relevance preservation (lambda closer
to 0.7). Compute lambda per query from the doc_id entropy of the pre-MMR
ranked list.

**Test.** BEIR 5-dataset ablation: fixed 0.5 vs adaptive. Target:
+0.3-1.0% macro NDCG@10 on heterogeneous corpora.

**Files.** `vstash/store.py:1672` (default), `vstash/store.py:2658-2810`
(`_mmr_dedup`), `vstash/store.py:2184` (search call site).

**Effort.** 2 days.

**Risk.** Extra complexity in a load-bearing function. Ship behind
`[scoring] mmr_adaptive = false` default-off until validated on BEIR.

---

### H-S3. FTS candidate pool sized from query IDF

**Statement.** `_fuse_rrf_scores()` gates FTS-only hits with
`rank < effective_k * 2`. For long rare-term queries this under-samples FTS
(which is where rare terms win). Size the FTS pool dynamically from the
harmonic-mean IDF of query terms.

**Test.** BEIR tail queries (> 20 words, < 5 common terms). Target:
+0.5% NDCG@10 on tail slice.

**Files.** `vstash/store.py` search pipeline around `_fuse_rrf_scores`
(ref line 1534 in pre-T1.5 diff; re-locate before implementing).

**Effort.** 1-2 days.

**Risk.** Widening the FTS pool raises per-query cost. Cap at 4x
`effective_k`.

---

### H-S4. Batch the MMR similarity update

**Statement.** `_mmr_dedup()` recomputes `max_sims` one chunk at a time in
Python. For large `top_k` (>= 100) and high-duplication corpora this is
O(N*K) Python. A precomputed all-pairs cosine matrix + numpy indexed
updates should be 5-10x faster.

**Test.** Bench on a synthetic 200-candidate / 50-deduplicate case. Target:
>= 5x speedup at top_k=100, no ranking diff.

**Files.** `vstash/store.py:2658-2810`.

**Effort.** 2 days.

**Risk.** Memory (K^2 floats). For K=1000, 8 MB. Acceptable.

---

### H-S5. Distance-cutoff from query-embedding entropy

**Statement.** `_compute_adaptive_rrf_params()` returns a fixed cutoff
(1.15 default, 5.0 for long queries). A query whose embedding is diffuse
across the vector space (high entropy of top-100 similarities) should get
a relaxed cutoff; a peaky embedding (one cluster) should get a tight one.

**Test.** BEIR 5-dataset recall@100 ablation. Target: +0.3-0.7% recall@100
on ambiguous queries (bottom-decile peak similarity).

**Files.** `vstash/store.py` (`search`, `_compute_adaptive_rrf_params`).

**Effort.** 2-3 days.

**Risk.** Speculative; may pay nothing. Keep behind a config flag.

---

## Roadmap next steps (after Tier 1 lands)

### H-T24. Cross-encoder reranker (T2.4 in roadmap)

**Statement.** A small cross-encoder (ms-marco-MiniLM-L-6-v2, ~20 MB) on
top of the current bi-encoder + adaptive-RRF stack should add 3-8 NDCG@10
reliably. Orthogonal to our retrain work, biggest product win per effort.

**Test.** Add `rerank()` stage after MMR, gated by `store.search(rerank=True)`
and `[rerank]` config. Target: +3% NDCG@10 on SciFact, no regression on
NFCorpus, < 100ms extra latency at top_k=50 on M1 CPU.

**Files.** New `vstash/rerank.py`; wire into `store.py:search`.

**Effort.** 1 week (~3 days model wiring, 2 days bench + tests).

**Risk.** Latency. Cap at `rerank_top_k=50`. Skip automatically when
search has < 20 candidates.

---

### H-T25. GISTEmbedLoss replacement for MNRL (T2.5)

**Statement.** GISTEmbedLoss (teacher-guided hard-negative filtering) is
2024 SOTA for small embedding models. Drop-in in `train_mnrl`, expose
`--loss {mnrl, gist}`.

**Test.** Re-run T1.5 notebook with `--loss gist`, teacher
`sentence-transformers/all-mpnet-base-v2`. Target: +1-3% macro NDCG@10 vs
MNRL at same budget.

**Files.** `vstash/retrain.py:train_mnrl`, `vstash/cli.py`.

**Effort.** 4-5 days (incl. teacher-model memory planning on T4).

**Risk.** Teacher model is 420 MB, needs careful batch sizing on T4 next
to the student. Fallback path: teacher on CPU.

---

## Tracking

- Shipped items move to `retrain_roadmap.md` with final numbers + PR.
- If an H-R fails its test cleanly, convert to a "tried, negative"
  footnote so we don't repeat the experiment.
- Re-prioritise monthly against the current macro NDCG@10 gap from
  `Stffens/bge-small-rrf-v2` published numbers.

Started 2026-04-19 after T1.5 landed. Owner: Jay + Claude.
