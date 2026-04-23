# Retrain -- fine-tune the embedding model on your own data

`vstash retrain` fine-tunes a base embedding model (default
`BAAI/bge-small-en-v1.5`) using your corpus's own
hybrid-retrieval disagreement signal. The result is an embedding
model that understands your data better: same speed, same
dimensions, higher retrieval NDCG@10.

No external labels, no GPU required to use the fine-tuned model,
no cloud calls. Training itself needs a GPU (T4 is enough); the
trained model then runs on CPU like any other embedding model.

**Two entry points:**

- [`vstash retrain`](#single-corpus-vstash-retrain) -- single
  store. For "tune bge-small on my notes / code / documentation".
- [`vstash retrain-multi`](#multi-corpus-vstash-retrain-multi) --
  N stores. For multi-domain personal retrieval (papers + code +
  notes) or for reproducing benchmark results with labeled
  queries (BEIR qrels etc.).

Published fine-tunes produced with this flow live on Hugging Face
under the `Stffens` namespace. **Current recommended model:
[`Stffens/bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3)**
(2026-04-19, `temperature=0.5 + total_triples=60000`). Previous
releases (`bge-small-rrf-v1`, `-v2`) remain valid.

---

## Track record

Retrain is not a one-off experiment. Each release is gated by an
honest NDCG@10 eval, validated on BEIR, and improves on the prior
version under apples-to-apples evaluation:

| Model       | Training recipe                                               | 5-dataset BEIR macro NDCG@10 | vs ColBERTv2 | Key signal                                    |
|-------------|---------------------------------------------------------------|------------------------------|--------------|-----------------------------------------------|
| base        | `BAAI/bge-small-en-v1.5` (no fine-tune)                       | 0.6118                       | 5/5          | reference                                     |
| rrf-v1      | Chunk-prefix disagreement, 1 hard neg per query               | (superseded)                 | 5/5          | first validation of the self-supervised idea  |
| rrf-v2      | Labeled queries + 76k triples, notebook + ad-hoc scripts      | 0.6246 (+0.013 vs base)       | 5/5          | first "paper-grade" result, still the NFCorpus specialist |
| **rrf-v3**  | `retrain-multi` CLI, 60k target, `temperature=0.5`, eval gate | **0.6405** (+0.029 vs base)  | **5/5**      | +0.016 macro over v2, +0.097 FiQA, +0.025 SciFact |

Each jump rests on validated infrastructure that also landed in the
codebase, not just the numbers:

- **v1 -> v2**: self-supervised pipeline moved from one-off scripts
  to `retrain`/`retrain-multi` entrypoints. Multi-dataset training
  (SciFact + NFCorpus + FiQA) and labeled-query mining arrived as
  a first-class API.
- **v2 -> v3**: the H-R9 ablation (temperature sweep + volume
  sweep) empirically chose the new defaults. H-R7 added seeded
  determinism so any re-run is reproducible; H-R5 added NDCG@3 and
  Recall@100 so regressions in head-quality and candidate-set
  health are visible before the user hits them. The published
  model ships under the same 33M / 384d footprint as v2.

Not every idea wins. Hypothesis H-R3 (hard-negative margin filter)
regressed macro NDCG@10 by -2.49pp in the 2026-04-19 arm_a
ablation: the infra was built, the eval gate caught the regression,
the branch was closed without merging. The retrain pipeline's job
is to refuse bad models, and it does.

Next steps that make this feature even stronger are queued:

- **T2.4 cross-encoder reranker** (design doc in
  `experiments/t24_reranker_design.md`): +3-8 NDCG@10 orthogonal
  to any embedding fine-tune. Expected to close v3's residual
  NFCorpus gap vs v2 and push every dataset further.
- **H-R8**: expose labeled queries + margin filter on single-corpus
  `retrain` so users with qrels over their own private store do
  not need to wrap their store in a dict.
- **T2.5 GISTEmbedLoss**: teacher-guided hard-negative filtering;
  drop-in replacement for MNRL where the assumption finally
  applies.

---

## Quick start

### Fine-tune on your own data

```bash
vstash retrain                                   # uses defaults
vstash reindex --model ~/.vstash/models/retrained
```

That's it. `retrain` samples pseudo-queries from your corpus,
generates disagreement-based training pairs, fine-tunes the base
model, evaluates on a held-out slice, and **refuses to save a
worse model** thanks to the NDCG@10 eval gate. `reindex` switches
your store to the new model.

### Fine-tune across multiple corpora

```bash
vstash retrain-multi \
    --store papers=/data/vstash-papers.db \
    --store code=/data/vstash-code.db \
    --store notes=/data/vstash-notes.db \
    --sampling-strategy temperature \
    --sampling-temperature 0.5 \
    --total-triples 30000 \
    --output ~/.vstash/models/multi-tuned
```

Temperature sampling prevents the largest corpus from dominating
the gradient budget; smaller corpora still receive enough signal.

---

## How it works

1. **Pseudo-queries.** Sample chunks and use the first 200
   characters of each as a pseudo-query. Or pass real labeled
   queries (BEIR-style qrels) via
   `--training-queries` on `retrain-multi`.
2. **Disagreement mining.** For each query, run two searches on
   your store: one vector-heavy (RRF weights 0.95/0.05) and one
   FTS-heavy (0.05/0.95). Top-K paths that appear in one but not
   the other are hard negatives.
3. **Triple assembly.** Emit `(query, positive, hard_neg)`
   triples. The positive is the query's source chunk (or the
   labeled gold doc's first chunk on `retrain-multi`).
4. **MNRL training.** Train with
   `MultipleNegativesRankingLoss` (SentenceTransformers) for
   2 epochs at `lr=3e-6`. In-batch negatives plus the explicit
   hard negative give a stronger gradient than in-batch alone.
5. **Eval gate.** Measure NDCG@10 on a held-out slice with the
   base model and with the fine-tuned candidate. If the delta
   falls below `--min-gain` (default 0.0, i.e. no regression),
   the candidate is left at `<output>.candidate/` for
   inspection and the user's active model is untouched.

---

## Single-corpus: `vstash retrain`

Simplest path. One store, one output model, one NDCG@10 check.

### Common flags

```bash
vstash retrain \
    --max-queries 5000 \         # pseudo-queries to generate
    --epochs 2 \                 # training epochs
    --lr 3e-6 \                  # learning rate
    --batch-size 64 \            # training batch size
    --eval-fraction 0.15 \       # held-out fraction for eval
    --eval-noise 1000 \          # distractor chunks in eval index
    --min-gain 0.0 \             # NDCG@10 improvement required
    --seed 42 \                  # deterministic runs
    --output ~/.vstash/models/retrained
```

`--quick` preset: 1 epoch, 1k queries, higher LR. Good smoke test
for a new corpus before committing to a full run.

### LLM-synthesized queries (optional)

Chunk-prefix pseudo-queries work well on technical corpora but
produce "statement-shaped" queries. If your corpus is
question-answering-heavy and you have a local LLM (Ollama / LM
Studio / any OpenAI-compatible server), add:

```bash
vstash retrain \
    --synthesize-queries \
    --synth-n 2 \
    --synth-cache ~/.vstash/synth_cache.jsonl
```

The LLM generates 2 short queries per chunk, cached so re-runs
are free. Empirically: synthesis is neutral on single-domain
corpora (tested on SciFact + NFCorpus, ~+0.2% NDCG@10), useful
mainly when combined with multi-dataset training.

### The eval gate

```
$ vstash retrain
...
Eval results
  Queries:          750
  Baseline NDCG@10: 0.7261
  Final NDCG@10:    0.7802
  Delta NDCG@10:    +0.0541
  Baseline NDCG@3:  0.6812
  Final NDCG@3:     0.7410
  Baseline MRR:     0.6929
  Final MRR:        0.7415
  Baseline Recall@100: 0.9400
  Final Recall@100:    0.9533

Model saved to: ~/.vstash/models/retrained
```

If the delta is negative or below `--min-gain`, `retrain` prints
the delta table, leaves the candidate at
`~/.vstash/models/retrained.candidate/` for inspection, and exits
non-zero. Your current model never gets overwritten without
passing the gate.

Override with `--no-eval` to skip the gate entirely (useful for
smoke tests). Override with `--min-gain -1` to always save the
candidate (useful when you want to inspect a known-regression
model for debugging).

---

## Multi-corpus: `vstash retrain-multi`

For heterogeneous use cases where a single flat corpus does not
fit:

1. **Multi-domain personal retrieval**: separate vstash stores
   for papers, blog posts, code, notes. Temperature sampling
   balances training signal so the largest corpus does not
   dominate.
2. **Benchmark / paper reproduction**: point `--store` at BEIR
   datasets with labelled qrels and pass them as
   `--training-queries`. This reproduces the v5 recipe that
   produced `Stffens/bge-small-rrf-v2` (+5% SciFact,
   +18.3% NFCorpus vs ColBERTv2 baselines).

### Typical invocation

```bash
vstash retrain-multi \
    --store papers=/data/vstash-papers.db \
    --store code=/data/vstash-code.db \
    --store notes=/data/vstash-notes.db \
    --sampling-strategy temperature \
    --sampling-temperature 0.5 \
    --total-triples 30000 \
    --epochs 2 --lr 3e-6 --batch-size 32 \
    --seed 42 \
    --output ~/.vstash/models/multi-tuned
```

The current profile's store is auto-included unless
`--exclude-primary` is passed.

### Sampling strategies

| Strategy | Formula | When to use |
|---|---|---|
| `uniform` | every corpus gets `total / N` triples | corpora of similar importance regardless of size |
| `proportional` | share = size | rarely useful; included for comparison |
| `temperature` | share proportional to `size ** alpha` | default. `alpha = temperature`, range `[0, 1]` |

`temperature=0.5` (default) is a sensible middle: larger corpora
get more share than equal but are damped vs proportional.
`temperature=0.3` gives noticeably more budget to small corpora
without starving the large ones. See the H-R9 ablation below for
empirical numbers.

### Bulk GPU paths (recommended on Colab / workstation)

```bash
vstash retrain-multi ... --bulk-mine --bulk-eval
```

- `--bulk-mine` batches triple mining via one GPU matmul.
  On a 57k-chunk corpus (BEIR FiQA), mining time drops from
  ~3 hours to ~2 minutes.
- `--bulk-eval` applies the same trick to baseline + final eval.
  3-corpus eval drops from ~60 min to ~2-5 min.

Both are trade-offs: the batched path skips the production
search pipeline's MMR dedup, IDF re-weighting, and distance
cutoff. Absolute NDCG@10 differs from the legacy path by a few
percent, but the baseline-vs-final delta is preserved because
both sides use the same eval. For paper-grade absolute numbers,
use the non-batched defaults.

### Labeled queries (v5 recipe)

When you have BEIR-style `qrels.jsonl`:

```python
from vstash.retrain import qrels_to_eval_queries, retrain_multi

eval_queries = {
    "scifact": qrels_to_eval_queries(scifact_queries, scifact_qrels, ...),
    "nfcorpus": qrels_to_eval_queries(nfcorpus_queries, nfcorpus_qrels, ...),
    "fiqa": qrels_to_eval_queries(fiqa_queries, fiqa_qrels, ...),
}

retrain_multi(
    stores=stores,
    base_model="BAAI/bge-small-en-v1.5",
    training_queries_by_dataset=eval_queries,   # v5 recipe
    eval_queries_by_dataset=eval_queries,
    bulk_mine=True, bulk_eval=True,
    total_triples=30000,
    output_path="~/.vstash/models/bge-small-rrf-custom",
)
```

This path produces paper-grade numbers. See
`experiments/retrain_t1_4_multi_beir.ipynb` for the full
end-to-end Colab notebook.

---

## Hyperparameter guide (with evidence)

All numbers below are from the 2026-04-19 three-arm ablation on
SciFact + NFCorpus + FiQA (see
`experiments/retrain_roadmap.md` for the source table).

### Sampling temperature

| temperature | NFCorpus share | SciFact | NFCorpus | FiQA | macro |
|---|---|---|---|---|---|
| 0.5 (default, 30k triples) | 17% | 0.7786 | 0.3677 | 0.4568 | 0.5344 |
| 0.3 (30k) | 26% | 0.7791 | 0.3732 | 0.4431 | 0.5318 |
| 0.0 (uniform, 30k) | 33% | 0.7765 | 0.3809 | 0.4222 | 0.5265 |

**Takeaway:** temperature is a controllable trade-off. Lower it
to lift small-corpus NDCG at the cost of large-corpus NDCG. The
knob is monotonic and predictable, so pick the value that matches
your priority:

- `0.5` (default): balanced mix of different-sized corpora.
- `0.3`: recommended when one corpus is < 10% of the total and
  you want to protect it.
- `0.0` (uniform): treat every corpus as equally important
  regardless of size. Only when all corpora matter equally and
  you accept the large-corpus regression.

### `total_triples`

| total | NFCorpus pairs | SciFact | NFCorpus | FiQA | macro |
|---|---|---|---|---|---|
| 30000 (default) | 4856 | 0.7786 | 0.3677 | 0.4568 | 0.5344 |
| 60000 (2x) | 9712 | 0.7818 | 0.3757 | 0.4818 | **0.5464** |

**Takeaway:** at the 30-60k scale, volume is a free lunch. All
three datasets improved simultaneously when doubling
`total_triples`. No observable trade-off. Recommended for any
3+ corpus mix where training time is not a hard constraint
(~55 min on Colab T4 for 60k triples).

### Other knobs

Defaults (verified equivalent to the v5 notebook):

- `epochs=2`, `lr=3e-6`, `warmup_steps=min(50, len(loader)//5)`.
- `batch_size=64` (single-corpus `retrain`) or `32`
  (`retrain-multi`, T4-safe when three corpora are live in eval).
- `use_amp=True` (FP16 mixed precision; halves GPU memory).
- `max_seq_length=None` (model default, 512 for bge-small).

Change these only if you know why. Specifically:

- `--batch-size 128` doubles throughput on A100+ but changes
  effective learning rate. If you raise batch size, scale LR
  sublinearly (`sqrt(k)` rule).
- `--epochs 3+` sometimes helps on very small corpora; normally
  2 is enough.
- `--max-seq-length 128` saves memory on short-chunk corpora
  without measurable quality impact.

---

## When NOT to retrain

- **Your corpus is small** (< 500 chunks). Not enough disagreement
  signal. `retrain` will warn and exit. Either ingest more
  content or skip this step and use the published
  `Stffens/bge-small-rrf-v2` directly.
- **Your corpus is already in the published model's training
  mix** (SciFact, NFCorpus, FiQA as BEIR datasets). Use
  `Stffens/bge-small-rrf-v3` directly -- no additional gain
  expected from retraining.
- **You rely on strict cross-lingual retrieval.** The MNRL signal
  is based on monolingual disagreement; fine-tuning can hurt
  cross-lingual NDCG. Consider a multilingual model instead
  (`intfloat/multilingual-e5-large`).

---

## Publishing a trained model

Manual path (the automated `--push-to-hub` helper is T3.8 on the
roadmap, not yet shipped):

```bash
# After training succeeds
huggingface-cli login
huggingface-cli upload YourHFName/my-model-v1 ~/.vstash/models/retrained .
```

Include a model card with:

- Base model: `BAAI/bge-small-en-v1.5`.
- Training corpus description (size, domain, languages).
- Hyperparameters (lr, epochs, batch, total_triples, sampling).
- `training_meta.json` contents (vstash writes this at
  `<output>/training_meta.json`).
- BEIR or domain-specific eval numbers if available.

---

## Troubleshooting

### `Training pairs: 0`

The disagreement mining found nothing. Either:

- Your corpus is highly homogeneous (vec and FTS agree on every
  query).
- `max_queries` is too low (raise it).
- FTS5 index is empty or broken (check `vstash check`).

### `Candidate gated out`

The fine-tuned model is worse than the base on your held-out
slice. Don't override with `--min-gain -1` without investigating.
Likely causes:

- Too-small eval split (`--eval-fraction` below 0.10 on a small
  corpus means few queries to judge by; raise it).
- Overfitting (try `--epochs 1` or lower `--lr`).
- Corpus distribution mismatch (training queries do not look like
  real queries -- consider `--synthesize-queries`).

Inspect the candidate at `<output>.candidate/training_meta.json`
for the per-dataset numbers.

### Margin-filter warning: "kept only X% of pairs"

Only applicable with `--margin-min` / `--margin-max` (currently
only on a closed-PR branch, not on develop). If you see this on
your own fork and X < 30%, widen the band. On the base
`bge-small`, cos(q, gold) / cos(q, hard_neg) margins sit at
0.02-0.08, so cutoffs above 0.05 remove real training signal
rather than noise.

---

## Internal references

- `experiments/retrain_roadmap.md` -- engineering roadmap + all
  validation runs.
- `experiments/hypotheses.md` -- living backlog of retrain +
  search hypotheses.
- `experiments/retrain_t1_4_multi_beir.ipynb` -- full Colab
  notebook reproducing `Stffens/bge-small-rrf-v2`.
- `experiments/retrain_t1_5_hr9_balance_ablation.ipynb` -- the
  three-arm ablation behind the temperature and volume tables
  above.
- `vstash/retrain.py`, `vstash/retrain_batch.py`,
  `vstash/retrain_synth.py` -- source.
- `Stffens/bge-small-rrf-v3` on Hugging Face -- current recommended
  fine-tune (2026-04-19, H-R9 winning config).
- `Stffens/bge-small-rrf-v2` on Hugging Face -- previous fine-tune,
  still valid.
