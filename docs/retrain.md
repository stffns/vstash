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

### Labeled training and eval queries (BEIR-style)

When you have real `(query, relevant_doc_paths)` labels --
BEIR qrels, search-log clicks, manual annotations -- pass them
to `vstash retrain` directly. Both flags accept JSONL files
where each line is one labeled query:

```jsonl
{"query": "what is the meaning of life", "relevant_paths": ["/notes/answer.md"]}
{"query": "how do mitochondria work", "relevant_paths": ["/biology/cell.md", "/biology/atp.md"]}
```

```bash
vstash retrain \
    --training-queries train.jsonl \
    --eval-queries eval.jsonl \
    --base-model BAAI/bge-small-en-v1.5 \
    --output ~/.vstash/models/my-fine-tune
```

What changes vs. the default flow:

- **Training pairs** come from real (query, gold) labels via
  `generate_labeled_triples_batched` instead of chunk-prefix
  pseudo-queries. This reuses the v5 recipe that produced
  `Stffens/bge-small-rrf-v2` and `bge-small-rrf-v3`. Pass
  `--training-pair-source labeled` to refuse to fall back to
  chunk-prefix when no training labels are present (forces a
  hard error rather than silent regression).
- **Eval gate** scores baseline and fine-tuned on `eval.jsonl`
  rather than the internal chunk-prefix split. This matters
  whenever you care about specific query distributions: the
  internal split's pseudo-queries are statement-shaped, so a
  +0.05 NDCG there can hide a regression on real questions.
- `--eval-fraction` is ignored (the held-out set is the file
  you provide). `--eval-noise` still applies and controls how
  many distractor chunks join the eval index.
- `--no-eval` and `--eval-queries` are mutually exclusive: pick
  one. If both are set the CLI exits non-zero.

#### Building a stratified holdout from BEIR-style qrels

`vstash.retrain.qrels_to_eval_queries` converts the standard
`(queries, qrels)` dicts into the JSONL shape above:

```python
import json
from vstash.retrain import qrels_to_eval_queries

queries = {"q1": "what is the meaning of life", ...}
qrels   = {"q1": {"answer.md": 1}, ...}

records = qrels_to_eval_queries(queries, qrels, path_for_doc_id=lambda d: d)

# 80/20 split
train, holdout = records[: int(len(records) * 0.8)], records[int(len(records) * 0.8) :]
with open("train.jsonl", "w") as fh:
    for r in train: fh.write(json.dumps(r) + "\n")
with open("eval.jsonl", "w") as fh:
    for r in holdout: fh.write(json.dumps(r) + "\n")
```

If your data is multi-domain (e.g., LongMemEval question types),
stratify the split by the relevant key before slicing -- a
random 80/20 over a heterogeneous benchmark hides per-category
regressions. The Python API (`retrain(..., eval_queries=...)`)
also accepts the records directly without writing JSONL.

#### Working from chat-style JSON (e.g., LongMemEval)

`vstash` does not ship a dedicated chat-JSON ingester -- the
schema varies too much across products. The pattern is:

1. **Reduce each chat session to a text blob** (one document).
   Concatenating turns with role markers
   (`[USER]\n...\n[ASSISTANT]\n...`) is a reasonable default;
   richer schemas may need timestamps or tool-call markup.
2. **Use the session id as the document path** so labeled
   queries can reference it directly in `relevant_paths`.
3. Ingest the blobs via the standard `Memory.remember` /
   `VstashStore.add_documents_batch` paths.

```python
# Sketch: turns -> blob -> doc, one per session
def _format_session(turns: list[dict]) -> str:
    return "\n\n".join(f"[{t['role'].upper()}]\n{t['content']}" for t in turns)


for session_id, turns in chat_sessions.items():
    text = _format_session(turns)
    chunks = chunk_text(text, cfg.chunking.size, cfg.chunking.overlap)
    embeddings = embed_texts(chunks, cfg.embeddings.model)
    store.add_documents_batch(
        [
            {
                "path": session_id,
                "title": session_id,
                "chunks": chunks,
                "embeddings": embeddings,
                "source_type": "chat",
                "collection": "my-chats",
            }
        ]
    )
```

The labeled-query JSONL then references each `session_id`
directly in `relevant_paths`.

### Case study: LongMemEval chat memory (+8pp NDCG@10 in 30 minutes)

We validated the labeled-retrain workflow end-to-end on LongMemEval-s,
a public chat-memory benchmark (Wu et al., 2024) with 500 questions
across six question types and ~115k tokens of haystack per question.

**Setup.** Stratified 80/20 split by `question_type` with a
deterministic seed: 398 train queries, 102 holdout. The 102-query
holdout is disjoint from training; the eval gate scores baseline +
candidate on it. ~25k unique haystack sessions ingested into one
vstash corpus, with paths namespaced as `{question_id}:{session_id}`
to handle the ~4300 sessions LongMemEval reuses across questions.

**Eval gate (102-query holdout, disjoint from train):**

| Model                             | NDCG@10 | Delta vs base |
|-----------------------------------|--------:|--------------:|
| BAAI/bge-small-en-v1.5 (vanilla)  | 0.6143  |   --          |
| Stffens/bge-small-rrf-v3          | 0.5898  | -0.0245       |
| **bge-small-rrf-lme-v1**          | 0.6878  | **+0.0735**   |

The chat fine-tune lifts NDCG@10 by 11.85% relative over vanilla
BGE. Two seeded runs with the same data and seed=42 produced 0.6872
and 0.6878 (delta 0.0006), confirming reproducibility. The gate
auto-passes both candidates over the `min_gain=0.0` threshold.

**Surprising finding.** Stffens/bge-small-rrf-v3, which we trained on
BEIR (SciFact + NFCorpus + FiQA) and which beats vanilla BGE on those
benchmarks, **loses** to vanilla on LongMemEval (-0.0245 NDCG@10,
-0.028 R@5 on temporal-reasoning specifically). This is the first
clean in-tree evidence that "best on BEIR" does not transfer to
"best on chat memory" -- domain matters more than absolute model
quality. The eval gate would have rejected v3 as a chat-memory
retriever even though it is a strict improvement on BEIR.

**Where the lift concentrates (per-type R@5, holdout):**

| Type                        | n  | base   | lme-v1 | delta    |
|-----------------------------|----|--------|--------|----------|
| single-session-user         | 14 | 0.929  | 0.929  |  0.000   |
| single-session-assistant    | 12 | 1.000  | 1.000  |  0.000   |
| single-session-preference   |  6 | 1.000  | 1.000  |  0.000   |
| **multi-session**           | 27 | 0.869  | 0.938  | **+0.069** |
| knowledge-update            | 16 | 0.969  | 1.000  | +0.031   |
| **temporal-reasoning**      | 27 | 0.774  | 0.829  | **+0.056** |

The chat fine-tune does its work where the failure analysis
predicted: multi-session (multiple gold sessions to surface) and
temporal-reasoning (cross-session date resolution). Single-session
types saturate at R@5 even on vanilla -- there was no headroom to
move there.

**Mechanism.** Pre-fine-tune failure analysis showed gold chunks
sitting at cosine distance 0.46-0.61 to the query (geometrically far
in embedding space). The fine-tune compresses (query, gold-chunk)
pairs in cosine space so the same gold sessions rank higher in
hybrid RRF. R@50 stays at 1.000 across all arms -- the embedder
already retrieves the gold; ranking is the lever, not coverage.

**Reproducing the case study end-to-end:**

```bash
# 1. Data prep (~9 min, Mac local) -- ingests haystacks + emits JSONLs
python -m experiments.lme_prepare_retrain \
    --output-db    experiments/lme_retrain_full.db \
    --output-train experiments/results/lme_train.jsonl \
    --output-eval  experiments/results/lme_eval.jsonl \
    --output-meta  experiments/results/lme_retrain_meta.json

# 2. Train on Colab T4 (~12 min) with the eval gate
VSTASH_DB_PATH=experiments/lme_retrain_full.db vstash retrain \
    --training-queries experiments/results/lme_train.jsonl \
    --eval-queries     experiments/results/lme_eval.jsonl \
    --base-model       BAAI/bge-small-en-v1.5 \
    --output           ~/.vstash/models/bge-small-rrf-lme-v1 \
    --bulk-mine --bulk-mine-device cuda

# 3. Score the trained model on the full LongMemEval-s for the
#    appendix table (Mac local, ~9 min)
python -m experiments.longmemeval_retrieval --all \
    --model ~/.vstash/models/bge-small-rrf-lme-v1 \
    --output experiments/results/lme_full_500_lme-v1.json
```

The full 4-arm comparison + reproducibility scripts are in
`experiments/`; the canonical paragraph + auditable tables for paper
v2 live at `paper/v2/chat_memory_paragraph.md`.

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
    training_queries_by_dataset=eval_queries,  # v5 recipe
    eval_queries_by_dataset=eval_queries,
    bulk_mine=True,
    bulk_eval=True,
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
