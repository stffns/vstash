---
language:
- en
license: mit
library_name: sentence-transformers
base_model: BAAI/bge-small-en-v1.5
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- retrieval
- bge
- beir
- vstash
- mnrl
- fine-tuned
datasets:
- BeIR/scifact
- BeIR/nfcorpus
- BeIR/fiqa
---

# bge-small-rrf-v4

A 33M-parameter (384-dim) English embedding model, fine-tuned from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
using [vstash](https://github.com/stffns/vstash)'s self-supervised
hybrid-retrieval disagreement signal.

**Reproduces the v3 training recipe on the post-v0.34 vstash pipeline.**
Same size, same speed, same recipe; trained against the corrected
cosine distance metric instead of the buggy L2-as-cosine that pre-v0.34
applied. Use this as the "current code" reference. v3 remains valid
as the pre-v0.34 reference snapshot.

## Why v4 exists

The v0.34 release (`vstash` 0.34.0) fixed a latent bug in the
`vec_chunks` table: `sqlite-vec`'s `vec0(embedding float[N])` defaults
to L2 distance, but every comment, threshold, and telemetry field in
vstash labelled the value "cosine distance." Worked accidentally for
unit-normalized BGE because L2 and cosine are monotonically related,
but the `distance_cutoff` and `relevance_tier` thresholds were
calibrated against L2 ratios while the documentation and downstream
code assumed cosine. v0.34 rescaled both (1.15 -> 1.3225 for the
default cutoff; 0.95/0.98 -> 0.4513/0.4802 for the tier thresholds)
to match the squared cosine convention.

`bge-small-rrf-v3` was trained before that fix landed. v4 retrains
the same recipe (`arm_vol` from the H-R9 ablation: `temperature=0.5`,
`total_triples=60000`, MNRL, 2 epochs at lr=3e-6, batch=64, seed=42)
on the post-v0.34 vstash code so the training mining runs through the
corrected pipeline.

Empirically the difference is small. On the 5-dataset BEIR macro
v4 vs v3 lands at delta = -0.0057 NDCG@10 -- within the calibration
band. Per-dataset, v3 wins on the 3 training datasets (scifact,
nfcorpus, fiqa) and v4 wins on the held-out 2 (scidocs, arguana),
consistent with v3 having mildly overfit to a metric quirk that v4's
training data does not contain. Reproduction notebook is at
[`experiments/retrain_v4_post_v034_validation.ipynb`](https://github.com/stffns/vstash/blob/experiment/retrain-v4-post-v034-validation/experiments/retrain_v4_post_v034_validation.ipynb)
in the vstash repo; the pre-committed decision rule applied during
the ablation is at
[`experiments/results/asymmetry_decision_2026_04_28.md`](https://github.com/stffns/vstash/blob/experiment/retrain-v4-post-v034-validation/experiments/results/asymmetry_decision_2026_04_28.md).

## Eval numbers

vstash hybrid pipeline (adaptive RRF + FTS5 + MMR dedup), v0.35 code,
sentence-transformers backend.

| Dataset  | base BGE-small | bge-small-rrf-v4 | Delta |
|----------|---------------:|-----------------:|------:|
| SciFact  |         0.7251 |           0.7575 | +0.0324 |
| NFCorpus |         0.3591 |           0.3707 | +0.0116 |
| FiQA     |         0.3917 |           0.4455 | +0.0538 |
| SciDocs  |         0.1945 |           0.1971 | +0.0026 |
| ArguAna  |         0.4367 |           0.4386 | +0.0019 |
| **macro** |     **0.4214** |       **0.4419** | **+0.0205 (+4.9%)** |

Comparison notes:
- Base BGE-small here is `BAAI/bge-small-en-v1.5` evaluated through
  the same vstash pipeline. Numbers above are the Colab T4 ST-CUDA
  measurement of v4. A local Mac ST-CPU re-measurement is pending and
  will be posted as a sidecar; the FastEmbed (ONNX) backend on current
  code is ~+0.003 NDCG@10 vs ST.
- vstash hybrid + base BGE on the v0.35 pipeline already exceeds
  ColBERTv2 published macro (0.4214 vs 0.402 = +4.8%). The fine-tune
  adds another +4.9% on top. Pipeline contribution dominates; the
  fine-tune is a modest edge for domain-specific use.
- v4 vs ColBERTv2 published per dataset: SciFact +9.3%, NFCorpus
  +7.8%, FiQA +25.1%, SciDocs +28.0%, ArguAna -5.3%.

## When to use v4

- General BEIR-style retrieval where you want the post-v0.34 reference
  weights. Recommended default for new vstash deployments.
- Reproducing the paper v2 benchmark numbers under current code.
- As a baseline for further fine-tuning via `vstash retrain`.

## When NOT to use v4

- **Chat-memory retrieval.** Use
  [`Stffens/bge-small-rrf-lme-v1`](https://huggingface.co/Stffens/bge-small-rrf-lme-v1)
  instead, which is trained on real labeled LongMemEval queries.
- **Pre-v0.34 reproducibility.** If you are reproducing measurements
  taken on a vstash version before 0.34.0, use
  [`Stffens/bge-small-rrf-v3`](https://huggingface.co/Stffens/bge-small-rrf-v3)
  to match the training-time pipeline.

## Usage

### vstash

```bash
pip install vstash
vstash reindex --model Stffens/bge-small-rrf-v4
```

### sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Stffens/bge-small-rrf-v4")
embs = model.encode(["query text", "document passage"], normalize_embeddings=True)
```

## Training

- **Base:** `BAAI/bge-small-en-v1.5` (33M params, 384 dim).
- **Loss:** MultipleNegativesRankingLoss (MNRL).
- **Sampling:** `temperature=0.5` across SciFact, NFCorpus, FiQA.
- **Total triples:** 60,000 target budget (39,852 actual after mining).
- **Per-dataset triples:** SciFact 4291, NFCorpus 9713, FiQA 25848.
- **Epochs:** 2.
- **LR:** 3e-6.
- **Batch size:** 64.
- **AMP:** enabled.
- **Seed:** 42.
- **Hardware:** Colab T4 (CUDA), `--bulk-mine --bulk-mine-device cuda`.
- **Wall time:** ~9 minutes training.

Triples are mined inside `vstash retrain` from vector-vs-FTS top-K
disagreement. The training distribution is dominated by FiQA (~65%)
because FiQA queries trigger more disagreement events than the more
homogeneous claim-style SciFact and NFCorpus corpora.

## Citation

If you use this model, please cite the vstash v2 paper (in
preparation). v1 is on arXiv:

```bibtex
@misc{steffens2026vstash,
  title  = {vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents},
  author = {Steffens, Jayson},
  year   = {2026},
  eprint = {2604.15484},
  archivePrefix = {arXiv}
}
```

## License

MIT, same as `BAAI/bge-small-en-v1.5`.
