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

# bge-small-rrf-v3

A 33M-parameter (384-dim) English embedding model, fine-tuned from
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
using [vstash](https://github.com/stffns/vstash)'s self-supervised
hybrid-retrieval disagreement signal.

**Same size and speed as the base model. Higher retrieval quality
on all three BEIR datasets it was trained on.**

## What changed vs v2

`bge-small-rrf-v3` is trained with the winning config from the
2026-04-19 H-R9 ablation:

- **2x training volume.** 60,000 target triples across three BEIR
  datasets instead of v2's 30,000. Volume was the single largest
  lever at this scale: every dataset improved simultaneously when
  doubling `total_triples`, no trade-off observed.
- **Same corpus balance.** `temperature=0.5` sampling keeps the
  same ratio v2 used; the volume increase scales every dataset
  proportionally rather than reshuffling.
- **Observability.** The training pipeline now records NDCG@3 and
  Recall@100 alongside NDCG@10 in `training_meta.json`.

## Eval numbers

Evaluated on BEIR SciFact + NFCorpus + FiQA held-out queries with
vstash's production retrieval pipeline (RRF hybrid + adaptive
weights + MMR dedup, widened top-100 candidate pool). Absolute
NDCG@10:

| Dataset  | Base (bge-small) | **v3 (this model)** | Delta    |
|----------|------------------|---------------------|----------|
| SciFact  | 0.7333           | **0.7818**          | +0.0485  |
| NFCorpus | 0.3538           | **0.3757**          | +0.0219  |
| FiQA     | 0.3916           | **0.4818**          | +0.0902  |
| Macro    | 0.4929           | **0.5464**          | +0.0535  |

Full per-arm ablation table is in
[vstash/experiments/retrain_roadmap.md](https://github.com/stffns/vstash/blob/main/experiments/retrain_roadmap.md).

## Usage

### Drop-in via sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Stffens/bge-small-rrf-v3")
embeddings = model.encode(["what is hybrid retrieval?"], normalize_embeddings=True)
```

### Inside vstash

```bash
vstash reindex --model Stffens/bge-small-rrf-v3
```

### As a search / RAG backbone

Same API as `bge-small-en-v1.5`: 384 dimensions, cosine similarity,
instruction-free encoding. Drop into any retrieval stack built
around the base model.

## Training recipe

Reproducible via the published notebook
[`retrain_t1_4_multi_beir.ipynb`](https://github.com/stffns/vstash/blob/main/experiments/retrain_t1_4_multi_beir.ipynb)
after setting `total_triples=60000`. Single command:

```bash
vstash retrain-multi \
    --store scifact=./scifact.db \
    --store nfcorpus=./nfcorpus.db \
    --store fiqa=./fiqa.db \
    --sampling-strategy temperature \
    --sampling-temperature 0.5 \
    --total-triples 60000 \
    --epochs 2 --lr 3e-6 --batch-size 32 \
    --bulk-mine --bulk-eval \
    --seed 42 \
    --output ./bge-small-rrf-v3
```

### Pipeline

1. Ingest SciFact (5183 docs), NFCorpus (3633), FiQA (57638) into
   separate vstash stores.
2. Sample training queries from BEIR `queries.jsonl` + qrels (v5
   labeled-query recipe).
3. Mine hard negatives via vec-heavy / FTS-heavy RRF disagreement
   on each store.
4. Train one model on the union with MNRL for 2 epochs.
5. Evaluate per-dataset; promote the candidate only if macro
   NDCG@10 exceeds the base.

### Hyperparameters

| Key | Value |
|---|---|
| Base model | `BAAI/bge-small-en-v1.5` |
| Loss | `MultipleNegativesRankingLoss` |
| Total training triples | 60,000 (target) / 39,852 (emitted) |
| Sampling | temperature, `alpha=0.5` |
| Epochs | 2 |
| Learning rate | 3e-6 |
| Batch size | 32 |
| Warmup steps | 50 |
| `max_seq_length` | 256 |
| Mixed precision | FP16 (AMP on) |
| Seed | 42 |
| Training hardware | NVIDIA A100 |
| Training time | ~15 minutes |

## Limitations

- **English only.** The base model and training data are English.
  Cross-lingual retrieval may regress vs a multilingual model.
- **NFCorpus still saturates.** Even at 2x volume, the NFCorpus
  NDCG@10 stays around 0.376, short of v5's published 0.409. The
  gap is likely model-capacity (33M params) and can be closed by
  a cross-encoder reranker on top; see
  [vstash's T2.4 design doc](https://github.com/stffns/vstash/blob/main/experiments/t24_reranker_design.md).
- **Domain-specific corpora** (clinical, legal, heavily-jargoned)
  may benefit more from retraining with
  `vstash retrain-multi --store mydomain=...` on top of v3 than
  from v3 out of the box.

## Citation

```bibtex
@software{vstash_bge_small_rrf_v3_2026,
  author  = {Steffens, Jay},
  title   = {bge-small-rrf-v3: self-supervised retrieval fine-tune of BGE-small via vstash},
  year    = {2026},
  url     = {https://huggingface.co/Stffens/bge-small-rrf-v3}
}
```

For vstash itself:

```bibtex
@software{vstash_2026,
  author  = {Steffens, Jay},
  title   = {vstash: local-first document memory with instant semantic search},
  year    = {2026},
  url     = {https://github.com/stffns/vstash}
}
```
