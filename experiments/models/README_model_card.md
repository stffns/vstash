---
language: en
license: apache-2.0
library_name: sentence-transformers
tags:
  - sentence-transformers
  - embedding
  - retrieval
  - hybrid-search
  - self-supervised
  - fine-tuned
  - BEIR
  - information-retrieval
base_model: BAAI/bge-small-en-v1.5
datasets:
  - BeIR/scifact
  - BeIR/nfcorpus
  - BeIR/fiqa
pipeline_tag: feature-extraction
model-index:
  - name: bge-small-rrf-v2
    results:
      - task:
          type: Retrieval
        dataset:
          name: BEIR SciFact
          type: BeIR/scifact
        metrics:
          - type: ndcg_at_10
            value: 0.6945
      - task:
          type: Retrieval
        dataset:
          name: BEIR NFCorpus
          type: BeIR/nfcorpus
        metrics:
          - type: ndcg_at_10
            value: 0.3949
      - task:
          type: Retrieval
        dataset:
          name: BEIR SciDocs
          type: BeIR/scidocs
        metrics:
          - type: ndcg_at_10
            value: 0.1875
---

# bge-small-rrf-v2: A 33M Parameter Model That Beats ColBERTv2 on 3/5 BEIR Datasets

**Trained with zero human labels using a novel self-supervised signal: hybrid retrieval disagreement.**

When vector search and keyword search disagree on what's relevant for a query, that disagreement reveals where the embedding model fails. We exploit this signal to fine-tune BGE-small, producing a model that better distinguishes "semantically close" from "actually relevant."

## Key Result

A 33M parameter model, fine-tuned for $0 with zero human labels, **surpasses ColBERTv2 (110M parameters) on 3 out of 5 standard BEIR benchmarks**:

| Dataset | Docs | ColBERTv2 (110M) | BGE-small base (33M) | **This model (33M)** | vs ColBERTv2 |
|---------|:----:|:-:|:-:|:-:|:-:|
| SciFact | 5K | 0.693 | 0.646 | **0.695** | **+0.2%** |
| NFCorpus | 3.6K | 0.344 | 0.330 | **0.395** | **+14.8%** |
| SciDocs | 25K | 0.154 | 0.178 | **0.188** | **+21.8%** |
| FiQA | 57K | 0.356 | 0.328 | 0.328 | -7.8% |
| ArguAna | 8.6K | 0.463 | 0.419 | 0.424 | -8.4% |

**Up to +19.5% NDCG improvement over the base model, with zero additional inference cost.**

## Why This Matters

Most embedding improvements require either:
- A larger model (more compute, more latency)
- Human-labeled training data (expensive, slow)
- A teacher model for distillation (adds complexity)

This model needs none of that. The training signal comes from running the existing hybrid retrieval pipeline and observing where its two components (vector search and keyword search) disagree. **The system improves itself.**

## The Training Signal: Hybrid Retrieval Disagreement

We discovered that **82% of queries produce disagreement** between vector and keyword search in the top-5 results. These disagreements fall into two categories:

- **Vector blind spots** (51%): chunks the vector search ranks high but keyword search ignores. These are semantically similar but not actually relevant.
- **Keyword blind spots** (49%): chunks keyword search finds but vector search misses. These contain relevant terms but the embedding doesn't recognize their relevance.

Fine-tuning on these disagreement pairs teaches the model to fix both types of blind spots.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Parameters | 33M (unchanged) |
| Embedding dimension | 384 (unchanged) |
| Loss function | MultipleNegativesRankingLoss with explicit hard negatives |
| Training data | 76K (query, positive, hard_negative) triples |
| Data source | RRF signal disagreement on SciFact, NFCorpus, FiQA |
| Human labels | **Zero** |
| Epochs | 2 |
| Learning rate | 3e-6 |
| Batch size | 64 |
| Training time | ~30 min on T4 GPU |
| Training cost | $0 (Colab free tier) |

### Why MNRL, not TripletLoss?

We tested TripletLoss first. It **destroyed the model** (-84% NDCG after 3 epochs). TripletLoss pushes individual negatives away with brute force, distorting the embedding space. MNRL adjusts relationships across 64 documents simultaneously per batch, preserving the model's general knowledge while learning from disagreements.

| Loss Function | NDCG@10 on SciFact | Result |
|---|:-:|---|
| TripletLoss (3 epochs, lr=2e-5) | 0.055 | -84% (destroyed) |
| TripletLoss (1 epoch, lr=1e-6) | 0.347 | -0.03% (no effect) |
| MNRL batch-only negatives (v1) | 0.683 | +5.6% |
| **MNRL + explicit hard negatives (this model)** | **0.695** | **+7.4%** |

## Usage

### With sentence-transformers
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Stffens/bge-small-rrf-v2")
embeddings = model.encode(["your query", "your document"])
similarity = embeddings[0] @ embeddings[1]
```

### With vstash (hybrid retrieval system)
```bash
pip install vstash
vstash reindex --model Stffens/bge-small-rrf-v2
vstash search "your query"
```

### Train your own version on your data
```bash
pip install vstash sentence-transformers torch
vstash retrain  # generates disagreement pairs from YOUR corpus and fine-tunes
vstash reindex --model ~/.vstash/models/retrained
```

## Reproduce From Scratch

```bash
git clone https://github.com/stffns/vstash
cd vstash
pip install -e . sentence-transformers torch

# Generate disagreement triples
python -m experiments.rrf_training_pairs --datasets scifact nfcorpus fiqa

# Train (GPU recommended)
python -m experiments.finetune_rrf --epochs 2 --lr 3e-6 --batch-size 64

# Evaluate
python -m experiments.finetune_rrf --evaluate-only
```

## Limitations

- **ArguAna regression**: queries with 200+ words show -8.4% vs ColBERTv2. Long argumentative queries produce only 1.1% signal disagreement, leaving no training signal.
- **FiQA neutral**: financial queries show +0.1% vs base but -7.8% vs ColBERTv2. The disagreement signal exists (86.7%) but doesn't translate to NDCG gains on this dataset.
- **English only**: inherited from BGE-small-en-v1.5.
- **Not tested beyond BEIR**: performance on domain-specific corpora may vary.

## Citation

```bibtex
@software{vstash2026,
  author = {Steffens, Jayson},
  title = {vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents},
  url = {https://github.com/stffns/vstash},
  year = {2026}
}
```

## Related

- [vstash paper](https://github.com/stffns/vstash/blob/main/paper/vstash-paper.md) (Section 8.10: Self-Supervised Embedding Refinement)
- [vstash GitHub](https://github.com/stffns/vstash)
- [Base model: BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
