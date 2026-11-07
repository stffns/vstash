# Query-side adapter probe (2026-11-07)

**Question.** Can a small adapter applied only to the query vector
(`q' = normalize(W q)`, document vectors frozen) replace full fine-tuning
(v3), so the self-supervised retrain loop never has to re-embed the corpus?

**Answer: no, not as a replacement.** The adapters gain nothing on queries
whose relevant documents were not already labelled in training. v3 does.
What looked like a large win on SciFact turned out to be a learned "popular
document" prior.

Script: `experiments/probe_query_adapter.py`. Raw numbers:
`query_adapter_probe.json` (hard negatives, lr 1e-3, includes v3) and
`query_adapter_probe_inbatch.json` (in-batch negatives only, lr 1e-4, the
adapter's best configuration).

## Setup

- Data: SciFact, NFCorpus and FiQA, the three corpora v3 was trained on.
  Adapters train on each dataset's `train` qrels, with 10% of those queries
  held out as a dev set for early stopping. Evaluation uses the `test` qrels
  (300 / 323 / 648 queries).
- Document vectors: base `BAAI/bge-small-en-v1.5` through vstash's FastEmbed
  path. v3 is re-embedded with a `transformers` CLS encoder, whose output
  matches FastEmbed's on the base model (cosine 1.00000).
- Arms:
  - `bias`: `q + b`, 384 parameters.
  - `lowrank`: `I + A Bᵀ` at rank 16, 12k parameters, 48 KB.
  - `full`: `I + D`, 147k parameters, 576 KB.
  - `lowrank_shared`: one rank-16 adapter trained on all three datasets.
  - `v3`: the full fine-tune, with the corpus re-embedded.
- Metrics:
  - NDCG@10, dense (exact cosine) and hybrid (the real `VstashStore.search`).
  - The same NDCG@10 on the "unseen" subset: test queries none of whose
    relevant documents is a train positive.
  - The median best distance and the share of queries in each
    `relevance_tier`.

## Results (hybrid NDCG@10, test)

| dataset | base | best adapter | v3 |
|---|---|---|---|
| SciFact, all 300 | 0.726 | 0.818 (`full`, hard neg) | 0.770 |
| SciFact, unseen 104 | 0.751 | 0.716 (same adapter, **−0.035**) | 0.793 (**+0.042**) |
| NFCorpus, 323 (unseen subset empty) | 0.359 | 0.367 (`full`, in-batch) | 0.375 |
| FiQA, 648 (all unseen, no overlap) | 0.392 | 0.392 (identity, never improved dev) | 0.464 (**+0.073**) |

Training cost really is negligible: 0.1 to 7 s on CPU for up to 100k pairs.
The adapters fail on generalization, not on cost.

## Findings

1. **The SciFact gain is leakage, not semantics.** 181 of the 283 relevant
   test documents are also train positives, and 193 of the 300 test queries
   cite only documents that train already labels relevant. The cited set is
   about 565 of 5,183 documents, so learning "prefer the cited set" pays off.
   On the 104 unseen queries every adapter is worse than the base model,
   while v3 improves them.
2. **FiQA has no document overlap, and the adapter learns nothing that
   transfers.** Training loss drops and NDCG on the training queries climbs
   from 0.398 to 0.472, but dev NDCG falls from the first epoch
   (0.403 → 0.33–0.37). This holds for every learning rate tried and with or
   without hard negatives, so early stopping correctly keeps the identity.
   On the same data v3 gains +0.10 dense and +0.07 hybrid.
3. **NFCorpus: the best case recovers about half of v3's gain**
   (+0.0075 vs +0.0155 hybrid). Even that case breaks calibration: 35% of
   queries fall into the `low` tier.
4. **Distance calibration breaks whenever the adapter actually changes
   ranking.** The median best distance goes from 0.18 to 0.58–0.62 on
   SciFact and from 0.25 to 0.45 on NFCorpus, with 35–97% of queries
   reported as `low`. The `relevance_tier` thresholds and the miss hints
   (#272) assume the base model's distance scale, so a deployed adapter
   would also need re-calibrated thresholds.
5. **Hard negatives mined from the top 30 hurt NFCorpus** (dev 0.330 → 0.25),
   because its top hits are full of relevant documents nobody labelled.
   In-batch negatives only is the safer default for any query-side training
   on this kind of qrels.

## Implications

- Keep full fine-tuning plus re-indexing as the way to improve the model.
  A query-only linear map is the wrong tool for that.
- The one thing an adapter did learn is a corpus prior ("these documents tend
  to be the answer"). For agent memory that prior might be wanted: it is what
  the usage log (#513) measures directly. If so, it is cheaper and safer as
  an explicit score feature (per-chunk use counts in RRF) than hidden inside
  `W`, where it also distorts distances. That would be a separate experiment.
- For datasets with label overlap, report generalization on an unseen subset;
  an all-query number alone overstated the SciFact result by about 0.1.
