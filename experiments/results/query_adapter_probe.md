# Query-side adapter probe (2026-11-07)

**Question.** Can a small adapter applied only to the query vector
(`q' = normalize(W q)`, with the document vectors frozen) replace full
fine-tuning (v3), so the self-supervised retrain loop never has to re-embed
the corpus?

**Answer: no.** On queries whose relevant documents were not already labelled
in training, the adapters gain nothing; v3 does. The large SciFact win is a
learned "popular document" prior. The one case where an adapter matches v3
(NFCorpus) cannot be separated from that prior, and it breaks distance
calibration completely.

Script: `experiments/probe_query_adapter.py`. Raw numbers:

- `query_adapter_probe.json`: top-30 hard negatives, lr 1e-3, includes v3.
- `query_adapter_probe_inbatch.json`: in-batch negatives only, lr 1e-4.

All numbers below are from the corrected training mask: every known positive
of a row's query is masked anywhere in the candidate block, not only that
row's own target.

## Setup

- **Data:** SciFact, NFCorpus and FiQA, the three corpora v3 was trained on.
  Adapters train on each dataset's `train` qrels, with 10% of those queries
  held out as a dev set for early stopping. Evaluation uses the `test` qrels
  (300 / 323 / 648 queries).
- **Embeddings:** document vectors come from base `BAAI/bge-small-en-v1.5`
  through vstash's FastEmbed path. v3 is re-embedded with a `transformers`
  CLS encoder, checked to give cosine 1.00000 against FastEmbed on the base
  model.
- **Arms:**
  - `bias`: `q + b`, 384 params.
  - `lowrank`: `I + A Bᵀ` at r=16, 12k params, 48 KB.
  - `full`: `I + D`, 147k params, 576 KB.
  - `lowrank_shared`: one r=16 adapter trained on all three datasets.
  - `v3`: the full fine-tune, with the corpus re-embedded.
- **Metrics:**
  - NDCG@10, both dense (exact cosine) and hybrid (the real
    `VstashStore.search`).
  - The same NDCG@10 on the **unseen** subset: test queries none of whose
    relevant documents is a train positive.
  - The median best distance and the share of queries in each
    `relevance_tier`.

## Results (hybrid NDCG@10, test)

| dataset | base | best adapter (config) | v3 |
|---|---|---|---|
| SciFact, all 300 | 0.726 | 0.819 (`full`, hard neg) | 0.770 |
| SciFact, **unseen** 104 | 0.751 | 0.717 (same, **−0.034**) | 0.793 (**+0.042**) |
| NFCorpus, 323 (unseen subset empty) | 0.359 | 0.376 (`full`, in-batch) | 0.375 |
| FiQA, 648 (all unseen) | 0.392 | 0.392 (identity: dev never improved) | 0.464 (**+0.073**) |

Training cost is negligible in every configuration: 0.1 to 50 s on CPU for up
to 113k pairs. What fails is generalisation.

## Findings

1. **The SciFact gain is leakage, not semantics.** 181 of the 283 relevant
   test documents are also train positives, and in 193 of the 300 test
   queries every relevant document is already a train positive. On the
   104 unseen queries, both adapters that improve the all-query number are
   worse than base: hybrid 0.705 for `lowrank` and 0.717 for `full`, against
   0.751. v3 improves the same queries to 0.793. With in-batch negatives
   only, every SciFact adapter stays within ±0.02 of base.
2. **FiQA: nothing transfers.** FiQA has no relevant document shared between
   train and test. Training loss falls and NDCG on the training queries rises
   from 0.398 to 0.473. Dev NDCG, however, falls from the first epoch
   (0.403 → 0.33–0.37) with every learning rate and with or without hard
   negatives, so early stopping keeps the identity. v3 gains +0.10 dense and
   +0.07 hybrid on the same queries.
3. **NFCorpus: the in-batch `full` adapter matches v3 (0.376 vs 0.375), but
   that cannot count as a win.** Every NFCorpus test query has at least one
   relevant document that is also a train positive, so there is no unseen
   subset to rule out the prior. That configuration also puts **100% of
   queries in the `low` tier**. With top-30 hard negatives, no NFCorpus
   adapter ever improved dev, because its top hits are full of relevant
   documents nobody labelled.
4. **Calibration breaks when the adapter moves far from the identity.** The
   median best distance and the share of queries tiered `low` change as
   follows:

   | adapter | median best distance | queries tiered `low` |
   |---|---|---|
   | SciFact `lowrank`, hard neg | 0.18 → 0.59 | 87% |
   | SciFact `full`, hard neg | 0.18 → 0.62 | 97% |
   | NFCorpus `bias`, in-batch | 0.25 → 0.57 | 82% (NDCG goes down) |
   | NFCorpus `full`, in-batch | 0.25 → 0.69 | 100% |

   Adapters that stay close to the identity keep the scale. The SciFact
   in-batch adapters move NDCG by at most 0.02, sit at a median of 0.19–0.22,
   and leave every query in `high`. Any adapter strong enough to matter would
   therefore need the `relevance_tier` thresholds and miss hints (#272)
   re-calibrated.

## Implications

- Keep full fine-tuning plus re-index as the way to improve the model. A
  query-only linear map is the wrong tool for that job.
- What the adapters do learn is a corpus prior: "these documents tend to be
  the answer". For agent memory that prior may be wanted, and it is exactly
  what the usage log (#513) measures. If it is wanted, it is cheaper and safer
  as an explicit score feature, such as per-chunk use counts in RRF, than
  hidden inside `W`, where it also distorts distances. Testing that is a
  separate experiment.
- When a dataset shares relevant documents between train and test, report an
  unseen subset alongside the all-query number. Here the all-query number
  overstated the SciFact result by about 0.1.
