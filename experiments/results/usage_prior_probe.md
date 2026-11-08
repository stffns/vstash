# Usage-log prior probe (2026-11-07)

**Question.** The query-adapter probe (#514) showed that what a query-side `W`
learns is a corpus prior ("these documents tend to be the answer"), not better
semantics. The usage log (#513) records that prior directly
(`search_results.used`). Does feeding it to the ranker *explicitly* keep the
gain without the harm?

**Answer: only when the prior is conditioned on the query.**

- **Global use-count boost: rejected.** It fails the same way the adapter
  did: SciFact's repeat queries go up and everything else goes down. Even the
  smallest β costs about 0.02 on the strict harm subset.
- **Query-conditioned kNN boost: works.** Boost only the documents that past
  queries *similar to this one* used, with a high similarity threshold. A
  dev-only "safe" selection rule picks τ = 0.9, β = 2.0. On test that setting
  gives SciFact 0.727 → 0.761, and the harm subsets stay within 0.001 of base.

Script: `experiments/probe_usage_prior.py`. Raw numbers:
`usage_prior_probe.json`.

## Setup

**Data and retrieval.** SciFact, NFCorpus and FiQA, with BGE-small base
embeddings. Retrieval goes through the real hybrid `VstashStore.search`: the
top 50 candidates are re-scored with the prior, then cut to 10.

**Simulated usage log.** Each dataset's `train` queries play the agent's past
searches. The log is generated from separate **`top_k=10`** searches.
`VstashStore.search` scales its candidate pool and its FTS admission with
`top_k`, so the head of a `top_k=50` search is not what a normal search
shows. Two versions of the log:

- **`logged` (realistic):** a document is "used" only if base search
  *returned* it and it is relevant, and each use is kept with p = 0.9. In
  addition, with p = 0.2 per query, one returned document that is *not*
  relevant is marked used.
- **`oracle` (upper bound):** every train positive counts as used.

**Priors.**

- **`mult`:** `score · (1 + β · log1p(uses))`, using one global count per
  document.
- **`rrf3`:** a third RRF list ordered by global use count, weighted by β.
- **`knn`:** uses only from past queries with embedding cosine ≥ τ to the
  current query (top 20, weighted by similarity), then the same formula as
  `mult`. A query with no near past query keeps the base ranking.

**Two selection rules.** Both use dev only: 10% of train, with usage counts
built from the other 90%, so a dev query never sees its own positives.

- **avg-dev:** the setting with the best mean dev NDCG over the three
  datasets.
- **safe:** the best mean dev gain, subject to no dataset losing more than
  0.002 on dev against β = 0.

**Subsets reported on `test`.**

- **all:** every test query.
- **unseen:** no relevant document is a train positive.
- **untouched:** no relevant document received *any* use in that log, false
  uses included. Only `untouched` is a strict harm test for the noisy
  `logged` log. A false use can land on a document relevant to an "unseen"
  query and help it, so an unchanged "unseen" mean could hide offsetting
  losses. Sizes:
  - SciFact: 104 unseen; 104 untouched with `oracle`, 139 with `logged`.
  - NFCorpus: 0 unseen; 0 untouched with `oracle`, 6 with `logged`.
  - FiQA: 648 unseen; 648 untouched with `oracle`, 582 with `logged`.

## Results (hybrid NDCG@10, test, `logged` log)

| prior | how chosen | SciFact all | SciFact untouched | NFCorpus all | FiQA all | FiQA untouched |
|---|---|---|---|---|---|---|
| base | none | 0.727 | 0.656 | 0.361 | 0.393 | 0.389 |
| `mult` β = 0.05 | avg-dev | 0.741 | **0.635** | 0.362 | **0.373** | **0.366** |
| `mult` / `rrf3` | safe | β = 0, i.e. base | | | | |
| `rrf3` β = 0.25 | sweep | 0.732 | 0.553 | 0.354 | 0.260 | 0.246 |
| `knn` τ = 0.85, β = 0.5 | avg-dev | 0.764 | 0.655 | 0.365 | 0.388 | 0.383 |
| **`knn` τ = 0.9, β = 2.0** | **safe** | **0.761** | 0.656 | 0.362 | 0.392 | 0.388 |
| `knn` τ = 0.8, β = 1.0 | sweep | 0.775 | 0.655 | 0.368 | 0.367 | 0.361 |

With the `oracle` log, the safe rule also picks τ = 0.9, β = 2.0. That
setting gives SciFact 0.776 (untouched 0.750, unchanged), NFCorpus 0.368 and
FiQA 0.392. The oracle's best case is `knn` τ = 0.8, β = 2.0, which reaches
0.800 on SciFact and 0.392 on NFCorpus but costs FiQA 0.038.

## Findings

1. **A global popularity prior is a trade, not a gain.** It helps queries
   that repeat a known answer and costs everything else. At the smallest β
   the avg-dev rule accepts (0.05), SciFact untouched falls 0.656 → 0.635 and
   FiQA untouched 0.389 → 0.366. The safe rule rejects every non-zero β for
   both `mult` and `rrf3`.
2. **Conditioning on the query removes the harm.** The `knn` prior changes a
   ranking only when a similar past query exists. At the safe-rule setting
   (τ = 0.9), SciFact gains +0.034 even with a noisy log (10% missed uses,
   20% wrong ones). The strict harm subsets move by at most 0.001: SciFact
   untouched by 0.000, FiQA untouched by −0.0008, NFCorpus untouched (only 6
   queries) by 0.000.
3. **τ is the safety knob, and the averaged choice is not safe.** BGE
   query–query cosines are compressed. FiQA's finance questions sit at
   0.75–0.85 cosine from each other yet need different answers, and FiQA
   loses −0.026 at τ = 0.8. Averaging dev over datasets picks τ = 0.85
   (FiQA −0.005); bounding each dataset's dev loss picks τ = 0.9 on its own.
   The τ = 0.9 default is therefore dev-selected under that rule, not read
   off the test set. It has not yet been confirmed on fresh held-out data or
   on real agent logs.
4. **The upside scales with how often questions repeat.** In BEIR, "past
   queries" are other people's questions. An agent's own memory repeats far
   more, which is where this signal pays. With a perfect log the ceiling is
   about +0.07 on SciFact and +0.04 on NFCorpus, but only at lower τ, which
   costs FiQA.
5. **Distances are untouched by construction.** Unlike the adapter, the
   prior cannot break the `relevance_tier` calibration.

## What shipping it would take (not in this PR)

- **Store the query embedding on `search_events`.** Today the log keeps the
  query text but not its vector: 1.5 KB per event, about 15 MB at the
  10,000-event cap, or a small vec0 table.
- **Add an opt-in kNN prior to `services/search.py` or `VstashStore.search`.**
  Find past events with cosine ≥ τ, sum similarity-weighted `used` counts per
  chunk, and boost the candidates before the `top_k` cut. Over 10,000 events
  this is a single 10k × 384 matrix-vector product, well under a millisecond.
- **Defaults:** τ = 0.9 and β ≈ 1–2, as selected by the safe rule here, opt-in.
  Add an `explain` field showing the prior's contribution so a boosted result
  is never a mystery.
- **Re-validate τ / β on real usage.** Once merken reports `mark_used`,
  re-check against real agent logs rather than BEIR's repeat rates. Until
  then the defaults are a dev-selected starting point, not a validated
  setting.
