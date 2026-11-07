# Usage-log prior probe (2026-11-07)

**Question.** The query-adapter probe (#514) showed that what a query-side `W`
learns is a corpus prior ("these documents tend to be the answer"), not better
semantics. The usage log (#513) records that prior directly
(`search_results.used`). Does feeding it to the ranker *explicitly* keep the
gain without the harm?

**Answer.** It works only when it is conditioned on the query.

- **Rejected: a global use-count boost.** This is the same failure as the
  adapter. It lifts SciFact's repeat queries, costs accuracy on everything
  else, and the best single default chosen on dev is β = 0, so it ships
  nothing.
- **Works: a query-conditioned boost.** Boost only the documents that *similar
  past queries* used, with kNN over past query embeddings and a high
  similarity threshold. This captures most of the gain with no measurable
  harm: SciFact rises 0.727 → 0.761, the unseen subset is untouched, and FiQA
  is flat to within 0.001.

Script: `experiments/probe_usage_prior.py`. Raw numbers:
`usage_prior_probe.json`.

## Setup

**Data.** SciFact, NFCorpus and FiQA, with BGE-small base embeddings and
retrieval through the real hybrid `VstashStore.search`. The top 50 candidates
are re-scored with the prior and then cut to 10.

**Simulated usage log.** Each dataset's `train` queries play the agent's past
searches. There are two versions:

- **`logged` (realistic):** a document is "used" only if base search
  *returned* it in the top 10 for that train query and it is relevant. Each
  such use is kept with p = 0.9. In addition, with p = 0.2 per query, one
  returned document that is *not* relevant is marked used.
- **`oracle` (upper bound):** every train positive counts as used.

**Priors.** Three ways of applying the log:

- **`mult`:** `score · (1 + β · log1p(uses))`, using one global count per
  document.
- **`rrf3`:** a third RRF list ordered by global use count, weighted by β.
- **`knn`:** the uses come only from past queries whose embedding has
  cosine ≥ τ with the current query (top 20, weighted by similarity), then the
  same formula as `mult`. A query with no near past query keeps the base
  ranking.

**Protocol.** β and τ are chosen on a dev split (10% of train). The dev split
gets usage counts built **only from the other 90%**, so a dev query never sees
its own positives. Test uses the full train log.

**Subsets reported.**

- **SciFact "unseen":** 104 queries none of whose relevant documents is a
  train positive. The prior can only hurt these.
- **FiQA:** train and test share no relevant document at all, so FiQA is
  a pure harm test.

## Results (hybrid NDCG@10, test, `logged` log)

| prior | SciFact all | SciFact unseen | NFCorpus | FiQA |
|---|---|---|---|---|
| base | 0.727 | 0.750 | 0.361 | 0.393 |
| global `mult`, β = 0.05 (best on dev) | 0.742 | **0.728** | 0.362 | 0.376 |
| global `mult`, β = 0.1 | 0.747 | 0.711 | 0.361 | 0.356 |
| global `rrf3`, β = 0.25 | 0.734 | 0.636 | 0.352 | 0.266 |
| **`knn` τ = 0.9, β = 1.0** | **0.761** | 0.750 | 0.363 | 0.392 |
| `knn` τ = 0.85, β = 1.0 | 0.766 | 0.750 | 0.366 | 0.387 |
| `knn` τ = 0.8, β = 1.0 | 0.775 | 0.750 | 0.367 | 0.364 |
| `knn` τ = 0.75, β = 1.0 | 0.764 | 0.723 | 0.371 | 0.327 |

The upper bound with a perfect log (`oracle`, `knn` τ = 0.8, β = 2.0) reaches
0.800 on SciFact and 0.392 on NFCorpus.

## Findings

1. **A global popularity prior is a trade, not a gain.** SciFact queries that
   repeat known answers go up, and every other query goes down: SciFact
   unseen falls 0.750 → 0.728 and FiQA 0.393 → 0.376 at the smallest useful
   β. Picking one β on dev across all three datasets gives β = 0 for both
   `mult` and `rrf3`. `rrf3` is strictly worse than `mult`; on FiQA it can
   halve NDCG.
2. **Conditioning on the query removes the harm.** The `knn` prior changes a
   ranking only when a similar past query exists. With τ ≥ 0.8 the SciFact
   unseen subset is bit-for-bit unchanged (0.7504). At τ = 0.9, FiQA moves by
   less than 0.001 while SciFact gains +0.034 even with a noisy log
   (10% missed uses, 20% wrong uses).
3. **τ is the safety knob, and it has to be high.** BGE query–query cosines
   are compressed: FiQA's finance questions sit at 0.75–0.85 cosine from each
   other yet need different answers. At τ = 0.8, FiQA already loses −0.029.
   The dev split, averaged over the three datasets, picks τ = 0.8 / β = 0.25
   (FiQA −0.020), so **an averaged-dev default is not safe**. τ = 0.9 should
   be chosen deliberately as the conservative default.
4. **The upside scales with how often questions repeat.** In BEIR, "past
   queries" are other people's questions. An agent's own memory repeats
   itself far more, which is where this signal pays most. With a perfect log
   the ceiling is +0.07 on SciFact and +0.03 on NFCorpus.
5. **Distances are untouched by construction.** Unlike the adapter, the
   `relevance_tier` calibration cannot break.

## What shipping it would take (not in this PR)

- **Store the query embedding on `search_events`.** Today the log keeps the
  query text but not its vector: 1.5 KB per event, about 15 MB at the
  10,000-event cap, or a small vec0 table.
- **Add an opt-in kNN prior to `services/search.py` or `VstashStore.search`.**
  Find past events with cosine ≥ τ, sum similarity-weighted `used` counts per
  chunk, and boost the candidates before the `top_k` cut. Over 10,000 events
  this is a single 10k × 384 matrix-vector product, well under a millisecond.
- **Defaults:** τ = 0.9 and β ≈ 1.0, opt-in, plus an `explain` field showing
  the prior's contribution so a boosted result is never a mystery.
- **Validate on real usage data.** Once merken reports `mark_used`, check the
  τ / β choice against real agent repeat rates, not BEIR's.
