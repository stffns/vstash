# snapvec x v3 sweeps + v4 mining diff -- findings

Date: 2026-04-20
Branch: `experiments/snapvec-v3-sweeps` (stacked on PR #254)
Model: `Stffens/bge-small-rrf-v3`, BAAI/bge-small pre-training, 384 dim, 33M params

Three experiments, all documented here:
1. **A1-bits**: flat snapvec `bits` sweep over BEIR 4-dataset (arguana excluded for runtime).
2. **A1-ivfpq**: ivfpq `(rerank_candidates, nprobe)` sweep over BEIR 4-dataset.
3. **B**: hard-negative mining diff -- sqlite-vec (exact) vs snapvec (ANN) -- for a v4 retrain candidate.

## TL;DR

- **bits=4 stays the default** on flat snapvec. `bits=3` is within 0.5% on
  macro NDCG@10 for 25% less disk; `bits=2` costs real NDCG.
- **ivfpq sweep**: `rerank_candidates` is a no-op at `top_k=100`
  (NDCG@10 identical across rerank in {50,100,200}). `nprobe=0`
  (library default `nlist // 8`) wins on every dataset; explicit
  lower values (8 or 16) regress NDCG by up to 0.045 on SciFact.
- **v4 from snapvec mining is NOT recommended**. On SciFact chunk-prefix
  queries, sqlite-vec (exact) emitted a hard negative for 2/300 queries;
  snapvec (ANN) emitted one for 273/300. The "extra" 271 negatives are
  almost certainly ANN-quantization noise rather than real
  disagreement-mined negatives -- training on them would likely hurt.

## Context

Jay asked for "fit snapvec codebook with v3 embeddings" analogues. Library-level
investigation showed:

- `snapvec.SnapIndex` (flat) has **no learnable codebook**. It exposes `add`,
  `add_batch`, `search`, `save`, `load`, `freeze`, `stats` -- no `fit()`.
  Polar quantization is deterministic given `(dim, bits, seed)`. The only
  "fit" knob on flat is `bits`, restricted to `{2, 3, 4}`.
- `snapvec-ivfpq` **does** have a learnable codebook: `store.fit_ivfpq()`
  trains coarse k-means + residual PQ from whatever is in `vec_chunks`.
  Relevant knobs after dataset-driven `nlist`: `rerank_candidates`, `nprobe`.
- `generate_triples` (retrain.py) calls `store.search()` which routes through
  the configured `vector_backend`. No retrain.py edit needed; swap backend at
  store construction time and hard-neg mining shifts from exact -> ANN.

## A1-bits. Flat snapvec bits sweep (v3, BEIR 4-dataset)

_Results file_: `experiments/results/beir_snapvec_flat_bits_sweep.json`
_Datasets_: scifact, nfcorpus, fiqa, scidocs. Arguana excluded (too many
super-slow FTS queries to fit in the runtime budget; backend pattern
was consistent in smoke runs).

### NDCG@10 (absolute)

| Dataset | BM25 | ColBERTv2 | bits=2 | bits=3 | bits=4 |
|---|---|---|---|---|---|
| scifact | 0.665 | 0.693 | 0.7644 | 0.7811 | 0.7856 |
| nfcorpus | 0.325 | 0.344 | 0.3791 | 0.3851 | 0.3832 |
| fiqa | 0.236 | 0.356 | 0.4677 | 0.4862 | 0.4892 |
| scidocs | 0.158 | 0.154 | 0.2005 | 0.2020 | 0.2072 |
| **macro** | - | - | **0.4529** | **0.4636** | **0.4663** |

### Storage (snpv MB, post-close)

| Dataset | bits=2 | bits=3 | bits=4 |
|---|---|---|---|
| scifact | 0.68 | 1.00 | 1.31 |
| nfcorpus | 0.48 | 0.70 | 0.92 |
| fiqa | 7.63 | 11.15 | 14.67 |
| scidocs | 3.39 | 4.96 | 6.52 |

### Observations

- **bits=4 is the best NDCG default** on macro (0.4663), but gains over
  bits=3 are tiny (+0.0027 macro, +0.5%). On nfcorpus bits=3 actually
  beats bits=4 (0.3851 vs 0.3832) -- noise-level difference but a clear
  pareto frontier exists.
- **bits=3 is the storage/quality sweet spot**: 75% of bits=4 disk for
  99.4% of bits=4 NDCG. Worth considering as the default when disk
  matters and the workload is mainly retrieval (not benchmarking).
- **bits=2 loses ~0.015 macro NDCG** (-2.9% relative). Not worth the
  extra compression for English BEIR.
- **Recall@100 barely moves** with bits: SciFact stays at 0.9967 across
  all three; NFCorpus drifts <0.01. Candidate-set health is not the
  bottleneck; top-10 ranking is where quantization noise shows up.
- **Latency is flat** across bits (within 2% spread). Polar-quant
  search is memory-bandwidth-bound, not bit-depth-bound.

### Recommendation

Keep `snapvec_bits=4` as the default. Document `bits=3` as the
disk-saving alternative with a ~0.5% NDCG cost. Do not ship `bits=2`
as a recommended config.

## A1-ivfpq. IVFPQ rerank x nprobe sweep (v3, BEIR 4-dataset)

_Results file_: `experiments/results/beir_snapvec_ivfpq_sweep.json`
_Datasets_: scifact, nfcorpus, fiqa, scidocs (arguana excluded for
the same runtime reason as A1-bits).
_Grid_: `rerank_candidates in {50, 100, 200} x nprobe in {0, 8, 16}`
where `nprobe=0` means "library default `nlist // 8`".

### NDCG@10 by (rerank, nprobe) x dataset

Collapsed because NDCG@10 is **identical across all three rerank
values** for every (dataset, nprobe) pair. Only nprobe moves.

| Dataset | nprobe=0 (default) | nprobe=8 | nprobe=16 |
|---|---|---|---|
| scifact | 0.7507 | 0.7061 | 0.7419 |
| nfcorpus | 0.3493 | 0.3421 | 0.3497 |
| fiqa | 0.4242 | 0.3906 | 0.4053 |
| scidocs | 0.1826 | 0.1792 | 0.1813 |
| **macro** | **0.4267** | **0.4045** | **0.4195** |

Library-default `nprobe` (= `nlist // 8`, which is 35/30/120/80 for
SciFact/NFCorpus/FiQA/SciDocs) is the best or tied on all 4 datasets.

### Observations

- **`rerank_candidates` has no measurable effect** on NDCG@10 across
  rerank in {50, 100, 200} with top_k=100 search. Hypothesis: the
  search layer floors rerank at `top_k`, so once rerank >= 100 the
  result set is identical. Latency p50 also essentially flat (within
  ~3% spread).
- **`nprobe=0` (library default `nlist // 8`) is the best lever**.
  Overrides to nprobe=8 or nprobe=16 always lose on NDCG@10, sometimes
  by a lot: SciFact -0.0446 at nprobe=8, FiQA -0.0336. Latency wins
  from explicit smaller nprobe are tiny (~5-10% p50).
- **Non-monotonic U-shape on nprobe**: nprobe=8 is worse than both 0
  and 16 on every dataset. Sign that the nprobe-8 sample lands in a
  bad local trough for our test set, not a real quality vs speed knob.
  For real tuning, sweep near the library default (nlist//8 ± a few),
  not these arbitrary low values.
- **Small-corpus under-training warning fires** on SciFact (nlist=287,
  training=5183, ratio=18) and NFCorpus (nlist=241, training=3633,
  ratio=15). Docstring threshold is 30 training vectors per cluster.
  SciDocs (nlist=640, training=25657, ratio=40) and FiQA (nlist=960,
  training=57638, ratio=60) satisfy the rule. NDCG gap to sqlite-vec
  on the small corpora is larger than on the well-trained ones, which
  tracks.
- **snpi disk is linear in N**: SciFact 5.10 MB (5.2k docs) -> FiQA
  49.66 MB (58k docs). ivfpq costs roughly 10x flat snapvec at bits=4
  for comparable corpora, which is the tradeoff the IVFPQ path is
  explicitly making for search-speed reasons (not flat's goal).

### Recommendation

- Do not override `rerank_candidates` without measuring -- default
  floor behavior (`max(rerank, top_k)`) makes the knob a no-op in
  practice for `top_k=100` searches.
- Do not override `nprobe` below the library default for
  BEIR-scale corpora. Only push it lower if latency is the bottleneck
  and you can tolerate measurable NDCG loss.
- For corpora where `nlist` can be small (e.g., <10k docs with
  auto-derived nlist above 30 training vectors/cluster), consider
  clamping nlist explicitly rather than fighting the default nprobe.

## B. sqlite-vec vs snapvec hard-neg mining diff (SciFact, 300 training chunks)

_Results file_: `experiments/results/retrain_v4_snapvec.json`

| Metric | sqlite-vec (exact) | snapvec (ANN) | Note |
|---|---|---|---|
| pairs generated | 300 | 300 | same queries, same chunk population |
| mining time | 26.3s | 13.9s | snapvec ~1.9x faster |
| pairs with a non-null hard negative | **2** | **273** | per-backend discovery count |
| queries where only this backend found a negative | 0 | 271 | asymmetric: snapvec >> sqlite-vec |
| queries where both found the same negative | 0 | 0 | no literal overlap |
| queries where both found different negatives | 2 | 2 | |
| queries where neither found a negative | 27 | 27 | |
| total divergence (`only_vec + only_snap + different`) | | 91.0% | of shared queries |

### Interpretation

The striking asymmetry: **sqlite-vec finds a hard negative in only 2 of
300 queries (0.7%); snapvec finds one in 273 (91%)**. The divergence is
not noise around a shared answer -- it's one backend finding negatives
that the other does not.

Why. The hard-neg rule in `retrain.generate_triples` is "something in
vec top-5 that is NOT in fts top-5 (and != doc path)". Training queries
here are the first-200-chars of each chunk, which are literal substrings
of the indexed corpus, so:

- On sqlite-vec (exact), vec top-5 and fts top-5 strongly agree, because
  a chunk's own prefix is near-perfectly retrievable by both signals.
  The disagreement set `vec_top5 \ fts_paths` is empty for ~99% of
  queries, and `generate_triples` falls through without emitting a
  negative. Result: 2 negatives out of 300.
- On snapvec (ANN, flat polar quantization), vec top-5 is lightly
  perturbed by quantization. Some "correct" near-matches drop out;
  some farther docs slip in. Those slipped-in docs are structurally
  unlikely to be in fts top-5 (which is exact), so the
  `vec_top5 \ fts_paths` set becomes non-empty for almost all queries.
  The "negative" the miner records is the first such slipped-in doc.

Which means the 273 snapvec-mined negatives are, with high probability,
**quantization-noise artifacts rather than genuinely hard negatives**.
Training on them would teach the model to distinguish between docs
that polar quantization happens to rank differently from exact -- not
useful contrastive signal, and likely harmful.

Subsidiary caveat: chunk-prefix pseudo-queries are a pathological case
for this diff. The v5 recipe (`training_queries_by_dataset`) uses real
BEIR queries + gold-label positives and runs through
`generate_labeled_triples_batched` which is a GPU matmul path that
**does not route through the vector_backend at all**. So switching a
store to snapvec during v5/v3-style multi-corpus training has zero
effect on mining. The `generate_triples` path in this experiment is
the one-corpus, chunk-prefix mining flow used in earlier retrain work.

### Recommendation

- **Do not** ship a v4 that uses snapvec-flat as the hard-neg miner
  for chunk-prefix queries. The mining signal is dominated by ANN
  noise, not real disagreement.
- If we want to test "faster mining via ANN", the cleaner version uses
  real (labeled) queries rather than chunk prefixes, and still routes
  through an exact backend because the speed bottleneck on chunk-prefix
  queries is the per-query sqlite-vec full scan, not the search itself
  -- `generate_triples_batched` (GPU matmul) already solves that.
- No train pass was run; the signal already says the negatives are
  wrong in kind, so a train-gate run would be a confirmation rather
  than a discovery.

### Known limitation

The monkeypatch in the experiment script routes `embed_query` through
SentenceTransformer because the library's `_HF_ONNX_MODELS` set does not
include `Stffens/bge-small-rrf-v3` yet. MLX rejects v3's legacy BERT
tensor names, so without the patch the experiment would crash on Apple
Silicon. This is an experiment-local workaround; the library side should
add v3 to `_HF_ONNX_MODELS` separately.

## Open questions / follow-ups

- A1-ivfpq summary pending job completion; the `(rerank, nprobe)`
  Pareto analysis will be added once the full grid lands.
- Validate the "snapvec mines quantization-noise negatives" hypothesis
  on real (labeled) BEIR queries. `generate_triples` only speaks
  chunk-prefix; a sibling miner that takes labeled queries and still
  routes through `store.search()` would distinguish "ANN adds real
  signal" from "ANN adds noise".
- B with `--train` on Colab: train v4-snapvec and v4-sqlite-vec
  head-to-head for 1-2 BEIR corpora. Hypothesis: v4-snapvec
  regresses because the 273 "negatives" are mostly ANN noise.
- Arguana backfill: re-run both sweeps on arguana with a reduced
  query budget (sample 200 queries) to confirm the bit-ordering and
  rerank-ordering trends also hold on the large-FTS-query dataset.
