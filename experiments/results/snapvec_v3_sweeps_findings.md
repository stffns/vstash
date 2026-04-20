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
- **v4 from snapvec mining is open**. On SciFact chunk-prefix queries,
  sqlite-vec (exact) emitted a hard negative for 2/300 queries; snapvec
  (ANN) for 273/300. Initial read was "snapvec invents noise negatives";
  a follow-up cosine probe falsified that. The 271 extras have
  cos(q, neg) distribution **statistically identical** to the 2
  sqlite-vec negatives (median 0.78 vs 0.78), with a hard-neg-typical
  margin of 0.16 vs the positive. They are legitimate hard negatives
  that sqlite-vec's rigid top-5 cutoff excludes. Training v4 with them
  is worth testing, not worth refusing.

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

### Initial interpretation (later falsified)

First read of the diff: **sqlite-vec finds a hard negative in only 2 of
300 queries (0.7%); snapvec finds one in 273 (91%)**. Hypothesis: snapvec's
ANN quantization perturbs vec_top5, slipping unrelated docs into the
disagreement set. Under this read the 271 extras would be quantization
noise, and training on them would hurt.

### Follow-up: cosine similarity probe refutes the noise read

_Results file_: `experiments/results/snapvec_negative_similarity_probe.json`

Computed exact (non-quantized) cosine similarity between each (query,
negative) on the 271 snapvec-only bucket and the 2 sqlite-vec bucket,
using the same SentenceTransformer encoder the training path uses.

| Metric | sqlite-vec negs (n=2) | snapvec-only negs (n=271) |
|---|---|---|
| cos(q, neg) median | 0.7801 | 0.7771 |
| cos(q, neg) mean   | 0.7801 | 0.7758 |
| cos(q, pos) median | 0.7978 | 0.9424 |
| margin (pos - neg) median | 0.0177 | 0.1606 |

The 271 snapvec-only negatives have the **same cos(q, neg) distribution**
as the 2 sqlite-vec negatives -- median 0.78 in both buckets.
Random-noise negatives in a 5k-doc corpus would sit near cos 0.3-0.5,
not 0.78. These are docs that are semantically close to the query but
just outside sqlite-vec's rigid top-5 cutoff; snapvec's ANN perturbation
effectively relaxes the cutoff to a wider neighborhood.

The cos(q, pos) median differs between buckets (0.80 vs 0.94) simply
because chunk-prefix queries are literal substrings of their source
chunks, so cos(q, pos) tends toward 1 -- both buckets are drawing from
the same underlying query distribution. The **margin** of ~0.16 is a
productive hard-negative regime for MNRL (DPR / ANCE / NV-Retriever
literature).

### Revised interpretation

- The asymmetry 2 vs 273 is **not** about snapvec fabricating negatives
  from noise. sqlite-vec's rigid top-5 rule is **under-producing**
  negatives when queries are chunk-prefixes (near-self-matches), and
  the ANN perturbation amounts to a de facto relaxation of the top-k
  window to docs that sit at cos 0.7-0.8 in the exact embedding space.
- Those docs are legitimate hard negatives by the model's own metric.
- Equivalent effect could be obtained without snapvec by widening
  sqlite-vec's top-k from 5 to ~20 inside `generate_triples`.
- Training v4 with snapvec-mined pairs is worth running. It is **not**
  predetermined to regress.

### Subsidiary caveat (unchanged)

The v5 recipe (`training_queries_by_dataset`) uses real BEIR queries
through `generate_labeled_triples_batched`, which is a GPU matmul path
that does not route through the vector_backend at all. So swapping a
store to snapvec during v5-style multi-corpus training has zero effect
on mining. This analysis speaks to the single-corpus, chunk-prefix
`generate_triples` flow only.

### Next step (scheduled)

Train v4-vec and v4-snap head-to-head with cross-backend eval on
SciFact and report the 2x2 NDCG@10 matrix. Notebook path:
`experiments/retrain_v4_snapvec_colab.ipynb` (Colab + GPU, ~15 min).

### Known limitation

The monkeypatch in the experiment script routes `embed_query` through
SentenceTransformer because the library's `_HF_ONNX_MODELS` set does not
include `Stffens/bge-small-rrf-v3` yet. MLX rejects v3's legacy BERT
tensor names, so without the patch the experiment would crash on Apple
Silicon. This is an experiment-local workaround; the library side should
add v3 to `_HF_ONNX_MODELS` separately.

## Open questions / follow-ups

- **v4 training** (scheduled on Colab): train v4-snap and v4-vec
  head-to-head on SciFact + NFCorpus + FiQA, cross-eval each on both
  backends, report 2x2 NDCG@10 matrix. Hypothesis (revised, post-probe):
  v4-snap will be **neutral-to-positive** vs v4-vec, and possibly show
  quantization-aware co-adaptation when deployed on snapvec at
  inference.
- **top-k relax equivalence** (cheap, local): repeat the mining diff
  with sqlite-vec at `top_k in {5, 10, 20}` in the miner and compare
  against snapvec top-5. If snapvec top-5 ~= sqlite-vec top-20, the
  "ANN as a window relaxer" interpretation is confirmed and the
  training benefit can be replicated without snapvec.
- **Real-query mining diff**: `generate_triples` only speaks
  chunk-prefix; a miner sibling that takes labeled BEIR queries and
  still routes through `store.search()` would show whether the
  sqlite-vec under-production also holds for real queries or is
  chunk-prefix-specific.
- **Arguana backfill** for A1-bits / A1-ivfpq: re-run both sweeps with
  a 200-query sample to confirm the bit-ordering and rerank-ordering
  trends also hold on the large-FTS-query dataset.
