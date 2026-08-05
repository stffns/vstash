# T2.4 -- Cross-encoder reranker design

Status: draft, 2026-04-19. No code yet. Motivated by the NFCorpus gap
that survived T1.5 (labeled queries) and H-R9 (corpus balance /
volume ablation). A small cross-encoder on top of the current
bi-encoder + adaptive-RRF stack is expected to close the residual
gap cleanly and is orthogonal to everything we have shipped.

## Problem

Current pipeline is bi-encoder only:

```
query -> embed -> vec top-K
query -> FTS5 -> keyword top-K
        \____ adaptive RRF ____/
              |
         MMR dedup
              |
         top_k results
```

RRF ordering is good for recall (Recall@100 is healthy on every
BEIR dataset we tested) but calibration of the top-10 head is
limited because both signals are symmetric and neither looks at
the query-document pair jointly. A cross-encoder scoring each
top-N candidate with the query re-ranks the head with much sharper
precision (literature: +3-8 NDCG@10 on BEIR vs pure bi-encoder).

## Goal

Add a `rerank(query, candidates, top_k)` stage after MMR dedup,
opt-in via config, that:

- Takes the top-N RRF results (default N=50).
- Re-scores each with a small cross-encoder.
- Re-orders and truncates to the user's top_k.
- Preserves backward compat: off by default, zero cost when off.

Target: +3 NDCG@10 on NFCorpus (0.3677 -> ~0.395), +2 on SciFact
(0.7786 -> ~0.800), +2 on FiQA (0.4568 -> ~0.477). Combined with
the current bi-encoder, this would lift the multi-domain macro to
~0.56 NDCG@10 and close the v5 NFCorpus gap without touching the
training data pipeline.

## Non-goals

- Training our own reranker (T2.5 territory, later). Start with a
  HuggingFace off-the-shelf model.
- Replacing the bi-encoder. Reranker runs on top, bi-encoder
  remains the retrieval engine.
- Streaming / on-device rerank for mobile. Desktop-class only in
  v1.

## Model selection

Two candidates from the ms-marco family, both battle-tested on
BEIR:

| Model | Params | Size | Latency (CPU top-50) | BEIR avg |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ~85 MB | ~50 ms | strong |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | ~130 MB | ~90 ms | stronger |

Default: L-6-v2. Same size class as bge-small, same latency
budget. L-12-v2 as an opt-in for users who want the extra +0.5
NDCG at 2x cost. Config:

```toml
[rerank]
enabled = false
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_n = 50   # how many RRF candidates to rerank
batch_size = 16
device = "auto"  # cpu / cuda / mps
```

## API surface

### Python SDK

```python
store.search(
    query_embedding=...,
    query_text="...",
    top_k=10,
    rerank=True,  # NEW: opt-in per call, overrides config
    rerank_top_n=50,  # NEW: how many pre-rerank candidates
)
```

Or static config via `vstash.toml`'s `[rerank]` section (the opt-in
default path so users do not have to pass `rerank=True` on every
call).

### CLI

```bash
vstash search "what is RRF?" --rerank
vstash search "what is RRF?" --no-rerank   # force off
vstash config set rerank.enabled true      # default-on per profile
```

### MCP

`rerank: bool` param on the `search` tool schema. Defaults to
`config.rerank.enabled` so MCP clients (Claude Desktop, etc.)
inherit the project default.

## Pipeline integration

Insert the rerank stage AFTER MMR dedup, BEFORE top_k truncation:

```
RRF fuse  ->  MMR dedup  ->  take top-N (N=rerank_top_n)  ->
rerank(query, candidates)  ->  sort by rerank score  ->  top_k
```

Placement rationale:

- After MMR: MMR already collapses intra-doc duplicates; running
  rerank on 50 unique docs is 50 cross-encoder calls, not 50
  chunks of the same doc.
- Before truncation: we want rerank to see a wider pool than
  `top_k` so it can promote docs ranked 11-50 into the top 10.
  `top_n = 5 * top_k` is the literature default.

Fallback when rerank fails (model load error, timeout, low memory):
log a warning, skip rerank, return the RRF-only ranking. No hard
failure on the search path.

## Implementation plan

1. **`vstash/rerank.py`** (new).
   - `class Reranker`: wraps `sentence_transformers.CrossEncoder`
     with lazy load, batch inference, optional MPS/CUDA.
   - `rerank(query: str, candidates: list[SearchResult],
     top_k: int) -> list[SearchResult]`.
   - Each `SearchResult` gets a new `rerank_score: float | None`
     field in `models.py`.

2. **`vstash/store.py`**.
   - `search(..., rerank: bool = False, rerank_top_n: int = 50)`
     plumbed through. When True, call `Reranker` after MMR.
   - If `cfg.rerank.enabled` is True and the call does not pass an
     explicit `rerank=...`, default to on.

3. **`vstash/cli.py`**.
   - `search` command: `--rerank / --no-rerank`,
     `--rerank-top-n`.
   - `config` command: accepts `rerank.*` keys.

4. **`vstash/config.py`**.
   - New `RerankConfig` Pydantic model with `enabled`, `model`,
     `top_n`, `batch_size`, `device`.

5. **`vstash/mcp.py`**.
   - Extend search tool schema with `rerank` bool.

6. **Tests (`tests/test_rerank.py`)**.
   - Mock `CrossEncoder` to return controlled scores.
   - Verify rerank reorders correctly (high-rerank candidates
     bubble up).
   - Verify `rerank=False` preserves RRF order.
   - Verify fallback on model load error keeps search working.
   - End-to-end: populate store, search with rerank on, assert a
     known relevant doc moves up.

7. **Benchmark (`experiments/rerank_beir_bench.py`)**.
   - Run BEIR SciFact + NFCorpus + FiQA with and without rerank,
     report NDCG@10 delta and latency p50 / p95 at top_k=10 with
     rerank_top_n=50.
   - Target: NFCorpus NDCG@10 >= 0.39 with rerank enabled;
     added latency < 300 ms at p95 on M1 CPU.

## Latency budget

Current search p50 on an M1 Pro for a medium corpus (bge-small +
sqlite-vec + FTS5 + adaptive RRF + MMR): ~15-25 ms.

Rerank adds one cross-encoder forward pass per candidate:

- L-6-v2, batch_size=16, CPU: ~1-2 ms per candidate -> ~50-100 ms
  for top_n=50.
- L-6-v2, batch_size=50, MPS: ~25 ms total.
- L-12-v2, CPU: ~2-4 ms per candidate -> ~100-200 ms.

So rerank-on raises search p50 from ~20 ms to ~70-120 ms on CPU.
Acceptable for most interactive use. For sub-50 ms workloads, the
user can leave rerank off or drop `top_n` to 20.

## Risk / open questions

1. **Model download on first use**: ~85 MB for L-6-v2. Handle
   like we handle embedding models today: auto-download on first
   call via FastEmbed / HF cache. Document in the `rerank`
   docstring.
2. **Memory residency**: loading the reranker adds ~250 MB RAM
   (params + activation buffers). Keep lazy: only instantiate on
   first rerank call, allow explicit `Reranker.unload()`.
3. **Training-data assumption**: ms-marco cross-encoders are
   trained on question-passage data. They generalize well to BEIR
   but may underperform on highly domain-specific corpora (code,
   legal, medical). For those, T2.5 (train our own reranker) is
   the follow-up.
4. **MCP streaming**: Claude Desktop benefits from streaming
   responses. Rerank adds ~100 ms upfront, delaying the first
   token. Either render progressively (top-5 from RRF first,
   reranked top-10 after) or document the added latency. Defer
   decision to implementation.
5. **Interaction with recency boost and scoring**: the current
   recency boost multiplies RRF score post-MMR. Rerank score
   should replace RRF score for ordering, but we might want to
   keep a `final_score = 0.8 * rerank_score + 0.2 * recency_mult`
   hybrid. Prototype first, calibrate via experiment.

## Effort

5-6 days of focused work.

- Day 1: `vstash/rerank.py` + config + basic unit tests.
- Day 2: wire into `store.search` + fallback + SDK tests.
- Day 3: CLI + MCP surfaces + end-to-end test.
- Day 4: BEIR benchmark + latency profiling.
- Day 5: docs (add `docs/rerank.md`, update README + constitution).
- Day 6: buffer + review fixes.

## Success criteria

- `vstash search --rerank` enabled on BEIR SciFact: NDCG@10
  >= 0.79 (bi-encoder baseline 0.7786, target +0.01 min).
- BEIR NFCorpus: NDCG@10 >= 0.39 (bi-encoder baseline 0.3677,
  target +0.022 min). **Closes the residual v5 gap.**
- BEIR FiQA: NDCG@10 >= 0.47 (bi-encoder baseline 0.4568,
  target +0.013 min).
- Added p95 latency < 300 ms at top_n=50 on M1 CPU.
- All 974+ existing tests still pass (rerank is opt-in, default
  off).
- New reranker module: >= 90% line coverage.

## Rollout

1. v1 ships as opt-in (`[rerank] enabled = false` default).
2. One minor release cycle with feedback from early adopters.
3. Consider flipping default to on in v2 if benchmarks confirm
   universal gain and latency acceptable.
4. Document the flip in a paper addendum / model card v3.

## Follow-ups (Tier 3 area)

- **T2.5**: train our own reranker on user-corpus disagreement
  pairs (reuse T1.5 labeled miner). Owns user-domain
  personalization.
- **T3.x**: calibrated rerank score as part of `SearchResult`,
  usable by apps for confidence thresholds.
