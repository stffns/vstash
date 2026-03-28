# vstash Product Improvements — Validation Report

Post-paper changes that improve vstash as a daily-use tool. Each change is driven by a hypothesis from experiment findings, validated with E2E tests and benchmarks on the full 24-paper corpus (786 chunks).

---

## Changes

### 1. Document-level deduplication in search

**Hypothesis:** Multiple chunks from the same document flood top-k results, reducing diversity without adding value.

**Change:** After RRF ranking, keep only the highest-scoring chunk per document before truncating to `top_k`.

**Validation:**

| Metric | Before | After |
|--------|-------:|------:|
| NDCG@5 | 0.814 | **0.829** (+1.8%) |
| Unique docs per top-5 | ~3.2 | **5.0** (perfect) |
| Queries with duplicate docs | 4/10 | **0/10** |

Dedup improves both diversity *and* retrieval quality — eliminating redundant chunks lets more relevant documents surface.

### 2. Context expansion for ask/chat

**Hypothesis:** A single chunk (~250 tokens) is too narrow for the LLM to generate a good answer. Including adjacent chunks (±1) gives richer context at minimal cost.

**Change:** `expand_context(results, window=1)` fetches neighboring chunks by `doc_id + seq` and concatenates them before sending to the LLM.

**Validation:**

| Metric | Value |
|--------|------:|
| Avg text expansion | **2.64x** |
| Metadata preserved | ✓ (title, path, chunk, score) |
| Latency overhead | **0.12ms** (8.5% of base search) |

2.64x more context per result with sub-millisecond overhead.

### 3. Relevance signal — distance-based (CLI + MCP)

**Hypothesis:** The score-spread signal (`spread < 0.15 → low relevance`) only discriminates at 1.22-1.24x even with heavy scoring history (30 rounds). The fundamental problem: spread depends on access count variance within the result batch, not on query-corpus semantic proximity. Vector distance of the best match directly measures "how far is this query from the nearest document?" — a much stronger signal.

**Analysis (combined_signal_analysis.py):**

| Signal | Relevant avg | Irrelevant avg | Ratio | Overlap |
|--------|----------:|----------:|------:|--------:|
| Spread (raw RRF) | 0.007 | 0.007 | 1.00x | 10/10 |
| Spread (scoring ON, 10 rounds) | 0.305 | 0.245 | 1.24x | 10/10 |
| **Vector distance** | **0.594** | **0.978** | **1.65x** | **0/10** |

Spread has complete overlap between relevant and irrelevant queries — no threshold can cleanly separate them. Distance has zero overlap.

**Change:** Replace spread-based signal with `store.last_best_distance > 0.95` across all interfaces:
- CLI search/ask/chat: warn when best match is semantically distant
- MCP: return `relevance: "low"/"high"` with `best_distance` in response JSON
- Works from first search — no scoring, no history, no warm-up needed

**Validation (classification strategies at threshold 0.95):**

| Strategy | Precision | Recall | F1 | Accuracy |
|----------|----------:|-------:|---:|--------:|
| spread > 0.15 (best fixed) | 0.500 | 1.000 | 0.667 | 0.500 |
| **distance < 0.95** | **0.909** | **1.000** | **0.952** | **0.950** |

F1 improves from 0.667 → 0.952. The signal is autonomous — no human tuning, no scoring dependency, works from day zero.

### 4. Access tracking decoupled from scoring

**Hypothesis:** If accesses are only tracked when scoring is enabled, users who disable scoring can never accumulate the history needed to benefit from scoring later. The cold start problem becomes permanent.

**Change:** Track accesses whenever `track_access=True`, regardless of `scoring.enabled`. Suggest enabling scoring after 500+ accumulated accesses.

**Validation:**

| Config | Accesses recorded |
|--------|:-:|
| `enabled=False, track_access=True` | ✓ (+15 per 3 queries) |
| `enabled=False, track_access=False` | ✗ (delta = 0) |
| `total_access_count()` vs manual SQL | Match (465 = 465) |

---

## Latency Impact

Full pipeline (search + scoring + dedup + context expansion) on 786 chunks:

| Pipeline | Avg | P50 | P95 | P99 |
|----------|----:|----:|----:|----:|
| Search + dedup | 1.43ms | 1.41ms | 1.57ms | 1.70ms |
| + Scoring | 2.92ms | 2.87ms | 3.44ms | 3.62ms |
| + Context expansion | **3.41ms** | **3.36ms** | 3.97ms | 4.10ms |

Context expansion adds 0.12ms. The full pipeline stays under 4ms at P50 — imperceptible.

---

## Iteration 2: Refinements

Based on the initial validation, three refinements were implemented:

### 5. MCP context injection

**Problem:** Claude Code received raw single chunks (~250 tokens) from `vstash_search`, limiting answer quality.

**Change:** `vstash_search` now calls `expand_context(window=1)` before returning results. The LLM gets 2.64x more context per result at +0.12ms cost.

### 6. Adaptive relevance threshold → superseded by distance signal

**Status:** The `search_stats` table, `record_spread()`, and `adaptive_relevance_threshold()` methods remain in `store.py` but are no longer called in production. The adaptive spread threshold (mean - 1σ) barely improved over fixed (F1 0.640 vs 0.621) because the underlying spread signal is too weak. The distance-based signal (§3) eliminates the need for adaptive thresholds entirely — it works with a fixed cutoff of 0.95 and achieves F1=0.952.

### 7. Scoring progress indicator

**Problem:** The 500-access threshold for scoring appeared suddenly. Users had no visibility into warm-up progress.

**Change:** After 50+ accesses (~10 searches), show a progress bar:

```
Learning preferences: ████████░░░░░░░░░░░░ 40% (200/500)
```

At 500+, the message changes to a call-to-action to enable scoring.

---

### 8. Tiered ghost warning

**Problem:** The binary "low relevance" warning was too blunt — it either warned or didn't. Users with borderline queries (distance 0.95-0.98) got no signal.

**Change:** Three-tier visual feedback based on vector distance:

| Distance | Tier | CLI behavior |
|----------|------|-------------|
| ≤ 0.95 | high | No indicator — full confidence |
| 0.95–0.98 | medium | Subtle `?` next to result rank + "Uncertain relevance" note |
| > 0.98 | low | Full warning: "Low relevance — results may not match" |

Applied consistently across `search`, `ask`, `chat`, and MCP.

### 9. Discard telemetry

**Problem:** The F1=0.952 is measured on a synthetic benchmark. No way to validate whether the relevance signal helps users in real-world usage.

**Change:** Every search records an event in `search_events` with query, distance, tier, and result count. A `dismissed` flag can be set when the user exits without engaging. `search_telemetry_summary()` returns dismiss rates grouped by tier.

**Validation scenario:** If users dismiss "low" tier results at 3-5x the rate of "high" tier, the signal is confirmed useful in production.

---

## Test Coverage

- **Unit tests:** 331 passed (19 new: dedup, expand_context, access tracking, adaptive threshold, telemetry × 4, relevance tier × 3)
- **E2E tests:** 11/11 passed across dedup, expansion, signal, and tracking
- **Benchmarks:** 4 latency benchmarks (50 iterations each)

---

## Remaining Opportunities

### Short-term

| Opportunity | Effort | Impact |
|-------------|--------|--------|
| **Configurable dedup** — `--group-by-doc` flag to optionally show multiple chunks per doc | 1h | Users who want chunk-level results get them back |
| **Expand window config** — `chunking.context_window` in vstash.toml | 30min | Users with large chunks can skip expansion |

### Medium-term

| Opportunity | Effort | Impact |
|-------------|--------|--------|
| **Chunk grouping in CLI** — show "3 more chunks from this doc" collapsed under the best chunk | 2-3h | Best of both worlds: diversity + depth |
| **Implicit feedback** — track which results the user expands/copies to refine adaptive threshold | 3-4h | Closes the loop between usage and quality |

### Research

| Opportunity | Effort | Impact |
|-------------|--------|--------|
| **Maximal Marginal Relevance (MMR)** — replace simple dedup with diversity-aware re-ranking | 2-3h | Better diversity/relevance tradeoff than hard dedup |
| **Late interaction re-ranking** — ColBERT-style token-level scoring as a second stage | 4-6h | Could close the gap between FTS and vector on diverse corpora |
| **Sliding window expansion** — dynamically choose window size based on chunk boundary quality | 2-3h | Smarter expansion for code vs prose |
