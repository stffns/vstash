# Memory Scoring: Frequency + Temporal Decay with Adaptive Activation

*Added in v0.5.0 · Adaptive maturity gate added in v0.7.0*

vstash learns which chunks matter to you. Every time you search or ask a question, vstash records which chunks were returned. Over time, frequently-accessed and recently-accessed chunks get a relevance boost, while chunks you haven't touched in months decay naturally.

This means vstash gets better the more you use it — the documents you actually rely on rise to the top.

---

## How It Works

After the standard hybrid search (vector similarity + keyword matching via RRF), vstash applies a second re-ranking pass:

```
final_score = α · normalized_rrf + (β · γ) · log(1 + access_count · e^(−λ · days_ago))
```

where **γ** is the adaptive maturity gate (see below).

| Term | Meaning |
|------|---------|
| `normalized_rrf` | The original search relevance score, normalized to [0, 1] |
| `access_count` | How many times this chunk has been returned in searches |
| `days_ago` | Days since the chunk was last accessed |
| `α` (alpha) | How much weight to give semantic relevance |
| `β` (beta) | How much weight to give access history |
| `λ` (decay_lambda) | How fast old accesses lose their influence |

The process:

1. **Over-fetch** — retrieve more candidates than needed (default: 50 instead of `top_k`)
2. **Normalize** — min-max normalize the RRF scores to [0, 1]
3. **Score** — apply the formula above to each candidate
4. **Truncate** — return the top `top_k` results

---

## Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `alpha` | `0.8` | Semantic relevance weight. Higher = trust the search engine more |
| `beta` | `0.2` | Access history weight. Higher = favor frequently-used chunks |
| `decay_lambda` | `0.05` | Decay speed. `0.05` = weeks matter, `0.1` = days matter |
| `over_fetch` | `50` | How many candidates to consider before re-ranking |
| `track_access` | `true` | Whether to record access counts on each search |

---

## Tuning Guide

**Most users don't need to change anything.** The defaults (α=0.8, β=0.2, λ=0.05) were validated across 16 configurations × 5 usage scenarios × 10 queries.

If you do want to tune:

- **"I want pure semantic search, no memory"** → set `enabled = false`
- **"vstash should learn my preferences faster"** → increase `beta` (e.g., 0.4) and decrease `alpha` (e.g., 0.6)
- **"Old documents stay relevant for months"** → decrease `decay_lambda` (e.g., 0.02)
- **"I only care about what I accessed this week"** → increase `decay_lambda` (e.g., 0.15)
- **"Reference docs that I check once should always rank high"** → keep `alpha` high (0.9), `beta` low (0.1)

---

## Configuration

In `vstash.toml`:

```toml
[scoring]
enabled = true        # set to false to disable entirely
alpha = 0.8           # RRF weight
beta = 0.2            # access history weight
decay_lambda = 0.05   # temporal decay rate
over_fetch = 50       # candidates before re-ranking
track_access = true   # record access counts
```

See [Configuration Reference](configuration.md) for the full TOML reference.

---

## Performance

Scoring adds **~0.12ms** to a ~0.7ms search pipeline — negligible overhead. The ANN vector lookup dominates at ~71% of total latency. All stages remain sub-millisecond at P99.

---

## Adaptive Maturity Gate (γ)

*Added in v0.7.0*

The adaptive maturity gate automatically suppresses scoring when access patterns don't carry meaningful signal. It uses the **max/mean ratio** of access counts among accessed chunks:

```
γ = clamp((R - 8.0) / (15.0 - 8.0), 0, 1)

where R = max(access_count) / mean(access_count)   [chunks with access_count > 0]
```

| R (max/mean) | γ | Effect |
|:---:|:---:|--------|
| < 8× | 0.0 | Scoring suppressed — pure RRF, no reranking overhead |
| 8× – 15× | 0.0 – 1.0 | Linear ramp — partial scoring |
| ≥ 15× | 1.0 | Full scoring active |

**Why this matters:** Without γ, a fixed β=0.5 degrades NDCG by -8.6% from day one because uniform access counts inject noise rather than signal. With γ, the system maintains 0.0% degradation across all 30 rounds of a 120-document experiment — scoring only activates when there's a genuine outlier in the access pattern.

**Short-circuit optimization:** When γ = 0, the system skips the re-ranking step entirely — no per-result metadata fetch, no decay computation. Some lightweight bookkeeping still occurs (maturity estimation query, over-fetch sizing), so overhead during cold start is minimal rather than strictly zero.

The gate requires at least 10 accessed chunks to activate, avoiding false triggering on tiny corpora.

---

## Cold Start

New chunks start with `access_count = 0` (ingestion does not count as an access). This means freshly-ingested documents aren't penalized — they compete on semantic relevance until they accumulate enough access history for the memory component to matter.

With the adaptive maturity gate (v0.7.0), scoring can be **enabled by default** with no cold start penalty. The system transitions seamlessly from pure RRF to frequency-augmented ranking as usage patterns mature, with zero user intervention.

---

## Intra-Document MMR Deduplication

*Added in v0.8.0*

Before v0.8, vstash used hard per-document dedup: only the highest-scoring chunk from each document appeared in results. This works well for short documents but hides relevant content in long documents — a book with two important chapters on different topics would only show one.

v0.8 replaces this with **Maximal Marginal Relevance (MMR)** applied within documents:

```
MMR(c) = λ · score(c) − (1 − λ) · max_sim(c, selected_same_doc)
```

| Term | Meaning |
|------|---------|
| `λ` (mmr_lambda) | Balance between relevance and diversity (default 0.5) |
| `score(c)` | Normalized RRF or final_score of the candidate |
| `max_sim` | Maximum cosine similarity to already-selected chunks from the same document |

**How it works:**
1. Chunks from *different* documents compete purely on score — no cross-document penalty.
2. When multiple chunks from the *same* document are candidates, the second chunk is penalized by its similarity to the first.
3. If two chapters are semantically diverse (low cosine similarity), both appear. If they're near-duplicates, the second is suppressed.
4. Selection stops when the best remaining candidate has negative MMR (redundancy exceeds relevance).

**Configuration:**

```toml
[scoring]
mmr_lambda = 0.5   # 0.0 = max diversity, 1.0 = hard dedup (one per doc)
```

**Impact:** On a 35-chunk paper, cross-section queries return 3–5× more relevant results than hard dedup. NDCG@5 improves from 0.814 to 0.829. Overhead is ~0.36ms (embedding fetch + similarity matrix).

---

## Relevance Signal

*Added in v0.6.0*

Separate from scoring, vstash provides a **distance-based relevance signal** that works from the very first search — no scoring or access history required. The cosine distance of the best vector match estimates confidence:

| Distance | Tier | F1 |
|----------|------|-----|
| ≤ 0.95 | high | 0.952 |
| 0.95–0.98 | medium | — |
| > 0.98 | low | — |

This is applied in CLI (search, ask, chat) and MCP server. See [How It Works](how-it-works.md) for details.

---

## Discard Telemetry

*Added in v0.6.0*

Every search records an event with query, distance, relevance tier, and result count to a `search_events` table. In chat mode, events are marked as "dismissed" when the user exits after a non-high result. The table is pruned to 1,000 entries and indexed by `(relevance_tier, created_at)`.

This allows validating the relevance signal against real-world usage patterns over time.

---

## Disabling Scoring

To revert to pure RRF ranking:

```toml
[scoring]
enabled = false
```

Access tracking also stops when scoring is disabled (unless you explicitly set `track_access = true`).
