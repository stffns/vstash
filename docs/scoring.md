# Memory Scoring: Frequency + Temporal Decay

*Added in v0.5.0*

vstash learns which chunks matter to you. Every time you search or ask a question, vstash records which chunks were returned. Over time, frequently-accessed and recently-accessed chunks get a relevance boost, while chunks you haven't touched in months decay naturally.

This means vstash gets better the more you use it — the documents you actually rely on rise to the top.

---

## How It Works

After the standard hybrid search (vector similarity + keyword matching via RRF), vstash applies a second re-ranking pass:

```
final_score = α · normalized_rrf + β · log(1 + access_count · e^(−λ · days_ago))
```

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

## Cold Start

New chunks start with `access_count = 1` (ingestion counts as the first access). This means freshly-ingested documents aren't penalized — they compete on semantic relevance until they accumulate enough access history for the memory component to matter.

---

## Disabling Scoring

To revert to pure RRF ranking:

```toml
[scoring]
enabled = false
```

Access tracking also stops when scoring is disabled (unless you explicitly set `track_access = true`).
