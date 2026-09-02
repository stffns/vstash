# Recency Boost & Temporal Filters

*Recency boost added in v0.19.0 · Replaces frequency+decay scoring (removed in v0.18.0)*

vstash supports two complementary mechanisms for time-aware search:

1. **Recency boost** — a soft ranking bias that favors recently created chunks
2. **Temporal filters** — hard date boundaries that exclude documents outside a range

Both are opt-in and off by default, so pure retrieval quality is unaffected.

---

## Recency Boost

After hybrid search (vector + FTS5 + RRF), an optional recency multiplier biases scores toward recent content:

```
boosted_score = rrf_score × (1 + recency_boost × e^(−0.05 × days_ago))
```

| Term | Meaning |
|------|---------|
| `rrf_score` | The original hybrid search score from RRF fusion |
| `recency_boost` | Multiplier strength (0.0 = off, 1.0 = strong) |
| `days_ago` | Days since the chunk was created |

**Decay curve:**

| Age | Decay factor | Effect at boost=1.0 |
|-----|:---:|---------|
| Today | 1.00 | Score doubled |
| 1 week | 0.70 | +70% boost |
| 1 month | 0.22 | +22% boost |
| 3 months | 0.01 | ~1% boost (negligible) |
| 1 year | ~0 | No effect |

### When to use it

- **Agentic memory** — an agent's recent context is more likely to be relevant
- **Active projects** — recent notes and decisions matter more than old ones
- **Conversational recall** — "what was I working on?" benefits from recency

### When NOT to use it

- **Reference retrieval** — "how does OAuth2 work?" has a timeless answer
- **Benchmarking** — leave it off (0.0) for BEIR-style evaluations
- **Mixed corpora** — if old documents are equally important, recency adds noise

---

## Temporal Filters

Hard date boundaries that filter at the SQL level — no ranking pollution:

```python
# Only documents added in Q1 2024
results = store.search(query, added_after="2024-01-01", added_before="2024-04-01")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `added_after` | ISO date string | Only documents added on or after this date |
| `added_before` | ISO date string | Only documents added before this date |

Filters apply to `documents.added_at` (ingestion timestamp). They work independently of `recency_boost` and can be combined with it.

---

## Usage

### Python SDK

```python
from vstash import Memory

mem = Memory(project="my_agent")

# Pure retrieval (no recency bias)
chunks = mem.search("OAuth2 PKCE flow")

# Agentic recall (favor recent context)
chunks = mem.search("what was the deploy issue?", recency_boost=1.0)

# Time-bounded search
chunks = mem.search("meeting notes", added_after="2024-06-01")

# Combined: recent + time window
chunks = mem.search("project decisions", recency_boost=0.5, added_after="2024-01-01")
```

### CLI

Recency boost and temporal filters are available via the Python SDK and MCP tools. CLI support for `--recency-boost`, `--added-after`, and `--added-before` flags is planned.

### MCP Server

`vstash_search` accepts `recency_boost`, `added_after`, and `added_before`. `vstash_ask` accepts `added_after` and `added_before` (no recency boost — ask is pure retrieval + LLM).

---

## Configuration

In `vstash.toml`:

```toml
[recency]
boost = 0.0   # default recency_boost for all searches (0.0 = off)
```

Per-call `recency_boost` overrides the config default.

---

## Intra-Document MMR Deduplication

After recency boost (or directly after RRF if boost is off), vstash applies **Maximal Marginal Relevance (MMR)** within documents to balance relevance and diversity:

```
MMR(c) = λ · score(c) − (1 − λ) · max_sim(c, selected_same_doc)
```

| Term | Meaning |
|------|---------|
| `λ` | Balance between relevance and diversity (fixed at 0.5) |
| `score(c)` | RRF score (or boosted score if recency is active) |
| `max_sim` | Maximum cosine similarity to already-selected chunks from the same document |

Chunks from *different* documents compete purely on score. When multiple chunks from the *same* document are candidates, the second is penalized by its similarity to the first. If two chapters are semantically diverse, both appear; if they're near-duplicates, the second is suppressed.

---

## Cross-Document Near-Duplicate Collapse

MMR is deliberately intra-document, which leaves one gap: when the *same
answer* is mirrored across many **different** documents -- audit logs, a
re-ingested file, notes copied between collections -- nothing suppresses
them, and they can take every result slot.

`dedup_threshold` is the opt-in cross-document counterpart. A chunk is
dropped when its cosine similarity to an already-kept, **higher-ranked**
chunk is `>= dedup_threshold`, regardless of which document each belongs to.

```python
store.search(q_emb, query, top_k=10)                        # unchanged
store.search(q_emb, query, top_k=10, dedup_threshold=0.95)  # collapse
```

| Property | Behaviour |
|----------|-----------|
| Default | `None` -- the collapse never runs and ranking is unchanged |
| Range | `(0.0, 1.0]`; out-of-range raises `ValueError` |
| `1.0` | Collapses exact duplicates only (with a float32 tolerance -- the dot product of identical vectors often lands on `0.99999994`) |
| `0.95` | Reasonable starting point for restated content |
| Order | Keep-first: the highest-scoring member of a cluster survives |
| Placement | After the recency boost, **before** MMR and the `top_k` cut, so a slot freed by a duplicate is refilled with distinct content rather than shortening the result list |
| Missing embedding | Always kept -- an unknown vector is never treated as a duplicate |

Available on every surface:

```bash
vstash search "query" --dedup 0.95
```

```python
memory.search("query", dedup_threshold=0.95)          # SDK
```

MCP exposes it as the `dedup_threshold` argument of `vstash_search`.
Under `--all-profiles`, the collapse is applied **per profile** before the
cross-profile RRF merge: two profiles that each hold a copy of the same
document still contribute one result apiece, because cross-profile
comparison would need embeddings from stores that are already closed by
merge time.

### When to use it

Reach for it when a corpus holds redundant *documents* rather than
redundant chunks: agent audit trails, mirrored note vaults, snapshots of
a file ingested repeatedly. A store of distinct documents gains nothing
from it and should leave it off.

Cost is one embedding fetch over the candidate pool plus a similarity
compare of each candidate against the ones kept so far. At the default
`top_k` the pool is 50-200 and this is negligible. It grows with `top_k`
(the pool is `min(top_k * 10, ...)` over the vector + FTS union), so a
`top_k` in the thousands on a large corpus makes it measurable: 20 000
candidates cost ~3.7 s and ~314 MB.

---

## Relevance Signal

Separate from recency and scoring, vstash provides a **distance-based relevance signal** that works from the very first search:

| Distance | Tier | F1 |
|----------|------|-----|
| ≤ 0.95 | high | 0.952 |
| 0.95–0.98 | medium | — |
| > 0.98 | low | — |

This is applied in CLI (search, ask, chat) and MCP server.

---

## Historical Note

v0.5.0–v0.17 included a frequency+decay scoring system that re-ranked results using access counts and temporal decay with an adaptive maturity gate (γ). This was removed in v0.18.0 after benchmarks showed it degraded NDCG on all tested datasets (SciFact: -1.6%, scoring grid: 0%, cross-encoder: -0.3% to -3.1%). The database columns `access_count` and `last_accessed_at` are preserved for backward compatibility. The `created_at` column is actively used by the v0.19 recency boost computation.

The v0.19.0 recency boost is a simpler, more targeted replacement: pure temporal decay without access counting, opt-in rather than global, and applied as a multiplicative boost rather than a weighted blend.
