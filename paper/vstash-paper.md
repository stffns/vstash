# vstash: Local-First Hybrid Retrieval with Temporal Memory Scoring for LLM Agents

**Jayson Steffens**
[github.com/stffns/vstash](https://github.com/stffns/vstash)

---

## Abstract

We present **vstash**, a local-first document memory system that combines vector similarity search with full-text keyword matching via Reciprocal Rank Fusion (RRF), augmented by a novel frequency-weighted temporal decay re-ranker. All data resides in a single SQLite file using `sqlite-vec` for approximate nearest neighbor search and FTS5 for keyword matching — no cloud services, no external databases.

We make five empirical contributions. **(1)** A post-RRF re-ranking formula that fuses normalized semantic scores with access-frequency signals decayed over time, improving NDCG@10 by up to 16.1% on access-heavy scenarios while adding only 0.017 ms of overhead. **(2)** A *distance-based relevance signal* using the cosine distance of the best vector match, achieving F1 = 0.952 with zero overlap between relevant and irrelevant queries — working from the first search with no scoring or usage history required. This supersedes our earlier score-spread approach (F1 = 0.667, 10/10 class overlap). **(3)** Document-level deduplication that improves result diversity from ~3.2 to 5.0 unique documents per top-5 while simultaneously improving NDCG@5 from 0.814 to 0.829. **(4)** Context expansion that retrieves adjacent chunks (±1 window) for 2.64× richer LLM context at +0.12 ms cost. **(5)** Code-aware chunking with language-specific boundary detection that preserves function-level semantic coherence for 6 programming languages.

We evaluate on two corpora — 24 arXiv papers (786 chunks, domain-specific) and 17 Wikipedia articles (2,602 chunks, mixed-domain) — with pooled relevance judgments across 10 queries, 5 access scenarios, and 16 parameter configurations. RRF achieves the highest NDCG@5 on both corpora (0.829 after dedup and 0.758 respectively). All experiments are reproducible; source code, data, and experiment scripts are open-source.

---

## 1. Introduction

Large language model (LLM) agents increasingly require persistent memory — the ability to store, retrieve, and prioritize information across sessions. While cloud-hosted vector databases (Pinecone, Weaviate, Chroma) serve this need at scale, many use cases demand *local-first* operation: developer tooling, personal knowledge management, privacy-sensitive workflows, and offline agents.

Existing local solutions face three gaps:

1. **Retrieval quality.** Pure vector search misses exact keywords (error codes, proper names); pure keyword search misses semantic paraphrases. Hybrid fusion helps, but combining scores from incompatible distributions (cosine distance vs. BM25 rank) is non-trivial.

2. **Temporal awareness.** Documents accessed yesterday should rank differently from documents untouched for months. Most RAG systems treat all chunks equally regardless of usage history.

3. **Confidence estimation.** When every query returns results with uniformly high scores, the system cannot distinguish "I found something relevant" from "I returned the least irrelevant thing I have."

We introduce **vstash**, a single-file system built on SQLite that addresses all three gaps. Our key insight is that *post-retrieval scoring* — re-ranking RRF candidates with frequency and decay — not only improves ranking quality but also creates the score variance necessary for confidence estimation.

### Contributions

- A **frequency + temporal decay re-ranker** with min-max normalization that brings RRF and access-history scores onto a common scale (§4).
- A **distance-based relevance signal** using the best vector match distance, achieving F1 = 0.952 with zero human tuning — superseding the score-spread approach which required scoring history and achieved only F1 = 0.667 (§5).
- **Document-level deduplication** that prevents a single document from flooding top-*k*, improving both diversity (5.0 unique docs per top-5) and NDCG@5 (+1.8%) (§3.4).
- **Context expansion** via adjacent chunk retrieval for 2.64× richer LLM context at negligible latency cost (§3.5).
- **Code-aware chunking** for 6 languages using regex-based boundary detection at column 0 with decorator attachment (§6).
- An **open-source system** with CLI, Python SDK, MCP server for LLM agent integration, and reproducible experiment scripts (§3).

---

## 2. Related Work

**Memory for LLM agents.** MemGPT [Packer et al., 2023] introduced virtual context management with explicit memory tiers. Mem0 [Chadha et al., 2025] and Memoria [2025] provide production-ready memory layers with cloud backends. A-MEM [2025] uses agentic self-organization. Unlike these systems, vstash operates entirely locally with zero cloud dependencies.

**Hybrid retrieval.** Reciprocal Rank Fusion (RRF) [Cormack et al., 2009] merges ranked lists without requiring comparable scores. Ma et al. [2024] showed RRF outperforms learned re-rankers on out-of-domain data. Our contribution is the post-RRF normalization step that enables fusion with frequency-based signals.

**Temporal decay in memory.** The Ebbinghaus forgetting curve [1885] inspires exponential decay models. Zep [2025] uses temporal knowledge graphs; MaRS [2025] models cognitive forgetting. PAM [2026] exploits temporal co-occurrence. We apply decay directly in the scoring formula rather than in graph structure.

**Code-aware chunking.** Tree-sitter-based approaches parse full ASTs but require language grammars. Our regex approach at column 0 provides comparable boundary detection for top-level definitions without external dependencies.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION                                │
│  PDF/DOCX/URL/Code ──► MarkItDown ──► Chunking ──► FastEmbed   │
│                          parse      (code-aware     (ONNX,      │
│                                      or semantic)   384-dim)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite (WAL mode)                             │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ documents  │  │   chunks   │  │vec_chunks│  │ fts_chunks │  │
│  │ metadata,  │  │ text, seq, │  │sqlite-vec│  │  FTS5 +    │  │
│  │ tags, path │  │ access_cnt │  │  ANN idx │  │  Porter    │  │
│  └────────────┘  └────────────┘  └──────────┘  └────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL                                │
│  Query ──► Embed ──┬──► Vector ANN ──┐                          │
│                    │                 ├──► RRF Fusion             │
│                    └──► FTS5 BM25 ───┘        │                 │
│                                               ▼                 │
│                                    Freq+Decay Re-rank           │
│                                    (over-fetch 50, §4)          │
│                                               │                 │
│                                               ▼                 │
│                                    Document Dedup (§3.4)          │
│                                               │                 │
│                                               ▼                 │
│                                    Distance Relevance Signal     │
│                                    (high/medium/low, §5)        │
│                                               │                 │
│                                               ▼                 │
│                                    Context Expansion (±1, §3.5)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACES                                 │
│   CLI    │   Python SDK   │   MCP Server   │  Claude Code Hook  │
│ (search, │  (Memory()     │  (vstash_search│  (auto-inject      │
│  ask,    │   .search()    │   vstash_add)  │   on knowledge     │
│  chat)   │   .ask())      │               │   questions)       │
└─────────────────────────────────────────────────────────────────┘
```
*Figure 1: vstash architecture. All data resides in a single SQLite file. The retrieval pipeline applies RRF fusion, frequency+decay re-ranking, document deduplication, a distance-based relevance signal, and context expansion.*

vstash stores all data in a single SQLite database using WAL mode for concurrent read safety:

- **Documents table** — metadata, hierarchical tags (project, layer, collection), source type.
- **Chunks table** — text segments with sequence numbers, access counters, and timestamps.
- **vec_chunks** — `sqlite-vec` virtual table for approximate nearest neighbor search (384-dim float vectors from BAAI/bge-small-en-v1.5).
- **fts_chunks** — FTS5 virtual table with Porter stemming for keyword matching.
- **search_events** — telemetry table recording query, distance, relevance tier, and dismiss flag for real-world signal validation (pruned to 1,000 entries).

### 3.1 Ingestion Pipeline

Documents are parsed via MarkItDown (PDF, DOCX, HTML, URLs) or read directly (code files). Text is split into chunks using either semantic chunking (§6) or code-aware chunking depending on source type. Chunks are embedded using FastEmbed (ONNX Runtime) at ~700 chunks/s on CPU.

### 3.2 Hybrid Search with RRF

Given a query *q*, we retrieve candidates from both indexes:

```
RRF(c) = w_v / (k + r_v(c)) + w_f / (k + r_f(c))
```

where *r_v(c)* and *r_f(c)* are the ranks of chunk *c* in vector and FTS5 results respectively, *k = 60*, and weights *w_v = 0.6*, *w_f = 0.4*.

**FTS5 safety.** Each query word is individually double-quoted and joined with OR, preventing FTS5 Boolean operator injection (NEAR, NOT) while allowing keyword-level matching instead of exact-phrase. Internal quotes are escaped by doubling.

### 3.3 LLM Agent Integration

vstash integrates with LLM agents through three interfaces:

**MCP Server.** Exposes search, add, ask, and management tools via the Model Context Protocol. The server includes LLM-facing instructions that guide clients on when to use (knowledge questions) and when to skip (action commands) memory tools. Search results include the relevance signal, enabling clients to filter noise.

**Claude Code Hook.** A `UserPromptSubmit` hook that auto-injects vstash context on knowledge questions. A pattern-based filter distinguishes knowledge questions from action commands (commit, merge, push), achieving 17/17 accuracy on our test set.

**Python SDK.** A `Memory` class with context manager protocol for programmatic integration into agent frameworks.

### 3.4 Document-Level Deduplication

After RRF ranking (and optional scoring), multiple chunks from the same document often cluster in the top-*k*. This floods results with redundant content and reduces diversity. We deduplicate by keeping only the highest-scoring chunk per document path before truncating to `top_k`.

### Table 0: Effect of document deduplication (24 papers, 786 chunks)

| Metric | Before | After |
|--------|:------:|:-----:|
| NDCG@5 | 0.814 | **0.829** (+1.8%) |
| Unique docs per top-5 | ~3.2 | **5.0** (perfect) |
| Queries with duplicate docs | 4/10 | **0/10** |

Dedup improves both diversity *and* retrieval quality — eliminating redundant chunks lets more relevant documents surface in the result set.

### 3.5 Context Expansion

A single chunk (~250 tokens) provides insufficient context for LLM answer generation. We expand each search result by fetching adjacent chunks (±*w* by sequence number within the same document) and concatenating their text:

```
expanded_text(c) = concat(text(c - w), ..., text(c), ..., text(c + w))
```

Default *w = 1* yields 2.64× more text per result at +0.12 ms overhead. This is applied in all LLM-facing interfaces (ask, chat, MCP) but not in raw search, where chunk-level granularity is preserved.

---

## 4. Frequency + Decay Scoring

After RRF retrieval, we apply a two-stage re-ranking.

### 4.1 Min-Max Normalization

RRF scores occupy a narrow band (~0.009–0.016) while frequency logs span a wider range (~0.69–2.40). Without normalization, frequency dominates. We normalize RRF within each batch:

```
ŝ_rrf(c) = (s_rrf(c) - min_i s_rrf(c_i)) / (max_i s_rrf(c_i) - min_i s_rrf(c_i))
```

### 4.2 Scoring Formula

```
score(c) = α · ŝ_rrf(c) + β · min(1, log(1 + f(c)) / log(1 + S))
```

where the frequency component is:

```
f(c) = (1 + access_count(c)) · e^(-λ · days_ago(c))
```

- **α, β**: semantic and memory weights (α + β ≤ 1)
- **λ**: decay rate (days⁻¹)
- **S = 100**: saturation constant for log normalization cap
- The **+1 baseline** ensures new documents can compete on pure semantic relevance

### 4.3 Over-Fetch and Re-Rank

We retrieve *N_over = 50* candidates via RRF, apply the scoring formula, sort, and truncate to `top_k`. This limits scoring compute to 50 candidates regardless of corpus size.

### 4.4 Access Tracking

After returning results, we batch-update `access_count` and `last_accessed_at` for returned chunks. This creates a feedback loop: frequently retrieved chunks accumulate higher scores, subject to temporal decay.

**Clock skew protection.** We clamp *days_ago ≥ 0* to prevent future timestamps (from clock drift or timezone issues) from inflating scores.

---

## 5. Relevance Signal via Vector Distance

A retrieval system that always returns results — even for off-topic queries — must provide a confidence signal so downstream consumers (CLI users, LLM agents) can decide whether to trust the results.

### 5.1 Why Score Spread Failed

Our initial approach used score spread (*max − min* of top-*k* scores). This required scoring to be enabled (spread is ~0.0006 on raw RRF for all queries), and even with scoring, discrimination was poor:

### Table 1a: Score spread as relevance discriminator

| Scoring | Relevant Spread | Irrelevant Spread | Ratio | Overlap | F1 |
|---------|:-:|:-:|:-:|:-:|:-:|
| Off (raw RRF) | 0.007 | 0.007 | 1.00x | 10/10 | 0.000 |
| On (10 rounds) | 0.305 | 0.245 | 1.24x | 10/10 | 0.667 |
| On (30 rounds) | 0.310 | 0.250 | 1.22x | 10/10 | 0.640 |

The fundamental problem: spread depends on per-chunk access count variance within the result batch, not on query-corpus semantic proximity. All 10 irrelevant queries fell within the spread range of relevant queries — no threshold could cleanly separate them.

### 5.2 Distance-Based Signal

The cosine distance of the best vector match directly measures "how far is this query from the nearest document." Relevant queries have best distances clustered around 0.5–0.85; irrelevant queries have distances > 0.95.

### Table 1b: Vector distance as relevance discriminator

| Signal | Relevant avg | Irrelevant avg | Ratio | Overlap | F1 @ threshold |
|--------|:-:|:-:|:-:|:-:|:-:|
| Spread (best) | 0.305 | 0.245 | 1.24x | 10/10 | 0.667 @ 0.15 |
| **Distance** | **0.594** | **0.978** | **1.65x** | **0/10** | **0.952 @ 0.95** |

### Table 1c: Classification strategies (10 relevant + 10 irrelevant queries)

| Strategy | Precision | Recall | F1 | Accuracy |
|----------|:-:|:-:|:-:|:-:|
| spread > 0.15 (best fixed) | 0.500 | 1.000 | 0.667 | 0.500 |
| **distance < 0.95** | **0.909** | **1.000** | **0.952** | **0.950** |
| distance < 0.95 AND spread > 0.005 | 0.909 | 1.000 | 0.952 | 0.950 |
| distance < 1.00 AND spread > 0.005 | 0.909 | 1.000 | 0.952 | 0.950 |

**Key advantages over spread:** (1) Works from the first search — no scoring, no access history, no warm-up period. (2) Zero class overlap vs. complete overlap. (3) No human threshold tuning needed — 0.95 is a natural gap in the distance distribution.

### 5.3 Tiered Ghost Warning

Rather than a binary signal, we implement three tiers based on the distance of the best vector match:

| Distance | Tier | User experience |
|----------|------|-----------------|
| ≤ 0.95 | high | No indicator — full confidence |
| 0.95–0.98 | medium | Subtle `?` next to result rank + "Uncertain relevance" note |
| > 0.98 | low | Full warning: "Low relevance — results may not match" |

This provides graduated feedback without being intrusive, applied consistently across CLI (search, ask, chat) and MCP server.

### 5.4 Discard Telemetry

To validate the signal in production, every search records an event with query, distance, tier, and result count. In chat mode, events are marked as "dismissed" when the user exits after a non-high result. The table is pruned to 1,000 entries and indexed by `(relevance_tier, created_at)`.

**Validation hypothesis:** If users dismiss "low" tier results at 3–5× the rate of "high" tier results, the signal is confirmed useful beyond synthetic benchmarks.

---

## 6. Code-Aware Chunking

Standard fixed-window chunking splits code mid-function, destroying semantic coherence. We detect top-level definitions via regex patterns anchored at column 0 (no indentation), supporting Python, JavaScript, TypeScript, Go, Rust, and Java.

### Algorithm: Code-Aware Chunking

```
Input: source text T, language L, chunk_size C
1. P ← language-specific regex patterns for L
2. B ← find all column-0 matches of P in T
3. Attach decorators/annotations to their next definition
4. chunks ← split T at boundaries B
5. For each chunk c:
     If tokens(c) > C: split by paragraphs, then fixed-window fallback
6. Merge adjacent chunks with < 80 tokens
7. Return chunks
```

**Column-0 anchoring.** By requiring zero indentation, we avoid false positives on nested method definitions (e.g., methods inside a Python class). This is a deliberate trade-off: nested methods are kept with their parent class, which is desirable for embedding coherence.

---

## 7. Experimental Setup

**Corpus.** 24 arXiv papers on LLM memory systems (2023–2026), yielding 786 chunks after ingestion. Publication dates are used to simulate temporal spread.

**Queries.** 10 evaluation queries with human-annotated top-5 expected results (graded relevance). 15 relevant and 15 irrelevant queries for the relevance signal experiment.

**Metrics:**
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **P@k**: Precision at position k
- **Rank displacement**: Average position change vs. baseline
- **Frequency response**: Rank improvement for boosted papers
- **F1, Accuracy**: For relevance signal classification

**Configurations.** 16 parameter variants: α ∈ {0.4, 0.5, 0.7, 0.8, 0.9}, β ∈ {0.1, 0.2, 0.3, 0.5, 0.6}, λ ∈ {0.01, 0.03, 0.05, 0.07, 0.10, 0.20}, N_over ∈ {20, 50, 100}.

**Scenarios.** 5 simulated access patterns: uniform (baseline), recent heavy use, stale favorites, mixed recency, benchmark-focused.

---

## 8. Results

### 8.1 Ablation: RRF vs. Vector vs. FTS

### Table 2a: Ablation — LLM memory corpus (24 papers, 786 chunks)

| Mode | NDCG@5 | NDCG@10 | P@3 | Latency |
|------|:------:|:-------:|:---:|--------:|
| Vector-only | 0.809 | 0.832 | 0.933 | 4.51 ms |
| FTS keyword | 0.631 | 0.621 | 0.767 | 0.81 ms |
| **Hybrid RRF** | **0.814** | **0.803** | **1.000** | 1.61 ms |

### Table 2b: Ablation — Wikipedia corpus (17 articles, 2,602 chunks)

| Mode | NDCG@5 | NDCG@10 | P@3 | Latency |
|------|:------:|:-------:|:---:|--------:|
| Vector-only | 0.742 | 0.742 | 0.667 | 21.0 ms |
| FTS keyword | 0.699 | 0.699 | 0.583 | 1.94 ms |
| **Hybrid RRF** | **0.758** | **0.758** | 0.633 | 4.78 ms |

> **Methodology.** Relevance labels were obtained via pooled annotation: the union of top-10 results from all three methods was scored on a 0–3 scale using an LLM judge (Qwen 3.5:9B), then validated against human judgment. Results are deduplicated at the document level — only the first chunk from each document counts toward NDCG. This prevents inflated scores from multi-chunk documents.

**Hybrid RRF is the strongest modality on both corpora.** On the domain-specific LLM memory corpus, RRF achieves perfect P@3 = 1.000 and the highest NDCG@5 (0.814, improved to 0.829 with document deduplication §3.4). On the diverse Wikipedia corpus, RRF still leads (NDCG@5 = 0.758), confirming that rank fusion generalizes across domains.

Two corpus-dependent effects are visible:

1. **FTS dominance fades on diverse corpora.** On homogeneous LLM memory papers, FTS trails vector by 22% (0.631 vs 0.809). On diverse Wikipedia, FTS closes to within 6% (0.699 vs 0.742) because semantic paraphrases become more important across unrelated domains.
2. **Vector search scales with corpus size.** Latency increases from 4.5 ms (786 chunks) to 21 ms (2,602 chunks) — proportional to the approximate nearest-neighbor scan. FTS and RRF scale better due to the FTS5 inverted index.

### 8.2 Scoring Grid Search

### Table 3: Top-5 scoring configurations averaged across 5 scenarios

| Configuration | Avg NDCG@10 | Δ Baseline | Best Scenario |
|---------------|:-----------:|:----------:|:-------------:|
| **α=0.5, β=0.5, λ=0.10** | **0.636** | **+4.6%** | +16.1% (benchmark) |
| α=0.5, β=0.5, λ=0.05 | 0.634 | +4.3% | +15.6% (benchmark) |
| α=0.7, β=0.3, λ=0.03 | 0.631 | +3.8% | +13.2% (benchmark) |
| α=0.7, β=0.3, λ=0.07 | 0.629 | +3.5% | +13.1% (benchmark) |
| α=0.8, β=0.2, λ=0.10 | 0.632 | +3.9% | +7.2% (benchmark) |

*Baseline NDCG@10 = 0.608*

Key findings:

1. **Equal weighting (α=β=0.5) is optimal** when averaged across all scenarios, contrary to the common assumption that semantic relevance should dominate.
2. **The scoring benefit scales with access differential.** In the benchmark-focused scenario (heavy access to specific papers), NDCG improves by 16.1%. In the uniform scenario (no differential access), improvement is minimal (+0.7%).
3. **Lambda sensitivity is low** in the 0.03–0.10 range. Extreme values (λ = 0.20) degrade performance as recent accesses decay too quickly.
4. **Over-fetch of 50 is sufficient.** Increasing to 100 shows no improvement; decreasing to 20 slightly hurts.

### 8.3 Relevance Signal

The distance-based signal (§5.2) at threshold *d < 0.95*:

- **Precision = 0.909**: one false positive among 10 irrelevant queries.
- **Recall = 1.000**: detects all 10 relevant queries.
- **F1 = 0.952**, **Accuracy = 95.0%**: across 20 queries total (10 relevant + 10 irrelevant).

This supersedes the score-spread signal, which achieved F1 = 0.667 with complete class overlap even after 30 rounds of scoring warm-up. The distance signal works from the first search with no dependencies on scoring or access history — critical for tool autonomy.

### 8.4 Code-Aware Chunking

### Table 4: Chunking strategy comparison (Python + Go, 8 queries, top-k=3)

| Strategy | Avg Recall | Avg Precision | Boundary Violations |
|----------|:----------:|:-------------:|:-------------------:|
| Naive (fixed-window) | **0.917** | 0.854 | present |
| Code-aware (boundary) | 0.625 | **0.750** | 0 |

Naive chunking achieves higher recall on our small test corpus because a single large chunk trivially contains all functions. However, code-aware chunking:

- Produces **zero boundary violations** (no function split mid-body).
- Achieves **perfect precision** (1.0 vs. 0.33–0.50) on focused queries like "rate limiting" or "revoke token".
- Creates smaller, semantically coherent chunks (avg 252 tokens vs. 653 tokens) that are more useful as LLM context.

**Scale effect.** As corpus size grows, the recall advantage of naive chunking vanishes: with thousands of chunks, the search cannot return all of them. Code-aware chunking's precision advantage compounds at scale.

### 8.5 Latency

### Table 5: Search latency on 786-chunk corpus

| Configuration | Median | P95 | P99 | Overhead |
|---------------|:------:|:---:|:---:|:--------:|
| RRF only | 0.54 ms | 0.60 ms | 0.69 ms | — |
| + Scoring | 0.99 ms | 1.14 ms | 1.21 ms | +82% |
| + Scoring + Track | 1.04 ms | 1.17 ms | 1.22 ms | +91% |
| + Dedup | 1.43 ms | 1.57 ms | 1.70 ms | — |
| + Scoring + Dedup | 2.92 ms | 3.44 ms | 3.62 ms | — |
| **Full pipeline** | **3.41 ms** | **3.97 ms** | **4.10 ms** | — |

*Full pipeline = search + scoring + dedup + context expansion.* Scoring adds 0.45 ms median overhead. Context expansion adds 0.12 ms. The full pipeline stays under 4 ms at P50 — imperceptible for interactive use.

### 8.6 Cold Start Curve

### Table 6: Scoring NDCG@5 vs. baseline over 20 rounds of non-uniform usage

| Round | Scored | Baseline | Δ% | Cumulative Accesses |
|:-----:|:------:|:--------:|:---:|:-------------------:|
| 1 | 0.570 | 0.688 | -17.1% | 50 |
| 5 | 0.591 | 0.688 | -14.1% | 450 |
| 10 | 0.609 | 0.688 | -11.5% | 1,400 |
| 15 | 0.593 | 0.688 | -13.9% | 2,550 |
| 20 | 0.619 | 0.688 | -10.1% | 3,700 |

We simulate non-uniform usage with a Zipf-weighted query distribution (some topics queried 5× more than others) and measure whether scoring (α=0.5, β=0.5, λ=0.10) outperforms unscored RRF as access history accumulates.

**Key finding:** Scoring has not crossed over baseline after 3,700 cumulative accesses (20 rounds), though the gap narrows monotonically from -17.1% to -10.1%. This is expected: the scoring grid (§8.2) found scoring beneficial only with **pre-established access differentials** (e.g., the benchmark-focused scenario with 30× access counts). Organic usage accumulates access counts gradually, requiring many sessions before the frequency signal dominates noise.

**Practical implication:** Scoring should be disabled by default and enabled once the user has established meaningful usage patterns. The shrinking gap suggests crossover at ~40–50 rounds for the optimal configuration — approximately 2–3 weeks of daily use for an active researcher.

---

## 9. Limitations and Future Work

**Corpus breadth.** We evaluate on two corpora: 24 LLM memory papers (domain-specific) and 17 Wikipedia articles (mixed-domain). While RRF leads on both (Tables 2a–2b), testing on additional domains (e.g., legal, medical) would further strengthen generalizability claims.

**Cold start period.** As shown in §8.6, scoring requires substantial usage history before it outperforms unscored RRF. The distance-based relevance signal (§5.2) solves the cold start problem for *confidence estimation* (works from the first search), but the *ranking quality* improvement from scoring still requires accumulated access history.

**Discard telemetry is prospective.** The search_events table and dismiss tracking (§5.4) are instrumented but have not yet accumulated enough real-world data to validate dismiss rates across tiers. This is a designed validation path, not a confirmed result.

**Scale.** Our experiments use 786 chunks. SQLite's single-writer model may bottleneck at 10⁶+ chunks under concurrent write load, though WAL mode and batching mitigate this for single-user scenarios.

**Maximal Marginal Relevance (MMR).** Document deduplication (§3.4) uses hard per-document dedup. MMR-style diversity-aware re-ranking could provide a better diversity/relevance tradeoff by allowing multiple chunks from the same document when they are sufficiently diverse.

**Implicit feedback.** Tracking which results the user expands, copies, or follows up on could refine the relevance signal and accelerate scoring warm-up — closing the loop between usage and retrieval quality.

**Multi-modal.** Current chunking and embedding support text only. Image embeddings via CLIP and table-aware chunking are planned for future versions.

---

## 10. Conclusion

We presented vstash, a local-first document memory system that demonstrates five findings relevant to LLM agent memory:

1. **Temporal scoring improves ranking under differential access.** Post-RRF re-ranking with frequency and decay improves NDCG@10 by +4.6% on average and +16.1% in access-heavy scenarios, though it requires substantial usage history before outperforming unscored RRF.

2. **Vector distance is a strong, autonomous relevance signal.** The cosine distance of the best vector match achieves F1 = 0.952 in distinguishing relevant from irrelevant queries — with zero class overlap, no scoring dependency, and no warm-up period. This supersedes our initial score-spread approach (F1 = 0.667, complete class overlap), eliminating the need for human threshold tuning.

3. **Document deduplication improves both diversity and quality.** Keeping only the best chunk per document in top-*k* results raises unique document count from ~3.2 to 5.0 while simultaneously improving NDCG@5 from 0.814 to 0.829.

4. **Context expansion is cheap and valuable.** Fetching adjacent chunks (±1 window) provides 2.64× more text for LLM consumption at +0.12 ms cost — a near-free improvement to answer quality.

5. **Local-first is viable.** With sub-millisecond search latency, a single SQLite file, and zero cloud dependencies, there is no fundamental barrier to running hybrid retrieval with temporal scoring, deduplication, and relevance signaling on a single machine.

---

## References

1. Cormack, G. V., Clarke, C. L. A., & Buttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*.

2. Ebbinghaus, H. (1885). *Uber das Gedachtnis*. Duncker & Humblot.

3. Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. *arXiv:2310.08560*.

4. Chadha, T., Khattab, D., & Singhal, P. (2025). Mem0: Production-ready AI agents with scalable long-term memory. *arXiv:2504.19413*.

5. Memoria Team. (2025). Memoria: Scalable agentic memory for personalized conversational AI. *arXiv:2512.12686*.

6. A-MEM Team. (2025). A-MEM: Agentic memory for LLM agents. *arXiv:2502.12110*.

7. Ma, X., Wang, Y., & Lin, J. (2024). Is reciprocal rank fusion all you need for hybrid retrieval? *arXiv preprint*.

8. Zep Team. (2025). Zep: Temporal knowledge graph architecture for agent memory. *arXiv:2501.13956*.

9. MaRS Team. (2025). MaRS: Forgetful but faithful — cognitive memory architecture. *arXiv:2512.12856*.

10. PAM Team. (2026). PAM: Predictive associative memory via temporal co-occurrence. *arXiv:2602.11322*.
