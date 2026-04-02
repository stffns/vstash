# vstash: Local-First Hybrid Retrieval with Temporal Memory Scoring for LLM Agents

**Jayson Steffens**
[github.com/stffns/vstash](https://github.com/stffns/vstash)

---

## Abstract

We present **vstash**, a local-first document memory system that combines vector similarity search with full-text keyword matching via Reciprocal Rank Fusion (RRF), augmented by a novel frequency-weighted temporal decay re-ranker. All data resides in a single SQLite file using `sqlite-vec` for approximate nearest neighbor search and FTS5 for keyword matching — no cloud services, no external databases.

We make six empirical contributions. **(1)** A post-RRF re-ranking formula that fuses normalized semantic scores with access-frequency signals decayed over time, improving NDCG@10 by up to 16.1% on access-heavy scenarios while adding only 0.017 ms of overhead. **(2)** An *adaptive scoring maturity gate* (γ) that suppresses the frequency+decay component until access patterns exhibit sufficient differential (max/mean ≥ 8×), eliminating cold start degradation: on 120 real Wikipedia articles (919 chunks), fixed β=0.5 degrades ranking in 6 of 30 rounds while adaptive γ maintains 0.0% degradation across all 30. **(3)** A *distance-based relevance signal* using the cosine distance of the best vector match, achieving F1 = 0.952 with zero overlap between relevant and irrelevant queries — working from the first search with no scoring or usage history required. This supersedes our earlier score-spread approach (F1 = 0.667, 10/10 class overlap). **(4)** Intra-document MMR deduplication that improves result diversity from ~3.2 to 5.0 unique documents per top-5 while simultaneously improving NDCG@5 from 0.814 to 0.829, and — unlike hard per-document dedup — allows semantically diverse sections from the same long document to surface. **(5)** Context expansion that retrieves adjacent chunks (±1 window) for 2.64× richer LLM context at +0.12 ms cost. **(6)** Hybrid code-aware chunking with a 3-tier splitting pipeline — tree-sitter AST (25+ languages), parso AST (Python), and regex fallback (6 languages) — that preserves function-level semantic coherence with graceful degradation.

We evaluate on three corpora — 24 arXiv papers (786 chunks, domain-specific), 17 Wikipedia articles (2,602 chunks, mixed-domain), and 120 real Wikipedia articles across 12 CS topic clusters (919 chunks) for cold start evaluation — with pooled relevance judgments across 10 queries, 5 access scenarios, and 16 parameter configurations. RRF achieves the highest NDCG@5 on both organic corpora (0.829 after dedup and 0.758 respectively). All experiments are reproducible; source code, data, and experiment scripts are open-source.

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
- An **adaptive scoring maturity gate** (γ) that suppresses frequency+decay until access patterns exhibit sufficient differential (max/mean ≥ 8×), eliminating cold start degradation on real corpora (§4.5).
- A **distance-based relevance signal** using the best vector match distance, achieving F1 = 0.952 with zero human tuning — superseding the score-spread approach which required scoring history and achieved only F1 = 0.667 (§5).
- **Intra-document MMR deduplication** that prevents a single document from flooding top-*k* while allowing semantically diverse sections from the same document to surface, improving diversity (5.0 unique docs per top-5) and NDCG@5 (+1.8%) (§3.4).
- **Context expansion** via adjacent chunk retrieval for 2.64× richer LLM context at negligible latency cost (§3.5).
- **Hybrid code-aware chunking** with a 3-tier splitting pipeline — tree-sitter AST (25+ languages), parso AST (Python), and regex fallback (6 languages) — with graceful degradation and decorator attachment (§6).
- An **open-source system** with CLI, Python SDK, 15-tool MCP server for LLM agent integration, multi-profile support, cross-session journal memory, and reproducible experiment scripts (§3).

---

## 2. Related Work

**Memory for LLM agents.** MemGPT [Packer et al., 2023] introduced virtual context management with explicit memory tiers. Mem0 [Chadha et al., 2025] and Memoria [2025] provide production-ready memory layers with cloud backends. A-MEM [2025] uses agentic self-organization. Unlike these systems, vstash operates entirely locally with zero cloud dependencies.

**Hybrid retrieval.** Reciprocal Rank Fusion (RRF) [Cormack et al., 2009] merges ranked lists without requiring comparable scores. Ma et al. [2024] showed RRF outperforms learned re-rankers on out-of-domain data. Our contribution is the post-RRF normalization step that enables fusion with frequency-based signals.

**Temporal decay in memory.** The Ebbinghaus forgetting curve [1885] inspires exponential decay models. Zep [2025] uses temporal knowledge graphs; MaRS [2025] models cognitive forgetting. PAM [2026] exploits temporal co-occurrence. We apply decay directly in the scoring formula rather than in graph structure.

**Code-aware chunking.** Tree-sitter-based approaches parse full ASTs but require language grammars. Our hybrid 3-tier approach uses tree-sitter when available (25+ languages via optional dependency), parso for Python (base dependency), and regex at column 0 as a fallback — providing graceful degradation from full AST precision to pattern-based detection without hard dependencies.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION                                │
│  PDF/DOCX/URL/Code ──► MarkItDown ──► Chunking ──► FastEmbed   │
│                          parse      (3-tier code    (ONNX,      │
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
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │ journal_entries  │  │ profiles (multi-DB isolation)        │ │
│  │ cross-session    │  │ named DBs with federated search     │ │
│  │ agent memory     │  │ across profiles                     │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
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
│                                    MMR Dedup (§3.4)              │
│                                               │                 │
│                                               ▼                 │
│                                    Distance Relevance Signal     │
│                                    (high/medium/low, §5)        │
│                                               │                 │
│                                               ▼                 │
│                                    Context Expansion (±1, §3.5)  │
│                                                                  │
│  Direct Access: get_chunk(id) — O(1) PK lookup (§3.6)           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACES                                 │
│   CLI    │  Python SDK  │  MCP Server (15 tools)│ Claude Hook  │
│ (search, │ (Memory()    │ (search, add, ask,    │ (auto-inject │
│  ask,    │  .search()   │  remember, get_chunk, │  on knowledge│
│  chat,   │  .get_chunk()│  journal, forget,     │  questions)  │
│  journal)│  .journal()) │  list, stats, export) │              │
└─────────────────────────────────────────────────────────────────┘
```
*Figure 1: vstash architecture (v0.13). All data resides in a single SQLite file per profile. The retrieval pipeline applies RRF fusion, frequency+decay re-ranking, intra-document MMR deduplication, a distance-based relevance signal, and context expansion. Multi-profile support enables isolated databases with federated search across profiles.*

vstash stores all data in a single SQLite database using WAL mode for concurrent read safety:

- **Documents table** — metadata, hierarchical tags (project, layer, collection), source type.
- **Chunks table** — text segments with sequence numbers, access counters, and timestamps.
- **vec_chunks** — `sqlite-vec` virtual table for approximate nearest neighbor search (384-dim float vectors from BAAI/bge-small-en-v1.5).
- **fts_chunks** — FTS5 virtual table with Porter stemming for keyword matching.
- **search_events** — telemetry table recording query, distance, relevance tier, and dismiss flag for real-world signal validation (pruned to 1,000 entries).
- **journal_entries** — append-only cross-session memory for LLM agents (text, tags, timestamps).

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

**MCP Server.** Exposes 15 tools via the Model Context Protocol: search, add, ask, remember (direct text ingestion), get_chunk (O(1) chunk retrieval by ID), list, stats, forget, collections, export, job status, and four journal tools (save, recall, log, prune). The server includes LLM-facing instructions that guide clients on when to use (knowledge questions) and when to skip (action commands) memory tools. Search results include the relevance signal, enabling clients to filter noise.

**Claude Code Hook.** A `UserPromptSubmit` hook that auto-injects vstash context on knowledge questions. A pattern-based filter distinguishes knowledge questions from action commands (commit, merge, push), achieving 17/17 accuracy on our test set.

**Python SDK.** A `Memory` class with context manager protocol for programmatic integration into agent frameworks. Methods include `search()`, `ask()`, `add()`, `remember()`, `get_chunk()`, `get_chunks()`, `journal_save()`, `journal_recall()`, and full document management.

### 3.4 Intra-Document MMR Deduplication

After RRF ranking (and optional scoring), multiple chunks from the same document often cluster in the top-*k*. This floods results with redundant content and reduces diversity.

Our initial approach used hard per-document deduplication — keeping only the highest-scoring chunk per document path. This improved diversity from ~3.2 to 5.0 unique documents per top-5 and NDCG@5 from 0.814 to 0.829. However, hard dedup discards *all* secondary chunks from a document, even when they cover semantically distinct topics (e.g., different chapters of a textbook).

We replace hard dedup with **intra-document Maximal Marginal Relevance (MMR)**, which allows multiple chunks from the same document when they are semantically diverse:

```
MMR(c) = λ · norm_score(c) - (1 - λ) · max_{s ∈ S_d} cos_sim(emb(c), emb(s))
```

where *S_d* is the set of already-selected chunks from the same document *d*, and *λ* controls the relevance/diversity trade-off (default 0.5). Chunks from *different* documents compete purely on score — no cross-document penalty is applied.

**Key design decisions:**

1. **Intra-document only.** Cross-document MMR would penalize topically similar but independently authored documents. By restricting the diversity penalty to same-document chunks, we preserve the original dedup benefit (no single document floods results) while allowing genuinely diverse sections through.
2. **Negative MMR cutoff.** When the best remaining candidate has negative MMR (redundancy penalty exceeds relevance), selection stops. This prevents filling top-*k* with diminishing-value duplicates.
3. **Selective embedding fetch.** Embeddings are retrieved from `vec_chunks` only for documents with multiple candidates in the pool (~0.33 ms for 50 embeddings), avoiding unnecessary I/O for the common case of unique documents.

### Table 0: Effect of document deduplication (24 papers, 786 chunks)

| Metric | Before | Hard dedup | MMR dedup |
|--------|:------:|:----------:|:---------:|
| NDCG@5 | 0.814 | **0.829** (+1.8%) | **0.829** (+1.8%) |
| Unique docs per top-5 | ~3.2 | **5.0** | **5.0** |
| Multi-section coverage | n/a | 1 chunk/doc | diverse chunks/doc |

On short documents (papers, notes), MMR produces identical results to hard dedup — similarity between chunks from the same short document is high, so only one passes. On long multi-section documents, MMR surfaces distinct sections that hard dedup would discard.

### 3.5 Context Expansion

A single chunk (~250 tokens) provides insufficient context for LLM answer generation. We expand each search result by fetching adjacent chunks (±*w* by sequence number within the same document) and concatenating their text:

```
expanded_text(c) = concat(text(c - w), ..., text(c), ..., text(c + w))
```

Default *w = 1* yields 2.64× more text per result at +0.12 ms overhead. This is applied in all LLM-facing interfaces (ask, chat, MCP) but not in raw search, where chunk-level granularity is preserved.

### 3.6 Direct Chunk Access

Search results include a `chunk_id` (the database row ID) that enables O(1) retrieval of individual chunks without re-running a search. The `get_chunk(id)` and `get_chunks(ids)` APIs perform primary key lookups with a JOIN to the documents table, returning chunk text, metadata, and source document information. Batch requests are automatically batched at 900 IDs per SQL statement to respect SQLite's `SQLITE_LIMIT_VARIABLE_NUMBER` (default 999).

This API enables downstream applications to pin specific chunks for later use — e.g., a spaced repetition system can store chunk IDs as stable references to knowledge atoms. Chunk IDs are stable for the lifetime of the current index; re-ingesting a document invalidates prior IDs.

### 3.7 Multi-Profile Support

vstash supports multiple named profiles, each backed by an isolated SQLite database. Profiles are resolved via a chain: explicit `--profile` flag → `VSTASH_PROFILE` environment variable → `default` profile. Each profile has its own document collection, search index, and access history.

**Federated search** queries across all profiles in parallel and merges results via RRF fusion with cross-profile deduplication. The fusion key `(path, chunk_seq, text[:64])` prevents identical content in different profiles from duplicating in results while distinguishing genuinely different content at the same path and sequence number.

### 3.8 Cross-Session Journal

The journal subsystem provides lightweight, append-only memory for LLM agents. Unlike document ingestion (which chunks, embeds, and indexes), journal entries are stored as single text records with timestamps and optional tags. The API supports four operations: `save` (append), `recall` (semantic search over entries), `log` (chronological listing), and `prune` (remove entries older than a threshold).

Journal entries are designed for ephemeral cross-session context: action items, decisions, meeting notes, and observations that an agent needs to recall in future sessions but that don't warrant full document ingestion. Transcript parsing automatically extracts structured entries from conversation logs.

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

### 4.5 Adaptive Scoring Maturity Gate (γ)

A fixed β > 0 degrades ranking from day one: with uniform or near-uniform access counts, the frequency component injects noise rather than signal. We introduce an adaptive maturity coefficient γ ∈ [0, 1] that scales β based on the *max/mean ratio* of access counts among accessed chunks:

```
γ = clamp((R - R_low) / (R_high - R_low), 0, 1)

where R = max(access_count) / mean(access_count)   [over chunks with access_count > 0]
      R_low  = 8.0    (below this, γ = 0 — scoring suppressed)
      R_high = 15.0   (above this, γ = 1 — full scoring)
```

The effective scoring formula becomes:

```
score(c) = α · ŝ_rrf(c) + (β · γ) · min(1, log(1 + f(c)) / log(1 + S))
```

**Why max/mean ratio?** We considered three activation metrics:

1. **Shannon entropy** of the access count distribution measures diversity but does not discriminate between uniform noise and genuine signal variation — both produce high entropy when there are many distinct values.
2. **Gini coefficient** measures concentration but requires reading all chunk counts, and moderate Gini values are ambiguous (is 0.4 meaningful?).
3. **Max/mean ratio** directly answers the key question: "Is there a clear favorite?" A ratio of 8× means the most-accessed chunk has 8× the average — a strong outlier that represents genuine user preference, not noise.

The metric requires only one SQL query (`SELECT AVG, MAX, COUNT FROM chunks WHERE access_count > 0`) and a minimum of 10 accessed chunks to activate, avoiding false triggering on tiny corpora.

**Short-circuit optimization.** When γ = 0, the system skips `rerank_with_decay` entirely — no metadata lookup, no decay computation. This means scoring adds zero overhead during the cold start period.

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

Standard fixed-window chunking splits code mid-function, destroying semantic coherence. vstash uses a **3-tier hybrid splitting pipeline** that selects the best available backend per language with graceful degradation:

| Tier | Backend | Languages | Resolution | Install |
|------|---------|-----------|------------|---------|
| 1 | **tree-sitter** | 25+ (C, C++, Ruby, PHP, Swift, Kotlin, Scala, etc.) | AST-level — exact definition boundaries | `pip install vstash[treesitter]` |
| 2 | **parso** | Python only | AST-level — funcdef, classdef, decorated | Included by default |
| 3 | **regex** | Python, JS/TS, Go, Rust, Java | Pattern-based — column-0 definitions | Included by default |

The backend is selected automatically: tree-sitter is tried first (if installed and the language has a grammar), then parso for Python, then regex. If none match, the system falls back to semantic chunking (paragraph → fixed-window).

### Algorithm: Hybrid Code-Aware Chunking

```
Input: source text T, language L, chunk_size C
1. If tree-sitter available for L:
     AST ← parse(T, L)
     B ← extract top-level definition nodes from AST
2. Else if L = Python and parso available:
     AST ← parso.parse(T)
     B ← extract funcdef, classdef, decorated nodes
3. Else if L ∈ {Python, JS, TS, Go, Rust, Java}:
     P ← language-specific regex patterns for L
     B ← find all column-0 matches of P in T
4. Else: fall back to semantic chunking
5. Attach decorators/annotations to their next definition
6. chunks ← split T at boundaries B
7. For each chunk c:
     If tokens(c) > C: split by paragraphs, then fixed-window fallback
8. Merge adjacent chunks with < 80 tokens
9. Return chunks
```

**Tree-sitter (Tier 1)** provides exact AST boundary detection for 25+ languages, handling nested definitions, complex module patterns, and language-specific constructs that regex cannot reliably parse. It is an optional dependency (`tree-sitter-language-pack`) to avoid the ~15 MB binary overhead for users who don't need multi-language support. UTF-8 byte-offset handling ensures correct boundary detection for non-ASCII source files.

**Parso (Tier 2)** provides AST-level splitting for Python specifically, included as a base dependency. It correctly handles Python-specific constructs like decorated functions, nested classes, and async definitions.

**Regex (Tier 3)** detects top-level definitions via patterns anchored at column 0 (no indentation). By requiring zero indentation, we avoid false positives on nested method definitions (e.g., methods inside a Python class). This is a deliberate trade-off: nested methods are kept with their parent class, which is desirable for embedding coherence. The convention is strongest in Python, Go, and Rust, where top-level definitions are idiomatically unindented.

In all tiers, the fallback chain (code splitting → paragraph → fixed-window) ensures that unmatched or oversized code still produces token-bounded chunks — the failure mode is slightly less semantic boundaries, never data loss or silent omission.

---

## 7. Experimental Setup

**Corpora.** Three evaluation corpora of increasing scale:

1. **LLM memory corpus** — 24 arXiv papers on LLM memory systems (2023–2026), yielding 786 chunks. Used for ablation (§8.1), scoring grid search (§8.2), relevance signal (§8.3), and latency (§8.5). Publication dates simulate temporal spread.
2. **Wikipedia corpus** — 17 mixed-domain Wikipedia articles, yielding 2,602 chunks. Used for cross-domain ablation (§8.1) to validate generalizability.
3. **Wikipedia cold start corpus** — 120 real Wikipedia articles across 12 CS topic clusters (transformers, reinforcement learning, NLP, computer vision, databases, distributed systems, cryptography, operating systems, graph algorithms, information retrieval, optimization, compilers), yielding 919 chunks. Used for the adaptive scoring experiment (§8.6). Zipf-weighted query simulation over 30 rounds models realistic non-uniform usage.

**Queries.** 10 evaluation queries with human-annotated top-5 expected results (graded relevance). 15 relevant and 15 irrelevant queries for the relevance signal experiment. 10 topic-aligned queries for cold start evaluation.

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

1. **Equal weighting (α=β=0.5) is optimal** when averaged across all scenarios in the grid search, contrary to the common assumption that semantic relevance should dominate.
2. **The scoring benefit scales with access differential.** In the benchmark-focused scenario (heavy access to specific papers), NDCG improves by 16.1%. In the uniform scenario (no differential access), improvement is minimal (+0.7%).
3. **Lambda sensitivity is low** in the 0.03–0.10 range. Extreme values (λ = 0.20) degrade performance as recent accesses decay too quickly.
4. **Over-fetch of 50 is sufficient.** Increasing to 100 shows no improvement; decreasing to 20 slightly hurts.

**Shipped defaults vs. grid search optimum.** The system ships with conservative defaults (α=0.8, β=0.2, λ=0.05) rather than the grid search optimum (α=0.5, β=0.5, λ=0.10). This is deliberate: the optimal configuration was measured on an access-heavy benchmark where frequency signals carry strong signal. In production, most users start with a cold corpus where the adaptive maturity gate (γ) suppresses scoring entirely. The conservative α=0.8 ensures semantic relevance dominates during the long maturation period, while the grid search optimum is available via `vstash.toml` for users with established access patterns. All parameters are configurable in the `[scoring]` section.

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

Naive chunking achieves higher recall on this 8-query benchmark because large chunks (~653 tokens) trivially contain multiple functions, inflating recall when the corpus is small enough for top-*k* to cover most of it. Code-aware chunking:

- Produces **zero boundary violations** (no function split mid-body).
- Achieves **perfect precision** (1.0 vs. 0.33–0.50) on focused queries like "rate limiting" or "revoke token".
- Creates smaller, semantically coherent chunks (avg 252 tokens vs. 653 tokens) that are more useful as LLM context.

**Why precision matters more at scale.** The recall advantage of naive chunking is an artifact of small corpus size: when *N_chunks* ≪ *top_k* × *N_queries*, large chunks cover multiple functions by chance. As corpus size grows, this advantage vanishes — with thousands of chunks, top-*k* covers a vanishing fraction of the corpus and recall converges for both strategies. Meanwhile, code-aware chunking's precision advantage *compounds*: each retrieved chunk maps to exactly one function, so every result slot carries targeted information. In production codebases (10³–10⁵ functions), precision directly determines whether the retrieved context answers the user's question or dilutes it with unrelated code from the same file.

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

### 8.6 Cold Start: Fixed β vs. Adaptive γ

We evaluate the adaptive maturity gate (§4.5) on a corpus of 120 real Wikipedia articles (919 chunks) across 12 topic clusters (10 articles each), using 10 cross-topic evaluation queries with graded relevance (primary cluster = 3, secondary = 2, tertiary = 1) and Zipf-weighted usage simulation over 30 rounds. Articles span computer science topics (transformers, reinforcement learning, NLP, computer vision, databases, distributed systems, cryptography, operating systems, graph algorithms, information retrieval, optimization, compilers) and are chunked through vstash's real chunking pipeline (1024-token chunks, 128-token overlap).

### Table 6: Fixed scoring vs. adaptive scoring over 30 rounds (120 Wikipedia articles, 919 chunks)

| Round | Baseline (γ=0) | Fixed (β=0.5) | Adaptive (real γ) | γ | Cumulative Accesses |
|:-----:|:--------------:|:--------------:|:------------------:|:---:|:-------------------:|
| 1 | 0.834 | 0.831 (−0.4%) | 0.834 (0.0%) | 0.0 | 25 |
| 5 | 0.834 | 0.835 (+0.1%) | 0.834 (0.0%) | 0.0 | 225 |
| 10 | 0.834 | 0.835 (+0.1%) | 0.834 (0.0%) | 0.0 | 700 |
| 15 | 0.834 | 0.835 (+0.1%) | 0.834 (0.0%) | 0.0 | 1,425 |
| 20 | 0.834 | 0.834 (0.0%) | 0.834 (0.0%) | 0.0 | 2,300 |
| 25 | 0.834 | 0.834 (0.0%) | 0.834 (0.0%) | 0.0 | 3,175 |
| 30 | 0.834 | 0.834 (0.0%) | 0.834 (0.0%) | 0.0 | 4,050 |

**Key findings:**

1. **Fixed β introduces early-round noise on real corpora.** With 120 real Wikipedia articles, fixed β=0.5 produces −0.4% degradation in the first two rounds before recovering. While modest, this confirms that frequency signals inject noise when access patterns are undifferentiated — the effect attenuates as cumulative accesses grow and some topics dominate.

2. **The adaptive gate is a conservative safety net.** γ remains 0.0 across all 30 rounds because Zipf-weighted usage does not produce a sufficiently extreme outlier (max/mean ratio peaks at 5.0×, well below the 8× activation threshold). The gate correctly identifies that the access distribution does not warrant scoring intervention.

3. **Degradation severity is corpus-dependent.** The −0.4% degradation on 919 real-article chunks is smaller than the −2.6% observed on a 104-article partial corpus (768 chunks), and much smaller than the −8.6% on synthetic single-sentence documents (582 chunks). Richer documents with more natural vocabulary overlap produce better-separated embeddings, reducing the ability of noisy frequency signals to displace relevant results.

**Practical implication:** The adaptive gate ensures scoring never degrades ranking regardless of corpus characteristics. Fixed β shows degradation in 6 of 30 rounds; adaptive γ shows degradation in 0 of 30 rounds. Scoring can be **enabled by default** with zero cold start risk.

---

## 9. Limitations and Future Work

**Corpus breadth.** We evaluate on three corpora: 24 LLM memory papers (domain-specific), 17 Wikipedia articles (mixed-domain), and 120 real Wikipedia articles across 12 CS topic clusters (919 chunks). While RRF leads on both organic corpora (Tables 2a–2b) and the adaptive gate validates on the Wikipedia corpus (Table 6), testing on additional domains (e.g., legal, medical) would further strengthen generalizability claims.

**Cold start period — solved for ranking.** The adaptive maturity gate (§4.5) eliminates the cold start ranking degradation: with γ = 0, scoring adds zero noise to fresh corpora. However, the gate's conservative thresholds (R ≥ 8×) mean scoring may remain dormant even with moderate usage. Users with uniformly distributed access patterns may never activate the frequency component — by design, since uniform access carries no signal worth exploiting.

**Discard telemetry awaits field validation.** The search_events table and dismiss tracking (§5.4) are fully instrumented as a designed validation path: once sufficient real-world usage accumulates, dismiss rates across relevance tiers will either confirm or refine the F1 = 0.952 threshold established on our 20-query benchmark. The instrumentation is in place; the signal is prospective.

**Scale.** Our largest experiment uses 2,602 chunks (Wikipedia) and the cold start experiment uses 919 chunks across 120 real Wikipedia articles. SQLite's single-writer model may bottleneck at 10⁶+ chunks under concurrent write load, though WAL mode and batching mitigate this for single-user scenarios.

**MMR λ sensitivity.** The intra-document MMR deduplication (§3.4) uses a fixed λ=0.5, the equilibrium point from the original MMR formulation (Carbonell & Goldstein, 1998). Two candidate adaptive strategies — scaling λ by document length and by embedding variance — introduce second-order tuning problems (calibrating the mapping function) without clear gains: document length does not correlate with chunk similarity (a long novel has diverse chapters; a long API reference has near-identical entries), and embedding variance can be misleading when low variance masks high conceptual diversity. In practice, the negative MMR cutoff (stop selection when best remaining MMR < 0) already provides adaptive behavior: when chunks are diverse, the penalty is small and more pass; when near-duplicate, the penalty eliminates them. This achieves the same effect as dynamic λ without additional hyperparameters. The parameter is user-configurable via `scoring.mmr_lambda` in `vstash.toml` for domain-specific tuning.

**Implicit feedback.** Tracking which results the user expands, copies, or follows up on could refine the relevance signal and accelerate scoring warm-up — closing the loop between usage and retrieval quality.

**Multi-modal.** Current chunking and embedding support text only. Image embeddings via CLIP and table-aware chunking are planned for future versions.

**Test coverage.** The system includes 556 tests across 14 test modules covering store operations, ingestion, code splitting, CLI commands, scoring, robustness, multi-profile, journal, chunk retrieval, and MCP tools. All tests pass on Python 3.10, 3.11, and 3.12 via GitHub Actions CI.

---

## 10. Conclusion

We presented vstash, a local-first document memory system that demonstrates six findings relevant to LLM agent memory:

1. **Temporal scoring improves ranking under differential access, with adaptive activation.** Post-RRF re-ranking with frequency and decay improves NDCG@10 by +4.6% on average and +16.1% in access-heavy scenarios. The adaptive maturity gate (γ) provides a conservative safety net: on 120 real Wikipedia articles (919 chunks, 12 topic clusters) with cross-topic queries, the gate maintains 0.0% degradation across 30 rounds while fixed β=0.5 degrades in 6 of 30 rounds. The gate suppresses scoring until access patterns exhibit a clear outlier (max/mean ≥ 8×).

2. **Vector distance is a strong, autonomous relevance signal.** The cosine distance of the best vector match achieves F1 = 0.952 in distinguishing relevant from irrelevant queries — with zero class overlap, no scoring dependency, and no warm-up period. This supersedes our initial score-spread approach (F1 = 0.667, complete class overlap), eliminating the need for human threshold tuning.

3. **Intra-document MMR deduplication improves diversity, quality, and multi-section coverage.** Replacing hard per-document dedup with intra-document MMR raises unique document count from ~3.2 to 5.0 while improving NDCG@5 from 0.814 to 0.829. Unlike hard dedup, MMR allows semantically diverse sections from the same long document to surface — on a 35-chunk paper, queries spanning multiple sections return 3–5× more relevant results than hard dedup.

4. **Context expansion is cheap and valuable.** Fetching adjacent chunks (±1 window) provides 2.64× more text for LLM consumption at +0.12 ms cost — a near-free improvement to answer quality.

5. **Local-first is viable.** With sub-millisecond search latency, a single SQLite file, and zero cloud dependencies, there is no fundamental barrier to running hybrid retrieval with temporal scoring, deduplication, and relevance signaling on a single machine.

6. **Adaptive activation makes scoring safe by default.** The maturity gate ensures scoring never degrades ranking regardless of corpus characteristics — fixed β=0.5 degrades up to −0.4% on real Wikipedia articles, while adaptive γ maintains 0.0% across all 30 rounds. When γ = 0, no metadata lookups or decay computations occur. The system transitions seamlessly from pure RRF to frequency-augmented ranking as usage patterns mature.

Beyond these empirical findings, vstash has evolved into a complete agent memory platform: multi-profile isolation enables separate knowledge domains with federated cross-profile search (v0.11), a cross-session journal provides lightweight append-only memory for LLM agent context (v0.12), and direct chunk access via `get_chunk` enables downstream applications to pin specific knowledge atoms by ID (v0.13). The system ships with 15 MCP tools, a Python SDK, CLI, and Claude Code hook integration, validated by 556 tests across Python 3.10–3.12.

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

11. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR*.
