# How It Works

vstash combines three retrieval strategies — vector similarity, keyword matching, and memory scoring — into a single fast pipeline. Everything runs locally except the optional LLM call.

---

## Ingestion Pipeline

When you add a file, vstash processes it through these stages:

```
file / URL
  → parse            (markitdown: PDF, DOCX, PPTX, XLSX, HTML, etc.)
  → chunk            (split into meaningful pieces)
  → embed            (FastEmbed ONNX: ~700 chunks/s)
  → store vectors    (sqlite-vec: cosine similarity index)
  → index text       (FTS5: keyword search with porter stemming)
```

### Parsing

Non-code files are parsed by [markitdown](https://github.com/microsoft/markitdown), which converts PDF, DOCX, PPTX, XLSX, HTML, and other formats to plain text while preserving structure.

Code files (when `code_aware = true`) skip markitdown entirely — they're read as raw UTF-8 to preserve indentation, syntax, and structure.

### Chunking

vstash uses two chunking strategies depending on the file type:

**Semantic chunking** (Markdown, PDF, DOCX, etc.):

```
text
  → split by Markdown headers    (# sections become chunk boundaries)
  → split by paragraphs          (double newlines within sections)
  → fixed-window fallback        (oversized paragraphs get windowed)
  → merge small chunks           (tiny fragments < 80 tokens get merged)
```

Headers stay with their body content. Paragraphs aren't torn mid-sentence. Tiny fragments are merged to avoid low-quality embeddings.

**Code-aware chunking** (25+ languages):

```
source code
  → detect language from extension        (.py → python, .go → go, etc.)
  → try tree-sitter AST splitting         (25+ languages, optional dependency)
  → try parso AST splitting               (Python only, base dependency)
  → fall back to regex splitting          (Python, JS/TS, Go, Rust, Java)
  → attach decorators to their function   (Python @decorator, Java @Annotation)
  → fallback for oversized functions      (paragraph → fixed-window)
  → merge small chunks                    (imports, constants get merged)
```

The splitting backend is selected automatically with graceful degradation:

| Backend | Languages | Resolution | Install |
|---------|-----------|------------|---------|
| **tree-sitter** | 25+ (C, C++, Ruby, PHP, Swift, Kotlin, Scala, etc.) | AST-level — exact definition boundaries | `pip install vstash[treesitter]` |
| **parso** | Python only | AST-level — funcdef, classdef, decorated | Included by default |
| **regex** | Python, JS/TS, Go, Rust, Java | Pattern-based — column-0 definitions | Included by default |

Each chunk starts at a top-level definition (`def`, `class`, `func`, `fn`, etc.). Indented methods stay inside their class. Decorators and annotations stay attached to their function. Preambles (imports, package declarations) are preserved as a separate chunk.

Disable code-aware chunking with `code_aware = false` in `[chunking]` — files will fall back to markitdown + semantic chunking.

---

## Search Pipeline

```
query
  → embed query            (FastEmbed ONNX)
  → vector search          (sqlite-vec: top-k × 10 candidates by cosine similarity)
  → keyword search         (FTS5: top-k × 10 candidates by BM25)
  → RRF fusion             (merge both rankings)
  → memory scoring         (frequency + temporal decay re-ranking)
  → document dedup         (one result per document path)
  → relevance signal       (distance-based confidence: high/medium/low)
  → context expansion      (±1 adjacent chunks for LLM context)
  → top-k results          (default: 5 chunks)
  → LLM                    (optional: generate answer from retrieved chunks)
```

### Reciprocal Rank Fusion (RRF)

RRF merges the vector and keyword result lists without needing comparable scores:

```
rrf_score = vec_weight / (k + vec_rank) + fts_weight / (k + fts_rank)
```

With `k=60`, `vec_weight=0.6`, `fts_weight=0.4`. This ensures:

- Semantic queries find conceptually related chunks (even without exact keywords)
- Exact keyword queries are never missed
- A chunk ranked high in both lists gets a strong combined score

### Memory Scoring

After RRF, an optional second pass re-ranks results using access frequency and recency:

```
final_score = α · normalized_rrf + (β · γ) · log(1 + access_count · e^(−λ · days_ago))
```

The **adaptive maturity gate (γ)** scales the frequency component based on how differentiated access patterns are. When usage is uniform or sparse (max/mean ratio < 8×), γ = 0 and scoring is completely suppressed — pure RRF results are returned with zero overhead. As access patterns develop clear favorites (ratio ≥ 15×), γ ramps to 1.0 and full scoring applies. This eliminates cold start degradation: fixed β=0.5 degrades NDCG by -8.6% from day one, while adaptive γ maintains 0.0% degradation.

Chunks you search for often and recently get a boost. See [Memory Scoring](scoring.md) for the full explanation, parameters, and tuning guide.

### Document Deduplication

*Added in v0.6.0*

After scoring, multiple chunks from the same document often cluster in the top-*k*. vstash deduplicates by keeping only the highest-scoring chunk per document path before truncating to `top_k`. This improves result diversity from ~3.2 to 5.0 unique documents per top-5 while also improving NDCG@5 by +1.8%.

### Distance-Based Relevance Signal

*Added in v0.6.0*

Every query returns results — even off-topic ones. vstash uses the cosine distance of the best vector match to estimate confidence:

| Distance | Tier | Behavior |
|----------|------|----------|
| ≤ 0.95 | **high** | Full confidence — no indicator |
| 0.95–0.98 | **medium** | Subtle `?` next to rank + "Uncertain relevance" note |
| > 0.98 | **low** | Full warning: "Low relevance — results may not match" |

This achieves F1 = 0.952 in distinguishing relevant from irrelevant queries — with zero class overlap and no warm-up period. It works from the very first search.

### Context Expansion

*Added in v0.6.0*

A single chunk (~250 tokens) is often too small for good LLM answers. In LLM-facing interfaces (ask, chat, MCP), vstash automatically fetches adjacent chunks (±1 by sequence number within the same document) and concatenates their text. This provides 2.64× more context at only +0.12 ms overhead.

Raw `vstash search` preserves chunk-level granularity — expansion only applies when chunks are sent to an LLM.

---

## Storage

Everything lives in a single SQLite database (default: `~/.vstash/memory.db`):

- **sqlite-vec** — vector index for approximate nearest neighbor search
- **FTS5** — full-text search index with porter stemming
- **Metadata** — document info, chunk text, access counts, timestamps
- **search_events** — telemetry table recording query, distance, relevance tier, and dismiss flag (pruned to 1,000 entries)

No external services. No running processes. Just a file.
