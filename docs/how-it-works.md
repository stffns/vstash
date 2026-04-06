# How It Works

vstash combines vector similarity, keyword matching, and adaptive ranking into a single fast pipeline. Everything runs locally except the optional LLM call.

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
  → adaptive RRF fusion    (IDF-weighted merge of both rankings)
  → recency boost          (optional: temporal decay favoring recent chunks)
  → MMR dedup              (intra-document diversity via Maximal Marginal Relevance)
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

With `k=60` and adaptive weights based on query characteristics:

- **Adaptive RRF** (default): Weights adjust per-query using mean IDF of query terms. Rare/technical terms boost FTS weight; common words boost vector weight. Long queries (>50 words) also relax the distance cutoff to handle diffuse embeddings.
- **Fixed weights**: `vec_weight=0.6`, `fts_weight=0.4` when `adaptive_rrf=False` or explicit weights are provided.

This ensures:
- Semantic queries find conceptually related chunks (even without exact keywords)
- Technical queries with rare terms benefit from exact keyword matching
- A chunk ranked high in both lists gets a strong combined score

### Recency Boost (optional)

*Added in v0.19.0*

After RRF, an optional recency multiplier biases scores toward recently created content:

```
boosted_score = rrf_score × (1 + recency_boost × e^(−0.05 × days_ago))
```

When `recency_boost=0.0` (default), this step is skipped entirely — pure RRF results are returned. When enabled, chunks created today get the full boost while chunks older than ~3 months are effectively unaffected.

Designed for agentic memory where recent context matters more than old context. See [Recency Boost & Temporal Filters](scoring.md) for usage, parameters, and examples.

**Temporal filters** (`added_after`/`added_before`) provide hard date boundaries at the SQL level, complementing the soft recency boost.

### Intra-Document MMR Deduplication

*Hard dedup added in v0.6.0 · Replaced by MMR in v0.8.0*

After scoring, multiple chunks from the same document often cluster in the top-*k*. vstash applies **Maximal Marginal Relevance (MMR)** within documents to balance relevance and diversity:

```
MMR(c) = λ · score(c) − (1 − λ) · max_sim(c, selected_same_doc)
```

Chunks from *different* documents compete purely on score. When multiple chunks from the *same* document are candidates, the second is penalized by its cosine similarity to the first. If two sections are semantically diverse, both appear; if they're near-duplicates, the second is suppressed.

Configure with `mmr_lambda` (default 0.5): 0.0 = maximum diversity, 1.0 = one chunk per document (old hard dedup behavior). See [Recency Boost & Temporal Filters](scoring.md#intra-document-mmr-deduplication) for the full explanation.

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
