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

**Code-aware chunking** (Python, JS/TS, Go, Rust, Java):

```
source code
  → split at function/class definitions   (regex at column 0)
  → attach decorators to their function   (Python @decorator, Java @Annotation)
  → fallback for oversized functions      (paragraph → fixed-window)
  → merge small chunks                    (imports, constants get merged)
```

Each chunk starts at a top-level definition (`def`, `class`, `func`, `fn`, etc.). Indented methods stay inside their class. Decorators and annotations stay attached to their function.

Supported languages: Python, JavaScript, TypeScript (including JSX/TSX), Go, Rust, Java.

Disable code-aware chunking with `code_aware = false` in `[chunking]` — files will fall back to markitdown + semantic chunking.

---

## Search Pipeline

```
query
  → embed query       (FastEmbed ONNX)
  → vector search     (sqlite-vec: top-k × 10 candidates by cosine similarity)
  → keyword search    (FTS5: top-k × 10 candidates by BM25)
  → RRF fusion        (merge both rankings)
  → memory scoring    (frequency + temporal decay re-ranking)
  → top-k results     (default: 5 chunks)
  → LLM              (optional: generate answer from retrieved chunks)
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
final_score = α · normalized_rrf + β · log(1 + access_count · e^(−λ · days_ago))
```

Chunks you search for often and recently get a boost. See [Memory Scoring](scoring.md) for the full explanation, parameters, and tuning guide.

---

## Storage

Everything lives in a single SQLite database (default: `~/.vstash/memory.db`):

- **sqlite-vec** — vector index for approximate nearest neighbor search
- **FTS5** — full-text search index with porter stemming
- **Metadata** — document info, chunk text, access counts, timestamps

No external services. No running processes. Just a file.
