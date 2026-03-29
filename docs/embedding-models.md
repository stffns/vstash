# Embedding Models

vstash uses [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings — no API calls, no server, fully in-process.

---

## Supported Models

### English-only

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| `BAAI/bge-small-en-v1.5` (default) | 384 | ~700 chunks/s | Great |
| `BAAI/bge-base-en-v1.5` | 768 | ~300 chunks/s | Excellent |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~300 chunks/s | Excellent |

### Multilingual

| Model | Dimensions | Languages | Speed | Quality |
|-------|-----------|-----------|-------|---------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ | ~600 chunks/s | Great |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ | ~250 chunks/s | Excellent |
| `intfloat/multilingual-e5-large` | 1024 | 100+ | ~100 chunks/s | Best |

**Recommendation:** For multilingual corpora, use `paraphrase-multilingual-MiniLM-L12-v2`. Same 384 dimensions as the default model = zero latency impact on search. Cross-lingual query similarity improves ~40% over English-only models.

---

## Changing the Model

In `vstash.toml`:

```toml
[embeddings]
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Then re-embed your existing chunks:

```bash
vstash reindex
```

The `reindex` command drops and recreates the vector index with the new model's dimensions, re-embeds all chunks in batches, and shows a progress bar. Text, metadata, and FTS index are preserved.

Options:

```bash
vstash reindex                          # uses model from vstash.toml
vstash reindex --model BAAI/bge-base-en-v1.5  # override model
vstash reindex --batch-size 128         # smaller batches (less RAM)
vstash reindex --yes                    # skip confirmation prompt
```

---

## When to upgrade dimensions

| Corpus size | Recommended dims | Model |
|-------------|:---:|-------|
| < 5,000 chunks | 384 | `bge-small-en-v1.5` or `multilingual-MiniLM` |
| 5,000 - 50,000 | 768 | `bge-base-en-v1.5` or `multilingual-mpnet` |
| 50,000+ | 1024 | `multilingual-e5-large` |

Higher dimensions give better discrimination between semantically similar chunks, but the improvement is negligible on small corpora. Search latency scales linearly with dimensions (~0.3ms at 384, ~0.7ms at 1024).

---

## Backend Selection

On Apple Silicon Macs, vstash can use MLX for GPU-accelerated embeddings:

```toml
[embeddings]
backend = "mlx"   # "onnx" (default, portable) | "mlx" (Apple Silicon) | "auto"
```

`auto` (the default) uses MLX when available, falling back to ONNX.
