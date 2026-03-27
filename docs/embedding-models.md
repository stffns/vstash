# Embedding Models

vstash uses [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings — no API calls, no server, fully in-process.

---

## Supported Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| `BAAI/bge-small-en-v1.5` (default) | 384 | ~700 chunks/s | Great |
| `BAAI/bge-base-en-v1.5` | 768 | ~300 chunks/s | Excellent |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~300 chunks/s | Excellent |

The default (`bge-small-en-v1.5`) offers the best speed/quality tradeoff for most use cases. Switch to a larger model if you need higher retrieval precision on technical or domain-specific content.

---

## Changing the Model

In `vstash.toml`:

```toml
[embeddings]
model = "BAAI/bge-base-en-v1.5"
```

> **Important:** Changing the embedding model requires re-ingesting all documents. The vector dimensions must match between stored embeddings and query embeddings. After changing the model, remove your database and re-add your files:
> ```bash
> rm ~/.vstash/memory.db
> vstash add <your files>
> ```

---

## Backend Selection

On Apple Silicon Macs, vstash can use MLX for GPU-accelerated embeddings:

```toml
[embeddings]
backend = "mlx"   # "onnx" (default, portable) | "mlx" (Apple Silicon) | "auto"
```

`auto` (the default) uses MLX when available, falling back to ONNX.
