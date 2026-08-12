# Limits and validation

*Added in v0.23.0 — [issue #133](https://github.com/stffns/vstash/issues/133).*

A good substrate is honest about its boundaries. vstash enforces explicit limits at every public API entry point so that pathological inputs (50 000-token queries, multi-megabyte chunks, `top_k=100000`) are rejected with a clear error instead of crashing inside SQLite, sqlite-vec, or the embedding model.

## Why

Frameworks built on top of vector databases hide failures behind retries and stack traces. vstash is a **glass box** — when you ask for something the substrate can't safely do, you get a single, named exception describing exactly which limit was hit and how to override it.

## Defaults

| Limit | Default | What it gates |
|---|---|---|
| `max_query_chars` | 10 000 | Maximum length of `query_text` passed to `search()` |
| `max_top_k` | 1 000 | Upper bound on `top_k` (lower bound is always 1) |
| `max_distance_cutoff` | 1 000.0 | Upper bound on `distance_cutoff` (must be ≥ 0) |
| `max_recency_boost` | 100.0 | Upper bound on `recency_boost` (must be ≥ 0) |
| `max_path_chars` | 4 096 | Maximum length of a document path (POSIX `PATH_MAX`) |
| `max_chunks_per_document` | 50 000 | Maximum chunks per `add_document` call |
| `max_chunk_chars` | 1 048 576 | Maximum size of a single chunk (1 MiB) |

These defaults are intentionally generous. Real workloads sit comfortably below all of them; the limits exist to catch caller bugs before they crash the store.

## Configuring

Override any limit in `vstash.toml`:

```toml
[limits]
max_query_chars = 50000      # allow longer queries
max_top_k = 100              # tighten top_k
max_chunks_per_document = 5000
max_chunk_chars = 524288     # 512 KiB
```

All fields are optional — anything you omit keeps its default.

## Exception hierarchy

All validators raise a subclass of `vstash.validation.LimitError`, which itself extends the standard `ValueError`. Existing `except ValueError` handlers continue to work; catch a specific subclass to react to one category.

```python
from vstash.validation import (
    LimitError,  # base class
    QueryInvalidError,
    QueryTooLongError,
    TopKOutOfRangeError,
    DistanceCutoffOutOfRangeError,
    RecencyBoostOutOfRangeError,
    PathTooLongError,
    EmptyDocumentError,
    TooManyChunksError,
    ChunkTooLargeError,
    EmbeddingMismatchError,
    InvalidIdentifierError,
)
```

## Where validation runs

- **`VstashStore.search()`** — validates `query_text`, `top_k`, `distance_cutoff`, `recency_boost` before any work.
- **`VstashStore.add_document()`** — validates `path`, chunk count, chunk sizes, and `len(chunks) == len(embeddings)` before opening the write transaction.
- **`Memory.__init__()`** — validates `project` and `collection` identifiers (rejects empty / whitespace-only / control characters).

Validation runs **once per public call**. The hot path is unaffected — each validator is a handful of comparisons.

## Example

```python
from vstash import Memory
from vstash.validation import TopKOutOfRangeError, ChunkTooLargeError

mem = Memory(project="research")

# Caller bug: top_k of 100 000 would OOM on a large corpus.
try:
    mem.search("transformers", top_k=100_000)
except TopKOutOfRangeError as e:
    print(f"rejected: {e}")
    # → rejected: top_k 100000 exceeds limit 1000 (set [limits] max_top_k to override)

# Caller bug: a single 5 MB "chunk" would crash the embedding model.
try:
    mem.remember("x" * 5_000_000, title="huge")
except ChunkTooLargeError as e:
    print(f"rejected: {e}")
```

## What's NOT validated (by design)

- **Runtime memory usage** — no `psutil` polling, no per-call memory budget. Out of scope.
- **CPU throttling** — not the substrate's job.
- **Rate limiting** — that's an upper-layer concern (proxy, gateway, agent loop).
- **Embedding dimension** — already validated by sqlite-vec at insert time.
- **Schema integrity** — handled by foreign keys and the SQLite checks themselves.

## Relationship to observability

Limits and observability (#132) are paired defenses. Observability tells you *what happened* after the fact; limits stop the worst inputs *before* they ever touch the hot path. Use both — they cost almost nothing and they buy you the difference between "vstash crashed at 3 AM" and "vstash refused at 3 AM with a one-line error message".
