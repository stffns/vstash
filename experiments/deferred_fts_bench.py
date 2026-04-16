"""Benchmark: deferred FTS indexing vs immediate FTS during batch ingest.

Measures ingest latency for N documents with and without defer_fts=True.
Run: python -m experiments.deferred_fts_bench
"""

from __future__ import annotations

import statistics
import tempfile
import time

from vstash.embed import embed_texts, get_embedding_dim
from vstash.store import VstashStore


def _make_docs(n: int, dim: int, embeddings: list[list[float]]):
    return [
        {
            "path": f"/bench/doc_{i}.md",
            "title": f"Document {i}",
            "chunks": [f"This is the content of document number {i} about topic {i % 10}."],
            "embeddings": [embeddings[i % len(embeddings)]],
            "source_type": "markdown",
        }
        for i in range(n)
    ]


def _bench_ingest(label: str, n: int, dim: int, docs: list, defer_fts: bool, rounds: int = 3):
    latencies = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VstashStore(f"{tmpdir}/bench.db", embedding_dim=dim)
            t0 = time.perf_counter()
            with store.batch_mode(defer_fts=defer_fts):
                for doc in docs:
                    store.add_document(
                        path=doc["path"],
                        title=doc["title"],
                        chunks=doc["chunks"],
                        embeddings=doc["embeddings"],
                    )
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            store.close()
    p50 = statistics.median(latencies)
    print(f"  {label:30s}  p50={p50:8.1f}ms  (n={n}, rounds={rounds})")
    return p50


def main():
    dim = get_embedding_dim("BAAI/bge-small-en-v1.5")
    seed_texts = [f"sample text for embedding {i}" for i in range(10)]
    embeddings = embed_texts(seed_texts, model_name="BAAI/bge-small-en-v1.5")

    for n in [50, 200, 500]:
        docs = _make_docs(n, dim, embeddings)
        print(f"\nBatch ingest: {n} documents (1 chunk each)")
        p50_imm = _bench_ingest("immediate FTS", n, dim, docs, defer_fts=False)
        p50_def = _bench_ingest("deferred FTS", n, dim, docs, defer_fts=True)
        speedup = p50_imm / p50_def if p50_def > 0 else float("inf")
        print(f"  {'speedup':30s}  {speedup:.2f}x")


if __name__ == "__main__":
    main()
