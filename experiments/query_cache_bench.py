"""Benchmark: query LRU cache hit vs cold search latency.

Measures repeated-query speedup when cache is enabled vs disabled.
Run: python -m experiments.query_cache_bench
"""

from __future__ import annotations

import statistics
import tempfile
import time

from vstash.config import CacheConfig
from vstash.embed import embed_texts, get_embedding_dim
from vstash.store import VstashStore


def _bench(label: str, store: VstashStore, q_emb: list[float], q_text: str, rounds: int = 50):
    latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        store.search(q_emb, q_text, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  {label:20s}  p50={p50:7.2f}ms  p99={p99:7.2f}ms  (n={rounds})")
    return p50


def main():
    dim = get_embedding_dim("BAAI/bge-small-en-v1.5")
    chunks = [
        "Python is a high-level programming language known for readability.",
        "Machine learning models require training data and compute.",
        "SQLite is a lightweight embedded database engine.",
        "Reciprocal Rank Fusion combines ranked lists from multiple retrievers.",
        "Vector embeddings capture semantic meaning in dense float arrays.",
    ] * 20  # 100 chunks

    embeddings = embed_texts(chunks, model_name="BAAI/bge-small-en-v1.5")
    q_emb = embed_texts(["python programming language"], model_name="BAAI/bge-small-en-v1.5")[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Without cache
        db_no_cache = f"{tmpdir}/no_cache.db"
        store_nc = VstashStore(
            db_no_cache, embedding_dim=dim, cache=CacheConfig(query_cache_size=0)
        )
        store_nc.add_document(
            path="/bench/corpus.md", title="Corpus", chunks=chunks, embeddings=embeddings
        )
        print("Repeated query (100 chunks, 50 rounds):")
        p50_cold = _bench("no cache", store_nc, q_emb, "python programming language")
        store_nc.close()

        # With cache
        db_cache = f"{tmpdir}/cache.db"
        store_c = VstashStore(db_cache, embedding_dim=dim, cache=CacheConfig(query_cache_size=128))
        store_c.add_document(
            path="/bench/corpus.md", title="Corpus", chunks=chunks, embeddings=embeddings
        )
        # Warm the cache with one search
        store_c.search(q_emb, "python programming language", top_k=5)
        p50_hot = _bench("cache (hot)", store_c, q_emb, "python programming language")
        store_c.close()

        speedup = p50_cold / p50_hot if p50_hot > 0 else float("inf")
        print(f"\n  Speedup: {speedup:.1f}x (p50 cold/hot)")


if __name__ == "__main__":
    main()
