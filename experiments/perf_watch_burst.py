"""Probe: watch.py burst ingestion shape (issue #266).

Goal: before fixing, confirm that the per-file ``ingest()`` path used by
the watch worker scales super-linearly on the snapvec flat backend.
The serialized queue in ``watch.py`` calls ``ingest(file)`` once per
dropped file; each ``add_document`` pays ``SnapIndex.add_batch`` which
internally vstacks against the growing buffer. For a burst of N files
that is O(N^2) memcpy.

Prints a per-N timing table for two shapes:
- ``serialized``: N x ``store.add_document`` (matches current watch).
- ``batched``:   one ``store.batch_mode() + add_documents_batch``
                  (matches the fix route via ingest_batch).

If the probe shows quadratic growth on serialized and near-linear on
batched, the fix in watch.py is justified. If the shapes match, the
hypothesis in #266 is incomplete -- stop, re-probe, do not write the
fix (probe-first discipline from the 2026-04-22 audit lessons).

Run: python -m experiments.perf_watch_burst [--backend snapvec|sqlite-vec]
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time

from vstash.embed import embed_texts, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"


def _seed_embeddings(k: int = 16) -> list[list[float]]:
    seed_texts = [f"seed embedding {i} for watch-burst probe" for i in range(k)]
    return embed_texts(seed_texts, model_name=MODEL)


def _make_docs(n: int, embeddings: list[list[float]]) -> list[dict]:
    return [
        {
            "path": f"/probe/burst_doc_{i}.md",
            "title": f"Burst Document {i}",
            "chunks": [f"content of burst document {i} mentioning topic {i % 8}."],
            "embeddings": [embeddings[i % len(embeddings)]],
            "source_type": "markdown",
        }
        for i in range(n)
    ]


def _bench_serialized(docs: list[dict], dim: int, backend: str, rounds: int) -> float:
    """Emulate the current watch worker: one add_document per file."""
    latencies = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            store = VstashStore(
                f"{tmp}/burst.db",
                embedding_dim=dim,
                vector_backend=backend,
                snapvec_bits=4,
            )
            t0 = time.perf_counter()
            for doc in docs:
                store.add_document(
                    path=doc["path"],
                    title=doc["title"],
                    chunks=doc["chunks"],
                    embeddings=doc["embeddings"],
                )
            store.close()
            latencies.append((time.perf_counter() - t0) * 1000)
    return statistics.median(latencies)


def _bench_batched(docs: list[dict], dim: int, backend: str, rounds: int) -> float:
    """Emulate the fix: one add_documents_batch for the burst."""
    latencies = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            store = VstashStore(
                f"{tmp}/burst.db",
                embedding_dim=dim,
                vector_backend=backend,
                snapvec_bits=4,
            )
            t0 = time.perf_counter()
            with store.batch_mode(defer_fts=True):
                store.add_documents_batch(docs)
            store.close()
            latencies.append((time.perf_counter() - t0) * 1000)
    return statistics.median(latencies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["sqlite-vec", "snapvec", "snapvec-ivfpq"],
        default="snapvec",
        help="Vector backend to probe (default: snapvec, the worst case).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 300, 1000, 3000],
        help="Burst sizes to sweep.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Runs per size (median).")
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)
    embeddings = _seed_embeddings()

    print(f"backend={args.backend}  model={MODEL}  dim={dim}  rounds={args.rounds}")
    print(f"{'N':>6}  {'serialized (ms)':>16}  {'batched (ms)':>14}  {'speedup':>8}")
    prev_ser = None
    prev_n = None
    for n in args.sizes:
        docs = _make_docs(n, embeddings)
        ser = _bench_serialized(docs, dim, args.backend, args.rounds)
        bat = _bench_batched(docs, dim, args.backend, args.rounds)
        speedup = ser / bat if bat > 0 else float("inf")
        if prev_ser is not None and prev_n is not None:
            ratio_n = n / prev_n
            ratio_t = ser / prev_ser if prev_ser > 0 else float("inf")
            shape = f"  (N x {ratio_n:.1f}, t x {ratio_t:.2f})"
        else:
            shape = ""
        print(f"{n:>6}  {ser:>16.1f}  {bat:>14.1f}  {speedup:>7.1f}x{shape}")
        prev_ser, prev_n = ser, n

    print()
    print("Reading the output:")
    print("  - If serialized t grows quadratically with N (t ratio ~ N^2), the O(N^2)")
    print("    hypothesis holds and the watcher fix is justified.")
    print("  - If both shapes grow linearly, the hypothesis is incomplete -- stop and")
    print("    re-probe (probe-first discipline from the 2026-04-22 audit).")


if __name__ == "__main__":
    main()
