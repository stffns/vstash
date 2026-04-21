"""
snapvec_ingest_scaling_probe.py -- isolate where per-doc add_document
gets slow on the flat snapvec backend.

Motivation. PR #249 / 2026-04-19 observed an O-something cliff in
per-doc ``store.add_document`` on the ``snapvec`` backend:

    N=5k   :   6.6s total (1.32 ms/doc)
    N=50k  :  16.4s total (0.33 ms/doc)   ← FASTER per-doc than 5k
    N=100k : >40 min      (>24 ms/doc)    ← 75x slower per-doc

The sub-linear 5k->50k followed by super-linear 50k->100k argues
against any single mechanism (disk rewrite, SQLite commit, FTS5
merge) on its own. Could be compounding: warm caches at 50k that
cold at 100k, OS page-cache eviction, FTS5 crisis-merge threshold
crossed, SnapIndex ``.snpv`` size crossing a buffer boundary, ...

Rather than keep guessing, measure directly. Use
``add_documents_batch`` to grow the store cheaply to each
N in [5k, 10k, 25k, 50k, 75k, 100k], then emit 10 per-doc
``add_document`` probes at each checkpoint and record median
latency. Also record how big ``.snpv`` and ``.db`` are at each
point (disk-cost proxy).

Keeps the expensive per-doc path bounded: 6 * 10 = 60 per-doc
probes. Even at worst case 25 ms/probe that is 1.5s of
instrumentation. Total runtime dominated by the batch fills
(~35s each at 100k, linear below).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from vstash.store import VstashStore


CHECKPOINTS = [5_000, 10_000, 25_000, 50_000, 75_000, 100_000]
DIM = 384
PROBES_PER_CHECKPOINT = 10


def _gen_docs(start: int, count: int, rng: np.random.Generator) -> list[dict]:
    """Generate `count` synthetic docs with unit-norm random embeddings."""
    vecs = rng.standard_normal((count, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return [
        {
            "path": f"probe/doc_{start + i}",
            "title": f"doc_{start + i}",
            "chunks": [f"synthetic document body number {start + i} for scaling probe"],
            "embeddings": [vecs[i].tolist()],
            "source_type": "text",
        }
        for i in range(count)
    ]


def _probe_per_doc_latency(
    store: VstashStore, rng: np.random.Generator, n_probes: int
) -> list[float]:
    """Insert ``n_probes`` docs one at a time via ``add_document`` and
    return each call's wall clock in ms."""
    base = store.stats().chunks
    latencies: list[float] = []
    for i in range(n_probes):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec).clip(min=1e-12)
        t0 = time.perf_counter()
        store.add_document(
            path=f"probe/single_{base + i}",
            title=f"single_{base + i}",
            chunks=[f"per-doc probe chunk {base + i}"],
            embeddings=[vec.tolist()],
            source_type="text",
        )
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def _disk_footprint(db_path: Path) -> dict:
    """Size of the sqlite db, wal, and .snpv companion, in bytes."""
    out = {}
    for suffix in ("", "-wal", "-shm"):
        p = db_path.with_suffix(db_path.suffix + suffix) if suffix else db_path
        out[f"db{suffix}"] = p.stat().st_size if p.exists() else 0
    snpv = db_path.with_suffix(".snpv")
    out["snpv"] = snpv.stat().st_size if snpv.exists() else 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="snapvec",
        choices=("sqlite-vec", "snapvec", "snapvec-ivfpq"),
        help="Vector backend to probe. Default: snapvec (the one with the cliff).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/snapvec_ingest_scaling_probe.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(seed=0xBEEF)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / f"{args.backend}.db"
        store = VstashStore(str(db_path), embedding_dim=DIM, vector_backend=args.backend)

        measurements: list[dict] = []
        last_n = 0
        for target in CHECKPOINTS:
            # Fill from last_n to target via batch (cheap).
            delta = target - last_n
            docs = _gen_docs(last_n, delta, rng)
            t0 = time.perf_counter()
            store.add_documents_batch(docs)
            batch_fill_s = time.perf_counter() - t0
            n_at_probe = store.stats().chunks

            # Measure per-doc add latency AT this N.
            probes = _probe_per_doc_latency(store, rng, PROBES_PER_CHECKPOINT)

            footprint = _disk_footprint(db_path)
            cell = {
                "n_at_probe": n_at_probe,
                "batch_fill_seconds": round(batch_fill_s, 2),
                "per_doc_probe_count": len(probes),
                "per_doc_ms_p50": round(statistics.median(probes), 3),
                "per_doc_ms_p95": round(
                    statistics.quantiles(probes, n=20, method="inclusive")[-1], 3
                ),
                "per_doc_ms_mean": round(statistics.mean(probes), 3),
                "per_doc_ms_min": round(min(probes), 3),
                "per_doc_ms_max": round(max(probes), 3),
                "disk_bytes": footprint,
            }
            measurements.append(cell)

            print(
                f"N={n_at_probe:>7}  fill={batch_fill_s:>6.2f}s  "
                f"per-doc p50={cell['per_doc_ms_p50']:>7.3f}ms "
                f"mean={cell['per_doc_ms_mean']:>7.3f}ms  "
                f"max={cell['per_doc_ms_max']:>7.3f}ms  "
                f"snpv={footprint['snpv'] / 1024 / 1024:>5.1f}MB  "
                f"db={footprint['db'] / 1024 / 1024:>6.1f}MB  "
                f"wal={footprint['db-wal'] / 1024 / 1024:>5.1f}MB",
                flush=True,
            )

            last_n = target + PROBES_PER_CHECKPOINT  # we added probes too

        store.close()

    result = {"backend": args.backend, "dim": DIM, "checkpoints": measurements}
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
