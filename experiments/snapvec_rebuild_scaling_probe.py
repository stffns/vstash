"""
snapvec_rebuild_scaling_probe.py -- isolate the cost of
``VstashStore._rebuild_snapvec_from_vec_chunks`` as a function of N.

Motivation. Issue #252 (O(N^2) audit Tier 1) flagged this path as
a likely quadratic because the loop streams 10k-row batches of
vectors out of ``vec_chunks`` and calls ``SnapIndex.add_batch``
per batch. ``SnapIndex.add_batch`` in turn does

    self._indices = np.vstack([self._indices, batch_idx])
    self._norms   = np.concatenate([self._norms, batch_norms])

(see snapvec/_index.py:206-214). Each call copies the full
current buffer, so over N/batch_size calls the total cost is
O(N^2 / batch_size).

This probe fills a store to target N via ``add_documents_batch``
(one snapvec add_batch for the whole fill, cheap), then calls
``_rebuild_snapvec_from_vec_chunks()`` and records wall clock.
It sweeps N across a geometric ladder and writes a JSON table so
we can compare the pre- and post-fix slopes directly.

Runtime budget: even at N=250k a single rebuild should land under
a minute on current HEAD (it's the vstack cost we're measuring,
not anything that spirals for hours).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from vstash.store import VstashStore


DEFAULT_CHECKPOINTS = [5_000, 10_000, 25_000, 50_000, 100_000]
DIM = 384


def _gen_docs(start: int, count: int, rng: np.random.Generator) -> list[dict]:
    """Generate `count` synthetic docs with unit-norm random embeddings.

    One chunk per doc keeps chunk_count == doc_count == N which makes
    the post-rebuild assertions simple.
    """
    vecs = rng.standard_normal((count, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return [
        {
            "path": f"probe/doc_{start + i}",
            "title": f"doc_{start + i}",
            "chunks": [f"synthetic body {start + i} for rebuild scaling probe"],
            "embeddings": [vecs[i].tolist()],
            "source_type": "text",
        }
        for i in range(count)
    ]


def _measure_rebuild(store: VstashStore, repeats: int) -> list[float]:
    """Run ``_rebuild_snapvec_from_vec_chunks`` `repeats` times and return
    per-call wall clock in seconds."""
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        store._rebuild_snapvec_from_vec_chunks()
        samples.append(time.perf_counter() - t0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Cumulative chunk counts at which to measure rebuild.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of rebuild calls per checkpoint (reports median).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/snapvec_rebuild_scaling_probe.json"),
    )
    parser.add_argument(
        "--backend",
        default="snapvec",
        choices=("snapvec", "snapvec-ivfpq"),
        help="Which snapvec variant to probe.",
    )
    parser.add_argument(
        "--fill-batch",
        type=int,
        default=5_000,
        help="Batch size for seeding the store (smaller = more progress ticks).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label written to the result JSON (e.g. 'before', 'after').",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(seed=0xC0FFEE)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / f"{args.backend}.db"
        store = VstashStore(str(db_path), embedding_dim=DIM, vector_backend=args.backend)

        measurements: list[dict] = []
        last_n = 0
        for target in args.checkpoints:
            delta = target - last_n
            # Seed in sub-batches so memory pressure stays sane at large N.
            cursor = 0
            while cursor < delta:
                chunk = min(args.fill_batch, delta - cursor)
                store.add_documents_batch(_gen_docs(last_n + cursor, chunk, rng))
                cursor += chunk
            last_n = target

            n_at_probe = store.stats().chunks
            samples = _measure_rebuild(store, args.repeats)

            cell = {
                "n_at_probe": n_at_probe,
                "rebuild_seconds_p50": round(statistics.median(samples), 3),
                "rebuild_seconds_min": round(min(samples), 3),
                "rebuild_seconds_max": round(max(samples), 3),
                "rebuild_seconds_mean": round(statistics.mean(samples), 3),
                "repeats": args.repeats,
            }
            measurements.append(cell)

            print(
                f"N={n_at_probe:>7}  rebuild p50={cell['rebuild_seconds_p50']:>7.3f}s "
                f"min={cell['rebuild_seconds_min']:>7.3f}s "
                f"max={cell['rebuild_seconds_max']:>7.3f}s",
                flush=True,
            )

        store.close()

    result = {
        "label": args.label,
        "backend": args.backend,
        "dim": DIM,
        "checkpoints": measurements,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
