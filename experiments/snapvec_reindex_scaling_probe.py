"""
snapvec_reindex_scaling_probe.py -- isolate the cost of
``VstashStore.reindex`` as a function of N on the snapvec backend.

Motivation. Issue #265 (spun off from #252 Tier 1) flags ``reindex``
as likely quadratic for the same two reasons #264 just fixed in
``_rebuild_snapvec_from_vec_chunks``:

1. ``LIMIT ? OFFSET ?`` pagination on ``chunks`` (store.py:4293).
   SQLite rescans ``offset`` rows per page so N/batch_size pages
   cost O(N^2/batch_size) reads.
2. ``self._snap.add_batch(ids, embeddings)`` called per page
   (store.py:4313). ``SnapIndex.add_batch`` does np.vstack on the
   growing index internally (snapvec/_index.py:206), so N/batch_size
   calls copy a growing buffer.

Fill the store with N random-vector docs via ``add_documents_batch``
(one snapvec add_batch for the whole fill, cheap), then call
``store.reindex(identity_embed_fn, new_dim=DIM)`` and record wall
clock. Sweep N across a geometric ladder.

The embed_fn returns deterministic random unit vectors per batch
(seeded from the probe RNG). It is NOT a true identity -- real
embedding latency is skipped on purpose, but a small per-call
RNG + normalization cost is paid so that the allocation pattern
is realistic. ``new_dim`` stays equal to the current dim so the
test exercises the re-embed loop without changing vec_chunks'
schema dimensionality.
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
    """Generate `count` synthetic docs with unit-norm random embeddings."""
    vecs = rng.standard_normal((count, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return [
        {
            "path": f"probe/doc_{start + i}",
            "title": f"doc_{start + i}",
            "chunks": [f"synthetic body {start + i} for reindex scaling probe"],
            "embeddings": [vecs[i].tolist()],
            "source_type": "text",
        }
        for i in range(count)
    ]


def _make_identity_embed_fn(rng: np.random.Generator):
    """Return an embed_fn that yields deterministic random vectors.

    ``store.reindex`` calls ``embed_fn(texts)`` per SQL batch. For
    the probe we do NOT want to measure real embedding latency, so
    we emit deterministic random vectors keyed off the probe RNG.
    The content of the vectors doesn't matter -- only the store
    write path is being measured.
    """

    def embed_fn(texts: list[str]) -> list[list[float]]:
        arr = rng.standard_normal((len(texts), DIM)).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr.tolist()

    return embed_fn


def _measure_reindex(store: VstashStore, rng: np.random.Generator, repeats: int) -> list[float]:
    """Run ``store.reindex`` `repeats` times and return per-call wall clock."""
    samples: list[float] = []
    for _ in range(repeats):
        embed_fn = _make_identity_embed_fn(rng)
        gc.collect()
        t0 = time.perf_counter()
        store.reindex(embed_fn, new_dim=DIM)
        samples.append(time.perf_counter() - t0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Cumulative chunk counts at which to measure reindex.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of reindex calls per checkpoint (reports median).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/snapvec_reindex_scaling_probe.json"),
    )
    parser.add_argument(
        "--backend",
        default="snapvec",
        choices=("sqlite-vec", "snapvec", "snapvec-ivfpq"),
        help="Which backend to probe.",
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

    rng = np.random.default_rng(seed=0xCAFE)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / f"{args.backend}.db"
        store = VstashStore(str(db_path), embedding_dim=DIM, vector_backend=args.backend)

        measurements: list[dict] = []
        last_n = 0
        for target in args.checkpoints:
            delta = target - last_n
            cursor = 0
            while cursor < delta:
                chunk = min(args.fill_batch, delta - cursor)
                store.add_documents_batch(_gen_docs(last_n + cursor, chunk, rng))
                cursor += chunk
            last_n = target

            n_at_probe = store.stats().chunks
            samples = _measure_reindex(store, rng, args.repeats)

            cell = {
                "n_at_probe": n_at_probe,
                "reindex_seconds_p50": round(statistics.median(samples), 3),
                "reindex_seconds_min": round(min(samples), 3),
                "reindex_seconds_max": round(max(samples), 3),
                "reindex_seconds_mean": round(statistics.mean(samples), 3),
                "repeats": args.repeats,
            }
            measurements.append(cell)

            print(
                f"N={n_at_probe:>7}  reindex p50={cell['reindex_seconds_p50']:>7.3f}s "
                f"min={cell['reindex_seconds_min']:>7.3f}s "
                f"max={cell['reindex_seconds_max']:>7.3f}s",
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
