"""
mmr_dedup_bench.py -- quantify the current Python-level cost of
``VstashStore._mmr_dedup`` across realistic workload shapes.

Context (H-S4 in experiments/hypotheses.md): the current MMR greedy
loop updates ``max_sims`` one same-doc chunk at a time, computing a
pure-Python cosine similarity inside the inner update. The hypothesis
is that a vectorised all-pairs cosine matrix + numpy max update would
be 5-10x faster at large ``top_k`` with high duplicate-doc ratios
(e.g. RAG over a book corpus with top_k=100).

Before refactoring the hot path, this bench:

1. Synthesises a ranked list of ``n`` fake chunks with an adjustable
   ``dup_ratio`` (fraction of chunks whose path is a duplicate of at
   least one other chunk's path) and a realistic ``dim`` embedding.
2. Calls the live ``_mmr_dedup`` at various ``(n, top_k, dup_ratio)``.
3. Reports wall-clock p50/p95 across 50 runs per cell, plus the
   selected-set size so we can see if the early-exit (mmr_score < 0)
   short-circuits the loop in practice.

Run:

    python -m experiments.mmr_dedup_bench

Outputs a Markdown table on stdout plus a JSON file at
``experiments/results/mmr_dedup_bench.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from vstash.store import VstashStore

# Default embedding dim matches BAAI/bge-small-en-v1.5 (the project's
# shipped default). Fixed rather than derived from VstashConfig so the
# benchmark runs even when the developer's local vstash.toml points at
# a model not registered in vstash.embed. Override via ``--dim``.
_DEFAULT_DIM = 384


def _make_chunk_row(
    chunk_id: int,
    path: str,
    score: float,
) -> dict[str, str | int | float]:
    """Produce a ranked-row shape matching what _fuse_rrf_scores emits."""
    return {
        "id": chunk_id,
        "path": path,
        "rrf": score,
        # Extra fields _mmr_dedup does not read but are present in real rows:
        "seq": 0,
        "text": "",
        "title": path,
    }


def _populate_store(
    store: VstashStore,
    n_chunks: int,
    dup_ratio: float,
    dim: int,
    rng: random.Random,
) -> list[dict[str, str | int | float]]:
    """Add ``n_chunks`` chunks into ``store`` and return the pre-MMR
    ranked list in a shape _mmr_dedup will accept.

    ``dup_ratio`` fraction of chunks share a path with at least one
    other chunk. The rest get unique paths. Vectors are raw
    ``np.random.randn`` 32-bit floats (not L2-normalised); the MMR
    path computes its own norms internally via ``math.hypot`` and does
    not assume unit-length inputs.
    """
    n_dup_chunks = int(n_chunks * dup_ratio)
    n_unique_chunks = n_chunks - n_dup_chunks

    paths: list[str] = []
    # Unique chunks: one path each.
    for i in range(n_unique_chunks):
        paths.append(f"/u/uniq_{i:05d}.md")
    # Duplicate chunks: cluster into 2-5 per doc, but make the clusters
    # sum exactly to ``n_dup_chunks`` so the realised dup ratio matches
    # the requested one. Fold any leftover remainder of 1 into the
    # previous cluster; never emit a solo "duplicate" path (that would
    # silently drop the dup ratio).
    cluster_sizes: list[int] = []
    remaining = n_dup_chunks
    while remaining > 0:
        if remaining == 1 and cluster_sizes:
            cluster_sizes[-1] += 1
            remaining = 0
            break
        size = min(rng.randint(2, 5), remaining)
        cluster_sizes.append(size)
        remaining -= size
    d = 0
    for size in cluster_sizes:
        for _ in range(size):
            paths.append(f"/d/book_{d:04d}.pdf")
        d += 1
    rng.shuffle(paths)

    # Group by path so we hand each document to add_document as a single
    # multi-chunk call (matches real ingestion shape).
    by_path: dict[str, list[int]] = {}
    for idx, p in enumerate(paths):
        by_path.setdefault(p, []).append(idx)

    ranked: list[dict[str, str | int | float]] = []
    for doc_path, indices in by_path.items():
        texts = [f"chunk text {i} for {doc_path}" for i in indices]
        # Batch-generate embeddings via a single np.random.randn call;
        # converting the array with .tolist() is ~10x faster than the
        # per-row list(map(float, ...)) pattern at dim=384.
        embeddings = np.random.randn(len(indices), dim).astype(np.float32).tolist()
        store.add_document(
            path=doc_path,
            title=doc_path,
            chunks=texts,
            embeddings=embeddings,
            source_type="text",
        )

    # Pull the chunk ids the store assigned so rrf scoring can reference
    # real ids (the _mmr_dedup loop fetches vec_chunks by rowid).
    # Explicit ORDER BY keeps the synthesised RRF ranking deterministic
    # across SQLite versions / query plans.
    rows = store._conn.execute(
        "SELECT c.id, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
    ).fetchall()
    # Synthesise a descending RRF score so the ranking is meaningful.
    # In reality ``rrf`` collides in ties but the MMR loop's tie-break
    # on index is deterministic.
    step = 0.001
    for i, r in enumerate(rows):
        ranked.append(_make_chunk_row(int(r["id"]), r["path"], 1.0 - i * step))
    return ranked


def _bench_cell(
    n_chunks: int,
    top_k: int,
    dup_ratio: float,
    mmr_lambda: float,
    repeats: int,
    tmp_path: Path,
    dim: int,
) -> dict[str, float | int]:
    """Run ``repeats`` back-to-back _mmr_dedup calls and report timings.

    The store is created once and reused across repeats so we only
    measure the Python-level MMR loop, not ingest overhead.
    """
    rng = random.Random(0x5A4D)

    store = VstashStore(str(tmp_path), embedding_dim=dim)
    try:
        ranked = _populate_store(store, n_chunks, dup_ratio, dim, rng)

        # Warm up sqlite + OS page cache.
        _ = store._mmr_dedup(ranked, top_k=top_k, mmr_lambda=mmr_lambda)

        times: list[float] = []
        selected_sizes: list[int] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = store._mmr_dedup(ranked, top_k=top_k, mmr_lambda=mmr_lambda)
            dt = time.perf_counter() - t0
            times.append(dt)
            selected_sizes.append(len(out))

        # Proper 95th percentile via ``statistics.quantiles``. The
        # naive ``sorted(times)[int(len * 0.95) - 1]`` pick is biased
        # low for small samples (at repeats=50 it lands on the 47th
        # value, ~p92). ``method="inclusive"`` matches
        # ``numpy.percentile`` exactly.
        p95 = (
            times[0]
            if len(times) == 1
            else statistics.quantiles(times, n=100, method="inclusive")[94]
        )
        return {
            "n_chunks": n_chunks,
            "top_k": top_k,
            "dup_ratio": dup_ratio,
            "mmr_lambda": mmr_lambda,
            "p50_ms": round(statistics.median(times) * 1000, 3),
            "p95_ms": round(p95 * 1000, 3),
            "mean_ms": round(statistics.mean(times) * 1000, 3),
            "selected_size_p50": int(statistics.median(selected_sizes)),
            "repeats": repeats,
        }
    finally:
        try:
            store.close()
        except Exception:
            pass
        for suffix in ("", "-wal", "-shm"):
            try:
                (tmp_path.parent / (tmp_path.name + suffix)).unlink(missing_ok=True)
            except Exception:
                pass


def _print_markdown(results: list[dict[str, float | int]]) -> None:
    """Summarise the grid as a Markdown table keyed by workload shape."""
    header = "| N chunks | top_k | dup_ratio | mmr_lambda | p50 ms | p95 ms | mean ms | selected |"
    sep = "| -------- | ----- | --------- | ---------- | ------ | ------ | ------- | -------- |"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r['n_chunks']:>8} | {r['top_k']:>5} | {r['dup_ratio']:>9.2f} | "
            f"{r['mmr_lambda']:>10.2f} | {r['p50_ms']:>6} | {r['p95_ms']:>6} | "
            f"{r['mean_ms']:>7} | {r['selected_size_p50']:>8} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim",
        type=int,
        default=_DEFAULT_DIM,
        help=(
            f"Embedding dimension. Default {_DEFAULT_DIM} (bge-small). "
            "Override to match a different model if you want realistic "
            "inner-product cost at that dim."
        ),
    )
    args = parser.parse_args()

    np.random.seed(0x5A4D)
    random.seed(0x5A4D)
    dim = args.dim

    # Realistic grid:
    # - N: production search returns ~100-200 pre-MMR candidates; RAG
    #   work with aggressive top_k can push this to 1000+.
    # - top_k: default 10; RAG pipelines use 50/100.
    # - dup_ratio: 0.0 skips the greedy loop entirely (hard-dedup fast
    #   path), 0.5 is a mixed corpus (code + docs), 1.0 is a single-book
    #   corpus.
    # - mmr_lambda: 0.5 default; 0.2 is heavy diversity, 0.8 is
    #   relevance-preserving.
    grid = [
        # (n_chunks, top_k, dup_ratio, mmr_lambda)
        (100, 10, 0.0, 0.5),  # hard-dedup fast path
        (100, 10, 0.5, 0.5),  # mixed corpus, small K
        (100, 10, 1.0, 0.5),  # single-book, small K
        (200, 20, 0.5, 0.5),  # RAG-ish
        (500, 50, 0.5, 0.5),  # large RAG
        (500, 50, 1.0, 0.5),  # pathological: all dup
        (1000, 100, 0.5, 0.5),  # very large RAG
        (1000, 100, 1.0, 0.5),  # pathological at large K
        (1000, 100, 0.5, 0.2),  # diversity-heavy at large K
    ]

    # ``TemporaryDirectory`` isolates this run from any previous
    # leftover files and from concurrent runs. Cross-platform: picks
    # ``/tmp`` on unix, ``%TEMP%`` on Windows.
    results: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory(prefix="vstash_mmr_bench_") as tmp_root:
        tmp_parent = Path(tmp_root)
        for i, (n, k, dup, lam) in enumerate(grid):
            tmp_path = tmp_parent / f"bench_{i}.db"
            print(f"-> cell {i + 1}/{len(grid)}: n={n}, k={k}, dup={dup}, lam={lam}")
            cell = _bench_cell(
                n_chunks=n,
                top_k=k,
                dup_ratio=dup,
                mmr_lambda=lam,
                repeats=50,
                tmp_path=tmp_path,
                dim=dim,
            )
            results.append(cell)

    print()
    _print_markdown(results)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "mmr_dedup_bench.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nJSON results saved to {out_file}")


if __name__ == "__main__":
    main()
