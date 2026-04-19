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

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from vstash.config import VstashConfig
from vstash.embed import get_embedding_dim
from vstash.store import VstashStore


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
    other chunk. The rest get unique paths. Vectors are random
    normalised 32-bit floats (same shape vstash stores).
    """
    n_dup_chunks = int(n_chunks * dup_ratio)
    n_unique_chunks = n_chunks - n_dup_chunks

    paths: list[str] = []
    # Unique chunks: one path each.
    for i in range(n_unique_chunks):
        paths.append(f"/u/uniq_{i:05d}.md")
    # Duplicate chunks: cluster them into docs with 2-5 chunks each.
    d = 0
    while len(paths) < n_chunks:
        cluster_size = rng.randint(2, 5)
        for _ in range(cluster_size):
            if len(paths) < n_chunks:
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
        embeddings = [list(map(float, np.random.randn(dim).astype(np.float32))) for _ in indices]
        store.add_document(
            path=doc_path,
            title=doc_path,
            chunks=texts,
            embeddings=embeddings,
            source_type="text",
        )

    # Pull the chunk ids the store assigned so rrf scoring can reference
    # real ids (the _mmr_dedup loop fetches vec_chunks by rowid).
    rows = store._conn.execute(
        "SELECT c.id, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
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

        return {
            "n_chunks": n_chunks,
            "top_k": top_k,
            "dup_ratio": dup_ratio,
            "mmr_lambda": mmr_lambda,
            "p50_ms": round(statistics.median(times) * 1000, 3),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1] * 1000, 3),
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
    np.random.seed(0x5A4D)
    random.seed(0x5A4D)

    cfg = VstashConfig()
    dim = get_embedding_dim(cfg.embeddings.model)

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

    tmp_parent = Path("/tmp/vstash_mmr_bench")
    tmp_parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int]] = []
    for i, (n, k, dup, lam) in enumerate(grid):
        tmp_path = tmp_parent / f"bench_{i}.db"
        for suffix in ("", "-wal", "-shm"):
            try:
                (tmp_parent / (tmp_path.name + suffix)).unlink(missing_ok=True)
            except Exception:
                pass
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
