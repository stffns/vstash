"""snapvec backend benchmark: recall@10 vs latency vs disk size.

Compares all 4 snapvec v0.6 index types against exact brute-force cosine
(the recall ceiling) on BGE-small embeddings of BEIR SciFact, padded with
FIQA to reach larger scales (10K / 50K / 100K).

Measures, per backend per scale:
  - build time (seconds, including fit() where applicable)
  - index size on disk (bytes, after save())
  - search latency (ms, median and p95 across BEIR SciFact test queries)
  - recall@10 (vs exact float32 brute-force top-10)

The benchmark includes a sqlite-vec baseline alongside the snapvec
backends but excludes the full vstash pipeline (FTS + RRF). Those
numbers already live in experiments/beir_benchmark.py. Here we answer
a narrower question: among the ANN backends vstash can plug in, which
wins at what scale?

Usage:
    python -m experiments.snapvec_backends_bench
    python -m experiments.snapvec_backends_bench --scales 10000 50000
    python -m experiments.snapvec_backends_bench --output results/snapvec_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.beir_benchmark import download_beir, load_beir
from vstash.embed import embed_query, embed_texts

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
TOP_K = 10
CACHE_DIR = Path("experiments/data/snapvec_bench_cache")


# ------------------------------------------------------------------ #
# Corpus assembly with embedding cache                                 #
# ------------------------------------------------------------------ #


def _embed_cache_path(dataset: str) -> Path:
    return CACHE_DIR / f"{dataset}_bge_small.npy"


def _ids_cache_path(dataset: str) -> Path:
    return CACHE_DIR / f"{dataset}_ids.json"


def embed_dataset(name: str) -> tuple[np.ndarray, list[str]]:
    """Embed a BEIR dataset corpus with BGE-small, cached on disk."""
    emb_path = _embed_cache_path(name)
    ids_path = _ids_cache_path(name)
    if emb_path.exists() and ids_path.exists():
        vecs = np.load(emb_path)
        with open(ids_path) as f:
            ids = json.load(f)
        return vecs, ids

    cache = download_beir(name)
    corpus, _, _ = load_beir(cache)
    ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in ids]
    print(f"  embedding {name}: {len(texts)} docs ...")
    # Chunk to stay under MLX Metal buffer limits on Apple Silicon.
    batch = 256
    parts: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        parts.append(np.asarray(embed_texts(texts[start:end], MODEL), dtype=np.float32))
        if start // batch % 10 == 0:
            print(f"    {end}/{len(texts)}")
    vecs = np.concatenate(parts, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, vecs)
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    return vecs, ids


def assemble_corpus(
    target_n: int,
) -> tuple[np.ndarray, list[int], np.ndarray, list[int], dict]:
    """Returns (vectors, int_ids, anchor_slice_vectors, anchor_slice_int_ids, meta).

    Uses SciFact as the anchor (queries + qrels live there). Padding to
    target_n comes from FIQA. Anchor docs always occupy positions
    [0, N_scifact) so qrels remain valid. Both ``int_ids`` and
    ``anchor_slice_int_ids`` are row indices into the assembled corpus.
    """
    sci_vecs, sci_ids = embed_dataset("scifact")
    cache = download_beir("scifact")
    _, queries, qrels = load_beir(cache)

    n_sci = len(sci_ids)
    if target_n <= n_sci:
        sliced = sci_vecs[:target_n]
        ids_str = sci_ids[:target_n]
        int_ids = list(range(target_n))
        id_map = {s: i for i, s in enumerate(ids_str)}
        return (
            sliced,
            int_ids,
            sliced,
            [id_map[s] for s in sci_ids if s in id_map],
            {
                "queries": queries,
                "qrels": qrels,
                "sci_count": target_n,
            },
        )

    pad_vecs, _ = embed_dataset("fiqa")
    need = target_n - n_sci
    if need > len(pad_vecs):
        # tile FIQA if still not enough
        repeats = (need // len(pad_vecs)) + 1
        pad_vecs = np.tile(pad_vecs, (repeats, 1))
    combined = np.concatenate([sci_vecs, pad_vecs[:need]], axis=0).astype(np.float32)
    int_ids = list(range(len(combined)))
    return (
        combined,
        int_ids,
        sci_vecs,
        list(range(n_sci)),
        {
            "queries": queries,
            "qrels": qrels,
            "sci_count": n_sci,
        },
    )


# ------------------------------------------------------------------ #
# Exact ground-truth search (the recall ceiling)                       #
# ------------------------------------------------------------------ #


def exact_topk(vectors: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Brute-force cosine top-k. Assumes inputs are L2-normalized."""
    scores = vectors @ query
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return idx[np.argsort(-scores[idx])].tolist()


# ------------------------------------------------------------------ #
# Backend builders                                                     #
# ------------------------------------------------------------------ #


def _train_sample(vectors: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    if len(vectors) <= max_n:
        return vectors
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(vectors), size=max_n, replace=False)
    return vectors[sel]


class SqliteVecIndex:
    """Minimal sqlite-vec wrapper with the same (ids, vectors) API as snapvec.

    Isolates the vec virtual table so we measure pure ANN behavior without
    vstash's FTS/RRF pipeline on top.
    """

    def __init__(self, dim: int, db_path: str) -> None:
        import sqlite3

        import sqlite_vec

        self.dim = dim
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{dim}])")

    def add_batch(self, ids: list[int], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        rows = [(int(i), v.tobytes()) for i, v in zip(ids, vectors)]
        self._conn.executemany("INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)", rows)
        self._conn.commit()

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        blob = np.ascontiguousarray(query, dtype=np.float32).tobytes()
        rows = self._conn.execute(
            "SELECT rowid, distance FROM vec_items "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (blob, k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def save(self, path: str) -> None:  # sqlite-vec persists via the db file
        self._conn.commit()


def build_sqlite_vec(vectors: np.ndarray, int_ids: list[int]) -> Any:
    import tempfile

    # Persist to a real file so .stat().st_size reflects disk footprint
    # (sqlite temp/:memory: would report 0).
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    idx = SqliteVecIndex(dim=DIM, db_path=tmp.name)
    idx.add_batch(int_ids, vectors)
    idx._db_file = tmp.name  # stash for size reporting
    return idx


def build_snapvec_flat(vectors: np.ndarray, int_ids: list[int]) -> Any:
    from snapvec import SnapIndex

    idx = SnapIndex(dim=DIM, bits=4, seed=0, normalized=True)
    idx.add_batch(int_ids, vectors)
    return idx


def build_snapvec_residual(vectors: np.ndarray, int_ids: list[int]) -> Any:
    from snapvec import ResidualSnapIndex

    idx = ResidualSnapIndex(dim=DIM, b1=3, b2=3, seed=0, normalized=True)
    idx.add_batch(int_ids, vectors)
    return idx


def build_snapvec_pq(vectors: np.ndarray, int_ids: list[int]) -> Any:
    from snapvec import PQSnapIndex

    idx = PQSnapIndex(dim=DIM, M=96, K=256, seed=0, normalized=True)
    idx.fit(_train_sample(vectors, 50_000))
    idx.add_batch(int_ids, vectors)
    return idx


def build_snapvec_ivfpq(
    vectors: np.ndarray,
    int_ids: list[int],
    *,
    keep_full_precision: bool,
) -> Any:
    from snapvec import IVFPQSnapIndex

    n = len(vectors)
    nlist = max(8, min(1024, int(4 * np.sqrt(n))))
    idx = IVFPQSnapIndex(
        dim=DIM,
        nlist=nlist,
        M=96,
        K=256,
        seed=0,
        normalized=True,
        keep_full_precision=keep_full_precision,
    )
    idx.fit(_train_sample(vectors, 50_000))
    idx.add_batch(int_ids, vectors)
    return idx, nlist


# ------------------------------------------------------------------ #
# Evaluation harness                                                   #
# ------------------------------------------------------------------ #


def run_backend(
    label: str,
    build_fn,
    vectors: np.ndarray,
    int_ids: list[int],
    query_vecs: np.ndarray,
    ground_truth: list[list[int]],
    *,
    search_kwargs: dict | None = None,
    save_suffix: str = ".snp",
) -> dict:
    t0 = time.perf_counter()
    built = build_fn(vectors, int_ids)
    build_s = time.perf_counter() - t0

    # Optional metadata returned by builder (nlist for IVFPQ).
    extra = {}
    idx = built
    if isinstance(built, tuple):
        idx, nlist = built
        extra["nlist"] = nlist

    if hasattr(idx, "_db_file"):
        # sqlite-vec already persisted to a real .db file
        disk_bytes = Path(idx._db_file).stat().st_size
    else:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"index{save_suffix}"
            idx.save(str(path))
            disk_bytes = path.stat().st_size

    kwargs = search_kwargs or {}
    latencies_ms = []
    recalls = []
    for qvec, gt in zip(query_vecs, ground_truth):
        t0 = time.perf_counter()
        res = idx.search(qvec, k=TOP_K, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed)
        hits = [r[0] for r in res]
        gt_set = set(gt)
        recalls.append(sum(1 for h in hits if h in gt_set) / len(gt_set))

    return {
        "backend": label,
        "build_s": round(build_s, 2),
        "disk_bytes": disk_bytes,
        "disk_mb": round(disk_bytes / 1024 / 1024, 2),
        "bytes_per_vec": round(disk_bytes / len(vectors), 1),
        "latency_p50_ms": round(statistics.median(latencies_ms), 3),
        "latency_p95_ms": round(
            statistics.quantiles(latencies_ms, n=20, method="inclusive")[-1], 3
        ),
        "recall_at_10": round(statistics.mean(recalls), 4),
        **extra,
    }


def bench_one_scale(n: int) -> list[dict]:
    print(f"\n=== scale N={n} ===")
    vectors, int_ids, sci_vecs, sci_int_ids, meta = assemble_corpus(n)
    # BGE-small is unit-normalized in output, re-assert for safety.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 1e-9, norms, 1.0)
    vectors = vectors.astype(np.float32)

    queries: dict[str, str] = meta["queries"]
    qrels: dict[str, dict[str, int]] = meta["qrels"]
    # Recall is measured against brute-force top-10 over the full vectors
    # array, so padded FIQA chunks are legitimate competitors — matches
    # what a real deployment would see at this scale.
    _ = sci_vecs, sci_int_ids
    eval_qids = [qid for qid in qrels if qid in queries]
    print(f"  embedding {len(eval_qids)} queries ...")
    qvecs = np.asarray(
        [embed_query(queries[qid], MODEL) for qid in eval_qids],
        dtype=np.float32,
    )
    qnorms = np.linalg.norm(qvecs, axis=1, keepdims=True)
    qvecs = qvecs / np.where(qnorms > 1e-9, qnorms, 1.0)

    print("  computing exact brute-force ground truth ...")
    t0 = time.perf_counter()
    gt = [exact_topk(vectors, q, TOP_K) for q in qvecs]
    print(f"    exact: {time.perf_counter() - t0:.1f}s")

    results: list[dict] = []

    def _run(label, build_fn, **kwargs):
        print(f"  running {label} ...")
        r = run_backend(label, build_fn, vectors, int_ids, qvecs, gt, **kwargs)
        r["n"] = n
        print(
            f"    build={r['build_s']}s  disk={r['disk_mb']}MB "
            f"({r['bytes_per_vec']} B/vec)  "
            f"p50={r['latency_p50_ms']}ms p95={r['latency_p95_ms']}ms  "
            f"recall@10={r['recall_at_10']}"
        )
        results.append(r)

    _run("sqlite-vec (baseline)", build_sqlite_vec, save_suffix=".db")
    _run("snapvec-flat (bits=4)", build_snapvec_flat, save_suffix=".snap")
    _run("snapvec-residual (b1=3,b2=3)", build_snapvec_residual, save_suffix=".snap")
    _run(
        "snapvec-residual (rerank_M=100)",
        build_snapvec_residual,
        search_kwargs={"rerank_M": 100},
        save_suffix=".snap",
    )
    _run("snapvec-pq (M=96,K=256)", build_snapvec_pq, save_suffix=".snpi")
    _run(
        "snapvec-ivfpq (default nprobe)",
        lambda v, i: build_snapvec_ivfpq(v, i, keep_full_precision=False),
        save_suffix=".snpi",
    )
    _run(
        "snapvec-ivfpq+rerank(100)",
        lambda v, i: build_snapvec_ivfpq(v, i, keep_full_precision=True),
        search_kwargs={"rerank_candidates": 100},
        save_suffix=".snpi",
    )
    return results


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[10_000, 50_000, 100_000],
        help="corpus sizes (N) to benchmark",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/snapvec_backends_bench.json",
    )
    args = parser.parse_args()

    all_results: list[dict] = []
    for n in args.scales:
        all_results.extend(bench_one_scale(n))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "dim": DIM,
                "top_k": TOP_K,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out}")

    # Summary table
    print("\n" + "=" * 96)
    print(f"{'backend':<34}{'N':>8}{'build':>8}{'MB':>8}{'p50ms':>8}{'p95ms':>8}{'recall':>9}")
    print("-" * 96)
    for r in all_results:
        print(
            f"{r['backend']:<34}{r['n']:>8}{r['build_s']:>8}{r['disk_mb']:>8}"
            f"{r['latency_p50_ms']:>8}{r['latency_p95_ms']:>8}{r['recall_at_10']:>9}"
        )


if __name__ == "__main__":
    main()
