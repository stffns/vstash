"""
beir_snapvec_ivfpq_sweep.py -- A1-ivfpq.

Unlike flat, snapvec-ivfpq DOES have a learnable codebook (coarse
k-means centroids + residual PQ codebooks). ``store.fit_ivfpq()``
trains them from whatever is in vec_chunks at fit time. In the
existing h2h (PR #254) we let the defaults ride: M=96, K=256,
rerank_candidates=100, nlist=4 * sqrt(N) clamped to [8, 1024],
nprobe=nlist // 8 (internal default).

This sweep varies the knobs that actually shape fit quality with
v3 embeddings: ``rerank_candidates`` (fp16 top-N re-ranked after
IVFPQ prefilter) and ``nprobe`` (clusters probed at query time).
Both affect the NDCG-vs-latency Pareto. M sweeps are deferred
(require dim-divisibility and recompiling the PQ code bank) and
nlist is data-size-derived by default.

Reuses the on-disk embedding cache from ``beir_snapvec_h2h`` so
the corpora are not re-embedded.

Usage:

    python -m experiments.beir_snapvec_ivfpq_sweep
    python -m experiments.beir_snapvec_ivfpq_sweep --rerank 50 100 200
    python -m experiments.beir_snapvec_ivfpq_sweep --datasets scifact nfcorpus
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

from experiments.beir_benchmark import download_beir, load_beir
from experiments.beir_snapvec_h2h import (
    BM25_NDCG_10,
    COLBERT_V2_NDCG_10,
    DEFAULT_DATASETS,
    DEFAULT_MODEL,
    RANK_K,
    SHALLOW_K,
    TOP_K,
    _embed_corpus,
    _embed_queries,
    _ndcg_at_k,
    _recall_at_k,
)
from vstash.embed import get_embedding_dim
from vstash.store import VstashStore


DEFAULT_RERANK = (50, 100, 200)
DEFAULT_NPROBE = (0, 8, 16)  # 0 = library default (nlist // 8)


def _evaluate_ivfpq(
    dataset: str,
    rerank: int,
    nprobe: int,
    dim: int,
    ids: list[str],
    texts: list[str],
    vecs: np.ndarray,
    queries: dict[str, str],
    query_vecs: dict[str, np.ndarray],
    qrels: dict[str, dict[str, int]],
    tmp_dir: Path,
) -> dict:
    db_path = tmp_dir / f"{dataset}_ivfpq_r{rerank}_p{nprobe}.db"
    # SQLite extensions append to ``.db``; snapvec companion replaces
    # the extension (``foo.db`` -> ``foo.snpi``).
    for p in (
        db_path,
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
        db_path.with_suffix(".snpi"),
    ):
        if p.exists():
            p.unlink()

    store = VstashStore(
        str(db_path),
        embedding_dim=dim,
        vector_backend="snapvec-ivfpq",
        ivfpq_rerank_candidates=rerank,
        ivfpq_nprobe=nprobe,
    )

    t0 = time.perf_counter()
    docs = [
        {
            "path": f"{dataset}/{doc_id}",
            "title": doc_id,
            "chunks": [text],
            "embeddings": [vecs[i].tolist()],
            "source_type": "text",
        }
        for i, (doc_id, text) in enumerate(zip(ids, texts))
    ]
    store.add_documents_batch(docs)
    ingest_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit_stats = store.fit_ivfpq()
    fit_s = time.perf_counter() - t0

    snpi_bytes = 0
    snpi_path = db_path.with_suffix(".snpi")
    if snpi_path.exists():
        snpi_bytes = snpi_path.stat().st_size

    ndcgs_10: list[float] = []
    ndcgs_3: list[float] = []
    recalls_10: list[float] = []
    recalls_100: list[float] = []
    latencies: list[float] = []
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        q_vec = query_vecs[qid].tolist()
        t0 = time.perf_counter()
        results = store.search(q_vec, qtext, top_k=RANK_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [Path(r.path).name for r in results]
        ndcgs_10.append(_ndcg_at_k(ranked, qrels[qid], TOP_K))
        ndcgs_3.append(_ndcg_at_k(ranked, qrels[qid], SHALLOW_K))
        recalls_10.append(_recall_at_k(ranked, qrels[qid], TOP_K))
        recalls_100.append(_recall_at_k(ranked, qrels[qid], RANK_K))

    store.close()
    for p in (
        db_path,
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
        db_path.with_suffix(".snpi"),
    ):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    def _avg(xs: list[float]) -> float:
        return round(statistics.mean(xs), 4) if xs else 0.0

    return {
        "rerank_candidates": rerank,
        "nprobe": nprobe,
        "nlist": fit_stats["nlist"],
        "training_sample": fit_stats["training_sample"],
        "n_queries": len(latencies),
        "ndcg_at_10": _avg(ndcgs_10),
        "ndcg_at_3": _avg(ndcgs_3),
        "recall_at_10": _avg(recalls_10),
        "recall_at_100": _avg(recalls_100),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(
            statistics.quantiles(latencies, n=20, method="inclusive")[-1]
            if len(latencies) > 1
            else latencies[0],
            2,
        )
        if latencies
        else 0.0,
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "ingest_seconds": round(ingest_s, 2),
        "fit_seconds": round(fit_s, 2),
        "snpi_bytes": snpi_bytes,
    }


def _print_table(
    results: list[dict],
    datasets: list[str],
) -> None:
    print("\n### IVFPQ NDCG@10 by (rerank, nprobe) x dataset (v3, BEIR)\n")
    # Collect unique (rerank, nprobe) pairs in stable order.
    configs = []
    seen = set()
    for row in results:
        key = (row["rerank"], row["nprobe"])
        if key not in seen:
            seen.add(key)
            configs.append(key)

    header = ["Dataset", "BM25", "ColBERTv2"] + [f"r={r}/p={p}" for r, p in configs]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    index: dict[tuple[str, int, int], dict] = {}
    for row in results:
        index[(row["dataset"], row["rerank"], row["nprobe"])] = row["cell"]
    for ds in datasets:
        cells = [ds, f"{BM25_NDCG_10.get(ds, 0):.3f}", f"{COLBERT_V2_NDCG_10.get(ds, 0):.3f}"]
        for r, p in configs:
            c = index.get((ds, r, p))
            cells.append(f"{c['ndcg_at_10']:.4f}" if c else "-")
        print("| " + " | ".join(cells) + " |")

    # Macro per config.
    macro_cells = ["**macro**", "-", "-"]
    for r, p in configs:
        vals = [index[(ds, r, p)]["ndcg_at_10"] for ds in datasets if (ds, r, p) in index]
        m = sum(vals) / len(vals) if vals else 0.0
        macro_cells.append(f"**{m:.4f}**")
    print("| " + " | ".join(macro_cells) + " |")

    print("\n### Latency p50 (ms) by config x dataset\n")
    header = ["Dataset"] + [f"r={r}/p={p}" for r, p in configs]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for ds in datasets:
        cells = [ds]
        for r, p in configs:
            c = index.get((ds, r, p))
            cells.append(f"{c['latency_p50_ms']:.2f}" if c else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n### Full metrics per (config, dataset)\n")
    print(
        "| Dataset | rerank | nprobe | nlist | NDCG@10 | R@10 | R@100 | "
        "p50 ms | mean ms | fit s | snpi MB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for r, p in configs:
            c = index.get((ds, r, p))
            if c is None:
                continue
            print(
                f"| {ds} | {r} | {p} | {c['nlist']} | "
                f"{c['ndcg_at_10']:.4f} | {c['recall_at_10']:.4f} | {c['recall_at_100']:.4f} | "
                f"{c['latency_p50_ms']:.2f} | {c['latency_mean_ms']:.2f} | "
                f"{c['fit_seconds']:.2f} | {c['snpi_bytes'] / 1024 / 1024:.2f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--rerank",
        nargs="+",
        type=int,
        default=list(DEFAULT_RERANK),
        help="rerank_candidates values to sweep",
    )
    parser.add_argument(
        "--nprobe",
        nargs="+",
        type=int,
        default=list(DEFAULT_NPROBE),
        help="nprobe values to sweep (0 = library default)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/data/beir_snapvec_h2h_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/beir_snapvec_ivfpq_sweep.json"),
    )
    args = parser.parse_args()

    dim = get_embedding_dim(args.model)
    print(
        f"=== IVFPQ knob sweep ===\n"
        f"model={args.model} dim={dim} rerank={args.rerank} nprobe={args.nprobe}"
        f" datasets={args.datasets}"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_corpus: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    per_dataset_queries: dict[str, tuple[dict, dict]] = {}
    per_dataset_query_vecs: dict[str, dict[str, np.ndarray]] = {}
    print("\n[1/2] loading embeddings ...")
    for ds in args.datasets:
        ids, texts, vecs = _embed_corpus(ds, args.model, args.cache_dir)
        per_dataset_corpus[ds] = (ids, texts, vecs)
        cache = download_beir(ds)
        _, queries, qrels = load_beir(cache)
        per_dataset_queries[ds] = (queries, qrels)
        per_dataset_query_vecs[ds] = _embed_queries(ds, args.model, queries, args.cache_dir)

    rows: list[dict] = []
    print("\n[2/2] sweeping ...")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for rerank in args.rerank:
            for nprobe in args.nprobe:
                for ds in args.datasets:
                    ids, texts, vecs = per_dataset_corpus[ds]
                    queries, qrels = per_dataset_queries[ds]
                    print(f"  [r={rerank}/p={nprobe}/{ds}] ...", flush=True)
                    cell = _evaluate_ivfpq(
                        dataset=ds,
                        rerank=rerank,
                        nprobe=nprobe,
                        dim=dim,
                        ids=ids,
                        texts=texts,
                        vecs=vecs,
                        queries=queries,
                        query_vecs=per_dataset_query_vecs[ds],
                        qrels=qrels,
                        tmp_dir=tmp_dir,
                    )
                    rows.append(
                        {"dataset": ds, "rerank": rerank, "nprobe": nprobe, "cell": cell}
                    )
                    print(
                        f"    NDCG@10={cell['ndcg_at_10']:.4f}  "
                        f"R@100={cell['recall_at_100']:.4f}  "
                        f"p50={cell['latency_p50_ms']}ms  "
                        f"fit={cell['fit_seconds']}s",
                        flush=True,
                    )

    _print_table(rows, list(args.datasets))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "dim": dim,
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
