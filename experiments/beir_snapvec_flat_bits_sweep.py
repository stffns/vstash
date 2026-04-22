"""
beir_snapvec_flat_bits_sweep.py -- A1-bits.

Flat snapvec has no learnable codebook (deterministic polar quantization),
so the only knob that behaves like a "fit" on v3 embeddings is the
``bits`` parameter on ``snapvec.SnapIndex``. The library restricts
``bits`` to {2, 3, 4}, so the full sweep is exactly those three values.
Measure NDCG@10 + Recall@100 + latency on the 5 BEIR datasets we
already publish in README. Baseline (bits=4) matches the h2h numbers
shipped in PR #254; the sweep tells us whether dropping bits costs
recall and how much disk a bit-drop saves.

Reuses the on-disk embedding cache from ``beir_snapvec_h2h`` so we
don't re-encode the corpora with the v3 model.

Usage:

    python -m experiments.beir_snapvec_flat_bits_sweep
    python -m experiments.beir_snapvec_flat_bits_sweep --bits 2 3 4
    python -m experiments.beir_snapvec_flat_bits_sweep --datasets scifact nfcorpus
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


DEFAULT_BITS = (2, 3, 4)


def _evaluate_bits(
    dataset: str,
    bits: int,
    dim: int,
    ids: list[str],
    texts: list[str],
    vecs: np.ndarray,
    queries: dict[str, str],
    query_vecs: dict[str, np.ndarray],
    qrels: dict[str, dict[str, int]],
    tmp_dir: Path,
) -> dict:
    db_path = tmp_dir / f"{dataset}_snapvec_b{bits}.db"
    # SQLite writes ``.db`` + ``.db-wal`` + ``.db-shm`` (extensions append).
    # Snapvec companion REPLACES the extension: ``foo.db`` -> ``foo.snpv``.
    for p in (
        db_path,
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
        db_path.with_suffix(".snpv"),
    ):
        if p.exists():
            p.unlink()

    store = VstashStore(
        str(db_path),
        embedding_dim=dim,
        vector_backend="snapvec",
        snapvec_bits=bits,
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

    snpv_bytes = 0
    snpv_path = db_path.with_suffix(".snpv")
    if snpv_path.exists():
        snpv_bytes = snpv_path.stat().st_size

    for p in (
        db_path,
        db_path.with_suffix(db_path.suffix + "-wal"),
        db_path.with_suffix(db_path.suffix + "-shm"),
        db_path.with_suffix(".snpv"),
    ):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    def _avg(xs: list[float]) -> float:
        return round(statistics.mean(xs), 4) if xs else 0.0

    return {
        "bits": bits,
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
        "snpv_bytes": snpv_bytes,
    }


def _print_table(results: dict, bits_list: list[int], datasets: list[str]) -> None:
    print("\n### Flat snapvec NDCG@10 by bits (v3, BEIR)\n")
    header = ["Dataset", "BM25", "ColBERTv2"] + [f"bits={b}" for b in bits_list]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for ds in datasets:
        cells = [ds, f"{BM25_NDCG_10.get(ds, 0):.3f}", f"{COLBERT_V2_NDCG_10.get(ds, 0):.3f}"]
        for b in bits_list:
            v = results.get(b, {}).get(ds, {}).get("ndcg_at_10")
            cells.append(f"{v:.4f}" if v is not None else "-")
        print("| " + " | ".join(cells) + " |")

    macros: dict[int, float] = {}
    for b in bits_list:
        vals = [results[b][ds]["ndcg_at_10"] for ds in datasets if ds in results.get(b, {})]
        macros[b] = sum(vals) / len(vals) if vals else 0.0
    macro_cells = ["**macro**", "-", "-"] + [f"**{macros[b]:.4f}**" for b in bits_list]
    print("| " + " | ".join(macro_cells) + " |")

    print("\n### Recall@100 by bits (candidate-set health)\n")
    header = ["Dataset"] + [f"bits={b}" for b in bits_list]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for ds in datasets:
        cells = [ds]
        for b in bits_list:
            v = results.get(b, {}).get(ds, {}).get("recall_at_100")
            cells.append(f"{v:.4f}" if v is not None else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n### Storage + latency per (bits, dataset)\n")
    print("| Dataset | bits | NDCG@10 | Recall@100 | p50 ms | mean ms | ingest s | snpv MB |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for b in bits_list:
            r = results.get(b, {}).get(ds, {})
            if not r:
                continue
            print(
                f"| {ds} | {b} | "
                f"{r['ndcg_at_10']:.4f} | {r['recall_at_100']:.4f} | "
                f"{r['latency_p50_ms']:.2f} | {r['latency_mean_ms']:.2f} | "
                f"{r['ingest_seconds']:.2f} | {r['snpv_bytes'] / 1024 / 1024:.2f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bits", nargs="+", type=int, default=list(DEFAULT_BITS))
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
        default=Path("experiments/results/beir_snapvec_flat_bits_sweep.json"),
    )
    args = parser.parse_args()

    dim = get_embedding_dim(args.model)
    print(
        f"=== flat snapvec bits sweep ===\n"
        f"model={args.model} dim={dim} bits={args.bits} datasets={args.datasets}"
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

    results: dict[int, dict[str, dict]] = {}
    print("\n[2/2] sweeping bits ...")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for bits in args.bits:
            results[bits] = {}
            for ds in args.datasets:
                ids, texts, vecs = per_dataset_corpus[ds]
                queries, qrels = per_dataset_queries[ds]
                print(f"  [bits={bits}/{ds}] ...", flush=True)
                cell = _evaluate_bits(
                    dataset=ds,
                    bits=bits,
                    dim=dim,
                    ids=ids,
                    texts=texts,
                    vecs=vecs,
                    queries=queries,
                    query_vecs=per_dataset_query_vecs[ds],
                    qrels=qrels,
                    tmp_dir=tmp_dir,
                )
                results[bits][ds] = cell
                print(
                    f"    NDCG@10={cell['ndcg_at_10']:.4f}  "
                    f"R@100={cell['recall_at_100']:.4f}  "
                    f"p50={cell['latency_p50_ms']}ms  "
                    f"snpv={cell['snpv_bytes'] / 1024 / 1024:.1f}MB",
                    flush=True,
                )

    _print_table(results, list(args.bits), list(args.datasets))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "dim": dim,
                "bits_sweep": {str(b): results[b] for b in args.bits},
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
