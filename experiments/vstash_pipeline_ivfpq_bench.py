"""Full-pipeline benchmark: vstash base (sqlite-vec) vs vstash-snapvec (ivfpq).

Unlike experiments/snapvec_backends_bench.py (which compares pure ANN
layers), this script exercises the full vstash search pipeline
(vector -> FTS5 -> adaptive RRF -> MMR) on both backends and reports
end-to-end latency + NDCG@10 on BEIR SciFact queries.

Pipeline:
    1. Ingest N=TARGET_N chunks into a temp store (SciFact corpus
       padded with FIQA to the target size) using pre-cached BGE
       embeddings. Single pass.
    2. Copy the temp store, reopen one copy as ``snapvec-ivfpq``,
       run ``fit_ivfpq`` on it.
    3. For each SciFact test query, call ``store.search(qe, qtext)``
       on both stores, recording wall latency + NDCG@10.
    4. Print the side-by-side comparison.

Usage:
    python -m experiments.vstash_pipeline_ivfpq_bench
    python -m experiments.vstash_pipeline_ivfpq_bench --n 10000
    python -m experiments.vstash_pipeline_ivfpq_bench --n 50000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.beir_benchmark import download_beir, load_beir
from experiments.snapvec_backends_bench import embed_dataset
from vstash.embed import embed_query
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
TOP_K = 10


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal, k)
    return _dcg(gains, k) / idcg if idcg > 0 else 0.0


_PAD_DATASETS: tuple[str, ...] = ("fiqa", "scidocs", "arguana", "nfcorpus")


def _load_dataset_for_padding(name: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (vecs, ids, texts) for a padding BEIR dataset."""
    vecs, ids = embed_dataset(name)
    corpus = load_beir(download_beir(name))[0]
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in ids]
    return vecs, ids, texts


def build_corpus(target_n: int) -> tuple[list[str], list[str], list[list[float]]]:
    """Returns (doc_ids, texts, embeddings) aligned row-wise.

    Anchor dataset is SciFact (so its qrel ids map directly onto the
    first 5k rows). Padding datasets are pulled in a fixed order
    (fiqa, scidocs, arguana, nfcorpus) and prefixed to avoid id
    collisions with SciFact's eval set. Embeddings for each padding
    dataset are cached on first use via embed_dataset().
    """
    sci_vecs, sci_ids = embed_dataset("scifact")
    sci_corpus = load_beir(download_beir("scifact"))[0]
    sci_texts = [
        (sci_corpus[d].get("title", "") + " " + sci_corpus[d].get("text", "")).strip()
        for d in sci_ids
    ]

    if target_n <= len(sci_ids):
        return sci_ids[:target_n], sci_texts[:target_n], sci_vecs[:target_n].tolist()

    ids: list[str] = list(sci_ids)
    texts: list[str] = list(sci_texts)
    vec_parts: list[np.ndarray] = [sci_vecs]

    for name in _PAD_DATASETS:
        if len(ids) >= target_n:
            break
        pad_vecs, pad_ids, pad_texts = _load_dataset_for_padding(name)
        take = min(len(pad_ids), target_n - len(ids))
        ids.extend(f"{name}_{i}" for i in pad_ids[:take])
        texts.extend(pad_texts[:take])
        vec_parts.append(pad_vecs[:take])

    if len(ids) < target_n:
        total = len(ids)
        raise RuntimeError(f"target_n={target_n} too large; SciFact + {_PAD_DATASETS} = {total}")

    vecs = np.concatenate(vec_parts, axis=0).tolist()
    return ids, texts, vecs


def ingest(db_path: str, ids: list[str], texts: list[str], vecs: list[list[float]]) -> None:
    store = VstashStore(db_path, embedding_dim=DIM, vector_backend="sqlite-vec")
    t0 = time.perf_counter()
    for i, (doc_id, text, emb) in enumerate(zip(ids, texts, vecs)):
        store.add_document(
            path=f"bench/{doc_id}",
            title=doc_id,
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )
        if (i + 1) % 5000 == 0:
            print(f"    ingested {i + 1}/{len(ids)}")
    print(f"    ingest done in {time.perf_counter() - t0:.1f}s")
    store.close()


def evaluate(
    db_path: str, backend: str, queries: dict, qrels: dict, doc_id_to_path: dict[str, str]
) -> dict:
    kwargs = {
        "embedding_dim": DIM,
        "vector_backend": backend,
    }
    if backend == "snapvec-ivfpq":
        kwargs["ivfpq_rerank_candidates"] = 100
    store = VstashStore(db_path, **kwargs)

    # fit is idempotent/checked inside; only runs when requested
    if backend == "snapvec-ivfpq" and not store._snap.fitted:
        print("    running fit_ivfpq ...")
        t0 = time.perf_counter()
        info = store.fit_ivfpq()
        print(
            f"    fit done in {time.perf_counter() - t0:.1f}s "
            f"(nlist={info['nlist']}, n={info['n_indexed']})"
        )

    ndcgs, latencies = [], []
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        qe = embed_query(qtext, MODEL)
        t0 = time.perf_counter()
        results = store.search(qe, qtext, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        # Recover original doc ids from path ("bench/<doc_id>")
        ranked = [Path(r.path).name for r in results]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))
    store.close()
    _ = doc_id_to_path  # reserved for future variations
    return {
        "backend": backend,
        "n_queries": len(latencies),
        "ndcg_at_10": round(statistics.mean(ndcgs), 4),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(statistics.quantiles(latencies, n=20, method="inclusive")[-1], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000, help="target corpus size (chunks)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to experiments/results/pipeline_ivfpq_bench_n{N}.json "
        "so re-runs at different N don't overwrite each other.",
    )
    args = parser.parse_args()

    print(f"=== vstash full-pipeline benchmark, N={args.n} ===")
    print("\n[1/3] building corpus ...")
    ids, texts, vecs = build_corpus(args.n)
    print(f"    {len(ids)} docs ready")

    sci_cache = download_beir("scifact")
    _, queries, qrels = load_beir(sci_cache)
    print(f"    {len(queries)} queries, {len(qrels)} qrels")

    with tempfile.TemporaryDirectory() as td:
        base_db = str(Path(td) / "base.db")
        ivfpq_db = str(Path(td) / "ivfpq.db")

        print(f"\n[2/3] ingesting into {base_db} ...")
        ingest(base_db, ids, texts, vecs)

        # Copy the ingested DB so both backends see identical chunks
        shutil.copy2(base_db, ivfpq_db)
        for ext in ("-wal", "-shm"):
            src = Path(base_db + ext)
            if src.exists():
                shutil.copy2(src, ivfpq_db + ext)

        print("\n[3/3] evaluating ...")
        print("  sqlite-vec baseline ...")
        base_stats = evaluate(base_db, "sqlite-vec", queries, qrels, {})

        print("  snapvec-ivfpq ...")
        ivfpq_stats = evaluate(ivfpq_db, "snapvec-ivfpq", queries, qrels, {})

    out_path = args.output or f"experiments/results/pipeline_ivfpq_bench_n{args.n}.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"n": args.n, "model": MODEL, "base": base_stats, "ivfpq": ivfpq_stats}
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'metric':<24}{'sqlite-vec':>16}{'snapvec-ivfpq':>16}{'delta':>14}")
    print("-" * 72)
    rows = [
        ("NDCG@10", base_stats["ndcg_at_10"], ivfpq_stats["ndcg_at_10"]),
        ("latency p50 (ms)", base_stats["latency_p50_ms"], ivfpq_stats["latency_p50_ms"]),
        ("latency p95 (ms)", base_stats["latency_p95_ms"], ivfpq_stats["latency_p95_ms"]),
        ("latency mean (ms)", base_stats["latency_mean_ms"], ivfpq_stats["latency_mean_ms"]),
    ]
    for name, a, b in rows:
        if "latency" in name and a > 0:
            delta = f"{b / a:.2f}x"
        else:
            delta = f"{b - a:+.4f}"
        print(f"{name:<24}{a:>16}{b:>16}{delta:>14}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
