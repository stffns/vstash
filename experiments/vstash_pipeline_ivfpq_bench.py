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
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.beir_benchmark import download_beir, load_beir
from experiments.snapvec_backends_bench import embed_dataset
from vstash.embed import embed_query
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"  # overridable via --model
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
        # Run out of real BEIR datasets (total ~99k). For scale-up
        # benchmarks (N >= 500k / 1M) we pad with synthetic
        # L2-normalised Gaussian vectors. NDCG is still measurable
        # because the real SciFact corpus (and its relevant docs)
        # stays at the front of the list; the synthetic tail acts as
        # distractor noise the retrieval system has to reject.
        missing = target_n - len(ids)
        print(
            f"    synthetic padding: {missing} random unit-norm vectors "
            f"(real BEIR corpora exhausted at N={len(ids)})"
        )
        rng = np.random.default_rng(seed=0xBEEF)
        dim = vec_parts[0].shape[1]
        raw = rng.standard_normal((missing, dim)).astype(np.float32)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-12)
        ids.extend(f"synth_{i}" for i in range(missing))
        texts.extend(f"synthetic distractor document number {i}" for i in range(missing))
        vec_parts.append(raw)

    vecs = np.concatenate(vec_parts, axis=0).tolist()
    return ids, texts, vecs


def ingest(
    db_path: str,
    ids: list[str],
    texts: list[str],
    vecs: list[list[float]],
    vector_backend: str,
    dim: int,
) -> float:
    """Ingest into a fresh store using ``vector_backend``.

    Per-backend ingest matters because ``snapvec`` flat populates its
    ``.snpv`` index during the write path (no ``fit_snapvec`` post-
    ingest step analogous to ``fit_ivfpq``). Copying a
    sqlite-vec-ingested db and reopening with ``vector_backend="snapvec"``
    does NOT build the flat index and silently falls back to sqlite-vec
    semantics (caught in PR #249 review).

    We ingest via ``add_documents_batch`` instead of a per-doc
    ``add_document`` loop. The per-doc loop is O(N^2) on disk I/O
    for the flat ``snapvec`` backend: ``_save_snapvec`` is called
    after every commit and rewrites the entire ``.snpv`` file, so
    at N=100k it writes ~1 TB over the run (~40 minutes wall-clock
    on NVMe). ``add_documents_batch`` calls ``_save_snapvec`` once
    after the whole batch, which is the intended code path at
    scale. Not a benchmark artefact -- any real vstash user
    ingesting tens of thousands of docs with the flat snapvec
    backend via per-doc add_document is hitting the same wall.
    """
    store = VstashStore(db_path, embedding_dim=dim, vector_backend=vector_backend)
    t0 = time.perf_counter()
    docs = [
        {
            "path": f"bench/{doc_id}",
            "title": doc_id,
            "chunks": [text],
            "embeddings": [emb],
            "source_type": "text",
        }
        for doc_id, text, emb in zip(ids, texts, vecs)
    ]
    store.add_documents_batch(docs)
    elapsed = time.perf_counter() - t0
    print(f"    ingest done in {elapsed:.1f}s  ({len(docs)} docs)")
    store.close()
    return elapsed


def evaluate(
    db_path: str,
    backend: str,
    queries: dict,
    qrels: dict,
    doc_id_to_path: dict[str, str],
    model: str,
    dim: int,
) -> dict:
    kwargs = {
        "embedding_dim": dim,
        "vector_backend": backend,
    }
    if backend == "snapvec-ivfpq":
        kwargs["ivfpq_rerank_candidates"] = 100
    store = VstashStore(db_path, **kwargs)

    # fit is idempotent/checked inside; only runs when requested
    fit_s: float | None = None
    if backend == "snapvec-ivfpq" and not store._snap.fitted:
        print("    running fit_ivfpq ...")
        t0 = time.perf_counter()
        info = store.fit_ivfpq()
        fit_s = time.perf_counter() - t0
        print(f"    fit done in {fit_s:.1f}s (nlist={info['nlist']}, n={info['n_indexed']})")

    ndcgs, latencies = [], []
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        qe = embed_query(qtext, model)
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
        "fit_seconds": round(fit_s, 2) if fit_s is not None else None,
    }


_ALL_BACKENDS = ("sqlite-vec", "snapvec", "snapvec-ivfpq")


def main() -> None:
    from vstash.embed import get_embedding_dim

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000, help="target corpus size (chunks)")
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Embedding model (HF name or local path). Default: BAAI/bge-small-en-v1.5.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(_ALL_BACKENDS),
        choices=list(_ALL_BACKENDS),
        help="Which vector backends to bench. Default: all three.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to experiments/results/pipeline_ivfpq_bench_n{N}.json "
        "so re-runs at different N don't overwrite each other.",
    )
    args = parser.parse_args()

    dim = get_embedding_dim(args.model)
    print(
        f"=== vstash full-pipeline benchmark, N={args.n}, model={args.model}, dim={dim}, "
        f"backends={args.backends} ==="
    )
    print("\n[1/3] building corpus ...")
    ids, texts, vecs = build_corpus(args.n)
    print(f"    {len(ids)} docs ready")

    # Guard against model/corpus dim mismatch. ``build_corpus`` pulls
    # embeddings from ``embed_dataset`` which is pinned to
    # ``BAAI/bge-small-en-v1.5`` (384d). Mixing a different model would
    # silently feed wrong-dimensional vectors to the store, not worth
    # the confusion.
    if vecs and len(vecs[0]) != dim:
        raise RuntimeError(
            f"Corpus embeddings are {len(vecs[0])}d but --model resolves to {dim}d. "
            f"The cached embeddings in experiments/snapvec_backends_bench live only for "
            f"BAAI/bge-small-en-v1.5 right now; passing --model with a different dim "
            f"is not yet supported. Stick to 384d models."
        )

    sci_cache = download_beir("scifact")
    _, queries, qrels = load_beir(sci_cache)
    print(f"    {len(queries)} queries, {len(qrels)} qrels")

    # Per-backend ingest. snapvec flat populates its .snpv index at
    # ingest time, so each backend needs its own from-scratch ingest
    # (copying a sqlite-vec-ingested db silently falls back to
    # sqlite-vec semantics for the snapvec column, caught in PR #249
    # review).
    with tempfile.TemporaryDirectory() as td:
        per_backend: dict[str, dict] = {}
        for backend in args.backends:
            bdb = str(Path(td) / f"{backend}.db")
            print(f"\n[2/3] ingesting {backend} into {bdb} ...")
            ingest_s = ingest(bdb, ids, texts, vecs, vector_backend=backend, dim=dim)
            print(f"\n[3/3] evaluating {backend} ...")
            stats = evaluate(bdb, backend, queries, qrels, {}, model=args.model, dim=dim)
            stats["ingest_seconds"] = round(ingest_s, 2)
            per_backend[backend] = stats

    out_path = args.output or f"experiments/results/pipeline_ivfpq_bench_n{args.n}.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "n": args.n,
        "model": args.model,
        "dim": dim,
        "backends": per_backend,
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Printable table.
    print("\n" + "=" * (24 + 16 * len(args.backends)))
    header = f"{'metric':<24}" + "".join(f"{b:>16}" for b in args.backends)
    print(header)
    print("-" * len(header))
    metric_keys = [
        ("NDCG@10", "ndcg_at_10"),
        ("latency p50 (ms)", "latency_p50_ms"),
        ("latency p95 (ms)", "latency_p95_ms"),
        ("latency mean (ms)", "latency_mean_ms"),
        ("ingest (s)", "ingest_seconds"),
        ("fit (s)", "fit_seconds"),
    ]
    for label, key in metric_keys:
        row = f"{label:<24}"
        for b in args.backends:
            val = per_backend[b].get(key)
            row += f"{'-' if val is None else val:>16}"
        print(row)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
