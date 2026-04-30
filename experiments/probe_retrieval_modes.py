"""Probe: retrieval mode coverage matrix (hybrid / vec_only / fts_only).

The pipeline benchmark only exercises hybrid mode. This probe sweeps
all three retrieval modes against each vector backend at a fixed
corpus size and reports the latency + NDCG@10 matrix. Used post
Sprint 2 to confirm:

- All three modes work on all three backends (no regression from
  the services / vectorbackend / embed-registry refactors).
- The v0.36 vec_only long-query distance-cutoff fix (#304) is still
  live (a regression would crater ArguAna-style NDCG to ~0.001).
- Mode dispatch in store.search and the new search service share
  the same mode resolution.

Usage:
    python -m experiments.probe_retrieval_modes --n 10000
    python -m experiments.probe_retrieval_modes --n 50000 \\
        --backends sqlite-vec snapvec snapvec-ivfpq \\
        --modes hybrid vec_only fts_only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path


from experiments.beir_benchmark import download_beir, load_beir
from experiments.snapvec_backends_bench import embed_dataset
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return _dcg(gains, k) / idcg


def _build_corpus(n: int) -> tuple[list[dict], dict, dict]:
    """SciFact + FIQA padding to N chunks, returns (docs, queries, qrels).

    Same shape as vstash_pipeline_ivfpq_bench: SciFact provides the
    queries / qrels (300 judged); FIQA pads with real BGE embeddings
    so the corpus density matches a realistic retrieval workload.

    Uses :func:`embed_dataset` from the snapvec backends bench, which
    caches BGE embeddings on disk -- second-run is near-instant.

    Raises:
        ValueError: if ``n`` is smaller than the SciFact corpus
            (5183 docs). Truncating below this would drop relevant
            documents that qrels still reference, making NDCG@10
            meaningless. The caller can pass ``n=5183`` to test on
            SciFact alone or ``n>=10000`` for realistic padding.
    """
    sf_cache = download_beir("scifact")
    sf_corpus, sf_queries, sf_qrels = load_beir(sf_cache)
    sf_vecs, sf_doc_ids = embed_dataset("scifact")
    if n < len(sf_doc_ids):
        raise ValueError(
            f"n={n} is smaller than the SciFact corpus ({len(sf_doc_ids)} docs). "
            f"Truncating below this drops relevant docs that qrels still "
            f"reference, making NDCG@10 incomparable to the published baseline. "
            f"Use n>={len(sf_doc_ids)} (no padding) or n>=10000 (with FIQA padding)."
        )

    docs: list[dict] = []
    for did, vec in zip(sf_doc_ids, sf_vecs, strict=True):
        text = (sf_corpus[did].get("title", "") + "\n" + sf_corpus[did].get("text", "")).strip()[
            :2048
        ]
        docs.append(
            {
                "path": f"/scifact/{did}",
                "title": sf_corpus[did].get("title", did),
                "chunks": [text],
                "embeddings": [vec.tolist()],
                "source_type": "text",
                "collection": "scifact",
            }
        )

    if len(docs) < n:
        fq_cache = download_beir("fiqa")
        fq_corpus, _, _ = load_beir(fq_cache)
        fq_vecs, fq_doc_ids = embed_dataset("fiqa")
        needed = n - len(docs)
        for did, vec in zip(fq_doc_ids[:needed], fq_vecs[:needed], strict=True):
            text = (
                fq_corpus[did].get("title", "") + "\n" + fq_corpus[did].get("text", "")
            ).strip()[:2048]
            docs.append(
                {
                    "path": f"/fiqa/{did}",
                    "title": fq_corpus[did].get("title", did),
                    "chunks": [text],
                    "embeddings": [vec.tolist()],
                    "source_type": "text",
                    "collection": "fiqa",
                }
            )

    return docs[:n], sf_queries, sf_qrels


def _ingest(docs: list[dict], db_path: str, dim: int, backend: str) -> VstashStore:
    store = VstashStore(db_path, embedding_dim=dim, vector_backend=backend, snapvec_bits=4)
    with store.batch_mode(defer_fts=True):
        store.add_documents_batch(docs)
    if backend == "snapvec-ivfpq" and not store._snap.fitted:
        store.fit_ivfpq(training_sample=min(50000, len(docs)))
    return store


def _eval_mode(store: VstashStore, queries: dict, qrels: dict, mode: str) -> dict:
    latencies: list[float] = []
    ndcgs: list[float] = []
    queries_with_qrel = [(qid, qtext) for qid, qtext in queries.items() if qid in qrels]
    if not queries_with_qrel:
        # Empty qrels (or empty intersection with queries) means the
        # caller built a corpus that has no measurable retrieval. Return
        # zeros rather than crashing in statistics.median([]).
        return {
            "n_queries": 0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
            "ndcg10_mean": 0.0,
            "ndcg10_zero_count": 0,
        }

    # Warmup: up to 3 untimed queries.
    for qid, qtext in queries_with_qrel[:3]:
        qe = embed_query(qtext, MODEL)
        store.search(query_embedding=qe, query_text=qtext, top_k=TOP_K, retrieval_mode=mode)

    for qid, qtext in queries_with_qrel:
        qe = embed_query(qtext, MODEL)
        t0 = time.perf_counter()
        results = store.search(
            query_embedding=qe, query_text=qtext, top_k=TOP_K, retrieval_mode=mode
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        # Map result paths back to SciFact doc ids for NDCG.
        ranked = []
        for r in results:
            if r.path.startswith("/scifact/"):
                ranked.append(r.path.split("/")[-1])
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p95_idx = min(n - 1, max(0, round(0.95 * (n - 1))))
    return {
        "n_queries": n,
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(sorted_lat[p95_idx], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "ndcg10_mean": round(statistics.mean(ndcgs), 4),
        "ndcg10_zero_count": sum(1 for x in ndcgs if x == 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000, help="target corpus size")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sqlite-vec", "snapvec", "snapvec-ivfpq"],
        choices=["sqlite-vec", "snapvec", "snapvec-ivfpq"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid", "vec_only", "fts_only"],
        choices=["hybrid", "vec_only", "fts_only"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"=== retrieval mode probe N={args.n} backends={args.backends} modes={args.modes} ===")
    print()

    print("[1/3] building corpus ...")
    docs, queries, qrels = _build_corpus(args.n)
    qrel_count = sum(1 for qid in queries if qid in qrels)
    print(f"    {len(docs)} docs ready; {qrel_count} judged queries")

    dim = get_embedding_dim(MODEL)

    out: dict = {
        "n": args.n,
        "model": MODEL,
        "results": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        for backend in args.backends:
            print(f"\n[2/3] ingesting {backend} ...")
            db = f"{tmp}/{backend.replace('-', '_')}.db"
            t0 = time.perf_counter()
            store = _ingest(docs, db, dim, backend)
            ingest_s = time.perf_counter() - t0
            print(f"    ingest {ingest_s:.1f}s")
            out["results"].setdefault(backend, {})

            for mode in args.modes:
                print(f"  [3/3] {backend} x {mode} ...", end=" ", flush=True)
                stats = _eval_mode(store, queries, qrels, mode)
                out["results"][backend][mode] = stats
                print(
                    f"p50={stats['latency_p50_ms']}ms "
                    f"p95={stats['latency_p95_ms']}ms "
                    f"NDCG={stats['ndcg10_mean']}"
                )
            store.close()

    print("\n" + "=" * 76)
    print(f"{'backend':>16} {'mode':>10} {'p50':>8} {'p95':>8} {'mean':>8} {'NDCG':>8} {'zero':>6}")
    print("-" * 76)
    for backend in args.backends:
        for mode in args.modes:
            s = out["results"][backend][mode]
            print(
                f"{backend:>16} {mode:>10} "
                f"{s['latency_p50_ms']:>7.1f}ms "
                f"{s['latency_p95_ms']:>7.1f}ms "
                f"{s['latency_mean_ms']:>7.1f}ms "
                f"{s['ndcg10_mean']:>8.4f} {s['ndcg10_zero_count']:>6}"
            )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
