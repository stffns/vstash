"""BEIR Benchmark — vstash vs published baselines and Chroma.

Downloads BEIR datasets on first run and caches them locally.
Evaluates vstash hybrid RRF against published baselines (BM25, dense-only,
ColBERTv2) and optionally against Chroma head-to-head.

Usage:
    python -m experiments.beir_benchmark
    python -m experiments.beir_benchmark --datasets scifact nfcorpus
    python -m experiments.beir_benchmark --no-chroma
    python -m experiments.beir_benchmark --model BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import statistics
import tempfile
import time
import urllib.request
import zipfile

from vstash.embed import (
    embed_query,
    embed_texts,
    get_embedding_dim,
    register_encoder_resolver,
    unregister_encoder_resolver,
)
from vstash.store import VstashStore


# ------------------------------------------------------------------ #
# Optional CUDA encoder resolver (used by Colab T4 runs).             #
# ------------------------------------------------------------------ #
# FastEmbed (vstash's default embedder) has no CUDA wheel, so when
# this script runs on Colab it pegs the shared CPU and FiQA's 57k-doc
# embed step takes hours.  --device cuda routes every embed_texts /
# embed_query call through SentenceTransformer pinned to T4 instead,
# matching what we did in experiments/longmemeval_retrieval.py.

_st_cache: dict = {}


class _STEncoder:
    """Minimal SentenceTransformer adapter to vstash's Encoder protocol."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._m = (
            SentenceTransformer(model_name, device=device)
            if device
            else SentenceTransformer(model_name)
        )
        self.embedding_dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]):
        return self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _make_st_resolver(device: str):
    """Return a resolver that claims any model_name and serves it via ST on ``device``."""

    def resolver(model_name: str):
        if model_name not in _st_cache:
            _st_cache[model_name] = _STEncoder(model_name, device=device)
        return _st_cache[model_name]

    return resolver


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10
CACHE_DIR = "experiments/data"

DATASETS = {
    "scifact": {"docs": 5183, "max_corpus": 60000},
    "nfcorpus": {"docs": 3633, "max_corpus": 60000},
    "fiqa": {"docs": 57638, "max_corpus": 60000},
    "scidocs": {"docs": 25657, "max_corpus": 60000},
    "arguana": {"docs": 8674, "max_corpus": 60000},
}

# Published NDCG@10 baselines (MTEB leaderboard / BEIR paper / ColBERTv2 paper)
BASELINES = {
    "scifact": {"BM25": 0.665, "BGE-small dense": 0.653, "ColBERTv2": 0.693},
    "nfcorpus": {"BM25": 0.325, "BGE-small dense": 0.338, "ColBERTv2": 0.344},
    "fiqa": {"BM25": 0.236, "BGE-small dense": 0.402, "ColBERTv2": 0.356},
    "scidocs": {"BM25": 0.158, "BGE-small dense": 0.163, "ColBERTv2": 0.154},
    "arguana": {"BM25": 0.315, "BGE-small dense": 0.584, "ColBERTv2": 0.463},
}


# ------------------------------------------------------------------ #
# BEIR data loading                                                    #
# ------------------------------------------------------------------ #


def download_beir(name: str) -> str:
    """Download and cache a BEIR dataset. Returns path to cache dir."""
    # Ensure the cache parent exists. CACHE_DIR is a gitignored path
    # inside the repo, so on a fresh clone it does not exist yet.
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/beir_{name}"
    if not os.path.exists(cache):
        print(f"  Downloading {name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        data = urllib.request.urlopen(url).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        raw_dir = f"{CACHE_DIR}/beir_{name}_raw"
        z.extractall(raw_dir)
        os.rename(f"{raw_dir}/{name}", cache)
        # Clean up raw dir
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
    return cache


def load_beir(cache: str) -> tuple[dict, dict, dict]:
    """Load corpus, queries, and qrels from a BEIR cache directory."""
    corpus: dict[str, dict] = {}
    with open(f"{cache}/corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    queries: dict[str, str] = {}
    with open(f"{cache}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(f"{cache}/qrels/test.tsv") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score

    return corpus, queries, qrels


# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal, k)
    return _dcg(gains, k) / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: list[str], qrel: dict[str, int]) -> float:
    for i, did in enumerate(ranked_ids):
        if qrel.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    relevant = {did for did, s in qrel.items() if s > 0}
    if not relevant:
        return 0.0
    return sum(1 for did in ranked_ids[:k] if did in relevant) / len(relevant)


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #


def evaluate_vstash(
    store: VstashStore,
    doc_id_map: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    model_id: str,
    retrieval_mode: str = "hybrid",
) -> dict:
    """Evaluate vstash on a BEIR dataset."""
    test_qids = list(qrels.keys())
    ndcgs, mrrs, recalls, latencies = [], [], [], []
    per_query: list[dict] = []

    for qid in test_qids:
        qe = embed_query(queries[qid], model_id)
        t0 = time.perf_counter()
        results = store.search(qe, queries[qid], top_k=TOP_K, retrieval_mode=retrieval_mode)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        ranked = [doc_id_map.get(r.path, "") for r in results]
        q_ndcg = ndcg_at_k(ranked, qrels[qid], TOP_K)
        q_mrr = mrr(ranked, qrels[qid])
        q_recall = recall_at_k(ranked, qrels[qid], TOP_K)
        ndcgs.append(q_ndcg)
        mrrs.append(q_mrr)
        recalls.append(q_recall)
        per_query.append(
            {
                "qid": qid,
                "ndcg_10": round(q_ndcg, 6),
                "mrr": round(q_mrr, 6),
                "recall_10": round(q_recall, 6),
            }
        )

    return {
        "ndcg_10": round(statistics.mean(ndcgs), 4),
        "mrr": round(statistics.mean(mrrs), 4),
        "recall_10": round(statistics.mean(recalls), 4),
        "latency_ms": round(statistics.mean(latencies), 1),
        "per_query": per_query,
    }


def evaluate_chroma(
    embeddings: list[list[float]],
    doc_ids: list[str],
    doc_texts: list[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    model_id: str,
) -> dict:
    """Evaluate Chroma on a BEIR dataset."""
    import chromadb

    chroma_dir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.create_collection(name="bench", metadata={"hnsw:space": "cosine"})

    t0 = time.perf_counter()
    for bs in range(0, len(doc_ids), 4000):
        end = min(bs + 4000, len(doc_ids))
        col.add(
            ids=doc_ids[bs:end],
            embeddings=embeddings[bs:end],
            documents=doc_texts[bs:end],
        )
    ingest_time = time.perf_counter() - t0
    print(f"  [Chroma] Ingested in {ingest_time:.1f}s")

    test_qids = list(qrels.keys())
    ndcgs, mrrs, recalls, latencies = [], [], [], []

    for qid in test_qids:
        qe = embed_query(queries[qid], model_id)
        t0 = time.perf_counter()
        results = col.query(query_embeddings=[qe], n_results=TOP_K)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        ranked = results["ids"][0]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))
        mrrs.append(mrr(ranked, qrels[qid]))
        recalls.append(recall_at_k(ranked, qrels[qid], TOP_K))

    shutil.rmtree(chroma_dir)

    return {
        "ndcg_10": round(statistics.mean(ndcgs), 4),
        "mrr": round(statistics.mean(mrrs), 4),
        "recall_10": round(statistics.mean(recalls), 4),
        "latency_ms": round(statistics.mean(latencies), 1),
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="BEIR benchmark for vstash")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vec_only", "fts_only"],
        default="hybrid",
        help="Retrieval mode: hybrid (default, RRF over vector + FTS5), "
        "vec_only (semantic-only, skip FTS5), fts_only (keyword-only, skip vector). "
        "Used to isolate substrate contribution in apples-to-apples evaluations.",
    )
    parser.add_argument("--no-chroma", action="store_true", help="Skip Chroma comparison")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Pin the embedder to a device.  --device cuda registers a "
        "SentenceTransformer-on-CUDA resolver so the embed step uses "
        "T4 instead of FastEmbed CPU (FastEmbed has no CUDA wheel).  "
        "On the local Mac the FastEmbed default is faster -- leave unset.",
    )
    args = parser.parse_args()

    model_id = args.model
    resolver = None
    if args.device in ("cuda", "cpu"):
        resolver = _make_st_resolver(args.device)
        register_encoder_resolver(resolver)
        print(f"  Registered SentenceTransformer-{args.device.upper()} resolver for {model_id}")
    dim = get_embedding_dim(model_id)
    run_chroma = not args.no_chroma

    if run_chroma:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            print("  chromadb not installed — skipping Chroma comparison")
            print("  Install with: pip install chromadb")
            run_chroma = False

    os.makedirs("experiments/results", exist_ok=True)
    all_results = []

    for ds_name in args.datasets:
        print(f"\n{'=' * 70}")
        print(f"  BEIR: {ds_name}")
        print(f"{'=' * 70}")

        cache = download_beir(ds_name)
        corpus, queries, qrels = load_beir(cache)
        test_qids = list(qrels.keys())
        print(f"  Corpus: {len(corpus)} docs, Queries: {len(test_qids)} with qrels")

        if len(corpus) > DATASETS[ds_name]["max_corpus"]:
            print(
                f"  SKIPPING — corpus too large ({len(corpus)} > {DATASETS[ds_name]['max_corpus']})"
            )
            continue

        # Pipelined embed + ingest: embed batch N while ingesting batch N-1
        from concurrent.futures import ThreadPoolExecutor, Future

        EMBED_BATCH = 256
        INGEST_BATCH = 500

        doc_ids = list(corpus.keys())
        doc_texts = [
            (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in doc_ids
        ]

        db_path = tempfile.mktemp(suffix=".db")
        store = VstashStore(db_path, embedding_dim=dim)
        vstash_id_map: dict[str, str] = {}
        all_embeddings: list[list[float]] = []
        t0 = time.perf_counter()

        pending_ingest: Future | None = None
        ingest_batch: list[dict] = []
        embedded_count = 0

        def _do_ingest(batch: list[dict]) -> None:
            store.add_documents_batch(batch)

        with ThreadPoolExecutor(max_workers=1) as executor:
            for bs in range(0, len(doc_texts), EMBED_BATCH):
                batch_texts = doc_texts[bs : bs + EMBED_BATCH]
                batch_ids = doc_ids[bs : bs + EMBED_BATCH]
                batch_embs = embed_texts(batch_texts, model_id)
                all_embeddings.extend(batch_embs)
                embedded_count += len(batch_texts)

                for doc_id, text, emb in zip(batch_ids, batch_texts, batch_embs):
                    path = f"/beir/{doc_id}"
                    vstash_id_map[path] = doc_id
                    ingest_batch.append(
                        {
                            "path": path,
                            "title": corpus[doc_id].get("title", ""),
                            "chunks": [text],
                            "embeddings": [emb],
                            "source_type": "text",
                        }
                    )

                    if len(ingest_batch) >= INGEST_BATCH:
                        if pending_ingest is not None:
                            pending_ingest.result()
                        pending_ingest = executor.submit(_do_ingest, ingest_batch)
                        ingest_batch = []

                if embedded_count % 2000 < EMBED_BATCH:
                    print(f"    {embedded_count}/{len(doc_texts)}...")

            # Flush remaining
            if pending_ingest is not None:
                pending_ingest.result()
            if ingest_batch:
                _do_ingest(ingest_batch)

        v_ingest = time.perf_counter() - t0
        print(f"  [vstash] Embed + ingest in {v_ingest:.1f}s ({len(doc_ids)} docs)")

        embed_query("warmup", model_id)
        v_metrics = evaluate_vstash(
            store,
            vstash_id_map,
            queries,
            qrels,
            model_id,
            retrieval_mode=args.retrieval_mode,
        )
        store.close()
        os.unlink(db_path)

        result: dict = {
            "dataset": ds_name,
            "docs": len(corpus),
            "queries": len(test_qids),
            "model": model_id,
            "vstash": v_metrics,
        }

        # Chroma
        if run_chroma:
            c_metrics = evaluate_chroma(
                all_embeddings, doc_ids, doc_texts, queries, qrels, model_id
            )
            result["chroma"] = c_metrics
            delta = ((v_metrics["ndcg_10"] - c_metrics["ndcg_10"]) / c_metrics["ndcg_10"]) * 100
            result["vstash_vs_chroma_pct"] = round(delta, 1)

        # Print results
        baselines = BASELINES.get(ds_name, {})
        print("\n  vstash hybrid RRF:")
        print(
            f"    NDCG@10={v_metrics['ndcg_10']:.4f}  "
            f"MRR={v_metrics['mrr']:.4f}  "
            f"R@10={v_metrics['recall_10']:.4f}  "
            f"Latency={v_metrics['latency_ms']:.1f}ms"
        )
        if run_chroma:
            print(
                f"  Chroma dense-only:\n"
                f"    NDCG@10={c_metrics['ndcg_10']:.4f}  "
                f"Latency={c_metrics['latency_ms']:.1f}ms  "
                f"(vstash {delta:+.1f}%)"
            )
        print("  Published baselines:")
        for name, score in baselines.items():
            d = ((v_metrics["ndcg_10"] - score) / score) * 100
            print(f"    {name:<20} NDCG@10={score:.3f}  (vstash {d:+.1f}%)")

        all_results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — vstash hybrid RRF ({model_id})")
    print(f"{'=' * 70}")
    header = f"  {'Dataset':<12} {'Docs':>6} {'NDCG@10':>8} {'vs BM25':>10} {'vs ColBERT':>11}"
    if run_chroma:
        header += f" {'vs Chroma':>11}"
    header += f" {'Latency':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for r in all_results:
        ds = r["dataset"]
        b = BASELINES.get(ds, {})
        v = r["vstash"]["ndcg_10"]
        bm25_d = f"{((v - b['BM25']) / b['BM25']) * 100:+.1f}%" if "BM25" in b else "—"
        col_d = (
            f"{((v - b['ColBERTv2']) / b['ColBERTv2']) * 100:+.1f}%" if "ColBERTv2" in b else "—"
        )
        line = f"  {ds:<12} {r['docs']:>6} {v:>8.4f} {bm25_d:>10} {col_d:>11}"
        if run_chroma:
            line += f" {r.get('vstash_vs_chroma_pct', 0):>+10.1f}%"
        line += f" {r['vstash']['latency_ms']:>7.1f}ms"
        print(line)

    # Save -- aggregate (compact, no per-query) goes to canonical path,
    # per-query NDCG goes to a sidecar JSON keyed by model slug for paired
    # bootstrap analysis (see experiments/paired_bootstrap_beir.py).
    model_slug = model_id.replace("/", "_").replace(" ", "_")
    if args.retrieval_mode != "hybrid":
        model_slug = f"{model_slug}_{args.retrieval_mode}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    aggregate_results = []
    perquery_results = []
    for r in all_results:
        v = dict(r["vstash"])
        v_pq = v.pop("per_query", None)
        compact = dict(r)
        compact["vstash"] = v
        aggregate_results.append(compact)
        if v_pq is not None:
            perquery_results.append(
                {
                    "dataset": r["dataset"],
                    "docs": r["docs"],
                    "queries": r["queries"],
                    "per_query": v_pq,
                }
            )

    output_path = "experiments/results/beir_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": model_id,
                "results": aggregate_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {output_path}")

    perquery_path = f"experiments/results/beir_perquery_{model_slug}.json"
    with open(perquery_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": model_id,
                "results": perquery_results,
            },
            f,
            indent=2,
        )
    print(f"  Per-query NDCG saved to {perquery_path}")

    if resolver is not None:
        unregister_encoder_resolver(resolver)


if __name__ == "__main__":
    main()
