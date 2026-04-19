"""
beir_snapvec_h2h.py -- 5-dataset BEIR benchmark across vector backends.

For each (backend in {sqlite-vec, snapvec, snapvec-ivfpq}, dataset in
the 5-BEIR cut), build a fresh ``VstashStore``, ingest the corpus,
query the held-out test set via the full production ``store.search``
pipeline, and record NDCG@10 + latency.

Why this exists. The existing
``experiments/v2_v3_head_to_head.py`` compares MODELS on a batched
eval path that skips MMR / IDF. ``experiments/vstash_pipeline_ivfpq_bench.py``
compares BACKENDS on a padded synthetic corpus. This script compares
BACKENDS on *real* BEIR datasets one at a time through the *full*
production pipeline, so the resulting table can be dropped straight
into the README's Retrieval Quality section if we ever change the
recommended default backend.

Output shape mirrors the model-h2h JSON so both can be diffed side
by side in a spreadsheet or notebook.

Usage:

    python -m experiments.beir_snapvec_h2h
    python -m experiments.beir_snapvec_h2h --model Stffens/bge-small-rrf-v3
    python -m experiments.beir_snapvec_h2h --backends sqlite-vec snapvec
    python -m experiments.beir_snapvec_h2h --datasets scifact nfcorpus

Runtime estimate (Apple Silicon, cache-hot):
  ingest per (backend, dataset):        5-20 s
  query eval per (backend, dataset):    10-60 s
  ivfpq fit per dataset:                5-60 s
  total across 3 backends x 5 datasets: 10-20 min
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.beir_benchmark import download_beir, load_beir
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore


DEFAULT_MODEL = "Stffens/bge-small-rrf-v3"
DEFAULT_DATASETS = ("scifact", "nfcorpus", "scidocs", "fiqa", "arguana")
DEFAULT_BACKENDS = ("sqlite-vec", "snapvec", "snapvec-ivfpq")
TOP_K = 10

# ColBERTv2 + BM25 reference numbers, pinned from the paper's
# Section 7.3 table. Match the README's Retrieval Quality framing.
COLBERT_V2_NDCG_10: dict[str, float] = {
    "scifact": 0.693,
    "nfcorpus": 0.344,
    "scidocs": 0.154,
    "fiqa": 0.356,
    "arguana": 0.463,
}
BM25_NDCG_10: dict[str, float] = {
    "scifact": 0.665,
    "nfcorpus": 0.325,
    "scidocs": 0.158,
    "fiqa": 0.236,
    "arguana": 0.315,
}


def _slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").lower()


def _embed_corpus(
    dataset: str,
    model: str,
    cache_dir: Path,
) -> tuple[list[str], list[str], np.ndarray]:
    """Embed a BEIR dataset's corpus, cached on disk per (dataset, model)."""
    cache_emb = cache_dir / f"{dataset}__{_slug(model)}.npy"
    cache_ids = cache_dir / f"{dataset}__ids.json"
    cache_texts = cache_dir / f"{dataset}__texts.json"

    beir_cache = download_beir(dataset)
    corpus, _, _ = load_beir(beir_cache)
    ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in ids]

    if cache_emb.exists() and cache_ids.exists() and cache_texts.exists():
        vecs = np.load(cache_emb)
        with open(cache_ids) as f:
            cached_ids = json.load(f)
        if cached_ids == ids:
            print(f"  [{dataset}] {len(ids)} docs (cached embeddings)")
            return ids, texts, vecs

    print(f"  [{dataset}] embedding {len(texts)} docs with {model} ...")
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model)
    vecs = st.encode(
        texts,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, vecs)
    cache_ids.write_text(json.dumps(ids))
    cache_texts.write_text(json.dumps(texts))
    return ids, texts, vecs


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def _ndcg_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal, k)
    return _dcg(gains, k) / idcg if idcg > 0 else 0.0


def _evaluate_backend(
    dataset: str,
    backend: str,
    model: str,
    dim: int,
    ids: list[str],
    texts: list[str],
    vecs: np.ndarray,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    tmp_dir: Path,
) -> dict:
    """Ingest corpus into a fresh store with ``backend``, run queries through
    the production search, return NDCG@10 + latency + timings."""
    db_path = tmp_dir / f"{dataset}_{backend}.db"
    for suffix in ("", "-wal", "-shm", ".snpv"):
        p = db_path.with_suffix(db_path.suffix + suffix) if suffix else db_path
        if os.path.exists(p):
            os.remove(p)

    kwargs = {"embedding_dim": dim, "vector_backend": backend}
    if backend == "snapvec-ivfpq":
        kwargs["ivfpq_rerank_candidates"] = 100
    store = VstashStore(str(db_path), **kwargs)

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

    fit_s: float | None = None
    if backend == "snapvec-ivfpq":
        t0 = time.perf_counter()
        store.fit_ivfpq()
        fit_s = time.perf_counter() - t0

    ndcgs: list[float] = []
    latencies: list[float] = []
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        q_vec = embed_query(qtext, model)
        t0 = time.perf_counter()
        results = store.search(q_vec, qtext, top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [Path(r.path).name for r in results]
        ndcgs.append(_ndcg_at_k(ranked, qrels[qid], TOP_K))

    store.close()
    for suffix in ("", "-wal", "-shm", ".snpv"):
        p = db_path.with_suffix(db_path.suffix + suffix) if suffix else db_path
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return {
        "n_queries": len(latencies),
        "ndcg_at_10": round(statistics.mean(ndcgs), 4) if ndcgs else 0.0,
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
        "fit_seconds": round(fit_s, 2) if fit_s is not None else None,
    }


def _print_table(results: dict, backends: list[str], datasets: list[str]) -> None:
    """Markdown-style comparison table: NDCG@10 per (dataset, backend)
    with BM25/ColBERTv2 reference columns."""
    print("\n### Absolute NDCG@10 (real BEIR corpora, full production pipeline)\n")
    header_cols = ["Dataset", "BM25", "ColBERTv2"] + backends
    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")
    for ds in datasets:
        cells = [ds, f"{BM25_NDCG_10.get(ds, 0):.3f}", f"{COLBERT_V2_NDCG_10.get(ds, 0):.3f}"]
        for b in backends:
            v = results.get(b, {}).get(ds, {}).get("ndcg_at_10")
            cells.append(f"{v:.4f}" if v is not None else "-")
        print("| " + " | ".join(cells) + " |")
    # Macro per backend.
    macros: dict[str, float] = {}
    for b in backends:
        vals = [results[b][ds]["ndcg_at_10"] for ds in datasets if ds in results[b]]
        macros[b] = sum(vals) / len(vals) if vals else 0.0
    macro_cells = ["**macro**", "-", "-"] + [f"**{macros[b]:.4f}**" for b in backends]
    print("| " + " | ".join(macro_cells) + " |")

    print("\n### Wins vs ColBERTv2 per backend\n")
    print("| Backend | wins / total | datasets won |")
    print("|---|---|---|")
    for b in backends:
        won: list[str] = []
        lost: list[str] = []
        for ds in datasets:
            colbert = COLBERT_V2_NDCG_10.get(ds)
            v = results.get(b, {}).get(ds, {}).get("ndcg_at_10")
            if colbert is None or v is None:
                continue
            (won if v > colbert else lost).append(ds)
        total = len(won) + len(lost)
        print(f"| {b} | {len(won)} / {total} | {', '.join(won) or '-'} |")

    print("\n### Latency + timings per (backend, dataset)\n")
    print("| Dataset | Backend | NDCG@10 | p50 ms | mean ms | ingest s | fit s |")
    print("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for b in backends:
            r = results.get(b, {}).get(ds, {})
            if not r:
                continue
            print(
                f"| {ds} | {b} | "
                f"{r.get('ndcg_at_10', 0):.4f} | "
                f"{r.get('latency_p50_ms', 0):.2f} | "
                f"{r.get('latency_mean_ms', 0):.2f} | "
                f"{r.get('ingest_seconds', 0):.2f} | "
                f"{'-' if r.get('fit_seconds') is None else r['fit_seconds']:.2f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=list(DEFAULT_BACKENDS),
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
        default=Path("experiments/results/beir_snapvec_h2h.json"),
    )
    args = parser.parse_args()

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print(
            "sentence-transformers required for corpus embedding. "
            "Install with: pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'"
        )
        return 1

    dim = get_embedding_dim(args.model)
    print(
        f"=== BEIR 5-dataset snapvec h2h ===\n"
        f"model={args.model} dim={dim} backends={args.backends} datasets={args.datasets}"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Embed corpora once per dataset; reuse across backends.
    per_dataset_corpus: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
    per_dataset_queries: dict[str, tuple[dict, dict]] = {}
    print("\n[1/2] embedding corpora ...")
    for ds in args.datasets:
        ids, texts, vecs = _embed_corpus(ds, args.model, args.cache_dir)
        per_dataset_corpus[ds] = (ids, texts, vecs)
        cache = download_beir(ds)
        _, queries, qrels = load_beir(cache)
        per_dataset_queries[ds] = (queries, qrels)
        print(f"  [{ds}] queries={len(queries)} qrels={len(qrels)}")

    # 2. Per-backend, per-dataset evaluation.
    results: dict[str, dict[str, dict]] = {}
    print("\n[2/2] evaluating ...")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for backend in args.backends:
            results[backend] = {}
            for ds in args.datasets:
                ids, texts, vecs = per_dataset_corpus[ds]
                queries, qrels = per_dataset_queries[ds]
                print(f"  [{backend}/{ds}] ...", flush=True)
                cell = _evaluate_backend(
                    dataset=ds,
                    backend=backend,
                    model=args.model,
                    dim=dim,
                    ids=ids,
                    texts=texts,
                    vecs=vecs,
                    queries=queries,
                    qrels=qrels,
                    tmp_dir=tmp_dir,
                )
                results[backend][ds] = cell
                fit_part = "" if cell["fit_seconds"] is None else f"  fit={cell['fit_seconds']}s"
                print(
                    f"    NDCG@10={cell['ndcg_at_10']:.4f}  "
                    f"p50={cell['latency_p50_ms']}ms  ingest={cell['ingest_seconds']}s{fit_part}",
                    flush=True,
                )

    _print_table(results, list(args.backends), list(args.datasets))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"model": args.model, "dim": dim, "backends": results},
            indent=2,
        )
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
