"""Scale Benchmark — Latency and NDCG at 1K/5K/10K/50K Chunks

Measures search latency and retrieval quality (NDCG@10) as the corpus
grows from 1K to 50K chunks. Uses the SciFact BEIR dataset as the
quality anchor, padded with synthetic chunks to reach target scales.

Closes paper Limitation #2: "50K-100K chunk performance remains untested."

Usage:
    python -m experiments.scale_benchmark [--scales 1000 5000 10000 50000]
    python -m experiments.scale_benchmark --output results/scale_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10


# ------------------------------------------------------------------ #
# BEIR SciFact loader (reuse from beir_benchmark)                      #
# ------------------------------------------------------------------ #


def load_scifact() -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    """Load SciFact corpus, queries, and qrels.

    Returns (corpus, queries, qrels) where:
      corpus: {doc_id: {"title": ..., "text": ...}}
      queries: {query_id: query_text}
      qrels: {query_id: {doc_id: relevance_score}}
    """
    try:
        from experiments.beir_benchmark import download_beir, load_beir

        cache = download_beir("scifact")
        return load_beir(cache)
    except Exception:
        raise RuntimeError(
            "SciFact dataset not available. Run "
            "'python -m experiments.beir_benchmark --datasets scifact' first "
            "to download it."
        )


# ------------------------------------------------------------------ #
# Synthetic chunk generation                                            #
# ------------------------------------------------------------------ #

# Vocabulary pool for generating synthetic text that is off-topic from
# SciFact (biomedical) so it does not interfere with relevance rankings.
_SYNTH_DOMAINS = [
    "architecture",
    "bridge",
    "skyscraper",
    "blueprint",
    "concrete",
    "medieval",
    "castle",
    "fortress",
    "moat",
    "drawbridge",
    "culinary",
    "sourdough",
    "fermentation",
    "broth",
    "seasoning",
    "astronomy",
    "nebula",
    "pulsar",
    "quasar",
    "supernova",
    "geology",
    "tectonic",
    "sedimentary",
    "volcanic",
    "erosion",
    "textiles",
    "weaving",
    "loom",
    "cotton",
    "dye",
    "maritime",
    "navigation",
    "compass",
    "buoyancy",
    "hull",
    "botany",
    "chlorophyll",
    "stamen",
    "germination",
    "rhizome",
    "ceramics",
    "kiln",
    "glaze",
    "porcelain",
    "stoneware",
    "typography",
    "serif",
    "kerning",
    "ligature",
    "baseline",
]


def generate_synthetic_chunks(n: int, dim: int, seed: int = 42) -> list[tuple[str, list[float]]]:
    """Generate n synthetic (text, embedding) pairs.

    Text is randomly assembled from off-topic vocabulary so it does not
    match SciFact queries. Embeddings are random unit vectors.
    """
    rng = random.Random(seed)
    chunks = []
    for i in range(n):
        words = rng.choices(_SYNTH_DOMAINS, k=rng.randint(20, 60))
        text = f"Synthetic document {i}: " + " ".join(words)
        # Random unit-ish vector (not true unit, but adequate for benchmarking)
        emb = [rng.gauss(0, 1) for _ in range(dim)]
        norm = sum(x * x for x in emb) ** 0.5
        emb = [x / norm for x in emb]
        chunks.append((text, emb))
    return chunks


# ------------------------------------------------------------------ #
# NDCG computation                                                      #
# ------------------------------------------------------------------ #


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Compute NDCG@k for a single query."""
    import math

    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ------------------------------------------------------------------ #
# Main benchmark                                                        #
# ------------------------------------------------------------------ #


def run_scale(
    scale: int,
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dim: int,
) -> dict:
    """Run benchmark at a given chunk scale. Returns metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  Scale: {scale:,} chunks")
    print(f"{'=' * 60}")

    db_path = tempfile.mktemp(suffix=".db", prefix=f"vstash_scale_{scale}_")
    store = VstashStore(db_path, embedding_dim=dim)

    # Step 1: Ingest SciFact corpus (real documents with ground truth)
    print(f"  Ingesting SciFact corpus ({len(corpus)} docs)...")
    doc_id_to_path: dict[str, str] = {}
    sci_chunks = 0
    for doc_id, doc in corpus.items():
        text = doc.get("title", "") + "\n" + doc.get("text", "")
        emb = embed_query(text[:512], MODEL)  # truncate for embedding
        path = f"/scifact/{doc_id}"
        doc_id_to_path[doc_id] = path
        store.add_document(
            path=path,
            title=doc.get("title", doc_id),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
            collection="scifact",
        )
        sci_chunks += 1

    # Step 2: Pad with synthetic chunks to reach target scale
    n_synthetic = max(0, scale - sci_chunks)
    if n_synthetic > 0:
        print(f"  Padding with {n_synthetic:,} synthetic chunks...")
        synth = generate_synthetic_chunks(n_synthetic, dim)
        batch_size = 500
        for batch_start in range(0, len(synth), batch_size):
            batch = synth[batch_start : batch_start + batch_size]
            texts = [t for t, _ in batch]
            embs = [e for _, e in batch]
            store.add_document(
                path=f"/synthetic/batch_{batch_start}",
                title=f"Synthetic batch {batch_start}",
                chunks=texts,
                embeddings=embs,
                source_type="text",
                collection="synthetic",
            )

    total_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
    print(f"  Total chunks in store: {total_chunks:,}")

    # Step 3: Run queries and measure latency + NDCG
    print(f"  Running {len(queries)} queries (top_k={TOP_K})...")
    path_to_doc_id = {v: k for k, v in doc_id_to_path.items()}

    latencies = []
    ndcg_scores = []

    # Warmup
    for qid in list(queries.keys())[:3]:
        emb = embed_query(queries[qid], MODEL)
        store.search(query_embedding=emb, query_text=queries[qid], top_k=TOP_K)

    for qid, qtext in queries.items():
        if qid not in qrels:
            continue

        emb = embed_query(qtext, MODEL)
        t0 = time.perf_counter()
        results = store.search(query_embedding=emb, query_text=qtext, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # Map result paths back to SciFact doc_ids for NDCG
        ranked_ids = []
        for r in results:
            did = path_to_doc_id.get(r.path)
            if did:
                ranked_ids.append(did)

        ndcg = ndcg_at_k(ranked_ids, qrels[qid], k=TOP_K)
        ndcg_scores.append(ndcg)

    # Compute summary stats
    sorted_latencies = sorted(latencies)
    n_queries = len(latencies)
    p50_idx = int(0.50 * n_queries) - 1
    p95_idx = int(0.95 * n_queries) - 1
    p99_idx = int(0.99 * n_queries) - 1

    metrics = {
        "scale": scale,
        "total_chunks": total_chunks,
        "n_scifact_docs": sci_chunks,
        "n_synthetic_chunks": n_synthetic,
        "n_queries": n_queries,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(sorted_latencies[max(p50_idx, 0)], 2),
            "p95": round(sorted_latencies[max(p95_idx, 0)], 2),
            "p99": round(sorted_latencies[max(p99_idx, 0)], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "ndcg_at_10": {
            "mean": round(statistics.mean(ndcg_scores), 4),
            "std": round(statistics.stdev(ndcg_scores), 4) if len(ndcg_scores) > 1 else 0.0,
        },
    }

    print(
        f"\n  Latency: mean={metrics['latency_ms']['mean']:.1f}ms, "
        f"p50={metrics['latency_ms']['median']:.1f}ms, "
        f"p95={metrics['latency_ms']['p95']:.1f}ms, "
        f"p99={metrics['latency_ms']['p99']:.1f}ms"
    )
    print(
        f"  NDCG@10: {metrics['ndcg_at_10']['mean']:.4f} (+/- {metrics['ndcg_at_10']['std']:.4f})"
    )

    store.close()
    # Clean up temp DB
    Path(db_path).unlink(missing_ok=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale benchmark (latency + NDCG)")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000, 50000],
        help="Chunk scales to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/scale_benchmark.json",
    )
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    print("Loading SciFact dataset...")
    corpus, queries, qrels = load_scifact()
    print(
        f"SciFact: {len(corpus)} docs, {len(queries)} queries, "
        f"{sum(len(v) for v in qrels.values())} judgments\n"
    )

    all_results = []
    for scale in sorted(args.scales):
        metrics = run_scale(scale, corpus, queries, qrels, dim)
        all_results.append(metrics)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Scale':>10} {'Chunks':>10} {'Latency p50':>12} {'Latency p95':>12} {'NDCG@10':>10}")
    print(f"  {'-' * 56}")
    for m in all_results:
        print(
            f"  {m['scale']:>10,} {m['total_chunks']:>10,} "
            f"{m['latency_ms']['median']:>10.1f}ms "
            f"{m['latency_ms']['p95']:>10.1f}ms "
            f"{m['ndcg_at_10']['mean']:>10.4f}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "scale_benchmark",
        "model": MODEL,
        "top_k": TOP_K,
        "scales": [m for m in all_results],
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
