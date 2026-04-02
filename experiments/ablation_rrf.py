"""Ablation Study: RRF vs Vector-only vs FTS-only

Compares retrieval quality across three modes:
  1. Vector-only (cosine similarity)
  2. FTS-only (BM25 via FTS5)
  3. Hybrid RRF (vector + FTS fusion)

Outputs JSON results for paper figures.

Usage:
    python -m experiments.ablation_rrf [--db PATH] [--output results/ablation.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.scoring_grid import (
    EVAL_QUERIES,
    NAME_TO_URL,
    PAPERS,
    ingest_papers,
    ndcg_at_k,
)
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ #
# Search modes                                                         #
# ------------------------------------------------------------------ #


def search_vector_only(store: VstashStore, query_embedding: list[float], top_k: int) -> list:
    """Vector search only — bypass FTS entirely."""
    dim = len(query_embedding)
    sql = """
        SELECT c.id, c.text, c.seq, d.title, d.path,
               vec_distance_cosine(v.embedding, ?) AS distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.rowid
        JOIN documents d ON d.id = c.doc_id
        ORDER BY distance ASC
        LIMIT ?
    """
    import struct

    blob = struct.pack(f"{dim}f", *query_embedding)
    rows = store._conn.execute(sql, [blob, top_k]).fetchall()

    from vstash.store import SearchResult

    results = []
    for r in rows:
        # Convert distance to similarity score (1 - distance for cosine)
        score = max(0.0, 1.0 - r["distance"])
        results.append(
            SearchResult(
                chunk_id=r["id"],
                text=r["text"],
                title=r["title"],
                path=r["path"],
                chunk=r["seq"],
                score=score,
            )
        )
    return results


def search_fts_only(store: VstashStore, query_text: str, top_k: int) -> list:
    """FTS5 search only — bypass vector entirely.

    Uses individual keyword matching (OR) for fair comparison,
    since exact phrase match is too restrictive for natural language queries.
    """
    # Split into individual words and quote each for safety
    words = query_text.split()
    safe_terms = ['"' + w.replace('"', '""') + '"' for w in words if len(w) > 2]
    if not safe_terms:
        return []
    fts_query = " OR ".join(safe_terms)

    sql = """
        SELECT c.id, c.text, c.seq, d.title, d.path,
               rank AS bm25_rank
        FROM fts_chunks f
        JOIN chunks c ON c.id = f.rowid
        JOIN documents d ON d.id = c.doc_id
        WHERE fts_chunks MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    rows = store._conn.execute(sql, [fts_query, top_k]).fetchall()

    from vstash.store import SearchResult

    results = []
    for i, r in enumerate(rows):
        score = 1.0 / (1 + i)  # reciprocal rank as score
        results.append(
            SearchResult(
                chunk_id=r["id"],
                text=r["text"],
                title=r["title"],
                path=r["path"],
                chunk=r["seq"],
                score=score,
            )
        )
    return results


def search_hybrid_rrf(
    store: VstashStore,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
) -> list:
    """Standard hybrid RRF search (the vstash default)."""
    return store.search(
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=top_k,
        scoring=None,  # Pure RRF, no frequency scoring
    )


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #


@dataclass
class AblationResult:
    mode: str
    query: str
    ndcg_5: float
    ndcg_10: float
    precision_3: float
    precision_5: float
    top_results: list[str]
    latency_ms: float


@dataclass
class AblationSummary:
    mode: str
    avg_ndcg_5: float
    avg_ndcg_10: float
    avg_precision_3: float
    avg_precision_5: float
    avg_latency_ms: float
    per_query: list[AblationResult] = field(default_factory=list)


def precision_at_k(result_titles: list[str], expected_urls: list[str], k: int) -> float:
    """Fraction of top-k results that are in the expected set."""
    hits = sum(1 for t in result_titles[:k] if t in expected_urls)
    return hits / min(k, len(expected_urls))


def run_ablation(store: VstashStore, top_k: int = 10) -> list[AblationSummary]:
    """Run all three search modes across all queries."""
    modes = {
        "vector_only": lambda emb, txt, k: search_vector_only(store, emb, k),
        "fts_only": lambda emb, txt, k: search_fts_only(store, txt, k),
        "hybrid_rrf": lambda emb, txt, k: search_hybrid_rrf(store, emb, txt, k),
    }

    summaries = []

    for mode_name, search_fn in modes.items():
        per_query = []
        ndcg5s, ndcg10s, p3s, p5s, lats = [], [], [], [], []

        for eq in EVAL_QUERIES:
            q = eq["query"]
            expected = eq["expected_top"]
            expected_urls = [NAME_TO_URL.get(n, "") for n in expected if n in NAME_TO_URL]

            emb = embed_query(q, MODEL)

            # Warmup
            for _ in range(3):
                search_fn(emb, q, top_k)

            # Timed run (average of 10)
            times = []
            results = None
            for _ in range(10):
                t0 = time.perf_counter()
                results = search_fn(emb, q, top_k)
                times.append((time.perf_counter() - t0) * 1000)

            avg_lat = sum(times) / len(times)
            result_titles = [r.title for r in results]

            n5 = ndcg_at_k(result_titles, expected, k=5)
            n10 = ndcg_at_k(result_titles, expected, k=10)
            p3 = precision_at_k(result_titles, expected_urls, k=3)
            p5 = precision_at_k(result_titles, expected_urls, k=5)

            ar = AblationResult(
                mode=mode_name,
                query=q,
                ndcg_5=n5,
                ndcg_10=n10,
                precision_3=p3,
                precision_5=p5,
                top_results=result_titles[:5],
                latency_ms=avg_lat,
            )
            per_query.append(ar)
            ndcg5s.append(n5)
            ndcg10s.append(n10)
            p3s.append(p3)
            p5s.append(p5)
            lats.append(avg_lat)

        summaries.append(
            AblationSummary(
                mode=mode_name,
                avg_ndcg_5=sum(ndcg5s) / len(ndcg5s),
                avg_ndcg_10=sum(ndcg10s) / len(ndcg10s),
                avg_precision_3=sum(p3s) / len(p3s),
                avg_precision_5=sum(p5s) / len(p5s),
                avg_latency_ms=sum(lats) / len(lats),
                per_query=per_query,
            )
        )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: RRF vs Vector vs FTS")
    parser.add_argument("--db", type=str, help="Reuse existing DB")
    parser.add_argument("--output", type=str, default="experiments/results/ablation.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    if args.db and Path(args.db).exists():
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"Reusing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_ablation_")
        db_path = str(Path(tmp_dir) / "ablation.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        print(f"Ingested: {n}/{len(PAPERS)}\n")

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ABLATION STUDY: Vector-only vs FTS-only vs Hybrid RRF")
    print(f"{sep}\n")

    summaries = run_ablation(store, top_k=args.top_k)

    # Print results
    print(f"  {'Mode':<15} {'NDCG@5':>8} {'NDCG@10':>8} {'P@3':>8} {'P@5':>8} {'Latency':>10}")
    print(f"  {'─' * 57}")
    for s in summaries:
        print(
            f"  {s.mode:<15} {s.avg_ndcg_5:>8.4f} {s.avg_ndcg_10:>8.4f} "
            f"{s.avg_precision_3:>8.4f} {s.avg_precision_5:>8.4f} "
            f"{s.avg_latency_ms:>8.2f}ms"
        )

    # Per-query breakdown
    print(f"\n{sep}")
    print("  PER-QUERY NDCG@5 COMPARISON")
    print(f"{sep}\n")

    for eq in EVAL_QUERIES:
        q = eq["query"]
        print(f"  Q: {q[:65]}")
        for s in summaries:
            qr = next(r for r in s.per_query if r.query == q)
            print(f"    {s.mode:<15} NDCG@5={qr.ndcg_5:.4f}  P@3={qr.precision_3:.4f}")
        print()

    # Improvement analysis
    vec_ndcg = next(s for s in summaries if s.mode == "vector_only").avg_ndcg_10
    fts_ndcg = next(s for s in summaries if s.mode == "fts_only").avg_ndcg_10
    rrf_ndcg = next(s for s in summaries if s.mode == "hybrid_rrf").avg_ndcg_10

    print(f"\n{sep}")
    print("  IMPROVEMENT ANALYSIS")
    print(f"{sep}\n")
    if vec_ndcg > 0:
        print(f"  RRF vs Vector-only:  {((rrf_ndcg / vec_ndcg) - 1) * 100:+.2f}% NDCG@10")
    if fts_ndcg > 0:
        print(f"  RRF vs FTS-only:     {((rrf_ndcg / fts_ndcg) - 1) * 100:+.2f}% NDCG@10")
    print(f"  RRF absolute NDCG@10: {rrf_ndcg:.4f}")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "ablation_rrf",
        "corpus_size": len(PAPERS),
        "queries": len(EVAL_QUERIES),
        "top_k": args.top_k,
        "db_path": db_path,
        "results": [asdict(s) for s in summaries],
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
