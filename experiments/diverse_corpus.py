"""Diverse Corpus Ablation — Does RRF generalize beyond homogeneous corpora?

Tests whether hybrid RRF maintains its advantage on a mixed-domain corpus
(Wikipedia articles across history, science, sports, technology, arts).

This addresses the "corpus homogeneity" limitation from the paper.

Usage:
    python -m experiments.diverse_corpus [--output results/diverse_corpus.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from vstash.config import VstashConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.ingest import ingest
from vstash.store import VstashStore

from experiments.ablation_rrf import (
    AblationResult,
    AblationSummary,
    search_fts_only,
    search_hybrid_rrf,
    search_vector_only,
)
from experiments.scoring_grid import dcg_at_k

MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ #
# Diverse Wikipedia corpus                                             #
# ------------------------------------------------------------------ #

# Mixed-domain articles: history, science, sports, tech, arts, geography
WIKI_ARTICLES: list[dict[str, str]] = [
    # History
    {"title": "French Revolution", "url": "https://en.wikipedia.org/wiki/French_Revolution"},
    {"title": "Roman Empire", "url": "https://en.wikipedia.org/wiki/Roman_Empire"},
    {"title": "World War II", "url": "https://en.wikipedia.org/wiki/World_War_II"},
    {"title": "Cold War", "url": "https://en.wikipedia.org/wiki/Cold_War"},
    # Science
    {"title": "General relativity", "url": "https://en.wikipedia.org/wiki/General_relativity"},
    {"title": "DNA", "url": "https://en.wikipedia.org/wiki/DNA"},
    {"title": "Photosynthesis", "url": "https://en.wikipedia.org/wiki/Photosynthesis"},
    {"title": "Quantum mechanics", "url": "https://en.wikipedia.org/wiki/Quantum_mechanics"},
    # Technology
    {
        "title": "Artificial intelligence",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    },
    {"title": "Internet", "url": "https://en.wikipedia.org/wiki/Internet"},
    {"title": "Semiconductor", "url": "https://en.wikipedia.org/wiki/Semiconductor"},
    # Sports
    {"title": "FIFA World Cup", "url": "https://en.wikipedia.org/wiki/FIFA_World_Cup"},
    {"title": "Olympic Games", "url": "https://en.wikipedia.org/wiki/Olympic_Games"},
    # Arts & Culture
    {"title": "Renaissance", "url": "https://en.wikipedia.org/wiki/Renaissance"},
    {"title": "Jazz", "url": "https://en.wikipedia.org/wiki/Jazz"},
    # Geography
    {"title": "Amazon rainforest", "url": "https://en.wikipedia.org/wiki/Amazon_rainforest"},
    {"title": "Mount Everest", "url": "https://en.wikipedia.org/wiki/Mount_Everest"},
]

WIKI_NAME_TO_URL = {a["title"]: a["url"] for a in WIKI_ARTICLES}

# Queries spanning multiple domains — tests semantic paraphrase handling
WIKI_QUERIES = [
    {
        "query": "wars and military conflicts in European history",
        "expected_top": ["World War II", "French Revolution", "Cold War", "Roman Empire"],
    },
    {
        "query": "fundamental physics theories about space and time",
        "expected_top": ["General relativity", "Quantum mechanics"],
    },
    {
        "query": "biological processes that convert light into energy",
        "expected_top": ["Photosynthesis", "DNA"],
    },
    {
        "query": "global sporting events and international competition",
        "expected_top": ["FIFA World Cup", "Olympic Games"],
    },
    {
        "query": "how computers and machines learn to think",
        "expected_top": ["Artificial intelligence", "Semiconductor", "Internet"],
    },
    {
        "query": "cultural and artistic movements in Europe",
        "expected_top": ["Renaissance", "Jazz", "French Revolution"],
    },
    {
        "query": "the genetic code and hereditary information in cells",
        "expected_top": ["DNA", "Photosynthesis"],
    },
    {
        "query": "extreme environments and natural wonders of the world",
        "expected_top": ["Mount Everest", "Amazon rainforest"],
    },
    {
        "query": "the rise and fall of ancient civilizations",
        "expected_top": ["Roman Empire", "French Revolution", "Renaissance"],
    },
    {
        "query": "digital networks that connect people globally",
        "expected_top": ["Internet", "Artificial intelligence", "Semiconductor"],
    },
]


# ------------------------------------------------------------------ #
# NDCG with URL-based matching (wiki titles are used as-is)            #
# ------------------------------------------------------------------ #


def wiki_ndcg_at_k(result_titles: list[str], expected_top: list[str], k: int = 5) -> float:
    """NDCG@k for wiki corpus. Deduplicates by URL."""
    max_rel = len(expected_top)
    expected_urls = {}
    for i, name in enumerate(expected_top):
        url = WIKI_NAME_TO_URL.get(name, "")
        if url:
            expected_urls[url] = max_rel - i

    # Deduplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for title in result_titles:
        if title not in seen:
            seen.add(title)
            deduped.append(title)

    actual_rel = [expected_urls.get(t, 0) for t in deduped[:k]]
    ideal_rel = sorted(expected_urls.values(), reverse=True)[:k]

    actual_dcg = dcg_at_k(actual_rel, k)
    ideal_dcg = dcg_at_k(ideal_rel, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ------------------------------------------------------------------ #
# Ingest                                                               #
# ------------------------------------------------------------------ #


def ingest_wiki(store: VstashStore) -> int:
    """Ingest Wikipedia articles. Returns count ingested."""
    cfg = VstashConfig()
    ingested = 0
    for article in WIKI_ARTICLES:
        print(f"  {article['title'][:50]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = ingest(source=article["url"], cfg=cfg, store=store, collection="wiki")
            elapsed = time.time() - t0
            print(f"✓ {r.chunks}ch {elapsed:.1f}s")
            ingested += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ {e} ({elapsed:.1f}s)")
    return ingested


# ------------------------------------------------------------------ #
# Run ablation on wiki corpus                                          #
# ------------------------------------------------------------------ #


def run_wiki_ablation(store: VstashStore, top_k: int = 10) -> list[AblationSummary]:
    """Run all three search modes across wiki queries."""
    modes = {
        "vector_only": lambda emb, txt, k: search_vector_only(store, emb, k),
        "fts_only": lambda emb, txt, k: search_fts_only(store, txt, k),
        "hybrid_rrf": lambda emb, txt, k: search_hybrid_rrf(store, emb, txt, k),
    }

    summaries = []

    for mode_name, search_fn in modes.items():
        per_query = []
        ndcg5s, ndcg10s, p3s, p5s, lats = [], [], [], [], []

        for eq in WIKI_QUERIES:
            q = eq["query"]
            expected = eq["expected_top"]
            expected_urls = [WIKI_NAME_TO_URL.get(n, "") for n in expected if n in WIKI_NAME_TO_URL]

            emb = embed_query(q, MODEL)

            t0 = time.perf_counter()
            results = search_fn(emb, q, top_k)
            lat = (time.perf_counter() - t0) * 1000

            result_titles = [r.title for r in results]

            n5 = wiki_ndcg_at_k(result_titles, expected, k=5)
            n10 = wiki_ndcg_at_k(result_titles, expected, k=10)

            # Precision: fraction of top-k that are expected (deduplicated)
            seen: set[str] = set()
            deduped: list[str] = []
            for t in result_titles:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            p3 = sum(1 for t in deduped[:3] if t in expected_urls) / min(3, len(expected_urls))
            p5 = sum(1 for t in deduped[:5] if t in expected_urls) / min(5, len(expected_urls))

            ar = AblationResult(
                mode=mode_name,
                query=q,
                ndcg_5=n5,
                ndcg_10=n10,
                precision_3=p3,
                precision_5=p5,
                top_results=result_titles[:5],
                latency_ms=lat,
            )
            per_query.append(ar)
            ndcg5s.append(n5)
            ndcg10s.append(n10)
            p3s.append(p3)
            p5s.append(p5)
            lats.append(lat)

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
    parser = argparse.ArgumentParser(description="Diverse Corpus Ablation")
    parser.add_argument("--output", type=str, default="experiments/results/diverse_corpus.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)
    tmp_dir = tempfile.mkdtemp(prefix="vstash_diverse_")
    db_path = str(Path(tmp_dir) / "diverse.db")
    store = VstashStore(db_path, embedding_dim=dim)
    print(f"New DB: {db_path}")
    print("Ingesting Wikipedia articles...")
    n = ingest_wiki(store)
    n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
    print(f"Ingested: {n}/{len(WIKI_ARTICLES)} ({n_chunks} chunks)\n")

    sep = "=" * 80
    print(f"\n{sep}")
    print("  DIVERSE CORPUS ABLATION: Vector vs FTS vs Hybrid RRF")
    print(f"  Corpus: {len(WIKI_ARTICLES)} Wikipedia articles, {n_chunks} chunks")
    print(f"{sep}\n")

    summaries = run_wiki_ablation(store, top_k=args.top_k)

    # Print results
    print(f"  {'Mode':<15} {'NDCG@5':>8} {'NDCG@10':>8} {'P@3':>8} {'P@5':>8} {'Latency':>10}")
    print(f"  {'─' * 57}")
    for s in summaries:
        print(
            f"  {s.mode:<15} {s.avg_ndcg_5:>8.4f} {s.avg_ndcg_10:>8.4f} "
            f"{s.avg_precision_3:>8.4f} {s.avg_precision_5:>8.4f} "
            f"{s.avg_latency_ms:>8.2f}ms"
        )

    # Per-query
    print(f"\n{sep}")
    print("  PER-QUERY NDCG@5")
    print(f"{sep}\n")
    for eq in WIKI_QUERIES:
        q = eq["query"]
        print(f"  Q: {q[:65]}")
        for s in summaries:
            qr = next(r for r in s.per_query if r.query == q)
            print(f"    {s.mode:<15} NDCG@5={qr.ndcg_5:.4f}  P@3={qr.precision_3:.4f}")
        print()

    # Comparison with paper corpus
    vec_ndcg = next(s for s in summaries if s.mode == "vector_only").avg_ndcg_5
    fts_ndcg = next(s for s in summaries if s.mode == "fts_only").avg_ndcg_5
    rrf_ndcg = next(s for s in summaries if s.mode == "hybrid_rrf").avg_ndcg_5

    print(f"\n{sep}")
    print("  CROSS-CORPUS COMPARISON (NDCG@5)")
    print(f"{sep}\n")
    print(f"  {'Corpus':<25} {'Vector':>8} {'FTS':>8} {'RRF':>8} {'Best':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'LLM Memory (24 papers)':<25} {'0.809':>8} {'0.631':>8} {'0.814':>8} {'RRF':>10}")
    print(
        f"  {'Wikipedia (17 articles)':<25} {vec_ndcg:>8.3f} {fts_ndcg:>8.3f} {rrf_ndcg:>8.3f} "
        f"{'RRF' if rrf_ndcg >= max(vec_ndcg, fts_ndcg) else 'Vector' if vec_ndcg > fts_ndcg else 'FTS':>10}"
    )

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "diverse_corpus",
        "corpus": "wikipedia_mixed",
        "corpus_size": len(WIKI_ARTICLES),
        "n_chunks": n_chunks,
        "queries": len(WIKI_QUERIES),
        "top_k": args.top_k,
        "db_path": db_path,
        "results": [asdict(s) for s in summaries],
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
