"""Expanded Frequency + Decay Scoring Experiment — Grid Search

Reuses the DB from scoring_experiment.py (or ingests fresh if needed).
Tests a grid of configurations across multiple access pattern scenarios
and evaluates with quantitative metrics.

Metrics:
  - NDCG@k: How well does scoring promote "expected" relevant results?
  - Rank displacement: Average position change vs baseline
  - Recency correlation: Do newer papers rank higher for temporal queries?
  - Frequency response: Does boosted content actually rise?

Usage:
    python -m experiments.scoring_grid [--db PATH]  # reuse existing DB
    python -m experiments.scoring_grid               # ingest fresh
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vstash.config import ScoringConfig, VstashConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Papers corpus (same as scoring_experiment.py)                        #
# ------------------------------------------------------------------ #

# Note: vstash stores the URL as both title and path when ingesting from URLs.
# All matching (NDCG, frequency_response, etc.) uses URL-based lookup.

PAPERS: list[dict[str, str]] = [
    {"title": "MemGPT", "url": "https://arxiv.org/pdf/2310.08560", "year": "2023"},
    {"title": "Generative Agents", "url": "https://arxiv.org/pdf/2304.03442", "year": "2023"},
    {"title": "RET-LLM", "url": "https://arxiv.org/pdf/2305.14322", "year": "2023"},
    {"title": "SCM", "url": "https://arxiv.org/pdf/2304.13343", "year": "2023"},
    {"title": "LoCoMo", "url": "https://arxiv.org/pdf/2402.17753", "year": "2024"},
    {"title": "SORT", "url": "https://arxiv.org/pdf/2410.08133", "year": "2024"},
    {"title": "Episodic Memories Benchmark", "url": "https://arxiv.org/pdf/2501.13121", "year": "2025"},
    {"title": "MemoryBank", "url": "https://arxiv.org/pdf/2305.10250", "year": "2024"},
    {"title": "Dynamic Tree Memory", "url": "https://arxiv.org/pdf/2410.14052", "year": "2024"},
    {"title": "A-MEM", "url": "https://arxiv.org/pdf/2502.12110", "year": "2025"},
    {"title": "Mem0", "url": "https://arxiv.org/pdf/2504.19413", "year": "2025"},
    {"title": "Memoria", "url": "https://arxiv.org/pdf/2512.12686", "year": "2025"},
    {"title": "Memori", "url": "https://arxiv.org/pdf/2603.19935", "year": "2025"},
    {"title": "Nemori", "url": "https://arxiv.org/pdf/2508.03341", "year": "2025"},
    {"title": "SeCom", "url": "https://arxiv.org/pdf/2502.05589", "year": "2025"},
    {"title": "R³Mem", "url": "https://arxiv.org/pdf/2502.15957", "year": "2025"},
    {"title": "Zep", "url": "https://arxiv.org/pdf/2501.13956", "year": "2025"},
    {"title": "MaRS", "url": "https://arxiv.org/pdf/2512.12856", "year": "2025"},
    {"title": "PAM", "url": "https://arxiv.org/pdf/2602.11322", "year": "2026"},
    {"title": "E-mem", "url": "https://arxiv.org/pdf/2601.21714", "year": "2026"},
    {"title": "Beyond Fact Retrieval (GSW)", "url": "https://arxiv.org/pdf/2511.07587", "year": "2026"},
    {"title": "MAGMA", "url": "https://arxiv.org/pdf/2601.03236", "year": "2026"},
    {"title": "EverMemOS", "url": "https://arxiv.org/pdf/2601.02163", "year": "2026"},
    {"title": "AI PERSONA", "url": "https://arxiv.org/pdf/2412.13103", "year": "2024"},
]

# Name → URL lookup for matching results (vstash stores URL as title/path)
NAME_TO_URL = {p["title"]: p["url"] for p in PAPERS}
URL_TO_NAME = {p["url"]: p["title"] for p in PAPERS}

# ------------------------------------------------------------------ #
# Expanded queries with expected relevance judgments                    #
# ------------------------------------------------------------------ #

# Each query has "expected_top" — papers that a human would expect in top-5.
# Used for NDCG calculation. Order matters (first = most relevant).
EVAL_QUERIES = [
    {
        "query": "long-term memory systems for conversational AI agents",
        "expected_top": ["LoCoMo", "Memoria", "Memori", "MemGPT", "Mem0"],
    },
    {
        "query": "temporal decay and forgetting mechanisms in memory",
        "expected_top": ["MaRS", "PAM", "Zep", "MemoryBank", "Nemori"],
    },
    {
        "query": "benchmarks for evaluating memory in language models",
        "expected_top": ["LoCoMo", "SORT", "Episodic Memories Benchmark", "MaRS", "SeCom"],
    },
    {
        "query": "episodic memory architecture for LLM agents",
        "expected_top": ["E-mem", "EverMemOS", "SORT", "Episodic Memories Benchmark", "MAGMA"],
    },
    {
        "query": "personalization and user modeling through persistent memory",
        "expected_top": ["AI PERSONA", "SeCom", "Memoria", "Memori", "Mem0"],
    },
    {
        "query": "knowledge graph approaches to agent memory",
        "expected_top": ["Zep", "MAGMA", "Dynamic Tree Memory", "A-MEM", "EverMemOS"],
    },
    {
        "query": "memory compression and retrieval efficiency",
        "expected_top": ["R³Mem", "SeCom", "SCM", "RET-LLM", "Generative Agents"],
    },
    {
        "query": "multi-agent memory sharing and coordination",
        "expected_top": ["E-mem", "MAGMA", "Generative Agents", "EverMemOS", "MemGPT"],
    },
    {
        "query": "read-write memory interfaces for language models",
        "expected_top": ["RET-LLM", "MemGPT", "A-MEM", "Mem0", "MemoryBank"],
    },
    {
        "query": "self-organizing and adaptive memory structures",
        "expected_top": ["Nemori", "EverMemOS", "A-MEM", "MAGMA", "Dynamic Tree Memory"],
    },
]

# ------------------------------------------------------------------ #
# Configuration grid                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ScoringVariant:
    name: str
    alpha: float
    beta: float
    decay_lambda: float
    over_fetch: int

GRID: list[ScoringVariant] = [
    # Baseline weights, varying lambda
    ScoringVariant("α.7_β.3_λ.01_of50",  0.7, 0.3, 0.01, 50),
    ScoringVariant("α.7_β.3_λ.03_of50",  0.7, 0.3, 0.03, 50),
    ScoringVariant("α.7_β.3_λ.05_of50",  0.7, 0.3, 0.05, 50),
    ScoringVariant("α.7_β.3_λ.07_of50",  0.7, 0.3, 0.07, 50),
    ScoringVariant("α.7_β.3_λ.10_of50",  0.7, 0.3, 0.10, 50),
    ScoringVariant("α.7_β.3_λ.20_of50",  0.7, 0.3, 0.20, 50),
    # Higher memory weight
    ScoringVariant("α.5_β.5_λ.05_of50",  0.5, 0.5, 0.05, 50),
    ScoringVariant("α.5_β.5_λ.07_of50",  0.5, 0.5, 0.07, 50),
    ScoringVariant("α.5_β.5_λ.10_of50",  0.5, 0.5, 0.10, 50),
    # Lower memory weight (subtle boost)
    ScoringVariant("α.8_β.2_λ.05_of50",  0.8, 0.2, 0.05, 50),
    ScoringVariant("α.8_β.2_λ.07_of50",  0.8, 0.2, 0.07, 50),
    ScoringVariant("α.8_β.2_λ.10_of50",  0.8, 0.2, 0.10, 50),
    # Extreme: semantic only vs memory heavy
    ScoringVariant("α.9_β.1_λ.07_of50",  0.9, 0.1, 0.07, 50),
    ScoringVariant("α.4_β.6_λ.07_of50",  0.4, 0.6, 0.07, 50),
    # Over-fetch variations
    ScoringVariant("α.7_β.3_λ.07_of20",  0.7, 0.3, 0.07, 20),
    ScoringVariant("α.7_β.3_λ.07_of100", 0.7, 0.3, 0.07, 100),
]

# ------------------------------------------------------------------ #
# Access pattern scenarios                                             #
# ------------------------------------------------------------------ #

@dataclass
class AccessScenario:
    """Defines a simulated usage pattern to apply before evaluation."""
    name: str
    description: str
    # {paper_short_name: (access_count, days_ago_of_last_access)}
    patterns: dict[str, tuple[int, float]]


SCENARIOS: list[AccessScenario] = [
    AccessScenario(
        name="uniform",
        description="All papers access_count=1, last_accessed=publication_date (baseline state)",
        patterns={},  # No changes — uses ingestion defaults
    ),
    AccessScenario(
        name="recent_heavy_use",
        description="LoCoMo(20x, today), PAM(10x, 2d ago), MemGPT(3x, 30d ago)",
        patterns={
            "LoCoMo": (20, 0),
            "PAM": (10, 2),
            "MemGPT": (3, 30),
        },
    ),
    AccessScenario(
        name="stale_favorites",
        description="MemGPT(50x, 180d), Generative Agents(30x, 120d), RET-LLM(20x, 90d)",
        patterns={
            "MemGPT": (50, 180),
            "Generative Agents": (30, 120),
            "RET-LLM": (20, 90),
        },
    ),
    AccessScenario(
        name="mixed_recency",
        description="Active research: recent papers boosted, old foundations stale",
        patterns={
            "EverMemOS": (15, 1),
            "MAGMA": (12, 3),
            "E-mem": (10, 5),
            "PAM": (8, 7),
            "MaRS": (5, 14),
            "LoCoMo": (20, 10),
            "MemGPT": (5, 60),
            "Generative Agents": (3, 90),
        },
    ),
    AccessScenario(
        name="benchmark_focused",
        description="User doing evaluation work — benchmarks heavily accessed",
        patterns={
            "LoCoMo": (30, 0),
            "SORT": (25, 1),
            "Episodic Memories Benchmark": (20, 2),
            "SeCom": (10, 5),
            "MaRS": (8, 7),
        },
    ),
]


# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #

def dcg_at_k(relevance: list[float], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))


def ndcg_at_k(result_titles: list[str], expected_top: list[str], k: int = 5) -> float:
    """Normalized DCG@k. Expected_top uses paper short names (mapped to URLs)."""
    max_rel = len(expected_top)
    # Map expected paper names → URLs → graded relevance
    expected_urls = {}
    for i, name in enumerate(expected_top):
        url = NAME_TO_URL.get(name, "")
        if url:
            expected_urls[url] = max_rel - i

    # Actual relevance of returned results (result titles ARE URLs)
    actual_rel = []
    for title in result_titles[:k]:
        actual_rel.append(expected_urls.get(title, 0))

    # Ideal relevance (sorted descending)
    ideal_rel = sorted(expected_urls.values(), reverse=True)[:k]

    actual_dcg = dcg_at_k(actual_rel, k)
    ideal_dcg = dcg_at_k(ideal_rel, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def avg_rank_displacement(
    baseline_titles: list[str], scored_titles: list[str]
) -> float:
    """Average absolute position change for shared results."""
    displacements = []
    for i, title in enumerate(scored_titles):
        if title in baseline_titles:
            old_pos = baseline_titles.index(title)
            displacements.append(abs(i - old_pos))
    return sum(displacements) / len(displacements) if displacements else 0.0


def recency_correlation(results: list, paper_years: dict[str, int]) -> float:
    """Spearman-like: do newer papers rank higher? Returns [-1, 1]."""
    # paper_years maps URL → year
    ranked_years = []
    for r in results:
        year = paper_years.get(r.title) or paper_years.get(r.path)
        if year:
            ranked_years.append(year)

    if len(ranked_years) < 2:
        return 0.0

    n = len(ranked_years)
    year_ranks = sorted(range(n), key=lambda i: -ranked_years[i])
    position_ranks = list(range(n))

    d_sq = sum((year_ranks[i] - position_ranks[i]) ** 2 for i in range(n))
    return 1 - (6 * d_sq) / (n * (n * n - 1))


def frequency_response(
    baseline_results: list, scored_results: list, boosted_papers: list[str]
) -> float:
    """Measures how much boosted papers improved their rank. Returns avg improvement."""
    if not boosted_papers:
        return 0.0

    improvements = []
    b_titles = [r.title for r in baseline_results]
    s_titles = [r.title for r in scored_results]
    k = len(s_titles)

    for paper in boosted_papers:
        # Map paper name → URL for matching
        url = NAME_TO_URL.get(paper, "")
        if not url:
            continue

        b_pos = b_titles.index(url) if url in b_titles else None
        s_pos = s_titles.index(url) if url in s_titles else None

        if b_pos is not None and s_pos is not None:
            improvements.append(b_pos - s_pos)  # positive = improved
        elif s_pos is not None and b_pos is None:
            improvements.append(k)  # appeared in top-k where it wasn't before

    return sum(improvements) / len(improvements) if improvements else 0.0


# ------------------------------------------------------------------ #
# DB snapshot/restore for clean scenario evaluation                    #
# ------------------------------------------------------------------ #

def snapshot_access_data(conn: sqlite3.Connection) -> list[tuple]:
    """Save current access_count and last_accessed_at for all chunks."""
    return conn.execute(
        "SELECT id, access_count, last_accessed_at, created_at FROM chunks"
    ).fetchall()


def restore_access_data(conn: sqlite3.Connection, snapshot: list[tuple]) -> None:
    """Restore access data from snapshot."""
    for row in snapshot:
        conn.execute(
            "UPDATE chunks SET access_count = ?, last_accessed_at = ?, created_at = ? "
            "WHERE id = ?",
            [row["access_count"], row["last_accessed_at"], row["created_at"], row["id"]],
        )
    conn.commit()


def apply_scenario(
    store: VstashStore, scenario: AccessScenario
) -> list[str]:
    """Apply access pattern scenario. Returns list of boosted paper names."""
    now = datetime.now(timezone.utc)
    boosted = []

    for paper_name, (count, days_ago) in scenario.patterns.items():
        access_date = (now - timedelta(days=days_ago)).isoformat()

        # Find chunks matching this paper (vstash stores URL as path when ingesting from URL)
        url = NAME_TO_URL.get(paper_name, "")
        rows = store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = ? OR d.title LIKE ? OR d.path LIKE ?",
            [url, f"%{paper_name}%", f"%{paper_name}%"],
        ).fetchall()

        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            store._conn.execute(
                f"UPDATE chunks SET access_count = ?, last_accessed_at = ? "
                f"WHERE id IN ({placeholders})",
                [count, access_date, *ids],
            )
            boosted.append(paper_name)

    store._conn.commit()
    return boosted


# ------------------------------------------------------------------ #
# Main grid evaluation                                                 #
# ------------------------------------------------------------------ #

@dataclass
class EvalResult:
    variant: str
    scenario: str
    avg_ndcg: float
    avg_displacement: float
    avg_recency_corr: float
    avg_freq_response: float
    per_query_ndcg: list[float] = field(default_factory=list)


def evaluate_variant(
    store: VstashStore,
    variant: ScoringVariant,
    scenario: AccessScenario,
    baseline_results: dict[str, list],
    boosted_papers: list[str],
    top_k: int = 10,
) -> EvalResult:
    """Run all queries with a given config variant and compute metrics."""
    paper_years = {p["url"]: int(p["year"]) for p in PAPERS}

    scoring = ScoringConfig(
        enabled=True,
        alpha=variant.alpha,
        beta=variant.beta,
        decay_lambda=variant.decay_lambda,
        over_fetch=variant.over_fetch,
        track_access=False,
    )

    ndcgs = []
    displacements = []
    recency_corrs = []
    freq_responses = []

    for eq in EVAL_QUERIES:
        q = eq["query"]
        expected = eq["expected_top"]

        emb = embed_query(q, MODEL)
        results = store.search(
            query_embedding=emb, query_text=q, top_k=top_k, scoring=scoring,
        )

        result_titles = [r.title for r in results]

        # NDCG
        ndcg = ndcg_at_k(result_titles, expected, k=top_k)
        ndcgs.append(ndcg)

        # Displacement vs baseline
        if q in baseline_results:
            b_titles = [r.title for r in baseline_results[q]]
            disp = avg_rank_displacement(b_titles, result_titles)
            displacements.append(disp)

        # Recency
        rc = recency_correlation(results, paper_years)
        recency_corrs.append(rc)

        # Frequency response
        if boosted_papers and q in baseline_results:
            fr = frequency_response(baseline_results[q], results, boosted_papers)
            freq_responses.append(fr)

    return EvalResult(
        variant=variant.name,
        scenario=scenario.name,
        avg_ndcg=sum(ndcgs) / len(ndcgs) if ndcgs else 0,
        avg_displacement=sum(displacements) / len(displacements) if displacements else 0,
        avg_recency_corr=sum(recency_corrs) / len(recency_corrs) if recency_corrs else 0,
        avg_freq_response=sum(freq_responses) / len(freq_responses) if freq_responses else 0,
        per_query_ndcg=ndcgs,
    )


def ingest_papers(store: VstashStore) -> int:
    """Ingest all papers and set temporal dates. Returns count ingested."""
    from vstash.ingest import ingest

    cfg = VstashConfig()
    ingested = 0

    for paper in PAPERS:
        print(f"  {paper['title'][:55]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = ingest(source=paper["url"], cfg=cfg, store=store, collection="experiment")
            elapsed = time.time() - t0
            if r.status == "ok":
                print(f"✓ {r.chunks}ch {elapsed:.1f}s")
                ingested += 1
                year = int(paper["year"])
                sim_date = datetime(year, 6, 15, tzinfo=timezone.utc).isoformat()
                store._conn.execute(
                    "UPDATE chunks SET created_at = ?, last_accessed_at = ? "
                    "WHERE doc_id = (SELECT id FROM documents WHERE path = ?)",
                    [sim_date, sim_date, paper["url"]],
                )
                store._conn.commit()
            elif r.status == "skipped":
                print(f"⊘ skipped {elapsed:.1f}s")
                ingested += 1
            else:
                print(f"✗ {r.status}")
        except Exception as e:
            print(f"✗ {e}")

    return ingested


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring grid search experiment")
    parser.add_argument("--db", type=str, help="Reuse existing DB path")
    parser.add_argument("--top-k", type=int, default=10, help="Results per query")
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    # Setup DB
    if args.db and Path(args.db).exists():
        db_path = args.db
        print(f"Reusing DB: {db_path}")
        store = VstashStore(db_path, embedding_dim=dim)
        chunk_count = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"  {chunk_count} chunks in DB")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_grid_")
        db_path = str(Path(tmp_dir) / "grid.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        print(f"  Ingested: {n}/{len(PAPERS)}\n")

    # Take clean snapshot (access_count=1, temporal dates set)
    clean_snapshot = snapshot_access_data(store._conn)

    # ============================================================== #
    # Run baseline (no scoring)                                        #
    # ============================================================== #
    sep = "=" * 80
    print(f"\n{sep}")
    print("  PHASE 1: Computing baseline (RRF, no scoring)")
    print(sep)

    baseline_results: dict[str, list] = {}
    for eq in EVAL_QUERIES:
        q = eq["query"]
        emb = embed_query(q, MODEL)
        results = store.search(
            query_embedding=emb, query_text=q, top_k=args.top_k, scoring=None,
        )
        baseline_results[q] = results

    # Compute baseline NDCG for reference
    baseline_ndcgs = []
    for eq in EVAL_QUERIES:
        titles = [r.title for r in baseline_results[eq["query"]]]
        ndcg = ndcg_at_k(titles, eq["expected_top"], k=args.top_k)
        baseline_ndcgs.append(ndcg)
    avg_baseline_ndcg = sum(baseline_ndcgs) / len(baseline_ndcgs)
    print(f"  Baseline avg NDCG@{args.top_k}: {avg_baseline_ndcg:.4f}")

    # ============================================================== #
    # Grid evaluation                                                  #
    # ============================================================== #
    all_results: list[EvalResult] = []

    for scenario in SCENARIOS:
        print(f"\n{sep}")
        print(f"  SCENARIO: {scenario.name}")
        print(f"  {scenario.description}")
        print(sep)

        # Restore clean state and apply scenario
        restore_access_data(store._conn, clean_snapshot)
        boosted = apply_scenario(store, scenario)
        if boosted:
            print(f"  Boosted papers: {', '.join(boosted)}")

        for variant in GRID:
            result = evaluate_variant(
                store, variant, scenario, baseline_results, boosted, args.top_k,
            )
            all_results.append(result)
            ndcg_delta = result.avg_ndcg - avg_baseline_ndcg
            sign = "+" if ndcg_delta >= 0 else ""
            print(
                f"  {variant.name:<25} "
                f"NDCG={result.avg_ndcg:.4f}({sign}{ndcg_delta:.4f}) "
                f"Disp={result.avg_displacement:.2f} "
                f"Rec={result.avg_recency_corr:+.3f} "
                f"Freq={result.avg_freq_response:+.2f}"
            )

    # ============================================================== #
    # Summary: best configs per scenario                               #
    # ============================================================== #
    print(f"\n{sep}")
    print("  SUMMARY: Best configs per scenario (by NDCG)")
    print(sep)

    print(f"\n  Baseline NDCG@{args.top_k}: {avg_baseline_ndcg:.4f}\n")

    for scenario in SCENARIOS:
        scenario_results = [r for r in all_results if r.scenario == scenario.name]
        scenario_results.sort(key=lambda r: r.avg_ndcg, reverse=True)

        best = scenario_results[0]
        worst = scenario_results[-1]
        delta_best = best.avg_ndcg - avg_baseline_ndcg
        print(f"  [{scenario.name}]")
        print(f"    Best:  {best.variant:<25} NDCG={best.avg_ndcg:.4f} ({'+' if delta_best >= 0 else ''}{delta_best:.4f} vs baseline)")
        print(f"    Worst: {worst.variant:<25} NDCG={worst.avg_ndcg:.4f}")
        print(f"    Best freq response: {best.avg_freq_response:+.2f}")
        print()

    # Overall best across all scenarios
    print(f"  {'─' * 70}")
    print("  OVERALL BEST (averaged across all scenarios):")

    variant_scores: dict[str, list[float]] = {}
    for r in all_results:
        variant_scores.setdefault(r.variant, []).append(r.avg_ndcg)

    ranked = sorted(variant_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

    print(f"\n  {'Rank':<5} {'Config':<27} {'Avg NDCG':<10} {'vs Baseline':<12} {'Min NDCG':<10} {'Max NDCG':<10}")
    print(f"  {'─' * 74}")
    for i, (name, ndcgs) in enumerate(ranked[:10]):
        avg = sum(ndcgs) / len(ndcgs)
        delta = avg - avg_baseline_ndcg
        print(
            f"  {i+1:<5} {name:<27} {avg:.4f}     "
            f"{'+' if delta >= 0 else ''}{delta:.4f}       "
            f"{min(ndcgs):.4f}     {max(ndcgs):.4f}"
        )

    # Recommended config
    best_name = ranked[0][0]
    best_variant = next(v for v in GRID if v.name == best_name)
    print(f"\n  ★ Recommended default config:")
    print(f"    alpha = {best_variant.alpha}")
    print(f"    beta = {best_variant.beta}")
    print(f"    decay_lambda = {best_variant.decay_lambda}")
    print(f"    over_fetch = {best_variant.over_fetch}")

    # ============================================================== #
    # Latency benchmark                                                #
    # ============================================================== #
    print(f"\n{sep}")
    print("  LATENCY BENCHMARK: scoring overhead measurement")
    print(sep)

    chunk_count = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
    print(f"\n  Corpus: {chunk_count} chunks")

    # Warm up embedding model (exclude from timing)
    warmup_q = EVAL_QUERIES[0]["query"]
    embed_query(warmup_q, MODEL)

    WARMUP_RUNS = 3
    BENCH_RUNS = 20

    scoring_off = ScoringConfig(enabled=False)
    scoring_on = ScoringConfig(
        enabled=True,
        alpha=best_variant.alpha,
        beta=best_variant.beta,
        decay_lambda=best_variant.decay_lambda,
        over_fetch=best_variant.over_fetch,
        track_access=False,
    )
    scoring_on_track = ScoringConfig(
        enabled=True,
        alpha=best_variant.alpha,
        beta=best_variant.beta,
        decay_lambda=best_variant.decay_lambda,
        over_fetch=best_variant.over_fetch,
        track_access=True,
    )

    configs = [
        ("RRF only (no scoring)", scoring_off),
        ("Scoring (rerank, no tracking)", scoring_on),
        ("Scoring + track_access", scoring_on_track),
    ]

    # Per-query latencies for each config
    latency_results: dict[str, list[float]] = {}

    for label, cfg_scoring in configs:
        times_ms: list[float] = []

        for eq in EVAL_QUERIES:
            q = eq["query"]
            emb = embed_query(q, MODEL)

            # Warmup runs (discard)
            for _ in range(WARMUP_RUNS):
                store.search(query_embedding=emb, query_text=q, top_k=args.top_k, scoring=cfg_scoring)

            # Timed runs
            query_times: list[float] = []
            for _ in range(BENCH_RUNS):
                t0 = time.perf_counter()
                store.search(query_embedding=emb, query_text=q, top_k=args.top_k, scoring=cfg_scoring)
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                query_times.append(elapsed)

            times_ms.extend(query_times)

        latency_results[label] = times_ms

    # Report
    print(f"\n  {'Config':<35} {'Median':>8} {'P95':>8} {'P99':>8} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─' * 83}")

    baseline_median = None
    for label, times in latency_results.items():
        times.sort()
        n = len(times)
        median = times[n // 2]
        p95 = times[int(n * 0.95)]
        p99 = times[int(n * 0.99)]
        mean = sum(times) / n
        mn, mx = times[0], times[-1]

        if baseline_median is None:
            baseline_median = median
            overhead = ""
        else:
            oh = ((median / baseline_median) - 1) * 100
            overhead = f"  ({oh:+.1f}%)"

        print(
            f"  {label:<35} {median:>7.2f}ms {p95:>7.2f}ms {p99:>7.2f}ms "
            f"{mean:>7.2f}ms {mn:>7.2f}ms {mx:>7.2f}ms{overhead}"
        )

    # Component-level breakdown
    print(f"\n  Component breakdown (median of {BENCH_RUNS} runs per query):")

    # Measure rerank_with_decay() in isolation
    restore_access_data(store._conn, clean_snapshot)
    apply_scenario(store, SCENARIOS[1])  # recent_heavy_use for realistic data

    sample_q = EVAL_QUERIES[0]["query"]
    sample_emb = embed_query(sample_q, MODEL)

    # Get candidates by running search without scoring
    raw_results = store.search(
        query_embedding=sample_emb, query_text=sample_q,
        top_k=best_variant.over_fetch, scoring=None,
    )

    # Build candidate dicts like search() does internally
    candidates = [
        {"id": i, "rrf": r.score, "text": r.text, "title": r.title,
         "path": r.path, "chunk": r.chunk}
        for i, r in enumerate(raw_results, 1)
    ]

    # Time rerank_with_decay
    rerank_times: list[float] = []
    for _ in range(WARMUP_RUNS):
        store.rerank_with_decay(copy.deepcopy(candidates))
    for _ in range(BENCH_RUNS * 5):
        c = copy.deepcopy(candidates)
        t0 = time.perf_counter()
        store.rerank_with_decay(c, alpha=best_variant.alpha, beta=best_variant.beta,
                                decay_lambda=best_variant.decay_lambda)
        rerank_times.append((time.perf_counter() - t0) * 1000)
    rerank_times.sort()

    # Time track_access
    sample_ids = list(range(1, min(11, chunk_count + 1)))  # 10 chunk IDs
    track_times: list[float] = []
    for _ in range(WARMUP_RUNS):
        store.track_access(sample_ids)
    for _ in range(BENCH_RUNS * 5):
        t0 = time.perf_counter()
        store.track_access(sample_ids)
        track_times.append((time.perf_counter() - t0) * 1000)
    track_times.sort()

    rr_med = rerank_times[len(rerank_times) // 2]
    ta_med = track_times[len(track_times) // 2]
    print(f"  rerank_with_decay({len(candidates)} candidates): {rr_med:.3f}ms median")
    print(f"  track_access({len(sample_ids)} chunks):            {ta_med:.3f}ms median")

    print(f"\n  DB preserved at: {db_path}")
    store.close()


if __name__ == "__main__":
    main()
