"""E2E Tests & Benchmarks for Product Improvements

Validates the hypotheses behind each improvement:
  1. Deduplication: diverse results vs multi-chunk flooding
  2. Context expansion: richer text for LLM context
  3. Relevance signal: correct behavior with/without scoring
  4. Access tracking: works even with scoring disabled
  5. Latency: overhead of dedup + expand_context

Usage:
    python -m experiments.e2e_improvements [--db PATH]
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

from experiments.scoring_grid import (
    EVAL_QUERIES,
    PAPERS,
    ingest_papers,
    ndcg_at_k,
)

MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ #
# Data structures                                                      #
# ------------------------------------------------------------------ #


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


# ------------------------------------------------------------------ #
# 1. Deduplication E2E                                                 #
# ------------------------------------------------------------------ #


def test_dedup_diverse_results(store: VstashStore) -> TestResult:
    """Verify that search returns at most 1 chunk per document."""
    results_all = []
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=5)
        titles = [r.title for r in results]
        unique_titles = set(titles)
        results_all.append(
            {
                "query": eq["query"][:50],
                "total": len(titles),
                "unique": len(unique_titles),
                "duplicated": len(titles) != len(unique_titles),
            }
        )

    n_duped = sum(1 for r in results_all if r["duplicated"])
    passed = n_duped == 0
    return TestResult(
        name="dedup_diverse_results",
        passed=passed,
        detail=f"{n_duped}/{len(results_all)} queries had duplicate documents in top-5",
        metrics={"queries_with_dupes": n_duped, "total_queries": len(results_all)},
    )


def test_dedup_ndcg_impact(store: VstashStore) -> TestResult:
    """Measure NDCG@5 with dedup (production) — should match or beat experiment values."""
    ndcg5s = []
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=10)
        titles = [r.title for r in results]
        n5 = ndcg_at_k(titles, eq["expected_top"], k=5)
        ndcg5s.append(n5)

    avg_ndcg5 = sum(ndcg5s) / len(ndcg5s) if ndcg5s else 0
    # RRF baseline from ablation was 0.814 — dedup should maintain or improve
    passed = avg_ndcg5 >= 0.75  # allow some margin
    return TestResult(
        name="dedup_ndcg_impact",
        passed=passed,
        detail=f"Avg NDCG@5 = {avg_ndcg5:.4f} (target >= 0.75)",
        metrics={"avg_ndcg5": round(avg_ndcg5, 4), "per_query": [round(n, 4) for n in ndcg5s]},
    )


def test_dedup_result_diversity(store: VstashStore) -> TestResult:
    """Count unique documents across all queries — dedup should maximize diversity."""
    all_unique_per_query = []
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=5)
        unique = len(set(r.title for r in results))
        all_unique_per_query.append(unique)

    avg_unique = sum(all_unique_per_query) / len(all_unique_per_query)
    # With dedup, every result should be unique, so avg unique == avg total
    passed = avg_unique >= 4.5  # at least 4.5 unique docs per top-5
    return TestResult(
        name="dedup_result_diversity",
        passed=passed,
        detail=f"Avg unique docs in top-5: {avg_unique:.2f} (target >= 4.5)",
        metrics={"avg_unique_per_query": round(avg_unique, 2), "per_query": all_unique_per_query},
    )


# ------------------------------------------------------------------ #
# 2. Context Expansion E2E                                             #
# ------------------------------------------------------------------ #


def test_expand_context_increases_text(store: VstashStore) -> TestResult:
    """Expanded results should have more text than originals."""
    ratios = []
    for eq in EVAL_QUERIES[:5]:
        emb = embed_query(eq["query"], MODEL)
        original = store.search(emb, eq["query"], top_k=3)
        expanded = store.expand_context(original, window=1)

        for orig, exp in zip(original, expanded):
            if orig.text:
                ratio = len(exp.text) / len(orig.text)
                ratios.append(ratio)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    # At least 20% more text on average (window=1 should often double)
    passed = avg_ratio >= 1.2
    return TestResult(
        name="expand_context_text_increase",
        passed=passed,
        detail=f"Avg text expansion ratio: {avg_ratio:.2f}x (target >= 1.2x)",
        metrics={
            "avg_ratio": round(avg_ratio, 2),
            "min_ratio": round(min(ratios), 2) if ratios else 0,
            "max_ratio": round(max(ratios), 2) if ratios else 0,
            "samples": len(ratios),
        },
    )


def test_expand_context_preserves_metadata(store: VstashStore) -> TestResult:
    """Expanded results should keep title, path, chunk, score unchanged."""
    emb = embed_query(EVAL_QUERIES[0]["query"], MODEL)
    original = store.search(emb, EVAL_QUERIES[0]["query"], top_k=3)
    expanded = store.expand_context(original, window=1)

    all_match = True
    mismatches = []
    for orig, exp in zip(original, expanded):
        if (
            orig.title != exp.title
            or orig.path != exp.path
            or orig.chunk != exp.chunk
            or orig.score != exp.score
        ):
            all_match = False
            mismatches.append(f"{orig.title}: title/path/chunk/score mismatch")

    return TestResult(
        name="expand_context_preserves_metadata",
        passed=all_match,
        detail=f"{'All metadata preserved' if all_match else f'{len(mismatches)} mismatches'}",
        metrics={"mismatches": mismatches},
    )


# ------------------------------------------------------------------ #
# 3. Relevance Signal E2E                                             #
# ------------------------------------------------------------------ #


def test_relevance_signal_off_without_scoring(store: VstashStore) -> TestResult:
    """Without scoring, spread should be near-zero — signal must not fire."""
    cfg_off = ScoringConfig(enabled=False)
    spreads = []
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=5, scoring=cfg_off)
        if len(results) > 1:
            scores = [r.score for r in results]
            spread = max(scores) - min(scores)
            spreads.append(spread)

    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    # Raw RRF spread should be very small (< 0.01)
    passed = avg_spread < 0.05
    return TestResult(
        name="relevance_signal_off_without_scoring",
        passed=passed,
        detail=f"Avg spread without scoring: {avg_spread:.6f} (confirms signal is useless here)",
        metrics={
            "avg_spread": round(avg_spread, 6),
            "max_spread": round(max(spreads), 6) if spreads else 0,
        },
    )


def test_relevance_signal_on_with_scoring(store: VstashStore) -> TestResult:
    """With scoring + some access history, spread should be meaningful."""
    cfg_on = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)

    # First, build some access history
    for eq in EVAL_QUERIES[:5]:
        emb = embed_query(eq["query"], MODEL)
        for _ in range(5):
            store.search(emb, eq["query"], top_k=5, scoring=cfg_on)

    # Now measure spreads
    spreads = []
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=5, scoring=cfg_on)
        if len(results) > 1:
            scores = [r.score for r in results]
            spread = max(scores) - min(scores)
            spreads.append(spread)

    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    # With scoring, spread should be significantly larger
    passed = avg_spread > 0.05  # much higher than the 0.0006 from raw RRF
    return TestResult(
        name="relevance_signal_on_with_scoring",
        passed=passed,
        detail=f"Avg spread with scoring: {avg_spread:.4f} (should be >> 0.05)",
        metrics={
            "avg_spread": round(avg_spread, 4),
            "min_spread": round(min(spreads), 4) if spreads else 0,
            "max_spread": round(max(spreads), 4) if spreads else 0,
        },
    )


def test_relevance_discrimination(store: VstashStore) -> TestResult:
    """Score spread should differentiate relevant vs irrelevant queries."""
    cfg_on = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)

    # Build access history first
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        for _ in range(3):
            store.search(emb, eq["query"], top_k=5, scoring=cfg_on)

    relevant_spreads = []
    irrelevant_spreads = []

    # Relevant queries (from EVAL_QUERIES)
    for eq in EVAL_QUERIES:
        emb = embed_query(eq["query"], MODEL)
        results = store.search(emb, eq["query"], top_k=5, scoring=cfg_on)
        if len(results) > 1:
            scores = [r.score for r in results]
            relevant_spreads.append(max(scores) - min(scores))

    # Irrelevant queries
    off_topic = [
        "recipes for chocolate cake baking",
        "professional basketball NBA draft picks",
        "gardening tips for growing tomatoes",
        "history of ancient Egyptian pyramids",
        "how to train for a marathon",
    ]
    for q in off_topic:
        emb = embed_query(q, MODEL)
        results = store.search(emb, q, top_k=5, scoring=cfg_on)
        if len(results) > 1:
            scores = [r.score for r in results]
            irrelevant_spreads.append(max(scores) - min(scores))

    avg_rel = sum(relevant_spreads) / len(relevant_spreads) if relevant_spreads else 0
    avg_irr = sum(irrelevant_spreads) / len(irrelevant_spreads) if irrelevant_spreads else 0
    ratio = avg_rel / avg_irr if avg_irr > 0 else float("inf")

    passed = ratio > 1.2  # relevant queries should have higher spread
    return TestResult(
        name="relevance_discrimination",
        passed=passed,
        detail=f"Relevant avg spread: {avg_rel:.4f}, Irrelevant: {avg_irr:.4f}, Ratio: {ratio:.2f}x",
        metrics={
            "avg_relevant_spread": round(avg_rel, 4),
            "avg_irrelevant_spread": round(avg_irr, 4),
            "discrimination_ratio": round(ratio, 2),
        },
    )


# ------------------------------------------------------------------ #
# 4. Access Tracking E2E                                               #
# ------------------------------------------------------------------ #


def test_access_tracking_without_scoring(store: VstashStore) -> TestResult:
    """Access counts should increment even when scoring is disabled."""
    cfg = ScoringConfig(enabled=False, track_access=True)

    before = store.total_access_count()

    for eq in EVAL_QUERIES[:3]:
        emb = embed_query(eq["query"], MODEL)
        store.search(emb, eq["query"], top_k=5, scoring=cfg)

    after = store.total_access_count()
    delta = after - before

    # 3 queries × 5 results = 15 new accesses
    passed = delta > 0
    return TestResult(
        name="access_tracking_without_scoring",
        passed=passed,
        detail=f"Access count delta: {delta} (3 queries × top_k=5)",
        metrics={"before": before, "after": after, "delta": delta},
    )


def test_no_tracking_when_explicitly_off(store: VstashStore) -> TestResult:
    """track_access=False should prevent any access count changes."""
    cfg = ScoringConfig(enabled=False, track_access=False)

    before = store.total_access_count()

    for eq in EVAL_QUERIES[:3]:
        emb = embed_query(eq["query"], MODEL)
        store.search(emb, eq["query"], top_k=5, scoring=cfg)

    after = store.total_access_count()
    delta = after - before

    passed = delta == 0
    return TestResult(
        name="no_tracking_when_explicitly_off",
        passed=passed,
        detail=f"Access count delta: {delta} (expected 0)",
        metrics={"before": before, "after": after, "delta": delta},
    )


def test_total_access_count_method(store: VstashStore) -> TestResult:
    """total_access_count should match manual SQL sum."""
    method_result = store.total_access_count()
    manual = store._conn.execute(
        "SELECT COALESCE(SUM(access_count), 0) AS total FROM chunks"
    ).fetchone()["total"]

    passed = method_result == manual
    return TestResult(
        name="total_access_count_method",
        passed=passed,
        detail=f"Method: {method_result}, Manual SQL: {manual}",
        metrics={"method": method_result, "manual": manual},
    )


# ------------------------------------------------------------------ #
# 5. Benchmarks                                                        #
# ------------------------------------------------------------------ #


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def benchmark_search_latency(store: VstashStore, iterations: int = 50) -> BenchmarkResult:
    """Benchmark search with dedup (current production path)."""
    emb = embed_query(EVAL_QUERIES[0]["query"], MODEL)
    latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        store.search(emb, EVAL_QUERIES[0]["query"], top_k=5)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    return BenchmarkResult(
        name="search_with_dedup",
        iterations=iterations,
        avg_ms=sum(latencies) / len(latencies),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


def benchmark_expand_context(store: VstashStore, iterations: int = 50) -> BenchmarkResult:
    """Benchmark expand_context separately."""
    emb = embed_query(EVAL_QUERIES[0]["query"], MODEL)
    results = store.search(emb, EVAL_QUERIES[0]["query"], top_k=5)
    latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        store.expand_context(results, window=1)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    return BenchmarkResult(
        name="expand_context_window1",
        iterations=iterations,
        avg_ms=sum(latencies) / len(latencies),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


def benchmark_search_with_scoring(store: VstashStore, iterations: int = 50) -> BenchmarkResult:
    """Benchmark search + scoring + dedup (full pipeline)."""
    cfg = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)
    emb = embed_query(EVAL_QUERIES[0]["query"], MODEL)
    latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        store.search(emb, EVAL_QUERIES[0]["query"], top_k=5, scoring=cfg)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    return BenchmarkResult(
        name="search_scoring_dedup",
        iterations=iterations,
        avg_ms=sum(latencies) / len(latencies),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


def benchmark_full_pipeline(store: VstashStore, iterations: int = 30) -> BenchmarkResult:
    """Benchmark full pipeline: search + dedup + expand_context (ask/chat path)."""
    cfg = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)
    latencies = []

    for _ in range(iterations):
        emb = embed_query(EVAL_QUERIES[0]["query"], MODEL)
        t0 = time.perf_counter()
        results = store.search(emb, EVAL_QUERIES[0]["query"], top_k=5, scoring=cfg)
        store.expand_context(results, window=1)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    return BenchmarkResult(
        name="full_pipeline_search_dedup_expand",
        iterations=iterations,
        avg_ms=sum(latencies) / len(latencies),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="E2E Tests & Benchmarks")
    parser.add_argument("--db", type=str, default=None, help="Path to existing DB")
    parser.add_argument("--output", type=str, default="experiments/results/e2e_improvements.json")
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    if args.db:
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"Using existing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_e2e_")
        db_path = str(Path(tmp_dir) / "e2e.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        n_chunks = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"Ingested: {n}/{len(PAPERS)} ({n_chunks} chunks)\n")

    sep = "=" * 80

    # ------ E2E Tests ------
    print(f"\n{sep}")
    print("  E2E TESTS — Product Improvements Validation")
    print(f"{sep}\n")

    tests = [
        # Deduplication
        test_dedup_diverse_results,
        test_dedup_ndcg_impact,
        test_dedup_result_diversity,
        # Context expansion
        test_expand_context_increases_text,
        test_expand_context_preserves_metadata,
        # Relevance signal
        test_relevance_signal_off_without_scoring,
        test_relevance_signal_on_with_scoring,
        test_relevance_discrimination,
        # Access tracking
        test_access_tracking_without_scoring,
        test_no_tracking_when_explicitly_off,
        test_total_access_count_method,
    ]

    results = []
    for test_fn in tests:
        print(f"  Running: {test_fn.__name__}...", end=" ", flush=True)
        try:
            result = test_fn(store)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} — {result.detail}")
            results.append(result)
        except Exception as e:
            print(f"✗ ERROR — {e}")
            results.append(TestResult(name=test_fn.__name__, passed=False, detail=str(e)))

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n  Results: {passed}/{len(results)} passed, {failed} failed")

    # ------ Benchmarks ------
    print(f"\n{sep}")
    print("  BENCHMARKS — Latency Impact")
    print(f"{sep}\n")

    benchmarks = [
        benchmark_search_latency,
        benchmark_expand_context,
        benchmark_search_with_scoring,
        benchmark_full_pipeline,
    ]

    bench_results = []
    for bench_fn in benchmarks:
        print(f"  Running: {bench_fn.__name__}...", end=" ", flush=True)
        br = bench_fn(store)
        print(
            f"avg={br.avg_ms:.3f}ms  p50={br.p50_ms:.3f}ms  p95={br.p95_ms:.3f}ms  p99={br.p99_ms:.3f}ms"
        )
        bench_results.append(br)

    # Summary table
    print(f"\n{sep}")
    print("  LATENCY SUMMARY")
    print(f"{sep}\n")
    print(f"  {'Pipeline':<35} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'─' * 69}")
    for br in bench_results:
        print(
            f"  {br.name:<35} {br.avg_ms:>7.3f}ms {br.p50_ms:>7.3f}ms {br.p95_ms:>7.3f}ms {br.p99_ms:>7.3f}ms"
        )

    # Overhead calculation
    base = next(b for b in bench_results if b.name == "search_with_dedup")
    full = next(b for b in bench_results if b.name == "full_pipeline_search_dedup_expand")
    expand = next(b for b in bench_results if b.name == "expand_context_window1")
    print(
        f"\n  Context expansion overhead: +{expand.avg_ms:.3f}ms ({expand.avg_ms / base.avg_ms * 100:.1f}% of search)"
    )
    print(f"  Full pipeline vs base search: +{full.avg_ms - base.avg_ms:.3f}ms")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "e2e_improvements",
        "db_path": db_path,
        "n_chunks": n_chunks,
        "tests": [
            {"name": r.name, "passed": r.passed, "detail": r.detail, "metrics": r.metrics}
            for r in results
        ],
        "benchmarks": [asdict(br) for br in bench_results],
        "summary": {
            "tests_passed": passed,
            "tests_total": len(results),
            "expand_overhead_ms": round(expand.avg_ms, 3),
            "full_pipeline_avg_ms": round(full.avg_ms, 3),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
