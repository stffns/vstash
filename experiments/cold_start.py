"""Cold Start Curve — How many queries before scoring helps?

Measures the warm-up period for frequency+decay scoring by simulating
incremental usage and comparing scored vs unscored NDCG at each step.

The experiment:
  1. Ingest corpus into a fresh DB (zero access history)
  2. Run all eval queries with scoring OFF → baseline NDCG
  3. For each "round" (1..N):
     a. Simulate realistic non-uniform usage: some queries are asked more
        frequently than others, creating differential access patterns
     b. Run all eval queries with scoring ON → record NDCG
     c. Compare scored NDCG vs baseline
  4. Find the crossover point where scoring consistently beats baseline

Key insight: Scoring only helps when access patterns are non-uniform.
Real users have "favorite" topics they query repeatedly — this creates
the differential signal that scoring exploits.

Usage:
    python -m experiments.cold_start [--db PATH] [--rounds 20] [--output results/cold_start.json]
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vstash.config import ScoringConfig, VstashConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

from experiments.scoring_grid import (
    EVAL_QUERIES,
    NAME_TO_URL,
    PAPERS,
    ingest_papers,
    ndcg_at_k,
)


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #


@dataclass
class RoundResult:
    round_num: int
    scored_ndcg_5: float
    baseline_ndcg_5: float
    delta: float  # scored - baseline
    delta_pct: float  # percentage improvement
    total_accesses: int  # cumulative access events so far
    usage_queries_this_round: int  # how many usage queries were simulated
    per_query_scored: list[float] = field(default_factory=list)
    per_query_baseline: list[float] = field(default_factory=list)


@dataclass
class ColdStartResult:
    rounds: list[RoundResult]
    crossover_round: int | None  # first round where scored > baseline for 3+ consecutive
    total_rounds: int
    corpus_size: int
    n_queries: int
    scoring_config: dict


# ------------------------------------------------------------------ #
# Usage simulation                                                     #
# ------------------------------------------------------------------ #

# Weighted query distribution simulating realistic usage:
# Some topics get queried much more than others.
# Weights roughly follow Zipf distribution.
QUERY_WEIGHTS = [
    5,   # "long-term memory systems" — heavily queried
    4,   # "temporal decay and forgetting" — frequently queried
    3,   # "benchmarks for evaluating memory" — moderate
    3,   # "episodic memory architecture" — moderate
    2,   # "personalization and user modeling" — occasional
    2,   # "knowledge graph approaches" — occasional
    1,   # "memory compression" — rare
    1,   # "multi-agent memory sharing" — rare
    1,   # "read-write memory interfaces" — rare
    1,   # "self-organizing and adaptive" — rare
]


def simulate_usage_round(
    store: VstashStore,
    query_embeddings: dict[str, list[float]],
    round_num: int,
    top_k: int = 10,
) -> int:
    """Simulate one round of non-uniform user queries. Returns access count."""
    rng = random.Random(42 + round_num)  # deterministic per round

    # Build weighted query list
    weighted_queries = []
    for eq, weight in zip(EVAL_QUERIES, QUERY_WEIGHTS):
        weighted_queries.extend([eq] * weight)

    # Each round simulates a "session" of queries
    # Scale up over time (user becomes more active)
    n_queries = min(5 + round_num * 2, len(weighted_queries))
    selected = rng.sample(weighted_queries, k=min(n_queries, len(weighted_queries)))

    total_tracked = 0
    for eq in selected:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q],
            query_text=q,
            top_k=top_k,
            scoring=None,  # Usage queries don't use scoring — they build history
        )

        # Track access on returned results
        chunk_ids = []
        for r in results[:top_k]:
            row = store._conn.execute(
                "SELECT id FROM chunks WHERE text = ? LIMIT 1", [r.text]
            ).fetchone()
            if row:
                chunk_ids.append(row["id"])
        if chunk_ids:
            store.track_access(chunk_ids)
            total_tracked += len(chunk_ids)

    return total_tracked


# ------------------------------------------------------------------ #
# Experiment                                                           #
# ------------------------------------------------------------------ #


def run_cold_start(
    store: VstashStore,
    n_rounds: int = 20,
    top_k: int = 10,
    alpha: float = 0.5,
    beta: float = 0.5,
    decay_lambda: float = 0.10,
) -> ColdStartResult:
    """Run the cold start curve experiment."""

    scoring = ScoringConfig(
        enabled=True,
        alpha=alpha,
        beta=beta,
        decay_lambda=decay_lambda,
        over_fetch=50,
        track_access=False,  # Don't track eval queries — only usage queries
    )

    # Pre-compute embeddings
    query_embeddings = {}
    for eq in EVAL_QUERIES:
        query_embeddings[eq["query"]] = embed_query(eq["query"], MODEL)

    # Step 1: Baseline — scoring OFF, no access history
    baseline_ndcgs = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q],
            query_text=q,
            top_k=top_k,
            scoring=None,
        )
        titles = [r.title for r in results]
        baseline_ndcgs.append(ndcg_at_k(titles, eq["expected_top"], k=5))

    avg_baseline = sum(baseline_ndcgs) / len(baseline_ndcgs)

    # Reset all access counts to 0 (clean slate)
    store._conn.execute("UPDATE chunks SET access_count = 0, last_accessed_at = NULL")
    store._conn.commit()

    rounds: list[RoundResult] = []
    cumulative_accesses = 0

    for round_num in range(n_rounds):
        # Phase A: Simulate non-uniform usage (builds access history)
        usage_accesses = simulate_usage_round(
            store, query_embeddings, round_num, top_k
        )
        cumulative_accesses += usage_accesses

        # Phase B: Evaluate — run all queries with scoring ON
        scored_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q],
                query_text=q,
                top_k=top_k,
                scoring=scoring,
            )
            titles = [r.title for r in results]
            scored_ndcgs.append(ndcg_at_k(titles, eq["expected_top"], k=5))

        avg_scored = sum(scored_ndcgs) / len(scored_ndcgs)
        delta = avg_scored - avg_baseline
        delta_pct = (delta / avg_baseline * 100) if avg_baseline > 0 else 0.0

        rounds.append(
            RoundResult(
                round_num=round_num + 1,
                scored_ndcg_5=avg_scored,
                baseline_ndcg_5=avg_baseline,
                delta=delta,
                delta_pct=delta_pct,
                total_accesses=cumulative_accesses,
                usage_queries_this_round=5 + round_num * 2,
                per_query_scored=scored_ndcgs,
                per_query_baseline=baseline_ndcgs,
            )
        )

    # Find crossover: first round where scored > baseline for 3+ consecutive
    crossover = None
    consecutive = 0
    for r in rounds:
        if r.delta > 0:
            consecutive += 1
            if consecutive >= 3 and crossover is None:
                crossover = r.round_num - 2  # first of the 3 consecutive
        else:
            consecutive = 0

    return ColdStartResult(
        rounds=rounds,
        crossover_round=crossover,
        total_rounds=n_rounds,
        corpus_size=len(PAPERS),
        n_queries=len(EVAL_QUERIES),
        scoring_config={
            "alpha": alpha,
            "beta": beta,
            "decay_lambda": decay_lambda,
        },
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold Start Curve Experiment")
    parser.add_argument("--db", type=str, help="Reuse existing DB")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=str, default="experiments/results/cold_start.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    if args.db and Path(args.db).exists():
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"Reusing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_coldstart_")
        db_path = str(Path(tmp_dir) / "cold_start.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        print(f"Ingested: {n}/{len(PAPERS)}\n")

    sep = "=" * 80
    print(f"\n{sep}")
    print("  COLD START CURVE: Queries needed before scoring helps")
    print(f"{sep}\n")

    result = run_cold_start(store, n_rounds=args.rounds, top_k=args.top_k)

    # Print results
    print(f"  Baseline NDCG@5 (no scoring): {result.rounds[0].baseline_ndcg_5:.4f}")
    print(f"  Scoring config: α={result.scoring_config['alpha']}, "
          f"β={result.scoring_config['beta']}, "
          f"λ={result.scoring_config['decay_lambda']}\n")

    print(f"  {'Round':>5} {'Scored':>8} {'Base':>8} {'Δ':>8} {'Δ%':>8} {'Uses':>6} {'Accesses':>10}")
    print(f"  {'─' * 58}")
    for r in result.rounds:
        marker = " ◄" if r.round_num == result.crossover_round else ""
        print(
            f"  {r.round_num:>5} {r.scored_ndcg_5:>8.4f} {r.baseline_ndcg_5:>8.4f} "
            f"{r.delta:>+8.4f} {r.delta_pct:>+7.2f}% {r.usage_queries_this_round:>6} "
            f"{r.total_accesses:>10}{marker}"
        )

    print(f"\n{sep}")
    if result.crossover_round is not None:
        crossover_data = result.rounds[result.crossover_round - 1]
        print(f"  CROSSOVER at round {result.crossover_round} "
              f"({crossover_data.total_accesses} cumulative accesses)")
        print(f"  Scoring starts consistently helping after "
              f"~{result.crossover_round} usage sessions")
    else:
        positive_rounds = sum(1 for r in result.rounds if r.delta > 0)
        if positive_rounds > 0:
            print(f"  No stable crossover found in {result.total_rounds} rounds")
            print(f"  Scoring was positive in {positive_rounds}/{result.total_rounds} rounds")
            best = max(result.rounds, key=lambda r: r.delta)
            print(f"  Best improvement: round {best.round_num} "
                  f"(Δ={best.delta:+.4f}, {best.delta_pct:+.2f}%)")
        else:
            print(f"  Scoring did not outperform baseline in {result.total_rounds} rounds")
            print("  This may indicate the corpus needs more diverse access patterns")
    print(f"{sep}\n")

    # Per-query analysis: which queries benefit most from scoring?
    if result.rounds:
        last = result.rounds[-1]
        print("  Per-query analysis (final round):")
        for i, eq in enumerate(EVAL_QUERIES):
            q = eq["query"][:55]
            scored = last.per_query_scored[i]
            base = last.per_query_baseline[i]
            d = scored - base
            print(f"    {q:<55} {scored:.3f} vs {base:.3f} ({d:+.3f})")
        print()

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "cold_start",
        "db_path": db_path,
        **asdict(result),
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
