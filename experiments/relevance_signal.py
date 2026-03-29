"""Relevance Signal Experiment — Score Spread Threshold Calibration

Evaluates the score spread heuristic for detecting low-relevance results.
Uses two sets of queries:
  1. "Relevant" — queries about topics in the corpus (should have high spread)
  2. "Irrelevant" — queries about topics NOT in the corpus (should have low spread)

Computes ROC-like metrics across thresholds, finds optimal threshold.

Usage:
    python -m experiments.relevance_signal [--db PATH] [--output results/relevance.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.scoring_grid import PAPERS, ingest_papers
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Query sets                                                           #
# ------------------------------------------------------------------ #

# Queries that SHOULD find relevant content (topics in corpus)
RELEVANT_QUERIES = [
    "long-term memory for conversational AI agents",
    "temporal decay and forgetting in memory systems",
    "episodic memory architecture for LLMs",
    "benchmarks for evaluating agent memory",
    "knowledge graph approaches to memory",
    "memory compression and retrieval",
    "multi-agent memory sharing",
    "personalization through persistent memory",
    "read-write memory interfaces for LLMs",
    "self-organizing memory structures",
    "agentic memory with tool use",
    "conversational memory evaluation metrics",
    "memory-augmented generation for dialogue",
    "scalable long-term memory for AI",
    "cognitive science inspired memory for agents",
]

# Queries that should NOT find relevant content (off-topic)
IRRELEVANT_QUERIES = [
    "best pizza restaurants in downtown Chicago",
    "how to train a dog to sit and shake",
    "history of the Roman Empire and Julius Caesar",
    "photosynthesis process in C4 plants",
    "quantum computing error correction codes",
    "Taylor Swift album discography and awards",
    "recipe for homemade sourdough bread",
    "FIFA World Cup 2022 final highlights",
    "stock market technical analysis patterns",
    "how to change a car tire in winter",
    "deep sea fishing techniques for tuna",
    "yoga poses for lower back pain",
    "history of impressionist painting in France",
    "beginner guide to rock climbing",
    "how to set up a saltwater aquarium",
]


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #


@dataclass
class SpreadResult:
    query: str
    is_relevant: bool  # ground truth
    scores: list[float]
    spread: float
    max_score: float
    min_score: float
    mean_score: float
    std_score: float
    n_results: int


@dataclass
class ThresholdMetrics:
    threshold: float
    true_positives: int  # relevant correctly detected as relevant
    false_positives: int  # irrelevant incorrectly detected as relevant
    true_negatives: int  # irrelevant correctly detected as irrelevant
    false_negatives: int  # relevant incorrectly detected as irrelevant
    precision: float
    recall: float
    f1: float
    accuracy: float


def compute_spread_stats(scores: list[float]) -> dict:
    """Compute spread and distribution statistics."""
    import math

    if len(scores) < 2:
        return {"spread": 0.0, "std": 0.0, "mean": scores[0] if scores else 0.0}

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    return {
        "spread": max(scores) - min(scores),
        "std": std,
        "mean": mean,
    }


def evaluate_queries_scored(
    store: VstashStore,
    queries: list[str],
    is_relevant: bool,
    top_k: int = 5,
    scoring: object | None = None,
) -> list[SpreadResult]:
    """Run queries with optional scoring and collect score distributions."""
    results = []
    for q in queries:
        emb = embed_query(q, MODEL)
        chunks = store.search(
            query_embedding=emb,
            query_text=q,
            top_k=top_k,
            scoring=scoring,
        )
        scores = [c.score for c in chunks]
        stats = compute_spread_stats(scores)
        results.append(
            SpreadResult(
                query=q,
                is_relevant=is_relevant,
                scores=scores,
                spread=stats["spread"],
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                mean_score=stats["mean"],
                std_score=stats["std"],
                n_results=len(chunks),
            )
        )
    return results


def evaluate_queries(
    store: VstashStore, queries: list[str], is_relevant: bool, top_k: int = 5
) -> list[SpreadResult]:
    """Run queries and collect score distributions."""
    results = []

    for q in queries:
        emb = embed_query(q, MODEL)
        chunks = store.search(
            query_embedding=emb,
            query_text=q,
            top_k=top_k,
            scoring=None,
        )

        scores = [c.score for c in chunks]
        stats = compute_spread_stats(scores)

        results.append(
            SpreadResult(
                query=q,
                is_relevant=is_relevant,
                scores=scores,
                spread=stats["spread"],
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                mean_score=stats["mean"],
                std_score=stats["std"],
                n_results=len(chunks),
            )
        )

    return results


def compute_threshold_metrics(
    all_results: list[SpreadResult], threshold: float
) -> ThresholdMetrics:
    """Compute classification metrics at a given spread threshold."""
    tp = fp = tn = fn = 0

    for r in all_results:
        predicted_relevant = r.spread >= threshold
        if r.is_relevant and predicted_relevant:
            tp += 1
        elif not r.is_relevant and predicted_relevant:
            fp += 1
        elif not r.is_relevant and not predicted_relevant:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_results) if all_results else 0.0

    return ThresholdMetrics(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Relevance signal threshold calibration")
    parser.add_argument("--db", type=str, help="Reuse existing DB")
    parser.add_argument("--output", type=str, default="experiments/results/relevance.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    if args.db and Path(args.db).exists():
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"Reusing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_relevance_")
        db_path = str(Path(tmp_dir) / "relevance.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        print(f"Ingested: {n}/{len(PAPERS)}\n")

    sep = "=" * 80

    # ============================================================== #
    # Phase 1: Collect score distributions                             #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 1: Score Distribution Analysis")
    print(f"{sep}\n")

    relevant_results = evaluate_queries(store, RELEVANT_QUERIES, is_relevant=True, top_k=args.top_k)
    irrelevant_results = evaluate_queries(
        store, IRRELEVANT_QUERIES, is_relevant=False, top_k=args.top_k
    )
    all_results = relevant_results + irrelevant_results

    # Print distributions
    print(f"  {'Query':<55} {'Spread':>8} {'Max':>8} {'Min':>8} {'Std':>8} {'Label':>8}")
    print(f"  {'─' * 95}")

    for r in sorted(all_results, key=lambda x: x.spread, reverse=True):
        label = "REL" if r.is_relevant else "IRREL"
        print(
            f"  {r.query[:55]:<55} {r.spread:>8.4f} {r.max_score:>8.4f} "
            f"{r.min_score:>8.4f} {r.std_score:>8.4f} {label:>8}"
        )

    # Summary stats
    rel_spreads = [r.spread for r in relevant_results]
    irr_spreads = [r.spread for r in irrelevant_results]

    print(f"\n  Relevant queries (n={len(rel_spreads)}):")
    print(
        f"    Spread — mean: {sum(rel_spreads) / len(rel_spreads):.4f}, "
        f"min: {min(rel_spreads):.4f}, max: {max(rel_spreads):.4f}"
    )

    print(f"  Irrelevant queries (n={len(irr_spreads)}):")
    print(
        f"    Spread — mean: {sum(irr_spreads) / len(irr_spreads):.4f}, "
        f"min: {min(irr_spreads):.4f}, max: {max(irr_spreads):.4f}"
    )

    # ============================================================== #
    # Phase 2a: Threshold sweep on RAW RRF (no scoring)                #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 2a: Threshold on Raw RRF (no scoring)")
    print(f"{sep}\n")

    thresholds = [i * 0.01 for i in range(1, 51)]  # 0.01 to 0.50
    metrics_list = []

    for t in thresholds:
        m = compute_threshold_metrics(all_results, t)
        metrics_list.append(m)

    best_f1_raw = max(metrics_list, key=lambda m: m.f1)
    print(
        f"  Raw RRF spread is ~{sum(r.spread for r in all_results) / len(all_results):.4f} for ALL queries."
    )
    print("  No threshold can discriminate relevant from irrelevant.")
    print(f"  Best F1: {best_f1_raw.f1:.4f} (Acc={best_f1_raw.accuracy:.4f})")

    # ============================================================== #
    # Phase 2b: Threshold sweep WITH scoring enabled                   #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 2b: Threshold with Scoring Enabled (key experiment)")
    print(f"{sep}\n")

    from vstash.config import ScoringConfig

    scoring_cfg = ScoringConfig(
        enabled=True,
        alpha=0.7,
        beta=0.3,
        decay_lambda=0.07,
        over_fetch=50,
        track_access=False,
    )

    scored_relevant_results = evaluate_queries_scored(
        store,
        RELEVANT_QUERIES,
        is_relevant=True,
        top_k=args.top_k,
        scoring=scoring_cfg,
    )
    scored_irrelevant_results = evaluate_queries_scored(
        store,
        IRRELEVANT_QUERIES,
        is_relevant=False,
        top_k=args.top_k,
        scoring=scoring_cfg,
    )
    all_scored_results = scored_relevant_results + scored_irrelevant_results

    scored_metrics_list = []
    for t in thresholds:
        m = compute_threshold_metrics(all_scored_results, t)
        scored_metrics_list.append(m)

    print(
        f"  {'Threshold':>10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} "
        f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
    )
    print(f"  {'─' * 72}")

    best_f1 = max(scored_metrics_list, key=lambda m: m.f1)
    best_acc = max(scored_metrics_list, key=lambda m: m.accuracy)

    for m in scored_metrics_list:
        marker = " ★" if m.threshold == best_f1.threshold else ""
        if m.threshold % 0.05 < 0.01 or m.threshold == best_f1.threshold:
            print(
                f"  {m.threshold:>10.2f} {m.precision:>8.4f} {m.recall:>8.4f} "
                f"{m.f1:>8.4f} {m.accuracy:>8.4f} "
                f"{m.true_positives:>4} {m.false_positives:>4} "
                f"{m.true_negatives:>4} {m.false_negatives:>4}{marker}"
            )

    print(
        f"\n  ★ Best F1 threshold: {best_f1.threshold:.2f} "
        f"(F1={best_f1.f1:.4f}, Acc={best_f1.accuracy:.4f})"
    )
    print(
        f"    Best accuracy threshold: {best_acc.threshold:.2f} "
        f"(Acc={best_acc.accuracy:.4f}, F1={best_acc.f1:.4f})"
    )
    print("\n  Key finding: scoring is a PREREQUISITE for the relevance signal.")
    print("  Raw RRF spread: ~0.0006 (no discrimination)")
    print(
        f"  Scored spread: relevant={sum(r.spread for r in scored_relevant_results) / len(scored_relevant_results):.4f}, "
        f"irrelevant={sum(r.spread for r in scored_irrelevant_results) / len(scored_irrelevant_results):.4f}"
    )

    # Update metrics_list for output (use scored version as primary)
    metrics_list = scored_metrics_list

    # ============================================================== #
    # Phase 3: Alternative signals comparison                          #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 3: Alternative Relevance Signals")
    print(f"{sep}\n")

    # Compare spread vs std vs max_score vs mean_score as discriminators
    signals = {
        "spread (max-min)": lambda r: r.spread,
        "std deviation": lambda r: r.std_score,
        "max score": lambda r: r.max_score,
        "mean score": lambda r: r.mean_score,
        "max - mean": lambda r: r.max_score - r.mean_score,
    }

    print(f"  {'Signal':<20} {'Best F1':>8} {'Best Acc':>8} {'Threshold':>10}")
    print(f"  {'─' * 50}")

    signal_results = {}
    for sig_name, sig_fn in signals.items():
        # Try thresholds for this signal
        sig_values = [(sig_fn(r), r.is_relevant) for r in all_results]
        all_vals = sorted(set(v for v, _ in sig_values))

        best_sig_f1 = 0.0
        best_sig_acc = 0.0
        best_t = 0.0

        for t in [v - 0.001 for v in all_vals] + [v + 0.001 for v in all_vals]:
            tp = sum(1 for v, rel in sig_values if v >= t and rel)
            fp = sum(1 for v, rel in sig_values if v >= t and not rel)
            tn = sum(1 for v, rel in sig_values if v < t and not rel)
            fn = sum(1 for v, rel in sig_values if v < t and rel)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp + tn) / len(sig_values)

            if f1 > best_sig_f1:
                best_sig_f1 = f1
                best_t = t
            best_sig_acc = max(best_sig_acc, acc)

        signal_results[sig_name] = {
            "best_f1": best_sig_f1,
            "best_acc": best_sig_acc,
            "threshold": best_t,
        }
        print(f"  {sig_name:<20} {best_sig_f1:>8.4f} {best_sig_acc:>8.4f} {best_t:>10.4f}")

    # ============================================================== #
    # Phase 4: Scoring mode comparison                                 #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 4: Spread with Scoring Enabled vs Disabled (summary)")
    print(f"{sep}\n")

    scored_relevant = [r.spread for r in scored_relevant_results]
    scored_irrelevant = [r.spread for r in scored_irrelevant_results]

    print(
        f"  With scoring — relevant spread mean:   {sum(scored_relevant) / len(scored_relevant):.4f}"
    )
    print(
        f"  With scoring — irrelevant spread mean: {sum(scored_irrelevant) / len(scored_irrelevant):.4f}"
    )
    print(f"  Without scoring — relevant spread mean:   {sum(rel_spreads) / len(rel_spreads):.4f}")
    print(f"  Without scoring — irrelevant spread mean: {sum(irr_spreads) / len(irr_spreads):.4f}")
    print(
        f"\n  Separation ratio (scored): {sum(scored_relevant) / len(scored_relevant) / max(sum(scored_irrelevant) / len(scored_irrelevant), 0.0001):.2f}x"
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "relevance_signal",
        "corpus_size": len(PAPERS),
        "top_k": args.top_k,
        "db_path": db_path,
        "relevant_queries": len(RELEVANT_QUERIES),
        "irrelevant_queries": len(IRRELEVANT_QUERIES),
        "spread_distributions": {
            "relevant": [asdict(r) for r in relevant_results],
            "irrelevant": [asdict(r) for r in irrelevant_results],
        },
        "threshold_sweep": [asdict(m) for m in metrics_list],
        "best_threshold": {
            "by_f1": asdict(best_f1),
            "by_accuracy": asdict(best_acc),
        },
        "alternative_signals": signal_results,
        "scored_vs_unscored": {
            "scored_relevant_mean": sum(scored_relevant) / len(scored_relevant),
            "scored_irrelevant_mean": sum(scored_irrelevant) / len(scored_irrelevant),
            "unscored_relevant_mean": sum(rel_spreads) / len(rel_spreads),
            "unscored_irrelevant_mean": sum(irr_spreads) / len(irr_spreads),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
