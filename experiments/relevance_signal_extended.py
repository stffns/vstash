"""Relevance Signal at Scale — Extended Threshold Calibration (n=100+)

Extends the original relevance_signal.py from n=20 to n=120 queries
(60 relevant + 60 irrelevant) and adds bootstrap confidence intervals
for F1, precision, recall, and accuracy.

Closes paper Limitation #1: "requires validation on 100+ queries."

Usage:
    python -m experiments.relevance_signal_extended [--db PATH] [--output PATH]
    python -m experiments.relevance_signal_extended --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.relevance_signal import (
    ThresholdMetrics,
)
from experiments.scoring_grid import PAPERS, ingest_papers
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Extended query sets — 60 relevant + 60 irrelevant                    #
# ------------------------------------------------------------------ #

# The original 15 relevant queries, plus 45 more on topics covered by
# the 24-paper LLM-memory corpus.
RELEVANT_QUERIES = [
    # --- original 15 ---
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
    # --- 45 additional ---
    "retrieval augmented generation for question answering",
    "vector database for semantic memory",
    "embedding similarity search for document retrieval",
    "context window limitations in large language models",
    "working memory versus long-term memory in AI",
    "memory consolidation strategies for chatbots",
    "persistent state management in LLM agents",
    "attention mechanism and memory in transformers",
    "external memory banks for neural networks",
    "recursive summarization for memory compression",
    "memory-based reinforcement learning agents",
    "experience replay in language model training",
    "hierarchical memory organization for AI",
    "associative memory models in neural computation",
    "memory retrieval latency optimization",
    "semantic search and approximate nearest neighbors",
    "BM25 versus dense retrieval for information retrieval",
    "hybrid search combining keyword and vector matching",
    "reciprocal rank fusion for multi-signal retrieval",
    "document chunking strategies for RAG systems",
    "embedding model selection for retrieval tasks",
    "fine-tuning embeddings for domain-specific retrieval",
    "knowledge distillation in retrieval systems",
    "cross-encoder reranking for document retrieval",
    "generative agents with persistent memory",
    "memory-augmented transformers architecture",
    "sparse and dense retrieval comparison",
    "passage retrieval for open domain question answering",
    "neural information retrieval benchmarks",
    "query expansion techniques for better recall",
    "negative sampling strategies for embedding training",
    "contrastive learning for text embeddings",
    "bi-encoder versus cross-encoder for retrieval",
    "multi-hop reasoning over retrieved documents",
    "chain of thought with retrieved context",
    "prompt engineering with retrieval augmented context",
    "memory injection strategies for language models",
    "session-based memory for conversational AI",
    "forgetting mechanism in lifelong learning agents",
    "continual learning and catastrophic forgetting",
    "memory priority and importance scoring",
    "reflection and self-evaluation in AI agents",
    "planning with memory retrieval in autonomous agents",
    "entity extraction for knowledge graph construction",
    "temporal reasoning in memory-augmented models",
    "memory-efficient inference in large models",
]

# 60 off-topic queries that should NOT match the memory-papers corpus.
IRRELEVANT_QUERIES = [
    # --- original 15 ---
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
    # --- 45 additional ---
    "plumbing repair for leaky kitchen faucet",
    "best hiking trails in Patagonia Argentina",
    "how to bake a chocolate lava cake",
    "rules of cricket test match format",
    "growing tomatoes in a greenhouse",
    "history of the Ottoman Empire expansion",
    "beginner guitar chords and strumming patterns",
    "how to knit a scarf for beginners",
    "vintage car restoration tips for 1960s Mustang",
    "scuba diving certification requirements",
    "home brewing craft beer IPA recipe",
    "training for a marathon in twelve weeks",
    "woodworking joints for beginner projects",
    "Mediterranean diet meal plan for weight loss",
    "how to build a raised garden bed",
    "basics of watercolor painting technique",
    "motorcycle road trip across Route 66",
    "origami crane folding step by step",
    "astrophotography camera settings for Milky Way",
    "history of jazz music in New Orleans",
    "building a model railroad layout HO scale",
    "essential oils for aromatherapy diffuser",
    "fly fishing techniques for trout in streams",
    "how to install hardwood flooring",
    "caring for houseplants in low light conditions",
    "basics of celestial navigation at sea",
    "how to make pottery on a wheel",
    "salsa dancing steps for beginners",
    "beekeeping for beginners getting started",
    "history of samurai warriors in feudal Japan",
    "how to grill the perfect steak",
    "candle making with essential oils at home",
    "bird watching guide for eastern United States",
    "how to play chess openings for beginners",
    "snowboarding technique for carving turns",
    "landscape photography composition rules",
    "history of ancient Egyptian pyramids",
    "how to tune a piano by ear",
    "organic composting methods for garden soil",
    "basics of calligraphy with brush pen",
    "surfing wave selection and paddle technique",
    "bookbinding by hand Japanese stab stitch",
    "fermentation basics for kimchi and sauerkraut",
    "how to plan a backpacking trip through Europe",
    "leather working tools and techniques for wallets",
    "basics of bread scoring patterns for artisan loaves",
]


# ------------------------------------------------------------------ #
# Evaluate (uses best_distance, the actual production signal)           #
# ------------------------------------------------------------------ #


@dataclass
class DistanceResult:
    """Result using best_distance as the relevance signal."""

    query: str
    is_relevant: bool
    best_distance: float
    relevance_tier: str
    n_results: int


def evaluate_queries_distance(
    store: VstashStore, queries: list[str], is_relevant: bool, top_k: int = 5
) -> list[DistanceResult]:
    """Run queries and collect best_distance (the actual production signal)."""
    from vstash.store import relevance_tier

    results = []
    for q in queries:
        emb = embed_query(q, MODEL)
        chunks = store.search(query_embedding=emb, query_text=q, top_k=top_k)
        bd = store.last_best_distance
        tier = relevance_tier(bd)
        results.append(
            DistanceResult(
                query=q,
                is_relevant=is_relevant,
                best_distance=bd,
                relevance_tier=tier,
                n_results=len(chunks),
            )
        )
    return results


def compute_distance_threshold_metrics(
    results: list[DistanceResult], threshold: float
) -> ThresholdMetrics:
    """Classify: predicted_relevant = best_distance <= threshold."""
    tp = fp = tn = fn = 0
    for r in results:
        predicted_relevant = r.best_distance <= threshold
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
    accuracy = (tp + tn) / len(results) if results else 0.0
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


def bootstrap_distance_ci(
    results: list[DistanceResult],
    threshold: float,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CI for distance-based classification."""
    rng = random.Random(seed)
    n = len(results)
    samples: dict[str, list[float]] = {"f1": [], "precision": [], "recall": [], "accuracy": []}

    for _ in range(n_bootstrap):
        sample = rng.choices(results, k=n)
        m = compute_distance_threshold_metrics(sample, threshold)
        samples["f1"].append(m.f1)
        samples["precision"].append(m.precision)
        samples["recall"].append(m.recall)
        samples["accuracy"].append(m.accuracy)

    alpha = (1 - ci) / 2
    result = {}
    for name, vals in samples.items():
        sorted_v = sorted(vals)
        lo = sorted_v[int(alpha * len(sorted_v))]
        hi = sorted_v[int((1 - alpha) * len(sorted_v))]
        result[name] = (round(lo, 4), round(hi, 4))
    return result


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Relevance signal at scale (n=120, bootstrap CI)")
    parser.add_argument("--db", type=str, help="Reuse existing DB")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/relevance_signal_extended.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    if args.db and Path(args.db).exists():
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"Reusing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_relevance_ext_")
        db_path = str(Path(tmp_dir) / "relevance_ext.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"New DB: {db_path}")
        print("Ingesting papers...")
        n = ingest_papers(store)
        print(f"Ingested: {n}/{len(PAPERS)}\n")

    n_rel = len(RELEVANT_QUERIES)
    n_irr = len(IRRELEVANT_QUERIES)
    n_total = n_rel + n_irr
    print(f"Queries: {n_rel} relevant + {n_irr} irrelevant = {n_total} total")
    print(f"Bootstrap: {args.n_bootstrap} iterations, 95% CI\n")

    # ============================================================== #
    # Phase 1: Collect best_distance for each query                    #
    # ============================================================== #
    print("Phase 1: Evaluating queries (distance-based signal)...")
    relevant_results = evaluate_queries_distance(
        store, RELEVANT_QUERIES, is_relevant=True, top_k=args.top_k
    )
    irrelevant_results = evaluate_queries_distance(
        store, IRRELEVANT_QUERIES, is_relevant=False, top_k=args.top_k
    )
    all_results = relevant_results + irrelevant_results

    # Summary stats
    rel_dists = [r.best_distance for r in relevant_results]
    irr_dists = [r.best_distance for r in irrelevant_results]

    print(f"\n  Relevant queries (n={len(rel_dists)}):")
    print(
        f"    Distance: mean={sum(rel_dists) / len(rel_dists):.4f}, "
        f"min={min(rel_dists):.4f}, max={max(rel_dists):.4f}"
    )
    print(f"  Irrelevant queries (n={len(irr_dists)}):")
    print(
        f"    Distance: mean={sum(irr_dists) / len(irr_dists):.4f}, "
        f"min={min(irr_dists):.4f}, max={max(irr_dists):.4f}"
    )

    # Tier distribution
    for label, results_list in [("Relevant", relevant_results), ("Irrelevant", irrelevant_results)]:
        tiers = {}
        for r in results_list:
            tiers[r.relevance_tier] = tiers.get(r.relevance_tier, 0) + 1
        print(f"  {label} tier distribution: {tiers}")

    # ============================================================== #
    # Phase 2: Distance threshold sweep + bootstrap CI                 #
    # ============================================================== #
    print("\nPhase 2: Distance threshold sweep with bootstrap CI...")

    # Sweep: predicted_relevant = best_distance <= threshold
    # Lower distance = more relevant, so threshold is an upper bound.
    thresholds = [0.80 + i * 0.005 for i in range(41)]  # 0.80 to 1.00
    best_f1 = 0.0
    best_threshold = 0.0
    best_metrics: ThresholdMetrics | None = None
    all_threshold_metrics = []

    for t in thresholds:
        m = compute_distance_threshold_metrics(all_results, t)
        all_threshold_metrics.append(m)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_threshold = t
            best_metrics = m

    assert best_metrics is not None

    print(f"\n  Best distance threshold: {best_threshold:.4f}")
    print(f"  F1:        {best_metrics.f1:.3f}")
    print(f"  Precision: {best_metrics.precision:.3f}")
    print(f"  Recall:    {best_metrics.recall:.3f}")
    print(f"  Accuracy:  {best_metrics.accuracy:.3f}")
    print(
        f"  TP={best_metrics.true_positives} FP={best_metrics.false_positives} "
        f"TN={best_metrics.true_negatives} FN={best_metrics.false_negatives}"
    )

    # Also report the production thresholds (0.95, 0.98)
    print("\n  Production thresholds (relevance_tier):")
    for t_name, t_val in [("high (<=0.95)", 0.95), ("medium (<=0.98)", 0.98)]:
        m = compute_distance_threshold_metrics(all_results, t_val)
        print(
            f"    {t_name}: F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f} "
            f"Acc={m.accuracy:.3f} (TP={m.true_positives} FP={m.false_positives} "
            f"TN={m.true_negatives} FN={m.false_negatives})"
        )

    # Bootstrap CI at best threshold
    print(f"\n  Bootstrap CI ({args.n_bootstrap} iterations, 95%):")
    ci_results = bootstrap_distance_ci(all_results, best_threshold, n_bootstrap=args.n_bootstrap)
    for metric_name, (lo, hi) in ci_results.items():
        print(f"    {metric_name:>10}: [{lo:.4f}, {hi:.4f}]")

    # Class separation check
    if max(rel_dists) < min(irr_dists):
        overlap = "none"
        print("\n  Class overlap: ZERO (clean separation)")
    else:
        overlap_count = sum(1 for rd in rel_dists for ird in irr_dists if rd >= ird)
        overlap = f"{overlap_count} crossings"
        print(f"\n  Class overlap: {overlap}")

    # ============================================================== #
    # Save results                                                      #
    # ============================================================== #
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "relevance_signal_extended",
        "signal": "best_distance",
        "n_relevant": n_rel,
        "n_irrelevant": n_irr,
        "n_total": n_total,
        "top_k": args.top_k,
        "model": MODEL,
        "n_bootstrap": args.n_bootstrap,
        "best_threshold": best_threshold,
        "best_metrics": asdict(best_metrics),
        "bootstrap_ci_95": ci_results,
        "class_overlap": overlap,
        "relevant_distance_summary": {
            "mean": sum(rel_dists) / len(rel_dists),
            "min": min(rel_dists),
            "max": max(rel_dists),
        },
        "irrelevant_distance_summary": {
            "mean": sum(irr_dists) / len(irr_dists),
            "min": min(irr_dists),
            "max": max(irr_dists),
        },
        "production_thresholds": {
            "high_0.95": asdict(compute_distance_threshold_metrics(all_results, 0.95)),
            "medium_0.98": asdict(compute_distance_threshold_metrics(all_results, 0.98)),
        },
        "threshold_sweep": [asdict(m) for m in all_threshold_metrics],
        "per_query": [
            {
                "query": r.query,
                "is_relevant": r.is_relevant,
                "best_distance": r.best_distance,
                "relevance_tier": r.relevance_tier,
                "n_results": r.n_results,
            }
            for r in all_results
        ],
    }

    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {output_path}")

    store.close()


if __name__ == "__main__":
    main()
