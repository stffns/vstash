"""Combined Relevance Signal Analysis

Tests whether combining vector distance (top-1) with spread improves
discrimination between relevant and irrelevant queries.

The hypothesis: spread alone gives 1.22x discrimination.
Vector distance should be much worse for off-topic queries,
providing the missing signal.

Usage:
    python -m experiments.combined_signal_analysis
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

from experiments.scoring_grid import EVAL_QUERIES, PAPERS, ingest_papers

MODEL = "BAAI/bge-small-en-v1.5"

IRRELEVANT_QUERIES = [
    "recipes for chocolate cake baking",
    "professional basketball NBA draft picks",
    "gardening tips for growing tomatoes",
    "history of ancient Egyptian pyramids",
    "how to train for a marathon",
    "best vacation destinations in Europe",
    "stock market investment strategies",
    "yoga poses for beginners",
    "how to fix a leaking faucet",
    "popular dog breeds for families",
]


@dataclass
class QuerySignal:
    query: str
    relevant: bool
    spread: float
    best_distance: float
    top1_score: float


def collect_signals(
    store: VstashStore,
    cfg: ScoringConfig | None,
    queries: list,
    relevant: bool,
    top_k: int = 5,
) -> list[QuerySignal]:
    signals = []
    for q in queries:
        query_text = q["query"] if isinstance(q, dict) else q
        emb = embed_query(query_text, MODEL)
        results = store.search(emb, query_text, top_k=top_k, scoring=cfg)
        if not results:
            continue
        scores = [r.score for r in results]
        spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
        signals.append(
            QuerySignal(
                query=query_text,
                relevant=relevant,
                spread=spread,
                best_distance=store.last_best_distance,
                top1_score=scores[0],
            )
        )
    return signals


def evaluate_signal(
    signals: list[QuerySignal],
    classify_fn,
    label: str,
) -> dict:
    tp = fp = tn = fn = 0
    for s in signals:
        predicted_relevant = classify_fn(s)
        if s.relevant and predicted_relevant:
            tp += 1
        elif not s.relevant and predicted_relevant:
            fp += 1
        elif not s.relevant and not predicted_relevant:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(signals) if signals else 0
    return {
        "method": label,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def main() -> None:
    dim = get_embedding_dim(MODEL)
    tmp_dir = tempfile.mkdtemp(prefix="vstash_combined_")
    db_path = str(Path(tmp_dir) / "combined.db")
    store = VstashStore(db_path, embedding_dim=dim)

    print("Ingesting papers...")
    n = ingest_papers(store)
    n_chunks = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Ingested: {n}/{len(PAPERS)} ({n_chunks} chunks)\n")

    sep = "=" * 80

    # ── Collect signals without scoring (raw RRF) ──
    print(f"{sep}")
    print("  PHASE 1: Raw RRF (no scoring)")
    print(f"{sep}\n")

    rel_signals = collect_signals(store, None, EVAL_QUERIES, relevant=True)
    irr_signals = collect_signals(store, None, IRRELEVANT_QUERIES, relevant=False)
    all_signals = rel_signals + irr_signals

    print(f"  {'Query':<55} {'Dist':>6} {'Spread':>8} {'Score':>7} {'Class':>10}")
    print(f"  {'─' * 88}")
    for s in sorted(all_signals, key=lambda x: x.best_distance):
        label = "relevant" if s.relevant else "IRRELEVANT"
        print(
            f"  {s.query[:55]:<55} {s.best_distance:>6.4f} {s.spread:>8.4f} {s.top1_score:>7.4f} {label:>10}"
        )

    rel_dists = [s.best_distance for s in rel_signals]
    irr_dists = [s.best_distance for s in irr_signals]
    dist_ratio = (sum(irr_dists) / len(irr_dists)) / (sum(rel_dists) / len(rel_dists))

    print(
        f"\n  Distance: relevant avg={sum(rel_dists) / len(rel_dists):.4f}, irrelevant avg={sum(irr_dists) / len(irr_dists):.4f}, ratio={dist_ratio:.2f}x"
    )
    print(
        f"  Overlap: {sum(1 for d in irr_dists if d <= max(rel_dists))}/{len(irr_dists)} irrelevant within relevant range"
    )

    # ── Test different classification strategies ──
    print(f"\n{sep}")
    print("  CLASSIFICATION STRATEGIES (Raw RRF)")
    print(f"{sep}\n")

    # Find optimal distance threshold
    all_dists = sorted(set(rel_dists + irr_dists))
    best_dist_f1 = 0
    best_dist_t = 0
    for d in all_dists:
        m = evaluate_signal(all_signals, lambda s, t=d: s.best_distance <= t, "")
        if m["f1"] > best_dist_f1:
            best_dist_f1 = m["f1"]
            best_dist_t = d

    strategies = [
        ("spread > 0.15 (fixed)", lambda s: s.spread > 0.15),
        (f"distance < {best_dist_t:.4f} (optimal)", lambda s: s.best_distance < best_dist_t),
        ("distance < 0.90", lambda s: s.best_distance < 0.90),
        ("distance < 0.95", lambda s: s.best_distance < 0.95),
        ("distance < 1.00", lambda s: s.best_distance < 1.00),
        (
            "distance < 0.95 AND spread > 0.005",
            lambda s: s.best_distance < 0.95 and s.spread > 0.005,
        ),
        (
            "distance < 1.00 AND spread > 0.005",
            lambda s: s.best_distance < 1.00 and s.spread > 0.005,
        ),
    ]

    print(f"  {'Strategy':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'─' * 71}")

    results = []
    for label, fn in strategies:
        m = evaluate_signal(all_signals, fn, label)
        print(
            f"  {label:<45} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f}"
        )
        results.append(m)

    # ── Now with scoring enabled + usage history ──
    print(f"\n{sep}")
    print("  PHASE 2: With scoring + 10 rounds usage")
    print(f"{sep}\n")

    cfg = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)
    for _ in range(10):
        for eq in EVAL_QUERIES:
            emb = embed_query(eq["query"], MODEL)
            store.search(emb, eq["query"], top_k=5, scoring=cfg)

    rel_signals_s = collect_signals(store, cfg, EVAL_QUERIES, relevant=True)
    irr_signals_s = collect_signals(store, cfg, IRRELEVANT_QUERIES, relevant=False)
    all_signals_s = rel_signals_s + irr_signals_s

    print(f"  {'Query':<55} {'Dist':>6} {'Spread':>8} {'Score':>7} {'Class':>10}")
    print(f"  {'─' * 88}")
    for s in sorted(all_signals_s, key=lambda x: x.best_distance):
        label = "relevant" if s.relevant else "IRRELEVANT"
        print(
            f"  {s.query[:55]:<55} {s.best_distance:>6.4f} {s.spread:>8.4f} {s.top1_score:>7.4f} {label:>10}"
        )

    rel_dists_s = [s.best_distance for s in rel_signals_s]
    irr_dists_s = [s.best_distance for s in irr_signals_s]

    # Find optimal combined
    all_dists_s = sorted(set(rel_dists_s + irr_dists_s))
    best_dist_f1_s = 0
    best_dist_t_s = 0
    for d in all_dists_s:
        m = evaluate_signal(all_signals_s, lambda s, t=d: s.best_distance <= t, "")
        if m["f1"] > best_dist_f1_s:
            best_dist_f1_s = m["f1"]
            best_dist_t_s = d

    strategies_s = [
        ("spread > 0.15 (fixed)", lambda s: s.spread > 0.15),
        ("spread > adaptive", lambda s: s.spread > 0.20),
        (f"distance < {best_dist_t_s:.4f} (optimal)", lambda s: s.best_distance < best_dist_t_s),
        ("distance < 0.95", lambda s: s.best_distance < 0.95),
        ("distance < 1.00", lambda s: s.best_distance < 1.00),
        ("distance < 0.95 OR spread > 0.30", lambda s: s.best_distance < 0.95 or s.spread > 0.30),
        ("distance < 1.00 AND spread > 0.15", lambda s: s.best_distance < 1.00 and s.spread > 0.15),
        ("distance < 1.00 OR spread > 0.35", lambda s: s.best_distance < 1.00 or s.spread > 0.35),
    ]

    print(f"\n  {'Strategy':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'─' * 71}")

    results_s = []
    for label, fn in strategies_s:
        m = evaluate_signal(all_signals_s, fn, label)
        print(
            f"  {label:<45} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f}"
        )
        results_s.append(m)

    # Save
    output_path = Path("experiments/results/combined_signal_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "raw_rrf": {
            "signals": [
                {
                    "query": s.query,
                    "relevant": s.relevant,
                    "distance": round(s.best_distance, 4),
                    "spread": round(s.spread, 4),
                }
                for s in all_signals
            ],
            "strategies": results,
        },
        "with_scoring": {
            "signals": [
                {
                    "query": s.query,
                    "relevant": s.relevant,
                    "distance": round(s.best_distance, 4),
                    "spread": round(s.spread, 4),
                }
                for s in all_signals_s
            ],
            "strategies": results_s,
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")
    store.close()


if __name__ == "__main__":
    main()
