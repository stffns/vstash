"""Relevance Signal Deep Analysis

Measures the actual discrimination power of the spread signal under
different conditions: fixed threshold, adaptive threshold, and with
varying amounts of access history. Tests whether the adaptive threshold
improves classification or just shifts the cutoff.

Usage:
    python -m experiments.relevance_signal_analysis
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

from experiments.scoring_grid import (
    EVAL_QUERIES,
    PAPERS,
    ingest_papers,
)

MODEL = "BAAI/bge-small-en-v1.5"

# Off-topic queries — should be flagged as irrelevant
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
class SignalMetrics:
    """Classification metrics for the relevance signal."""

    threshold: float
    true_positives: int  # relevant correctly flagged as high
    false_positives: int  # irrelevant incorrectly flagged as high
    true_negatives: int  # irrelevant correctly flagged as low
    false_negatives: int  # relevant incorrectly flagged as low
    relevant_spreads: list[float]
    irrelevant_spreads: list[float]

    @property
    def precision(self) -> float:
        tp = self.true_positives
        fp = self.false_positives
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.true_positives
        fn = self.false_negatives
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0


def collect_spreads(
    store: VstashStore,
    cfg: ScoringConfig,
    queries: list[dict | str],
    top_k: int = 5,
) -> list[float]:
    """Run queries and collect spreads."""
    spreads = []
    for q in queries:
        query_text = q["query"] if isinstance(q, dict) else q
        emb = embed_query(query_text, MODEL)
        results = store.search(emb, query_text, top_k=top_k, scoring=cfg)
        if len(results) > 1:
            scores = [r.score for r in results]
            spreads.append(max(scores) - min(scores))
    return spreads


def evaluate_threshold(
    relevant_spreads: list[float],
    irrelevant_spreads: list[float],
    threshold: float,
) -> SignalMetrics:
    """Evaluate a threshold against spread distributions."""
    tp = sum(1 for s in relevant_spreads if s >= threshold)
    fn = sum(1 for s in relevant_spreads if s < threshold)
    fp = sum(1 for s in irrelevant_spreads if s >= threshold)
    tn = sum(1 for s in irrelevant_spreads if s < threshold)
    return SignalMetrics(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        relevant_spreads=relevant_spreads,
        irrelevant_spreads=irrelevant_spreads,
    )


def find_optimal_threshold(
    relevant_spreads: list[float],
    irrelevant_spreads: list[float],
) -> tuple[float, SignalMetrics]:
    """Find the threshold that maximizes F1."""
    all_spreads = sorted(set(relevant_spreads + irrelevant_spreads))
    best_f1 = 0.0
    best_threshold = 0.15
    best_metrics = None
    for s in all_spreads:
        m = evaluate_threshold(relevant_spreads, irrelevant_spreads, s)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_threshold = s
            best_metrics = m
    if best_metrics is None:
        best_metrics = evaluate_threshold(relevant_spreads, irrelevant_spreads, best_threshold)
    return best_threshold, best_metrics


def main() -> None:
    dim = get_embedding_dim(MODEL)
    tmp_dir = tempfile.mkdtemp(prefix="vstash_signal_")
    db_path = str(Path(tmp_dir) / "signal.db")
    store = VstashStore(db_path, embedding_dim=dim)

    print("Ingesting papers...")
    n = ingest_papers(store)
    n_chunks = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Ingested: {n}/{len(PAPERS)} ({n_chunks} chunks)\n")

    sep = "=" * 80

    # ── Phase 1: No access history (cold start) ──
    print(f"{sep}")
    print("  PHASE 1: Cold Start (no access history)")
    print(f"{sep}\n")

    cfg_scoring = ScoringConfig(enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05)

    rel_spreads_cold = collect_spreads(store, cfg_scoring, EVAL_QUERIES)
    irr_spreads_cold = collect_spreads(store, cfg_scoring, IRRELEVANT_QUERIES)

    avg_rel = sum(rel_spreads_cold) / len(rel_spreads_cold)
    avg_irr = sum(irr_spreads_cold) / len(irr_spreads_cold)
    ratio_cold = avg_rel / avg_irr if avg_irr > 0 else float("inf")

    print(
        f"  Relevant spreads:   min={min(rel_spreads_cold):.4f}  avg={avg_rel:.4f}  max={max(rel_spreads_cold):.4f}"
    )
    print(
        f"  Irrelevant spreads: min={min(irr_spreads_cold):.4f}  avg={avg_irr:.4f}  max={max(irr_spreads_cold):.4f}"
    )
    print(f"  Discrimination ratio: {ratio_cold:.2f}x")
    print(
        f"  Overlap: {sum(1 for s in irr_spreads_cold if s >= min(rel_spreads_cold))}/{len(irr_spreads_cold)} irrelevant exceed min relevant"
    )

    # Fixed threshold
    m_fixed = evaluate_threshold(rel_spreads_cold, irr_spreads_cold, 0.15)
    print(
        f"\n  Fixed τ=0.15:   Precision={m_fixed.precision:.3f}  Recall={m_fixed.recall:.3f}  F1={m_fixed.f1:.3f}  Acc={m_fixed.accuracy:.3f}"
    )

    # Optimal threshold
    opt_t, m_opt = find_optimal_threshold(rel_spreads_cold, irr_spreads_cold)
    print(
        f"  Optimal τ={opt_t:.4f}: Precision={m_opt.precision:.3f}  Recall={m_opt.recall:.3f}  F1={m_opt.f1:.3f}  Acc={m_opt.accuracy:.3f}"
    )

    # ── Phase 2: Build access history (warm up) ──
    print(f"\n{sep}")
    print("  PHASE 2: After 10 rounds of access history")
    print(f"{sep}\n")

    # Simulate non-uniform usage: query relevant topics more
    for _ in range(10):
        for eq in EVAL_QUERIES:
            emb = embed_query(eq["query"], MODEL)
            store.search(emb, eq["query"], top_k=5, scoring=cfg_scoring)

    rel_spreads_warm = collect_spreads(store, cfg_scoring, EVAL_QUERIES)
    irr_spreads_warm = collect_spreads(store, cfg_scoring, IRRELEVANT_QUERIES)

    avg_rel_w = sum(rel_spreads_warm) / len(rel_spreads_warm)
    avg_irr_w = sum(irr_spreads_warm) / len(irr_spreads_warm)
    ratio_warm = avg_rel_w / avg_irr_w if avg_irr_w > 0 else float("inf")

    print(
        f"  Relevant spreads:   min={min(rel_spreads_warm):.4f}  avg={avg_rel_w:.4f}  max={max(rel_spreads_warm):.4f}"
    )
    print(
        f"  Irrelevant spreads: min={min(irr_spreads_warm):.4f}  avg={avg_irr_w:.4f}  max={max(irr_spreads_warm):.4f}"
    )
    print(f"  Discrimination ratio: {ratio_warm:.2f}x")
    print(
        f"  Overlap: {sum(1 for s in irr_spreads_warm if s >= min(rel_spreads_warm))}/{len(irr_spreads_warm)} irrelevant exceed min relevant"
    )

    m_fixed_w = evaluate_threshold(rel_spreads_warm, irr_spreads_warm, 0.15)
    print(
        f"\n  Fixed τ=0.15:   Precision={m_fixed_w.precision:.3f}  Recall={m_fixed_w.recall:.3f}  F1={m_fixed_w.f1:.3f}  Acc={m_fixed_w.accuracy:.3f}"
    )

    opt_t_w, m_opt_w = find_optimal_threshold(rel_spreads_warm, irr_spreads_warm)
    print(
        f"  Optimal τ={opt_t_w:.4f}: Precision={m_opt_w.precision:.3f}  Recall={m_opt_w.recall:.3f}  F1={m_opt_w.f1:.3f}  Acc={m_opt_w.accuracy:.3f}"
    )

    # Adaptive threshold (from stored spreads during warm-up)
    # Feed the warm spreads into the adaptive system
    for s in rel_spreads_warm + irr_spreads_warm:
        store.record_spread(s)
    adaptive_t = store.adaptive_relevance_threshold()
    m_adaptive = evaluate_threshold(rel_spreads_warm, irr_spreads_warm, adaptive_t)
    print(
        f"  Adaptive τ={adaptive_t:.4f}: Precision={m_adaptive.precision:.3f}  Recall={m_adaptive.recall:.3f}  F1={m_adaptive.f1:.3f}  Acc={m_adaptive.accuracy:.3f}"
    )

    # ── Phase 3: Heavy usage (30 rounds) ──
    print(f"\n{sep}")
    print("  PHASE 3: After 30 rounds (heavy usage)")
    print(f"{sep}\n")

    for _ in range(20):  # 20 more rounds
        for eq in EVAL_QUERIES:
            emb = embed_query(eq["query"], MODEL)
            store.search(emb, eq["query"], top_k=5, scoring=cfg_scoring)

    rel_spreads_heavy = collect_spreads(store, cfg_scoring, EVAL_QUERIES)
    irr_spreads_heavy = collect_spreads(store, cfg_scoring, IRRELEVANT_QUERIES)

    avg_rel_h = sum(rel_spreads_heavy) / len(rel_spreads_heavy)
    avg_irr_h = sum(irr_spreads_heavy) / len(irr_spreads_heavy)
    ratio_heavy = avg_rel_h / avg_irr_h if avg_irr_h > 0 else float("inf")

    print(
        f"  Relevant spreads:   min={min(rel_spreads_heavy):.4f}  avg={avg_rel_h:.4f}  max={max(rel_spreads_heavy):.4f}"
    )
    print(
        f"  Irrelevant spreads: min={min(irr_spreads_heavy):.4f}  avg={avg_irr_h:.4f}  max={max(irr_spreads_heavy):.4f}"
    )
    print(f"  Discrimination ratio: {ratio_heavy:.2f}x")
    print(
        f"  Overlap: {sum(1 for s in irr_spreads_heavy if s >= min(rel_spreads_heavy))}/{len(irr_spreads_heavy)} irrelevant exceed min relevant"
    )

    m_fixed_h = evaluate_threshold(rel_spreads_heavy, irr_spreads_heavy, 0.15)
    print(
        f"\n  Fixed τ=0.15:   Precision={m_fixed_h.precision:.3f}  Recall={m_fixed_h.recall:.3f}  F1={m_fixed_h.f1:.3f}  Acc={m_fixed_h.accuracy:.3f}"
    )

    opt_t_h, m_opt_h = find_optimal_threshold(rel_spreads_heavy, irr_spreads_heavy)
    print(
        f"  Optimal τ={opt_t_h:.4f}: Precision={m_opt_h.precision:.3f}  Recall={m_opt_h.recall:.3f}  F1={m_opt_h.f1:.3f}  Acc={m_opt_h.accuracy:.3f}"
    )

    for s in rel_spreads_heavy + irr_spreads_heavy:
        store.record_spread(s)
    adaptive_t_h = store.adaptive_relevance_threshold()
    m_adaptive_h = evaluate_threshold(rel_spreads_heavy, irr_spreads_heavy, adaptive_t_h)
    print(
        f"  Adaptive τ={adaptive_t_h:.4f}: Precision={m_adaptive_h.precision:.3f}  Recall={m_adaptive_h.recall:.3f}  F1={m_adaptive_h.f1:.3f}  Acc={m_adaptive_h.accuracy:.3f}"
    )

    # ── Per-query spread breakdown ──
    print(f"\n{sep}")
    print("  PER-QUERY SPREADS (Phase 3 — heavy usage)")
    print(f"{sep}\n")
    print(f"  {'Query':<55} {'Spread':>8} {'Class':>10}")
    print(f"  {'─' * 75}")
    for eq, s in zip(EVAL_QUERIES, rel_spreads_heavy):
        print(f"  {eq['query'][:55]:<55} {s:>8.4f} {'relevant':>10}")
    for q, s in zip(IRRELEVANT_QUERIES, irr_spreads_heavy):
        print(f"  {q[:55]:<55} {s:>8.4f} {'irrelevant':>10}")

    # ── Summary ──
    print(f"\n{sep}")
    print("  SUMMARY: Discrimination ratio over time")
    print(f"{sep}\n")
    print(f"  {'Phase':<25} {'Ratio':>8} {'Fixed F1':>10} {'Optimal F1':>12} {'Adaptive F1':>13}")
    print(f"  {'─' * 70}")
    print(
        f"  {'Cold (0 rounds)':<25} {ratio_cold:>8.2f}x {m_fixed.f1:>10.3f} {m_opt.f1:>12.3f} {'n/a':>13}"
    )
    print(
        f"  {'Warm (10 rounds)':<25} {ratio_warm:>8.2f}x {m_fixed_w.f1:>10.3f} {m_opt_w.f1:>12.3f} {m_adaptive.f1:>13.3f}"
    )
    print(
        f"  {'Heavy (30 rounds)':<25} {ratio_heavy:>8.2f}x {m_fixed_h.f1:>10.3f} {m_opt_h.f1:>12.3f} {m_adaptive_h.f1:>13.3f}"
    )

    # ── Verdict ──
    print(f"\n  Optimal thresholds: cold={opt_t:.4f}, warm={opt_t_w:.4f}, heavy={opt_t_h:.4f}")
    print(f"  Adaptive thresholds: warm={adaptive_t:.4f}, heavy={adaptive_t_h:.4f}")

    if ratio_heavy < 1.5:
        print(
            f"\n  ⚠ VERDICT: Even with heavy usage, discrimination ratio ({ratio_heavy:.2f}x) stays below 1.5x."
        )
        print("    The spread signal has limited discriminative power on this corpus.")
        print("    Consider: (a) showing it only as metadata, not as a warning,")
        print("    or (b) combining spread with other signals (e.g., top-1 absolute score).")
    else:
        print(
            f"\n  ✓ VERDICT: Discrimination ratio improved to {ratio_heavy:.2f}x with usage history."
        )

    # Save
    output_path = Path("experiments/results/relevance_signal_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "phases": {
            "cold": {
                "ratio": round(ratio_cold, 4),
                "fixed_f1": round(m_fixed.f1, 4),
                "optimal_f1": round(m_opt.f1, 4),
                "optimal_threshold": round(opt_t, 4),
            },
            "warm": {
                "ratio": round(ratio_warm, 4),
                "fixed_f1": round(m_fixed_w.f1, 4),
                "optimal_f1": round(m_opt_w.f1, 4),
                "adaptive_f1": round(m_adaptive.f1, 4),
                "adaptive_threshold": round(adaptive_t, 4),
            },
            "heavy": {
                "ratio": round(ratio_heavy, 4),
                "fixed_f1": round(m_fixed_h.f1, 4),
                "optimal_f1": round(m_opt_h.f1, 4),
                "adaptive_f1": round(m_adaptive_h.f1, 4),
                "adaptive_threshold": round(adaptive_t_h, 4),
            },
        },
        "relevant_spreads_heavy": [round(s, 4) for s in rel_spreads_heavy],
        "irrelevant_spreads_heavy": [round(s, 4) for s in irr_spreads_heavy],
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
