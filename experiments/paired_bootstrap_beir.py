"""Paired bootstrap of per-query NDCG@10 for two models on BEIR.

Loads the per-query sidecar JSONs produced by experiments/beir_benchmark.py
for two models (typically base BGE-small and a fine-tuned variant) and
computes paired bootstrap confidence intervals on:

- per-dataset delta NDCG@10
- macro NDCG@10 averaged across the 5 BEIR datasets

The decision rule pre-committed for the chat-to-BEIR asymmetry validation
is in experiments/results/asymmetry_decision_2026_04_28.md and is also
applied automatically here at the bottom of the report.

Usage::

    python -m experiments.paired_bootstrap_beir \
        --base experiments/results/beir_perquery_BAAI_bge-small-en-v1.5.json \
        --tuned experiments/results/beir_perquery_Stffens_bge-small-rrf-lme-v1.json \
        --out experiments/results/paired_bootstrap_beir.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

B_DEFAULT = 1000
SEED_DEFAULT = 42
ROBUST_DELTA_THRESHOLD = 0.04


def load_perquery(path: str) -> dict[str, dict[str, float]]:
    """Return ``{dataset: {qid: ndcg_10}}`` from a sidecar JSON."""

    raw = json.loads(Path(path).read_text())
    out: dict[str, dict[str, float]] = {}
    for ds in raw["results"]:
        out[ds["dataset"]] = {q["qid"]: float(q["ndcg_10"]) for q in ds["per_query"]}
    return out


def assert_pairable(base: dict, tuned: dict) -> list[str]:
    """Verify the same datasets and the same qids per dataset are present."""

    if set(base) != set(tuned):
        raise ValueError(f"Dataset mismatch: base={sorted(base)} vs tuned={sorted(tuned)}")
    for ds in base:
        if set(base[ds]) != set(tuned[ds]):
            missing_in_tuned = set(base[ds]) - set(tuned[ds])
            missing_in_base = set(tuned[ds]) - set(base[ds])
            raise ValueError(
                f"qid mismatch on {ds}: "
                f"in base only={list(missing_in_tuned)[:5]}..., "
                f"in tuned only={list(missing_in_base)[:5]}..."
            )
    return sorted(base)


def bootstrap_paired(
    base: dict[str, dict[str, float]],
    tuned: dict[str, dict[str, float]],
    *,
    B: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Resample qids with replacement per dataset, compute deltas + macro.

    Returns a structured dict with point estimates, per-dataset 95% CIs,
    and a macro 95% CI computed by averaging per-dataset bootstrap means
    in each iteration (propagates per-query variance through the macro,
    rather than treating the 5 datasets as 5 i.i.d. data points).
    """

    rng = random.Random(seed)
    datasets = sorted(base)

    qids_per_ds = {ds: sorted(base[ds]) for ds in datasets}

    point_base = {ds: statistics.mean(base[ds].values()) for ds in datasets}
    point_tuned = {ds: statistics.mean(tuned[ds].values()) for ds in datasets}
    point_delta = {ds: point_tuned[ds] - point_base[ds] for ds in datasets}
    point_macro_base = statistics.mean(point_base[ds] for ds in datasets)
    point_macro_tuned = statistics.mean(point_tuned[ds] for ds in datasets)
    point_macro_delta = point_macro_tuned - point_macro_base

    per_ds_deltas: dict[str, list[float]] = {ds: [] for ds in datasets}
    macro_deltas: list[float] = []

    for _ in range(B):
        per_ds_means: list[float] = []
        for ds in datasets:
            qids = qids_per_ds[ds]
            n = len(qids)
            sampled = [qids[rng.randrange(n)] for _ in range(n)]
            base_vals = [base[ds][q] for q in sampled]
            tuned_vals = [tuned[ds][q] for q in sampled]
            base_mean = statistics.mean(base_vals)
            tuned_mean = statistics.mean(tuned_vals)
            delta = tuned_mean - base_mean
            per_ds_deltas[ds].append(delta)
            per_ds_means.append(delta)
        macro_deltas.append(statistics.mean(per_ds_means))

    def ci(samples: list[float], lo_pct: float = 2.5, hi_pct: float = 97.5) -> tuple[float, float]:
        s = sorted(samples)
        n = len(s)
        lo_idx = max(0, int(n * lo_pct / 100))
        hi_idx = min(n - 1, int(n * hi_pct / 100))
        return s[lo_idx], s[hi_idx]

    per_ds_summary: dict[str, dict] = {}
    for ds in datasets:
        lo, hi = ci(per_ds_deltas[ds])
        per_ds_summary[ds] = {
            "n_queries": len(qids_per_ds[ds]),
            "base_ndcg_10": round(point_base[ds], 6),
            "tuned_ndcg_10": round(point_tuned[ds], 6),
            "delta_point": round(point_delta[ds], 6),
            "ci_2_5": round(lo, 6),
            "ci_97_5": round(hi, 6),
            "ci_excludes_zero": lo > 0 or hi < 0,
            "direction": "tuned > base" if point_delta[ds] > 0 else "base > tuned",
        }

    macro_lo, macro_hi = ci(macro_deltas)
    macro_summary = {
        "base_macro_ndcg_10": round(point_macro_base, 6),
        "tuned_macro_ndcg_10": round(point_macro_tuned, 6),
        "delta_point": round(point_macro_delta, 6),
        "ci_2_5": round(macro_lo, 6),
        "ci_97_5": round(macro_hi, 6),
        "ci_excludes_zero": macro_lo > 0 or macro_hi < 0,
        "B": B,
        "seed": seed,
    }
    return {
        "macro": macro_summary,
        "per_dataset": per_ds_summary,
    }


def apply_decision_rule(
    macro: dict, *, threshold: float = ROBUST_DELTA_THRESHOLD
) -> tuple[str, str]:
    """Apply the pre-committed decision rule from asymmetry_decision_2026_04_28.md."""

    delta = macro["delta_point"]
    lo = macro["ci_2_5"]
    if lo <= 0:
        return (
            "ARTIFACT",
            "CI lower bound crosses zero. Retire the asymmetry claim. "
            "The original v2 thesis ('BEIR-tune regresses on chat') is still "
            "supported by S5.7 / S6 evidence; the reverse direction is not.",
        )
    if delta > threshold:
        return (
            "ROBUST",
            f"CI lower bound > 0 and delta_macro = {delta:.4f} > {threshold}. "
            "Update paper thesis to 'training-signal quality > target alignment', "
            "add Section 5.8 with the asymmetry table including per-dataset CIs.",
        )
    return (
        "DIRECTIONAL",
        f"CI lower bound > 0 but delta_macro = {delta:.4f} <= {threshold}. "
        "Update paper but with caveat 'within the backend calibration band of "
        "+0.02 to +0.08 NDCG@10 disclosed in S8.10; leave the asymmetry magnitude "
        "as preliminary.'",
    )


def format_report(result: dict) -> str:
    macro = result["macro"]
    out_lines = [
        "Paired bootstrap of per-query NDCG@10",
        f"B={macro['B']}  seed={macro['seed']}",
        "",
        f"{'dataset':<12} {'n':>5} {'base':>8} {'tuned':>8} {'delta':>9} {'CI 95% [lo, hi]':>22} {'sig':>4}",
    ]
    for ds, s in sorted(result["per_dataset"].items()):
        sig = "*" if s["ci_excludes_zero"] else ""
        out_lines.append(
            f"{ds:<12} {s['n_queries']:>5} {s['base_ndcg_10']:>8.4f} "
            f"{s['tuned_ndcg_10']:>8.4f} {s['delta_point']:>+9.4f} "
            f"  [{s['ci_2_5']:>+7.4f}, {s['ci_97_5']:>+7.4f}] {sig:>4}"
        )
    out_lines.append("")
    out_lines.append(
        f"{'macro':<12} {'':>5} {macro['base_macro_ndcg_10']:>8.4f} "
        f"{macro['tuned_macro_ndcg_10']:>8.4f} {macro['delta_point']:>+9.4f} "
        f"  [{macro['ci_2_5']:>+7.4f}, {macro['ci_97_5']:>+7.4f}] "
        f"{'*' if macro['ci_excludes_zero'] else '':>4}"
    )
    out_lines.append("")
    decision, rationale = apply_decision_rule(macro)
    out_lines.append(f"DECISION (pre-committed rule): {decision}")
    out_lines.append(rationale)
    return "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="Per-query JSON for the baseline model")
    parser.add_argument("--tuned", required=True, help="Per-query JSON for the tuned model")
    parser.add_argument("--out", default="experiments/results/paired_bootstrap_beir.json")
    parser.add_argument("--B", type=int, default=B_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    base = load_perquery(args.base)
    tuned = load_perquery(args.tuned)
    assert_pairable(base, tuned)

    result = bootstrap_paired(base, tuned, B=args.B, seed=args.seed)
    decision, rationale = apply_decision_rule(result["macro"])
    result["decision"] = {
        "label": decision,
        "rationale": rationale,
        "threshold_delta": ROBUST_DELTA_THRESHOLD,
        "rule_source": "experiments/results/asymmetry_decision_2026_04_28.md",
        "base_input": args.base,
        "tuned_input": args.tuned,
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(format_report(result))
    print(f"\n  Detailed JSON saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
