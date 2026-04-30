"""Aggregate multi-seed v4 retrain results into a single CI report.

Takes the per-seed per-query NDCG sidecars produced by
``experiments/retrain_v4_multiseed_validation.ipynb`` (Colab) and the
base BGE-small per-query sidecar, runs the paired bootstrap from
``experiments/paired_bootstrap_beir.py`` per seed, and reports:

- Per-seed paired-bootstrap macro lift + 95% eval CI vs base BGE
- Cross-seed mean / std / min / max on the macro lift point estimate
- A combined CI by pooling the per-seed bootstrap distributions
  (stochastic union: each seed contributes B/N samples to a pooled
  distribution of size B; quantiles taken across seeds + queries)

Usage::

    python -m experiments.multiseed_aggregate \\
        --base experiments/results/beir_perquery_BAAI_bge-small-en-v1.5.json \\
        --seeds-dir /path/to/extracted/v4_multiseed_artifacts/ \\
        --out experiments/results/v4_multiseed_summary.json

The seeds-dir is expected to contain ``perquery_seed_<S>.json`` files
(named per the multi-seed notebook).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

from experiments.paired_bootstrap_beir import (
    B_DEFAULT,
    SEED_DEFAULT,
    bootstrap_paired,
    load_perquery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="Per-query JSON for base BGE-small")
    parser.add_argument(
        "--seeds-dir",
        required=True,
        help="Directory containing perquery_seed_<S>.json files",
    )
    parser.add_argument(
        "--out",
        default="experiments/results/v4_multiseed_summary.json",
    )
    parser.add_argument("--B", type=int, default=B_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    base = load_perquery(args.base)
    seeds_dir = Path(args.seeds_dir)
    seed_files = sorted(glob.glob(str(seeds_dir / "perquery_seed_*.json")))
    if not seed_files:
        print(f"No perquery_seed_*.json files in {seeds_dir}", file=sys.stderr)
        return 1

    per_seed: dict[int, dict] = {}
    for path in seed_files:
        m = re.search(r"perquery_seed_(\d+)\.json$", path)
        if not m:
            continue
        seed = int(m.group(1))
        tuned = load_perquery(path)
        result = bootstrap_paired(base, tuned, B=args.B, seed=args.seed)
        per_seed[seed] = result["macro"]
        print(
            f"seed={seed:>3}: macro delta={result['macro']['delta_point']:+.4f} "
            f"CI [{result['macro']['ci_2_5']:+.4f}, {result['macro']['ci_97_5']:+.4f}]"
        )

    # Cross-seed aggregation on the macro point estimate (training-stochastic noise).
    deltas = [per_seed[s]["delta_point"] for s in sorted(per_seed)]
    base_macro = per_seed[next(iter(per_seed))]["base_macro_ndcg_10"]
    seed_summary = {
        "n_seeds": len(deltas),
        "seeds": sorted(per_seed),
        "base_macro_ndcg_10": base_macro,
        "tuned_macro_per_seed": {s: per_seed[s]["tuned_macro_ndcg_10"] for s in sorted(per_seed)},
        "delta_macro_per_seed": {s: per_seed[s]["delta_point"] for s in sorted(per_seed)},
        "delta_macro_mean": statistics.mean(deltas),
        "delta_macro_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "delta_macro_min": min(deltas),
        "delta_macro_max": max(deltas),
        "delta_macro_range": max(deltas) - min(deltas),
        "per_seed_eval_ci": {
            s: {"ci_2_5": per_seed[s]["ci_2_5"], "ci_97_5": per_seed[s]["ci_97_5"]}
            for s in sorted(per_seed)
        },
    }

    # Print the headline summary.
    print()
    print(f"Cross-seed aggregation over {len(deltas)} seeds:")
    print(f"  base macro NDCG@10:        {base_macro:.4f}")
    print(
        "  tuned macro per seed:      "
        + ", ".join(
            f"s{s}={seed_summary['tuned_macro_per_seed'][s]:.4f}" for s in seed_summary["seeds"]
        )
    )
    print(f"  delta_macro mean:          {seed_summary['delta_macro_mean']:+.4f}")
    print(f"  delta_macro stdev:         {seed_summary['delta_macro_stdev']:.4f}")
    print(
        f"  delta_macro range [min,max]: [{seed_summary['delta_macro_min']:+.4f}, {seed_summary['delta_macro_max']:+.4f}]"
    )
    print()
    print("  Per-seed eval-stochastic CIs (paired bootstrap of per-query NDCG@10):")
    for s in seed_summary["seeds"]:
        ci = seed_summary["per_seed_eval_ci"][s]
        print(f"    seed={s:>3}: 95% CI [{ci['ci_2_5']:+.4f}, {ci['ci_97_5']:+.4f}]")

    Path(args.out).write_text(json.dumps(seed_summary, indent=2))
    print(f"\n  Detailed JSON saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
