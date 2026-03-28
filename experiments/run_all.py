"""Run All Experiments — Unified Runner

Executes all experiments sequentially and produces a combined results file.
Reuses a single DB across experiments that share the same corpus.

Usage:
    python -m experiments.run_all [--skip-ingest]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "chunking_eval",
        "module": "experiments.chunking_eval",
        "description": "Code-aware vs naive chunking retrieval quality",
        "output": "experiments/results/chunking.json",
        "needs_corpus": False,  # uses synthetic code
    },
    {
        "name": "ablation_rrf",
        "module": "experiments.ablation_rrf",
        "description": "RRF vs Vector-only vs FTS-only ablation",
        "output": "experiments/results/ablation.json",
        "needs_corpus": True,
    },
    {
        "name": "scoring_grid",
        "module": "experiments.scoring_grid",
        "description": "Frequency+decay scoring grid search with NDCG",
        "output": None,  # prints to stdout
        "needs_corpus": True,
    },
    {
        "name": "relevance_signal",
        "module": "experiments.relevance_signal",
        "description": "Score spread threshold calibration",
        "output": "experiments/results/relevance.json",
        "needs_corpus": True,
    },
    {
        "name": "cold_start",
        "module": "experiments.cold_start",
        "description": "Cold start curve: queries needed before scoring helps",
        "output": "experiments/results/cold_start.json",
        "needs_corpus": True,
    },
]


def run_experiment(exp: dict, db_path: str | None = None) -> tuple[bool, float]:
    """Run a single experiment. Returns (success, elapsed_seconds)."""
    cmd = [sys.executable, "-m", exp["module"]]

    if db_path and exp.get("needs_corpus"):
        cmd.extend(["--db", db_path])

    print(f"\n{'=' * 80}")
    print(f"  Running: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    return result.returncode == 0, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--db", type=str, help="Shared DB path (skip re-ingestion)")
    parser.add_argument(
        "--only", type=str, nargs="*",
        help="Run only specified experiments",
        choices=[e["name"] for e in EXPERIMENTS],
    )
    args = parser.parse_args()

    # Ensure results directory exists
    Path("experiments/results").mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] in args.only]

    results_summary = []
    total_start = time.time()

    for exp in experiments:
        success, elapsed = run_experiment(exp, db_path=args.db)
        status = "PASS" if success else "FAIL"
        results_summary.append({
            "name": exp["name"],
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "output": exp.get("output"),
        })
        print(f"\n  [{status}] {exp['name']} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 80}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 80}\n")

    print(f"  {'Experiment':<25} {'Status':<8} {'Time':>8}")
    print(f"  {'─' * 45}")
    for r in results_summary:
        print(f"  {r['name']:<25} {r['status']:<8} {r['elapsed_s']:>7.1f}s")

    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    print(f"\n  Total: {passed}/{len(results_summary)} passed in {total_elapsed:.1f}s")

    # List output files
    results_dir = Path("experiments/results")
    if results_dir.exists():
        files = sorted(results_dir.glob("*.json"))
        if files:
            print(f"\n  Output files:")
            for f in files:
                size = f.stat().st_size
                print(f"    {f} ({size:,} bytes)")

    # Save summary
    summary_path = results_dir / "summary.json"
    summary_data = {
        "total_elapsed_s": round(total_elapsed, 1),
        "experiments": results_summary,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
