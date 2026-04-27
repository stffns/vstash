"""Bootstrap 95% CI for the holdout R@K delta (lme-v1 vs vanilla BGE).

Holdout question_ids are loaded from ``experiments/results/lme_eval.jsonl``
(the 102-query stratified disjoint split, seed 42).  Per-question recall
values come from the full N=500 result JSONs under
``experiments/results/lme_full_500_*.json``; we restrict to the holdout
102 before resampling.  Bootstrap: 1000 resamples with replacement,
seed 42.  Reports point estimate and 95% percentile CI for R@1, R@3,
R@5, R@10.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("experiments/results")
HOLDOUT_JSONL = RESULTS / "lme_eval.jsonl"
BASE_JSON = RESULTS / "lme_full_500_base.json"
TUNED_JSON = RESULTS / "lme_full_500_lme-v1.json"

KS = (1, 3, 5, 10)
B = 1000
SEED = 42


def load_per_question(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {row["question_id"]: row for row in payload["per_question"]}


def load_holdout_qids(path: Path) -> list[str]:
    qids = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        qids.append(json.loads(line)["query_id"])
    return qids


def main() -> None:
    holdout_qids = load_holdout_qids(HOLDOUT_JSONL)
    base_pq = load_per_question(BASE_JSON)
    tuned_pq = load_per_question(TUNED_JSON)

    missing_base = [q for q in holdout_qids if q not in base_pq]
    missing_tuned = [q for q in holdout_qids if q not in tuned_pq]
    if missing_base or missing_tuned:
        raise SystemExit(
            f"Missing per-question rows: base={len(missing_base)} tuned={len(missing_tuned)}"
        )

    n = len(holdout_qids)
    assert n == 102, f"Expected 102 holdout queries, got {n}"

    rng = np.random.default_rng(SEED)
    print(f"Bootstrap on n={n} holdout queries, B={B} resamples, seed={SEED}")
    print(f"{'K':>3}  {'base':>8}  {'tuned':>8}  {'delta':>9}  {'95% CI':>22}")

    for k in KS:
        base_arr = np.array([base_pq[q][f"recall@{k}"] for q in holdout_qids])
        tuned_arr = np.array([tuned_pq[q][f"recall@{k}"] for q in holdout_qids])

        base_mean = base_arr.mean()
        tuned_mean = tuned_arr.mean()
        point = tuned_mean - base_mean

        deltas = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, size=n)
            deltas[i] = tuned_arr[idx].mean() - base_arr[idx].mean()
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

        print(
            f"{k:>3}  {base_mean:>8.4f}  {tuned_mean:>8.4f}  "
            f"{point:>+9.4f}  [{ci_low:>+7.4f}, {ci_high:>+7.4f}]"
        )


if __name__ == "__main__":
    main()
