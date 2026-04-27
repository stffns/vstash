"""Combine local vstash results + Colab ColBERTv2 results into the H2H table.

Inputs (all under ``experiments/results/`` by default):

  - LongMemEval, vstash side (one or many):
      * lme_full_500_v3.json
      * lme_full_500_lme-v1.json
      * lme_full_500_base.json
      * lme_full_500_lme-v1-from-v3.json
  - LongMemEval, ColBERT side:
      * lme_full_500_colbertv2.json (downloaded from Drive)
  - BEIR, vstash side:
      * beir_benchmark.json
  - BEIR, ColBERT side:
      * beir_colbertv2.json (downloaded from Drive)

Outputs:

  - Prints a markdown-friendly table to stdout.
  - Writes the same table + raw structured JSON to
    ``experiments/results/h2h_full.json`` so the paper can pull
    numbers without re-parsing the per-engine outputs.

Usage:
    python -m experiments.h2h_combine
    python -m experiments.h2h_combine --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("experiments/results")
LME_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)
LME_KS = (1, 3, 5, 10, 20, 50)
BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa", "scidocs", "arguana")
LME_LABEL_BY_FILE = {
    "lme_full_500_base.json": "base BGE",
    "lme_full_500_v3.json": "v3 (BEIR-tuned)",
    "lme_full_500_lme-v1.json": "lme-v1 (chat from base)",
    "lme_full_500_lme-v1-from-v3.json": "lme-v1-from-v3 (chat from v3)",
    "lme_full_500_colbertv2.json": "ColBERTv2",
}


def _load_lme(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _load_beir(path: Path, key: str) -> dict[str, dict]:
    """Index a BEIR results JSON by dataset name.

    ``key`` is the metrics field on each dataset row -- ``vstash`` for
    the local benchmark output, ``colbertv2`` for the Colab output.
    Both files share ``[{dataset, ...}, ...]`` shape.
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    return {r["dataset"]: r.get(key, {}) for r in rows if "dataset" in r}


def _fmt_num(v: float | None, width: int = 7, prec: int = 4) -> str:
    if v is None:
        return f"{'-':>{width}}"
    return f"{v:>{width}.{prec}f}"


def render_lme_macro(arms: dict[str, dict]) -> list[str]:
    lines = ["LongMemEval-s (Recall@K, session-level, n=500):"]
    header = f"  {'Model':<32}" + "".join(f" | {f'R@{k}':>6}" for k in LME_KS)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for label, summary in arms.items():
        macro = summary["macro"]
        row = f"  {label:<32}"
        for k in LME_KS:
            row += f" | {macro[f'recall@{k}']:>6.4f}"
        lines.append(row)
    return lines


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interp percentile (numpy-free) on a sorted copy."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(s):
        return s[-1]
    return s[low] + frac * (s[low + 1] - s[low])


def render_lme_latency(arms_per_question: dict[str, list[dict]]) -> list[str]:
    """Per-query latency (ms) -- P50, P99, mean.

    Read from each arm's ``per_question`` rows; the harness records
    ``search_s`` per question for both vstash and the minimal ColBERT
    engine.  Reports search latency only -- ingest / corpus encoding
    is excluded since in production the index is persistent.
    """
    lines = ["", "LongMemEval-s per-query search latency (ms, n=500):"]
    header = f"  {'Model':<32} | {'P50':>7} | {'P99':>7} | {'mean':>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for label, rows in arms_per_question.items():
        if not rows:
            lines.append(f"  {label:<32} |       - |       - |       -")
            continue
        ms = [r["search_s"] * 1000 for r in rows if "search_s" in r and r["search_s"] is not None]
        if not ms:
            lines.append(f"  {label:<32} |       - |       - |       -")
            continue
        p50 = _percentile(ms, 50)
        p99 = _percentile(ms, 99)
        mean = sum(ms) / len(ms)
        lines.append(f"  {label:<32} | {p50:>7.1f} | {p99:>7.1f} | {mean:>7.1f}")
    return lines


def render_lme_by_type(arms: dict[str, dict]) -> list[str]:
    lines = ["", "LongMemEval per question_type (Recall@10):"]
    header = f"  {'Type':<28} {'n':>4}" + "".join(f" | {label[:14]:>14}" for label in arms)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for t in LME_TYPES:
        # Pull n from the first arm that has the type
        nq = None
        cells = []
        for label, summary in arms.items():
            agg = summary.get("by_question_type", {}).get(t)
            if agg is None:
                cells.append(_fmt_num(None, width=14, prec=4))
                continue
            if nq is None:
                nq = agg["n"]
            cells.append(_fmt_num(agg["recall@10"], width=14, prec=4))
        lines.append(f"  {t:<28} {nq if nq is not None else '-':>4} | " + " | ".join(cells))
    return lines


def render_beir(beir_arms: dict[str, dict[str, dict]]) -> list[str]:
    lines = ["", "BEIR (NDCG@10):"]
    header = f"  {'Dataset':<10}" + "".join(f" | {label[:14]:>14}" for label in beir_arms)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for ds in BEIR_DATASETS:
        cells = []
        for label, by_ds in beir_arms.items():
            row = by_ds.get(ds)
            cells.append(_fmt_num(None if row is None else row.get("ndcg_10"), width=14))
        lines.append(f"  {ds:<10} | " + " | ".join(cells))

    lines.append("")
    lines.append("BEIR (Recall@10):")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for ds in BEIR_DATASETS:
        cells = []
        for label, by_ds in beir_arms.items():
            row = by_ds.get(ds)
            cells.append(_fmt_num(None if row is None else row.get("recall_10"), width=14))
        lines.append(f"  {ds:<10} | " + " | ".join(cells))
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "h2h_full.json")
    args = p.parse_args()

    rd = args.results_dir

    # LongMemEval arms in canonical display order.
    lme_arms: dict[str, dict] = {}
    lme_arms_per_question: dict[str, list[dict]] = {}
    for fname, label in LME_LABEL_BY_FILE.items():
        data = _load_lme(rd / fname)
        if data is not None:
            lme_arms[label] = data["summary"]
            lme_arms_per_question[label] = data.get("per_question", [])

    # BEIR arms.
    beir_v_path = rd / "beir_benchmark.json"
    beir_c_path = rd / "beir_colbertv2.json"
    beir_arms: dict[str, dict[str, dict]] = {}
    if beir_v_path.is_file():
        beir_arms["vstash"] = _load_beir(beir_v_path, "vstash")
    if beir_c_path.is_file():
        beir_arms["ColBERTv2"] = _load_beir(beir_c_path, "colbertv2")

    if not lme_arms and not beir_arms:
        raise SystemExit(
            f"No result files found under {rd}. Expected at least one of "
            f"{list(LME_LABEL_BY_FILE)} or beir_benchmark.json / beir_colbertv2.json."
        )

    lines: list[str] = []
    if lme_arms:
        lines += render_lme_macro(lme_arms)
        lines += render_lme_by_type(lme_arms)
        lines += render_lme_latency(lme_arms_per_question)
    if beir_arms:
        lines += render_beir(beir_arms)

    table = "\n".join(lines)
    print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "lme_arms": lme_arms,
                "beir_arms": beir_arms,
                "table": table,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
