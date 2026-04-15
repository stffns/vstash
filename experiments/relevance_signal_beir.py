"""Relevance Signal on BEIR — Cross-Domain Distance Threshold Validation

Tests whether best_distance discriminates relevant from irrelevant queries
on real IR benchmarks, not just a hand-curated corpus.

For each BEIR dataset:
  - Ingest the corpus
  - "Relevant" queries: native queries (have qrels in this corpus)
  - "Irrelevant" queries: queries from OTHER BEIR datasets (cross-domain)
  - Measure best_distance for each, sweep thresholds, report F1 + bootstrap CI

This is the hard test: a SciFact query against an NFCorpus store produces
a near-miss distance (both are biomedical), not the 0.48 vs 1.09 gap from
the hand-curated experiment.

Usage:
    python -m experiments.relevance_signal_beir
    python -m experiments.relevance_signal_beir --datasets scifact nfcorpus
    python -m experiments.relevance_signal_beir --n-cross 50
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.beir_benchmark import DATASETS, download_beir, load_beir
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore, relevance_tier

MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10


@dataclass
class DistanceObservation:
    query: str
    is_relevant: bool
    best_distance: float
    relevance_tier: str
    source_dataset: str  # which BEIR dataset the query came from
    target_dataset: str  # which corpus it was run against


@dataclass
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def sweep_thresholds(
    observations: list[DistanceObservation],
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    """Sweep distance thresholds, return (best, all)."""
    thresholds = [0.60 + i * 0.01 for i in range(51)]  # 0.60 to 1.10
    best: ThresholdResult | None = None
    all_results = []

    for t in thresholds:
        tp = fp = tn = fn = 0
        for o in observations:
            predicted = o.best_distance <= t
            if o.is_relevant and predicted:
                tp += 1
            elif not o.is_relevant and predicted:
                fp += 1
            elif not o.is_relevant and not predicted:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(observations) if observations else 0.0
        r = ThresholdResult(
            threshold=t,
            f1=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )
        all_results.append(r)
        if best is None or f1 > best.f1:
            best = r

    assert best is not None
    return best, all_results


def bootstrap_ci(
    observations: list[DistanceObservation],
    threshold: float,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for F1, precision, recall, accuracy."""
    rng = random.Random(seed)
    n = len(observations)
    samples: dict[str, list[float]] = {"f1": [], "precision": [], "recall": [], "accuracy": []}

    for _ in range(n_bootstrap):
        sample = rng.choices(observations, k=n)
        best, _ = sweep_thresholds(sample)
        # Use the fixed threshold, not the per-bootstrap-best
        tp = fp = tn = fn = 0
        for o in sample:
            predicted = o.best_distance <= threshold
            if o.is_relevant and predicted:
                tp += 1
            elif not o.is_relevant and predicted:
                fp += 1
            elif not o.is_relevant and not predicted:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        samples["f1"].append(f1)
        samples["precision"].append(precision)
        samples["recall"].append(recall)
        samples["accuracy"].append(accuracy)

    result = {}
    for name, vals in samples.items():
        sorted_v = sorted(vals)
        lo = sorted_v[int(0.025 * len(sorted_v))]
        hi = sorted_v[int(0.975 * len(sorted_v))]
        result[name] = (round(lo, 4), round(hi, 4))
    return result


def run_dataset(
    target_name: str,
    all_datasets: dict[str, tuple[dict, dict, dict]],
    n_cross: int,
    dim: int,
) -> dict:
    """Run relevance signal evaluation for one target corpus."""
    print(f"\n{'=' * 60}")
    print(f"  Target corpus: {target_name}")
    print(f"{'=' * 60}")

    corpus, queries, qrels = all_datasets[target_name]

    # Ingest corpus
    db_path = tempfile.mktemp(suffix=".db", prefix=f"vstash_relsig_{target_name}_")
    store = VstashStore(db_path, embedding_dim=dim)

    print(f"  Ingesting {len(corpus)} docs...")
    for doc_id, doc in corpus.items():
        text = doc.get("title", "") + "\n" + doc.get("text", "")
        emb = embed_query(text[:512], MODEL)
        store.add_document(
            path=f"/{target_name}/{doc_id}",
            title=doc.get("title", doc_id),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )

    # Relevant queries: native queries that have qrels
    native_qids = [qid for qid in queries if qid in qrels]
    # Cap to n_cross for balanced eval
    rng = random.Random(42)
    if len(native_qids) > n_cross:
        native_qids = rng.sample(native_qids, n_cross)

    print(f"  Running {len(native_qids)} native queries (relevant)...")
    observations: list[DistanceObservation] = []
    for qid in native_qids:
        emb = embed_query(queries[qid], MODEL)
        store.search(query_embedding=emb, query_text=queries[qid], top_k=TOP_K)
        bd = store.last_best_distance
        observations.append(
            DistanceObservation(
                query=queries[qid],
                is_relevant=True,
                best_distance=bd,
                relevance_tier=relevance_tier(bd),
                source_dataset=target_name,
                target_dataset=target_name,
            )
        )

    # Irrelevant queries: queries from OTHER datasets
    cross_queries = []
    for other_name, (_, other_queries, _) in all_datasets.items():
        if other_name == target_name:
            continue
        for qid, qtext in other_queries.items():
            cross_queries.append((other_name, qtext))

    if len(cross_queries) > n_cross:
        cross_queries = rng.sample(cross_queries, n_cross)

    print(f"  Running {len(cross_queries)} cross-domain queries (irrelevant)...")
    for source_name, qtext in cross_queries:
        emb = embed_query(qtext, MODEL)
        store.search(query_embedding=emb, query_text=qtext, top_k=TOP_K)
        bd = store.last_best_distance
        observations.append(
            DistanceObservation(
                query=qtext,
                is_relevant=False,
                best_distance=bd,
                relevance_tier=relevance_tier(bd),
                source_dataset=source_name,
                target_dataset=target_name,
            )
        )

    # Stats
    rel_dists = [o.best_distance for o in observations if o.is_relevant]
    irr_dists = [o.best_distance for o in observations if not o.is_relevant]

    print(
        f"\n  Relevant (n={len(rel_dists)}): "
        f"mean={statistics.mean(rel_dists):.4f}, "
        f"min={min(rel_dists):.4f}, max={max(rel_dists):.4f}"
    )
    print(
        f"  Irrelevant (n={len(irr_dists)}): "
        f"mean={statistics.mean(irr_dists):.4f}, "
        f"min={min(irr_dists):.4f}, max={max(irr_dists):.4f}"
    )

    # Overlap
    if max(rel_dists) < min(irr_dists):
        gap = min(irr_dists) - max(rel_dists)
        print(f"  Class separation: CLEAN GAP of {gap:.4f}")
    else:
        overlap_range = min(max(rel_dists), max(irr_dists)) - max(min(rel_dists), min(irr_dists))
        print(f"  Class separation: OVERLAP zone width {overlap_range:.4f}")

    # Threshold sweep
    best, all_thresholds = sweep_thresholds(observations)
    print(f"\n  Best threshold: {best.threshold:.2f}")
    print(f"  F1={best.f1:.3f} P={best.precision:.3f} R={best.recall:.3f} Acc={best.accuracy:.3f}")
    print(f"  TP={best.tp} FP={best.fp} TN={best.tn} FN={best.fn}")

    # Bootstrap CI
    ci = bootstrap_ci(observations, best.threshold, n_bootstrap=2000)
    print("  Bootstrap 95% CI:")
    for name, (lo, hi) in ci.items():
        print(f"    {name:>10}: [{lo:.4f}, {hi:.4f}]")

    store.close()
    Path(db_path).unlink(missing_ok=True)

    return {
        "target_dataset": target_name,
        "n_relevant": len(rel_dists),
        "n_irrelevant": len(irr_dists),
        "relevant_distance": {
            "mean": round(statistics.mean(rel_dists), 4),
            "min": round(min(rel_dists), 4),
            "max": round(max(rel_dists), 4),
        },
        "irrelevant_distance": {
            "mean": round(statistics.mean(irr_dists), 4),
            "min": round(min(irr_dists), 4),
            "max": round(max(irr_dists), 4),
        },
        "best_threshold": best.threshold,
        "best_metrics": asdict(best),
        "bootstrap_ci_95": ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relevance signal validation on BEIR (cross-domain)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--n-cross", type=int, default=100, help="Number of queries per class (relevant/irrelevant)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/relevance_signal_beir.json",
    )
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    # Load all requested datasets
    print("Loading BEIR datasets...")
    all_datasets: dict[str, tuple[dict, dict, dict]] = {}
    for name in args.datasets:
        cache = download_beir(name)
        all_datasets[name] = load_beir(cache)
        c, q, qr = all_datasets[name]
        print(
            f"  {name}: {len(c)} docs, {len(q)} queries, {sum(len(v) for v in qr.values())} judgments"
        )

    # Run per-dataset evaluation
    results = []
    for name in args.datasets:
        r = run_dataset(name, all_datasets, args.n_cross, dim)
        results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':>12} {'Threshold':>10} {'F1':>8} {'P':>8} {'R':>8} {'Acc':>8}")
    print(f"  {'-' * 56}")
    for r in results:
        m = r["best_metrics"]
        print(
            f"  {r['target_dataset']:>12} {r['best_threshold']:>10.2f} "
            f"{m['f1']:>8.3f} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['accuracy']:>8.3f}"
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "relevance_signal_beir",
        "model": MODEL,
        "top_k": TOP_K,
        "n_cross_per_class": args.n_cross,
        "datasets": args.datasets,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
