"""Scoring Lifecycle Experiment — does memory improve with use?

Demonstrates the full scoring lifecycle on a real BEIR dataset (SciFact):
  cold corpus → simulated usage → γ activation → ranking improvement

Design:
  1. Ingest SciFact (5K docs, 300 queries with human relevance judgments)
  2. Measure baseline NDCG@10 with pure RRF (no scoring)
  3. Simulate 30 rounds of Zipf-weighted usage on "hot" queries (30%)
  4. After each round, measure γ (scoring maturity) and NDCG@10
  5. Compare: adaptive γ vs fixed β=0.5 vs no-scoring baseline

Expected outcome:
  - γ activates after enough usage rounds (non-uniform access → ratio ≥ 8×)
  - Hot queries improve: frequently-accessed relevant docs get boosted
  - Cold queries don't degrade under adaptive γ (may degrade under fixed β)

Usage:
    python -m experiments.scoring_lifecycle
    python -m experiments.scoring_lifecycle --dataset nfcorpus
    python -m experiments.scoring_lifecycle --rounds 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
from datetime import datetime, timezone

from experiments.beir_benchmark import download_beir, load_beir, ndcg_at_k
from vstash.config import ScoringConfig
from vstash.embed import embed_query, embed_texts, get_embedding_dim
from vstash.store import VstashStore

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10
SEED = 42
HOT_FRACTION = 0.3  # 30% of queries are "hot topics"


def zipf_weights(n: int, s: float = 1.0) -> list[float]:
    """Zipf distribution weights for n items."""
    raw = [1.0 / (i + 1) ** s for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


# ------------------------------------------------------------------ #
# Experiment                                                           #
# ------------------------------------------------------------------ #


def run_experiment(
    dataset: str = "scifact",
    model_id: str = DEFAULT_MODEL,
    n_rounds: int = 30,
    queries_per_round: int = 15,
) -> dict:
    """Run the full scoring lifecycle experiment."""

    dim = get_embedding_dim(model_id)
    random.seed(SEED)

    # ── Load BEIR dataset ──
    print(f"\n{'=' * 70}")
    print(f"  Scoring Lifecycle Experiment — {dataset}")
    print(f"{'=' * 70}")

    cache = download_beir(dataset)
    corpus, queries, qrels = load_beir(cache)
    test_qids = [qid for qid in qrels if qid in queries]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(test_qids)} with qrels")

    # ── Split hot/cold queries ──
    random.shuffle(test_qids)
    n_hot = max(1, int(len(test_qids) * HOT_FRACTION))
    hot_qids = sorted(test_qids[:n_hot])
    cold_qids = sorted(test_qids[n_hot:])
    print(f"  Hot queries: {len(hot_qids)}, Cold queries: {len(cold_qids)}")

    # ── Embed and ingest corpus ──
    print("  Embedding corpus...")
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    all_embeddings: list[list[float]] = []
    for bs in range(0, len(doc_texts), 64):
        all_embeddings.extend(embed_texts(doc_texts[bs : bs + 64], model_id))
        if (bs + 64) % 2000 < 64:
            print(f"    {min(bs + 64, len(doc_texts))}/{len(doc_texts)}...")

    db_path = tempfile.mktemp(suffix=".db")
    store = VstashStore(db_path, embedding_dim=dim)
    id_map: dict[str, str] = {}
    for doc_id, text, emb in zip(doc_ids, doc_texts, all_embeddings):
        path = f"/beir/{doc_id}"
        store.add_document(
            path=path,
            title=corpus[doc_id].get("title", ""),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )
        id_map[path] = doc_id
    print(f"  Ingested {len(doc_ids)} docs")

    # ── Pre-compute query embeddings ──
    print("  Embedding queries...")
    query_embs: dict[str, list[float]] = {}
    for qid in test_qids:
        query_embs[qid] = embed_query(queries[qid], model_id)

    # ── Scoring configs ──
    # Usage simulation: tracks access to build history
    usage_scoring = ScoringConfig(
        enabled=True, alpha=0.8, beta=0.2, decay_lambda=0.05, track_access=True
    )
    # Evaluation configs: NEVER track access (read-only measurement)
    adaptive_eval = ScoringConfig(
        enabled=True, alpha=0.8, beta=0.2, decay_lambda=0.05, track_access=False
    )
    fixed_eval = ScoringConfig(
        enabled=True, alpha=0.5, beta=0.5, decay_lambda=0.05, track_access=False
    )

    # ── Helper: evaluate a set of queries (read-only, no side effects) ──
    def evaluate(
        qids: list[str],
        scoring: ScoringConfig | None,
        gamma_override: float | None = None,
    ) -> float:
        ndcgs = []
        for qid in qids:
            results = store.search(
                query_embs[qid],
                queries[qid],
                top_k=TOP_K,
                scoring=scoring,
                _gamma_override=gamma_override,
            )
            ranked = [id_map.get(r.path, "") for r in results]
            ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))
        return round(statistics.mean(ndcgs), 4) if ndcgs else 0.0

    # ── Phase 1: Baseline (no scoring) ──
    print("\n  Phase 1: Baseline (no scoring)...")
    baseline_all = evaluate(test_qids, None)
    baseline_hot = evaluate(hot_qids, None)
    baseline_cold = evaluate(cold_qids, None)
    print(f"    All:  NDCG@10 = {baseline_all}")
    print(f"    Hot:  NDCG@10 = {baseline_hot}")
    print(f"    Cold: NDCG@10 = {baseline_cold}")

    # ── Phase 2: Usage simulation ──
    print(f"\n  Phase 2: Simulating {n_rounds} rounds of usage...")
    weights = zipf_weights(len(hot_qids))
    rounds_data: list[dict] = []

    for round_num in range(1, n_rounds + 1):
        # Pick queries for this round (Zipf-weighted from hot set)
        selected = random.choices(hot_qids, weights=weights, k=queries_per_round)

        # Search with access tracking (builds usage history)
        for qid in selected:
            store.search(
                query_embs[qid],
                queries[qid],
                top_k=TOP_K,
                scoring=usage_scoring,
            )

        # Measure maturity (uses only chunks WHERE access_count > 0)
        gamma = store.scoring_maturity()

        # Access stats (match scoring_maturity: only accessed chunks)
        row = store._conn.execute(
            "SELECT AVG(access_count) AS mean, MAX(access_count) AS max_val, "
            "SUM(access_count) AS total, COUNT(*) AS accessed "
            "FROM chunks WHERE access_count > 0"
        ).fetchone()

        mean_access = float(row["mean"] or 0)
        max_access = int(row["max_val"] or 0)
        total_access = int(row["total"] or 0)
        accessed_chunks = int(row["accessed"] or 0)
        ratio = max_access / mean_access if mean_access > 0 else 0

        # Evaluate with adaptive scoring (read-only, no access tracking)
        adaptive_all = evaluate(test_qids, adaptive_eval)
        adaptive_hot = evaluate(hot_qids, adaptive_eval)
        adaptive_cold = evaluate(cold_qids, adaptive_eval)

        # Evaluate with fixed beta (no maturity gate, read-only)
        fixed_all = evaluate(test_qids, fixed_eval, gamma_override=1.0)
        fixed_hot = evaluate(hot_qids, fixed_eval, gamma_override=1.0)
        fixed_cold = evaluate(cold_qids, fixed_eval, gamma_override=1.0)

        round_data = {
            "round": round_num,
            "gamma": round(gamma, 4),
            "total_accesses": total_access,
            "max_access": max_access,
            "mean_access": round(mean_access, 2),
            "ratio": round(ratio, 2),
            "accessed_chunks": accessed_chunks,
            "adaptive": {"all": adaptive_all, "hot": adaptive_hot, "cold": adaptive_cold},
            "fixed": {"all": fixed_all, "hot": fixed_hot, "cold": fixed_cold},
        }
        rounds_data.append(round_data)

        if round_num % 5 == 0 or round_num == 1 or gamma > 0:
            print(
                f"    Round {round_num:2d}: γ={gamma:.3f}  ratio={ratio:.1f}  "
                f"accesses={total_access}  "
                f"adaptive={adaptive_all:.4f}  fixed={fixed_all:.4f}"
            )

    # ── Phase 3: Final evaluation ──
    final = rounds_data[-1]
    print(f"\n  Phase 3: Final results (round {n_rounds})")
    print(f"    γ = {final['gamma']}  (ratio = {final['ratio']})")
    print()
    print(f"    {'':20s} {'Baseline':>10s} {'Adaptive':>10s} {'Fixed β=0.5':>12s}")
    print(f"    {'─' * 55}")
    print(
        f"    {'All queries':20s} {baseline_all:10.4f} {final['adaptive']['all']:10.4f} "
        f"{final['fixed']['all']:12.4f}"
    )
    print(
        f"    {'Hot queries':20s} {baseline_hot:10.4f} {final['adaptive']['hot']:10.4f} "
        f"{final['fixed']['hot']:12.4f}"
    )
    print(
        f"    {'Cold queries':20s} {baseline_cold:10.4f} {final['adaptive']['cold']:10.4f} "
        f"{final['fixed']['cold']:12.4f}"
    )

    # Degradation analysis
    fixed_cold_degrades = sum(1 for r in rounds_data if r["fixed"]["cold"] < baseline_cold - 1e-6)
    adaptive_cold_degrades = sum(
        1 for r in rounds_data if r["adaptive"]["cold"] < baseline_cold - 1e-6
    )
    gamma_activated = any(r["gamma"] > 0 for r in rounds_data)
    gamma_first_round = next((r["round"] for r in rounds_data if r["gamma"] > 0), None)

    print()
    print(
        f"    γ activated: {'yes (round ' + str(gamma_first_round) + ')' if gamma_activated else 'no'}"
    )
    print(
        f"    Cold query degradation — fixed: {fixed_cold_degrades}/{n_rounds} rounds, "
        f"adaptive: {adaptive_cold_degrades}/{n_rounds} rounds"
    )

    # Hot query improvement
    hot_delta_adaptive = final["adaptive"]["hot"] - baseline_hot
    hot_delta_fixed = final["fixed"]["hot"] - baseline_hot
    print(
        f"    Hot query Δ vs baseline — adaptive: {hot_delta_adaptive:+.4f} "
        f"({hot_delta_adaptive / baseline_hot * 100:+.1f}%), "
        f"fixed: {hot_delta_fixed:+.4f} ({hot_delta_fixed / baseline_hot * 100:+.1f}%)"
    )

    # ── Save results ──
    result = {
        "experiment": "scoring_lifecycle",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "model": model_id,
        "seed": SEED,
        "config": {
            "hot_fraction": HOT_FRACTION,
            "n_rounds": n_rounds,
            "queries_per_round": queries_per_round,
            "adaptive": {"alpha": 0.8, "beta": 0.2, "decay_lambda": 0.05},
            "fixed": {"alpha": 0.5, "beta": 0.5, "decay_lambda": 0.05},
        },
        "query_split": {
            "total": len(test_qids),
            "hot": len(hot_qids),
            "cold": len(cold_qids),
        },
        "baseline": {
            "all": baseline_all,
            "hot": baseline_hot,
            "cold": baseline_cold,
        },
        "rounds": rounds_data,
        "summary": {
            "gamma_activated": gamma_activated,
            "gamma_first_round": gamma_first_round,
            "gamma_final": final["gamma"],
            "cold_degradation_fixed": fixed_cold_degrades,
            "cold_degradation_adaptive": adaptive_cold_degrades,
            "final_adaptive": final["adaptive"],
            "final_fixed": final["fixed"],
            "hot_delta_adaptive_pct": round(hot_delta_adaptive / baseline_hot * 100, 2),
            "hot_delta_fixed_pct": round(hot_delta_fixed / baseline_hot * 100, 2),
        },
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = f"experiments/results/scoring_lifecycle_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    store.close()
    os.unlink(db_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring lifecycle experiment")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--rounds", type=int, default=30, help="Number of usage rounds")
    parser.add_argument("--queries-per-round", type=int, default=15, help="Queries per round")
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        model_id=args.model,
        n_rounds=args.rounds,
        queries_per_round=args.queries_per_round,
    )


if __name__ == "__main__":
    main()
