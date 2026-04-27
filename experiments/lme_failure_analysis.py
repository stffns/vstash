"""Failure analysis for LongMemEval retrieval results.

Loads a previously-saved results JSON from `longmemeval_retrieval`, finds
the questions where the gold session did not land in the top-K, then
re-runs the same ingest + `VstashStore.miss_analysis()` on each gold
session to capture:

    - the pipeline stage that dropped the gold chunk
    - the gold chunk's vector cosine distance to the query
    - the final rank of the gold chunk (if any)
    - the actual top-5 sessions for comparison

Aggregates by `question_type` so we can see whether failures cluster on
the vector pool, the distance cutoff, FTS, RRF, or MMR. This is the
input to the "should we retrain?" decision.

Usage:
    python -m experiments.lme_failure_analysis \
        --results experiments/results/lme_stratified_5pt_v3.json \
        --output  experiments/results/lme_failures_5pt_v3.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from experiments.longmemeval_retrieval import (
    DATA_PATH,
    _ST_FALLBACK_MODELS,
    _ingest_question,
    _st_resolver,
)
from vstash.config import VstashConfig
from vstash.embed import (
    embed_query,
    get_embedding_dim,
    register_encoder_resolver,
    unregister_encoder_resolver,
)
from vstash.store import VstashStore

DEFAULT_FAIL_THRESHOLD_K = 10  # a question is "failed" if recall@K < 1.0


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _build_question_index(data_path: Path) -> dict[str, dict]:
    raw = json.loads(Path(data_path).read_text())
    return {q["question_id"]: q for q in raw}


def _bucket_distance(d: float | None) -> str:
    if d is None:
        return "no_dist"
    if d < 0.20:
        return "<0.20"
    if d < 0.30:
        return "0.20-0.30"
    if d < 0.45:
        return "0.30-0.45"
    if d < 0.60:
        return "0.45-0.60"
    return ">=0.60"


def _gold_chunk_distance(store: VstashStore, q_emb: list[float], session_id: str) -> float | None:
    """Find the smallest cosine distance from the query to any chunk in the gold session.

    Bypasses the search pipeline entirely so we can see the raw geometric
    relationship even when the cutoff or RRF rejected the chunk.
    """
    rows = store._conn.execute(
        """
        SELECT vc.distance AS dist
          FROM vec_chunks vc
          JOIN chunks c ON c.id = vc.rowid
          JOIN documents d ON d.id = c.doc_id
         WHERE d.path = ?
           AND vc.embedding MATCH ?
           AND k = 50
         ORDER BY vc.distance ASC
         LIMIT 1
        """,
        [session_id, _serialize_vec(q_emb)],
    ).fetchall()
    if not rows:
        return None
    return float(rows[0]["dist"])


def _serialize_vec(v: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(v)}f", *v)


def _gold_best_chunk(
    store: VstashStore, q_emb: list[float], session_id: str
) -> tuple[int | None, float | None]:
    """Find the (chunk_id, cosine_distance) of the gold session's best-matching chunk.

    Bypasses sqlite-vec MATCH (which silently fails when combined with a
    rowid-IN filter inside vstash's miss_analysis path) and computes
    distance directly from raw embeddings. Each haystack has at most a
    few hundred chunks so the O(N) cost is negligible.
    """
    import math
    import struct

    rows = store._conn.execute(
        """
        SELECT vc.rowid AS chunk_id, vc.embedding AS emb
          FROM vec_chunks vc
          JOIN chunks c ON c.id = vc.rowid
          JOIN documents d ON d.id = c.doc_id
         WHERE d.path = ?
        """,
        [session_id],
    ).fetchall()
    if not rows:
        return None, None

    dim = len(q_emb)
    q_norm = math.sqrt(sum(x * x for x in q_emb)) or 1.0
    best_id: int | None = None
    best: float | None = None
    for r in rows:
        emb = struct.unpack(f"{dim}f", r["emb"])
        dot = sum(a * b for a, b in zip(q_emb, emb))
        e_norm = math.sqrt(sum(x * x for x in emb)) or 1.0
        cos_dist = 1.0 - dot / (q_norm * e_norm)
        if best is None or cos_dist < best:
            best = cos_dist
            best_id = int(r["chunk_id"])
    return best_id, best


def analyze(
    results_path: Path,
    output_path: Path,
    fail_at_k: int,
    model: str | None,
    retrieval_mode: str,
    max_failures: int | None,
) -> dict:
    cached = _load_results(results_path)
    summary = cached["summary"]
    per_question = cached["per_question"]

    # miss_analysis() in vstash always runs the hybrid pipeline (no
    # retrieval_mode parameter). If the source results were generated
    # under vec_only or fts_only, the dropped_at attribution would not
    # match what produced the failures -- refuse rather than silently
    # mislead.
    src_mode = summary.get("retrieval_mode", "hybrid")
    if src_mode != "hybrid":
        raise SystemExit(
            f"Source results used retrieval_mode={src_mode!r}; failure "
            "analysis only supports hybrid because miss_analysis() does "
            "not honour retrieval_mode."
        )

    if model is None:
        model = summary.get("embedding_model")
    if model is None:
        raise SystemExit("model not provided and not in summary; pass --model")

    failed = [r for r in per_question if r[f"recall@{fail_at_k}"] < 1.0]
    if max_failures is not None:
        failed = failed[:max_failures]

    print(f"Total questions in results:    {len(per_question)}")
    print(f"Failures at recall@{fail_at_k} < 1:    {len(failed)}")
    if not failed:
        print("Nothing to analyze.")
        return {"summary": summary, "failures": []}

    qindex = _build_question_index(DATA_PATH)
    cfg = VstashConfig()
    cfg = cfg.model_copy(update={"embeddings": cfg.embeddings.model_copy(update={"model": model})})
    resolver_registered = False
    if cfg.embeddings.model in _ST_FALLBACK_MODELS:
        register_encoder_resolver(_st_resolver)
        resolver_registered = True
    dim = get_embedding_dim(cfg.embeddings.model)

    failures: list[dict] = []
    dropped_by_type: dict[str, Counter] = defaultdict(Counter)
    dist_by_dropstage: dict[str, list[float]] = defaultdict(list)

    overall_start = time.time()
    for i, row in enumerate(failed):
        q = qindex[row["question_id"]]
        question = q["question"]
        gold = q["answer_session_ids"]
        session_ids = q["haystack_session_ids"]
        sessions = q["haystack_sessions"]
        qtype = q["question_type"]

        with tempfile.TemporaryDirectory(prefix="lme_fa_") as tmp:
            db_path = str(Path(tmp) / "lme.db")
            store = VstashStore(
                db_path,
                embedding_dim=dim,
                vector_backend=cfg.storage.vector_backend,
                snapvec_bits=cfg.storage.snapvec_bits,
                observability=cfg.observability,
                limits=cfg.limits,
                cache=cfg.cache,
            )
            try:
                _ingest_question(store, cfg, session_ids, sessions)
                q_emb = embed_query(question, cfg.embeddings.model)
                gold_traces: list[dict] = []
                for sid in gold:
                    best_chunk_id, best_dist = _gold_best_chunk(store, q_emb, sid)
                    if best_chunk_id is None:
                        gold_traces.append({"session_id": sid, "error": "no chunks"})
                        continue
                    try:
                        ma = store.miss_analysis(
                            query_embedding=q_emb,
                            query_text=question,
                            expected_chunk_id=best_chunk_id,
                            top_k=fail_at_k,
                            collection="lme",
                        )
                    except ValueError as e:
                        gold_traces.append({"session_id": sid, "error": str(e)})
                        continue
                    gold_traces.append(
                        {
                            "session_id": sid,
                            "appeared_in_results": ma.appeared_in_results,
                            "final_rank": ma.final_rank,
                            "dropped_at": ma.dropped_at,
                            "best_chunk_id": best_chunk_id,
                            "best_chunk_cos_dist": round(best_dist, 4)
                            if best_dist is not None
                            else None,
                            "suggestions": list(ma.suggestions),
                        }
                    )
                    stage = ma.dropped_at or ("retrieved" if ma.appeared_in_results else "unknown")
                    dropped_by_type[qtype][stage] += 1
                    if best_dist is not None:
                        dist_by_dropstage[stage].append(best_dist)

                actual_top_k_sessions: list[str] = []
                seen: set[str] = set()
                results = store.search(
                    query_embedding=q_emb,
                    query_text=question,
                    top_k=fail_at_k * 4,
                    collection="lme",
                    retrieval_mode=retrieval_mode,
                )
                for r in results:
                    if r.path in seen:
                        continue
                    seen.add(r.path)
                    actual_top_k_sessions.append(r.path)
                    if len(actual_top_k_sessions) >= fail_at_k:
                        break
            finally:
                store.close()

        failures.append(
            {
                "question_id": q["question_id"],
                "question_type": qtype,
                "question": question,
                "gold_session_ids": gold,
                "actual_top_k_sessions": actual_top_k_sessions,
                "first_gold_rank": row.get("first_gold_rank"),
                f"recall@{fail_at_k}": row[f"recall@{fail_at_k}"],
                "gold_traces": gold_traces,
            }
        )
        elapsed = time.time() - overall_start
        print(
            f"[{i + 1}/{len(failed)}] {qtype:25s} q={q['question_id']} "
            f"R@{fail_at_k}={row[f'recall@{fail_at_k}']:.2f} "
            f"first_gold={row.get('first_gold_rank')} "
            f"elapsed={elapsed:.0f}s"
        )

    print()
    print("=" * 72)
    print("Drop-stage histogram by question_type:")
    for qtype in sorted(dropped_by_type):
        c = dropped_by_type[qtype]
        total = sum(c.values())
        items = ", ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  {qtype:30s} (n={total}) {items}")

    print()
    print("Best gold-chunk cosine distance by drop stage:")
    print(f"  {'stage':22s} {'n':>4} {'mean':>7} {'median':>7} {'max':>7}")
    for stage in sorted(dist_by_dropstage):
        vals = dist_by_dropstage[stage]
        print(
            f"  {stage:22s} {len(vals):>4} "
            f"{statistics.mean(vals):>7.4f} "
            f"{statistics.median(vals):>7.4f} "
            f"{max(vals):>7.4f}"
        )

    out = {
        "source_results": str(results_path),
        "model": model,
        "retrieval_mode": retrieval_mode,
        "fail_threshold_k": fail_at_k,
        "n_total": len(per_question),
        "n_failed": len(failed),
        "drop_stage_by_type": {k: dict(v) for k, v in dropped_by_type.items()},
        "dist_stats_by_drop_stage": {
            stage: {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "max": round(max(vals), 4),
            }
            for stage, vals in dist_by_dropstage.items()
        },
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {output_path}")
    if resolver_registered:
        unregister_encoder_resolver(_st_resolver)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fail-at-k", type=int, default=DEFAULT_FAIL_THRESHOLD_K)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model. Defaults to the one in the results file's summary.",
    )
    p.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vec_only", "fts_only"],
        default="hybrid",
    )
    p.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Limit number of failures analyzed (for fast iteration)",
    )
    args = p.parse_args()
    analyze(
        results_path=args.results,
        output_path=args.output,
        fail_at_k=args.fail_at_k,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        max_failures=args.max_failures,
    )


if __name__ == "__main__":
    main()
