"""
v2_v3_head_to_head.py -- apples-to-apples comparison of
``Stffens/bge-small-rrf-v2`` vs ``Stffens/bge-small-rrf-v3`` on the
same BEIR eval pipeline.

Why this script exists: v2 and v3 were validated on different eval
pipelines over time (v2 under the pre-PR-#243 top-10 matmul, v3
under the current top-100 matmul + doc-dedup). The per-dataset
deltas published on each model's HF card therefore are not
directly comparable. Running both through one common evaluator
fixes that.

What it does:

1. Download BEIR SciFact + NFCorpus + FiQA (cached under
   ``experiments/data``) -- same utilities as the retrain
   notebook.
2. Ingest each into its own ``VstashStore``.
3. For each of {base bge-small, v2, v3}, call
   ``retrain_batch.evaluate_model_batched`` per dataset with the
   same eval queries (real BEIR qrels).
4. Emit a Markdown table of absolute NDCG@10 / NDCG@3 / Recall@100
   per (model, dataset) plus per-model macro.

Runtime: ~3 datasets x 3 models x ~1 min/model on A100 = ~10 min.
T4: ~20-25 min.

Usage:

    python -m experiments.v2_v3_head_to_head

    # or pin a subset of datasets / models
    python -m experiments.v2_v3_head_to_head \\
        --datasets scifact nfcorpus \\
        --models base v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODEL_ALIASES: dict[str, str] = {
    "base": "BAAI/bge-small-en-v1.5",
    "v1": "stffens/bge-small-rrf-v1",
    "v2": "Stffens/bge-small-rrf-v2",
    "v3": "Stffens/bge-small-rrf-v3",
}
# Full BEIR set we bench against in the paper. Previous v2 result
# beat ColBERTv2 on 4/5 (all except ArguAna); this h2h re-checks the
# same cut for v3 in the current eval pipeline.
DEFAULT_DATASETS = ("scifact", "nfcorpus", "scidocs", "fiqa", "arguana")
DEFAULT_MODELS = ("base", "v2", "v3")

# ColBERTv2 + BM25 reference numbers, pinned from the paper's
# Section 7.3 table. Used only for display so the h2h output
# slots into the existing framing without a separate re-run.
COLBERT_V2_NDCG_10: dict[str, float] = {
    "scifact": 0.693,
    "nfcorpus": 0.344,
    "scidocs": 0.154,
    "fiqa": 0.356,
    "arguana": 0.463,
}
BM25_NDCG_10: dict[str, float] = {
    "scifact": 0.665,
    "nfcorpus": 0.325,
    "scidocs": 0.158,
    "fiqa": 0.236,
    "arguana": 0.315,
}


def _ingest_dataset(name: str, data_dir: Path, store_dir: Path) -> tuple:
    """Download BEIR dataset + ingest into a fresh VstashStore.
    Returns (store, eval_queries)."""
    from sentence_transformers import SentenceTransformer

    from experiments.beir_benchmark import download_beir, load_beir
    from vstash.retrain import qrels_to_eval_queries
    from vstash.store import VstashStore

    cache = download_beir(name)
    corpus, queries, qrels = load_beir(cache)
    doc_ids = list(corpus.keys())
    print(
        f"[{name}] corpus: {len(doc_ids)} docs | queries: {len(queries)} | qrels: {len(qrels)}",
        flush=True,
    )

    # Embed corpus with the BASE model. ``evaluate_model_batched``
    # re-embeds for each candidate model internally, so the initial
    # ingest just needs to populate the store with correctly-shaped
    # vectors and the FTS5 index. Using base embeddings keeps ingest
    # fast on a cold run.
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)

    store_path = str(store_dir / f"h2h_{name}.db")
    for suffix in ("", "-wal", "-shm"):
        p = store_path + suffix
        if os.path.isfile(p):
            os.remove(p)

    store = VstashStore(store_path, embedding_dim=int(vecs.shape[1]))
    for doc_id, text, vec in zip(doc_ids, texts, vecs):
        store.add_document(
            path=f"{name}://{doc_id}",
            title=corpus[doc_id].get("title", "")[:80] or doc_id,
            chunks=[text],
            embeddings=[list(map(float, vec))],
        )
    stats = store.stats()
    print(f"  -> ingested {stats.documents} docs / {stats.chunks} chunks", flush=True)

    eval_queries = qrels_to_eval_queries(
        queries=queries,
        qrels=qrels,
        path_for_doc_id=lambda doc_id, d=name: f"{d}://{doc_id}",
    )
    print(f"  -> eval_queries: {len(eval_queries)}", flush=True)
    return store, eval_queries


def _evaluate_one(store, model_name: str, eval_queries: list[dict]) -> dict:
    from vstash.retrain_batch import evaluate_model_batched

    metrics = evaluate_model_batched(
        store,
        model_name_or_path=model_name,
        eval_queries=eval_queries,
        noise_sample_size=0,  # we want to score on the actual corpus, not a synthetic noise mix
        seed=42,
    )
    return metrics.as_dict()


def _print_table(results: dict) -> None:
    """Results shape: results[model_alias][dataset] = metrics dict."""
    models = list(results.keys())
    datasets = sorted({ds for m in results.values() for ds in m.keys()})

    # Per-dataset absolute NDCG@10 table with BM25 + ColBERTv2 as
    # external reference columns. Same shape as the paper's Section
    # 7.3 table so the h2h result reads 1:1 against the published
    # framing.
    print("\n### Absolute NDCG@10 (with external references)\n")
    header_cols = ["Dataset", "BM25", "ColBERTv2"] + models
    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")
    for ds in datasets:
        cells: list[str] = [ds]
        bm25 = BM25_NDCG_10.get(ds)
        colbert = COLBERT_V2_NDCG_10.get(ds)
        cells.append(f"{bm25:.3f}" if bm25 is not None else "-")
        cells.append(f"{colbert:.3f}" if colbert is not None else "-")
        for m in models:
            v = results[m].get(ds, {}).get("ndcg_at_10")
            cells.append(f"{v:.4f}" if v is not None else "-")
        print("| " + " | ".join(cells) + " |")
    # Macro row (only for vstash models; BM25/ColBERTv2 macros live
    # in the paper and we do not recompute them here).
    macros: dict[str, float] = {}
    for m in models:
        vals = [results[m][ds]["ndcg_at_10"] for ds in datasets if ds in results[m]]
        macros[m] = sum(vals) / len(vals) if vals else 0.0
    macro_cells = ["**macro**", "-", "-"] + [f"**{macros[m]:.4f}**" for m in models]
    print("| " + " | ".join(macro_cells) + " |")

    # "Wins vs ColBERTv2" per model -- the headline figure the paper
    # frames as "4/5 BEIR datasets beaten".
    print("\n### Per-model wins vs ColBERTv2 (absolute NDCG@10)\n")
    print("| Model | wins / total | datasets won |")
    print("|---|---|---|")
    for m in models:
        if m == "base":
            continue
        won: list[str] = []
        lost: list[str] = []
        for ds in datasets:
            colbert = COLBERT_V2_NDCG_10.get(ds)
            v = results[m].get(ds, {}).get("ndcg_at_10")
            if colbert is None or v is None:
                continue
            (won if v > colbert else lost).append(ds)
        total = len(won) + len(lost)
        print(f"| {m} | {len(won)} / {total} | {', '.join(won) or '-'} |")

    # Secondary metrics table.
    print("\n### Supporting metrics (NDCG@3 / Recall@100)\n")
    print("| Dataset | Model | NDCG@10 | NDCG@3 | Recall@100 | MRR | Hit@10 |")
    print("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for m in models:
            r = results[m].get(ds, {})
            if not r:
                continue
            print(
                f"| {ds} | {m} | "
                f"{r.get('ndcg_at_10', 0):.4f} | "
                f"{r.get('ndcg_at_3', 0):.4f} | "
                f"{r.get('recall_at_100', 0):.4f} | "
                f"{r.get('mrr', 0):.4f} | "
                f"{r.get('hit_at_10', 0):.4f} |"
            )

    # Per-model win counts vs base.
    if "base" in results:
        print("\n### v2 / v3 deltas vs base (absolute NDCG@10)\n")
        print("| Dataset | v2 delta | v3 delta | winner |")
        print("|---|---|---|---|")
        for ds in datasets:
            base = results.get("base", {}).get(ds, {}).get("ndcg_at_10")
            v2 = results.get("v2", {}).get(ds, {}).get("ndcg_at_10")
            v3 = results.get("v3", {}).get(ds, {}).get("ndcg_at_10")
            if base is None:
                continue
            d2 = f"{(v2 - base):+.4f}" if v2 is not None else "-"
            d3 = f"{(v3 - base):+.4f}" if v3 is not None else "-"
            if v2 is not None and v3 is not None:
                winner = "v3" if v3 > v2 else ("v2" if v2 > v3 else "tie")
            else:
                winner = "-"
            print(f"| {ds} | {d2} | {d3} | {winner} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="BEIR dataset names to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model aliases to evaluate. Aliases: "
        + ", ".join(f"{k}={v}" for k, v in MODEL_ALIASES.items()),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp"),
        help="Where to create the per-dataset VstashStores.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("experiments/results/v2_v3_head_to_head.json"),
        help="Where to write the combined metrics JSON.",
    )
    args = parser.parse_args()

    for m in args.models:
        if m not in MODEL_ALIASES:
            print(
                f"error: unknown model alias {m!r}. Valid: {list(MODEL_ALIASES.keys())}",
                file=sys.stderr,
            )
            return 1

    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"missing dependency: {exc}.  Install with:\n"
            "  pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'",
            file=sys.stderr,
        )
        return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs("experiments/data", exist_ok=True)

    # Ingest each dataset once; we reuse the store across models.
    stores: dict[str, tuple] = {}
    for ds in args.datasets:
        stores[ds] = _ingest_dataset(ds, Path("experiments/data"), args.data_dir)

    # For each model, evaluate on every dataset.
    results: dict[str, dict[str, dict]] = {}
    for alias in args.models:
        model_name = MODEL_ALIASES[alias]
        print(f"\n=== Evaluating {alias} ({model_name}) ===", flush=True)
        results[alias] = {}
        for ds in args.datasets:
            store, eval_queries = stores[ds]
            metrics = _evaluate_one(store, model_name, eval_queries)
            results[alias][ds] = metrics
            print(
                f"  [{alias}/{ds}] NDCG@10={metrics['ndcg_at_10']:.4f}  "
                f"NDCG@3={metrics.get('ndcg_at_3', 0):.4f}  "
                f"Recall@100={metrics.get('recall_at_100', 0):.4f}",
                flush=True,
            )

    _print_table(results)

    args.results_json.write_text(json.dumps(results, indent=2))
    print(f"\nCombined metrics JSON: {args.results_json}")

    for ds, (store, _) in stores.items():
        try:
            store.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
