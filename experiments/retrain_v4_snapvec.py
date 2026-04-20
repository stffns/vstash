"""
retrain_v4_snapvec.py -- B. Train (or diff) a v4 variant whose hard
negatives were mined via snapvec ANN instead of sqlite-vec exact.

Key insight. ``generate_triples`` (the non-batched path) calls
``store.search()``, which routes through the store's
``vector_backend``. Swap the backend to ``snapvec`` and the hard
negatives in each training triple are now ANN-approximate rather
than the exact k-NN that sqlite-vec returns. The research question:
does that shift the negatives enough to change the trained model,
and if so, is the change positive, neutral, or regressive?

Two phases, both optional:

  1. Mine-and-diff (cheap, CPU-only). Ingest BEIR SciFact into
     two fresh stores -- one sqlite-vec, one snapvec flat --
     sample the same training chunks with a fixed seed, and run
     ``generate_triples`` against each store. Compare pair counts,
     per-query negative overlap, and disagreement rate.

  2. Train-and-gate (GPU recommended). For each backend, call
     ``train_mnrl`` on its pair set, run the eval harness on the
     same BEIR qrels, and report delta NDCG@10 vs the base model.
     The gate decides which (if any) candidate to keep.

Usage:

    # Fast diff, no training (runs in ~30-60 s on Apple Silicon):
    python -m experiments.retrain_v4_snapvec --dataset scifact

    # Full train + gate (GPU-ish; slow on CPU):
    python -m experiments.retrain_v4_snapvec --dataset scifact --train

    # Wider: run on nfcorpus + fiqa too. Training uses one model per backend,
    # so datasets are evaluated independently.
    python -m experiments.retrain_v4_snapvec --dataset nfcorpus --train

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.beir_benchmark import download_beir, load_beir
from experiments.beir_snapvec_h2h import (
    DEFAULT_MODEL,
    _embed_corpus,
    _embed_queries,
    _get_st_model,
)
from vstash import embed as vstash_embed
from vstash import retrain as vstash_retrain
from vstash.embed import get_embedding_dim
from vstash.retrain import (
    evaluate_model,
    generate_triples,
    sample_training_chunks,
)
from vstash.store import VstashStore


def _evaluate_on_backend(
    base_store: VstashStore,
    model_path: str,
    eval_queries: list[dict],
    backend: str,
    seed: int,
):
    """Wrap ``evaluate_model`` so its internal eval_store uses ``backend``.

    ``evaluate_model`` hard-codes ``VstashStore(tmp_db, embedding_dim=dim)``
    on its internal eval index (``vstash/retrain.py`` line 874). The
    ``base_store`` we pass is only used for relevant-chunk + noise lookup,
    not for search. To run a meaningful "v4 on snapvec" cell, we need the
    eval index itself to use snapvec. We do that by patching the symbol
    ``vstash.retrain.VstashStore`` with a shim that defaults
    ``vector_backend=backend`` for the duration of the call, then restore.
    """
    original_cls = vstash_retrain.VstashStore

    def _forced(*args, **kwargs):
        kwargs.setdefault("vector_backend", backend)
        return original_cls(*args, **kwargs)

    vstash_retrain.VstashStore = _forced
    try:
        return evaluate_model(
            base_store,
            model_name_or_path=model_path,
            eval_queries=eval_queries,
            seed=seed,
        )
    finally:
        vstash_retrain.VstashStore = original_cls


def _train_triplet(
    pairs: list[dict],
    base_model: str,
    output_path: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    margin: float,
) -> int:
    """TripletLoss wrapper. Filters pairs to those with a non-null
    explicit hard negative (required by TripletLoss) and trains with
    ``sentence_transformers.losses.TripletLoss``.

    Returns the number of triplets actually used (post-filter). The
    call site should record this so the final report distinguishes
    "0 pairs, training skipped" from "N pairs trained"."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader
    import random as _random
    import torch as _torch

    triplets = [p for p in pairs if p.get("negative")]
    if not triplets:
        return 0

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(base_model)
    examples = [
        InputExample(texts=[p["query"], p["positive"], p["negative"]])
        for p in triplets
    ]

    rng = _random.Random(seed)
    rng.shuffle(examples)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    gen = _torch.Generator()
    gen.manual_seed(seed)
    loader_kwargs["generator"] = gen
    loader = DataLoader(examples, **loader_kwargs)

    loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=margin,
    )
    warmup_steps = min(50, max(1, len(loader) // 5))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output_path,
        show_progress_bar=True,
    )
    model.save(output_path)

    meta = {
        "base_model": base_model,
        "loss": "triplet",
        "distance_metric": "cosine",
        "triplet_margin": margin,
        "n_pairs_total": len(pairs),
        "n_triplets_used": len(triplets),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }
    (Path(output_path) / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return len(triplets)


def _patch_embed_query_for_finetuned_model() -> None:
    """Fine-tuned v3 is published as safetensors only and isn't in
    ``vstash.embed._HF_ONNX_MODELS``. On Apple Silicon the default
    resolver routes to MLX, which rejects v3's legacy BERT tensor
    names (``embeddings.LayerNorm.beta/gamma`` etc). Replace
    ``embed_query`` for the duration of this script with a
    SentenceTransformer-backed path so generate_triples works.
    This is intentionally local to the experiment; the library
    should expose v3 via its own channel separately."""

    def _embed_query_via_st(text: str, model_name: str, backend=None) -> list[float]:
        st = _get_st_model(model_name)
        vec = st.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32).tolist()

    vstash_embed.embed_query = _embed_query_via_st
    # retrain.py imports the symbol at module load time.
    from vstash import retrain as vstash_retrain

    vstash_retrain.embed_query = _embed_query_via_st


def _build_store(
    db_path: Path,
    backend: str,
    dim: int,
    ids: list[str],
    texts: list[str],
    vecs: np.ndarray,
    path_prefix: str,
) -> VstashStore:
    """Fresh store of the given backend with the BEIR corpus already ingested.

    ``path_prefix`` must match the prefix used by ``_qrels_to_eval_queries``
    so ``evaluate_model`` can resolve qrels paths to chunks. Earlier versions
    derived the prefix from ``db_path.stem``, which was unique per backend
    and therefore never matched the qrels -- NDCG collapsed to 0."""
    for suffix in ("", "-wal", "-shm", ".snpv", ".snpi"):
        p = db_path.with_suffix(db_path.suffix + suffix) if suffix else db_path
        if os.path.exists(p):
            os.remove(p)

    store = VstashStore(str(db_path), embedding_dim=dim, vector_backend=backend)
    docs = [
        {
            "path": f"{path_prefix}/{doc_id}",
            "title": doc_id,
            "chunks": [text],
            "embeddings": [vecs[i].tolist()],
            "source_type": "text",
        }
        for i, (doc_id, text) in enumerate(zip(ids, texts))
    ]
    store.add_documents_batch(docs)
    return store


def _pair_key(pair: dict) -> str:
    """Hash a triple by (query, positive) so sqlite-vec and snapvec
    pairs sharing the same chunk-prefix query can be joined."""
    h = hashlib.md5()
    h.update((pair.get("query") or "").encode())
    h.update(b"\x00")
    h.update((pair.get("positive") or "").encode())
    return h.hexdigest()


def _diff_pair_sets(pairs_vec: list[dict], pairs_snap: list[dict]) -> dict:
    """Compare two triple lists: pair counts, shared-query count,
    per-query negative overlap."""
    vec_by_key = {_pair_key(p): p for p in pairs_vec}
    snap_by_key = {_pair_key(p): p for p in pairs_snap}
    shared = set(vec_by_key) & set(snap_by_key)

    same_neg = 0
    diff_neg = 0
    only_vec_has = 0
    only_snap_has = 0
    neither_has_neg = 0
    for k in shared:
        v_neg = vec_by_key[k].get("negative")
        s_neg = snap_by_key[k].get("negative")
        if v_neg is None and s_neg is None:
            neither_has_neg += 1
        elif v_neg is not None and s_neg is None:
            only_vec_has += 1
        elif v_neg is None and s_neg is not None:
            only_snap_has += 1
        elif v_neg == s_neg:
            same_neg += 1
        else:
            diff_neg += 1

    vec_has_neg = sum(1 for p in pairs_vec if p.get("negative"))
    snap_has_neg = sum(1 for p in pairs_snap if p.get("negative"))

    return {
        "n_pairs_vec": len(pairs_vec),
        "n_pairs_snap": len(pairs_snap),
        "n_shared_queries": len(shared),
        "vec_has_negative": vec_has_neg,
        "snap_has_negative": snap_has_neg,
        "negative_overlap": {
            "same": same_neg,
            "different": diff_neg,
            "only_vec_has": only_vec_has,
            "only_snap_has": only_snap_has,
            "both_missing": neither_has_neg,
        },
        "divergence_pct": round(
            100 * (diff_neg + only_vec_has + only_snap_has) / max(len(shared), 1), 2
        ),
    }


def _qrels_to_eval_queries(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dataset_prefix: str,
) -> list[dict]:
    """Convert BEIR qrels + queries into retrain.evaluate_model's format.
    Paths are prefixed by dataset so they line up with the ingested
    ``{dataset_prefix}/{doc_id}`` scheme."""
    out: list[dict] = []
    for qid, rels in qrels.items():
        text = queries.get(qid)
        if not text:
            continue
        relevant = [f"{dataset_prefix}/{did}" for did, rel in rels.items() if rel > 0]
        if not relevant:
            continue
        out.append({"query": text, "relevant_paths": relevant})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1500,
        help="Training-chunk sample size (shared across both backends).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train one model per backend and evaluate against BEIR qrels.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="MNRL epochs (only when --train)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="MNRL batch size (only when --train)."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-6, help="Learning rate (only when --train)."
    )
    parser.add_argument(
        "--loss",
        choices=("mnrl", "triplet"),
        default="mnrl",
        help=(
            "Training loss. 'mnrl' uses MultipleNegativesRankingLoss with "
            "in-batch negatives (default; matches v3 recipe). 'triplet' "
            "uses TripletLoss which requires an explicit hard negative per "
            "example and filters pairs accordingly -- this exposes the "
            "mining-backend signal that MNRL's in-batch average hides."
        ),
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=0.3,
        help=(
            "Margin for TripletLoss with COSINE distance (only when "
            "--loss triplet). Default 0.3: cosine distance for L2-normalized "
            "vectors sits in [0, 2] with same-topic pairs at ~0-0.5 and "
            "random pairs at ~0.7-1.0; 0.3 asks for d(a,n) >= d(a,p) + 0.3, "
            "achievable without saturating the loss on easy negatives."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/data/beir_snapvec_h2h_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/retrain_v4_snapvec.json"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("experiments/results/v4_snapvec_models"),
        help="Directory for trained candidate models (only when --train).",
    )
    args = parser.parse_args()

    dim = get_embedding_dim(args.model)
    print(
        f"=== retrain v4 (snapvec vs sqlite-vec hard-neg mining) ===\n"
        f"dataset={args.dataset} model={args.model} dim={dim} "
        f"max_queries={args.max_queries} train={args.train}"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    _patch_embed_query_for_finetuned_model()

    ids, texts, vecs = _embed_corpus(args.dataset, args.model, args.cache_dir)
    cache = download_beir(args.dataset)
    _, queries, qrels = load_beir(cache)
    print(f"  corpus={len(ids)} queries={len(queries)} qrels={len(qrels)}")
    _embed_queries(args.dataset, args.model, queries, args.cache_dir)

    eval_queries = _qrels_to_eval_queries(queries, qrels, args.dataset)
    print(f"  eval_queries={len(eval_queries)}")

    report: dict = {
        "dataset": args.dataset,
        "model": args.model,
        "max_queries": args.max_queries,
        "seed": args.seed,
    }

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        store_vec = _build_store(
            tmp_dir / f"{args.dataset}_vec.db",
            "sqlite-vec",
            dim,
            ids,
            texts,
            vecs,
            path_prefix=args.dataset,
        )
        # Pre-sample training chunks from sqlite-vec so both backends
        # mine triples from the exact same chunk population (the only
        # variable is which vector backend answers the search calls).
        training_chunks = sample_training_chunks(
            store_vec, max_queries=args.max_queries, seed=args.seed
        )
        print(f"  training_chunks={len(training_chunks)}")

        print("\n[mine] sqlite-vec ...")
        t0 = time.perf_counter()
        pairs_vec = generate_triples(
            store_vec,
            args.model,
            max_queries=args.max_queries,
            seed=args.seed,
            pre_sampled_chunks=training_chunks,
        )
        mine_vec_s = time.perf_counter() - t0
        print(f"  -> {len(pairs_vec)} pairs in {mine_vec_s:.1f}s")

        store_snap = _build_store(
            tmp_dir / f"{args.dataset}_snap.db",
            "snapvec",
            dim,
            ids,
            texts,
            vecs,
            path_prefix=args.dataset,
        )

        print("\n[mine] snapvec flat ...")
        t0 = time.perf_counter()
        pairs_snap = generate_triples(
            store_snap,
            args.model,
            max_queries=args.max_queries,
            seed=args.seed,
            pre_sampled_chunks=training_chunks,
        )
        mine_snap_s = time.perf_counter() - t0
        print(f"  -> {len(pairs_snap)} pairs in {mine_snap_s:.1f}s")

        diff = _diff_pair_sets(pairs_vec, pairs_snap)
        print("\n### Triple mining diff\n")
        print(json.dumps(diff, indent=2))
        report["mining"] = {
            "sqlite-vec": {"n_pairs": len(pairs_vec), "seconds": round(mine_vec_s, 2)},
            "snapvec": {"n_pairs": len(pairs_snap), "seconds": round(mine_snap_s, 2)},
            "diff": diff,
        }

        if args.train:
            from vstash.retrain import train_mnrl

            args.models_dir.mkdir(parents=True, exist_ok=True)

            # Baseline eval on BOTH backends so the 2x2 is complete.
            # _evaluate_on_backend forces the backend on evaluate_model's
            # internal eval index (see helper docstring).
            print("\n[eval] baseline (v3) on sqlite-vec ...")
            baseline_vec = _evaluate_on_backend(
                store_vec, args.model, eval_queries, "sqlite-vec", args.seed
            )
            print(
                f"  baseline/sqlite-vec NDCG@10={baseline_vec.ndcg_at_10:.4f} "
                f"Recall@100={baseline_vec.recall_at_100:.4f}"
            )

            print("[eval] baseline (v3) on snapvec ...")
            baseline_snap = _evaluate_on_backend(
                store_snap, args.model, eval_queries, "snapvec", args.seed
            )
            print(
                f"  baseline/snapvec NDCG@10={baseline_snap.ndcg_at_10:.4f} "
                f"Recall@100={baseline_snap.recall_at_100:.4f}"
            )

            report["eval"] = {
                "baseline": {
                    "sqlite-vec": baseline_vec.as_dict(),
                    "snapvec": baseline_snap.as_dict(),
                }
            }
            baseline_by_backend = {
                "sqlite-vec": baseline_vec,
                "snapvec": baseline_snap,
            }

            trained_models: dict[str, str] = {}
            for tag, pairs in (("sqlite-vec", pairs_vec), ("snapvec", pairs_snap)):
                out_dir = args.models_dir / f"{args.dataset}_v4_{tag}_{args.loss}"
                if out_dir.exists():
                    import shutil

                    shutil.rmtree(out_dir)

                if args.loss == "triplet":
                    n_triplets = _train_triplet(
                        pairs,
                        base_model=args.model,
                        output_path=str(out_dir),
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        margin=args.triplet_margin,
                    )
                    print(
                        f"\n[train triplet] v4-{tag} ({n_triplets}/{len(pairs)} triplets "
                        f"usable, margin={args.triplet_margin}) -> {out_dir}"
                    )
                    if n_triplets == 0:
                        print(
                            f"  [skip] v4-{tag}: no pairs with explicit hard negative; "
                            f"TripletLoss needs them. Falling back to baseline for this cell."
                        )
                        report["eval"][f"v4_{tag}"] = {
                            "model_path": None,
                            "n_pairs": len(pairs),
                            "n_triplets_used": 0,
                            "skipped": "no explicit hard negatives",
                        }
                        continue
                    effective_pairs = n_triplets
                else:
                    print(f"\n[train mnrl] v4-{tag} ({len(pairs)} pairs) -> {out_dir}")
                    train_mnrl(
                        pairs,
                        base_model=args.model,
                        output_path=str(out_dir),
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        seed=args.seed,
                    )
                    effective_pairs = len(pairs)

                trained_models[tag] = str(out_dir)
                report["eval"][f"v4_{tag}"] = {
                    "model_path": str(out_dir),
                    "n_pairs": len(pairs),
                    "n_effective_pairs": effective_pairs,
                    "loss": args.loss,
                }

            # 2x2 cross eval: each trained model evaluated on each backend.
            print("\n### Cross-backend eval (NDCG@10)\n")
            print("| trained_on \\ eval_on | sqlite-vec | snapvec |")
            print("|---|---|---|")
            for tag, path in trained_models.items():
                row = f"| v4-{tag}"
                for backend_tag, store in (
                    ("sqlite-vec", store_vec),
                    ("snapvec", store_snap),
                ):
                    print(
                        f"  [eval] v4-{tag} on {backend_tag} ...",
                        end="",
                        flush=True,
                    )
                    metrics = _evaluate_on_backend(
                        store, path, eval_queries, backend_tag, args.seed
                    )
                    base_line = baseline_by_backend[backend_tag]
                    delta = metrics.ndcg_at_10 - base_line.ndcg_at_10
                    print(
                        f" NDCG={metrics.ndcg_at_10:.4f} delta={delta:+.4f}",
                        flush=True,
                    )
                    report["eval"][f"v4_{tag}"][f"on_{backend_tag}"] = {
                        **metrics.as_dict(),
                        "delta_ndcg_at_10_vs_baseline": round(delta, 5),
                    }
                    row += f" | {metrics.ndcg_at_10:.4f} ({delta:+.4f})"
                row += " |"
                print(row)
            print(
                f"| baseline (v3) | {baseline_vec.ndcg_at_10:.4f} "
                f"| {baseline_snap.ndcg_at_10:.4f} |"
            )

        store_vec.close()
        store_snap.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
