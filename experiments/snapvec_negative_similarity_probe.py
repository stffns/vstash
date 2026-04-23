"""
snapvec_negative_similarity_probe.py -- discriminate noise vs real-hard-neg.

Given the pairs produced by ``retrain_v4_snapvec.py``, compute exact
(non-quantized) cosine similarity between each (query, negative) and
each (query, positive). Separate the negatives by origin:

- ``both``: sqlite-vec and snapvec both emitted a negative for this
  query (may be different texts).
- ``only_vec``: only sqlite-vec emitted a negative.
- ``only_snap``: only snapvec emitted a negative. This is the bucket
  that the v4 discussion hinges on -- are these negatives close to
  the query (real hard negatives the ANN surfaced) or are they
  scattered low-similarity docs (quantization noise)?

Hypothesis A (noise): distribution of cos(q, neg) for ``only_snap``
lies well below cos(q, pos), with a long low tail.

Hypothesis B (hard-neg): distribution of cos(q, neg) for ``only_snap``
overlaps with the upper-end of cos(q, random-doc) or cos(q, real hard
negatives), close to but below cos(q, pos).

The probe uses the SAME exact sentence-transformer encoder as the
training path, so "noise" means noise in the embedding space the
model actually works in -- not noise in the quantized space only.

Usage:

    python -m experiments.snapvec_negative_similarity_probe \
        --dataset scifact --max-queries 300

Reads the existing pair JSONs from ``retrain_v4_snapvec.py`` by
re-running the mining (cached) rather than persisting the raw pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
from experiments.retrain_v4_snapvec import (
    _build_store,
    _pair_key,
    _patch_embed_query_for_finetuned_model,
)
from vstash.embed import get_embedding_dim
from vstash.retrain import generate_triples, sample_training_chunks


def _encode_many(texts: list[str], model_name: str) -> np.ndarray:
    st = _get_st_model(model_name)
    return st.encode(
        texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)


def _summarize(name: str, cosines: np.ndarray) -> dict:
    if len(cosines) == 0:
        return {"count": 0}
    return {
        "count": int(len(cosines)),
        "min": round(float(cosines.min()), 4),
        "p10": round(float(np.percentile(cosines, 10)), 4),
        "p25": round(float(np.percentile(cosines, 25)), 4),
        "median": round(float(np.median(cosines)), 4),
        "p75": round(float(np.percentile(cosines, 75)), 4),
        "p90": round(float(np.percentile(cosines, 90)), 4),
        "max": round(float(cosines.max()), 4),
        "mean": round(float(cosines.mean()), 4),
    }


def _print_table(label: str, summaries: dict[str, dict]) -> None:
    cols = ["count", "min", "p10", "p25", "median", "p75", "p90", "max", "mean"]
    print(f"\n### {label}\n")
    print("| bucket | " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for bucket, s in summaries.items():
        if s["count"] == 0:
            print(f"| {bucket} | 0 | - | - | - | - | - | - | - | - |")
            continue
        print("| " + bucket + " | " + " | ".join(str(s[c]) for c in cols) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/data/beir_snapvec_h2h_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/snapvec_negative_similarity_probe.json"),
    )
    args = parser.parse_args()

    dim = get_embedding_dim(args.model)
    print(
        f"=== snapvec negative similarity probe ===\n"
        f"dataset={args.dataset} model={args.model} dim={dim} "
        f"max_queries={args.max_queries}"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    _patch_embed_query_for_finetuned_model()

    ids, texts, vecs = _embed_corpus(args.dataset, args.model, args.cache_dir)
    cache = download_beir(args.dataset)
    _, queries, qrels = load_beir(cache)
    _embed_queries(args.dataset, args.model, queries, args.cache_dir)

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
        training_chunks = sample_training_chunks(
            store_vec, max_queries=args.max_queries, seed=args.seed
        )

        print("\n[mine] sqlite-vec ...")
        t0 = time.perf_counter()
        pairs_vec = generate_triples(
            store_vec,
            args.model,
            max_queries=args.max_queries,
            seed=args.seed,
            pre_sampled_chunks=training_chunks,
        )
        print(f"  {len(pairs_vec)} pairs in {time.perf_counter() - t0:.1f}s")

        store_snap = _build_store(
            tmp_dir / f"{args.dataset}_snap.db",
            "snapvec",
            dim,
            ids,
            texts,
            vecs,
            path_prefix=args.dataset,
        )

        print("[mine] snapvec flat ...")
        t0 = time.perf_counter()
        pairs_snap = generate_triples(
            store_snap,
            args.model,
            max_queries=args.max_queries,
            seed=args.seed,
            pre_sampled_chunks=training_chunks,
        )
        print(f"  {len(pairs_snap)} pairs in {time.perf_counter() - t0:.1f}s")

        store_vec.close()
        store_snap.close()

    vec_by_key = {_pair_key(p): p for p in pairs_vec}
    snap_by_key = {_pair_key(p): p for p in pairs_snap}
    shared = sorted(set(vec_by_key) & set(snap_by_key))

    # Precompute exact (non-quantized) embeddings for every unique
    # query / positive / negative text in one batched encode call.
    text_to_idx: dict[str, int] = {}

    def _register(t: str | None) -> int:
        if not t:
            return -1
        if t not in text_to_idx:
            text_to_idx[t] = len(text_to_idx)
        return text_to_idx[t]

    entries = []
    for k in shared:
        pv = vec_by_key[k]
        ps = snap_by_key[k]
        entries.append(
            {
                "key": k,
                "query_idx": _register(pv.get("query")),
                "positive_idx": _register(pv.get("positive")),
                "vec_neg_idx": _register(pv.get("negative")),
                "snap_neg_idx": _register(ps.get("negative")),
            }
        )

    unique_texts = [None] * len(text_to_idx)
    for t, i in text_to_idx.items():
        unique_texts[i] = t

    print(f"\n[encode] {len(unique_texts)} unique texts ...")
    embeddings = _encode_many(unique_texts, args.model)

    def _cos(i: int, j: int) -> float:
        if i < 0 or j < 0:
            return float("nan")
        # Embeddings are L2-normalized so dot product == cosine similarity.
        return float(np.dot(embeddings[i], embeddings[j]))

    # Bucket each shared query by negative origin and collect cosines.
    buckets_pos: dict[str, list[float]] = {
        "both_have_neg": [],
        "only_vec_has_neg": [],
        "only_snap_has_neg": [],
        "neither_has_neg": [],
    }
    buckets_vec_neg: dict[str, list[float]] = {
        "both_have_neg": [],
        "only_vec_has_neg": [],
    }
    buckets_snap_neg: dict[str, list[float]] = {
        "both_have_neg": [],
        "only_snap_has_neg": [],
    }

    for e in entries:
        v_has = e["vec_neg_idx"] >= 0
        s_has = e["snap_neg_idx"] >= 0
        if v_has and s_has:
            bucket = "both_have_neg"
        elif v_has:
            bucket = "only_vec_has_neg"
        elif s_has:
            bucket = "only_snap_has_neg"
        else:
            bucket = "neither_has_neg"

        buckets_pos[bucket].append(_cos(e["query_idx"], e["positive_idx"]))
        if v_has:
            buckets_vec_neg[bucket if bucket in buckets_vec_neg else "both_have_neg"].append(
                _cos(e["query_idx"], e["vec_neg_idx"])
            )
        if s_has:
            buckets_snap_neg[bucket if bucket in buckets_snap_neg else "both_have_neg"].append(
                _cos(e["query_idx"], e["snap_neg_idx"])
            )

    pos_summary = {k: _summarize(k, np.array(v)) for k, v in buckets_pos.items()}
    vec_neg_summary = {k: _summarize(k, np.array(v)) for k, v in buckets_vec_neg.items()}
    snap_neg_summary = {k: _summarize(k, np.array(v)) for k, v in buckets_snap_neg.items()}

    # Also: per-bucket margin (cos(q, pos) - cos(q, neg)) for the
    # "real hard negative" test. Hard negatives have SMALL positive
    # margins; noise negatives have LARGE margins.
    margins_vec = []
    for e in entries:
        if e["vec_neg_idx"] < 0:
            continue
        margins_vec.append(
            _cos(e["query_idx"], e["positive_idx"]) - _cos(e["query_idx"], e["vec_neg_idx"])
        )
    margins_snap_all = []
    margins_snap_onlysnap = []
    for e in entries:
        if e["snap_neg_idx"] < 0:
            continue
        m = _cos(e["query_idx"], e["positive_idx"]) - _cos(e["query_idx"], e["snap_neg_idx"])
        margins_snap_all.append(m)
        if e["vec_neg_idx"] < 0:
            margins_snap_onlysnap.append(m)

    _print_table("cos(query, positive) per bucket", pos_summary)
    _print_table("cos(query, NEGATIVE mined by sqlite-vec)", vec_neg_summary)
    _print_table("cos(query, NEGATIVE mined by snapvec)", snap_neg_summary)

    print("\n### margin = cos(q, pos) - cos(q, neg)\n")
    print("Lower margin = harder negative (closer to positive in embedding space).\n")
    print(
        f"sqlite-vec negatives (n={len(margins_vec)}): "
        f"median={statistics.median(margins_vec):.4f}  "
        f"p25={statistics.quantiles(margins_vec, n=4)[0]:.4f}  "
        f"p75={statistics.quantiles(margins_vec, n=4)[-1]:.4f}"
        if len(margins_vec) >= 4
        else f"sqlite-vec negatives (n={len(margins_vec)}): "
        f"median={statistics.median(margins_vec) if margins_vec else float('nan'):.4f}"
    )
    print(
        f"snapvec negatives (all, n={len(margins_snap_all)}): "
        f"median={statistics.median(margins_snap_all):.4f}  "
        f"p25={statistics.quantiles(margins_snap_all, n=4)[0]:.4f}  "
        f"p75={statistics.quantiles(margins_snap_all, n=4)[-1]:.4f}"
        if len(margins_snap_all) >= 4
        else f"snapvec negatives (n={len(margins_snap_all)}): median=-"
    )
    print(
        f"snapvec-only negatives (n={len(margins_snap_onlysnap)}): "
        f"median={statistics.median(margins_snap_onlysnap):.4f}  "
        f"p25={statistics.quantiles(margins_snap_onlysnap, n=4)[0]:.4f}  "
        f"p75={statistics.quantiles(margins_snap_onlysnap, n=4)[-1]:.4f}"
        if len(margins_snap_onlysnap) >= 4
        else f"snapvec-only negatives (n={len(margins_snap_onlysnap)}): median=-"
    )

    out = {
        "dataset": args.dataset,
        "model": args.model,
        "max_queries": args.max_queries,
        "n_shared_queries": len(entries),
        "bucket_counts": {k: pos_summary[k]["count"] for k in pos_summary},
        "cos_query_positive_by_bucket": pos_summary,
        "cos_query_negative_vec_by_bucket": vec_neg_summary,
        "cos_query_negative_snap_by_bucket": snap_neg_summary,
        "margin_pos_minus_neg": {
            "sqlite-vec (all)": _summarize("vec", np.array(margins_vec)),
            "snapvec (all)": _summarize("snap", np.array(margins_snap_all)),
            "snapvec-only (noise test bucket)": _summarize(
                "snap_only", np.array(margins_snap_onlysnap)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
