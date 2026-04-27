"""ColBERTv2 on BEIR (head-to-head with experiments/beir_benchmark.py).

Runs the same NDCG@10 / MRR / Recall@10 evaluation on the same 5 BEIR
datasets we ship in the paper, but with ColBERTv2 (multi-vector
MaxSim) instead of vstash (single-vector + FTS5 + adaptive RRF).
Outputs the same JSON shape as ``experiments.beir_benchmark`` so the
H2H notebook can build the comparison table mechanically.

Reuses ``download_beir``, ``load_beir``, ``ndcg_at_k``, ``mrr``, and
``recall_at_k`` from ``beir_benchmark`` so the corpus fetching, qrels
parsing, and metric definitions stay in lockstep.

CUDA-required: ColBERT MaxSim on CPU at BEIR scale (5K-57K docs * tens
of tokens each * many queries) is impractical.  pylate auto-uses CUDA
when available.

Usage (Colab T4):
    python -m experiments.beir_colbert \\
        --datasets scifact nfcorpus fiqa scidocs arguana \\
        --output experiments/results/beir_colbertv2.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from experiments.beir_benchmark import (
    DATASETS,
    TOP_K,
    download_beir,
    load_beir,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

DEFAULT_MODEL = "colbert-ir/colbertv2.0"


def _load_engine(engine: str, model_name: str, device: str):
    """Engine-agnostic loader. Returns an object with encode_query /
    encode_docs / score on top of either pylate or the raw-HF
    MinimalColBERT (see experiments/colbert_minimal.py).
    """
    if engine == "pylate":
        try:
            from pylate import models  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "pylate is required for --engine pylate. Install with: "
                "pip install pylate, or pass --engine minimal to use the "
                "raw-HF fallback (no pylate / sentence-transformers needed)."
            ) from exc

        pylate_model = models.ColBERT(model_name_or_path=model_name, device=device)

        class _PylateEngine:
            def encode_queries(self, texts, batch_size=64):
                return pylate_model.encode(
                    texts, is_query=True, batch_size=batch_size, show_progress_bar=False
                )

            def encode_docs(self, texts, batch_size=64):
                return pylate_model.encode(
                    texts, is_query=False, batch_size=batch_size, show_progress_bar=True
                )

            def score(self, q_batch, d_all):
                from pylate import scores as colbert_scores  # type: ignore

                return colbert_scores.colbert_scores(q_batch, d_all)

        return _PylateEngine()

    if engine == "minimal":
        from experiments.colbert_minimal import MinimalColBERT

        impl = MinimalColBERT(model_name=model_name, device=device)

        class _MinimalEngine:
            def encode_queries(self, texts, batch_size=64):
                # MinimalColBERT.encode_queries returns list of (seq_q, 128)
                # We pad them into a single tensor for batched matmul.
                import torch

                emb_list = impl.encode_queries(texts, batch_size=batch_size)
                if not emb_list:
                    return torch.empty(0, 0, 128, device=impl.device)
                max_len = max(e.shape[0] for e in emb_list)
                padded = []
                for e in emb_list:
                    pad = max_len - e.shape[0]
                    if pad:
                        e = torch.nn.functional.pad(e, (0, 0, 0, pad))
                    padded.append(e)
                return torch.stack(padded, dim=0)

            def encode_docs(self, texts, batch_size=64):
                doc_embs, _ = impl.encode_documents(texts, batch_size=batch_size)
                return doc_embs

            def score(self, q_batch, d_all):
                # q_batch: (n_q, seq_q, 128); d_all: (n_d, seq_d, 128).
                # Returns (n_q, n_d).  einsum is the cleanest way.
                import torch

                # For each query, compute MaxSim against every doc.
                sims = torch.einsum("qmh,dnh->qdmn", q_batch, d_all)
                # MaxSim: max over d-tokens, sum over q-tokens.
                return sims.max(dim=3).values.sum(dim=2)

        return _MinimalEngine()

    raise SystemExit(f"Unknown --engine: {engine!r}; expected 'pylate' or 'minimal'.")


def evaluate_colbert(
    engine,
    corpus: dict,
    queries: dict,
    qrels: dict,
    *,
    encode_batch_size: int,
    query_batch_size: int,
) -> dict:
    """Index the corpus once, score every test query.

    Indexing strategy: encode all docs in one pass and keep the multi-
    vector tensor on GPU.  Per-query scoring is a single
    ``colbert_scores(q, all_docs)`` call -- pylate handles the
    block-matmul internally and returns one row per query.

    For BEIR-class corpora (~5K-60K docs at ~80-220 tokens each) this
    fits in 4-8 GB of T4 VRAM with the default batch sizes; FiQA at
    57K docs is the tight upper bound.
    """
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]

    t0 = time.perf_counter()
    doc_emb = engine.encode_docs(doc_texts, batch_size=encode_batch_size)
    encode_corpus_s = time.perf_counter() - t0
    print(f"  [ColBERT] encoded {len(doc_ids)} docs in {encode_corpus_s:.1f}s")

    test_qids = list(qrels.keys())
    query_texts = [queries[qid] for qid in test_qids]
    t0 = time.perf_counter()
    query_emb = engine.encode_queries(query_texts, batch_size=query_batch_size)
    encode_queries_s = time.perf_counter() - t0

    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []

    # Score in chunks so we can time per-query without paying for the
    # full all-vs-all matmul on every loop iteration.
    import numpy as np

    BATCH = 32
    t0_search = time.perf_counter()
    for start in range(0, len(test_qids), BATCH):
        end = min(start + BATCH, len(test_qids))
        qb = query_emb[start:end]
        sim = engine.score(qb, doc_emb)
        sim_np = sim.cpu().numpy() if hasattr(sim, "cpu") else np.asarray(sim)
        for j, qid in enumerate(test_qids[start:end]):
            row = sim_np[j]
            t0 = time.perf_counter()
            if TOP_K >= len(row):
                idx = np.argsort(-row)
            else:
                part = np.argpartition(-row, TOP_K - 1)[:TOP_K]
                idx = part[np.argsort(-row[part])]
            ranked = [doc_ids[k] for k in idx]
            latencies.append((time.perf_counter() - t0) * 1000)
            ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))
            mrrs.append(mrr(ranked, qrels[qid]))
            recalls.append(recall_at_k(ranked, qrels[qid], TOP_K))
    search_s = time.perf_counter() - t0_search

    return {
        "ndcg_10": round(statistics.mean(ndcgs), 4),
        "mrr": round(statistics.mean(mrrs), 4),
        "recall_10": round(statistics.mean(recalls), 4),
        "latency_ms": round(statistics.mean(latencies), 3),
        "encode_corpus_s": round(encode_corpus_s, 1),
        "encode_queries_s": round(encode_queries_s, 2),
        "search_total_s": round(search_s, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="ColBERTv2 BEIR benchmark")
    p.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--output", type=Path,
                   default=Path("experiments/results/beir_colbertv2.json"))
    p.add_argument(
        "--engine",
        choices=["pylate", "minimal"],
        default="pylate",
        help="ColBERT inference backend.  'minimal' is the raw-HF fallback "
        "(experiments/colbert_minimal.py) -- no pylate, no sentence-transformers, "
        "needs only torch + transformers.  Use it when Colab's pylate / "
        "sentence-transformers / torchcodec dependency cascade is unstable.",
    )
    args = p.parse_args()

    print(f"Loading {args.model} on {args.device} (engine={args.engine})...", flush=True)
    engine = _load_engine(args.engine, args.model, args.device)
    print("Loaded.", flush=True)

    os.makedirs(args.output.parent, exist_ok=True)
    all_results: list[dict] = []

    for ds_name in args.datasets:
        print(f"\n{'=' * 70}\n  ColBERTv2 on BEIR: {ds_name}\n{'=' * 70}")
        cache = download_beir(ds_name)
        corpus, queries, qrels = load_beir(cache)
        test_qids = list(qrels.keys())
        print(f"  Corpus: {len(corpus)} docs, Queries: {len(test_qids)} with qrels")

        if len(corpus) > DATASETS[ds_name]["max_corpus"]:
            print(f"  SKIPPING ({len(corpus)} > {DATASETS[ds_name]['max_corpus']})")
            continue

        metrics = evaluate_colbert(
            engine,
            corpus,
            queries,
            qrels,
            encode_batch_size=args.encode_batch_size,
            query_batch_size=args.query_batch_size,
        )
        result = {
            "dataset": ds_name,
            "docs": len(corpus),
            "queries": len(test_qids),
            "model": args.model,
            "device": args.device,
            "colbertv2": metrics,
        }
        all_results.append(result)
        print(
            f"  [ColBERT] NDCG@10={metrics['ndcg_10']:.4f}  "
            f"Recall@10={metrics['recall_10']:.4f}  MRR={metrics['mrr']:.4f}"
        )

        # Free GPU memory between datasets so a long FiQA run does not
        # leave us OOM on SciDocs.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
