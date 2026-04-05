"""EmbeddingGemma vs BGE-small — BEIR benchmark comparison.

Compares vstash NDCG@10 using:
  - BGE-small-en-v1.5 (384d, current default)
  - EmbeddingGemma-300m (768d, Google)

Uses the community ONNX version (onnx-community/embeddinggemma-300m-ONNX)
which doesn't require HuggingFace gated access.

Usage:
    python -m experiments.embedding_gemma_eval
    python -m experiments.embedding_gemma_eval --datasets scifact nfcorpus
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from experiments.beir_benchmark import download_beir, load_beir, ndcg_at_k
from vstash.embed import embed_query, embed_texts, get_embedding_dim
from vstash.store import VstashStore

# ------------------------------------------------------------------ #
# EmbeddingGemma ONNX wrapper                                         #
# ------------------------------------------------------------------ #

_gemma_session: ort.InferenceSession | None = None
_gemma_tokenizer: Tokenizer | None = None
GEMMA_DIM = 768
GEMMA_MAX_TOKENS = 2048


def _init_gemma() -> tuple[ort.InferenceSession, Tokenizer]:
    global _gemma_session, _gemma_tokenizer
    if _gemma_session is None:
        repo = "onnx-community/embeddinggemma-300m-ONNX"
        model_path = hf_hub_download(repo, "onnx/model.onnx")
        hf_hub_download(repo, "onnx/model.onnx_data")  # ensure data file exists
        tokenizer_path = hf_hub_download(repo, "tokenizer.json")
        _gemma_session = ort.InferenceSession(model_path)
        _gemma_tokenizer = Tokenizer.from_file(tokenizer_path)
        _gemma_tokenizer.enable_truncation(max_length=GEMMA_MAX_TOKENS)
    return _gemma_session, _gemma_tokenizer


def gemma_embed_one(text: str) -> list[float]:
    """Embed a single text with EmbeddingGemma."""
    session, tokenizer = _init_gemma()
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids[:GEMMA_MAX_TOKENS]], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask[:GEMMA_MAX_TOKENS]], dtype=np.int64)
    outputs = session.run(
        ["sentence_embedding"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    emb = outputs[0]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0].tolist()


def gemma_embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with EmbeddingGemma (one at a time for simplicity)."""
    return [gemma_embed_one(t) for t in texts]


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #

TOP_K = 10
BGE_MODEL = "BAAI/bge-small-en-v1.5"


def evaluate_model(
    dataset: str,
    embed_fn,
    query_fn,
    dim: int,
    model_name: str,
) -> dict:
    """Run BEIR eval with a given embedding function."""
    cache = download_beir(dataset)
    corpus, queries, qrels = load_beir(cache)
    test_qids = [qid for qid in qrels if qid in queries]

    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in doc_ids
    ]

    # Embed corpus
    print(f"    Embedding {len(doc_texts)} docs with {model_name}...")
    t0 = time.perf_counter()
    all_embeddings: list[list[float]] = []
    for bs in range(0, len(doc_texts), 32):
        batch = doc_texts[bs : bs + 32]
        all_embeddings.extend(embed_fn(batch))
        if (bs + 32) % 1000 < 32:
            print(f"      {min(bs + 32, len(doc_texts))}/{len(doc_texts)}...")
    embed_time = time.perf_counter() - t0
    print(f"    Embedded in {embed_time:.1f}s")

    # Ingest
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

    # Evaluate
    ndcgs, latencies = [], []
    for qid in test_qids:
        qe = query_fn(queries[qid])
        t0 = time.perf_counter()
        results = store.search(qe, queries[qid], top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [id_map.get(r.path, "") for r in results]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))

    store.close()
    os.unlink(db_path)

    return {
        "model": model_name,
        "dim": dim,
        "ndcg_10": round(statistics.mean(ndcgs), 4),
        "mrr": None,
        "latency_ms": round(statistics.mean(latencies), 1),
        "embed_time_s": round(embed_time, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbeddingGemma vs BGE-small benchmark")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["scifact", "nfcorpus"],
        help="BEIR datasets to evaluate",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  EmbeddingGemma vs BGE-small — BEIR Benchmark")
    print("=" * 70)

    # Warmup Gemma
    print("\n  Warming up EmbeddingGemma...")
    gemma_embed_one("warmup")

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 60}")

        # BGE-small
        print("\n  [BGE-small-en-v1.5]")
        bge_dim = get_embedding_dim(BGE_MODEL)
        bge = evaluate_model(
            dataset,
            embed_fn=lambda texts: embed_texts(texts, BGE_MODEL),
            query_fn=lambda q: embed_query(q, BGE_MODEL),
            dim=bge_dim,
            model_name="BGE-small-en-v1.5",
        )

        # EmbeddingGemma
        print("\n  [EmbeddingGemma-300m]")
        gemma = evaluate_model(
            dataset,
            embed_fn=gemma_embed_batch,
            query_fn=gemma_embed_one,
            dim=GEMMA_DIM,
            model_name="EmbeddingGemma-300m",
        )

        delta = gemma["ndcg_10"] - bge["ndcg_10"]
        pct = (delta / bge["ndcg_10"] * 100) if bge["ndcg_10"] > 0 else 0

        print(f"\n  {'':25s} {'BGE-small':>12s} {'Gemma-300m':>12s} {'Delta':>10s}")
        print(f"  {'─' * 62}")
        print(f"  {'NDCG@10':25s} {bge['ndcg_10']:12.4f} {gemma['ndcg_10']:12.4f} {delta:+10.4f}")
        print(f"  {'':25s} {'':12s} {'':12s} {pct:+9.2f}%")
        print(f"  {'Latency (ms)':25s} {bge['latency_ms']:12.1f} {gemma['latency_ms']:12.1f}")
        print(f"  {'Embed time (s)':25s} {bge['embed_time_s']:12.1f} {gemma['embed_time_s']:12.1f}")

        all_results.append(
            {
                "dataset": dataset,
                "bge": bge,
                "gemma": gemma,
                "delta": round(delta, 4),
                "delta_pct": round(pct, 2),
            }
        )

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/embedding_gemma_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n  Results saved to experiments/results/embedding_gemma_eval.json")


if __name__ == "__main__":
    main()
