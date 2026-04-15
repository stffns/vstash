"""Fine-tune BGE-small using RRF disagreement hard negatives.

Takes the training triples from rrf_training_pairs.py and fine-tunes
BGE-small-en-v1.5 with TripletLoss. The result is a model that should
better distinguish "semantically close" from "actually relevant" by
learning from the cases where vector and FTS signals disagree.

Requirements:
    pip install sentence-transformers torch

Usage:
    # Generate pairs first (if not already done):
    python -m experiments.rrf_training_pairs

    # Fine-tune:
    python -m experiments.finetune_rrf

    # With custom settings:
    python -m experiments.finetune_rrf --epochs 5 --batch-size 32
    python -m experiments.finetune_rrf --input path/to/pairs.jsonl

    # Evaluate after training:
    python -m experiments.finetune_rrf --evaluate-only --model-path experiments/models/vstash-bge-rrf
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

MODEL_BASE = "BAAI/bge-small-en-v1.5"
DEFAULT_PAIRS = "experiments/results/rrf_training_pairs.jsonl"
DEFAULT_OUTPUT = "experiments/models/vstash-bge-rrf"


def load_triples(path: str, max_triples: int = 0) -> list[dict]:
    """Load training triples from JSONL."""
    triples = []
    with open(path) as f:
        for line in f:
            triples.append(json.loads(line))
            if max_triples and len(triples) >= max_triples:
                break
    return triples


def train(
    pairs_path: str,
    output_path: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_triples: int = 0,
    warmup_ratio: float = 0.1,
) -> None:
    """Fine-tune BGE-small with triplet loss on RRF disagreement pairs."""
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: sentence-transformers and torch required.")
        print("  pip install sentence-transformers torch")
        return

    print(f"Loading triples from {pairs_path}...")
    raw_triples = load_triples(pairs_path, max_triples)
    print(f"  Loaded {len(raw_triples)} triples")

    # Convert to InputExamples
    examples = [InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in raw_triples]

    print(f"\nLoading base model: {MODEL_BASE}")
    model = SentenceTransformer(MODEL_BASE)

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model=model)

    warmup_steps = int(len(loader) * epochs * warmup_ratio)
    print(f"  Training: {len(examples)} examples, {epochs} epochs, batch={batch_size}")
    print(f"  Steps/epoch: {len(loader)}, warmup: {warmup_steps}")
    print(f"  LR: {lr}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining...")
    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")

    model.save(str(output_dir))
    print(f"  Model saved to {output_dir}")

    # Save training metadata
    meta = {
        "base_model": MODEL_BASE,
        "pairs_file": pairs_path,
        "n_triples": len(raw_triples),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "training_time_s": round(elapsed, 1),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))


def evaluate(model_path: str) -> None:
    """Compare fine-tuned model against base on SciFact NDCG@10."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers required.")
        return

    from experiments.beir_benchmark import download_beir, load_beir
    from vstash.store import VstashStore

    cache = download_beir("scifact")
    corpus, queries, qrels = load_beir(cache)
    doc_items = list(corpus.items())[:2000]
    query_items = [(qid, qt) for qid, qt in queries.items() if qid in qrels]

    def ndcg_at_k(ranked_ids: list[str], qr: dict[str, int], k: int = 10) -> float:
        gains = [qr.get(did, 0) for did in ranked_ids[:k]]
        ideal = sorted(qr.values(), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        dcg_val = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
        return dcg_val / idcg if idcg > 0 else 0.0

    models_to_test = {
        "BGE-small (base)": MODEL_BASE,
        "vstash-bge-rrf (tuned)": model_path,
    }

    print(f"Evaluating on SciFact ({len(doc_items)} docs, {len(query_items)} queries)")
    print()
    print(f"  {'Model':>30} {'NDCG@10':>10} {'Embed/doc':>10} {'Search':>10}")
    print(f"  {'-' * 62}")

    for label, mpath in models_to_test.items():
        st_model = SentenceTransformer(mpath)
        dim = st_model.get_sentence_embedding_dimension()

        db_path = tempfile.mktemp(suffix=".db")
        store = VstashStore(db_path, embedding_dim=dim)

        doc_id_to_path: dict[str, str] = {}
        embed_times = []
        for doc_id, doc in doc_items:
            text = doc.get("title", "") + "\n" + doc.get("text", "")
            t0 = time.perf_counter()
            emb = st_model.encode(text[:512]).tolist()
            embed_times.append((time.perf_counter() - t0) * 1000)
            path = f"/sf/{doc_id}"
            doc_id_to_path[doc_id] = path
            store.add_document(
                path=path,
                title=doc.get("title", doc_id),
                chunks=[text],
                embeddings=[emb],
                source_type="text",
            )

        path_to_id = {v: k for k, v in doc_id_to_path.items()}

        latencies = []
        ndcgs = []
        for qid, qtext in query_items:
            emb = st_model.encode(qtext).tolist()
            t0 = time.perf_counter()
            results = store.search(query_embedding=emb, query_text=qtext, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            ranked = [path_to_id.get(r.path) for r in results if r.path in path_to_id]
            ndcgs.append(ndcg_at_k(ranked, qrels.get(qid, {}), 10))

        embed_med = statistics.median(embed_times)
        search_med = statistics.median(latencies)
        ndcg_mean = statistics.mean(ndcgs)

        print(f"  {label:>30} {ndcg_mean:>10.4f} {embed_med:>8.1f}ms {search_med:>8.1f}ms")

        store.close()
        Path(db_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BGE-small with RRF disagreement pairs")
    parser.add_argument("--input", type=str, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-triples", type=int, default=0, help="0 = use all")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate(args.model_path)
    else:
        train(
            pairs_path=args.input,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_triples=args.max_triples,
        )
        print("\nRunning evaluation...")
        evaluate(args.output)


if __name__ == "__main__":
    main()
