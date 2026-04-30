"""RRF Disagreement Training Pair Generator

Generates (query, positive, hard_negative) triples from the disagreement
between vector and FTS signals in vstash's hybrid retrieval pipeline.

Thesis: chunks ranked high by one signal but absent from the other's top-k
are natural hard negatives that require no human labeling. Fine-tuning a
dense encoder on these pairs should teach it to distinguish "semantically
close" from "actually relevant."

Usage:
    python -m experiments.rrf_training_pairs
    python -m experiments.rrf_training_pairs --datasets scifact nfcorpus fiqa
    python -m experiments.rrf_training_pairs --output experiments/results/training_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from experiments.beir_benchmark import DATASETS, download_beir, load_beir
from vstash.embed import _embed_onnx, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 10


def generate_pairs_for_dataset(
    name: str,
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dim: int,
    max_docs: int = 5000,
) -> tuple[list[dict], dict]:
    """Generate training pairs from RRF disagreement for one BEIR dataset.

    Returns (pairs, stats) where pairs is a list of training triples and
    stats is a summary dict.
    """
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {name}")
    print(f"{'=' * 60}")

    db_path = tempfile.mktemp(suffix=".db", prefix=f"vstash_pairs_{name}_")
    store = VstashStore(db_path, embedding_dim=dim)

    # Ingest corpus
    doc_items = list(corpus.items())[:max_docs]
    print(f"  Ingesting {len(doc_items)} docs...")
    doc_id_to_path: dict[str, str] = {}
    doc_texts: dict[str, str] = {}
    for doc_id, doc in doc_items:
        text = doc.get("title", "") + "\n" + doc.get("text", "")
        emb = _embed_onnx([text[:512]], MODEL)[0]
        path = f"/{name}/{doc_id}"
        doc_id_to_path[doc_id] = path
        doc_texts[path] = text
        store.add_document(
            path=path,
            title=doc.get("title", doc_id),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )

    query_items = [(qid, qt) for qid, qt in queries.items() if qid in qrels]

    print(f"  Processing {len(query_items)} queries...")

    pairs = []
    total_disagreements = 0
    total_queries = 0

    for qid, qtext in query_items:
        emb = _embed_onnx([qtext], MODEL)[0]
        gold_paths = {
            doc_id_to_path[did]
            for did in qrels.get(qid, {})
            if qrels[qid][did] > 0 and did in doc_id_to_path
        }

        if not gold_paths:
            continue

        total_queries += 1

        # Vector-heavy and FTS-heavy searches
        vec_results = store.search(
            query_embedding=emb,
            query_text=qtext,
            top_k=TOP_K,
            vec_weight=0.95,
            fts_weight=0.05,
            adaptive_rrf=False,
        )
        fts_results = store.search(
            query_embedding=emb,
            query_text=qtext,
            top_k=TOP_K,
            vec_weight=0.05,
            fts_weight=0.95,
            adaptive_rrf=False,
        )

        vec_paths = [r.path for r in vec_results[:TOP_K]]
        fts_paths = [r.path for r in fts_results[:TOP_K]]
        vec_set = set(vec_paths)
        fts_set = set(fts_paths)

        has_disagreement = vec_set != fts_set
        if has_disagreement:
            total_disagreements += 1

        # Hard negatives from vector: high in vec, absent from FTS, not gold
        vec_only_neg = [p for p in vec_paths if p not in fts_set and p not in gold_paths]
        # Hard negatives from FTS: high in FTS, absent from vec, not gold
        fts_only_neg = [p for p in fts_paths if p not in vec_set and p not in gold_paths]

        # Build triples: (query, positive_text, hard_negative_text)
        for gold_path in gold_paths:
            if gold_path not in doc_texts:
                continue
            positive_text = doc_texts[gold_path][:512]

            # Type 1: vector found irrelevant, FTS didn't (dense hard negative)
            for neg_path in vec_only_neg:
                if neg_path in doc_texts:
                    pairs.append(
                        {
                            "query": qtext,
                            "positive": positive_text,
                            "negative": doc_texts[neg_path][:512],
                            "neg_type": "vec_only",
                            "dataset": name,
                            "query_id": qid,
                        }
                    )

            # Type 2: FTS found irrelevant, vector didn't (lexical hard negative)
            for neg_path in fts_only_neg:
                if neg_path in doc_texts:
                    pairs.append(
                        {
                            "query": qtext,
                            "positive": positive_text,
                            "negative": doc_texts[neg_path][:512],
                            "neg_type": "fts_only",
                            "dataset": name,
                            "query_id": qid,
                        }
                    )

    store.close()
    Path(db_path).unlink(missing_ok=True)

    vec_neg = sum(1 for p in pairs if p["neg_type"] == "vec_only")
    fts_neg = sum(1 for p in pairs if p["neg_type"] == "fts_only")

    stats = {
        "dataset": name,
        "docs_ingested": len(doc_items),
        "queries_with_gold": total_queries,
        "queries_with_disagreement": total_disagreements,
        "disagreement_rate": round(total_disagreements / total_queries, 3) if total_queries else 0,
        "total_triples": len(pairs),
        "vec_only_negatives": vec_neg,
        "fts_only_negatives": fts_neg,
    }

    print(f"  Queries with gold: {total_queries}")
    print(f"  Disagreement rate: {stats['disagreement_rate']:.1%}")
    print(f"  Training triples: {len(pairs)} (vec_neg={vec_neg}, fts_neg={fts_neg})")

    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RRF disagreement training pairs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "fiqa"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/rrf_training_pairs.jsonl",
    )
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)

    # Load datasets
    print("Loading BEIR datasets...")
    all_data: dict[str, tuple[dict, dict, dict]] = {}
    for name in args.datasets:
        cache = download_beir(name)
        all_data[name] = load_beir(cache)
        c, q, _qr = all_data[name]
        print(f"  {name}: {len(c)} docs, {len(q)} queries")

    # Generate pairs
    all_pairs = []
    all_stats = []
    for name in args.datasets:
        corpus, queries, qrels = all_data[name]
        pairs, stats = generate_pairs_for_dataset(
            name, corpus, queries, qrels, dim, max_docs=args.max_docs
        )
        all_pairs.extend(pairs)
        all_stats.append(stats)

    # Shuffle
    rng = random.Random(42)
    rng.shuffle(all_pairs)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'Dataset':>12} {'Queries':>8} {'Disagree':>10} {'Triples':>10} {'VecNeg':>8} {'FtsNeg':>8}"
    )
    print(f"  {'-' * 58}")
    for s in all_stats:
        print(
            f"  {s['dataset']:>12} {s['queries_with_gold']:>8} "
            f"{s['disagreement_rate']:>9.1%} {s['total_triples']:>10} "
            f"{s['vec_only_negatives']:>8} {s['fts_only_negatives']:>8}"
        )
    print(f"  {'TOTAL':>12} {'':>8} {'':>10} {len(all_pairs):>10}")

    # Save as JSONL (sentence-transformers InputExample format)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n  Saved {len(all_pairs)} triples to {output_path}")

    # Also save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "datasets": args.datasets,
                "per_dataset": all_stats,
                "total_triples": len(all_pairs),
            },
            f,
            indent=2,
        )
    print(f"  Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
