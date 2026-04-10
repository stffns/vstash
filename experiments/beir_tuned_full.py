"""BEIR Full Pipeline Benchmark — Base vs Tuned Model

Runs the complete vstash pipeline (vector + FTS5 + adaptive RRF + MMR dedup)
on all 5 BEIR datasets, comparing BGE-small base against the RRF-tuned model.

Unlike eval_beir_tuned.py which only measures the vector component,
this script uses VstashStore.search() with all pipeline stages active,
producing numbers directly comparable to the paper's Table 8/10.

Usage (run in Colab with sentence-transformers installed):
    PYTHONPATH=/content/vstash python experiments/beir_tuned_full.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Import vstash store and BEIR loading utilities
from vstash.store import VstashStore

BEIR_CACHE = "experiments/data"
TOP_K = 10

MODELS = {
    "BGE-small (base)": "BAAI/bge-small-en-v1.5",
    "vstash-bge-rrf (tuned)": "stffens/bge-small-rrf-v1",
}

DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs", "arguana"]

BASELINES = {
    "scifact": {"BM25": 0.665, "ColBERTv2": 0.693},
    "nfcorpus": {"BM25": 0.325, "ColBERTv2": 0.344},
    "fiqa": {"BM25": 0.236, "ColBERTv2": 0.356},
    "scidocs": {"BM25": 0.158, "ColBERTv2": 0.154},
    "arguana": {"BM25": 0.315, "ColBERTv2": 0.463},
}


def download_beir(name: str) -> str:
    """Download and cache a BEIR dataset."""
    import io
    import urllib.request
    import zipfile
    import shutil

    cache = f"{BEIR_CACHE}/beir_{name}"
    if not os.path.exists(cache):
        print(f"  Downloading {name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        data = urllib.request.urlopen(url).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        raw_dir = f"{BEIR_CACHE}/beir_{name}_raw"
        z.extractall(raw_dir)
        os.rename(f"{raw_dir}/{name}", cache)
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
    return cache


def load_beir(cache: str):
    """Load corpus, queries, qrels."""
    corpus = {}
    with open(f"{cache}/corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    queries = {}
    with open(f"{cache}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels = {}
    with open(f"{cache}/qrels/test.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score
    return corpus, queries, qrels


def ndcg_at_k(ranked_ids, qr, k=10):
    gains = [qr.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qr.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    os.makedirs("experiments/results", exist_ok=True)
    all_results = {}

    for ds_name in DATASETS:
        cache = download_beir(ds_name)
        corpus, queries, qrels = load_beir(cache)
        test_qids = list(qrels.keys())

        print(f"\n{'=' * 70}")
        print(f"  {ds_name}: {len(corpus)} docs, {len(test_qids)} queries")
        print(f"{'=' * 70}")

        all_results[ds_name] = {}

        for label, model_path in MODELS.items():
            print(f"\n  [{label}]")
            st_model = SentenceTransformer(model_path)
            dim = st_model.get_sentence_embedding_dimension()

            # Ingest with full vstash pipeline
            db_path = tempfile.mktemp(suffix=".db")
            store = VstashStore(db_path, embedding_dim=dim)

            print(f"    Embedding + ingesting {len(corpus)} docs...")
            doc_id_map = {}
            batch_texts = []
            batch_ids = []
            for doc_id, doc in corpus.items():
                text = (doc.get("title", "") + "\n" + doc.get("text", "")).strip()
                batch_texts.append(text[:512])
                batch_ids.append(doc_id)

            # Batch embed
            print(f"    Encoding {len(batch_texts)} texts...")
            all_embs = st_model.encode(batch_texts, show_progress_bar=True, batch_size=256)

            print("    Ingesting into vstash...")
            for doc_id, text, emb in zip(batch_ids, batch_texts, all_embs):
                path = f"/beir/{doc_id}"
                store.add_document(
                    path=path,
                    title=corpus[doc_id].get("title", ""),
                    chunks=[text],
                    embeddings=[emb.tolist()],
                    source_type="text",
                )
                doc_id_map[path] = doc_id

            # Search with full vstash pipeline (adaptive RRF, FTS5, MMR)
            print(f"    Running {len(test_qids)} queries (full pipeline)...")
            ndcgs = []
            latencies = []
            for qid in test_qids:
                qemb = st_model.encode(queries[qid]).tolist()
                t0 = time.perf_counter()
                results = store.search(
                    query_embedding=qemb,
                    query_text=queries[qid],
                    top_k=TOP_K,
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                ranked = [doc_id_map.get(r.path, "") for r in results]
                ndcgs.append(ndcg_at_k(ranked, qrels[qid], TOP_K))

            mean_ndcg = statistics.mean(ndcgs)
            med_lat = statistics.median(latencies)
            print(f"    NDCG@10={mean_ndcg:.4f}  latency={med_lat:.1f}ms")

            all_results[ds_name][label] = {
                "ndcg_10": round(mean_ndcg, 4),
                "latency_ms": round(med_lat, 1),
            }

            store.close()
            Path(db_path).unlink(missing_ok=True)

    # Summary table
    print(f"\n{'=' * 80}")
    print("  FULL PIPELINE RESULTS (adaptive RRF + FTS5 + MMR dedup)")
    print(f"{'=' * 80}")
    print(
        f"  {'Dataset':>12} {'BM25':>8} {'ColBERTv2':>10} {'Base':>10} {'Tuned':>10} "
        f"{'Delta':>8} {'vs ColBERT':>10}"
    )
    print(f"  {'-' * 70}")

    for ds in DATASETS:
        if ds not in all_results:
            continue
        bm25 = BASELINES.get(ds, {}).get("BM25", 0)
        colbert = BASELINES.get(ds, {}).get("ColBERTv2", 0)
        base = all_results[ds].get("BGE-small (base)", {}).get("ndcg_10", 0)
        tuned = all_results[ds].get("vstash-bge-rrf (tuned)", {}).get("ndcg_10", 0)
        delta = (tuned - base) / base * 100 if base > 0 else 0
        vs_colbert = (tuned - colbert) / colbert * 100 if colbert > 0 else 0
        print(
            f"  {ds:>12} {bm25:>8.3f} {colbert:>10.3f} {base:>10.4f} {tuned:>10.4f} "
            f"{delta:>+7.1f}% {vs_colbert:>+9.1f}%"
        )

    # Save
    output_path = Path("experiments/results/beir_tuned_full_pipeline.json")
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
