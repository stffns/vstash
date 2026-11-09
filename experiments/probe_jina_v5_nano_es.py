"""Spanish retrieval: jina-embeddings-v5-text-nano-retrieval vs current options.

Motivation: vstash's default (``BAAI/bge-small-en-v1.5``) is English-only, and
a Spanish memory (KakesuI) was running on it. This probe compares three
encoders on MTEB ``SpanishPassageRetrievalS2P`` (10 037 passages, 167 queries,
binary qrels):

* ``BAAI/bge-small-en-v1.5`` -- the default (384d, English-only);
* ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` -- the
  multilingual model the config docs recommend (384d);
* ``jinaai/jina-embeddings-v5-text-nano-retrieval`` -- the new built-in
  backend (768d, asymmetric ``Query:`` / ``Document:`` prompts).

Two modes per model:

* **dense** -- brute-force cosine over the passage embeddings, so the
  encoder is measured on its own;
* **hybrid** -- vstash's real pipeline (vector + FTS5 + adaptive RRF + MMR)
  on a fresh store, which is what a user gets.

Metrics: NDCG@10 and Recall@100 (dense) / Recall@top_k (hybrid), plus
corpus embedding throughput and warm query latency.

Run:
    python -m experiments.probe_jina_v5_nano_es [--models ...] [--limit-corpus N]

Results are written to ``experiments/results/jina_v5_nano_es.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
    "embeddinggemma-300m",
]
DATASET = "mteb/SpanishPassageRetrievalS2P"
OUT = Path(__file__).parent / "results" / "jina_v5_nano_es.json"
HYBRID_TOP_K = 10


def _load() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def table(part: str):
        path = hf_hub_download(DATASET, f"{part}/test-00000-of-00001.parquet", repo_type="dataset")
        return pq.read_table(path).to_pylist()

    corpus = {
        r["_id"]: (f"{r['title']}\n{r['text']}" if r.get("title") else r["text"])
        for r in table("corpus")
    }
    queries = {r["_id"]: r["text"] for r in table("queries")}
    qrels: dict[str, set[str]] = {}
    for r in table("qrels"):
        if float(r["score"]) > 0:
            qrels.setdefault(r["query-id"], set()).add(r["corpus-id"])
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def _ndcg_at_10(ranked: list[str], relevant: set[str]) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    return dcg / ideal if ideal else 0.0


def _recall(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _run_model(model: str, corpus, queries, qrels) -> dict:
    from vstash.config import VstashConfig
    from vstash.embed import embed_query, embed_texts, get_embedding_dim
    from vstash.store import VstashStore

    ids = list(corpus)
    texts = [corpus[i] for i in ids]

    embed_texts(texts[:2], model)  # load + warm
    t0 = time.perf_counter()
    doc_vecs = np.asarray(embed_texts(texts, model), dtype=np.float32)
    embed_s = time.perf_counter() - t0

    q_ids = list(queries)
    q_lat: list[float] = []
    q_vecs = []
    for qid in q_ids:
        t = time.perf_counter()
        q_vecs.append(embed_query(queries[qid], model))
        q_lat.append((time.perf_counter() - t) * 1000.0)
    q_mat = np.asarray(q_vecs, dtype=np.float32)

    # Dense: brute-force cosine (all vectors are unit-normalised).
    sims = q_mat @ doc_vecs.T
    dense_ndcg, dense_r100 = [], []
    for row, qid in zip(sims, q_ids, strict=True):
        ranked = [ids[j] for j in np.argsort(-row)[:100]]
        dense_ndcg.append(_ndcg_at_10(ranked, qrels[qid]))
        dense_r100.append(_recall(ranked, qrels[qid], 100))

    # Hybrid: the real vstash pipeline on a fresh store.
    cfg = VstashConfig()
    hyb_ndcg, hyb_recall = [], []
    with tempfile.TemporaryDirectory() as tmp:
        store = VstashStore(
            os.path.join(tmp, "es.db"),
            embedding_dim=get_embedding_dim(model),
            cache=cfg.cache,
        )
        try:
            with store.batch_mode(defer_fts=True):
                for doc_id, text, vec in zip(ids, texts, doc_vecs.tolist(), strict=True):
                    store.add_document(
                        path=doc_id,
                        title=doc_id,
                        chunks=[text],
                        embeddings=[vec],
                        source_type="text",
                    )
            for qid, qv in zip(q_ids, q_mat, strict=True):
                results = store.search(qv, queries[qid], top_k=HYBRID_TOP_K)
                ranked = [r.path for r in results]
                hyb_ndcg.append(_ndcg_at_10(ranked, qrels[qid]))
                hyb_recall.append(_recall(ranked, qrels[qid], HYBRID_TOP_K))
        finally:
            store.close()

    return {
        "model": model,
        "dim": int(doc_vecs.shape[1]),
        "dense_ndcg@10": round(statistics.mean(dense_ndcg), 4),
        "dense_recall@100": round(statistics.mean(dense_r100), 4),
        "hybrid_ndcg@10": round(statistics.mean(hyb_ndcg), 4),
        f"hybrid_recall@{HYBRID_TOP_K}": round(statistics.mean(hyb_recall), 4),
        "corpus_embed_s": round(embed_s, 1),
        "passages_per_s": round(len(texts) / embed_s, 1),
        "query_embed_p50_ms": round(statistics.median(q_lat), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--limit-corpus", type=int, default=None, help="debug: first N passages")
    args = ap.parse_args()

    corpus, queries, qrels = _load()
    if args.limit_corpus:
        keep = set(list(corpus)[: args.limit_corpus])
        corpus = {k: v for k, v in corpus.items() if k in keep}
    print(f"{DATASET}: {len(corpus)} passages, {len(queries)} queries")

    rows = []
    for model in args.models:
        print(f"-> {model}", flush=True)
        row = _run_model(model, corpus, queries, qrels)
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)

    if not args.limit_corpus:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        # Merge by model so a run with --models adds to earlier results.
        merged: dict[str, dict] = {}
        if OUT.exists():
            merged = {r["model"]: r for r in json.loads(OUT.read_text())["results"]}
        merged.update({r["model"]: r for r in rows})
        payload = {"dataset": DATASET, "results": list(merged.values())}
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
