"""Retrain pipeline sanity check against snapvec 0.7.1.

Builds a small vstash store from the SciFact corpus, runs
`generate_triples` with a handful of pseudo-queries, and reports
how many disagreement pairs came out. Skips the MNRL training step
(that takes minutes + GPU and is covered by the published model).

What this validates:
  - VstashStore ingest still works after the snapvec 0.7.1 bump
  - hybrid search (vector + FTS + RRF) still returns sensible results
  - retrain's disagreement detector still finds pairs

Usage:
    python -m experiments.retrain_smoke
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from experiments.beir_benchmark import download_beir, load_beir
from vstash.embed import embed_texts
from vstash.retrain import generate_triples
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
N_DOCS = 500
MAX_QUERIES = 50


def main() -> None:
    cache = download_beir("scifact")
    corpus, _, _ = load_beir(cache)
    doc_ids = list(corpus.keys())[:N_DOCS]
    texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    print(f"embedding {len(texts)} SciFact docs ...")
    t0 = time.perf_counter()
    batch = 256
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        embeddings.extend(embed_texts(texts[start:end], MODEL))
    print(f"  embedded in {time.perf_counter() - t0:.1f}s")

    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "smoke.db")
        store = VstashStore(db_path, embedding_dim=384, vector_backend="sqlite-vec")

        print("ingesting ...")
        t0 = time.perf_counter()
        for i, (doc_id, text, emb) in enumerate(zip(doc_ids, texts, embeddings)):
            store.add_document(
                path=f"scifact/{doc_id}",
                title=doc_id,
                chunks=[text],
                embeddings=[emb],
                source_type="text",
            )
            if i % 100 == 99:
                print(f"  {i + 1}/{len(texts)}")
        print(f"  ingested in {time.perf_counter() - t0:.1f}s")

        stats = store.stats()
        print(f"  store: {stats.documents} docs, {stats.chunks} chunks")

        print(f"generating triples (max_queries={MAX_QUERIES}) ...")
        t0 = time.perf_counter()
        pairs = generate_triples(store, MODEL, max_queries=MAX_QUERIES)
        print(f"  generated {len(pairs)} pairs in {time.perf_counter() - t0:.1f}s")

        if pairs:
            print("\nsample pair:")
            p = pairs[0]
            print(f"  query:    {p['query'][:80]} ...")
            print(f"  positive: {p['positive'][:80]} ...")


if __name__ == "__main__":
    main()
