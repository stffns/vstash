"""Reindex roundtrip: verify reindex() creates cosine-typed vec_chunks."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from vstash.embed import embed_query, embed_texts, get_embedding_dim
from vstash.store import VstashStore

SRC_MODEL = "BAAI/bge-small-en-v1.5"
DST_MODEL = "BAAI/bge-small-en-v1.5"  # same model, but reindex still rebuilds

CORPUS = [
    "SQLite is embedded.",
    "Transformers use attention.",
    "Ribosomes translate mRNA.",
]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "reindex.db"
        src_dim = get_embedding_dim(SRC_MODEL)

        store = VstashStore(str(db_path), embedding_dim=src_dim)
        embs = embed_texts(CORPUS, SRC_MODEL)
        store.add_document(
            path="/r.md",
            title="r",
            chunks=CORPUS,
            embeddings=[list(e) for e in embs],
            source_type="markdown",
        )
        # sanity: pre-reindex DDL
        row = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert "distance_metric=cosine" in (row["sql"] or "").lower(), (
            f"fresh v2 DB missing cosine DDL: {row['sql']!r}"
        )

        dst_dim = get_embedding_dim(DST_MODEL)
        processed = store.reindex(
            embed_fn=lambda texts: [list(e) for e in embed_texts(texts, DST_MODEL)],
            new_dim=dst_dim,
        )
        print(f"Reindexed {processed} chunks.")

        row = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert "distance_metric=cosine" in (row["sql"] or "").lower(), (
            f"post-reindex DDL lost cosine metric: {row['sql']!r}"
        )
        print(f"Post-reindex DDL: {row['sql']}")

        # Verify search still works end-to-end.
        q_emb = embed_query("relational database storage", DST_MODEL)
        results = store.search(
            query_embedding=q_emb, query_text="relational database storage", top_k=3
        )
        print(f"Search returned {len(results)} hits, best text={results[0].text!r}")
        assert results, "search returned no results after reindex"
        assert "SQLite" in results[0].text, (
            f"expected best hit to be the SQLite chunk, got {results[0].text!r}"
        )

        store.close()
        print("PASS: reindex preserves cosine metric and search works.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
