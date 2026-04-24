"""Migration parity probe: v1 -> v2 must preserve retrieval ordering.

The cosine migration only swaps how sqlite-vec computes distance; the
stored float bytes are identical.  For unit-normalized embeddings (BGE)
L2 and cosine are monotonic, so ranking must not change.  This script:

1. Creates a v2 store with real BGE-small embeddings over a synthetic
   corpus.
2. Runs a batch of queries, snapshots (id, path) of the top-10 results
   and the exact score for each.
3. Forces the store back to a v1 (L2) table with the same byte content.
4. Reopens it with the v2 build, which runs the migration.
5. Re-runs the same queries and compares top-10 order.

If this probe fails, the migration changes user-visible retrieval
behavior on the baseline model and the paper numbers are at risk.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import sqlite_vec

from vstash.embed import embed_query, embed_texts, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

CORPUS = [
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "A mitochondrion is the powerhouse of the cell.",
    "The Pythagorean theorem relates sides of a right triangle.",
    "Quantum entanglement describes non-local particle correlations.",
    "Neural networks learn representations from labeled data.",
    "SQLite is an embedded transactional relational database.",
    "The French Revolution began in 1789.",
    "Newton's third law: every action has an equal and opposite reaction.",
    "Ribosomes synthesize proteins from mRNA templates.",
    "Rust's borrow checker enforces memory safety at compile time.",
    "A Turing machine is an abstract model of computation.",
    "The Krebs cycle produces ATP in aerobic respiration.",
    "Graph neural networks operate on non-Euclidean data.",
    "The speed of light is approximately 3e8 m/s in vacuum.",
    "Transformers use self-attention over sequence elements.",
    "DNA is a double helix with complementary base pairs.",
    "A hash function maps arbitrary inputs to fixed-size outputs.",
    "Black holes warp spacetime according to general relativity.",
    "The CAP theorem bounds distributed system guarantees.",
    "Mitosis produces two genetically identical daughter cells.",
]

QUERIES = [
    "how does a cell produce energy",
    "what is the powerhouse of the cell",
    "explain quantum mechanics",
    "protein synthesis in biology",
    "what are transformers in machine learning",
    "describe compile-time memory safety",
    "cellular respiration pathway",
]


def _downgrade_to_v1(db_path: Path, dim: int) -> None:
    """Rewrite the v2 cosine vec_chunks as a v1 L2 vec_chunks in place."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("UPDATE store_meta SET value='1' WHERE key='schema_version'")
    rows = conn.execute("SELECT rowid, embedding FROM vec_chunks").fetchall()
    conn.execute("DROP TABLE vec_chunks")
    conn.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dim}])")
    conn.executemany("INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _snapshot_queries(store: VstashStore, dim: int) -> list[list[tuple[int, str, float]]]:
    """Return top-10 (chunk_id, path, score) for each query."""
    snaps: list[list[tuple[int, str, float]]] = []
    for q in QUERIES:
        q_emb = embed_query(q, MODEL)
        results = store.search(
            query_embedding=q_emb,
            query_text=q,
            top_k=10,
        )
        snaps.append([(r.chunk_id, r.path, round(r.score, 8)) for r in results])
    return snaps


def _compare(
    pre: list[list[tuple[int, str, float]]],
    post: list[list[tuple[int, str, float]]],
) -> tuple[int, int]:
    """Return (identical_queries, total_queries)."""
    identical = 0
    for i, (a, b) in enumerate(zip(pre, post)):
        # Compare ids only; score may differ numerically since RRF with
        # cosine distances vs L2 distances over non-identical vec_weight
        # pools produces slightly different absolute scores.  The
        # RANKING is what must hold.
        a_ids = [row[0] for row in a]
        b_ids = [row[0] for row in b]
        if a_ids == b_ids:
            identical += 1
        else:
            print(f"  Query {i}: diff")
            print(f"    pre:  {a_ids}")
            print(f"    post: {b_ids}")
    return identical, len(pre)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "parity.db"
        dim = get_embedding_dim(MODEL)

        print(f"Dim: {dim}, Corpus: {len(CORPUS)}, Queries: {len(QUERIES)}")
        print("Step 1: build a v2 store and snapshot queries")

        store = VstashStore(str(db_path), embedding_dim=dim)
        embs = embed_texts(CORPUS, MODEL)
        store.add_document(
            path="/probe.md",
            title="probe",
            chunks=CORPUS,
            embeddings=[list(e) for e in embs],
            source_type="markdown",
        )
        pre = _snapshot_queries(store, dim)
        assert store.get_meta("schema_version") == "2"
        store.close()

        print("Step 2: downgrade the DB back to v1 (L2) on disk")
        _downgrade_to_v1(db_path, dim)

        print("Step 3: reopen -- triggers v1->v2 migration")
        store2 = VstashStore(str(db_path), embedding_dim=dim)
        assert store2.get_meta("schema_version") == "2"
        row = store2._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert "distance_metric=cosine" in (row["sql"] or "").lower()

        print("Step 4: re-run same queries and compare")
        post = _snapshot_queries(store2, dim)
        store2.close()

        identical, total = _compare(pre, post)
        print(f"\nResult: {identical}/{total} queries have identical top-10 ordering")
        if identical == total:
            print("PASS: migration preserves retrieval ordering.")
            return 0
        else:
            print("FAIL: migration changes retrieval ordering on some queries.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
