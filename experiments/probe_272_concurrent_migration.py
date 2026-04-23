"""Concurrent migration probe: two processes racing on the same v1 DB.

Asserts that the ``BEGIN IMMEDIATE`` in ``_migrate_v1_to_v2`` serializes
two concurrent migrations: one wins, the other either waits and sees
the already-migrated state, or cleanly observes it as v2 via the
sqlite_master guard.  No half-migrated state, no data loss.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import sqlite_vec

from vstash.embed import embed_texts, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"
CORPUS = [f"chunk {i}: the quick brown fox jumps over lazy dog #{i}." for i in range(50)]


def _downgrade(db_path: str, dim: int) -> None:
    conn = sqlite3.connect(db_path)
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


def _open_worker(db_path: str, dim: int, result_q: "mp.Queue[tuple[str, str]]") -> None:
    """Open the store -- blocks on BEGIN IMMEDIATE if another process is mid-migration."""
    try:
        store = VstashStore(db_path, embedding_dim=dim)
        version = store.get_meta("schema_version")
        count = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        store.close()
        result_q.put(("ok", f"version={version}, vec_chunks_count={count}"))
    except Exception as e:
        result_q.put(("error", f"{type(e).__name__}: {e}"))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "concurrent.db"
        dim = get_embedding_dim(MODEL)

        print("Setup: v2 store with 50 chunks")
        store = VstashStore(str(db_path), embedding_dim=dim)
        embs = embed_texts(CORPUS, MODEL)
        store.add_document(
            path="/c.md",
            title="c",
            chunks=CORPUS,
            embeddings=[list(e) for e in embs],
            source_type="markdown",
        )
        store.close()

        print("Downgrade to v1 (L2) on disk")
        _downgrade(str(db_path), dim)

        print("Launching 4 worker processes that all want to migrate the same DB")
        ctx = mp.get_context("spawn")
        result_q: mp.Queue = ctx.Queue()
        workers = [
            ctx.Process(target=_open_worker, args=(str(db_path), dim, result_q))
            for _ in range(4)
        ]
        t0 = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=60)
        dt = time.perf_counter() - t0
        print(f"All workers joined in {dt:.2f}s")

        results = []
        while not result_q.empty():
            results.append(result_q.get())
        print(f"\nWorker outcomes ({len(results)}):")
        for status, detail in results:
            print(f"  {status}: {detail}")

        # All workers must succeed.
        assert len(results) == 4, f"expected 4 results, got {len(results)}"
        assert all(s == "ok" for s, _ in results), (
            f"some workers failed: {[r for r in results if r[0] != 'ok']}"
        )
        # All must see version=2 and the full 50 rows.
        for _, detail in results:
            assert "version=2" in detail and "vec_chunks_count=50" in detail, detail

        # Final DB state: still 50 rows, schema v2, DDL cosine.
        conn = sqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()["sql"]
        count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        version = conn.execute(
            "SELECT value FROM store_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        conn.close()
        print(f"\nFinal DB: version={version}, rows={count}")
        print(f"DDL: {ddl}")
        assert "distance_metric=cosine" in (ddl or "").lower()
        assert count == 50
        assert version == "2"
        print("PASS: concurrent migration serialized cleanly, no data loss.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
