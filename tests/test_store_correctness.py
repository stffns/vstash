"""Regression tests for store-level correctness fixes (PR: fix/store-correctness).

- C1: ``reindex`` wraps DROP/CREATE/INSERT in BEGIN IMMEDIATE so a mid-reindex
  failure rolls back to the original ``vec_chunks`` instead of an empty one.
- C2: ``prune_documents`` canonicalises ``before_iso`` to UTC so the lexical
  ``added_at < ?`` comparison agrees with chronological order at TZ boundaries.
- C5: ``_build_idf_cache`` only swallows ``OperationalError`` (missing vocab
  table); a real ``DatabaseError`` (corruption) now propagates.
"""

from __future__ import annotations

import sqlite3

import pytest

from vstash.store import VstashStore


class _ConnProxy:
    """Wraps a real connection, raising ``exc`` for the fts_chunks_vocab query
    and delegating everything else, so we can inject a vocab-query failure
    without monkeypatching sqlite3.Connection.execute (which is read-only)."""

    def __init__(self, real: sqlite3.Connection, exc: Exception) -> None:
        self._real = real
        self._exc = exc

    def execute(self, sql: str, *args, **kwargs):
        if "fts_chunks_vocab" in sql:
            raise self._exc
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


# --------------------------------------------------------------------------- #
# C1: reindex atomicity
# --------------------------------------------------------------------------- #


def test_reindex_rolls_back_vec_chunks_on_embed_failure(populated_store: VstashStore) -> None:
    """A failure mid-reindex must leave the original vector index intact."""
    store = populated_store
    before = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    dim_before = store.embedding_dim
    assert before > 0

    def boom(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed failed mid-reindex")

    with pytest.raises(RuntimeError, match="embed failed"):
        store.reindex(boom, new_dim=dim_before)

    after = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert after == before, "reindex must roll back the DROP/CREATE, not wipe vec_chunks"
    assert store.embedding_dim == dim_before
    # The store is still searchable afterwards.
    results = store.search(query_embedding=[0.1] * dim_before, query_text="python", top_k=3)
    assert results, "search should still work after a rolled-back reindex"


# --------------------------------------------------------------------------- #
# C2: prune_documents UTC canonicalisation
# --------------------------------------------------------------------------- #


def test_prune_before_rejects_non_iso(populated_store: VstashStore) -> None:
    with pytest.raises(ValueError, match="ISO"):
        populated_store.prune_documents(before_iso="not-a-date")


def test_prune_before_canonicalises_timezone(populated_store: VstashStore) -> None:
    """A non-UTC offset that is chronologically *after* the stored added_at must
    delete the row. Lexically the raw string sorts the wrong way, so this only
    passes once before_iso is canonicalised to the +00:00 form."""
    store = populated_store
    # Pin both docs to a known UTC instant.
    store._conn.execute("BEGIN IMMEDIATE")
    store._conn.execute("UPDATE documents SET added_at = ?", ["2026-06-08T12:00:00+00:00"])
    store._conn.commit()

    n_docs = store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert n_docs > 0

    # 07:00:01-05:00 == 12:00:01Z, one second AFTER the stored 12:00:00Z, so the
    # docs should be pruned. As a raw lexical string "07:..." < "12:..." would
    # sort before added_at and delete nothing.
    res = store.prune_documents(before_iso="2026-06-08T07:00:01-05:00")
    assert res["deleted"] == n_docs


# --------------------------------------------------------------------------- #
# C5: _build_idf_cache narrowed except
# --------------------------------------------------------------------------- #


def test_idf_cache_degrades_on_missing_vocab(populated_store: VstashStore) -> None:
    """A missing fts_chunks_vocab table (OperationalError) still degrades to no
    adaptive IDF, as before."""
    store = populated_store
    store._idf_cache = None
    store._conn = _ConnProxy(
        store._conn, sqlite3.OperationalError("no such table: fts_chunks_vocab")
    )
    idf, total = store._build_idf_cache()
    assert idf == {}
    assert total == 0


def test_idf_cache_propagates_db_corruption(populated_store: VstashStore) -> None:
    """A real DatabaseError (corruption) must propagate, not be swallowed into a
    silently-disabled adaptive ranking."""
    store = populated_store
    store._idf_cache = None
    store._conn = _ConnProxy(store._conn, sqlite3.DatabaseError("database disk image is malformed"))
    with pytest.raises(sqlite3.DatabaseError, match="malformed"):
        store._build_idf_cache()
