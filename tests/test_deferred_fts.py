"""Tests for deferred FTS indexing in batch_mode."""

from __future__ import annotations

import pytest

from vstash.store import VstashStore

from .conftest import requires_sqlite_vec


@requires_sqlite_vec
class TestDeferredFTS:
    """Deferred FTS indexing: correctness, flush, and edge cases."""

    @pytest.fixture
    def store(self, tmp_db_path: str) -> VstashStore:
        dim = 384
        s = VstashStore(tmp_db_path, embedding_dim=dim)
        yield s
        s.close()

    def _add_doc(self, store: VstashStore, path: str, text: str) -> str:
        dim = store.embedding_dim
        return store.add_document(
            path=path,
            title=path,
            chunks=[text],
            embeddings=[[0.1] * dim],
        )

    def _fts_has_terms(self, store: VstashStore) -> bool:
        """Check if FTS5 index has any indexed terms."""
        return store._conn.execute("SELECT COUNT(*) FROM fts_chunks_vocab").fetchone()[0] > 0

    def _fts_match(self, store: VstashStore, term: str) -> int:
        """Count FTS MATCH results for a term."""
        safe = '"' + term.replace('"', '""') + '"'
        try:
            return len(
                store._conn.execute(
                    "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?",
                    [safe],
                ).fetchall()
            )
        except Exception:
            return 0

    def test_defer_fts_skips_inserts_during_batch(self, store: VstashStore):
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/a.md", "alpha bravo charlie")
            assert not self._fts_has_terms(store)
            assert len(store._deferred_fts_rows) == 1

    def test_defer_fts_flushes_on_exit(self, store: VstashStore):
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/a.md", "alpha bravo charlie")
            self._add_doc(store, "/b.md", "delta echo foxtrot")
        assert self._fts_has_terms(store)
        assert len(store._deferred_fts_rows) == 0

    def test_fts_search_works_after_flush(self, store: VstashStore):
        dim = store.embedding_dim
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/a.md", "quantum entanglement physics")
            self._add_doc(store, "/b.md", "classical mechanics newton")

        results = store.search([0.1] * dim, "quantum", top_k=5)
        texts = [r.text for r in results]
        assert any("quantum" in t for t in texts)

    def test_no_defer_still_inserts_immediately(self, store: VstashStore):
        with store.batch_mode(defer_fts=False):
            self._add_doc(store, "/a.md", "immediate fts insert")
            assert self._fts_has_terms(store)

    def test_default_batch_mode_no_defer(self, store: VstashStore):
        with store.batch_mode():
            self._add_doc(store, "/a.md", "immediate fts insert")
            assert self._fts_has_terms(store)

    def test_add_documents_batch_with_defer(self, store: VstashStore):
        dim = store.embedding_dim
        docs = [
            {
                "path": f"/doc{i}.md",
                "title": f"Doc {i}",
                "chunks": [f"content for document {i} xyzzy"],
                "embeddings": [[0.1 * (i + 1)] * dim],
                "source_type": "markdown",
            }
            for i in range(5)
        ]
        with store.batch_mode(defer_fts=True):
            store.add_documents_batch(docs)
            assert not self._fts_has_terms(store)
        assert self._fts_match(store, "xyzzy") == 5

    def test_mixed_add_and_batch_with_defer(self, store: VstashStore):
        dim = store.embedding_dim
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/single.md", "single doc add foobar")
            store.add_documents_batch(
                [
                    {
                        "path": "/batch1.md",
                        "title": "Batch 1",
                        "chunks": ["batch content one foobar"],
                        "embeddings": [[0.2] * dim],
                        "source_type": "markdown",
                    },
                    {
                        "path": "/batch2.md",
                        "title": "Batch 2",
                        "chunks": ["batch content two foobar"],
                        "embeddings": [[0.3] * dim],
                        "source_type": "markdown",
                    },
                ]
            )
            assert not self._fts_has_terms(store)
        assert self._fts_match(store, "foobar") == 3

    def test_nested_batch_flushes_at_outermost(self, store: VstashStore):
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/outer.md", "outer document")
            with store.batch_mode(defer_fts=True):
                self._add_doc(store, "/inner.md", "inner document")
            assert not self._fts_has_terms(store)
        assert self._fts_has_terms(store)

    def test_large_batch_deferred(self, store: VstashStore):
        n = 50
        with store.batch_mode(defer_fts=True):
            for i in range(n):
                self._add_doc(store, f"/doc{i}.md", f"document number {i} zztoken")
        assert self._fts_match(store, "zztoken") == n

    def test_error_during_batch_still_flushes(self, store: VstashStore):
        try:
            with store.batch_mode(defer_fts=True):
                self._add_doc(store, "/ok.md", "this is fine xqtoken")
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert self._fts_match(store, "xqtoken") == 1
        assert len(store._deferred_fts_rows) == 0

    def test_fts_not_searchable_during_defer(self, store: VstashStore):
        with store.batch_mode(defer_fts=True):
            self._add_doc(store, "/a.md", "uniquetoken12345")
            assert self._fts_match(store, "uniquetoken12345") == 0
        assert self._fts_match(store, "uniquetoken12345") == 1
