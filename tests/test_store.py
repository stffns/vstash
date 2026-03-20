"""Tests for vstash.store — VstashStore CRUD and RRF search."""

from __future__ import annotations

import sqlite3

import pytest

from vstash.models import DocumentInfo, SearchResult, StoreStats
from vstash.store import VstashStore


def _sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension loading is available."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("SELECT vec_version()")
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason="sqlite-vec extension or enable_load_extension not available",
)


class TestStoreContextManager:
    """Test context manager protocol."""

    def test_context_manager_opens_and_closes(self, tmp_db_path: str) -> None:
        with VstashStore(tmp_db_path, embedding_dim=384) as store:
            assert store is not None
            s = store.stats()
            assert s.documents == 0

    def test_context_manager_returns_self(self, tmp_db_path: str) -> None:
        store = VstashStore(tmp_db_path, embedding_dim=384)
        with store as s:
            assert s is store


class TestStoreCRUD:
    """Test document CRUD operations."""

    def test_add_document(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        doc_id = sample_store.add_document(
            path="/test/file.md",
            title="Test File",
            chunks=["chunk one", "chunk two"],
            embeddings=[[0.1] * dim, [0.2] * dim],
            source_type="markdown",
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) == 16

    def test_doc_exists(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/existing.md",
            title="Existing",
            chunks=["text"],
            embeddings=[[0.1] * dim],
        )
        assert sample_store.doc_exists("/test/existing.md") is True
        assert sample_store.doc_exists("/test/nonexistent.md") is False

    def test_list_documents(self, populated_store: VstashStore) -> None:
        docs = populated_store.list_documents()
        assert len(docs) == 2
        assert all(isinstance(d, DocumentInfo) for d in docs)
        titles = {d.title for d in docs}
        assert "Python Guide" in titles
        assert "ML Introduction" in titles

    def test_stats(self, populated_store: VstashStore) -> None:
        s = populated_store.stats()
        assert isinstance(s, StoreStats)
        assert s.documents == 2
        assert s.chunks == 5  # 2 + 3
        assert s.db_size_mb >= 0

    def test_delete_document(self, populated_store: VstashStore) -> None:
        deleted = populated_store.delete_document("/test/python_guide.md")
        assert deleted is True
        docs = populated_store.list_documents()
        assert len(docs) == 1
        assert docs[0].title == "ML Introduction"

    def test_delete_nonexistent_returns_false(self, sample_store: VstashStore) -> None:
        deleted = sample_store.delete_document("/nonexistent/file.md")
        assert deleted is False

    def test_re_ingest_replaces_document(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        # First ingestion
        sample_store.add_document(
            path="/test/doc.md",
            title="V1",
            chunks=["version one"],
            embeddings=[[0.1] * dim],
        )
        # Second ingestion (same path)
        sample_store.add_document(
            path="/test/doc.md",
            title="V2",
            chunks=["version two", "extra chunk"],
            embeddings=[[0.2] * dim, [0.3] * dim],
        )
        docs = sample_store.list_documents()
        assert len(docs) == 1
        assert docs[0].title == "V2"
        assert docs[0].chunk_count == 2


class TestStoreSearch:
    """Test hybrid search and RRF ranking."""

    def test_search_returns_search_results(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        query_vec = [0.1] * dim
        results = populated_store.search(query_vec, "Python programming")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_respects_top_k(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        query_vec = [0.1] * dim
        results = populated_store.search(query_vec, "Python", top_k=2)
        assert len(results) <= 2

    def test_search_returns_scores(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        query_vec = [0.3] * dim
        results = populated_store.search(query_vec, "machine learning")
        if results:
            assert all(r.score > 0 for r in results)

    def test_empty_store_search(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        query_vec = [0.1] * dim
        results = sample_store.search(query_vec, "anything")
        assert results == []

    def test_search_result_has_all_fields(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        query_vec = [0.1] * dim
        results = populated_store.search(query_vec, "Python")
        if results:
            r = results[0]
            assert r.text
            assert r.title
            assert r.path
            assert isinstance(r.chunk, int)
            assert isinstance(r.score, float)
