"""Tests for vstash.store — VstashStore CRUD and RRF search."""

from __future__ import annotations

from tests.conftest import requires_sqlite_vec
from vstash.models import DocumentInfo, SearchResult, StoreStats
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


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
        assert len(doc_id) == 32

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


class TestStoreDeduplication:
    """Test document-level deduplication in search results."""

    def test_search_deduplicates_by_document(self, sample_store: VstashStore) -> None:
        """Multiple chunks from the same document should not flood top-k."""
        dim = sample_store.embedding_dim
        # Add one document with many chunks that all match the query
        sample_store.add_document(
            path="/test/big_doc.md",
            title="Big Document",
            chunks=[f"machine learning topic {i}" for i in range(5)],
            embeddings=[[0.1 + i * 0.01] * dim for i in range(5)],
        )
        # Add another document with a single relevant chunk
        sample_store.add_document(
            path="/test/small_doc.md",
            title="Small Document",
            chunks=["machine learning basics"],
            embeddings=[[0.12] * dim],
        )
        results = sample_store.search([0.11] * dim, "machine learning", top_k=5)
        titles = [r.title for r in results]
        # Each document should appear at most once
        assert titles.count("Big Document") <= 1
        assert titles.count("Small Document") <= 1
        # Both documents should be represented
        assert "Big Document" in titles
        assert "Small Document" in titles


class TestExpandContext:
    """Test context expansion with adjacent chunks."""

    def test_expand_context_includes_neighbors(self, sample_store: VstashStore) -> None:
        """Expanding a middle chunk should include previous and next chunks."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/multi.md",
            title="Multi Chunk Doc",
            chunks=["chunk zero", "chunk one", "chunk two", "chunk three"],
            embeddings=[[0.1 + i * 0.01] * dim for i in range(4)],
        )
        # Search to get a result for the middle chunk
        results = sample_store.search([0.11] * dim, "chunk one", top_k=1)
        assert len(results) == 1
        assert results[0].title == "Multi Chunk Doc"

        expanded = sample_store.expand_context(results, window=1)
        assert len(expanded) == 1
        # The expanded text should contain adjacent chunk text
        assert "chunk" in expanded[0].text
        # Should have more text than original single chunk
        assert len(expanded[0].text) >= len(results[0].text)

    def test_expand_context_window_zero_returns_unchanged(self, sample_store: VstashStore) -> None:
        """Window=0 should return results unchanged."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/doc.md",
            title="Test Doc",
            chunks=["hello world"],
            embeddings=[[0.1] * dim],
        )
        results = sample_store.search([0.1] * dim, "hello", top_k=1)
        expanded = sample_store.expand_context(results, window=0)
        assert expanded == results

    def test_expand_context_empty_results(self, sample_store: VstashStore) -> None:
        """Empty results should return empty."""
        expanded = sample_store.expand_context([], window=1)
        assert expanded == []


class TestTotalAccessCount:
    """Test total access count aggregation."""

    def test_total_access_count_initial(self, sample_store: VstashStore) -> None:
        """Fresh store should have zero total accesses."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/doc.md",
            title="Doc",
            chunks=["some text"],
            embeddings=[[0.1] * dim],
        )
        assert sample_store.total_access_count() == 0

    def test_total_access_count_after_tracking(self, sample_store: VstashStore) -> None:
        """Total should reflect tracked accesses."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/doc.md",
            title="Doc",
            chunks=["some text", "more text"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        # Get chunk ids
        rows = sample_store._conn.execute("SELECT id FROM chunks").fetchall()
        chunk_ids = [r["id"] for r in rows]
        # Track 3 times
        for _ in range(3):
            sample_store.track_access(chunk_ids)
        # 2 chunks × 3 accesses = 6
        assert sample_store.total_access_count() == 6


class TestAdaptiveRelevanceThreshold:
    """Test adaptive relevance threshold based on spread history."""

    def test_fallback_with_no_history(self, sample_store: VstashStore) -> None:
        """Should return fallback when no spreads recorded."""
        assert sample_store.adaptive_relevance_threshold(fallback=0.15) == 0.15

    def test_fallback_with_few_samples(self, sample_store: VstashStore) -> None:
        """Should return fallback when fewer than 10 spreads."""
        for i in range(5):
            sample_store.record_spread(0.3)
        assert sample_store.adaptive_relevance_threshold(fallback=0.15) == 0.15

    def test_adaptive_with_enough_history(self, sample_store: VstashStore) -> None:
        """With 10+ uniform spreads, threshold should be near mean - 1σ."""
        for _ in range(15):
            sample_store.record_spread(0.30)
        threshold = sample_store.adaptive_relevance_threshold()
        # All same value → σ=0, threshold = mean = 0.30
        assert abs(threshold - 0.30) < 0.01

    def test_adaptive_responds_to_variance(self, sample_store: VstashStore) -> None:
        """Higher variance should lower the threshold (more lenient)."""
        # Mix of high and low spreads
        for v in [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]:
            sample_store.record_spread(v)
        threshold = sample_store.adaptive_relevance_threshold()
        # mean=0.3, σ=0.2, threshold=0.3-0.2=0.1
        assert threshold < 0.15  # more lenient than fixed 0.15

    def test_ring_buffer_prunes_old(self, sample_store: VstashStore) -> None:
        """Only last 50 entries should be kept."""
        for i in range(60):
            sample_store.record_spread(float(i) / 100)
        count = sample_store._conn.execute(
            "SELECT COUNT(*) AS n FROM search_stats"
        ).fetchone()["n"]
        assert count == 50


class TestStoreCollections:
    """Test collection-scoped operations."""

    def test_add_to_named_collection(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/file.md",
            title="Test",
            chunks=["hello"],
            embeddings=[[0.1] * dim],
            collection="work",
        )
        docs = sample_store.list_documents(collection="work")
        assert len(docs) == 1
        assert docs[0].collection == "work"
        assert docs[0].title == "Test"

    def test_list_collections(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["a"],
            embeddings=[[0.1] * dim],
            collection="alpha",
        )
        sample_store.add_document(
            path="/b.md",
            title="B",
            chunks=["b"],
            embeddings=[[0.2] * dim],
            collection="beta",
        )
        sample_store.add_document(
            path="/c.md",
            title="C",
            chunks=["c"],
            embeddings=[[0.3] * dim],
            collection="alpha",
        )
        cols = sample_store.list_collections()
        assert set(cols) == {"alpha", "beta"}

    def test_list_filtered_by_collection(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["a"],
            embeddings=[[0.1] * dim],
            collection="proj1",
        )
        sample_store.add_document(
            path="/b.md",
            title="B",
            chunks=["b"],
            embeddings=[[0.2] * dim],
            collection="proj2",
        )
        docs_p1 = sample_store.list_documents(collection="proj1")
        assert len(docs_p1) == 1
        assert docs_p1[0].title == "A"

        docs_all = sample_store.list_documents()
        assert len(docs_all) == 2

    def test_search_scoped_by_collection(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/work/a.md",
            title="Work Doc",
            chunks=["work content"],
            embeddings=[[0.1] * dim],
            collection="work",
        )
        sample_store.add_document(
            path="/personal/b.md",
            title="Personal Doc",
            chunks=["personal content"],
            embeddings=[[0.2] * dim],
            collection="personal",
        )
        query_vec = [0.1] * dim
        # Search only within "work"
        results = sample_store.search(query_vec, "content", collection="work")
        paths_in_results = {r.path for r in results}
        assert "/work/a.md" in paths_in_results or len(results) == 0
        # Ensure personal docs are NOT in work-scoped results
        assert "/personal/b.md" not in paths_in_results

    def test_find_document_scoped_by_collection(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/docs/readme.md",
            title="Readme",
            chunks=["text"],
            embeddings=[[0.1] * dim],
            collection="docs",
        )
        sample_store.add_document(
            path="/notes/readme.md",
            title="Notes Readme",
            chunks=["notes"],
            embeddings=[[0.2] * dim],
            collection="notes",
        )
        match = sample_store.find_document("readme", collection="docs")
        assert match is not None
        assert "docs" in match

    def test_stats_includes_collection_count(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["a"],
            embeddings=[[0.1] * dim],
            collection="x",
        )
        sample_store.add_document(
            path="/b.md",
            title="B",
            chunks=["b"],
            embeddings=[[0.2] * dim],
            collection="y",
        )
        s = sample_store.stats()
        assert s.collections == 2

    def test_same_path_different_collections(self, sample_store: VstashStore) -> None:
        """Same file can exist in multiple collections (different doc_id hash)."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/shared/file.md",
            title="Shared",
            chunks=["shared"],
            embeddings=[[0.1] * dim],
            collection="team-a",
        )
        sample_store.add_document(
            path="/shared/file.md",
            title="Shared",
            chunks=["shared"],
            embeddings=[[0.1] * dim],
            collection="team-b",
        )
        docs = sample_store.list_documents()
        assert len(docs) == 2
        cols = {d.collection for d in docs}
        assert cols == {"team-a", "team-b"}

    def test_default_collection(self, sample_store: VstashStore) -> None:
        """Documents without explicit collection go to 'default'."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/x.md",
            title="X",
            chunks=["x"],
            embeddings=[[0.1] * dim],
        )
        docs = sample_store.list_documents()
        assert len(docs) == 1
        assert docs[0].collection == "default"


class TestSearchTelemetry:
    """Tests for search event telemetry (discard tracking)."""

    def test_record_search_event(self, sample_store: VstashStore) -> None:
        event_id = sample_store.record_search_event(
            query="test", best_distance=0.5, relevance_tier="high", result_count=5,
        )
        assert event_id > 0

    def test_mark_search_dismissed(self, sample_store: VstashStore) -> None:
        event_id = sample_store.record_search_event(
            query="test", best_distance=0.99, relevance_tier="low", result_count=3,
        )
        sample_store.mark_search_dismissed(event_id)
        row = sample_store._conn.execute(
            "SELECT dismissed FROM search_events WHERE id = ?", [event_id]
        ).fetchone()
        assert row["dismissed"] == 1

    def test_telemetry_summary_groups_by_tier(self, sample_store: VstashStore) -> None:
        # Record events across tiers
        sample_store.record_search_event("q1", 0.5, "high", 5)
        sample_store.record_search_event("q2", 0.6, "high", 5)
        eid = sample_store.record_search_event("q3", 0.99, "low", 3)
        sample_store.mark_search_dismissed(eid)

        summary = sample_store.search_telemetry_summary()
        assert summary["high"]["total"] == 2
        assert summary["high"]["dismissed"] == 0
        assert summary["low"]["total"] == 1
        assert summary["low"]["dismissed"] == 1

    def test_telemetry_empty(self, sample_store: VstashStore) -> None:
        summary = sample_store.search_telemetry_summary()
        assert summary == {}
