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


class TestMMRDedup:
    """Test intra-document MMR diversity deduplication."""

    def test_similar_chunks_same_doc_are_deduped(self, sample_store: VstashStore) -> None:
        """Near-identical chunks from the same document should be deduped to one."""
        dim = sample_store.embedding_dim
        # One document with 5 near-identical chunks (same embedding direction)
        sample_store.add_document(
            path="/test/repetitive.md",
            title="Repetitive Doc",
            chunks=[f"machine learning topic variation {i}" for i in range(5)],
            embeddings=[[0.1 + i * 0.001] * dim for i in range(5)],  # very similar
        )
        # Another document
        sample_store.add_document(
            path="/test/other.md",
            title="Other Doc",
            chunks=["machine learning basics"],
            embeddings=[[0.12] * dim],
        )
        results = sample_store.search([0.11] * dim, "machine learning", top_k=5)
        titles = [r.title for r in results]
        # Similar chunks should be deduped — at most 1 from the repetitive doc
        assert titles.count("Repetitive Doc") <= 1
        assert "Other Doc" in titles

    def test_diverse_chunks_same_doc_both_appear(self, sample_store: VstashStore) -> None:
        """Semantically different chunks from the same document should both appear."""
        dim = sample_store.embedding_dim
        # One document with two very different chapters
        # Chapter 1: about ML (embedding direction: [1, 0, 0, ...])
        # Chapter 2: about databases (embedding direction: [0, 1, 0, ...])
        emb_ml = [1.0] + [0.0] * (dim - 1)
        emb_db = [0.0] + [1.0] + [0.0] * (dim - 2)
        # Add some padding chunks between to simulate a real book
        emb_filler = [0.5] * dim
        sample_store.add_document(
            path="/test/book.pdf",
            title="Big Book",
            chunks=[
                "chapter one about machine learning and neural networks",
                "filler content about nothing relevant",
                "filler content more stuff",
                "chapter four about database indexing and SQL queries",
            ],
            embeddings=[emb_ml, emb_filler, emb_filler, emb_db],
        )
        # Search with a query that's between both topics
        query_emb = [0.7] + [0.7] + [0.0] * (dim - 2)
        results = sample_store.search(query_emb, "machine learning databases", top_k=5)
        book_results = [r for r in results if r.title == "Big Book"]
        # Both diverse chapters should appear
        assert len(book_results) == 2
        texts = [r.text for r in book_results]
        assert any("machine learning" in t for t in texts)
        assert any("database" in t for t in texts)

    def test_mmr_does_not_affect_cross_document_ranking(self, sample_store: VstashStore) -> None:
        """Chunks from different documents should never penalize each other."""
        dim = sample_store.embedding_dim
        # Three documents with identical embeddings — MMR should keep all three
        for i in range(3):
            sample_store.add_document(
                path=f"/test/doc_{i}.md",
                title=f"Doc {i}",
                chunks=["identical content about machine learning"],
                embeddings=[[0.1] * dim],
            )
        results = sample_store.search([0.1] * dim, "machine learning", top_k=5)
        titles = {r.title for r in results}
        # All three documents should appear (no cross-doc penalty)
        assert titles == {"Doc 0", "Doc 1", "Doc 2"}


class TestReindex:
    """Test re-embedding chunks with a new model."""

    def test_reindex_changes_embeddings(self, tmp_db_path: str) -> None:
        """Reindex should replace all vec_chunks with new embeddings."""
        dim = 384
        store = VstashStore(tmp_db_path, embedding_dim=dim)
        with store:
            store.add_document(
                path="/test/doc.md",
                title="Test Doc",
                chunks=["hello world", "foo bar"],
                embeddings=[[0.1] * dim, [0.2] * dim],
            )
            assert store.stats().chunks == 2

            # Reindex with a "new model" (same dim, different vectors)
            new_dim = 384
            call_count = [0]

            def fake_embed(texts: list[str]) -> list[list[float]]:
                call_count[0] += 1
                return [[0.9] * new_dim for _ in texts]

            count = store.reindex(embed_fn=fake_embed, new_dim=new_dim)
            assert count == 2
            assert call_count[0] >= 1

            # Search should still work with new embeddings
            results = store.search([0.9] * new_dim, "hello", top_k=2)
            assert len(results) >= 1

    def test_reindex_with_different_dim(self, tmp_db_path: str) -> None:
        """Reindex should handle dimension changes."""
        old_dim = 384
        store = VstashStore(tmp_db_path, embedding_dim=old_dim)
        with store:
            store.add_document(
                path="/test/doc.md",
                title="Test Doc",
                chunks=["hello world"],
                embeddings=[[0.1] * old_dim],
            )

            new_dim = 768

            def fake_embed(texts: list[str]) -> list[list[float]]:
                return [[0.5] * new_dim for _ in texts]

            count = store.reindex(embed_fn=fake_embed, new_dim=new_dim)
            assert count == 1
            assert store.embedding_dim == new_dim

            # Search with new dim should work
            results = store.search([0.5] * new_dim, "hello", top_k=1)
            assert len(results) == 1

    def test_reindex_empty_store(self, tmp_db_path: str) -> None:
        """Reindex on empty store should return 0."""
        store = VstashStore(tmp_db_path, embedding_dim=384)
        with store:
            count = store.reindex(
                embed_fn=lambda texts: [[0.1] * 384 for _ in texts],
                new_dim=384,
            )
            assert count == 0

    def test_reindex_progress_callback(self, tmp_db_path: str) -> None:
        """Progress callback should be called during reindex."""
        dim = 384
        store = VstashStore(tmp_db_path, embedding_dim=dim)
        with store:
            store.add_document(
                path="/test/doc.md",
                title="Test Doc",
                chunks=["chunk one", "chunk two", "chunk three"],
                embeddings=[[0.1 + i * 0.01] * dim for i in range(3)],
            )

            progress_calls: list[tuple[int, int]] = []

            def on_progress(processed: int, total: int) -> None:
                progress_calls.append((processed, total))

            store.reindex(
                embed_fn=lambda texts: [[0.5] * dim for _ in texts],
                new_dim=dim,
                batch_size=2,
                progress_cb=on_progress,
            )
            assert len(progress_calls) >= 1
            # Last call should show all chunks processed
            assert progress_calls[-1][0] == 3
            assert progress_calls[-1][1] == 3


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
        count = sample_store._conn.execute("SELECT COUNT(*) AS n FROM search_stats").fetchone()["n"]
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
            query="test",
            best_distance=0.5,
            relevance_tier="high",
            result_count=5,
        )
        assert event_id > 0

    def test_mark_search_dismissed(self, sample_store: VstashStore) -> None:
        event_id = sample_store.record_search_event(
            query="test",
            best_distance=0.99,
            relevance_tier="low",
            result_count=3,
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


# ------------------------------------------------------------------ #
# search(explain=True)                                                 #
# ------------------------------------------------------------------ #


class TestSearchExplain:
    """Test search with explain=True returns diagnostic breakdowns."""

    def test_explain_false_returns_no_explain(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python")
        for r in results:
            assert r.explain is None

    def test_explain_true_returns_explain_info(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python programming", explain=True)
        assert len(results) > 0
        for r in results:
            assert r.explain is not None
            assert r.explain.rrf_total > 0

    def test_explain_has_vec_or_fts_rank(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python", explain=True)
        for r in results:
            ex = r.explain
            assert ex is not None
            # Every result must come from at least one source
            assert ex.vec_rank is not None or ex.fts_rank is not None

    def test_explain_rrf_components_sum(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python", explain=True)
        for r in results:
            ex = r.explain
            assert ex is not None
            # vec + fts should approximately equal total (within float precision)
            assert abs(ex.rrf_vec + ex.rrf_fts - ex.rrf_total) < 1e-5

    def test_explain_fts_terms_populated(self, populated_store: VstashStore) -> None:
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python programming", explain=True)
        assert len(results) > 0
        # At least one result should have fts_terms
        assert any(r.explain and r.explain.fts_terms for r in results)

    def test_explain_empty_store(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        results = sample_store.search([0.1] * dim, "anything", explain=True)
        assert results == []


# ------------------------------------------------------------------ #
# Adaptive RRF weights                                                 #
# ------------------------------------------------------------------ #


class TestAdaptiveRRF:
    """Test adaptive RRF weight computation based on query IDF."""

    def test_adaptive_returns_valid_weights(self, populated_store: VstashStore) -> None:
        vec_w, fts_w, cutoff = populated_store._compute_adaptive_rrf_params("Python programming")
        assert 0.0 <= vec_w <= 1.0
        assert 0.0 <= fts_w <= 1.0
        assert cutoff > 0
        assert abs(vec_w + fts_w - 1.0) < 1e-6

    def test_long_query_favors_vector(self, populated_store: VstashStore) -> None:
        """Queries >50 words should heavily favor vector search."""
        long_query = " ".join(["word"] * 60)
        vec_w, fts_w, cutoff = populated_store._compute_adaptive_rrf_params(long_query)
        assert vec_w == 0.9
        assert fts_w == 0.1
        assert cutoff == 5.0  # relaxed for diffuse embeddings

    def test_rare_terms_boost_fts(self, populated_store: VstashStore) -> None:
        """OOV / very rare terms should increase FTS weight."""
        # Use a term unlikely to be in the test corpus
        vec_w_rare, fts_w_rare, _ = populated_store._compute_adaptive_rrf_params(
            "XYZZY_UNKNOWN_TERM"
        )
        vec_w_common, fts_w_common, _ = populated_store._compute_adaptive_rrf_params("Python")
        # Rare term should give higher FTS weight than common term
        assert fts_w_rare > fts_w_common

    def test_empty_store_returns_defaults(self, sample_store: VstashStore) -> None:
        vec_w, fts_w, cutoff = sample_store._compute_adaptive_rrf_params("anything")
        assert vec_w == 0.6
        assert fts_w == 0.4
        assert cutoff == 1.15

    def test_adaptive_enabled_by_default(self, populated_store: VstashStore) -> None:
        """search() uses adaptive weights by default."""
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python", explain=True, adaptive_rrf=True)
        if results and results[0].explain:
            ex = results[0].explain
            # Weights should be set (not None) and potentially differ from 0.6/0.4
            assert ex.rrf_vec_weight is not None
            assert ex.rrf_fts_weight is not None

    def test_adaptive_disabled_uses_fixed(self, populated_store: VstashStore) -> None:
        """adaptive_rrf=False uses the fixed default weights."""
        dim = populated_store.embedding_dim
        results = populated_store.search([0.1] * dim, "Python", explain=True, adaptive_rrf=False)
        if results and results[0].explain:
            ex = results[0].explain
            assert ex.rrf_vec_weight == 0.6
            assert ex.rrf_fts_weight == 0.4

    def test_vector_empty_fallback_boosts_fts_score(self, sample_store: VstashStore) -> None:
        """When vec pool is empty post-cutoff, FTS gets full weight (#156).

        Observable contract: a literal FTS match at rank 0 scores
        ``1.0 * 1/(60 + 0) ≈ 0.01667`` after the fallback, vs
        ``0.4 * 1/60 ≈ 0.00667`` without it. The 2.5x score increase
        reflects the corrected weighting, not a corpus change.
        """
        dim = sample_store.embedding_dim
        # Three docs with orthogonal embeddings — distance cutoff will
        # eliminate all of them. FTS will still find the one that
        # contains the query term.
        for i in range(3):
            emb = [0.0] * dim
            emb[i] = 1.0
            sample_store.add_document(
                path=f"/test/fallback_doc_{i}.md",
                title=f"Fallback Doc {i}",
                chunks=[f"zebrafish xyzzy_{i} is a known literal term"],
                embeddings=[emb],
            )

        # Query embedding is orthogonal to every doc → distance_cutoff=0
        # drops all vec candidates. FTS5 still finds "xyzzy_1".
        query_emb = [0.0] * dim
        query_emb[200] = 1.0

        results = sample_store.search(
            query_emb,
            "xyzzy_1",
            top_k=5,
            distance_cutoff=0.0,
        )

        # Exactly one doc should surface via FTS.
        assert len(results) == 1
        assert "xyzzy_1" in results[0].text

        # And its score should reflect fts_weight=1.0 at rank 0:
        # 1.0 * 1/(60 + 0) = 0.016666..., NOT 0.4 * 1/60 = 0.00666...
        # Tolerate tiny float noise.
        assert abs(results[0].score - (1.0 / 60.0)) < 1e-4, (
            f"expected ~0.01667 (fts_weight=1.0 at rank 0), got {results[0].score}"
        )

    def test_vector_empty_fallback_sets_worst_distance(self, sample_store: VstashStore) -> None:
        """The fallback must reset last_best_distance to the worst value.

        Without the reset, the relevance-tier signal would report
        "high" confidence even though every vector candidate was
        eliminated — a lie.
        """
        dim = sample_store.embedding_dim
        for i in range(2):
            emb = [0.0] * dim
            emb[i] = 1.0
            sample_store.add_document(
                path=f"/test/tier_doc_{i}.md",
                title=f"Tier Doc {i}",
                chunks=[f"literalmark_{i} appears here"],
                embeddings=[emb],
            )

        query_emb = [0.0] * dim
        query_emb[300] = 1.0

        # First a hybrid call to prime last_best_distance to a non-
        # worst value, so we can assert the fallback actually resets it.
        sample_store.search(query_emb, "literalmark_0", top_k=3)
        assert sample_store.last_best_distance < 2.0

        # Now trigger the fallback.
        sample_store.search(query_emb, "literalmark_0", top_k=3, distance_cutoff=0.0)
        assert sample_store.last_best_distance == 2.0, (
            f"fallback did not reset last_best_distance to 2.0, "
            f"got {sample_store.last_best_distance}"
        )

    def test_vector_empty_fallback_increments_counter(self, sample_store: VstashStore) -> None:
        """The fallback must increment adaptive_rrf_vector_empty_fallback_total."""
        from vstash.metrics import registry

        dim = sample_store.embedding_dim
        emb = [0.0] * dim
        emb[0] = 1.0
        sample_store.add_document(
            path="/test/ctr_doc.md",
            title="Counter Doc",
            chunks=["countkeyword lives here"],
            embeddings=[emb],
        )

        snap_before = registry.snapshot()
        before = snap_before.get("counters", {}).get("adaptive_rrf_vector_empty_fallback_total", 0)

        # Orthogonal query embedding → impossible cutoff → vec empty.
        query_emb = [0.0] * dim
        query_emb[300] = 1.0
        sample_store.search(query_emb, "countkeyword", top_k=3, distance_cutoff=0.0)

        snap_after = registry.snapshot()
        after = snap_after.get("counters", {}).get("adaptive_rrf_vector_empty_fallback_total", 0)
        assert after == before + 1, f"counter did not increment: before={before}, after={after}"

    def test_hybrid_path_does_not_trigger_fallback(self, sample_store: VstashStore) -> None:
        """Regression guard: normal hybrid search must NOT touch the fallback.

        The fallback is only for the degenerate vec-empty case. A
        healthy hybrid query with surviving vec candidates must leave
        the counter alone and keep the adaptive-RRF weights in play.
        """
        from vstash.metrics import registry

        dim = sample_store.embedding_dim
        emb = [1.0] + [0.0] * (dim - 1)
        sample_store.add_document(
            path="/test/healthy.md",
            title="Healthy Doc",
            chunks=["healthy hybrid content for a normal search"],
            embeddings=[emb],
        )

        snap_before = registry.snapshot()
        before = snap_before.get("counters", {}).get("adaptive_rrf_vector_empty_fallback_total", 0)

        # Query is nearly aligned — vec_rows will have the doc, the
        # default distance cutoff keeps it, and no fallback fires.
        results = sample_store.search([1.0] + [0.0] * (dim - 1), "healthy", top_k=3)
        assert len(results) >= 1

        snap_after = registry.snapshot()
        after = snap_after.get("counters", {}).get("adaptive_rrf_vector_empty_fallback_total", 0)
        assert after == before, (
            f"hybrid search triggered the vector-empty fallback: before={before}, after={after}"
        )

    def test_idf_cache_built_once(self, populated_store: VstashStore) -> None:
        """IDF cache should be built on first call and reused."""
        # First call builds cache
        populated_store._compute_adaptive_rrf_params("test")
        assert populated_store._idf_cache is not None

        # Cache should contain terms
        idf_dict, total_chunks = populated_store._idf_cache
        assert total_chunks > 0
        assert len(idf_dict) > 0

        # Second call reuses (same object)
        cache_ref = populated_store._idf_cache
        populated_store._compute_adaptive_rrf_params("another test")
        assert populated_store._idf_cache is cache_ref

    def test_idf_cache_invalidated_on_add(self, sample_store: VstashStore) -> None:
        """Adding a document should invalidate the IDF cache."""
        dim = sample_store.embedding_dim
        sample_store._compute_adaptive_rrf_params("test")
        assert sample_store._idf_cache is not None

        sample_store.add_document(
            path="/test/new.md",
            title="New",
            chunks=["new content here"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        assert sample_store._idf_cache is None

    def test_idf_cache_invalidated_on_delete(self, populated_store: VstashStore) -> None:
        """Deleting a document should invalidate the IDF cache."""

        populated_store._compute_adaptive_rrf_params("test")
        assert populated_store._idf_cache is not None

        populated_store.delete_document("/test/python_guide.md")
        assert populated_store._idf_cache is None

    def test_no_regression_vs_fixed(self, populated_store: VstashStore) -> None:
        """Adaptive should not degrade results vs fixed on the test corpus."""
        dim = populated_store.embedding_dim
        query = "Python programming"
        qvec = [0.1] * dim

        fixed_results = populated_store.search(qvec, query, adaptive_rrf=False)
        adaptive_results = populated_store.search(qvec, query, adaptive_rrf=True)

        # Both should return results
        assert len(fixed_results) > 0
        assert len(adaptive_results) > 0
        # Top result should be the same (or at least overlap)
        fixed_paths = {r.path for r in fixed_results}
        adaptive_paths = {r.path for r in adaptive_results}
        assert len(fixed_paths & adaptive_paths) > 0

    def test_long_query_cutoff_relaxation_recovers_results(self, sample_store: VstashStore) -> None:
        """Long queries (>50 words) should relax distance cutoff to 5.0x,
        recovering results that the tight 1.15x cutoff would discard.

        This protects the ArguAna BEIR claim (+19% from cutoff relaxation).
        """
        dim = sample_store.embedding_dim

        # Create docs with spread-out embeddings (simulating diffuse long-query distances)
        close_emb = [0.1] * dim
        far_emb = [0.9] * dim  # far from query but still relevant
        sample_store.add_document(
            path="/test/close.md",
            title="Close Doc",
            chunks=["close content about programming"],
            embeddings=[close_emb],
            source_type="markdown",
        )
        sample_store.add_document(
            path="/test/far.md",
            title="Far Doc",
            chunks=["far content about programming"],
            embeddings=[far_emb],
            source_type="markdown",
        )

        query_emb = [0.12] * dim  # near close_emb, far from far_emb
        long_query = " ".join(["programming"] * 60)

        # Tight cutoff (1.15x) should drop the far doc
        tight_results = sample_store.search(
            query_emb, long_query, adaptive_rrf=False, distance_cutoff=1.15
        )
        tight_paths = {r.path for r in tight_results}

        # Adaptive with long query should relax cutoff to 5.0x, recovering far doc
        adaptive_results = sample_store.search(query_emb, long_query, adaptive_rrf=True)
        adaptive_paths = {r.path for r in adaptive_results}

        # The adaptive path must return at least as many results
        assert len(adaptive_results) >= len(tight_results)
        # If the far doc was dropped by tight cutoff, adaptive should recover it
        if "/test/far.md" not in tight_paths:
            assert "/test/far.md" in adaptive_paths, (
                "Adaptive cutoff relaxation failed to recover far document — "
                "this would break ArguAna BEIR gains"
            )

    def test_idf_weighting_shifts_fts_weight(self, populated_store: VstashStore) -> None:
        """IDF weighting must produce measurably different weights for rare vs common
        terms, and these weights must flow through to search results via explain.

        This protects the claim that IDF-based sigmoid correctly identifies query regimes.
        """
        dim = populated_store.embedding_dim
        qvec = [0.1] * dim

        # Rare term: not in corpus → high IDF → higher FTS weight
        rare_results = populated_store.search(
            qvec, "XYZZY_NONEXISTENT_TERM", explain=True, adaptive_rrf=True
        )
        # Common term: in corpus → lower IDF → lower FTS weight
        common_results = populated_store.search(qvec, "Python", explain=True, adaptive_rrf=True)

        # Both should have explain info with adaptive weights
        assert rare_results and rare_results[0].explain
        assert common_results and common_results[0].explain

        rare_fts_w = rare_results[0].explain.rrf_fts_weight
        common_fts_w = common_results[0].explain.rrf_fts_weight
        assert rare_fts_w is not None
        assert common_fts_w is not None

        # Rare terms MUST get higher FTS weight than common terms
        assert rare_fts_w > common_fts_w, (
            f"IDF weighting broken: rare term FTS weight ({rare_fts_w}) "
            f"should exceed common term FTS weight ({common_fts_w})"
        )

    def test_adaptive_weights_are_not_always_default(self, populated_store: VstashStore) -> None:
        """Guard against a regression where _compute_adaptive_rrf_params
        always returns the fixed defaults (0.6/0.4). At least one query
        type must produce non-default weights.
        """
        queries = [
            "XYZZY_UNKNOWN",  # rare → should boost FTS
            " ".join(["word"] * 60),  # long → should return 0.9/0.1
        ]
        saw_non_default = False
        for q in queries:
            vec_w, fts_w, _ = populated_store._compute_adaptive_rrf_params(q)
            if (vec_w, fts_w) != (0.6, 0.4):
                saw_non_default = True
                break
        assert saw_non_default, (
            "Adaptive RRF always returned default weights (0.6/0.4) — "
            "the adaptive logic is effectively dead code"
        )


# ------------------------------------------------------------------ #
# batch_mode — deferred IDF cache invalidation                        #
# ------------------------------------------------------------------ #


class TestBatchMode:
    """Tests for ``VstashStore.batch_mode()`` context manager."""

    def test_single_invalidation_during_batch(self, sample_store: "VstashStore"):
        """Inside batch_mode, _invalidate_idf_cache defers the actual clear."""
        dim = sample_store.embedding_dim

        # Prime the IDF cache so we can observe invalidation
        sample_store.add_document(
            path="/seed.md",
            title="seed",
            chunks=["seed text"],
            embeddings=[[0.1] * dim],
        )
        sample_store._build_idf_cache()
        assert sample_store._idf_cache is not None

        with sample_store.batch_mode():
            # Add a doc — should NOT clear the cache yet
            sample_store.add_document(
                path="/a.md",
                title="a",
                chunks=["alpha"],
                embeddings=[[0.2] * dim],
            )
            assert sample_store._idf_cache is not None, "Cache was cleared inside batch_mode"
            assert sample_store._batch_dirty is True

            # Add another doc — still deferred
            sample_store.add_document(
                path="/b.md",
                title="b",
                chunks=["beta"],
                embeddings=[[0.3] * dim],
            )
            assert sample_store._idf_cache is not None

        # After exiting batch_mode, cache should be invalidated
        assert sample_store._idf_cache is None
        assert sample_store._batch_dirty is False

    def test_no_invalidation_when_batch_clean(self, sample_store: "VstashStore"):
        """If no mutations happen inside batch_mode, cache stays intact."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/seed.md",
            title="seed",
            chunks=["seed"],
            embeddings=[[0.1] * dim],
        )
        sample_store._build_idf_cache()
        assert sample_store._idf_cache is not None

        with sample_store.batch_mode():
            pass  # no mutations

        assert sample_store._idf_cache is not None, "Cache was invalidated despite no mutations"

    def test_nested_batch_mode(self, sample_store: "VstashStore"):
        """Nested batch_mode defers invalidation until outermost exits."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/seed.md",
            title="seed",
            chunks=["seed"],
            embeddings=[[0.1] * dim],
        )
        sample_store._build_idf_cache()

        with sample_store.batch_mode():
            with sample_store.batch_mode():
                sample_store.add_document(
                    path="/inner.md",
                    title="inner",
                    chunks=["inner"],
                    embeddings=[[0.2] * dim],
                )
                assert sample_store._idf_cache is not None
            # inner exited but outer still active — should NOT invalidate
            assert sample_store._idf_cache is not None
            assert sample_store._batch_depth == 1

        # outermost exited — now invalidate
        assert sample_store._idf_cache is None
        assert sample_store._batch_depth == 0

    def test_batch_mode_exception_still_invalidates(self, sample_store: "VstashStore"):
        """If an exception occurs inside batch_mode, cache is still invalidated."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/seed.md",
            title="seed",
            chunks=["seed"],
            embeddings=[[0.1] * dim],
        )
        sample_store._build_idf_cache()

        try:
            with sample_store.batch_mode():
                sample_store.add_document(
                    path="/x.md",
                    title="x",
                    chunks=["x"],
                    embeddings=[[0.2] * dim],
                )
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass

        assert sample_store._idf_cache is None
        assert sample_store._batch_depth == 0
        assert sample_store._batch_dirty is False

    def test_search_works_after_batch(self, sample_store: "VstashStore"):
        """Search returns correct results after batch_mode exits."""
        dim = sample_store.embedding_dim

        with sample_store.batch_mode():
            sample_store.add_document(
                path="/doc.md",
                title="Python Guide",
                chunks=["Python is a programming language"],
                embeddings=[[0.5] * dim],
            )

        results = sample_store.search([0.5] * dim, "Python programming", top_k=5)
        assert len(results) >= 1
        assert results[0].path == "/doc.md"

    def test_delete_inside_batch(self, sample_store: "VstashStore"):
        """delete_document inside batch_mode defers cache invalidation."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/to_delete.md",
            title="ephemeral",
            chunks=["temporary content"],
            embeddings=[[0.4] * dim],
        )
        sample_store._build_idf_cache()
        assert sample_store._idf_cache is not None

        with sample_store.batch_mode():
            sample_store.delete_document("/to_delete.md")
            assert sample_store._idf_cache is not None, (
                "Cache was cleared inside batch_mode during delete"
            )

        assert sample_store._idf_cache is None

    def test_delete_by_prefix_inside_batch(self, sample_store: "VstashStore"):
        """delete_by_path_prefix inside batch_mode defers cache invalidation."""
        dim = sample_store.embedding_dim
        for i in range(3):
            sample_store.add_document(
                path=f"/bulk/doc_{i}.md",
                title=f"bulk {i}",
                chunks=[f"bulk content {i}"],
                embeddings=[[0.1 * (i + 1)] * dim],
            )
        sample_store._build_idf_cache()
        assert sample_store._idf_cache is not None

        with sample_store.batch_mode():
            deleted = sample_store.delete_by_path_prefix("/bulk/")
            assert deleted == 3
            assert sample_store._idf_cache is not None

        assert sample_store._idf_cache is None

    def test_nested_exception_with_recovery(self, sample_store: "VstashStore"):
        """Inner batch raises, outer batch catches and continues normally."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/seed.md",
            title="seed",
            chunks=["seed"],
            embeddings=[[0.1] * dim],
        )
        sample_store._build_idf_cache()

        with sample_store.batch_mode():
            sample_store.add_document(
                path="/before.md",
                title="before",
                chunks=["before"],
                embeddings=[[0.2] * dim],
            )
            try:
                with sample_store.batch_mode():
                    sample_store.add_document(
                        path="/inner.md",
                        title="inner",
                        chunks=["inner"],
                        embeddings=[[0.3] * dim],
                    )
                    raise ValueError("inner failure")
            except ValueError:
                pass

            # Outer batch still active — cache still deferred
            assert sample_store._idf_cache is not None
            assert sample_store._batch_depth == 1

            sample_store.add_document(
                path="/after.md",
                title="after",
                chunks=["after"],
                embeddings=[[0.4] * dim],
            )

        # Outermost exited — now invalidated
        assert sample_store._idf_cache is None
        assert sample_store._batch_depth == 0


# ------------------------------------------------------------------ #
# Recency boost + temporal filters                                     #
# ------------------------------------------------------------------ #


class TestRecencyBoost:
    """Tests for recency_boost parameter in search()."""

    def test_recency_boost_zero_is_noop(self, populated_store: "VstashStore"):
        """recency_boost=0.0 should not change result ordering."""
        dim = populated_store.embedding_dim
        q = [0.1] * dim
        base = populated_store.search(q, "Python", top_k=5, recency_boost=0.0)
        normal = populated_store.search(q, "Python", top_k=5)
        assert [r.chunk_id for r in base] == [r.chunk_id for r in normal]

    def test_recency_boost_favors_recent(self, sample_store: "VstashStore"):
        """Chunks created recently should rank higher with recency_boost > 0."""
        dim = sample_store.embedding_dim
        from datetime import datetime, timedelta, timezone

        old_time = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        # Add old doc
        sample_store.add_document(
            path="/old.md",
            title="Old Doc",
            chunks=["Python programming basics"],
            embeddings=[[0.5] * dim],
        )
        # Manually backdate the chunk's created_at
        sample_store._conn.execute(
            "UPDATE chunks SET created_at = ? WHERE doc_id = (SELECT id FROM documents WHERE path = '/old.md')",
            [old_time],
        )
        sample_store._conn.commit()

        # Add recent doc with same semantic content
        sample_store.add_document(
            path="/new.md",
            title="New Doc",
            chunks=["Python programming basics"],
            embeddings=[[0.5] * dim],
        )

        # Without recency boost — both should have equal scores
        q = [0.5] * dim
        results_no_boost = sample_store.search(q, "Python programming", top_k=5, recency_boost=0.0)
        assert len(results_no_boost) >= 2

        # With strong recency boost — new doc should rank first
        results_boosted = sample_store.search(q, "Python programming", top_k=5, recency_boost=2.0)
        assert results_boosted[0].path == "/new.md"

    def test_recency_boost_preserves_results(self, populated_store: "VstashStore"):
        """recency_boost should not add or remove results, only reorder."""
        dim = populated_store.embedding_dim
        q = [0.15] * dim
        base = populated_store.search(q, "programming", top_k=5)
        boosted = populated_store.search(q, "programming", top_k=5, recency_boost=1.0)
        assert set(r.chunk_id for r in base) == set(r.chunk_id for r in boosted)


class TestTemporalFilters:
    """Tests for added_after/added_before date filters."""

    def test_added_after_filters_old_docs(self, sample_store: "VstashStore"):
        """added_after should exclude documents added before the cutoff."""
        dim = sample_store.embedding_dim

        sample_store.add_document(
            path="/old.md",
            title="Old",
            chunks=["old content about Python"],
            embeddings=[[0.5] * dim],
        )
        # Backdate the document
        sample_store._conn.execute(
            "UPDATE documents SET added_at = '2023-01-01T00:00:00' WHERE path = '/old.md'"
        )
        sample_store._conn.commit()

        sample_store.add_document(
            path="/new.md",
            title="New",
            chunks=["new content about Python"],
            embeddings=[[0.5] * dim],
        )

        q = [0.5] * dim
        # Filter: only docs from 2024 onwards
        results = sample_store.search(q, "Python", top_k=5, added_after="2024-01-01")
        paths = [r.path for r in results]
        assert "/new.md" in paths
        assert "/old.md" not in paths

    def test_added_before_filters_new_docs(self, sample_store: "VstashStore"):
        """added_before should exclude documents added after the cutoff."""
        dim = sample_store.embedding_dim

        sample_store.add_document(
            path="/old.md",
            title="Old",
            chunks=["old content"],
            embeddings=[[0.5] * dim],
        )
        sample_store._conn.execute(
            "UPDATE documents SET added_at = '2023-06-01T00:00:00' WHERE path = '/old.md'"
        )
        sample_store._conn.commit()

        sample_store.add_document(
            path="/new.md",
            title="New",
            chunks=["new content"],
            embeddings=[[0.5] * dim],
        )

        q = [0.5] * dim
        results = sample_store.search(q, "content", top_k=5, added_before="2024-01-01")
        paths = [r.path for r in results]
        assert "/old.md" in paths
        assert "/new.md" not in paths

    def test_date_range_filter(self, sample_store: "VstashStore"):
        """Combining added_after and added_before creates a date range."""
        dim = sample_store.embedding_dim

        for date, name in [("2023-01-01", "jan"), ("2023-06-15", "jun"), ("2024-01-01", "new")]:
            sample_store.add_document(
                path=f"/{name}.md",
                title=name,
                chunks=[f"{name} content about testing"],
                embeddings=[[0.5] * dim],
            )
            sample_store._conn.execute(
                "UPDATE documents SET added_at = ? WHERE path = ?",
                [f"{date}T00:00:00", f"/{name}.md"],
            )
            sample_store._conn.commit()

        q = [0.5] * dim
        results = sample_store.search(
            q, "testing", top_k=5, added_after="2023-03-01", added_before="2023-12-31"
        )
        paths = [r.path for r in results]
        assert "/jun.md" in paths
        assert "/jan.md" not in paths
        assert "/new.md" not in paths

    def test_no_filter_returns_all(self, populated_store: "VstashStore"):
        """Without temporal filters, all documents are returned."""
        dim = populated_store.embedding_dim
        q = [0.15] * dim
        results = populated_store.search(q, "programming", top_k=10)
        assert len(results) >= 1
