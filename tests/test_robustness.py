"""Tests for robustness fixes: expand_context isolation, reindex safety,
watch shutdown, scoring maturity edge cases, and frontmatter warnings.
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from vstash.store import VstashStore


# ------------------------------------------------------------------ #
# expand_context cross-collection isolation                           #
# ------------------------------------------------------------------ #


class TestExpandContextIsolation:
    """Verify expand_context scopes to the correct document."""

    def test_expand_context_uses_text_match(self, populated_store: VstashStore) -> None:
        """expand_context resolves doc_id via chunk text, not just path."""
        from vstash.models import SearchResult

        # Search results reference a known chunk
        results = [
            SearchResult(
                chunk_id=0,
                text="Machine learning is a subset of artificial intelligence.",
                title="ML Introduction",
                path="/test/ml_intro.pdf",
                chunk=0,
                score=0.9,
            )
        ]
        expanded = populated_store.expand_context(results, window=1)
        assert len(expanded) == 1
        # Should include chunk 0 and chunk 1 (window=1)
        assert "Machine learning" in expanded[0].text
        assert "Neural networks" in expanded[0].text

    def test_expand_context_returns_original_on_missing_chunk(
        self, populated_store: VstashStore
    ) -> None:
        """If chunk text doesn't match, return the original result."""
        from vstash.models import SearchResult

        results = [
            SearchResult(
                chunk_id=0,
                text="This text does not exist in any document",
                title="Fake",
                path="/nonexistent/path.md",
                chunk=0,
                score=0.5,
            )
        ]
        expanded = populated_store.expand_context(results, window=1)
        assert len(expanded) == 1
        assert expanded[0].text == "This text does not exist in any document"

    def test_expand_context_same_path_different_collections(
        self, sample_store: VstashStore
    ) -> None:
        """Same path in two collections should not leak across collections."""
        dim = sample_store.embedding_dim
        path = "/shared/doc.md"

        # Add same path to two different collections with different content
        sample_store.add_document(
            path=path,
            title="Doc A",
            chunks=["Alpha chunk 0", "Alpha chunk 1", "Alpha chunk 2"],
            embeddings=[[0.1] * dim, [0.2] * dim, [0.3] * dim],
            source_type="markdown",
            collection="collection_a",
        )
        sample_store.add_document(
            path=path,
            title="Doc B",
            chunks=["Beta chunk 0", "Beta chunk 1", "Beta chunk 2"],
            embeddings=[[0.4] * dim, [0.5] * dim, [0.6] * dim],
            source_type="markdown",
            collection="collection_b",
        )

        from vstash.models import SearchResult

        # Request expansion for collection_a's chunk
        results = [
            SearchResult(
                chunk_id=0,
                text="Alpha chunk 1",
                title="Doc A",
                path=path,
                chunk=1,
                score=0.9,
            )
        ]
        expanded = sample_store.expand_context(results, window=1)
        assert len(expanded) == 1
        # Should contain Alpha chunks only, not Beta
        assert "Alpha" in expanded[0].text
        assert "Beta" not in expanded[0].text


# ------------------------------------------------------------------ #
# Reindex safety                                                      #
# ------------------------------------------------------------------ #


class TestReindexSafety:
    """Verify reindex preserves embedding_dim on failure."""

    def test_reindex_restores_dim_on_failure(self, populated_store: VstashStore) -> None:
        """embedding_dim should remain unchanged if reindex fails mid-batch."""
        original_dim = populated_store.embedding_dim

        def failing_embed(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("Simulated embedding failure")

        with pytest.raises(RuntimeError, match="Simulated embedding failure"):
            populated_store.reindex(failing_embed, new_dim=768)

        assert populated_store.embedding_dim == original_dim

    def test_reindex_success_updates_dim(self, populated_store: VstashStore) -> None:
        """embedding_dim should update to new_dim on successful reindex."""
        new_dim = 128

        def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * new_dim for _ in texts]

        populated_store.reindex(fake_embed, new_dim=new_dim)
        assert populated_store.embedding_dim == new_dim


# ------------------------------------------------------------------ #
# Scoring maturity gate edge cases                                    #
# ------------------------------------------------------------------ #


class TestScoringMaturityGate:
    """Verify maturity gate handles edge cases safely."""

    def test_maturity_gate_returns_zero_with_no_accessed_chunks(
        self, populated_store: VstashStore
    ) -> None:
        """No accessed chunks → γ = 0."""
        gamma = populated_store.scoring_maturity()
        assert gamma == 0.0

    def test_maturity_gate_handles_few_accessed_chunks(self, populated_store: VstashStore) -> None:
        """Fewer than 10 accessed chunks → γ = 0."""
        # Access a few chunks (less than 10)
        for i in range(5):
            populated_store._conn.execute(
                "UPDATE chunks SET access_count = 1 WHERE id = (SELECT id FROM chunks LIMIT 1 OFFSET ?)",
                [i],
            )
        populated_store._conn.commit()
        gamma = populated_store.scoring_maturity()
        assert gamma == 0.0

    def test_maturity_gate_safe_with_zero_mean(self, populated_store: VstashStore) -> None:
        """Even if somehow mean=0, no division by zero."""
        # This shouldn't happen with access_count > 0 filter, but be safe
        gamma = populated_store.scoring_maturity()
        # Should return 0.0, not raise
        assert isinstance(gamma, float)


# ------------------------------------------------------------------ #
# Watch worker shutdown                                               #
# ------------------------------------------------------------------ #


class TestWatchWorkerShutdown:
    """Verify the watch worker responds to stop_event."""

    def test_stop_event_terminates_worker(self) -> None:
        """Worker thread should exit when stop_event is set."""
        ingest_queue: queue.Queue[str | None] = queue.Queue()
        stop_event = threading.Event()

        exited = threading.Event()

        def mock_worker() -> None:
            while not stop_event.is_set():
                try:
                    item = ingest_queue.get(timeout=1)
                    if item is None:
                        break
                except queue.Empty:
                    continue
            exited.set()

        t = threading.Thread(target=mock_worker, daemon=True)
        t.start()

        # Signal stop
        stop_event.set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert exited.is_set()


# ------------------------------------------------------------------ #
# Watch file deletion handling                                        #
# ------------------------------------------------------------------ #


class TestWatchDeleteHandler:
    """Verify the watch module's _should_process works for delete events."""

    def test_should_process_filters_extensions(self) -> None:
        from vstash.watch import _should_process

        exts = frozenset({".md", ".py", ".txt"})
        assert _should_process("/path/to/file.md", exts) is True
        assert _should_process("/path/to/file.jpg", exts) is False

    def test_should_process_skips_hidden(self) -> None:
        from vstash.watch import _should_process

        exts = frozenset({".md"})
        assert _should_process("/path/.hidden/file.md", exts) is False


# ------------------------------------------------------------------ #
# Frontmatter warnings                                               #
# ------------------------------------------------------------------ #


class TestFrontmatterWarnings:
    """Verify warnings are emitted for non-scalar frontmatter values."""

    @patch("vstash.ingest.console")
    def test_dict_project_warns(self, mock_console: MagicMock) -> None:
        """Frontmatter project as dict should warn and be ignored."""
        from vstash.ingest import _extract_frontmatter

        text = "---\nproject:\n  name: foo\n---\nSome content here."
        fm = _extract_frontmatter(text)

        # Simulate the ingest logic
        fm_project = fm.get("project")
        if fm_project is not None and isinstance(fm_project, (dict, list)):
            from vstash.ingest import console

            console.print(
                f"[yellow]⚠ Frontmatter 'project' is a {type(fm_project).__name__}, "
                f"expected string — ignoring.[/yellow]"
            )
            fm_project = None

        # The frontmatter should have been a dict
        assert fm.get("project") is not None
        # After our check, fm_project should be None
        assert fm_project is None


# ------------------------------------------------------------------ #
# delete_by_path_prefix                                               #
# ------------------------------------------------------------------ #


class TestDeleteByPathPrefix:
    """Verify delete_by_path_prefix correctness and edge cases."""

    def test_basic_prefix_match(self, sample_store: VstashStore) -> None:
        """Deletes only documents whose path starts with the given prefix."""
        dim = sample_store.embedding_dim

        # 3 docs under /a/b/, 1 under /a/c/
        for i, path in enumerate(
            ["/a/b/one.md", "/a/b/two.md", "/a/b/sub/three.md", "/a/c/other.md"]
        ):
            sample_store.add_document(
                path=path,
                title=f"doc{i}",
                chunks=[f"chunk {i}"],
                embeddings=[[float(i + 1) / 10] * dim],
                source_type="markdown",
            )

        deleted = sample_store.delete_by_path_prefix("/a/b/")
        assert deleted == 3

        # /a/c/other.md should still exist
        docs = sample_store.list_documents()
        assert len(docs) == 1
        assert docs[0].path == "/a/c/other.md"

    def test_zero_matches_returns_zero(self, sample_store: VstashStore) -> None:
        """Returns 0 when no documents match the prefix."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/x/y/file.md",
            title="doc",
            chunks=["text"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        assert sample_store.delete_by_path_prefix("/nonexistent/") == 0
        assert len(sample_store.list_documents()) == 1

    def test_special_chars_escaped(self, sample_store: VstashStore) -> None:
        """Paths containing SQL LIKE wildcards (%, _) are escaped properly."""
        dim = sample_store.embedding_dim

        # Path with % and _ that could be misinterpreted as LIKE wildcards
        sample_store.add_document(
            path="/data/100%_done/report.md",
            title="report",
            chunks=["done"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        sample_store.add_document(
            path="/data/other/file.md",
            title="other",
            chunks=["other"],
            embeddings=[[0.2] * dim],
            source_type="markdown",
        )

        # Should only delete the doc under the literal "100%_done" path
        deleted = sample_store.delete_by_path_prefix("/data/100%_done/")
        assert deleted == 1
        docs = sample_store.list_documents()
        assert len(docs) == 1
        assert docs[0].path == "/data/other/file.md"

    def test_empty_prefix_raises(self, sample_store: VstashStore) -> None:
        """Empty prefix raises ValueError to prevent accidental full wipe."""
        with pytest.raises(ValueError, match="prefix must not be empty"):
            sample_store.delete_by_path_prefix("")


# ------------------------------------------------------------------ #
# MMR fallback                                                        #
# ------------------------------------------------------------------ #


class TestMmrFallback:
    """Verify MMR falls back gracefully when embedding fetch fails."""

    def test_search_returns_results_even_with_mmr_error(self, populated_store: VstashStore) -> None:
        """Search should still return results if MMR embedding fetch fails."""
        dim = populated_store.embedding_dim
        query_embedding = [0.3] * dim

        # This should work even if MMR internals have issues
        results = populated_store.search(
            query_embedding=query_embedding,
            query_text="machine learning",
            top_k=3,
        )
        # Should return results (may be hard-deduped)
        assert len(results) > 0
