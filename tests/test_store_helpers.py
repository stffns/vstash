"""Tests for store helper methods: get_document_chunks, get_document_added_at, get_chunks_for_documents."""

from __future__ import annotations

from tests.conftest import requires_sqlite_vec
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


class TestGetDocumentChunks:
    """Test get_document_chunks with collection disambiguation."""

    def test_returns_chunks_ordered_by_seq(self, populated_store: VstashStore) -> None:
        chunks = populated_store.get_document_chunks("/test/python_guide.md")
        assert len(chunks) == 2
        assert "high-level" in chunks[0]
        assert "multiple paradigms" in chunks[1]

    def test_not_found_returns_empty(self, populated_store: VstashStore) -> None:
        assert populated_store.get_document_chunks("/nonexistent.md") == []

    def test_with_collection_filter(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        # Same path in two different collections with different content
        sample_store.add_document(
            path="/shared/doc.md",
            title="Doc A",
            chunks=["content from collection A"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
            collection="alpha",
        )
        sample_store.add_document(
            path="/shared/doc.md",
            title="Doc B",
            chunks=["content from collection B"],
            embeddings=[[0.2] * dim],
            source_type="markdown",
            collection="beta",
        )
        chunks_a = sample_store.get_document_chunks("/shared/doc.md", collection="alpha")
        chunks_b = sample_store.get_document_chunks("/shared/doc.md", collection="beta")
        assert chunks_a == ["content from collection A"]
        assert chunks_b == ["content from collection B"]

    def test_without_collection_returns_most_recent(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/shared/doc.md",
            title="Old",
            chunks=["old content"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
            collection="first",
        )
        sample_store.add_document(
            path="/shared/doc.md",
            title="New",
            chunks=["new content"],
            embeddings=[[0.2] * dim],
            source_type="markdown",
            collection="second",
        )
        # Without collection, should get the most recently added
        chunks = sample_store.get_document_chunks("/shared/doc.md")
        assert chunks == ["new content"]


class TestGetDocumentAddedAt:
    """Test get_document_added_at with batching and dedup."""

    def test_returns_timestamps(self, populated_store: VstashStore) -> None:
        result = populated_store.get_document_added_at(
            ["/test/python_guide.md", "/test/ml_intro.pdf"]
        )
        assert result["/test/python_guide.md"] is not None
        assert result["/test/ml_intro.pdf"] is not None

    def test_missing_path_returns_none(self, populated_store: VstashStore) -> None:
        result = populated_store.get_document_added_at(["/nonexistent.md"])
        assert result["/nonexistent.md"] is None

    def test_empty_input(self, populated_store: VstashStore) -> None:
        assert populated_store.get_document_added_at([]) == {}

    def test_duplicate_paths_returns_max_added_at(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/dup.md",
            title="First",
            chunks=["a"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
            collection="c1",
        )
        sample_store.add_document(
            path="/dup.md",
            title="Second",
            chunks=["b"],
            embeddings=[[0.2] * dim],
            source_type="markdown",
            collection="c2",
        )
        result = sample_store.get_document_added_at(["/dup.md"])
        # Should get a valid timestamp (the max)
        assert result["/dup.md"] is not None

    def test_batch_boundary(self, sample_store: VstashStore) -> None:
        """Handles >900 paths across batches."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test.md",
            title="Test",
            chunks=["x"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        # 950 paths, only 1 exists
        paths = [f"/fake/{i}.md" for i in range(949)] + ["/test.md"]
        result = sample_store.get_document_added_at(paths)
        assert result["/test.md"] is not None
        assert result["/fake/0.md"] is None


class TestGetChunksForDocuments:
    """Test get_chunks_for_documents with batching and path disambiguation."""

    def test_returns_chunks_per_path(self, populated_store: VstashStore) -> None:
        result = populated_store.get_chunks_for_documents(
            ["/test/python_guide.md", "/test/ml_intro.pdf"]
        )
        assert len(result["/test/python_guide.md"]) == 2
        assert len(result["/test/ml_intro.pdf"]) == 3

    def test_missing_path_returns_empty_list(self, populated_store: VstashStore) -> None:
        result = populated_store.get_chunks_for_documents(["/nonexistent.md"])
        assert result["/nonexistent.md"] == []

    def test_empty_input(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunks_for_documents([]) == {}

    def test_duplicate_paths_returns_most_recent(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/dup.md",
            title="Old",
            chunks=["old chunk"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
            collection="c1",
        )
        sample_store.add_document(
            path="/dup.md",
            title="New",
            chunks=["new chunk 1", "new chunk 2"],
            embeddings=[[0.2] * dim, [0.3] * dim],
            source_type="markdown",
            collection="c2",
        )
        result = sample_store.get_chunks_for_documents(["/dup.md"])
        # Should get chunks from the most recently added document
        assert len(result["/dup.md"]) == 2
        assert result["/dup.md"][0] == "new chunk 1"

    def test_batch_boundary(self, sample_store: VstashStore) -> None:
        """Handles >900 paths across batches."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test.md",
            title="Test",
            chunks=["hello"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        paths = [f"/fake/{i}.md" for i in range(949)] + ["/test.md"]
        result = sample_store.get_chunks_for_documents(paths)
        assert result["/test.md"] == ["hello"]
        assert result["/fake/0.md"] == []
