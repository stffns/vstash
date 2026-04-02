"""Tests for get_chunk / get_chunks — direct chunk retrieval by ID."""

from __future__ import annotations

from tests.conftest import requires_sqlite_vec
from vstash.models import ChunkInfo
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


class TestGetChunk:
    """Test single chunk retrieval by ID."""

    def test_get_chunk_returns_chunk_info(self, populated_store: VstashStore) -> None:
        chunk = populated_store.get_chunk(1)
        assert chunk is not None
        assert isinstance(chunk, ChunkInfo)
        assert chunk.chunk_id == 1
        assert chunk.chunk == 0
        assert chunk.text == "Python is a high-level programming language known for its simplicity."
        assert chunk.title == "Python Guide"
        assert chunk.path == "/test/python_guide.md"

    def test_get_chunk_second_document(self, populated_store: VstashStore) -> None:
        chunk = populated_store.get_chunk(3)
        assert chunk is not None
        assert chunk.title == "ML Introduction"
        assert chunk.path == "/test/ml_intro.pdf"
        assert chunk.chunk == 0

    def test_get_chunk_not_found(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunk(9999) is None

    def test_get_chunk_empty_store(self, sample_store: VstashStore) -> None:
        assert sample_store.get_chunk(1) is None

    def test_get_chunk_includes_collection(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/test/doc.md",
            title="Doc",
            chunks=["hello world"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
            collection="notes",
        )
        chunk = sample_store.get_chunk(1)
        assert chunk is not None
        assert chunk.collection == "notes"


class TestGetChunks:
    """Test batch chunk retrieval by IDs."""

    def test_get_chunks_returns_ordered(self, populated_store: VstashStore) -> None:
        chunks = populated_store.get_chunks([3, 1])
        assert len(chunks) == 2
        assert chunks[0].chunk_id == 3
        assert chunks[1].chunk_id == 1

    def test_get_chunks_skips_missing(self, populated_store: VstashStore) -> None:
        chunks = populated_store.get_chunks([1, 9999, 2])
        assert len(chunks) == 2
        assert chunks[0].chunk_id == 1
        assert chunks[1].chunk_id == 2

    def test_get_chunks_empty_list(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunks([]) == []

    def test_get_chunks_all_missing(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunks([9998, 9999]) == []

    def test_get_chunks_duplicate_ids_preserves_repeats(self, populated_store: VstashStore) -> None:
        """Repeated IDs return one entry per occurrence (not deduplicated)."""
        chunks = populated_store.get_chunks([1, 1, 1])
        assert len(chunks) == 3
        assert all(c.chunk_id == 1 for c in chunks)

    def test_get_chunks_batch_boundary(self, sample_store: VstashStore) -> None:
        """Batch logic handles >900 IDs across multiple batches."""
        dim = sample_store.embedding_dim
        # Add enough chunks to test batching (we only need IDs, not 900+ real chunks)
        sample_store.add_document(
            path="/test/big.md",
            title="Big",
            chunks=["chunk"] * 5,
            embeddings=[[0.1] * dim] * 5,
            source_type="markdown",
        )
        # Request 950 IDs — most won't exist, but the batching code path runs
        ids = list(range(1, 951))
        chunks = sample_store.get_chunks(ids)
        assert len(chunks) == 5  # only the 5 real chunks


class TestGetChunkEdgeCases:
    """Edge case tests."""

    def test_get_chunk_negative_id(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunk(-1) is None

    def test_get_chunk_zero_id(self, populated_store: VstashStore) -> None:
        assert populated_store.get_chunk(0) is None


class TestGetChunkSDK:
    """Test get_chunk via Memory SDK."""

    def test_sdk_get_chunk(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            mem.add("tests/conftest.py")
            results = mem.search("fixture")
            assert len(results) > 0
            chunk_id = results[0].chunk_id
            chunk = mem.get_chunk(chunk_id)
            assert chunk is not None
            assert isinstance(chunk, ChunkInfo)
            assert chunk.chunk_id == chunk_id

    def test_sdk_get_chunk_not_found(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            assert mem.get_chunk(9999) is None

    def test_sdk_get_chunks_batch(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            mem.add("tests/conftest.py")
            results = mem.search("fixture", top_k=3)
            ids = [r.chunk_id for r in results]
            chunks = mem.get_chunks(ids)
            assert len(chunks) == len(ids)
            for chunk, expected_id in zip(chunks, ids):
                assert chunk.chunk_id == expected_id


class TestGetDocumentChunksSDK:
    """Test get_document_chunks via Memory SDK."""

    def test_sdk_get_document_chunks(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            mem.add("tests/conftest.py")
            chunks = mem.get_document_chunks("tests/conftest.py")
            assert len(chunks) > 0
            assert all(isinstance(c, str) for c in chunks)

    def test_sdk_get_document_chunks_not_found(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            assert mem.get_document_chunks("/nonexistent.md") == []

    def test_sdk_get_document_chunks_respects_collection(self, tmp_db_path: str) -> None:
        from vstash import Memory

        with Memory(db=tmp_db_path, collection="notes") as mem:
            mem.add("tests/conftest.py")
            chunks = mem.get_document_chunks("tests/conftest.py")
            assert len(chunks) > 0

        # Different collection should not find it
        with Memory(db=tmp_db_path, collection="other") as mem:
            chunks = mem.get_document_chunks("tests/conftest.py")
            assert chunks == []

    def test_sdk_get_document_chunks_text_path(self, tmp_db_path: str) -> None:
        """text:// paths from remember() should be retrievable without normalization."""
        from vstash import Memory

        with Memory(db=tmp_db_path) as mem:
            mem.remember("Hello world, this is a test note.", title="my note")
            chunks = mem.get_document_chunks("text://my note")
            assert len(chunks) > 0
            assert "Hello world" in chunks[0]
