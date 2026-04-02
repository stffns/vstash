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
        assert chunk.seq == 0
        assert chunk.text == "Python is a high-level programming language known for its simplicity."
        assert chunk.title == "Python Guide"
        assert chunk.path == "/test/python_guide.md"

    def test_get_chunk_second_document(self, populated_store: VstashStore) -> None:
        chunk = populated_store.get_chunk(3)
        assert chunk is not None
        assert chunk.title == "ML Introduction"
        assert chunk.path == "/test/ml_intro.pdf"
        assert chunk.seq == 0

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
