"""Tests for the optional snapvec vector search backend."""

from __future__ import annotations

import numpy as np  # noqa: E402
import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

from vstash.store import VstashStore, _HAS_SNAPVEC  # noqa: E402


DIM = 32  # small dim for fast tests


@pytest.fixture
def snap_store(tmp_path):
    """Create a VstashStore with snapvec backend."""
    db = str(tmp_path / "snap_test.db")
    store = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
    yield store
    store.close()


@pytest.fixture
def populated_snap_store(snap_store):
    """Snap store with two documents."""
    # Document 1
    emb1 = [np.random.default_rng(1).standard_normal(DIM).tolist() for _ in range(3)]
    snap_store.add_document(
        path="/test/doc1.md",
        title="Document One",
        chunks=["chunk one alpha", "chunk one beta", "chunk one gamma"],
        embeddings=emb1,
        source_type="markdown",
    )
    # Document 2
    emb2 = [np.random.default_rng(2).standard_normal(DIM).tolist() for _ in range(2)]
    snap_store.add_document(
        path="/test/doc2.md",
        title="Document Two",
        chunks=["chunk two delta", "chunk two epsilon"],
        embeddings=emb2,
        source_type="markdown",
    )
    return snap_store


class TestSnapvecDetection:
    def test_has_snapvec(self):
        assert _HAS_SNAPVEC is True

    def test_snap_index_created(self, snap_store):
        assert snap_store._snap is not None
        assert snap_store._vector_backend == "snapvec"

    def test_fallback_when_sqlite_vec(self, tmp_path):
        """Default backend should not create a SnapIndex."""
        db = str(tmp_path / "fallback.db")
        store = VstashStore(db, embedding_dim=DIM)
        assert store._snap is None
        assert store._vector_backend == "sqlite-vec"
        store.close()


class TestSnapvecAdd:
    def test_add_populates_snap_index(self, snap_store):
        emb = [np.random.default_rng(42).standard_normal(DIM).tolist()]
        snap_store.add_document(
            path="/test/single.md",
            title="Single",
            chunks=["hello world"],
            embeddings=emb,
        )
        assert len(snap_store._snap) == 1

    def test_add_multiple_chunks(self, populated_snap_store):
        assert len(populated_snap_store._snap) == 5  # 3 + 2


class TestSnapvecSearch:
    def test_search_returns_results(self, populated_snap_store):
        query_emb = np.random.default_rng(1).standard_normal(DIM).tolist()
        results = populated_snap_store.search(
            query_embedding=query_emb,
            query_text="chunk alpha",
            top_k=3,
        )
        assert len(results) > 0
        assert all(r.text for r in results)

    def test_search_respects_top_k(self, populated_snap_store):
        query_emb = np.random.default_rng(1).standard_normal(DIM).tolist()
        results = populated_snap_store.search(
            query_embedding=query_emb,
            query_text="chunk",
            top_k=2,
        )
        assert len(results) <= 2


class TestSnapvecDelete:
    def test_delete_removes_from_snap_index(self, populated_snap_store):
        initial_count = len(populated_snap_store._snap)
        populated_snap_store.delete_document("/test/doc1.md")
        assert len(populated_snap_store._snap) == initial_count - 3

    def test_delete_nonexistent(self, populated_snap_store):
        count_before = len(populated_snap_store._snap)
        result = populated_snap_store.delete_document("/nonexistent.md")
        assert result is False
        assert len(populated_snap_store._snap) == count_before


class TestSnapvecPersistence:
    def test_save_and_reload(self, tmp_path):
        """SnapIndex should persist and reload across store instances."""
        db = str(tmp_path / "persist.db")

        # Create and populate
        store1 = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        emb = [np.random.default_rng(99).standard_normal(DIM).tolist() for _ in range(3)]
        store1.add_document(
            path="/test/persist.md",
            title="Persist",
            chunks=["a", "b", "c"],
            embeddings=emb,
        )
        store1.close()

        # Reload
        store2 = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        assert store2._snap is not None
        assert len(store2._snap) == 3

        # Search should still work
        results = store2.search(
            query_embedding=emb[0],
            query_text="a",
            top_k=3,
        )
        assert len(results) > 0
        store2.close()


class TestSnapvecReindex:
    def test_reindex_rebuilds_snap_index(self, populated_snap_store):
        def fake_embed(texts):
            rng = np.random.default_rng(0)
            return [rng.standard_normal(DIM).tolist() for _ in texts]

        count = populated_snap_store.reindex(fake_embed, new_dim=DIM)
        assert count == 5
        assert len(populated_snap_store._snap) == 5


class TestSnapvecDimMismatch:
    def test_dim_mismatch_rebuilds(self, tmp_path):
        """If saved index has wrong dim, it should rebuild gracefully."""
        db = str(tmp_path / "mismatch.db")

        # Create with dim=32
        store1 = VstashStore(db, embedding_dim=32, vector_backend="snapvec", snapvec_bits=4)
        emb = [[0.1] * 32]
        store1.add_document(path="/test/x.md", title="X", chunks=["x"], embeddings=emb)
        store1.close()

        # Reopen with dim=64 — should warn and rebuild, not crash
        store2 = VstashStore(db, embedding_dim=64, vector_backend="snapvec", snapvec_bits=4)
        assert store2._snap is not None
        assert store2._snap.dim == 64
        assert len(store2._snap) == 0  # rebuilt empty
        store2.close()
