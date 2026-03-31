"""Tests for the TurboQuant ANN backend in VstashStore.

Covers:
- add / search / delete coherence between SQLite and .tqvs
- persistence round-trip (save → reload → search)
- dim/bits mismatch detection on load
- post-filtering correctness (collection filter)
- atomic save (no .tqvs.tmp left on disk after write)
- fallback to sqlite-vec when backend is not "turboquant"
- O(1) delete via _id_to_pos dict
"""
from __future__ import annotations

import numpy as np
import pytest

from vstash.store import VstashStore


DIM = 32  # small dim for fast tests
BITS = 4
RNG = np.random.default_rng(0)


def _rand_embs(n: int, dim: int = DIM) -> list[list[float]]:
    vecs = RNG.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.tolist()


def _store(tmp_path, backend: str = "turboquant") -> VstashStore:
    return VstashStore(
        str(tmp_path / "mem.db"),
        embedding_dim=DIM,
        vector_backend=backend,
        turboquant_bits=BITS,
    )


# ------------------------------------------------------------------ #
# Basic coherence                                                     #
# ------------------------------------------------------------------ #

class TestTurboQuantBasic:
    def test_add_increases_tq_size(self, tmp_path):
        store = _store(tmp_path)
        assert store._tq_index is not None
        assert len(store._tq_index) == 0

        embs = _rand_embs(5)
        store.add_document("/doc.txt", "Doc", [f"chunk {i}" for i in range(5)], embs)
        assert len(store._tq_index) == 5
        store.close()

    def test_tq_size_matches_sqlite_chunks(self, tmp_path):
        store = _store(tmp_path)
        for i in range(3):
            store.add_document(f"/doc{i}.txt", f"Doc{i}", [f"chunk {i}"], _rand_embs(1))
        db_count = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert len(store._tq_index) == db_count == 3
        store.close()

    def test_search_returns_results(self, tmp_path):
        store = _store(tmp_path)
        embs = _rand_embs(10)
        store.add_document("/doc.txt", "Doc", [f"item {i}" for i in range(10)], embs)
        results = store.search(embs[0], "item", top_k=3)
        assert len(results) > 0
        store.close()

    def test_vec_chunks_not_populated_for_tq(self, tmp_path):
        """vec_chunks must stay empty when TurboQuant is active (size optimisation)."""
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1))
        count = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        assert count == 0, "vec_chunks should be skipped when TurboQuant is active"
        store.close()

    def test_vec_chunks_populated_for_sqlite_vec(self, tmp_path):
        """sqlite-vec backend must still populate vec_chunks."""
        store = _store(tmp_path, backend="sqlite-vec")
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1))
        count = store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        assert count == 1
        store.close()


# ------------------------------------------------------------------ #
# Delete coherence                                                    #
# ------------------------------------------------------------------ #

class TestTurboQuantDelete:
    def test_delete_reduces_tq_size(self, tmp_path):
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["a", "b", "c"], _rand_embs(3))
        assert len(store._tq_index) == 3
        store.delete_document("/doc.txt")
        assert len(store._tq_index) == 0
        store.close()

    def test_delete_keeps_db_and_tq_in_sync(self, tmp_path):
        store = _store(tmp_path)
        store.add_document("/a.txt", "A", ["chunk a"], _rand_embs(1))
        store.add_document("/b.txt", "B", ["chunk b"], _rand_embs(1))
        store.delete_document("/a.txt")
        db_count = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert len(store._tq_index) == db_count == 1
        store.close()

    def test_o1_delete_via_id_to_pos(self, tmp_path):
        """_id_to_pos dict enables O(1) id lookup (no list.index scan)."""
        store = _store(tmp_path)
        n = 50
        embs = _rand_embs(n)
        store.add_document("/doc.txt", "Doc", [f"c{i}" for i in range(n)], embs)
        tq = store._tq_index
        # Every id must appear in _id_to_pos with the correct position
        for pos, id_val in enumerate(tq._ids):
            assert tq._id_to_pos[id_val] == pos
        # After delete, positions must be compacted
        del_id = tq._ids[10]
        tq.delete(del_id)
        assert del_id not in tq._id_to_pos
        for pos, id_val in enumerate(tq._ids):
            assert tq._id_to_pos[id_val] == pos, "positions must be contiguous after delete"
        store.close()


# ------------------------------------------------------------------ #
# Persistence round-trip                                             #
# ------------------------------------------------------------------ #

class TestTurboQuantPersistence:
    def test_reload_preserves_count(self, tmp_path):
        store = _store(tmp_path)
        embs = _rand_embs(8)
        store.add_document("/doc.txt", "Doc", [f"c{i}" for i in range(8)], embs)
        store.close()

        store2 = _store(tmp_path)
        assert len(store2._tq_index) == 8
        store2.close()

    def test_reload_returns_correct_results(self, tmp_path):
        embs = _rand_embs(20)
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", [f"item {i}" for i in range(20)], embs)
        store.close()

        store2 = _store(tmp_path)
        results = store2.search(embs[5], "item", top_k=3)
        assert any("item 5" in r.text for r in results)
        store2.close()

    def test_atomic_save_no_tmp_leftover(self, tmp_path):
        """No .tqvs.tmp file should remain after a successful save."""
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1))
        store.close()
        tmp_file = tmp_path / "mem.tqvs.tmp"
        assert not tmp_file.exists(), ".tqvs.tmp must be cleaned up after atomic rename"

    def test_tqvs_file_created_on_disk(self, tmp_path):
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1))
        store.close()
        assert (tmp_path / "mem.tqvs").exists()


# ------------------------------------------------------------------ #
# Validation on load                                                 #
# ------------------------------------------------------------------ #

class TestTurboQuantValidation:
    def test_dim_mismatch_raises(self, tmp_path):
        """Loading a .tqvs with wrong dim must raise ValueError."""
        # Create index with dim=32
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1, dim=DIM))
        store.close()

        # Try to open with different dim
        with pytest.raises(ValueError, match="dim mismatch"):
            VstashStore(
                str(tmp_path / "mem.db"),
                embedding_dim=DIM * 2,  # wrong
                vector_backend="turboquant",
            )

    def test_bits_mismatch_warns(self, tmp_path, caplog):
        """Loading a .tqvs with different bits emits a warning, not an error."""
        import logging
        store = _store(tmp_path)
        store.add_document("/doc.txt", "Doc", ["hello"], _rand_embs(1))
        store.close()

        with caplog.at_level(logging.WARNING):
            store2 = VstashStore(
                str(tmp_path / "mem.db"),
                embedding_dim=DIM,
                vector_backend="turboquant",
                turboquant_bits=2,  # different from saved 4-bit
            )
        assert "bits mismatch" in caplog.text.lower()
        store2.close()


# ------------------------------------------------------------------ #
# Post-filtering (collection filter)                                 #
# ------------------------------------------------------------------ #

class TestTurboQuantFiltering:
    def test_collection_filter_returns_correct_subset(self, tmp_path):
        store = _store(tmp_path)
        embs_a = _rand_embs(5)
        embs_b = _rand_embs(5)
        store.add_document("/a.txt", "A", [f"alpha {i}" for i in range(5)], embs_a,
                           collection="col_a")
        store.add_document("/b.txt", "B", [f"beta {i}" for i in range(5)], embs_b,
                           collection="col_b")

        results = store.search(embs_a[0], "alpha", top_k=5, collection="col_a")
        assert all("alpha" in r.text for r in results), \
            "Filter should restrict results to col_a only"
        store.close()

    def test_filtered_search_finds_result_with_oversample(self, tmp_path):
        """When filter is highly selective, oversampling must still surface results."""
        store = _store(tmp_path)
        # 50 docs in "noise" collection
        for i in range(50):
            store.add_document(f"/noise{i}.txt", f"Noise{i}", [f"noise {i}"],
                               _rand_embs(1), collection="noise")
        # 1 doc in target collection
        target_emb = _rand_embs(1)
        store.add_document("/target.txt", "Target", ["specific content"],
                           target_emb, collection="rare")

        results = store.search(target_emb[0], "specific", top_k=3, collection="rare")
        assert len(results) >= 1, "Must find the one doc in rare collection"
        assert results[0].text == "specific content"
        store.close()
