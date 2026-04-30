"""Tests for the snapvec-ivfpq backend adapter and VstashStore integration."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

from vstash.store import VstashStore
from vstash.vectorbackend import IVFPQBackend

DIM = 96  # multiple of ivfpq_M so no param juggling is needed
N = 400  # big enough for the default nlist without overfitting warnings


def _unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


class TestIVFPQBackendWrapper:
    def test_pre_fit_is_empty_and_search_returns_nothing(self):
        be = IVFPQBackend(dim=DIM, nlist=16, M=24, K=32)
        assert len(be) == 0
        assert not be.fitted
        assert be.search(np.zeros(DIM, dtype=np.float32), k=5) == []

    def test_pre_fit_add_batch_is_no_op(self):
        be = IVFPQBackend(dim=DIM, nlist=16, M=24, K=32)
        be.add_batch([1, 2, 3], _unit_vectors(np.random.default_rng(0), 3, DIM))
        assert len(be) == 0

    def test_fit_then_add_then_search(self, tmp_path):
        rng = np.random.default_rng(0)
        vecs = _unit_vectors(rng, N, DIM)
        ids = list(range(N))
        be = IVFPQBackend(dim=DIM, nlist=16, M=24, K=32, rerank_candidates=32)
        be.fit(vecs)
        be.add_batch(ids, vecs)
        assert len(be) == N
        # First vector should find itself at rank 0
        hits = be.search(vecs[0], k=5)
        assert hits, "expected non-empty results post-fit"
        assert hits[0][0] == 0
        # Distances should be in [0, 2] cosine-like range
        assert all(0.0 <= d <= 2.0 for _, d in hits)

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(1)
        vecs = _unit_vectors(rng, N, DIM)
        ids = list(range(N))
        path = str(tmp_path / "idx.snpi")
        be = IVFPQBackend(dim=DIM, nlist=16, M=24, K=32, rerank_candidates=32)
        be.fit(vecs)
        be.add_batch(ids, vecs)
        be.save(path)

        be2 = IVFPQBackend.load(path, dim=DIM, nlist=16, M=24, K=32, rerank_candidates=32)
        assert be2.fitted
        assert len(be2) == N
        hits = be2.search(vecs[3], k=5)
        assert hits[0][0] == 3

    def test_load_missing_file_returns_unfit(self, tmp_path):
        missing = str(tmp_path / "nope.snpi")
        be = IVFPQBackend.load(missing, dim=DIM, nlist=16, M=24, K=32)
        assert not be.fitted
        assert len(be) == 0

    def test_load_treats_file_existence_as_authoritative(self, tmp_path):
        """A fitted .snpi survives a missing sidecar (crash resilience)."""
        from pathlib import Path as _P

        rng = np.random.default_rng(5)
        vecs = _unit_vectors(rng, N, DIM)
        path = str(tmp_path / "idx.snpi")
        be = IVFPQBackend(dim=DIM, nlist=16, M=24, K=32, rerank_candidates=32)
        be.fit(vecs)
        be.add_batch(list(range(N)), vecs)
        be.save(path)
        # Previously the loader required a .fitted sidecar; deleting it
        # must not downgrade load() to the unfit path.
        sidecar = _P(path + ".fitted")
        if sidecar.exists():
            sidecar.unlink()
        loaded = IVFPQBackend.load(path, dim=DIM, nlist=16, M=24, K=32)
        assert loaded.fitted
        assert len(loaded) == N

    def test_constructor_rejects_invalid_dim_M_combo(self):
        with pytest.raises(ValueError, match="must divide embedding_dim"):
            IVFPQBackend(dim=384, nlist=16, M=10, K=32)  # 384 % 10 != 0


class TestVstashStoreIVFPQIntegration:
    def test_backend_initialization(self, tmp_path):
        store = VstashStore(
            str(tmp_path / "s.db"),
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
            ivfpq_rerank_candidates=32,
        )
        assert store._vector_backend == "snapvec-ivfpq"
        assert isinstance(store._snap, IVFPQBackend)
        assert not store._snap.fitted
        store.close()

    def test_ingest_falls_through_to_sqlite_vec_pre_fit(self, tmp_path):
        store = VstashStore(
            str(tmp_path / "s.db"),
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        rng = np.random.default_rng(2)
        vecs = _unit_vectors(rng, 5, DIM)
        store.add_document(
            path="/t/a.md",
            title="A",
            chunks=[f"chunk {i}" for i in range(5)],
            embeddings=vecs.tolist(),
            source_type="text",
        )
        # sqlite-vec still answers pre-fit
        results = store.search(vecs[0].tolist(), "chunk 0", top_k=3)
        assert len(results) > 0
        store.close()

    def test_fit_ivfpq_end_to_end(self, tmp_path):
        store = VstashStore(
            str(tmp_path / "s.db"),
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
            ivfpq_rerank_candidates=32,
        )
        rng = np.random.default_rng(3)
        vecs = _unit_vectors(rng, N, DIM)
        # Ingest as a single batch for speed
        store.add_document(
            path="/t/bulk.md",
            title="Bulk",
            chunks=[f"chunk {i}" for i in range(N)],
            embeddings=vecs.tolist(),
            source_type="text",
        )
        stats = store.fit_ivfpq(training_sample=N)
        assert stats["n_indexed"] == N
        assert store._snap.fitted

        # Post-fit search should now go through IVFPQ
        results = store.search(vecs[0].tolist(), "chunk 0", top_k=5)
        assert len(results) > 0
        # Index file should exist
        assert store._ivfpq_path.exists()
        store.close()

    def test_ivfpq_last_best_distance_in_cosine_range(self, tmp_path):
        """Regression test for #289: the per-backend branch in
        ``VstashStore.search`` must not double-invert ``IVFPQBackend
        .search`` output.  ``IVFPQBackend`` already returns cosine
        distance in ``[0, 2]`` (see ``vectorbackend/snapvec_ivfpq.py``), so a same-
        vector query under ``snapvec-ivfpq`` must land in the
        ``"high"`` tier, not the ``"low"`` tier that double-inversion
        would produce.
        """
        from vstash.store import RELEVANCE_TIER_HIGH_MAX, relevance_tier

        store = VstashStore(
            str(tmp_path / "ivfpq_dist.db"),
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
            ivfpq_rerank_candidates=32,
        )
        rng = np.random.default_rng(5)
        vecs = _unit_vectors(rng, N, DIM)
        store.add_document(
            path="/t/dist.md",
            title="Dist",
            chunks=[f"chunk {i}" for i in range(N)],
            embeddings=vecs.tolist(),
            source_type="text",
        )
        store.fit_ivfpq(training_sample=N)

        store.search(vecs[0].tolist(), "chunk 0", top_k=1)
        assert 0.0 <= store.last_best_distance <= 2.0
        assert store.last_best_distance <= RELEVANCE_TIER_HIGH_MAX
        assert relevance_tier(store.last_best_distance) == "high"
        store.close()

    def test_fit_ivfpq_rejects_non_ivfpq_backend(self, tmp_path):
        store = VstashStore(
            str(tmp_path / "s.db"),
            embedding_dim=DIM,
            vector_backend="sqlite-vec",
        )
        with pytest.raises(RuntimeError, match="snapvec-ivfpq"):
            store.fit_ivfpq()
        store.close()

    def test_fit_ivfpq_rejects_empty_corpus(self, tmp_path):
        store = VstashStore(
            str(tmp_path / "s.db"),
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        with pytest.raises(RuntimeError, match="empty"):
            store.fit_ivfpq()
        store.close()

    def test_reopen_after_fit_loads_index(self, tmp_path):
        db_path = str(tmp_path / "reopen.db")
        store = VstashStore(
            db_path,
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        rng = np.random.default_rng(4)
        vecs = _unit_vectors(rng, N, DIM)
        store.add_document(
            path="/t/bulk.md",
            title="Bulk",
            chunks=[f"chunk {i}" for i in range(N)],
            embeddings=vecs.tolist(),
            source_type="text",
        )
        store.fit_ivfpq(training_sample=N)
        store.close()

        store2 = VstashStore(
            db_path,
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        assert store2._snap.fitted
        assert len(store2._snap) == N
        store2.close()

    def test_stale_ivfpq_index_downgrades_to_unfitted(self, tmp_path, caplog):
        """Issue #329: a crash post-fit can leave .snpi behind vec_chunks.

        Reopen must detect that staleness and downgrade the backend to
        its unfitted state so search falls back to sqlite-vec instead of
        silently missing the rows ingested in the last write burst. The
        repro shape:

        1. Open store, ingest + fit, close (so .snpi flushes).
        2. Reopen, ingest more rows. These land in vec_chunks AND in
           the in-memory IVFPQ index, but .snpi is NOT flushed.
        3. Force-close the second store WITHOUT calling close() to
           simulate a process crash. Use the underlying ``_conn.close``
           and drop the snap reference -- this leaves .snpi at its
           v1 size while vec_chunks has the new rows.
        4. Reopen a third time. The init path must:
           - log a warning about the staleness,
           - leave ``self._snap.fitted == False``,
           - leave ``len(self._snap) == 0`` so the search code path
             falls back to sqlite-vec via the ``len(self._snap) > 0``
             gate in VstashStore.search.
        """
        import logging

        db_path = str(tmp_path / "stale.db")

        # 1. Initial fit + save.
        store1 = VstashStore(
            db_path,
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        rng = np.random.default_rng(7)
        v1 = _unit_vectors(rng, N, DIM)
        store1.add_document(
            path="/t/initial.md",
            title="initial",
            chunks=[f"c{i}" for i in range(N)],
            embeddings=v1.tolist(),
            source_type="text",
        )
        store1.fit_ivfpq(training_sample=N)
        store1.close()  # .snpi now has N rows.

        # 2. Reopen and add more rows; do NOT close (crash sim).
        store2 = VstashStore(
            db_path,
            embedding_dim=DIM,
            vector_backend="snapvec-ivfpq",
            ivfpq_M=24,
            ivfpq_K=32,
            ivfpq_nlist=16,
        )
        assert store2._snap.fitted
        assert len(store2._snap) == N
        v2 = _unit_vectors(rng, 50, DIM)
        store2.add_document(
            path="/t/burst.md",
            title="burst",
            chunks=[f"burst_c{i}" for i in range(50)],
            embeddings=v2.tolist(),
            source_type="text",
        )
        # vec_chunks now has N+50 rows; .snpi is still at N because
        # _save_snapvec writes only on close()/checkpoint and we are
        # simulating a crash. Drop the snap reference + close conn
        # without calling store.close() to avoid the flush.
        store2._snap = None
        store2._conn.close()

        # 3. Reopen. Staleness detection must fire.
        with caplog.at_level(logging.WARNING, logger="vstash.store"):
            store3 = VstashStore(
                db_path,
                embedding_dim=DIM,
                vector_backend="snapvec-ivfpq",
                ivfpq_M=24,
                ivfpq_K=32,
                ivfpq_nlist=16,
            )

        assert any("stale" in r.message.lower() for r in caplog.records), (
            f"staleness warning not emitted; got: {[r.message for r in caplog.records]}"
        )
        # The downgrade contract: unfitted -> search falls back to sqlite-vec.
        assert not store3._snap.fitted
        assert len(store3._snap) == 0
        # Sanity: search still works (via sqlite-vec fallback).
        results = store3.search(v1[0].tolist(), "c0", top_k=3)
        assert len(results) > 0
        store3.close()
