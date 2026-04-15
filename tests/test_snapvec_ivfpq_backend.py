"""Tests for the snapvec-ivfpq backend adapter and VstashStore integration."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

from vstash._ivfpq_backend import IVFPQBackend  # noqa: E402
from vstash.store import VstashStore  # noqa: E402

DIM = 96  # multiple of ivfpq_M so no param juggling is needed
N = 400   # big enough for the default nlist without overfitting warnings


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

        be2 = IVFPQBackend.load(
            path, dim=DIM, nlist=16, M=24, K=32, rerank_candidates=32
        )
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
