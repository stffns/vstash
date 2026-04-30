"""Tests for the vstash.vectorbackend Protocol and registry.

The Protocol is structural; existing classes that already match its
shape (notably IVFPQBackend, which is the only concrete adapter
extracted in v0.37) satisfy it without inheritance. These tests
pin three things:

1. ``vstash.vectorbackend.IVFPQBackend`` is the same class object
   as ``vstash.vectorbackend.snapvec_ivfpq.IVFPQBackend`` (the
   re-export contract).
2. ``vstash._ivfpq_backend`` still re-exports the same class with
   a DeprecationWarning (compat shim contract for v0.37 -> v0.40).
3. IVFPQBackend satisfies the runtime-checkable VectorBackend
   Protocol -- i.e. has all the required methods with the right
   signatures.

Snapvec is an optional dep; the whole module skips when it is not
installed.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

import numpy as np

from vstash.vectorbackend import IVFPQBackend, VectorBackend


class TestPackageExports:
    """Re-export contract: IVFPQBackend is one class, three import paths."""

    def test_package_exports_ivfpq_backend(self) -> None:
        from vstash.vectorbackend import IVFPQBackend as Pkg
        from vstash.vectorbackend.snapvec_ivfpq import IVFPQBackend as Mod

        assert Pkg is Mod, "package re-export drifted from module-level class"

    def test_legacy_shim_warns_and_resolves(self) -> None:
        # The shim emits a DeprecationWarning on import and resolves
        # to the same class object as the new path. Tests for the
        # warning are easier to write at module-import time, but
        # python caches modules, so we use importlib.reload to force
        # a fresh import.
        import importlib

        import vstash._ivfpq_backend

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            importlib.reload(vstash._ivfpq_backend)

        assert any(
            issubclass(w.category, DeprecationWarning) and "vstash._ivfpq_backend" in str(w.message)
            for w in captured
        ), "deprecation warning not emitted from shim"

        from vstash._ivfpq_backend import IVFPQBackend as Shim
        from vstash.vectorbackend import IVFPQBackend as Pkg

        assert Shim is Pkg, "shim re-export drifted from canonical class"


class TestProtocolConformance:
    """IVFPQBackend satisfies the VectorBackend Protocol."""

    @pytest.fixture
    def backend(self) -> IVFPQBackend:
        # M=96 divides dim=384 (the default BGE small dim).
        return IVFPQBackend(dim=384, nlist=4, M=96, K=64)

    def test_isinstance_check(self, backend: IVFPQBackend) -> None:
        # runtime_checkable Protocol -- isinstance is the contract
        # external code uses for defensive type checks.
        assert isinstance(backend, VectorBackend)

    def test_has_required_attributes(self, backend: IVFPQBackend) -> None:
        # Every protocol member must exist on the instance.
        assert hasattr(backend, "dim")
        assert backend.dim == 384

    def test_len_returns_zero_pre_fit(self, backend: IVFPQBackend) -> None:
        # Pre-fit IVFPQ reports 0 routable vectors so the store falls
        # back to sqlite-vec. This is the contract VstashStore.search
        # depends on at line 2204-ish (`elif self._snap is not None
        # and len(self._snap) > 0`).
        assert len(backend) == 0

    def test_search_pre_fit_returns_empty(self, backend: IVFPQBackend) -> None:
        query = np.random.default_rng(42).standard_normal(384).astype(np.float32)
        results = backend.search(query, k=5)
        assert results == []

    def test_add_batch_pre_fit_silently_drops(self, backend: IVFPQBackend) -> None:
        # The store dual-populates sqlite-vec + snapvec on every
        # add_document; pre-fit IVFPQ must accept the call without
        # raising or sqlite-vec writes would be rolled back too.
        rng = np.random.default_rng(42)
        ids = list(range(10))
        vectors = rng.standard_normal((10, 384)).astype(np.float32)
        # Should not raise.
        backend.add_batch(ids, vectors)
        # Still 0 routable vectors because we never fit().
        assert len(backend) == 0

    def test_delete_pre_fit_returns_false(self, backend: IVFPQBackend) -> None:
        # Pre-fit deletes are no-ops returning False so the caller
        # knows the snapvec side did not actually remove anything.
        assert backend.delete(123) is False
