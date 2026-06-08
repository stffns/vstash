"""Focused tests for ``_IndexBackendMixin`` (the #280 vector-index concern).

Heavy snapvec / IVFPQ behaviour is covered by ``test_snapvec_backend.py`` and
``test_snapvec_ivfpq_backend.py``; this module pins the mixin into the MRO and
exercises the pure ``_nlist_for`` helper and the sidecar-path properties.
"""

from __future__ import annotations

import math

from vstash.store import VstashStore
from vstash.store._index import _IndexBackendMixin


def test_index_mixin_in_mro() -> None:
    assert _IndexBackendMixin in VstashStore.__mro__


def test_nlist_for_clamps_and_is_monotonic() -> None:
    # n <= 0 is a placeholder; the real nlist is derived at fit() time.
    assert VstashStore._nlist_for(0) == 256
    # FAISS rule of thumb 4*sqrt(N), clamped to [8, 1024].
    assert VstashStore._nlist_for(1) == 8  # 4*1 -> clamped up to the floor
    assert VstashStore._nlist_for(100) == int(4 * math.sqrt(100))
    assert VstashStore._nlist_for(10**9) == 1024  # clamped down to the ceiling
    positive = [VstashStore._nlist_for(n) for n in (1, 100, 1_000, 50_000, 10**9)]
    assert positive == sorted(positive)
    assert all(8 <= v <= 1024 for v in positive)


def test_index_paths_are_db_sidecars(sample_store: VstashStore) -> None:
    db = sample_store.db_path
    assert sample_store._snapvec_path == db.with_suffix(".snpv")
    assert sample_store._ivfpq_path == db.with_suffix(".snpi")
    assert sample_store._snapvec_path != sample_store._ivfpq_path
