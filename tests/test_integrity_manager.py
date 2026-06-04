"""Focused tests for ``_IntegrityMixin`` (the #280 integrity concern).

Full invariant + repair behaviour is covered by ``test_integrity.py``; this
pins the mixin into the MRO and confirms a freshly populated store reports a
clean bill of health.
"""

from __future__ import annotations

from vstash.store import VstashStore
from vstash.store._integrity import _IntegrityMixin


def test_integrity_mixin_in_mro() -> None:
    assert _IntegrityMixin in VstashStore.__mro__


def test_clean_store_passes_all_invariants(populated_store: VstashStore) -> None:
    checks = populated_store.integrity_check()
    assert checks, "integrity_check should return the list of invariants"
    failed = [c.name for c in checks if not c.passed]
    assert not failed, f"freshly populated store failed invariants: {failed}"
