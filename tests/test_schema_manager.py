"""Focused tests for ``_SchemaManagerMixin`` (the #280 schema concern).

``test_schema_versioning.py`` covers the v1->v2 migration in depth; this module
exercises the rest of the mixin's surface — ``store_meta`` key/value access and
``check_embedding_drift`` — and pins the mixin into ``VstashStore``'s MRO.
"""

from __future__ import annotations

from vstash.store import VstashStore
from vstash.store._schema import _SchemaManagerMixin


def test_schema_mixin_in_mro() -> None:
    assert _SchemaManagerMixin in VstashStore.__mro__


def test_meta_round_trip(sample_store: VstashStore) -> None:
    assert sample_store.get_meta("does-not-exist") is None
    sample_store.set_meta("custom_key", "value-1")
    assert sample_store.get_meta("custom_key") == "value-1"
    # set_meta upserts.
    sample_store.set_meta("custom_key", "value-2")
    assert sample_store.get_meta("custom_key") == "value-2"


def test_check_embedding_drift_claims_then_detects(sample_store: VstashStore) -> None:
    # Establish a known claimed model.
    sample_store.set_meta("embedding_model", "model-A")
    # Same model -> no drift.
    assert sample_store.check_embedding_drift("model-A") is None
    # Different model -> a mismatch warning that names both models.
    warning = sample_store.check_embedding_drift("model-B")
    assert warning is not None
    assert "model-A" in warning
    assert "model-B" in warning


def test_check_embedding_drift_first_open_is_silent(sample_store: VstashStore) -> None:
    # Clear any model meta so this exercises the first-open claim path.
    sample_store._conn.execute("DELETE FROM store_meta WHERE key = 'embedding_model'")
    sample_store._conn.commit()
    assert sample_store.check_embedding_drift("fresh-model") is None
    # The claim was persisted, so a re-check with the same model stays silent.
    assert sample_store.check_embedding_drift("fresh-model") is None
