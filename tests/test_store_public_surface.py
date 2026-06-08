"""Locks the public + re-export surface of ``vstash.store`` (#280 split).

``vstash/store.py`` is being decomposed into a package of focused submodules
(``_common``, ``_schema``, ``_index``, ``_search``, ``_integrity``) behind a thin
``VstashStore`` facade. These tests guarantee that, however the internals are
reorganised, the symbols other modules and tests import directly from
``vstash.store`` keep resolving. A later extraction PR that drops a re-export
fails here instead of breaking a distant import at runtime.
"""

from __future__ import annotations

import importlib

import pytest

# Every name that MUST stay importable from ``vstash.store``, sourced from a
# grep of ``from vstash.store import X`` across vstash/ and tests/. The
# underscore-prefixed entries are part of the stable internal surface.
_PUBLIC_SURFACE = [
    "VstashStore",
    "SchemaVersionError",
    "SCHEMA_VERSION",
    "KNOWN_SCHEMA_VERSIONS",
    "relevance_tier",
    "RELEVANCE_TIER_HIGH_MAX",
    "RELEVANCE_TIER_MEDIUM_MAX",
    "RRF_K",
    "_PipelineTracer",
    "_cosine_sim",
    "_serialize",
    "_deserialize",
    "_normalize_tags",
    "_SQLITE_PARAM_BATCH",
    "_HAS_SNAPVEC",
]


@pytest.mark.parametrize("name", _PUBLIC_SURFACE)
def test_symbol_importable_from_vstash_store(name: str) -> None:
    mod = importlib.import_module("vstash.store")
    assert hasattr(mod, name), f"vstash.store dropped its re-export of {name!r}"


def test_from_import_form_resolves() -> None:
    """The literal ``from vstash.store import ...`` form used across the codebase."""
    from vstash.store import (
        RRF_K,
        SchemaVersionError,
        VstashStore,
        _cosine_sim,
        _normalize_tags,
        relevance_tier,
    )

    assert isinstance(RRF_K, int)
    assert issubclass(SchemaVersionError, Exception)
    assert isinstance(VstashStore, type)
    assert callable(_cosine_sim)
    assert callable(_normalize_tags)
    assert relevance_tier(0.0) == "high"


def test_store_is_a_package() -> None:
    """The #280 split converts the module into a package without changing its path."""
    import vstash.store

    # __path__ only exists on packages; confirms store.py became store/__init__.py.
    assert hasattr(vstash.store, "__path__")
    # The leaf helper module is reachable and self-contained.
    from vstash.store._common import _cosine_sim

    assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
