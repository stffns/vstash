"""Focused tests for ``_SearchEngineMixin`` (the #280 search concern).

The full hybrid pipeline is covered by ``test_store.py``, ``test_query_cache.py``,
``test_retrieval_mode.py`` and ``test_miss_analysis.py``; this pins the mixin
into the MRO and exercises the pure ``_resolve_retrieval_mode`` contract.
"""

from __future__ import annotations

import pytest

from vstash.store import VstashStore
from vstash.store._search import _SearchEngineMixin


def test_search_mixin_in_mro() -> None:
    assert _SearchEngineMixin in VstashStore.__mro__


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("hybrid", "hybrid"), ("vec_only", "vec_only"), ("fts_only", "fts_only"), (None, "hybrid")],
)
def test_resolve_retrieval_mode_canonicalises(mode: str | None, expected: str) -> None:
    assert VstashStore._resolve_retrieval_mode(mode) == expected


def test_resolve_retrieval_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        VstashStore._resolve_retrieval_mode("bogus")
