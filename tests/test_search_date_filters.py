"""#384: ``added_after`` / ``added_before`` are UTC-canonicalised before the
lexical ``added_at`` comparison used by search and ``list_documents``.

``added_at`` is stored as UTC ISO (``...+00:00``) and SQLite compares the column
lexically, so a non-UTC offset would sort out of chronological order and return
the wrong rows. The store now canonicalises the inputs (and rejects non-ISO).
"""

from __future__ import annotations

import pytest

from vstash.store._common import _canonicalize_added_filter


class TestCanonicalizeAddedFilter:
    def test_date_only_becomes_utc_midnight(self) -> None:
        assert _canonicalize_added_filter("2026-01-15", label="x") == "2026-01-15T00:00:00+00:00"

    def test_naive_timestamp_treated_as_utc(self) -> None:
        assert (
            _canonicalize_added_filter("2026-01-15T08:30:00", label="x")
            == "2026-01-15T08:30:00+00:00"
        )

    def test_non_utc_offset_converted_to_utc(self) -> None:
        # 00:00 at -05:00 is 05:00 UTC; the raw string would have sorted
        # lexically before the equivalent UTC instant (the #384 bug).
        assert (
            _canonicalize_added_filter("2026-01-15T00:00:00-05:00", label="x")
            == "2026-01-15T05:00:00+00:00"
        )

    def test_already_utc_is_idempotent(self) -> None:
        value = "2026-01-15T05:00:00+00:00"
        assert _canonicalize_added_filter(value, label="x") == value

    @pytest.mark.parametrize("bad", ["not-a-date", "", "2026-13-99", "30d"])
    def test_invalid_raises_with_label(self, bad: str) -> None:
        with pytest.raises(ValueError, match="added_after"):
            _canonicalize_added_filter(bad, label="added_after")


class TestSearchAcceptsDateFilters:
    def test_search_with_non_utc_offset_does_not_crash(self, sample_store) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/d.md", title="d", chunks=["alpha"], embeddings=[[0.5] * dim]
        )
        # Non-UTC offset must be accepted and canonicalised, not rejected.
        results = sample_store.search(
            [0.5] * dim, "alpha", top_k=5, added_after="2020-01-01T00:00:00-05:00"
        )
        assert isinstance(results, list)

    def test_search_rejects_garbage_date(self, sample_store) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(ValueError):
            sample_store.search([0.5] * dim, "alpha", top_k=5, added_after="not-a-date")
