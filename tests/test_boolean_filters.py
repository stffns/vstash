"""#106: boolean cross-field filter expressions (AND / OR / NOT over metadata).

Covers the `_compile_filter_tree` compiler (SQL shape + validation) and an
end-to-end `store.search(filters=...)` proving OR across fields actually filters.
"""

from __future__ import annotations

import pytest

from vstash.store import VstashStore
from vstash.store._common import _compile_filter_tree


class TestCompiler:
    def test_single_field_leaf(self) -> None:
        sql, params = _compile_filter_tree({"collection": "docs"}, "d.")
        assert sql == "d.collection = ?"
        assert params == ["docs"]

    def test_or_across_fields(self) -> None:
        sql, params = _compile_filter_tree({"or": [{"collection": "a"}, {"layer": "b"}]}, "")
        assert sql == "(collection = ? OR layer = ?)"
        assert params == ["a", "b"]

    def test_and_with_nested_or(self) -> None:
        node = {"and": [{"collection": "a"}, {"or": [{"layer": "x"}, {"layer": "y"}]}]}
        sql, params = _compile_filter_tree(node, "")
        assert sql == "(collection = ? AND (layer = ? OR layer = ?))"
        assert params == ["a", "x", "y"]

    def test_not(self) -> None:
        sql, params = _compile_filter_tree({"not": {"collection": "secret"}}, "")
        assert sql == "(NOT collection = ?)"
        assert params == ["secret"]

    def test_tags_leaf_is_comma_anchored(self) -> None:
        sql, params = _compile_filter_tree({"tags": ["a", "b"]}, "")
        assert "LIKE ?" in sql
        assert params == ["%,a,%", "%,b,%"]

    def test_added_after_canonicalised_to_utc(self) -> None:
        sql, params = _compile_filter_tree({"added_after": "2026-01-15T00:00:00-05:00"}, "")
        assert sql == "added_at >= ?"
        assert params == ["2026-01-15T05:00:00+00:00"]

    def test_multi_field_leaf_is_anded(self) -> None:
        sql, params = _compile_filter_tree({"collection": "a", "layer": "b"}, "")
        assert sql == "(collection = ? AND layer = ?)"
        assert params == ["a", "b"]

    def test_none_value_compiles_to_is_null(self) -> None:
        sql, params = _compile_filter_tree({"project": None}, "d.")
        assert sql == "d.project IS NULL"
        assert params == []
        # tags None -> "untagged" (NULL or empty string), no bind params.
        sql, params = _compile_filter_tree({"tags": None}, "")
        assert sql == "(tags IS NULL OR tags = '')"
        assert params == []

    def test_none_date_value_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a date value"):
            _compile_filter_tree({"added_after": None}, "")

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown filter field"):
            _compile_filter_tree({"bogus": "x"}, "")

    def test_operator_node_must_be_single_key(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _compile_filter_tree({"and": [{"collection": "a"}], "collection": "b"}, "")

    def test_or_must_be_nonempty_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            _compile_filter_tree({"or": []}, "")

    def test_non_dict_node_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            _compile_filter_tree(["collection", "a"], "")


class TestSearchIntegration:
    def _seed(self, store: VstashStore) -> int:
        dim = store.embedding_dim
        for path, coll in (("/a.md", "work"), ("/b.md", "home"), ("/c.md", "other")):
            store.add_document(
                path=path, title=path, chunks=["alpha"], embeddings=[[0.5] * dim], collection=coll
            )
        return dim

    def test_or_matches_either_collection(self, sample_store: VstashStore) -> None:
        dim = self._seed(sample_store)
        results = sample_store.search(
            [0.5] * dim,
            "alpha",
            top_k=10,
            filters={"or": [{"collection": "work"}, {"collection": "home"}]},
        )
        assert {r.path for r in results} == {"/a.md", "/b.md"}

    def test_not_excludes(self, sample_store: VstashStore) -> None:
        dim = self._seed(sample_store)
        results = sample_store.search(
            [0.5] * dim, "alpha", top_k=10, filters={"not": {"collection": "other"}}
        )
        assert {r.path for r in results} == {"/a.md", "/b.md"}

    def test_filters_compose_with_flat_kwargs(self, sample_store: VstashStore) -> None:
        dim = self._seed(sample_store)
        # Flat collection='work' AND the OR tree -> only /a.md survives both.
        results = sample_store.search(
            [0.5] * dim,
            "alpha",
            top_k=10,
            collection="work",
            filters={"or": [{"collection": "work"}, {"collection": "home"}]},
        )
        assert {r.path for r in results} == {"/a.md"}

    def test_invalid_filter_field_raises(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(ValueError, match="unknown filter field"):
            sample_store.search([0.5] * dim, "alpha", top_k=5, filters={"nope": "x"})
