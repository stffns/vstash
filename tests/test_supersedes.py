"""``supersedes``: a newer document hides the one it replaces from search.

Covers the store primitives (``supersede`` / ``unsupersede``, validation,
cycles, collection scoping), the search filter on every branch (hybrid,
vec_only, fts_only) and its interaction with the query cache, the automatic
restore when the newer document is deleted, the additive migration, and the
SDK / MCP / ingest surfaces.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import CacheConfig, VstashConfig
from vstash.store import VstashStore

DIM = 384


def _vec(cos: float, axis: int) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cos
    v[axis] = math.sqrt(1.0 - cos * cos)
    return v.tolist()


Q = _vec(1.0, 1)
OLD, NEW, OTHER = "text://modelo v1", "text://modelo v2", "text://clima"


@pytest.fixture
def store(tmp_path: Path):
    s = VstashStore(
        str(tmp_path / "sup.db"), embedding_dim=DIM, cache=CacheConfig(query_cache_size=16)
    )
    for path, text, emb in (
        (OLD, "decision modelo embeddings bge small", _vec(0.95, 2)),
        (NEW, "decision modelo embeddings jina nano", _vec(0.94, 3)),
        (OTHER, "decision clima bogota lluvia", _vec(0.90, 4)),
    ):
        s.add_document(
            path=path,
            title=path.removeprefix("text://"),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )
    yield s
    s.close()


def _paths(store: VstashStore, **kw) -> list[str]:
    return [r.path for r in store.search(Q, "decision modelo", top_k=5, **kw)]


@requires_sqlite_vec
class TestStorePrimitives:
    def test_no_supersede_no_filter(self, store) -> None:
        assert not store.has_active_supersedes()
        assert set(_paths(store)) == {OLD, NEW, OTHER}

    def test_supersede_hides_old(self, store) -> None:
        out = store.supersede(OLD, NEW)
        assert out == {"superseded": [OLD], "not_found": []}
        assert store.has_active_supersedes()
        assert OLD not in _paths(store)
        assert NEW in _paths(store)

    @pytest.mark.parametrize("mode", ["hybrid", "vec_only", "fts_only"])
    def test_hidden_on_every_branch(self, store, mode) -> None:
        store.supersede(OLD, NEW)
        assert OLD not in _paths(store, retrieval_mode=mode)

    def test_include_superseded_marks_it(self, store) -> None:
        store.supersede(OLD, NEW)
        res = {
            r.path: r for r in store.search(Q, "decision modelo", top_k=5, include_superseded=True)
        }
        assert res[OLD].superseded_by == NEW
        assert res[NEW].superseded_by is None

    def test_deleting_new_restores_old(self, store) -> None:
        store.supersede(OLD, NEW)
        store.delete_document(NEW)
        assert not store.has_active_supersedes()
        assert OLD in _paths(store)

    def test_unsupersede(self, store) -> None:
        store.supersede(OLD, NEW)
        assert store.unsupersede(OLD) is True
        assert store.unsupersede(OLD) is False
        assert OLD in _paths(store)

    def test_not_found_reported(self, store) -> None:
        out = store.supersede([OLD, "text://nope"], NEW)
        assert out == {"superseded": [OLD], "not_found": ["text://nope"]}

    def test_missing_new_rejected(self, store) -> None:
        with pytest.raises(ValueError, match="not found"):
            store.supersede(OLD, "text://ghost")

    def test_self_rejected(self, store) -> None:
        with pytest.raises(ValueError, match="itself"):
            store.supersede(NEW, NEW)

    def test_cycle_rejected(self, store) -> None:
        store.supersede(OLD, NEW)
        with pytest.raises(ValueError, match="cycle"):
            store.supersede(NEW, OLD)
        store.supersede(NEW, OTHER)  # chain OLD -> NEW -> OTHER is fine
        with pytest.raises(ValueError, match="cycle"):
            store.supersede(OTHER, OLD)

    def test_collection_scoped(self, store) -> None:
        store.add_document(
            path=OLD,
            title="copia",
            chunks=["decision modelo embeddings bge small copia"],
            embeddings=[_vec(0.95, 5)],
            source_type="text",
            collection="otra",
        )
        store.supersede(OLD, NEW)  # default collection only
        assert OLD in _paths(store, collection="otra")
        assert OLD not in _paths(store, collection="default")

    def test_query_cache_invalidated(self, store) -> None:
        assert OLD in _paths(store)  # cached
        store.supersede(OLD, NEW)
        assert OLD not in _paths(store)
        store.unsupersede(OLD)
        assert OLD in _paths(store)

    def test_include_flag_in_cache_key(self, store) -> None:
        store.supersede(OLD, NEW)
        assert OLD not in _paths(store)
        assert OLD in _paths(store, include_superseded=True)
        assert OLD not in _paths(store)


@requires_sqlite_vec
def test_migration_adds_columns(tmp_path: Path) -> None:
    db = str(tmp_path / "legacy.db")
    VstashStore(db, embedding_dim=DIM).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX idx_documents_superseded")
    conn.execute("ALTER TABLE documents DROP COLUMN superseded_by")
    conn.execute("ALTER TABLE documents DROP COLUMN superseded_at")
    conn.commit()
    conn.close()
    s = VstashStore(db, embedding_dim=DIM)
    try:
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(documents)")}
        assert {"superseded_by", "superseded_at"} <= cols
    finally:
        s.close()


@requires_sqlite_vec
class TestSurfaces:
    @pytest.fixture
    def embeds(self):
        def batch(texts, _model):
            return [_vec(0.95 - 0.01 * (i % 5), 2 + (i % 50)) for i, _ in enumerate(texts)]

        with (
            patch("vstash.ingest.embed_texts", side_effect=batch),
            patch("vstash.services.search.embed_query", return_value=Q),
        ):
            yield

    def test_memory_remember_supersedes(self, tmp_path: Path, embeds) -> None:
        from vstash import Memory

        mem = Memory(db=tmp_path / "m.db")
        try:
            mem.remember("usamos bge small para la memoria de kakesu", title="modelo v1")
            res = mem.remember(
                "cambiamos a jina nano para la memoria de kakesu",
                title="modelo v2",
                supersedes="text://modelo v1",
            )
            assert res.superseded == ["text://modelo v1"]
            paths = [r.path for r in mem.search("memoria de kakesu")]
            assert "text://modelo v1" not in paths
            hist = mem.search("memoria de kakesu", include_superseded=True)
            assert {r.path: r.superseded_by for r in hist}["text://modelo v1"] == "text://modelo v2"
            assert mem.unsupersede("text://modelo v1") is True
        finally:
            mem.close()

    def test_remember_reports_not_found(self, tmp_path: Path, embeds) -> None:
        from vstash import Memory

        mem = Memory(db=tmp_path / "n.db")
        try:
            res = mem.remember("nota nueva sobre algo", title="n", supersedes="text://nada")
            assert res.superseded == []
            assert res.supersede_not_found == ["text://nada"]
        finally:
            mem.close()

    def test_mcp_tools(self, store, embeds) -> None:
        from vstash.mcp import vstash_search, vstash_supersede

        with (
            patch("vstash.mcp._get_store", return_value=store),
            patch("vstash.mcp._get_config", return_value=VstashConfig()),
        ):
            out = json.loads(vstash_supersede(OLD, NEW))
            assert out["superseded"] == [OLD]
            hidden = json.loads(vstash_search("decision modelo"))
            assert OLD not in [c["path"] for c in hidden["chunks"]]
            shown = json.loads(vstash_search("decision modelo", include_superseded=True))
            by = {c["path"]: c.get("superseded_by") for c in shown["chunks"]}
            assert by[OLD] == NEW
            assert json.loads(vstash_supersede([OLD], "", undo=True))["restored"] == [OLD]
            err = json.loads(vstash_supersede(OLD, "text://ghost"))
            assert "error" in err
