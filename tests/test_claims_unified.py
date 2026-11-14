"""Claim retirement, unified with what #523 (``supersedes``) had solved.

Regression tests: a content edit must not un-retire a claim; self and
cyclic replacement are rejected before any write; ``restore`` brings a
retired claim back; ``include_retired`` shows history (and is part of the
query-cache key); retired chunks cannot starve the candidate pool; the flag
reaches federated search.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import CacheConfig, VstashConfig
from vstash.ingest import ingest_text
from vstash.store import VstashStore

DIM = 384
WORD = "memo"


def _unit(cos: float, axis: int) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cos
    v[axis] = math.sqrt(1.0 - cos * cos)
    return v.tolist()


Q = _unit(1.0, 1)


@pytest.fixture
def store(tmp_path):
    s = VstashStore(
        str(tmp_path / "u.db"), embedding_dim=DIM, cache=CacheConfig(query_cache_size=16)
    )
    yield s
    s.close()


@pytest.fixture
def cfg():
    c = MagicMock()
    c.chunking.size = 1024
    c.chunking.overlap = 0
    c.embeddings.model = "BAAI/bge-small-en-v1.5"
    return c


def _remember(store, cfg, text, title, emb=None, **kw):
    emb = emb or _unit(0.95, 2)
    with patch("vstash.ingest._embed_with_progress", return_value=[emb]):
        body = f"{text} This sentence pads the claim past the minimum chunk size. {WORD}"
        kw.setdefault("source_ref", "charla")
        return ingest_text(body, cfg, store, title=title, layer="decision", **kw)


def _paths(store, **kw) -> list[str]:
    return [r.path for r in store.search(Q, WORD, top_k=10, **kw)]


@requires_sqlite_vec
class TestStickyRetirement:
    def test_reingest_with_new_text_stays_retired(self, store, cfg) -> None:
        _remember(store, cfg, "We use bge small.", "v1")
        _remember(store, cfg, "We moved to jina nano.", "v2", supersedes="text://v1")
        res = _remember(store, cfg, "We use bge small (corrected text).", "v1")
        assert res.claim_status == "superado"  # what is stored, not what was derived
        assert "text://v1" not in _paths(store)
        row = store._conn.execute(
            "SELECT status FROM documents WHERE path = 'text://v1'"
        ).fetchone()
        assert row[0] == "superado"

    def test_memory_update_text_stays_retired(self, tmp_path: Path) -> None:
        from vstash import Memory

        with (
            patch(
                "vstash.ingest._embed_with_progress",
                side_effect=lambda c, m, **k: [_unit(0.95, 2) for _ in c],
            ),
            patch(
                "vstash.embed.embed_texts",
                side_effect=lambda t, m, *a, **k: [_unit(0.95, 2) for _ in t],
            ),
            patch(
                "vstash.memory.embed_texts",
                side_effect=lambda t, m, *a, **k: [_unit(0.95, 2) for _ in t],
                create=True,
            ),
            patch("vstash.services.search.embed_query", return_value=Q),
        ):
            mem = Memory(db=tmp_path / "m.db")
            try:
                pad = " This sentence pads the claim past the minimum chunk size. memo"
                mem.remember(
                    "We use bge small." + pad, title="v1", layer="decision", source_ref="a"
                )
                mem.remember(
                    "We moved to jina." + pad,
                    title="v2",
                    layer="decision",
                    source_ref="b",
                    supersedes="text://v1",
                )
                mem.update("text://v1", text="We use bge small, edited." + pad)
                assert "text://v1" not in [r.path for r in mem.search("memo")]
            finally:
                mem.close()


@requires_sqlite_vec
def test_identical_reremember_still_supersedes(store, cfg) -> None:
    _remember(store, cfg, "We use bge small.", "v1")
    _remember(store, cfg, "We moved to jina nano.", "v2")
    res = _remember(store, cfg, "We moved to jina nano.", "v2", supersedes="text://v1")
    assert res.status == "skipped"
    assert res.superseded is True
    assert "text://v1" not in _paths(store)


@requires_sqlite_vec
def test_identical_reremember_validates_target(store, cfg) -> None:
    _remember(store, cfg, "We moved to jina nano.", "v2")
    with pytest.raises(ValueError, match="no document"):
        _remember(store, cfg, "We moved to jina nano.", "v2", supersedes="text://nope")


@requires_sqlite_vec
class TestSelfAndCycles:
    def test_self_supersede_rejected_before_write(self, store, cfg) -> None:
        _remember(store, cfg, "We use bge small.", "v1")
        with pytest.raises(ValueError, match="itself"):
            _remember(store, cfg, "We use bge small again.", "v1", supersedes="text://v1")
        assert "text://v1" in _paths(store)

    def test_cycle_rejected(self, store, cfg) -> None:
        _remember(store, cfg, "A says one thing.", "a")
        _remember(store, cfg, "B replaces A.", "b", supersedes="text://a")
        with pytest.raises(ValueError, match="restore"):
            _remember(store, cfg, "A replaces B now.", "a", supersedes="text://b")
        assert "text://b" in _paths(store)


@requires_sqlite_vec
class TestRestore:
    def test_restore_brings_it_back_with_derived_status(self, store, cfg) -> None:
        _remember(store, cfg, "We use bge small.", "v1")
        _remember(store, cfg, "We moved to jina nano.", "v2", supersedes="text://v1")
        assert store.restore_claim(store.doc_id_for("default", "text://v1")) is True
        assert "text://v1" in _paths(store)
        row = store._conn.execute(
            "SELECT status, valid_until FROM documents WHERE path = 'text://v1'"
        ).fetchone()
        assert row[0] == "vigente" and row[1] is None

    def test_restore_live_claim_is_noop(self, store, cfg) -> None:
        _remember(store, cfg, "We use bge small.", "v1")
        assert store.restore_claim(store.doc_id_for("default", "text://v1")) is False


@requires_sqlite_vec
class TestIncludeRetired:
    def test_flag_and_cache_key(self, store, cfg) -> None:
        _remember(store, cfg, "We use bge small.", "v1")
        _remember(store, cfg, "We moved to jina nano.", "v2", supersedes="text://v1")
        assert "text://v1" not in _paths(store)
        shown = {r.path: r.status for r in store.search(Q, WORD, top_k=10, include_retired=True)}
        assert shown["text://v1"] == "superado"
        assert "text://v1" not in _paths(store)


def _crowded(tmp_path: Path, **kw) -> VstashStore:
    s = VstashStore(str(tmp_path / "crowd.db"), embedding_dim=DIM, **kw)
    s.add_document(
        path="text://old",
        title="old",
        chunks=["alpha uno", "alpha dos", "alpha tres"],
        embeddings=[_unit(0.99, 2), _unit(0.985, 3), _unit(0.98, 4)],
        source_type="text",
    )
    s.add_document(
        path="text://new",
        title="new",
        chunks=["alpha nuevo"],
        embeddings=[_unit(0.90, 5)],
        source_type="text",
    )
    s.supersede_claim(s.doc_id_for("default", "text://old"), s.doc_id_for("default", "text://new"))
    return s


@requires_sqlite_vec
class TestCandidatePool:
    def test_vec0(self, tmp_path: Path) -> None:
        s = _crowded(tmp_path)
        try:
            assert [r.path for r in s.search(Q, "zzz", top_k=1, retrieval_mode="vec_only")] == [
                "text://new"
            ]
        finally:
            s.close()

    def test_snapvec(self, tmp_path: Path) -> None:
        pytest.importorskip("snapvec")
        s = _crowded(tmp_path, vector_backend="snapvec", snapvec_bits=4)
        try:
            assert [r.path for r in s.search(Q, "zzz", top_k=1, retrieval_mode="vec_only")] == [
                "text://new"
            ]
        finally:
            s.close()


@requires_sqlite_vec
def test_federated_forwards_include_retired(tmp_path: Path, monkeypatch) -> None:
    import vstash.profile as prof

    db = tmp_path / "fed.db"
    s = VstashStore(str(db), embedding_dim=DIM)
    for path, emb in (("text://old", _unit(0.95, 2)), ("text://new", _unit(0.94, 3))):
        s.add_document(
            path=path, title=path, chunks=[f"{WORD} {path}"], embeddings=[emb], source_type="text"
        )
    s.supersede_claim(s.doc_id_for("default", "text://old"), s.doc_id_for("default", "text://new"))
    s.close()
    monkeypatch.setattr(prof, "DEFAULT_DB", db)
    monkeypatch.setattr(prof, "_find_local_db", lambda: None)
    monkeypatch.setattr(prof, "list_profiles", lambda: [])

    def run(**kw):
        return {
            r.path for _, r in prof.federated_search(Q, WORD, cfg=VstashConfig(), top_k=5, **kw)
        }

    assert "text://old" not in run()
    assert "text://old" in run(include_retired=True)


@requires_sqlite_vec
def test_mcp_retire_restore_and_include_retired(store, cfg) -> None:
    import json

    from vstash.mcp import vstash_remember, vstash_retire, vstash_search

    pad = " This sentence pads the claim past the minimum chunk size. memo"
    with (
        patch("vstash.mcp._get_store", return_value=store),
        patch("vstash.mcp._get_config", return_value=VstashConfig()),
        patch(
            "vstash.ingest._embed_with_progress",
            side_effect=lambda c, m, **k: [_unit(0.95, 2) for _ in c],
        ),
        patch("vstash.services.search.embed_query", return_value=Q),
    ):
        json.loads(
            vstash_remember("We use bge small." + pad, title="v1", layer="decision", source_ref="a")
        )
        bad = json.loads(
            vstash_remember(
                "Typo target." + pad,
                title="v9",
                layer="decision",
                source_ref="a",
                supersedes="text://nope",
            )
        )
        assert "error" in bad

        assert json.loads(vstash_retire("text://v1", refuted=True)) == {"retired": True}
        assert "text://v1" not in [c["path"] for c in json.loads(vstash_search("memo"))["chunks"]]
        hist = json.loads(vstash_search("memo", include_retired=True))["chunks"]
        assert {c["path"]: c.get("status") for c in hist}["text://v1"] == "refutado"

        assert json.loads(vstash_retire("text://v1", restore=True)) == {"restored": True}
        assert "text://v1" in [c["path"] for c in json.loads(vstash_search("memo"))["chunks"]]
