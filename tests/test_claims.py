"""Claim state on memories: status, source, validity window and replacement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from vstash.ingest import ingest_text
from vstash.store import VstashStore

DIM = 384
WORD = "memo"  # every test text carries it, so the FTS side matches too


@pytest.fixture
def store(tmp_path):
    s = VstashStore(db_path=str(tmp_path / "claims.db"), embedding_dim=DIM)
    yield s
    s.close()


@pytest.fixture
def cfg():
    c = MagicMock()
    c.chunking.size = 1024
    c.chunking.overlap = 0
    c.embeddings.model = "BAAI/bge-small-en-v1.5"
    return c


def _vec(seed: float) -> list[float]:
    return [seed] * DIM


def _remember(store, cfg, text, title, seed=0.1, **kw):
    with patch("vstash.ingest._embed_with_progress", return_value=[_vec(seed)]):
        body = f"{text} This sentence pads the claim past the minimum chunk size. {WORD}"
        result = ingest_text(body, cfg, store, title=title, layer="hecho", **kw)
        assert result.status == "ok", result
        return result


def _titles(store, seed=0.1):
    return {r.title for r in store.search(_vec(seed), WORD, top_k=10)}


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


class TestClaimGate:
    def test_fact_without_source_is_not_vigente(self, store, cfg):
        assert _remember(store, cfg, "The trip leaves on the 17th.", "trip").claim_status == (
            "sin-fuente"
        )

    def test_fact_with_source_is_vigente(self, store, cfg):
        r = _remember(store, cfg, "The trip leaves on the 17th.", "trip", source_ref="session 1")
        assert r.claim_status == "vigente"


class TestSupersede:
    def test_replaced_claim_leaves_search_but_is_kept(self, store, cfg):
        _remember(store, cfg, "The trip leaves on the 17th.", "trip v1", source_ref="s1")
        r = _remember(
            store,
            cfg,
            "The trip leaves on the 19th.",
            "trip v2",
            source_ref="s2",
            supersedes="text://trip v1",
        )
        assert r.superseded is True
        assert _titles(store) == {"trip v2"}
        by_title = {d.title: d for d in store.list_documents()}
        assert by_title["trip v1"].status == "superado"
        assert by_title["trip v2"].supersedes == by_title["trip v1"].id

    def test_unknown_target_raises_and_writes_nothing(self, store, cfg):
        with pytest.raises(ValueError, match="supersedes"):
            _remember(store, cfg, "New.", "new", source_ref="s", supersedes="trip v1")
        assert store.list_documents() == []

    def test_already_retired_target_reports_false(self, store, cfg):
        _remember(store, cfg, "Old.", "old", source_ref="s")
        old_id = store.list_documents()[0].id
        assert store.supersede_claim(old_id) == 1
        r = _remember(store, cfg, "New.", "new", source_ref="s", supersedes=old_id)
        assert r.superseded is False

    def test_row_without_status_can_be_retired(self, store):
        # A row written before the claim schema has NULL status.
        doc_id = store.add_document(
            path="text://legacy",
            title="legacy",
            chunks=[f"A legacy note {WORD}."],
            embeddings=[_vec(0.1)],
            source_type="text",
        )
        assert store.list_documents()[0].status is None
        assert store.supersede_claim(doc_id) == 1
        assert _titles(store) == set()


class TestRetire:
    def test_retire_without_replacement(self, store, cfg):
        _remember(store, cfg, "Rome is the venue.", "venue", source_ref="s")
        doc = store.list_documents()[0]
        assert store.supersede_claim(doc.id, status="refutado") == 1
        assert _titles(store) == set()
        assert store.list_documents(status="refutado")[0].title == "venue"
        assert store.supersede_claim(doc.id) == 0  # already retired

    def test_retire_rejects_other_statuses(self, store):
        with pytest.raises(ValueError):
            store.supersede_claim("x" * 32, status="vigente")

    def test_retire_invalidates_cached_search(self, store, cfg):
        _remember(store, cfg, "Rome is the venue.", "venue", source_ref="s")
        assert _titles(store) == {"venue"}
        store.supersede_claim(store.list_documents()[0].id)
        assert _titles(store) == set()


class TestValidity:
    def test_expired_claim_leaves_search(self, store, cfg):
        _remember(
            store,
            cfg,
            "Fare is 800.",
            "fare",
            source_ref="s",
            valid_until=_iso(timedelta(hours=-1)),
        )
        _remember(
            store,
            cfg,
            "Hotel is booked.",
            "hotel",
            source_ref="s",
            valid_until=_iso(timedelta(days=30)),
        )
        _remember(store, cfg, "Name is Jay.", "name", source_ref="s")
        assert _titles(store) == {"hotel", "name"}
        # Listing still shows it: that is how history is read.
        assert {d.title for d in store.list_documents()} == {"fare", "hotel", "name"}

    def test_date_only_valid_until(self, store, cfg):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()
        _remember(store, cfg, "Was true.", "past", source_ref="s", valid_until=past)
        _remember(store, cfg, "Still true.", "future", source_ref="s", valid_until=future)
        assert _titles(store) == {"future"}

    def test_valid_until_is_canonicalised_to_utc(self, store, cfg):
        _remember(store, cfg, "X.", "x", source_ref="s", valid_until="2099-01-15T00:00:00+05:00")
        assert store.list_documents()[0].valid_until == "2099-01-14T19:00:00+00:00"

    def test_non_iso_valid_until_is_rejected(self, store, cfg):
        with pytest.raises(ValueError):
            _remember(store, cfg, "X.", "x", source_ref="s", valid_until="next tuesday")


class TestListing:
    def test_list_carries_claim_fields_and_filters_by_status(self, store, cfg):
        _remember(store, cfg, "Sourced.", "a", source_ref="session 9")
        _remember(store, cfg, "Unsourced.", "b")
        vigentes = store.list_documents(status="vigente")
        assert [d.title for d in vigentes] == ["a"]
        assert vigentes[0].source_ref == "session 9" and vigentes[0].id
        assert {d.title for d in store.list_documents(status=["vigente", "sin-fuente"])} == {
            "a",
            "b",
        }


class TestMemoryApi:
    def test_retire_and_list_through_memory(self, tmp_path):
        from vstash import Memory

        mem = Memory(db=tmp_path / "m.db", collection="notes")
        try:
            with patch("vstash.ingest._embed_with_progress", return_value=[_vec(0.1)]):
                mem.remember(
                    f"The venue is Rome, as agreed in the planning session. {WORD}",
                    "venue",
                    layer="decision",
                    source_ref="session 3",
                )
            assert [d.status for d in mem.list(status="vigente")] == ["vigente"]
            assert mem.retire("text://venue", refuted=True) is True
            assert mem.retire("text://venue") is False  # already retired
            assert mem.retire("text://nothing") is False
            assert [d.status for d in mem.list()] == ["refutado"]
        finally:
            mem.close()
