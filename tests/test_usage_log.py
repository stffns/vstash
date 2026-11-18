"""Usage log: which chunks each search returned, and which the caller used.

Covers the ``search_results`` table (schema + additive migration), the
write path in ``vstash.services.search`` (cache hits, opt-out, failure
isolation), ``mark_result_used`` / ``mark_used``, the training export,
and pruning (rolling cap, document deletion cascade, ``compact``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import CacheConfig, ObservabilityConfig, VstashConfig
from vstash.models import SearchResult, SearchResults
from vstash.services import mark_used, search_with_embedding
from vstash.store import SCHEMA_VERSION, VstashStore
from vstash.validation import LimitError

DIM = 384


@pytest.fixture
def patched_embed_query():
    with patch("vstash.services.search.embed_query") as mock:
        mock.return_value = [0.3] * DIM
        yield mock


def _populate(store: VstashStore) -> None:
    store.add_document(
        path="/test/python_guide.md",
        title="Python Guide",
        chunks=[
            "Python is a high-level programming language known for its simplicity.",
            "Python supports multiple paradigms including object-oriented and functional.",
        ],
        embeddings=[[0.1] * DIM, [0.2] * DIM],
        source_type="markdown",
    )
    store.add_document(
        path="/test/ml_intro.pdf",
        title="ML Introduction",
        chunks=[
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are a key component of deep learning.",
            "Training data quality affects model performance significantly.",
        ],
        embeddings=[[0.3] * DIM, [0.4] * DIM, [0.5] * DIM],
        source_type="pdf",
    )


def _rows(store: VstashStore, event_id: int) -> list[sqlite3.Row]:
    return store._conn.execute(
        "SELECT rank, chunk_id, score, used, used_at FROM search_results "
        "WHERE event_id = ? ORDER BY rank",
        [event_id],
    ).fetchall()


def _cache_hits(registry) -> int:
    return int(registry.snapshot()["counters"].get("query_cache_hits_total", 0))


def _search(store: VstashStore, cfg: VstashConfig, query: str = "python programming", **kw):
    return search_with_embedding(cfg=cfg, store=store, query=query, top_k=3, **kw)


# --------------------------------------------------------------------- #
# Schema / migration                                                     #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestSchema:
    def test_fresh_store_has_table(self, sample_store: VstashStore) -> None:
        cols = {r[1] for r in sample_store._conn.execute("PRAGMA table_info(search_results)")}
        assert cols == {"event_id", "rank", "chunk_id", "score", "used", "used_at"}

    def test_old_db_gains_table_on_open_without_version_bump(self, tmp_path: Path) -> None:
        db = str(tmp_path / "old.db")
        store = VstashStore(db, embedding_dim=DIM)
        _populate(store)
        eid = store.record_search_event("q", 0.1, "high", 0)
        # Simulate a DB written by a pre-usage-log build.
        store._conn.execute("DROP TABLE search_results")
        store.close()

        reopened = VstashStore(db, embedding_dim=DIM)
        try:
            names = {
                r[0]
                for r in reopened._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert "search_results" in names
            assert reopened.get_meta("schema_version") == SCHEMA_VERSION
            # Pre-existing events and documents survive untouched.
            assert (
                reopened._conn.execute(
                    "SELECT COUNT(*) FROM search_events WHERE id = ?", [eid]
                ).fetchone()[0]
                == 1
            )
            assert reopened.stats().documents == 2
        finally:
            reopened.close()
        # Idempotent: a second open is a no-op.
        VstashStore(db, embedding_dim=DIM).close()


# --------------------------------------------------------------------- #
# Write path                                                             #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestWritePath:
    def test_record_event_writes_ranked_rows(self, populated_store: VstashStore) -> None:
        results = [
            SearchResult(chunk_id=5, text="a", title="t", path="/p", chunk=0, score=0.03),
            SearchResult(chunk_id=2, text="b", title="t", path="/p", chunk=1, score=0.02),
        ]
        eid = populated_store.record_search_event("q", 0.1, "high", 2, results=results)
        rows = _rows(populated_store, eid)
        assert [(r["rank"], r["chunk_id"], r["used"]) for r in rows] == [(0, 5, 0), (1, 2, 0)]
        assert rows[0]["score"] == pytest.approx(0.03)

    def test_record_event_without_results_writes_no_rows(
        self, populated_store: VstashStore
    ) -> None:
        eid = populated_store.record_search_event("q", 0.1, "high", 0)
        assert _rows(populated_store, eid) == []

    def test_service_search_logs_returned_chunks(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        results = _search(populated_store, sample_config)
        assert isinstance(results, SearchResults)
        assert isinstance(results.event_id, int)
        assert results
        rows = _rows(populated_store, results.event_id)
        # Logged ids/ranks are the ones the caller got (expansion keeps ids).
        assert [r["chunk_id"] for r in rows] == [r.chunk_id for r in results]
        ev = populated_store._conn.execute(
            "SELECT query, result_count FROM search_events WHERE id = ?", [results.event_id]
        ).fetchone()
        assert (ev["query"], ev["result_count"]) == ("python programming", len(results))

    def test_opt_out_via_config(self, populated_store: VstashStore, patched_embed_query) -> None:
        cfg = VstashConfig(observability=ObservabilityConfig(log_search_results=False))
        results = _search(populated_store, cfg)
        assert results.event_id is None
        assert (
            populated_store._conn.execute("SELECT COUNT(*) FROM search_events").fetchone()[0] == 0
        )

    def test_log_usage_false_skips(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        assert _search(populated_store, sample_config, log_usage=False).event_id is None
        assert (
            populated_store._conn.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == 0
        )

    def test_store_search_primitive_does_not_log(self, populated_store: VstashStore) -> None:
        """Internal callers (retrain mining, benchmarks, explain, miss_analysis)
        hit ``VstashStore.search`` directly and must not pollute the log."""
        populated_store.search([0.3] * DIM, "python", top_k=3)
        populated_store.search([0.3] * DIM, "python", top_k=3, explain=True)
        populated_store.miss_analysis([0.3] * DIM, "python", expected_path="/test/ml_intro.pdf")
        assert (
            populated_store._conn.execute("SELECT COUNT(*) FROM search_events").fetchone()[0] == 0
        )

    def test_cache_hit_is_logged_as_its_own_event(
        self, tmp_db_path: str, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        store = VstashStore(tmp_db_path, embedding_dim=DIM, cache=CacheConfig(query_cache_size=8))
        try:
            _populate(store)
            from vstash.metrics import registry

            first = _search(store, sample_config)
            hits_before = _cache_hits(registry)
            second = _search(store, sample_config)
            assert _cache_hits(registry) == hits_before + 1  # served from the cache
            assert first.event_id != second.event_id
            assert [r["chunk_id"] for r in _rows(store, second.event_id)] == [
                r["chunk_id"] for r in _rows(store, first.event_id)
            ]
        finally:
            store.close()

    def test_log_failure_does_not_fail_search(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        with patch.object(
            populated_store,
            "record_search_event",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            results = _search(populated_store, sample_config)
        assert results and results.event_id is None

    def test_rolling_cap_cascades_to_results(self, tmp_db_path: str) -> None:
        store = VstashStore(
            tmp_db_path,
            embedding_dim=DIM,
            observability=ObservabilityConfig(search_log_max_events=3),
        )
        try:
            _populate(store)
            cid = store._conn.execute("SELECT MIN(id) FROM chunks").fetchone()[0]
            hit = [SearchResult(chunk_id=cid, text="x", title="t", path="/p", chunk=0, score=0.1)]
            ids = [
                store.record_search_event(f"q{i}", 0.1, "high", 1, results=hit) for i in range(5)
            ]
            kept = [r[0] for r in store._conn.execute("SELECT id FROM search_events ORDER BY id")]
            assert kept == ids[-3:]
            logged = {r[0] for r in store._conn.execute("SELECT event_id FROM search_results")}
            assert logged == set(ids[-3:])
        finally:
            store.close()


# --------------------------------------------------------------------- #
# mark_used                                                              #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestMarkUsed:
    def test_marks_only_returned_chunk(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        results = _search(populated_store, sample_config)
        target = results[1].chunk_id
        assert mark_used(store=populated_store, event_id=results.event_id, chunk_ids=target) == 1
        rows = {r["chunk_id"]: r for r in _rows(populated_store, results.event_id)}
        assert rows[target]["used"] == 1 and rows[target]["used_at"]
        assert sum(r["used"] for r in rows.values()) == 1

    def test_idempotent_and_unknown_ids_are_noops(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        results = _search(populated_store, sample_config)
        cid = results[0].chunk_id
        assert mark_used(store=populated_store, event_id=results.event_id, chunk_ids=[cid]) == 1
        first_used_at = _rows(populated_store, results.event_id)[0]["used_at"]
        assert mark_used(store=populated_store, event_id=results.event_id, chunk_ids=[cid]) == 0
        assert _rows(populated_store, results.event_id)[0]["used_at"] == first_used_at
        assert mark_used(store=populated_store, event_id=results.event_id, chunk_ids=[999999]) == 0
        assert mark_used(store=populated_store, event_id=424242, chunk_ids=[cid]) == 0
        assert mark_used(store=populated_store, event_id=results.event_id, chunk_ids=[]) == 0

    def test_rejects_non_int(self, populated_store: VstashStore) -> None:
        with pytest.raises(LimitError):
            mark_used(store=populated_store, event_id="1", chunk_ids=[1])  # type: ignore[arg-type]
        with pytest.raises(LimitError):
            mark_used(store=populated_store, event_id=1, chunk_ids=[True])

    def test_mcp_tool_and_search_event_id(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        from vstash import mcp

        with (
            patch("vstash.mcp._get_store", return_value=populated_store),
            patch("vstash.mcp._get_config", return_value=sample_config),
        ):
            payload = json.loads(mcp.vstash_search("python programming", top_k=3))
            eid = payload["event_id"]
            cid = payload["chunks"][0]["chunk_id"]
            assert json.loads(mcp.vstash_mark_used(eid, str(cid))) == {"marked": 1}
            assert json.loads(mcp.vstash_mark_used(eid, [cid])) == {"marked": 0}
            assert "error" in json.loads(mcp.vstash_mark_used(eid, "x"))
        # vstash_search must not have written a second event of its own.
        assert (
            populated_store._conn.execute("SELECT COUNT(*) FROM search_events").fetchone()[0] == 1
        )

    def test_web_endpoint(self, populated_store: VstashStore, monkeypatch) -> None:
        from starlette.testclient import TestClient

        import vstash.web as web_mod

        cid = populated_store._conn.execute("SELECT MIN(id) FROM chunks").fetchone()[0]
        hit = [SearchResult(chunk_id=cid, text="x", title="t", path="/p", chunk=0, score=0.1)]
        eid = populated_store.record_search_event("q", 0.1, "high", 1, results=hit)
        monkeypatch.setattr(web_mod, "_get_store", lambda: populated_store)
        monkeypatch.setattr(web_mod, "_store", None, raising=False)
        with TestClient(web_mod.create_app()) as client:
            r = client.post("/api/search/used", json={"event_id": eid, "chunk_ids": [cid]})
            assert r.status_code == 200 and r.json() == {"marked": 1}
            bad = client.post("/api/search/used", json={"event_id": "1", "chunk_ids": [cid]})
            assert bad.status_code == 400


# --------------------------------------------------------------------- #
# Export + prune                                                         #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestExportAndPrune:
    def _two_events(self, store: VstashStore, cfg: VstashConfig):
        a = _search(store, cfg, "python programming")
        b = _search(store, cfg, "deep learning")
        mark_used(store=store, event_id=a.event_id, chunk_ids=a[0].chunk_id)
        return a, b

    def test_export_feedback_only(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        a, _b = self._two_events(populated_store, sample_config)
        rows = populated_store.export_usage_labels()
        assert {r["event_id"] for r in rows} == {a.event_id}
        assert [r["rank"] for r in rows] == list(range(len(a)))
        assert [r["used"] for r in rows] == [True] + [False] * (len(a) - 1)
        first = rows[0]
        assert first["query"] == "python programming"
        assert first["chunk_id"] == a[0].chunk_id
        assert first["text"] and first["path"] and first["title"]

    def test_export_all_since_limit_no_text(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        a, b = self._two_events(populated_store, sample_config)
        rows = populated_store.export_usage_labels(feedback_only=False, include_text=False)
        assert {r["event_id"] for r in rows} == {a.event_id, b.event_id}
        assert "text" not in rows[0]
        assert (
            populated_store.export_usage_labels(feedback_only=False, since_iso="2999-01-01") == []
        )
        limited = populated_store.export_usage_labels(feedback_only=False, limit=1)
        assert len(limited) == 1 and limited[0]["event_id"] == b.event_id
        with pytest.raises(ValueError):
            populated_store.export_usage_labels(limit=0)

    def test_deleting_document_drops_its_rows(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        results = _search(populated_store, sample_config)
        doomed = {r.chunk_id for r in results if r.path == "/test/ml_intro.pdf"}
        assert doomed
        populated_store.delete_document("/test/ml_intro.pdf")
        left = {r["chunk_id"] for r in _rows(populated_store, results.event_id)}
        assert left and not (left & doomed)
        # The event itself survives; only rows for vanished chunks go.
        assert (
            populated_store._conn.execute(
                "SELECT COUNT(*) FROM search_events WHERE id = ?", [results.event_id]
            ).fetchone()[0]
            == 1
        )

    def test_prune_search_events_before(
        self, populated_store: VstashStore, sample_config: VstashConfig, patched_embed_query
    ) -> None:
        results = _search(populated_store, sample_config)
        assert populated_store.prune_search_events(before_iso="2000-01-01") == 0
        assert populated_store.prune_search_events(before_iso="2999-01-01T00:00:00Z") == 1
        assert _rows(populated_store, results.event_id) == []
        with pytest.raises(ValueError):
            populated_store.prune_search_events(before_iso="invalid")


@requires_sqlite_vec
class TestMemoryEndToEnd:
    def test_search_mark_export_compact(self, tmp_path: Path) -> None:
        from vstash.memory import Memory

        doc = tmp_path / "doc.md"
        doc.write_text("Key rotation: rotate the signing keys every ninety days.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            hits = mem.search("how often do I rotate keys?")
            assert hits.event_id is not None
            assert mem.mark_used(hits.event_id, hits[0].chunk_id) == 1
            rows = mem.export_usage()
            assert rows and rows[0]["used"] is True and "rotate" in rows[0]["text"]
            report = mem.compact(before="2999-01-01")
            # compact(before=...) bounds the usage log with the same cutoff.
            assert report["search_events_pruned"] >= 1
            assert mem.export_usage(feedback_only=False) == []
