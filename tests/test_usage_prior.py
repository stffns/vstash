"""Opt-in query-conditioned usage prior (``[usage_prior]``).

Covers the stored query embedding on ``search_events`` (write path +
additive migration), ``VstashStore.usage_prior_doc_weights`` (tau,
neighbour cap, dimension guard), the services re-rank (off by default,
per-call override, untouched ranking without a similar past search,
candidate pool, failure isolation), the SDK passthrough and the reindex
reset.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import UsagePriorConfig, VstashConfig
from vstash.services import search_with_embedding
from vstash.store import VstashStore

DIM = 384


def _unit(*pairs: tuple[int, float]) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    for i, x in pairs:
        v[i] = x
    return (v / np.linalg.norm(v)).tolist()


Q = _unit((0, 1.0))


def _at(cos: float, axis: int) -> list[float]:
    return _unit((0, cos), (axis, math.sqrt(1.0 - cos * cos)))


# Ranked by cosine to Q: A > B > C > D, all inside the store's relative
# distance cutoff (best_distance * 1.3225) so a top_k=4 search returns all.
EMB = {"A": _at(0.900, 1), "B": _at(0.895, 2), "C": _at(0.890, 3), "D": _at(0.880, 4)}
# cos(Q, FAR) ~= 0.6: well below the default tau of 0.9.
FAR = _unit((0, 0.6), (5, 0.8))


@pytest.fixture
def store(tmp_path: Path):
    s = VstashStore(str(tmp_path / "prior.db"), embedding_dim=DIM)
    for name, emb in EMB.items():
        s.add_document(
            path=f"/t/{name}.md",
            title=name,
            chunks=[f"document {name} body text"],
            embeddings=[emb],
            source_type="markdown",
        )
    yield s
    s.close()


@pytest.fixture
def embed():
    with patch("vstash.services.search.embed_query") as mock:
        mock.return_value = Q
        yield mock


def _cfg(**prior) -> VstashConfig:
    return VstashConfig(usage_prior=UsagePriorConfig(**prior))


def _titles(results) -> list[str]:
    return [r.title for r in results]


def _chunk_of(store: VstashStore, title: str) -> int:
    return int(
        store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.title = ?",
            [title],
        ).fetchone()[0]
    )


def _past_search(store: VstashStore, emb: list[float], used: str, *, rank_of_used=1) -> int:
    """Log a past search with ``emb`` as its query vector where ``used`` was used."""
    results = store.search(np.asarray(emb, dtype=np.float32), "zzz", top_k=4)
    eid = store.record_search_event(
        query="past", best_distance=0.1, relevance_tier="high", result_count=len(results),
        results=results, query_embedding=emb,
    )  # fmt: skip
    assert store.mark_result_used(eid, _chunk_of(store, used)) == 1
    return eid


# --------------------------------------------------------------------- #
# Storage                                                                #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestStorage:
    def test_service_search_stores_query_embedding(self, store, embed) -> None:
        res = search_with_embedding(cfg=_cfg(), store=store, query="zzz", top_k=2)
        blob = store._conn.execute(
            "SELECT query_embedding FROM search_events WHERE id = ?", [res.event_id]
        ).fetchone()[0]
        assert np.allclose(np.frombuffer(blob, dtype=np.float32), Q)

    def test_record_event_without_embedding_stores_null(self, store) -> None:
        eid = store.record_search_event("q", 0.1, "high", 0)
        row = store._conn.execute(
            "SELECT query_embedding FROM search_events WHERE id = ?", [eid]
        ).fetchone()
        assert row[0] is None

    def test_old_db_gains_column_on_open(self, tmp_path: Path) -> None:
        db = str(tmp_path / "old.db")
        s = VstashStore(db, embedding_dim=DIM)
        eid = s.record_search_event("q", 0.1, "high", 0)
        s._conn.execute("ALTER TABLE search_events DROP COLUMN query_embedding")
        s.close()

        reopened = VstashStore(db, embedding_dim=DIM)
        try:
            cols = {r[1] for r in reopened._conn.execute("PRAGMA table_info(search_events)")}
            assert "query_embedding" in cols
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM search_events WHERE id = ?", [eid]
            ).fetchone()[0] == 1  # fmt: skip
        finally:
            reopened.close()


# --------------------------------------------------------------------- #
# Store primitive                                                        #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestDocWeights:
    def test_similar_past_search_credits_used_doc(self, store) -> None:
        _past_search(store, Q, used="B")
        w = store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20)
        assert len(w) == 1
        assert next(iter(w.values())) == pytest.approx(1.0, abs=1e-5)

    def test_dissimilar_past_search_is_ignored(self, store) -> None:
        _past_search(store, FAR, used="B")
        assert store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}

    def test_events_without_a_use_are_ignored(self, store) -> None:
        store.record_search_event(
            "q", 0.1, "high", 1, results=store.search(np.asarray(Q), "zzz", top_k=2),
            query_embedding=Q,
        )  # fmt: skip
        assert store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}

    def test_max_neighbors_keeps_most_similar(self, store) -> None:
        near = _unit((0, 1.0), (6, 0.2))  # cos ~0.98
        _past_search(store, near, used="C")
        _past_search(store, Q, used="B")  # cos 1.0
        w = store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=1)
        doc_b = store.chunk_doc_ids([_chunk_of(store, "B")])[_chunk_of(store, "B")]
        assert list(w) == [doc_b]

    def test_weights_accumulate_over_neighbours(self, store) -> None:
        _past_search(store, Q, used="B")
        _past_search(store, Q, used="B")
        w = store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20)
        assert next(iter(w.values())) == pytest.approx(2.0, abs=1e-5)

    def test_cache_sees_new_marks_and_deleted_docs(self, store) -> None:
        assert store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}
        _past_search(store, Q, used="B")  # new mark after a cached (empty) lookup
        assert len(store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20)) == 1
        store.delete_document("/t/B.md")  # cascades the used row away
        assert store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}

    def test_wrong_dimension_rows_are_skipped(self, store) -> None:
        eid = _past_search(store, Q, used="B")
        store._conn.execute(
            "UPDATE search_events SET query_embedding = ? WHERE id = ?",
            [np.ones(8, dtype=np.float32).tobytes(), eid],
        )
        assert store.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}


# --------------------------------------------------------------------- #
# Services re-rank                                                       #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestRerank:
    def test_base_order(self, store, embed) -> None:
        res = search_with_embedding(cfg=_cfg(), store=store, query="zzz", top_k=3)
        assert _titles(res) == ["A", "B", "C"]
        assert all(r.usage_boost is None for r in res)

    def test_off_by_default(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        res = search_with_embedding(cfg=VstashConfig(), store=store, query="zzz", top_k=3)
        assert _titles(res) == ["A", "B", "C"]
        assert all(r.usage_boost is None for r in res)

    def test_enabled_promotes_doc_used_by_similar_search(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        res = search_with_embedding(cfg=_cfg(enabled=True), store=store, query="zzz", top_k=3)
        assert _titles(res)[0] == "C"
        boosts = {r.title: r.usage_boost for r in res}
        assert boosts["C"] == pytest.approx(1.0 + 2.0 * math.log1p(1.0), abs=1e-4)
        assert boosts["A"] == 1.0 and boosts["B"] == 1.0

    def test_per_call_override(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        on = search_with_embedding(cfg=_cfg(), store=store, query="zzz", top_k=3, usage_prior=True)
        assert _titles(on)[0] == "C"
        off = search_with_embedding(
            cfg=_cfg(enabled=True), store=store, query="zzz", top_k=3, usage_prior=False
        )
        assert _titles(off) == ["A", "B", "C"]

    def test_no_similar_past_search_leaves_ranking_untouched(self, store, embed) -> None:
        _past_search(store, FAR, used="C")
        with patch.object(store, "search", wraps=store.search) as spy:
            res = search_with_embedding(cfg=_cfg(enabled=True), store=store, query="zzz", top_k=3)
        assert _titles(res) == ["A", "B", "C"]
        assert all(r.usage_boost is None for r in res)
        assert spy.call_args.kwargs["top_k"] == 3  # no pooled search

    def test_doc_outside_top_k_is_reached_through_pool(self, store, embed) -> None:
        _past_search(store, Q, used="D")  # D is not in the base top 2
        with patch.object(store, "search", wraps=store.search) as spy:
            res = search_with_embedding(
                cfg=_cfg(enabled=True, candidate_pool=4), store=store, query="zzz", top_k=2
            )
        assert spy.call_args.kwargs["top_k"] == 4
        assert len(res) == 2
        assert "D" in _titles(res)

    def test_beta_zero_is_a_noop(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        res = search_with_embedding(
            cfg=_cfg(enabled=True, beta=0.0), store=store, query="zzz", top_k=3
        )
        assert _titles(res) == ["A", "B", "C"]

    def test_logs_the_reranked_list(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        res = search_with_embedding(cfg=_cfg(enabled=True), store=store, query="zzz", top_k=3)
        logged = [
            r[0]
            for r in store._conn.execute(
                "SELECT chunk_id FROM search_results WHERE event_id = ? ORDER BY rank",
                [res.event_id],
            )
        ]
        assert logged == [r.chunk_id for r in res]

    def test_lookup_failure_degrades_to_no_prior(self, store, embed) -> None:
        _past_search(store, Q, used="C")
        with patch.object(store, "usage_prior_doc_weights", side_effect=RuntimeError("boom")):
            res = search_with_embedding(cfg=_cfg(enabled=True), store=store, query="zzz", top_k=3)
        assert _titles(res) == ["A", "B", "C"]

    def test_cached_results_are_not_mutated(self, tmp_path: Path, embed) -> None:
        from vstash.config import CacheConfig

        s = VstashStore(
            str(tmp_path / "c.db"), embedding_dim=DIM, cache=CacheConfig(query_cache_size=8)
        )
        try:
            for name, emb in EMB.items():
                s.add_document(
                    path=f"/t/{name}.md", title=name, chunks=[f"document {name}"],
                    embeddings=[emb], source_type="markdown",
                )  # fmt: skip
            _past_search(s, Q, used="C")
            search_with_embedding(cfg=_cfg(enabled=True), store=s, query="zzz", top_k=3)
            again = s.search(np.asarray(Q, dtype=np.float32), "zzz", top_k=50)
            assert all(r.usage_boost is None for r in again)
        finally:
            s.close()


# --------------------------------------------------------------------- #
# SDK + reindex                                                          #
# --------------------------------------------------------------------- #


@requires_sqlite_vec
class TestSurfaces:
    def test_memory_search_passthrough(self, store, embed) -> None:
        from vstash.memory import Memory

        _past_search(store, Q, used="C")
        mem = Memory.__new__(Memory)
        mem._cfg = _cfg()
        mem._store = store
        mem._collection = None
        mem._project = None
        with (
            patch.object(Memory, "_resolve_collection", return_value=None),
            patch.object(Memory, "_resolve_project", return_value=None),
        ):
            res = mem.search("zzz", top_k=3, usage_prior=True)
        assert _titles(res)[0] == "C"

    def test_reindex_clears_query_embeddings(self, store) -> None:
        _past_search(store, Q, used="C")
        store.reindex(lambda texts: [Q for _ in texts], new_dim=DIM)
        n = store._conn.execute(
            "SELECT COUNT(*) FROM search_events WHERE query_embedding IS NOT NULL"
        ).fetchone()[0]
        assert n == 0


# --------------------------------------------------------------------- #
# Review follow-ups (#516)                                               #
# --------------------------------------------------------------------- #


class _FakeStore:
    def __init__(self, doc_of: dict[int, str]) -> None:
        self._doc_of = doc_of

    def chunk_doc_ids(self, chunk_ids):
        return {c: self._doc_of[c] for c in chunk_ids if c in self._doc_of}


def _sr(chunk_id: int, score: float, title: str):
    from vstash.models import SearchResult

    return SearchResult(
        chunk_id=chunk_id, text=title, title=title, path=f"/{title}", chunk=0, score=score
    )


class TestMmrOrder:
    # MMR order: A1, B, A2 -- A2 (same doc as A1) was demoted below B by the
    # diversity penalty even though its raw fusion score is higher.
    RESULTS = [_sr(1, 0.016, "A1"), _sr(2, 0.012, "B"), _sr(3, 0.015, "A2")]
    DOCS = {1: "docA", 2: "docB", 3: "docA"}

    def test_unboosted_candidates_keep_mmr_order(self) -> None:
        from vstash.services.search import apply_usage_prior

        out = apply_usage_prior(
            store=_FakeStore(self.DOCS), results=self.RESULTS, weights={"other": 1.0},
            beta=2.0, top_k=3,
        )  # fmt: skip
        assert _titles(out) == ["A1", "B", "A2"]

    def test_boost_lifts_only_the_boosted_doc(self) -> None:
        from vstash.services.search import apply_usage_prior

        out = apply_usage_prior(
            store=_FakeStore(self.DOCS), results=self.RESULTS, weights={"docB": 1.0},
            beta=2.0, top_k=3,
        )  # fmt: skip
        assert _titles(out) == ["B", "A1", "A2"]
        assert out[0].score == pytest.approx(0.012 * (1 + 2 * math.log1p(1.0)))


@requires_sqlite_vec
def test_reindex_through_another_store_invalidates_warm_cache(tmp_path: Path) -> None:
    db = str(tmp_path / "shared.db")
    a = VstashStore(db, embedding_dim=DIM)
    try:
        for name, emb in EMB.items():
            a.add_document(
                path=f"/t/{name}.md", title=name, chunks=[f"document {name}"],
                embeddings=[emb], source_type="markdown",
            )  # fmt: skip
        _past_search(a, Q, used="B")
        assert len(a.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20)) == 1  # warm
        b = VstashStore(db, embedding_dim=DIM)
        try:
            b.reindex(lambda texts: [Q for _ in texts], new_dim=DIM)  # same dim
        finally:
            b.close()
        assert a.usage_prior_doc_weights(Q, tau=0.9, max_neighbors=20) == {}
    finally:
        a.close()
