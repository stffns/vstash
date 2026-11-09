"""Opt-in "nothing found" gate (``min_relevance``) and the cache tier fix.

``min_relevance`` drops the whole result list when the query's best vector
match falls below a :func:`vstash.store.relevance_tier`. The tier is read
from ``store.last_best_distance``, which a query-cache hit used to leave
holding the previous query's value -- so the cache now stores it too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import CacheConfig, VstashConfig
from vstash.services import search_with_embedding
from vstash.store import VstashStore, relevance_tier
from vstash.validation import LimitError

DIM = 384


def _query(cos: float) -> list[float]:
    """Unit vector whose cosine to the stored document (axis 0) is ``cos``."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cos
    v[1] = math.sqrt(1.0 - cos * cos)
    return v.tolist()


DOC = _query(1.0)
# Cosine distance = 1 - cos. Tier bounds: high <= 0.4513 < medium <= 0.4802 < low.
NEAR = _query(0.90)  # distance 0.10 -> high
MID = _query(0.535)  # distance 0.465 -> medium
FAR = _query(0.30)  # distance 0.70 -> low


@pytest.fixture
def store(tmp_path: Path):
    s = VstashStore(
        str(tmp_path / "gate.db"),
        embedding_dim=DIM,
        cache=CacheConfig(query_cache_size=16),
    )
    s.add_document(
        path="/t/apple.md",
        title="apple",
        chunks=["apple orchard harvest notes"],
        embeddings=[DOC],
        source_type="markdown",
    )
    yield s
    s.close()


def _search(store: VstashStore, emb: list[float], query: str = "zzz", **kw):
    with patch("vstash.services.search.embed_query", return_value=emb):
        return search_with_embedding(
            cfg=VstashConfig(), store=store, query=query, top_k=5, expand_window=0, **kw
        )


@requires_sqlite_vec
class TestMinRelevance:
    def test_sanity_tiers(self, store) -> None:
        for emb, tier in ((NEAR, "high"), (MID, "medium"), (FAR, "low")):
            store.search(np.asarray(emb, dtype=np.float32), "zzz", top_k=5)
            assert relevance_tier(store.last_best_distance) == tier

    def test_default_returns_off_topic_results(self, store) -> None:
        assert len(_search(store, FAR)) == 1

    @pytest.mark.parametrize(
        ("emb", "gate", "expected"),
        [
            (NEAR, "medium", 1),
            (MID, "medium", 1),
            (FAR, "medium", 0),
            (NEAR, "high", 1),
            (MID, "high", 0),
            (FAR, "high", 0),
        ],
    )
    def test_gate(self, store, emb, gate, expected) -> None:
        assert len(_search(store, emb, min_relevance=gate)) == expected

    def test_fts_only_is_not_gated(self, store) -> None:
        res = _search(store, FAR, query="apple", retrieval_mode="fts_only", min_relevance="high")
        assert [r.title for r in res] == ["apple"]

    @pytest.mark.parametrize("bad", ["certain", "low", ""])
    def test_invalid_value_rejected(self, store, bad) -> None:
        with pytest.raises(LimitError):
            _search(store, NEAR, min_relevance=bad)

    def test_gated_search_is_logged_as_empty(self, store) -> None:
        res = _search(store, FAR, min_relevance="medium")
        assert res == []
        assert res.event_id is not None
        count = store._conn.execute(
            "SELECT result_count FROM search_events WHERE id = ?", [res.event_id]
        ).fetchone()[0]
        assert count == 0

    def test_memory_passthrough(self, store) -> None:
        from vstash.memory import Memory

        mem = Memory.__new__(Memory)
        mem._cfg = VstashConfig()
        mem._store = store
        mem._collection = None
        mem._project = None
        with (
            patch.object(Memory, "_resolve_collection", return_value=None),
            patch.object(Memory, "_resolve_project", return_value=None),
            patch("vstash.services.search.embed_query", return_value=FAR),
        ):
            assert mem.search("zzz", min_relevance="medium") == []
            assert len(mem.search("zzz")) == 1

    def test_mcp_reports_best_relevance(self, store) -> None:
        from vstash.mcp import vstash_search

        with (
            patch("vstash.mcp._get_store", return_value=store),
            patch("vstash.mcp._get_config", return_value=VstashConfig()),
            patch("vstash.services.search.embed_query", return_value=FAR),
        ):
            gated = json.loads(vstash_search("zzz", min_relevance="medium"))
            plain = json.loads(vstash_search("zzz"))
        assert gated["chunks"] == []
        assert gated["relevance"] == "none"
        assert gated["best_relevance"] == "low"
        assert isinstance(gated["event_id"], int)
        assert len(plain["chunks"]) == 1
        assert plain["relevance"] == "low"


@requires_sqlite_vec
class TestCacheRestoresBestDistance:
    def test_cache_hit_restores_best_distance(self, store) -> None:
        far = np.asarray(FAR, dtype=np.float32)
        store.search(far, "zzz", top_k=5)
        far_best = store.last_best_distance
        store.search(np.asarray(NEAR, dtype=np.float32), "zzz", top_k=5)
        entries = len(store._query_cache)
        store.search(far, "zzz", top_k=5)  # served from the cache
        assert len(store._query_cache) == entries
        assert store.last_best_distance == pytest.approx(far_best)

    def test_gate_holds_on_cache_hit(self, store) -> None:
        assert _search(store, FAR, min_relevance="medium") == []
        assert len(_search(store, NEAR, min_relevance="medium")) == 1
        # Same FAR query again hits the cache; NEAR's distance must not leak.
        assert _search(store, FAR, min_relevance="medium") == []
