"""End-to-end tests for frequency + decay scoring through the full pipeline.

These tests exercise the complete flow: ingest → search → re-rank → track,
verifying that scoring actually changes result ordering and that access
tracking persists across searches.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import ScoringConfig, VstashConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.models import SearchResult
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def e2e_config() -> VstashConfig:
    """Config with scoring enabled."""
    return VstashConfig.model_validate(
        {
            "scoring": {
                "enabled": True,
                "alpha": 0.5,
                "beta": 0.5,
                "decay_lambda": 0.05,
                "over_fetch": 20,
                "track_access": True,
            }
        }
    )


@pytest.fixture
def e2e_config_no_tracking() -> VstashConfig:
    """Config with scoring enabled but tracking disabled."""
    return VstashConfig.model_validate(
        {
            "scoring": {
                "enabled": True,
                "track_access": False,
            }
        }
    )


@pytest.fixture
def e2e_store(tmp_path, e2e_config: VstashConfig) -> VstashStore:
    """Store with realistic documents for e2e testing."""
    dim = get_embedding_dim(e2e_config.embeddings.model)
    store = VstashStore(str(tmp_path / "e2e.db"), embedding_dim=dim)

    # Ingest several docs with real embeddings
    docs = [
        (
            "Python decorators simplify function wrapping and metaprogramming.",
            "python_decorators.md",
        ),
        ("Rust ownership model prevents memory leaks at compile time.", "rust_ownership.md"),
        ("Machine learning requires large datasets for training models.", "ml_basics.md"),
        ("Database indexing improves query performance significantly.", "db_indexing.md"),
        ("Python list comprehensions provide concise iteration syntax.", "python_lists.md"),
    ]

    for text, filename in docs:
        embedding = embed_query(text, e2e_config.embeddings.model)
        store.add_document(
            path=f"/test/{filename}",
            title=filename.replace(".md", "").replace("_", " ").title(),
            chunks=[text],
            embeddings=[embedding],
            source_type="markdown",
        )

    yield store  # type: ignore[misc]
    store.close()


# ------------------------------------------------------------------ #
# E2E: Scoring disabled (baseline)                                     #
# ------------------------------------------------------------------ #


class TestScoringDisabled:
    """When scoring is disabled, search should behave exactly as before."""

    def test_search_without_scoring_returns_results(self, e2e_store: VstashStore) -> None:
        """Baseline search should return results without scoring."""
        cfg = ScoringConfig(enabled=False)
        query_emb = embed_query("Python programming", VstashConfig().embeddings.model)

        results = e2e_store.search(
            query_embedding=query_emb,
            query_text="Python programming",
            top_k=3,
            scoring=cfg,
        )
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_access_tracking_when_scoring_disabled_but_track_on(
        self, e2e_store: VstashStore
    ) -> None:
        """Access counts should still increment when scoring is disabled but track_access=True.

        This builds usage history so scoring can be enabled later.
        """
        cfg = ScoringConfig(enabled=False, track_access=True)
        query_emb = embed_query("Python decorators", VstashConfig().embeddings.model)

        before = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }

        e2e_store.search(
            query_embedding=query_emb,
            query_text="Python decorators",
            top_k=3,
            scoring=cfg,
        )

        after = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }
        assert any(after[cid] > before[cid] for cid in before)

    def test_no_access_tracking_when_track_access_false(
        self, e2e_store: VstashStore
    ) -> None:
        """Access counts should not change when track_access is explicitly False."""
        cfg = ScoringConfig(enabled=False, track_access=False)
        query_emb = embed_query("Python decorators", VstashConfig().embeddings.model)

        before = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }

        e2e_store.search(
            query_embedding=query_emb,
            query_text="Python decorators",
            top_k=3,
            scoring=cfg,
        )

        after = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }
        assert before == after


# ------------------------------------------------------------------ #
# E2E: Scoring enabled                                                 #
# ------------------------------------------------------------------ #


class TestScoringEnabled:
    """When scoring is enabled, frequency+decay should influence ranking."""

    def test_search_with_scoring_returns_results(
        self, e2e_store: VstashStore, e2e_config: VstashConfig
    ) -> None:
        """Search with scoring enabled should return valid results."""
        query_emb = embed_query("Python programming", e2e_config.embeddings.model)

        results = e2e_store.search(
            query_embedding=query_emb,
            query_text="Python programming",
            top_k=3,
            scoring=e2e_config.scoring,
        )
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)

    def test_access_tracking_increments_on_search(
        self, e2e_store: VstashStore, e2e_config: VstashConfig
    ) -> None:
        """Searching with tracking enabled should increment access counts."""
        query_emb = embed_query("Python decorators", e2e_config.embeddings.model)

        e2e_store.search(
            query_embedding=query_emb,
            query_text="Python decorators",
            top_k=3,
            scoring=e2e_config.scoring,
        )

        # At least some chunks should have access_count > 0 (incremented from initial 0)
        rows = e2e_store._conn.execute(
            "SELECT access_count FROM chunks WHERE access_count > 0"
        ).fetchall()
        assert len(rows) > 0

    def test_no_tracking_when_track_access_false(
        self, e2e_store: VstashStore, e2e_config_no_tracking: VstashConfig
    ) -> None:
        """Scoring enabled but track_access=False should not update counts."""
        cfg = e2e_config_no_tracking
        query_emb = embed_query("Python decorators", cfg.embeddings.model)

        before = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }

        e2e_store.search(
            query_embedding=query_emb,
            query_text="Python decorators",
            top_k=3,
            scoring=cfg.scoring,
        )

        after = {
            r["id"]: r["access_count"]
            for r in e2e_store._conn.execute("SELECT id, access_count FROM chunks").fetchall()
        }
        assert before == after

    def test_repeated_searches_boost_ranking(
        self, e2e_store: VstashStore, e2e_config: VstashConfig
    ) -> None:
        """Repeatedly searching for a topic should boost those chunks over time."""
        cfg = e2e_config
        model = cfg.embeddings.model

        # Search for "database indexing" multiple times to boost that chunk
        for _ in range(10):
            query_emb = embed_query("database indexing query performance", model)
            e2e_store.search(
                query_embedding=query_emb,
                query_text="database indexing query performance",
                top_k=5,
                scoring=cfg.scoring,
            )

        # Now search with a broad query that could match multiple docs
        broad_emb = embed_query("performance optimization techniques", model)
        results = e2e_store.search(
            query_embedding=broad_emb,
            query_text="performance optimization techniques",
            top_k=5,
            scoring=cfg.scoring,
        )

        # The db_indexing doc should be boosted due to frequent access
        paths = [r.path for r in results]
        if "/test/db_indexing.md" in paths:
            db_idx = paths.index("/test/db_indexing.md")
            # It should rank relatively high due to access frequency
            assert db_idx < 3, f"db_indexing.md ranked at {db_idx}, expected top 3"

    def test_old_accesses_decay(self, e2e_store: VstashStore, e2e_config: VstashConfig) -> None:
        """Chunks accessed long ago should have diminished boost."""
        cfg = e2e_config

        # Manually set high access count but old date for one chunk
        chunk_row = e2e_store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = '/test/rust_ownership.md' LIMIT 1"
        ).fetchone()
        if chunk_row:
            old_date = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
            e2e_store._conn.execute(
                "UPDATE chunks SET access_count = 100, last_accessed_at = ? WHERE id = ?",
                [old_date, chunk_row["id"]],
            )
            e2e_store._conn.commit()

        # Search — the old high-access chunk should not dominate
        query_emb = embed_query("Python programming syntax", cfg.embeddings.model)
        results = e2e_store.search(
            query_embedding=query_emb,
            query_text="Python programming syntax",
            top_k=3,
            scoring=cfg.scoring,
        )

        # Python docs should still rank above old Rust doc for a Python query
        if len(results) >= 2:
            top_paths = [r.path for r in results[:2]]
            assert any("python" in p for p in top_paths), (
                f"Expected Python docs in top 2, got {top_paths}"
            )


# ------------------------------------------------------------------ #
# E2E: Over-fetch verification                                        #
# ------------------------------------------------------------------ #


class TestOverFetch:
    """Verify that over-fetching retrieves more candidates for re-ranking."""

    def test_over_fetch_with_small_top_k(
        self, e2e_store: VstashStore, e2e_config: VstashConfig
    ) -> None:
        """With top_k=1 and over_fetch=20, scoring should still consider many candidates."""
        cfg = e2e_config
        query_emb = embed_query("programming languages", cfg.embeddings.model)

        results = e2e_store.search(
            query_embedding=query_emb,
            query_text="programming languages",
            top_k=1,
            scoring=cfg.scoring,
        )
        assert len(results) == 1
        assert results[0].score > 0

    def test_scoring_vs_no_scoring_can_differ(
        self, e2e_store: VstashStore, e2e_config: VstashConfig
    ) -> None:
        """Scoring may reorder results compared to pure RRF."""
        model = e2e_config.embeddings.model

        # Boost a specific chunk
        chunk_row = e2e_store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = '/test/ml_basics.md' LIMIT 1"
        ).fetchone()
        if chunk_row:
            for _ in range(15):
                e2e_store.track_access([chunk_row["id"]])

        query_emb = embed_query("data training models", model)

        # Without scoring
        no_scoring = e2e_store.search(
            query_embedding=query_emb,
            query_text="data training models",
            top_k=5,
            scoring=None,
        )

        # With scoring
        with_scoring = e2e_store.search(
            query_embedding=query_emb,
            query_text="data training models",
            top_k=5,
            scoring=e2e_config.scoring,
        )

        # Both should return results
        assert len(no_scoring) > 0
        assert len(with_scoring) > 0

        # Scores should differ (scoring adds final_score based on frequency)
        no_scores = [r.score for r in no_scoring]
        with_scores = [r.score for r in with_scoring]
        assert no_scores != with_scores, "Scoring should change at least one result score"
