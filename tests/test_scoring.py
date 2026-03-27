"""Tests for frequency + decay memory scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import ScoringConfig, VstashConfig
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def scoring_store(tmp_path) -> VstashStore:
    """Store with scoring-related data pre-loaded."""
    store = VstashStore(str(tmp_path / "scoring.db"), embedding_dim=4)

    # Doc A: about Python
    store.add_document(
        path="/test/python.md",
        title="Python Guide",
        chunks=["Python is great for scripting", "Python has many libraries"],
        embeddings=[[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
        source_type="markdown",
    )

    # Doc B: about Rust
    store.add_document(
        path="/test/rust.md",
        title="Rust Guide",
        chunks=["Rust is great for systems", "Rust has zero-cost abstractions"],
        embeddings=[[0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.8, 0.2]],
        source_type="markdown",
    )

    yield store  # type: ignore[misc]
    store.close()


# ------------------------------------------------------------------ #
# Schema Migration Tests                                               #
# ------------------------------------------------------------------ #


class TestScoringMigration:
    """Test that scoring columns are added correctly."""

    def test_chunks_have_access_count(self, scoring_store: VstashStore) -> None:
        """New chunks should have access_count = 1."""
        rows = scoring_store._conn.execute(
            "SELECT access_count FROM chunks"
        ).fetchall()
        assert all(row["access_count"] == 1 for row in rows)

    def test_chunks_have_created_at(self, scoring_store: VstashStore) -> None:
        """New chunks should have created_at set."""
        rows = scoring_store._conn.execute(
            "SELECT created_at FROM chunks"
        ).fetchall()
        assert all(row["created_at"] is not None for row in rows)

    def test_chunks_have_null_last_accessed(self, scoring_store: VstashStore) -> None:
        """New chunks should have last_accessed_at = NULL (never searched)."""
        rows = scoring_store._conn.execute(
            "SELECT last_accessed_at FROM chunks"
        ).fetchall()
        assert all(row["last_accessed_at"] is None for row in rows)

    def test_migration_on_existing_db(self, tmp_path) -> None:
        """Opening an old DB (pre-scoring schema) should add scoring columns via migration."""
        import sqlite3 as _sqlite3

        import sqlite_vec

        db_path = str(tmp_path / "old.db")

        # Manually create an old-style DB without scoring columns
        conn = _sqlite3.connect(db_path)
        conn.row_factory = _sqlite3.Row
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except AttributeError:
            conn.close()
            conn = sqlite_vec.Connection(db_path)
            conn.row_factory = _sqlite3.Row

        conn.executescript("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'file',
                collection TEXT NOT NULL DEFAULT 'default',
                project TEXT, layer TEXT, tags TEXT,
                char_count INTEGER DEFAULT 0, chunk_count INTEGER DEFAULT 0,
                added_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL, text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[4]);
            CREATE VIRTUAL TABLE fts_chunks
            USING fts5(text, content=chunks, content_rowid=id, tokenize='porter ascii');
        """)
        conn.execute(
            "INSERT INTO documents (id, path, title, source_type, added_at) "
            "VALUES ('d1', '/test/old.md', 'Old Doc', 'file', '2026-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO chunks (doc_id, seq, text) VALUES ('d1', 0, 'old content')"
        )
        conn.commit()

        # Verify scoring columns do NOT exist yet
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        assert "access_count" not in cols
        conn.close()

        # Re-open via VstashStore — migration should add columns + backfill
        store = VstashStore(db_path, embedding_dim=4)
        row = store._conn.execute(
            "SELECT access_count, created_at FROM chunks LIMIT 1"
        ).fetchone()
        assert row["access_count"] == 1
        assert row["created_at"] == "2026-01-01T00:00:00+00:00"  # backfilled from added_at
        store.close()


# ------------------------------------------------------------------ #
# Track Access Tests                                                   #
# ------------------------------------------------------------------ #


class TestTrackAccess:
    """Test access tracking functionality."""

    def test_track_access_increments_count(self, scoring_store: VstashStore) -> None:
        """track_access should increment access_count."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 2").fetchall()
        ]
        scoring_store.track_access(chunk_ids)

        rows = scoring_store._conn.execute(
            f"SELECT access_count FROM chunks WHERE id IN ({','.join('?' * len(chunk_ids))})",
            chunk_ids,
        ).fetchall()
        assert all(row["access_count"] == 2 for row in rows)  # 1 (initial) + 1

    def test_track_access_sets_last_accessed(self, scoring_store: VstashStore) -> None:
        """track_access should set last_accessed_at."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchall()
        ]
        scoring_store.track_access(chunk_ids)

        row = scoring_store._conn.execute(
            "SELECT last_accessed_at FROM chunks WHERE id = ?", [chunk_ids[0]]
        ).fetchone()
        assert row["last_accessed_at"] is not None

    def test_track_access_multiple_times(self, scoring_store: VstashStore) -> None:
        """Multiple track_access calls should accumulate."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchall()
        ]
        for _ in range(5):
            scoring_store.track_access(chunk_ids)

        row = scoring_store._conn.execute(
            "SELECT access_count FROM chunks WHERE id = ?", [chunk_ids[0]]
        ).fetchone()
        assert row["access_count"] == 6  # 1 (initial) + 5

    def test_track_access_empty_list(self, scoring_store: VstashStore) -> None:
        """track_access with empty list should be a no-op."""
        scoring_store.track_access([])  # Should not raise


# ------------------------------------------------------------------ #
# Rerank with Decay Tests                                              #
# ------------------------------------------------------------------ #


class TestRerankWithDecay:
    """Test the rerank_with_decay scoring function."""

    def test_higher_rrf_wins_with_equal_access(self, scoring_store: VstashStore) -> None:
        """With equal access history, higher RRF should still rank first."""
        candidates = [
            {"id": 1, "rrf": 0.016, "text": "a", "title": "t", "path": "p", "chunk": 0},
            {"id": 2, "rrf": 0.009, "text": "b", "title": "t", "path": "p", "chunk": 1},
        ]
        result = scoring_store.rerank_with_decay(candidates)
        assert result[0]["id"] == 1

    def test_high_access_can_boost_lower_rrf(self, scoring_store: VstashStore) -> None:
        """A chunk with many recent accesses can overtake one with higher RRF."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 2").fetchall()
        ]
        # Give second chunk many accesses
        for _ in range(20):
            scoring_store.track_access([chunk_ids[1]])

        candidates = [
            {"id": chunk_ids[0], "rrf": 0.016, "text": "a", "title": "t", "path": "p", "chunk": 0},
            {"id": chunk_ids[1], "rrf": 0.014, "text": "b", "title": "t", "path": "p", "chunk": 1},
        ]
        result = scoring_store.rerank_with_decay(candidates, beta=0.5)
        # With enough accesses and beta=0.5, the boosted chunk should win
        assert result[0]["id"] == chunk_ids[1]

    def test_decay_reduces_old_access_influence(self, scoring_store: VstashStore) -> None:
        """Old accesses should have less influence than recent ones."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 2").fetchall()
        ]
        # Set chunk 1 with many accesses but long ago
        old_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        scoring_store._conn.execute(
            "UPDATE chunks SET access_count = 50, last_accessed_at = ? WHERE id = ?",
            [old_date, chunk_ids[0]],
        )
        scoring_store._conn.commit()

        # Set chunk 2 with few accesses but recent
        scoring_store.track_access([chunk_ids[1]])

        candidates = [
            {"id": chunk_ids[0], "rrf": 0.012, "text": "a", "title": "t", "path": "p", "chunk": 0},
            {"id": chunk_ids[1], "rrf": 0.012, "text": "b", "title": "t", "path": "p", "chunk": 1},
        ]
        result = scoring_store.rerank_with_decay(candidates, decay_lambda=0.1)
        # Equal RRF, but chunk 2 has recent access — should rank higher
        assert result[0]["id"] == chunk_ids[1]

    def test_empty_candidates(self, scoring_store: VstashStore) -> None:
        """Empty candidates should return empty list."""
        result = scoring_store.rerank_with_decay([])
        assert result == []

    def test_single_candidate(self, scoring_store: VstashStore) -> None:
        """Single candidate should get final_score."""
        chunk_id = scoring_store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
        candidates = [
            {"id": chunk_id, "rrf": 0.016, "text": "a", "title": "t", "path": "p", "chunk": 0},
        ]
        result = scoring_store.rerank_with_decay(candidates)
        assert "final_score" in result[0]
        assert result[0]["final_score"] > 0

    def test_min_max_scaling_normalizes_rrf(self, scoring_store: VstashStore) -> None:
        """Min-max scaling should normalize best RRF to 1.0 contribution."""
        chunk_ids = [
            r["id"] for r in scoring_store._conn.execute("SELECT id FROM chunks LIMIT 3").fetchall()
        ]
        candidates = [
            {"id": chunk_ids[0], "rrf": 0.020, "text": "a", "title": "t", "path": "p", "chunk": 0},
            {"id": chunk_ids[1], "rrf": 0.010, "text": "b", "title": "t", "path": "p", "chunk": 1},
            {"id": chunk_ids[2], "rrf": 0.005, "text": "c", "title": "t", "path": "p", "chunk": 2},
        ]
        # With beta=0 (no frequency component), scores should be purely normalized RRF
        result = scoring_store.rerank_with_decay(candidates, alpha=1.0, beta=0.0)
        assert result[0]["final_score"] == pytest.approx(1.0)  # max RRF → 1.0
        assert result[-1]["final_score"] == pytest.approx(0.0)  # min RRF → 0.0

    def test_clock_skew_protection(self, scoring_store: VstashStore) -> None:
        """Future dates should not inflate scores (days_ago clamped to 0)."""
        chunk_id = scoring_store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
        future_date = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        scoring_store._conn.execute(
            "UPDATE chunks SET last_accessed_at = ? WHERE id = ?",
            [future_date, chunk_id],
        )
        scoring_store._conn.commit()

        candidates = [
            {"id": chunk_id, "rrf": 0.016, "text": "a", "title": "t", "path": "p", "chunk": 0},
        ]
        result = scoring_store.rerank_with_decay(candidates)
        # Score should be reasonable, not astronomically inflated
        assert result[0]["final_score"] < 2.0


# ------------------------------------------------------------------ #
# ScoringConfig Tests                                                  #
# ------------------------------------------------------------------ #


class TestScoringConfig:
    """Test ScoringConfig validation and defaults."""

    def test_defaults(self) -> None:
        cfg = ScoringConfig()
        assert cfg.enabled is True
        assert cfg.alpha == 0.5
        assert cfg.beta == 0.5
        assert cfg.decay_lambda == 0.05
        assert cfg.over_fetch == 50
        assert cfg.track_access is True

    def test_from_toml_dict(self) -> None:
        raw = {
            "scoring": {
                "enabled": True,
                "alpha": 0.6,
                "beta": 0.4,
                "decay_lambda": 0.1,
                "over_fetch": 100,
                "track_access": False,
            }
        }
        cfg = VstashConfig.model_validate(raw)
        assert cfg.scoring.enabled is True
        assert cfg.scoring.alpha == 0.6
        assert cfg.scoring.over_fetch == 100

    def test_vstash_config_includes_scoring(self) -> None:
        cfg = VstashConfig()
        assert hasattr(cfg, "scoring")
        assert cfg.scoring.enabled is True
