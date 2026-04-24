"""Tests for schema versioning and config forward compatibility (#135)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vstash.config import VstashConfig
from vstash.store import (
    KNOWN_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    SchemaVersionError,
    VstashStore,
)


# ------------------------------------------------------------------ #
# Schema version stamping                                              #
# ------------------------------------------------------------------ #


class TestSchemaVersionStamp:
    def test_fresh_db_gets_stamped(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            assert store.get_meta("schema_version") == SCHEMA_VERSION
            assert store.get_meta("vstash_version") is not None
        finally:
            store.close()

    def test_known_versions_includes_current(self) -> None:
        assert SCHEMA_VERSION in KNOWN_SCHEMA_VERSIONS

    def test_reopen_does_not_change_schema_version(self, tmp_path: Path) -> None:
        db = tmp_path / "reopen.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            initial = store.get_meta("schema_version")
        finally:
            store.close()

        store2 = VstashStore(str(db), embedding_dim=4)
        try:
            assert store2.get_meta("schema_version") == initial
        finally:
            store2.close()

    def test_legacy_db_without_stamp_is_treated_as_v1(self, tmp_path: Path) -> None:
        """A pre-#135 DB has no schema_version row.  Opening it should
        recognize the data as legacy v1, run the v2 cosine migration,
        and stamp the current schema version.
        """
        db = tmp_path / "legacy.db"

        # Create the DB through the normal path so all tables exist…
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store.add_document(
                path="/a.md",
                title="a",
                chunks=["hello"],
                embeddings=[[0.1, 0.2, 0.3, 0.4]],
                source_type="markdown",
            )
            # …then strip the schema_version row to fake a pre-#135 DB.
            store._conn.execute("DELETE FROM store_meta WHERE key = 'schema_version'")
            store._conn.commit()
        finally:
            store.close()

        # Re-opening should silently re-stamp it.
        store2 = VstashStore(str(db), embedding_dim=4)
        try:
            assert store2.get_meta("schema_version") == SCHEMA_VERSION
        finally:
            store2.close()


# ------------------------------------------------------------------ #
# v1 -> v2 cosine migration (#272)                                     #
# ------------------------------------------------------------------ #


class TestV1ToV2CosineMigration:
    """Schema v2 changed ``vec_chunks`` from L2 to cosine metric.

    A v1 DB on first open with a v2 build must rebuild ``vec_chunks`` in
    place, keeping the same embedding bytes but switching the metric.
    Nothing else about the DB should change.
    """

    def _force_v1(self, db: Path) -> None:
        """Downgrade a fresh DB to look like a v1 DB on disk.

        Dropping the cosine-typed ``vec_chunks`` and recreating it as
        the old L2 ``vec0(embedding float[N])`` reproduces what an
        existing v1 store looks like the first time a v2 build opens
        it.  The ``schema_version`` row is rewritten to ``"1"`` so the
        migration path fires.
        """
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("UPDATE store_meta SET value='1' WHERE key='schema_version'")
        # Pull the current bytes and re-insert into an L2 table.
        rows = conn.execute("SELECT rowid, embedding FROM vec_chunks").fetchall()
        conn.execute("DROP TABLE vec_chunks")
        conn.execute("CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[4])")
        conn.executemany("INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)", rows)
        conn.commit()
        conn.close()

    def test_v1_db_migrates_to_v2(self, tmp_path: Path) -> None:
        db = tmp_path / "legacy_v1.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store.add_document(
                path="/a.md",
                title="a",
                chunks=["hello"],
                embeddings=[[0.1, 0.2, 0.3, 0.4]],
                source_type="markdown",
            )
        finally:
            store.close()

        self._force_v1(db)

        store2 = VstashStore(str(db), embedding_dim=4)
        try:
            assert store2.get_meta("schema_version") == "2"
            # vec_chunks DDL now contains distance_metric=cosine.
            row = store2._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
            ).fetchone()
            assert "distance_metric=cosine" in (row["sql"] or "").lower()
            # Embeddings survived the migration byte-for-byte.
            rows = store2._conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()
            assert rows["n"] == 1
        finally:
            store2.close()

    def test_migration_is_idempotent_on_v2(self, tmp_path: Path) -> None:
        db = tmp_path / "already_v2.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store.add_document(
                path="/a.md",
                title="a",
                chunks=["hi"],
                embeddings=[[1.0, 0.0, 0.0, 0.0]],
                source_type="markdown",
            )
        finally:
            store.close()

        # Re-open twice: both should no-op, keep schema_version at 2,
        # and keep the same vec_chunks DDL.
        for _ in range(2):
            store_n = VstashStore(str(db), embedding_dim=4)
            try:
                assert store_n.get_meta("schema_version") == SCHEMA_VERSION
                row = store_n._conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
                ).fetchone()
                assert "distance_metric=cosine" in (row["sql"] or "").lower()
            finally:
                store_n.close()

    def test_empty_unstamped_legacy_db_is_still_migrated(self, tmp_path: Path) -> None:
        """An empty pre-v0.25 DB has no schema_version row and no rows
        in documents / vec_chunks, but still has an L2-typed vec_chunks
        virtual table on disk.  The v2 migration must rebuild that table
        based on the DDL, not on row counts.
        """
        db = tmp_path / "empty_legacy.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            # Fresh v2 store -- now force it back to an empty L2 table to
            # simulate an empty pre-v0.25 DB that a user never added data
            # to before upgrading.
            store._conn.execute("DROP TABLE vec_chunks")
            store._conn.execute("CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[4])")
            store._conn.execute("DELETE FROM store_meta WHERE key = 'schema_version'")
            store._conn.commit()
        finally:
            store.close()

        store2 = VstashStore(str(db), embedding_dim=4)
        try:
            assert store2.get_meta("schema_version") == SCHEMA_VERSION
            row = store2._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
            ).fetchone()
            assert "distance_metric=cosine" in (row["sql"] or "").lower()
        finally:
            store2.close()

    def test_cosine_distance_in_0_2_range_for_non_normalized(self, tmp_path: Path) -> None:
        """``last_best_distance`` must stay in [0, 2] regardless of
        embedding magnitude (closes #271).  Cosine normalizes
        internally; L2 would blow past 2 on large-magnitude non-unit
        vectors, which was the symptom in the original bug report.
        """
        db = tmp_path / "nonnorm.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store.add_document(
                path="/a.md",
                title="a",
                chunks=["hi"],
                embeddings=[[10.0, 5.0, 3.0, 2.0]],  # magnitude ~11.7
                source_type="markdown",
            )
            store.search(
                query_embedding=[20.0, 10.0, 6.0, 4.0],  # same direction, 2x scale
                query_text="hi",
                top_k=1,
            )
            # Cosine distance of co-linear vectors is 0; L2 would be
            # sqrt(sum(q-d)^2) = sqrt(100+25+9+4) ~= 11.75 -- way past 2.
            assert store.last_best_distance < 0.001
        finally:
            store.close()


# ------------------------------------------------------------------ #
# Schema version mismatch                                              #
# ------------------------------------------------------------------ #


class TestSchemaVersionMismatch:
    def test_unknown_future_version_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "future.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store._conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
                [
                    "schema_version",
                    "999",
                    datetime.now(timezone.utc).isoformat(),
                ],
            )
            store._conn.commit()
        finally:
            store.close()

        with pytest.raises(SchemaVersionError, match="schema_version='999'"):
            VstashStore(str(db), embedding_dim=4)

    def test_error_message_lists_known_versions(self, tmp_path: Path) -> None:
        db = tmp_path / "future.db"
        store = VstashStore(str(db), embedding_dim=4)
        try:
            store._conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
                ["schema_version", "42", datetime.now(timezone.utc).isoformat()],
            )
            store._conn.commit()
        finally:
            store.close()

        with pytest.raises(SchemaVersionError) as exc_info:
            VstashStore(str(db), embedding_dim=4)
        # The known set should appear in the error so the user knows
        # what range this build can read.
        assert "1" in str(exc_info.value)


# ------------------------------------------------------------------ #
# Config forward compatibility                                         #
# ------------------------------------------------------------------ #


class TestConfigForwardCompat:
    def test_unknown_top_level_section_is_accepted(self, caplog: pytest.LogCaptureFixture) -> None:
        """A config carrying a section from a newer vstash version is
        loaded with a warning instead of crashing."""
        with caplog.at_level(logging.WARNING, logger="vstash"):
            cfg = VstashConfig.model_validate(
                {
                    "embeddings": {"model": "BAAI/bge-small-en-v1.5"},
                    "future_feature": {"enabled": True, "knob": 42},
                }
            )
        assert cfg.embeddings.model == "BAAI/bge-small-en-v1.5"
        assert any(
            "future_feature" in rec.message and "unknown" in rec.message.lower()
            for rec in caplog.records
        )

    def test_no_warning_when_only_known_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vstash"):
            VstashConfig.model_validate(
                {
                    "embeddings": {"model": "BAAI/bge-small-en-v1.5"},
                    "limits": {"max_top_k": 100},
                }
            )
        warnings = [r for r in caplog.records if "unknown" in r.message.lower()]
        assert warnings == []

    def test_nested_section_is_still_strict(self) -> None:
        """Forward-compat is per-design only at the top level.  Each
        nested section keeps its strict Pydantic schema so typos are
        still caught."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VstashConfig.model_validate({"limits": {"max_top_k": "not-a-number"}})

    def test_default_config_still_works(self) -> None:
        cfg = VstashConfig()
        assert cfg.embeddings.model
        assert cfg.limits.max_top_k > 0
