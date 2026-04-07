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
        stamp it as v1, not crash."""
        db = tmp_path / "legacy.db"

        # Create the DB through the normal path so all tables exist…
        store = VstashStore(str(db), embedding_dim=4)
        try:
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
