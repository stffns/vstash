"""Tests for vstash journal — cross-session memory for agents."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tests.conftest import requires_sqlite_vec


pytestmark = requires_sqlite_vec


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _isolate_journal(tmp_path, monkeypatch):
    """Route journal to a temp directory so tests don't touch real profiles."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    monkeypatch.setattr("vstash.journal.PROFILES_DIR", profiles_dir)
    # Also patch profile module to match
    monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)


# ------------------------------------------------------------------ #
# journal_save                                                         #
# ------------------------------------------------------------------ #


class TestJournalSave:
    def test_save_basic(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("Test decision: use OAuth2 PKCE", source="test")
        assert result["status"] == "ok"
        assert result["chunks"] >= 1
        assert "journal" in result["tags"]
        assert "test" in result["tags"]

    def test_save_with_title(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("Some content", title="my-custom-title", source="test")
        assert result["title"] == "my-custom-title"

    def test_save_auto_title(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("Authentication uses PKCE flow for public clients")
        assert "journal" in result["title"]
        assert "Authentication" in result["title"]

    def test_save_empty_text(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("")
        assert result["status"] == "empty"

    def test_save_with_project(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("Content", project="myproj", source="test")
        assert result["project"] == "myproj"

    def test_save_with_tags(self) -> None:
        from vstash.journal import journal_save

        result = journal_save("Content", tags="auth,security", source="hook")
        tags = result["tags"]
        assert "journal" in tags
        assert "auth" in tags
        assert "security" in tags
        assert "hook" in tags


# ------------------------------------------------------------------ #
# journal_recall                                                       #
# ------------------------------------------------------------------ #


class TestJournalRecall:
    def test_recall_empty(self) -> None:
        from vstash.journal import journal_recall

        entries = journal_recall()
        assert entries == []

    def test_recall_recent(self) -> None:
        from vstash.journal import journal_save, journal_recall

        journal_save("First entry about authentication", source="test")
        journal_save("Second entry about deployment", source="test")

        entries = journal_recall()
        assert len(entries) >= 1

    def test_recall_with_query(self) -> None:
        from vstash.journal import journal_save, journal_recall

        journal_save("OAuth2 PKCE flow is used for mobile clients", source="test")
        journal_save("Deployment uses Kubernetes on GCP", source="test")

        entries = journal_recall(query="authentication OAuth")
        assert len(entries) >= 1
        # Should find the auth entry
        texts = " ".join(e["text"] for e in entries)
        assert "OAuth" in texts or "PKCE" in texts

    def test_recall_with_project_filter(self) -> None:
        from vstash.journal import journal_save, journal_recall

        journal_save("Project A content", project="proj-a", source="test")
        journal_save("Project B content", project="proj-b", source="test")

        entries = journal_recall(project="proj-a")
        assert len(entries) >= 1
        # Entries from proj-b should not appear
        for e in entries:
            assert "Project B" not in e.get("text", "")

    def test_recall_recent_with_tag_filter(self) -> None:
        """``tags=`` on the recent-entries branch restricts by exact
        tag membership (not substring)."""
        from vstash.journal import journal_recall, journal_save

        journal_save("Auth note about OAuth", tags="auth", source="test")
        journal_save("Auth-note longer", tags="authentication", source="test")
        journal_save("Deploy notes", tags="deploy", source="test")

        entries = journal_recall(tags="auth")
        titles = " | ".join(e.get("title", "") for e in entries)
        # The "auth" entry must appear; "authentication" must NOT
        # (comma-anchored membership, not substring).
        for e in entries:
            assert "authentication" not in (e.get("tags") or ""), (
                f"tag=auth false-matched 'authentication' entry: {e.get('tags')}"
            )
        assert any("auth" in (e.get("tags") or "").split(",") for e in entries), (
            f"expected the auth-tagged entry in results, got titles: {titles}"
        )

    def test_recall_query_with_tag_filter(self) -> None:
        """``tags=`` on the semantic-search branch flows through to
        ``store.search`` and constrains the candidate pool."""
        from vstash.journal import journal_recall, journal_save

        journal_save("Decision: switch from JWT to opaque tokens", tags="decision", source="test")
        journal_save("Note: explored JWT alternatives", tags="note", source="test")

        entries = journal_recall(query="tokens JWT", tags="decision")
        # The semantic-search branch returns ``text/title/score/added_at``
        # (no ``tags`` field surfaced), so we cannot assert per-entry tag
        # values. The functional guarantee is that the ``note``-tagged
        # entry must NOT appear in results -- its content alone would
        # otherwise win on relevance.
        assert all("alternatives" not in e.get("text", "") for e in entries), (
            f"tag=decision filter leaked the note-tagged entry: {[e.get('text') for e in entries]}"
        )


# ------------------------------------------------------------------ #
# journal_log                                                          #
# ------------------------------------------------------------------ #


class TestJournalLog:
    def test_log_empty(self) -> None:
        from vstash.journal import journal_log

        entries = journal_log()
        assert entries == []

    def test_log_shows_entries(self) -> None:
        from vstash.journal import journal_save, journal_log

        journal_save("Entry one", title="first", source="test")
        journal_save("Entry two", title="second", source="test")

        entries = journal_log()
        assert len(entries) == 2
        titles = [e["title"] for e in entries]
        assert "first" in titles
        assert "second" in titles

    def test_log_limit(self) -> None:
        from vstash.journal import journal_save, journal_log

        for i in range(5):
            journal_save(f"Entry {i}", title=f"entry-{i}", source="test")

        entries = journal_log(limit=3)
        assert len(entries) == 3

    def test_log_recent_filter(self) -> None:
        from vstash.journal import journal_save, journal_log

        journal_save("Recent entry", title="recent", source="test")

        # All entries were just created, so --recent 7d should include them
        entries = journal_log(recent="7d")
        assert len(entries) >= 1

        # 0h window should exclude everything (entries are at least a few ms old)
        # but since our timestamps are in seconds, 0h = now, so entries at same second pass.
        # Use 0d instead — entries created "today" are within 0 days ago.
        entries_none = journal_log(recent="0h")
        # May or may not include depending on timing — just verify no crash
        assert isinstance(entries_none, list)

    def test_log_recent_cli(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()
        runner.invoke(app, ["journal", "save", "For recent test", "--source", "test"])

        result = runner.invoke(app, ["journal", "log", "--recent", "7d"])
        assert result.exit_code == 0


# ------------------------------------------------------------------ #
# journal_prune                                                        #
# ------------------------------------------------------------------ #


class TestJournalPrune:
    def test_prune_dry_run(self) -> None:
        from vstash.journal import journal_save, journal_prune

        journal_save("Old content", source="test")

        result = journal_prune("0d", dry_run=True)
        assert result["status"] == "dry_run"
        assert result["would_delete"] >= 1

    def test_prune_nothing_old(self) -> None:
        from vstash.journal import journal_save, journal_prune

        journal_save("Recent content", source="test")

        result = journal_prune("30d")
        assert result["deleted"] == 0

    def test_prune_invalid_age(self) -> None:
        from vstash.journal import journal_prune

        with pytest.raises(ValueError, match="Invalid age format"):
            journal_prune("bad")

    def test_parse_age_formats(self) -> None:
        from vstash.journal import _parse_age

        assert _parse_age("30d") == timedelta(days=30)
        assert _parse_age("2w") == timedelta(weeks=2)
        assert _parse_age("24h") == timedelta(hours=24)


# ------------------------------------------------------------------ #
# parse_transcript                                                     #
# ------------------------------------------------------------------ #


class TestParseTranscript:
    def test_parse_empty_file(self, tmp_path) -> None:
        from vstash.journal import parse_transcript

        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert parse_transcript(str(f)) == ""

    def test_parse_nonexistent_file(self) -> None:
        from vstash.journal import parse_transcript

        assert parse_transcript("/nonexistent/path.jsonl") == ""

    def test_parse_extracts_user_messages(self, tmp_path) -> None:
        from vstash.journal import parse_transcript

        f = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "Add multi-profile support"}),
            json.dumps({"role": "assistant", "content": [{"type": "text", "text": "OK"}]}),
            json.dumps({"role": "user", "content": "Run the tests"}),
        ]
        f.write_text("\n".join(lines))

        result = parse_transcript(str(f))
        assert "Add multi-profile support" in result
        assert "Run the tests" in result

    def test_parse_extracts_file_edits(self, tmp_path) -> None:
        from vstash.journal import parse_transcript

        f = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "/src/main.py"},
                        }
                    ],
                }
            ),
        ]
        f.write_text("\n".join(lines))

        result = parse_transcript(str(f))
        assert "/src/main.py" in result


# ------------------------------------------------------------------ #
# CLI commands                                                         #
# ------------------------------------------------------------------ #


class TestJournalCLI:
    def test_save_and_log(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()

        # Save
        result = runner.invoke(app, ["journal", "save", "CLI test entry", "--source", "test"])
        assert result.exit_code == 0
        assert "Saved" in result.stdout

        # Log
        result = runner.invoke(app, ["journal", "log"])
        assert result.exit_code == 0
        assert "CLI test entry" in result.stdout or "journal" in result.stdout.lower()

    def test_recall_empty(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["journal", "recall"])
        assert result.exit_code == 0

    def test_recall_raw(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()

        # Save first
        runner.invoke(app, ["journal", "save", "Raw output test", "--source", "test"])

        # Recall with --raw
        result = runner.invoke(app, ["journal", "recall", "--raw"])
        assert result.exit_code == 0
        assert "Raw output test" in result.stdout

    def test_prune_dry_run(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()
        runner.invoke(app, ["journal", "save", "To prune", "--source", "test"])

        result = runner.invoke(app, ["journal", "prune", "0d", "--dry-run"])
        assert result.exit_code == 0
        assert "Would delete" in result.stdout


# ------------------------------------------------------------------ #
# SDK Memory.journal_* methods                                         #
# ------------------------------------------------------------------ #


class TestMemoryJournal:
    def test_memory_journal_save(self, monkeypatch) -> None:
        from unittest.mock import patch

        with (
            patch("vstash.memory.open_store_for_config"),
            patch("vstash.memory.load_config"),
        ):
            from vstash.memory import Memory

            mem = Memory(project="test-agent")

            # journal_save delegates to journal module

            with patch("vstash.journal.journal_save") as mock_save:
                mock_save.return_value = {"status": "ok", "title": "test", "chunks": 1}
                result = mem.journal_save("Test content", source="agent")
                assert result["status"] == "ok"
                mock_save.assert_called_once()
                # Verify project was passed through
                assert mock_save.call_args.kwargs["project"] == "test-agent"

            mem.close()

    def test_memory_journal_recall(self, monkeypatch) -> None:
        from unittest.mock import patch

        with (
            patch("vstash.memory.open_store_for_config"),
            patch("vstash.memory.load_config"),
        ):
            from vstash.memory import Memory

            mem = Memory(project="test-agent")

            with patch("vstash.journal.journal_recall") as mock_recall:
                mock_recall.return_value = [{"text": "context", "title": "entry"}]
                result = mem.journal_recall("query")
                assert len(result) == 1
                mock_recall.assert_called_once()
                assert mock_recall.call_args.kwargs["project"] == "test-agent"

            mem.close()

    def test_memory_journal_log(self, monkeypatch) -> None:
        from unittest.mock import patch

        with (
            patch("vstash.memory.open_store_for_config"),
            patch("vstash.memory.load_config"),
        ):
            from vstash.memory import Memory

            mem = Memory(project="test-agent")

            with patch("vstash.journal.journal_log") as mock_log:
                mock_log.return_value = [{"title": "entry", "added_at": "2026-01-01"}]
                result = mem.journal_log(limit=10, recent="7d")
                assert len(result) == 1
                mock_log.assert_called_once()
                assert mock_log.call_args.kwargs["project"] == "test-agent"
                assert mock_log.call_args.kwargs["limit"] == 10
                assert mock_log.call_args.kwargs["recent"] == "7d"

            mem.close()

    def test_memory_journal_prune(self, monkeypatch) -> None:
        from unittest.mock import patch

        with (
            patch("vstash.memory.open_store_for_config"),
            patch("vstash.memory.load_config"),
        ):
            from vstash.memory import Memory

            mem = Memory(project="test-agent")

            with patch("vstash.journal.journal_prune") as mock_prune:
                mock_prune.return_value = {"status": "ok", "deleted": 2}
                result = mem.journal_prune("30d", dry_run=True)
                assert result["status"] == "ok"
                mock_prune.assert_called_once()
                assert mock_prune.call_args.kwargs["project"] == "test-agent"
                assert mock_prune.call_args.kwargs["dry_run"] is True

            mem.close()
