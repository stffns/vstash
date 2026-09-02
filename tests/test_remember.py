"""Tests for vstash remember — direct text ingestion."""

from __future__ import annotations

import re

from unittest.mock import MagicMock, patch

from vstash.ingest import ingest_text


class TestIngestText:
    """Test the ingest_text function."""

    def test_empty_text_returns_empty(self) -> None:
        cfg = MagicMock()
        store = MagicMock()
        result = ingest_text("", cfg, store, title="title")
        assert result.status == "empty"

    def test_whitespace_only_returns_empty(self) -> None:
        cfg = MagicMock()
        store = MagicMock()
        result = ingest_text("   \n  ", cfg, store, title="title")
        assert result.status == "empty"

    @patch("vstash.ingest._embed_with_progress")
    def test_basic_text_ingestion(self, mock_embed) -> None:
        mock_embed.return_value = [[0.1] * 384]

        store = MagicMock()
        store.add_document.return_value = "abc123"

        cfg = MagicMock()
        cfg.chunking.size = 1024
        cfg.chunking.overlap = 128
        cfg.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = ingest_text(
            "This is some important text about architecture decisions.",
            cfg,
            store,
            title="arch-notes",
            collection="docs",
            project="myproj",
        )

        assert result.status == "ok"
        assert result.title == "arch-notes"
        assert result.chunks >= 1
        assert result.doc_id == "abc123"

        # Verify store was called with synthetic path
        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["path"] == "text://arch-notes"
        assert call_kwargs[1]["source_type"] == "text"
        assert call_kwargs[1]["collection"] == "docs"
        assert call_kwargs[1]["project"] == "myproj"

    @patch("vstash.ingest._embed_with_progress")
    def test_frontmatter_extraction(self, mock_embed) -> None:
        mock_embed.return_value = [[0.1] * 384]

        store = MagicMock()
        store.add_document.return_value = "def456"

        cfg = MagicMock()
        cfg.chunking.size = 1024
        cfg.chunking.overlap = 128
        cfg.embeddings.model = "BAAI/bge-small-en-v1.5"

        text = """---
project: my-project
tags: [api, design]
---

# API Design Notes

The API uses REST with JSON payloads."""

        result = ingest_text(text, cfg, store, title="api-notes")
        assert result.status == "ok"

        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["project"] == "my-project"
        assert call_kwargs[1]["tags"] == "api,design"

    @patch("vstash.ingest._embed_with_progress")
    def test_metadata_params_override_frontmatter(self, mock_embed) -> None:
        mock_embed.return_value = [[0.1] * 384]

        store = MagicMock()
        store.add_document.return_value = "ghi789"

        cfg = MagicMock()
        cfg.chunking.size = 1024
        cfg.chunking.overlap = 128
        cfg.embeddings.model = "BAAI/bge-small-en-v1.5"

        text = """---
project: frontmatter-project
---

This is enough content to generate at least one chunk for the embedding pipeline to process correctly."""

        result = ingest_text(text, cfg, store, title="note", project="override-project")
        assert result.status == "ok"

        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["project"] == "override-project"

    @patch("vstash.ingest._embed_with_progress")
    def test_auto_generated_title(self, mock_embed) -> None:
        """When title is None, a descriptive title is generated from text."""
        mock_embed.return_value = [[0.1] * 384]

        store = MagicMock()
        store.add_document.return_value = "auto123"

        cfg = MagicMock()
        cfg.chunking.size = 1024
        cfg.chunking.overlap = 128
        cfg.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = ingest_text(
            "OAuth2 uses PKCE for public clients",
            cfg,
            store,
        )

        assert result.status == "ok"
        # Title should contain slugified words from content, not "note"
        assert result.title != "note"
        assert "oauth2" in result.title
        # Source path should match
        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["path"].startswith("text://oauth2")

    @patch("vstash.ingest._embed_with_progress")
    def test_explicit_title_preserved(self, mock_embed) -> None:
        """When title is explicitly provided, it should be used as-is."""
        mock_embed.return_value = [[0.1] * 384]

        store = MagicMock()
        store.add_document.return_value = "expl123"

        cfg = MagicMock()
        cfg.chunking.size = 1024
        cfg.chunking.overlap = 128
        cfg.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = ingest_text(
            "Some content here that is long enough to be chunked properly by the ingestion pipeline.",
            cfg,
            store,
            title="my-custom-title",
        )

        assert result.status == "ok"
        assert result.title == "my-custom-title"
        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["path"] == "text://my-custom-title"


class TestGenerateTitle:
    """Test the _generate_title helper."""

    def test_generates_slug_from_content(self) -> None:
        from vstash.ingest import _generate_title

        title = _generate_title("OAuth2 uses PKCE for public clients")
        assert title.startswith("oauth2-uses-pkce-for-public-")

    def test_strips_special_characters(self) -> None:
        from vstash.ingest import _generate_title

        title = _generate_title("Hello, world! @#$ Test 123")
        assert "hello" in title
        assert "@" not in title

    def test_limits_to_five_words(self) -> None:
        from vstash.ingest import _generate_title

        title = _generate_title("one two three four five six seven eight")
        # Strip the YYYYMMDD-HHMMSSffffff timestamp suffix
        slug_part = re.sub(r"-\d{8}-\d{12,}$", "", title)
        word_parts = slug_part.split("-")
        assert len(word_parts) == 5

    def test_fallback_for_empty_slug(self) -> None:
        from vstash.ingest import _generate_title

        title = _generate_title("!!! @@@")
        assert title.startswith("note-")

    def test_includes_microseconds(self) -> None:
        from vstash.ingest import _generate_title

        title = _generate_title("test content")
        # Timestamp should include microseconds (14+ digits after last slug word)
        match = re.search(r"-(\d{8}-\d{12,})$", title)
        assert match is not None


class TestRememberContentHashDedup:
    """Tier-0 write-time dedup (memory-manager design doc, Phase 1
    precursor): a byte-identical re-remember of the same (collection,
    title) is a NOOP -- no chunking, no embedding, no store write.
    Changed text still replaces; legacy rows without a stored hash and
    partial copies are never wrongly skipped."""

    TEXT = "The aardvark memo: decision record about local coder models."

    def _remember(self, mem, text: str):
        return mem.remember(text, title="dedup-note")

    def test_identical_re_remember_is_skipped(self, tmp_path) -> None:
        from tests.conftest import requires_sqlite_vec  # noqa: F401
        from vstash.memory import Memory

        with Memory(db=tmp_path / "t.db") as mem:
            r1 = self._remember(mem, self.TEXT)
            assert r1.status == "ok"
            chunk_count_before = mem._store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                0
            ]

            # The skip path must not touch the embed pipeline at all.
            with patch(
                "vstash.ingest._embed_with_progress",
                side_effect=AssertionError("dedup skip must not embed"),
            ):
                r2 = self._remember(mem, self.TEXT)
            assert r2.status == "skipped"
            assert r2.source == "text://dedup-note"

            chunk_count_after = mem._store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                0
            ]
            assert chunk_count_after == chunk_count_before

    def test_changed_text_replaces(self, tmp_path) -> None:
        from vstash.memory import Memory

        with Memory(db=tmp_path / "t.db") as mem:
            assert self._remember(mem, self.TEXT).status == "ok"
            r2 = self._remember(mem, self.TEXT + " Updated with a zebra clause.")
            assert r2.status == "ok"
            texts = [
                row[0] for row in mem._store._conn.execute("SELECT text FROM chunks").fetchall()
            ]
            assert any("zebra" in t for t in texts), "changed text must replace the old copy"

    def test_legacy_row_without_hash_is_not_skipped(self, tmp_path) -> None:
        """Rows written before the content_hash column (or by paths
        that don't set it) must never be treated as identical."""
        from vstash.memory import Memory

        with Memory(db=tmp_path / "t.db") as mem:
            assert self._remember(mem, self.TEXT).status == "ok"
            # Fake a legacy row: NULL the stored hash.
            mem._store._conn.execute("UPDATE documents SET content_hash = NULL")
            mem._store._conn.commit()

            r2 = self._remember(mem, self.TEXT)
            assert r2.status == "ok", "legacy row must re-ingest, not skip"
            # The re-ingest stored the hash, so the THIRD call skips.
            r3 = self._remember(mem, self.TEXT)
            assert r3.status == "skipped"

    def test_partial_copy_is_healed_not_skipped(self, tmp_path) -> None:
        """A hash match on a PARTIAL copy (crash mid-ingest) must fall
        through to re-ingest, mirroring the file-ingest healing path."""
        from vstash.memory import Memory

        with Memory(db=tmp_path / "t.db") as mem:
            assert self._remember(mem, self.TEXT).status == "ok"
            # Simulate a torn ingest: declared chunk_count no longer
            # matches the actual chunk rows.
            mem._store._conn.execute("UPDATE documents SET chunk_count = chunk_count + 1")
            mem._store._conn.commit()
            assert (
                mem._store.doc_completeness("text://dedup-note", collection="default") == "partial"
            )

            r2 = self._remember(mem, self.TEXT)
            assert r2.status == "ok", "partial copy must heal via re-ingest"
            assert (
                mem._store.doc_completeness("text://dedup-note", collection="default") == "complete"
            )

    def test_content_hash_column_migrates_on_open(self, tmp_path) -> None:
        """An existing DB without the content_hash column gains it on
        the next open (additive ALTER migration)."""
        import sqlite3

        from vstash.memory import Memory

        if sqlite3.sqlite_version_info < (3, 35, 0):
            import pytest

            pytest.skip("ALTER TABLE DROP COLUMN needs sqlite >= 3.35")

        db = tmp_path / "t.db"
        with Memory(db=db) as mem:
            assert self._remember(mem, self.TEXT).status == "ok"
        # Strip the column to fake a pre-migration DB.
        conn = sqlite3.connect(str(db))
        conn.execute("ALTER TABLE documents DROP COLUMN content_hash")
        conn.commit()
        conn.close()

        with Memory(db=db) as mem:
            cols = {
                row[1]
                for row in mem._store._conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            assert "content_hash" in cols
            # Hash is NULL post-migration, so the same text re-ingests
            # (no false skip) and re-stamps the hash.
            assert self._remember(mem, self.TEXT).status == "ok"
            assert self._remember(mem, self.TEXT).status == "skipped"


class TestRememberCLI:
    """Test the vstash remember CLI command."""

    def test_remember_help(self) -> None:
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["remember", "--help"])
        assert result.exit_code == 0
        assert "Ingest text directly" in result.output

    def test_remember_no_input(self) -> None:
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        # No argument and no stdin → should fail
        result = runner.invoke(app, ["remember"])
        assert result.exit_code != 0
