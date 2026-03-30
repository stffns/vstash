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
        assert "note" != result.title
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
