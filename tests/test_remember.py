"""Tests for vstash remember — direct text ingestion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vstash.ingest import ingest_text


class TestIngestText:
    """Test the ingest_text function."""

    def test_empty_text_returns_empty(self) -> None:
        cfg = MagicMock()
        store = MagicMock()
        result = ingest_text("", "title", cfg, store)
        assert result.status == "empty"

    def test_whitespace_only_returns_empty(self) -> None:
        cfg = MagicMock()
        store = MagicMock()
        result = ingest_text("   \n  ", "title", cfg, store)
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
            "arch-notes",
            cfg,
            store,
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

        result = ingest_text(text, "api-notes", cfg, store)
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

        result = ingest_text(
            text, "note", cfg, store, project="override-project"
        )
        assert result.status == "ok"

        call_kwargs = store.add_document.call_args
        assert call_kwargs[1]["project"] == "override-project"


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
