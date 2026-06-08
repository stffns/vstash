"""Tests for vstash export — export_chunks store method and CLI command."""

from __future__ import annotations

import csv
import json

import pytest

from tests._cli_helpers import patch_cli_attr

from tests.conftest import requires_sqlite_vec
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec


@pytest.fixture
def export_store(tmp_db_path: str) -> VstashStore:
    """Store with documents in different projects/layers for export tests."""
    store = VstashStore(tmp_db_path, embedding_dim=384)
    dim = 384

    store.add_document(
        path="/test/paper1.pdf",
        title="Research Paper 1",
        chunks=["Neural networks learn representations.", "Attention is all you need."],
        embeddings=[[0.1] * dim, [0.2] * dim],
        source_type="pdf",
        collection="research",
        project="ml",
        layer="papers",
        tags="deep-learning,transformers",
    )
    store.add_document(
        path="/test/readme.md",
        title="Project Readme",
        chunks=["This project uses Python and FastAPI."],
        embeddings=[[0.3] * dim],
        source_type="markdown",
        collection="default",
        project="webapp",
        layer="docs",
        tags="python,api",
    )
    store.add_document(
        path="/test/notes.md",
        title="Meeting Notes",
        chunks=["Discussed deployment strategy.", "Agreed on weekly sprints."],
        embeddings=[[0.4] * dim, [0.5] * dim],
        source_type="markdown",
        collection="default",
        project="webapp",
        layer="notes",
    )

    yield store  # type: ignore[misc]
    store.close()


class TestExportChunks:
    """Test export_chunks store method."""

    def test_export_all_chunks(self, export_store: VstashStore) -> None:
        """Export all chunks without filters."""
        result = export_store.export_chunks()
        assert len(result) == 5
        assert all("text" in r for r in result)
        assert all("title" in r for r in result)
        assert all("project" in r for r in result)

    def test_export_filter_by_project(self, export_store: VstashStore) -> None:
        """Filter export by project."""
        result = export_store.export_chunks(project="ml")
        assert len(result) == 2
        assert all(r["project"] == "ml" for r in result)

    def test_export_filter_by_layer(self, export_store: VstashStore) -> None:
        """Filter export by layer."""
        result = export_store.export_chunks(layer="papers")
        assert len(result) == 2
        assert all(r["layer"] == "papers" for r in result)

    def test_export_filter_by_collection(self, export_store: VstashStore) -> None:
        """Filter export by collection."""
        result = export_store.export_chunks(collection="research")
        assert len(result) == 2

    def test_export_filter_by_tags(self, export_store: VstashStore) -> None:
        """Filter export by tags substring."""
        result = export_store.export_chunks(tags="python")
        assert len(result) == 1
        assert result[0]["title"] == "Project Readme"

    def test_export_combined_filters(self, export_store: VstashStore) -> None:
        """Combine project and layer filters."""
        result = export_store.export_chunks(project="webapp", layer="docs")
        assert len(result) == 1
        assert result[0]["title"] == "Project Readme"

    def test_export_no_match(self, export_store: VstashStore) -> None:
        """Empty filters return empty list."""
        result = export_store.export_chunks(project="nonexistent")
        assert result == []

    def test_export_with_embeddings(self, export_store: VstashStore) -> None:
        """Include embedding vectors in export."""
        result = export_store.export_chunks(project="ml", include_embeddings=True)
        assert len(result) == 2
        assert "embedding" in result[0]
        assert isinstance(result[0]["embedding"], list)
        assert len(result[0]["embedding"]) == 384

    def test_export_contains_metadata(self, export_store: VstashStore) -> None:
        """Verify all metadata fields are present."""
        result = export_store.export_chunks(project="ml")
        chunk = result[0]
        assert chunk["project"] == "ml"
        assert chunk["layer"] == "papers"
        assert chunk["collection"] == "research"
        assert "deep-learning" in (chunk["tags"] or "")

    def test_export_chunk_ordering(self, export_store: VstashStore) -> None:
        """Chunks within a doc are ordered by seq."""
        result = export_store.export_chunks(project="ml")
        assert result[0]["chunk"] < result[1]["chunk"]


class TestExportCLI:
    """Test CLI export command using typer CliRunner."""

    @pytest.fixture(autouse=True)
    def _patch_store(self, export_store: VstashStore, monkeypatch) -> None:
        """Monkeypatch _get_store so CLI uses the test store."""
        from vstash.config import load_config

        patch_cli_attr(monkeypatch, "_get_store",
            lambda cfg=None, warm=False, profile=None: (cfg or load_config(), export_store),
        )

    def test_cli_export_jsonl(self, tmp_path) -> None:
        """CLI export to JSONL creates valid file."""
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        out_file = str(tmp_path / "out.jsonl")

        result = runner.invoke(app, ["export", "-o", out_file])
        assert result.exit_code == 0
        assert "Exported 5 chunks" in result.stdout

        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5
        parsed = json.loads(lines[0])
        assert "text" in parsed
        assert "title" in parsed

    def test_cli_export_csv(self, tmp_path) -> None:
        """CLI export to CSV creates valid file with headers."""
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        out_file = str(tmp_path / "out.csv")

        result = runner.invoke(app, ["export", "-o", out_file, "-f", "csv"])
        assert result.exit_code == 0
        assert "Exported 5 chunks" in result.stdout

        with (tmp_path / "out.csv").open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "text" in rows[0]
            # None values should be empty strings, not "None"
            for row in rows:
                assert row.get("tags", "") != "None"

    def test_cli_export_with_filter(self, tmp_path) -> None:
        """CLI export with --project filter exports only matching chunks."""
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        out_file = str(tmp_path / "filtered.jsonl")

        result = runner.invoke(app, ["export", "-o", out_file, "--project", "ml"])
        assert result.exit_code == 0
        assert "Exported 2 chunks" in result.stdout

    def test_cli_export_invalid_format(self, tmp_path) -> None:
        """CLI export with invalid format raises error."""
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        out_file = str(tmp_path / "bad.txt")

        result = runner.invoke(app, ["export", "-o", out_file, "-f", "xml"])
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout

    def test_cli_export_no_match(self, tmp_path) -> None:
        """CLI export with no matching chunks exits with error."""
        from typer.testing import CliRunner
        from vstash.cli import app

        runner = CliRunner()
        out_file = str(tmp_path / "empty.jsonl")

        result = runner.invoke(app, ["export", "-o", out_file, "--project", "nonexistent"])
        assert result.exit_code == 1
        assert "No chunks match" in result.stdout
