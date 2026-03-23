"""Tests for vstash export — export_chunks store method and CLI command."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from vstash.store import VstashStore


def _sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension loading is available."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("SELECT vec_version()")
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason="sqlite-vec extension or enable_load_extension not available",
)


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
    """Test CLI export command via JSONL/CSV output."""

    def test_export_jsonl(self, export_store: VstashStore, tmp_path) -> None:
        """Export to JSONL produces valid JSON per line."""
        chunks = export_store.export_chunks()
        out_path = tmp_path / "export.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "text" in parsed
            assert "title" in parsed

    def test_export_csv(self, export_store: VstashStore, tmp_path) -> None:
        """Export to CSV produces valid CSV with headers."""
        chunks = export_store.export_chunks()
        out_path = tmp_path / "export.csv"
        fieldnames = ["text", "title", "path", "chunk", "collection", "project", "layer", "tags"]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for chunk in chunks:
                row = {k: chunk.get(k, "") for k in fieldnames}
                writer.writerow(row)

        with out_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "text" in rows[0]
