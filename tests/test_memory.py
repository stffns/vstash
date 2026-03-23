"""Tests for vstash.memory — the Python SDK (Memory class)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vstash.config import VstashConfig
from vstash.memory import Memory, _load_config_from
from vstash.models import DocumentInfo, IngestResult, SearchResult, StoreStats

# All tests that touch the store need sqlite-vec.
# Import the skip marker from conftest.
from tests.conftest import requires_sqlite_vec


# ------------------------------------------------------------------ #
# Config loader                                                        #
# ------------------------------------------------------------------ #


class TestLoadConfig:
    """Test config resolution for the Memory class."""

    def test_load_default_config(self) -> None:
        """No path → falls back to load_config() defaults."""
        cfg = _load_config_from(None)
        assert isinstance(cfg, VstashConfig)

    def test_load_explicit_path(self, tmp_path: Path) -> None:
        """Explicit path to a vstash.toml file loads correctly."""
        toml = tmp_path / "vstash.toml"
        toml.write_text('[inference]\nbackend = "ollama"\nmodel = "test-model"\n')
        cfg = _load_config_from(str(toml))
        assert cfg.inference.backend == "ollama"
        assert cfg.inference.model == "test-model"

    def test_load_nonexistent_path_raises(self) -> None:
        """Non-existent explicit config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_config_from("/nonexistent/vstash.toml")


# ------------------------------------------------------------------ #
# Memory constructor                                                   #
# ------------------------------------------------------------------ #


class TestMemoryInit:
    """Test Memory class initialization."""

    @requires_sqlite_vec
    def test_basic_init(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        mem = Memory(db=db)
        try:
            assert mem._store is not None
            assert mem._project is None
            assert mem._collection == "default"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_init_with_project(self, tmp_path: Path) -> None:
        mem = Memory(db=tmp_path / "test.db", project="agent_x")
        try:
            assert mem._project == "agent_x"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_init_with_collection(self, tmp_path: Path) -> None:
        mem = Memory(db=tmp_path / "test.db", collection="research")
        try:
            assert mem._collection == "research"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            s = mem.stats()
            assert isinstance(s, StoreStats)
        # After exiting, store should be closed
        # (accessing it would raise, but we just verify no error on exit)


# ------------------------------------------------------------------ #
# Memory.add                                                           #
# ------------------------------------------------------------------ #


class TestMemoryAdd:
    """Test document ingestion via Memory.add()."""

    @requires_sqlite_vec
    def test_add_file(self, tmp_path: Path) -> None:
        """Add a text file and verify it's ingested."""
        doc = tmp_path / "test.md"
        doc.write_text("# Hello\n\nThis is a test document with enough content to chunk.")

        with Memory(db=tmp_path / "test.db") as mem:
            result = mem.add(doc)
            assert isinstance(result, IngestResult)
            assert result.status == "ok"
            assert result.chunks > 0

    @requires_sqlite_vec
    def test_add_with_project(self, tmp_path: Path) -> None:
        """Add with explicit project tag."""
        doc = tmp_path / "notes.txt"
        doc.write_text("Meeting notes about the project requirements and deadlines.")

        with Memory(db=tmp_path / "test.db") as mem:
            result = mem.add(doc, project="my_project")
            assert result.status == "ok"
            docs = mem.list(project="my_project")
            assert len(docs) >= 1

    @requires_sqlite_vec
    def test_add_duplicate_skipped(self, tmp_path: Path) -> None:
        """Adding the same file twice without force=True skips it."""
        doc = tmp_path / "dup.md"
        doc.write_text("Some content that should only be ingested once for testing.")

        with Memory(db=tmp_path / "test.db") as mem:
            r1 = mem.add(doc)
            assert r1.status == "ok"
            r2 = mem.add(doc)
            assert r2.status == "skipped"

    @requires_sqlite_vec
    def test_add_force_reingests(self, tmp_path: Path) -> None:
        """force=True re-ingests an existing document."""
        doc = tmp_path / "force.md"
        doc.write_text("Content that will be re-ingested with force flag enabled.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            r2 = mem.add(doc, force=True)
            assert r2.status == "ok"


# ------------------------------------------------------------------ #
# Memory.search                                                        #
# ------------------------------------------------------------------ #


class TestMemorySearch:
    """Test semantic search via Memory.search()."""

    @requires_sqlite_vec
    def test_search_returns_results(self, tmp_path: Path) -> None:
        doc = tmp_path / "python.md"
        doc.write_text(
            "# Python Programming\n\n"
            "Python is a high-level programming language known for simplicity and readability. "
            "It supports object-oriented, functional, and procedural programming paradigms."
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            results = mem.search("programming language")
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].score > 0

    @requires_sqlite_vec
    def test_search_empty_store(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            results = mem.search("anything")
            assert results == []

    @requires_sqlite_vec
    def test_search_top_k(self, tmp_path: Path) -> None:
        doc = tmp_path / "multi.md"
        doc.write_text(
            "\n\n".join(
                f"## Section {i}\n\nContent about topic {i} with enough detail." * 5
                for i in range(10)
            )
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            results = mem.search("topic", top_k=3)
            assert len(results) <= 3


# ------------------------------------------------------------------ #
# Memory.remove                                                        #
# ------------------------------------------------------------------ #


class TestMemoryRemove:
    """Test document removal via Memory.remove()."""

    @requires_sqlite_vec
    def test_remove_existing(self, tmp_path: Path) -> None:
        doc = tmp_path / "remove_me.md"
        doc.write_text("This document will be removed after ingestion for testing.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            assert len(mem.list()) >= 1
            removed = mem.remove(str(doc.resolve()))
            assert removed is True

    @requires_sqlite_vec
    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            removed = mem.remove("/nonexistent/file.pdf")
            assert removed is False


# ------------------------------------------------------------------ #
# Memory.list and Memory.stats                                         #
# ------------------------------------------------------------------ #


class TestMemoryListAndStats:
    """Test listing and statistics."""

    @requires_sqlite_vec
    def test_list_empty(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            docs = mem.list()
            assert docs == []

    @requires_sqlite_vec
    def test_list_returns_document_info(self, tmp_path: Path) -> None:
        doc = tmp_path / "listed.md"
        doc.write_text("A document that should appear in the list of ingested files.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            docs = mem.list()
            assert len(docs) >= 1
            assert isinstance(docs[0], DocumentInfo)

    @requires_sqlite_vec
    def test_stats_returns_store_stats(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            s = mem.stats()
            assert isinstance(s, StoreStats)
            assert s.documents == 0
            assert s.chunks == 0


# ------------------------------------------------------------------ #
# Memory with project scoping                                          #
# ------------------------------------------------------------------ #


class TestMemoryProjectScoping:
    """Test that constructor-level project/collection filters work."""

    @requires_sqlite_vec
    def test_project_filter_on_search(self, tmp_path: Path) -> None:
        """Search with project scoping only returns matching docs."""
        doc_a = tmp_path / "doc_a.md"
        doc_a.write_text("Alpha project documentation about the API design and architecture.")
        doc_b = tmp_path / "doc_b.md"
        doc_b.write_text("Beta project documentation about the deployment and infrastructure.")

        db = tmp_path / "test.db"

        # Ingest with different projects using a base memory
        with Memory(db=db) as mem:
            mem.add(doc_a, project="alpha")
            mem.add(doc_b, project="beta")

        # Search scoped to alpha
        with Memory(db=db, project="alpha") as mem_alpha:
            results = mem_alpha.search("documentation")
            paths = {r.path for r in results}
            # Should find alpha, not beta
            assert any("doc_a" in p for p in paths)

    @requires_sqlite_vec
    def test_db_override(self, tmp_path: Path) -> None:
        """db= parameter overrides the config db_path."""
        custom_db = tmp_path / "custom.db"
        with Memory(db=custom_db) as mem:
            s = mem.stats()
            assert str(custom_db) in s.db_path

    @requires_sqlite_vec
    def test_explicit_none_overrides_project(self, tmp_path: Path) -> None:
        """Passing project=None explicitly clears the constructor default."""
        doc = tmp_path / "doc.md"
        doc.write_text("Content for sentinel override testing across projects.")

        db = tmp_path / "test.db"
        with Memory(db=db) as mem:
            mem.add(doc, project="alpha")

        # Memory scoped to alpha, but explicit None should search all projects
        with Memory(db=db, project="alpha") as mem:
            scoped = mem.search("content")
            unscoped = mem.search("content", project=None)
            # Both should find the doc (only one project here),
            # but the key is that project=None doesn't raise
            assert len(unscoped) >= len(scoped)


# ------------------------------------------------------------------ #
# Memory.ask                                                           #
# ------------------------------------------------------------------ #


class TestMemoryAsk:
    """Test LLM-backed ask via Memory.ask()."""

    @requires_sqlite_vec
    def test_ask_calls_chat(self, tmp_path: Path) -> None:
        """ask() retrieves chunks and passes them to chat.ask."""
        doc = tmp_path / "askable.md"
        doc.write_text("Python is a high-level language for general purpose programming.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            with patch("vstash.memory._chat_ask", return_value="mocked answer") as mock:
                answer = mem.ask("what is python?")
                assert answer == "mocked answer"
                mock.assert_called_once()
                # First arg is the query
                assert mock.call_args[0][0] == "what is python?"
                # Second arg is the chunks list
                assert isinstance(mock.call_args[0][1], list)
