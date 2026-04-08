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
            results = mem.add(doc)
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], IngestResult)
            assert results[0].status == "ok"
            assert results[0].chunks > 0

    @requires_sqlite_vec
    def test_add_with_project(self, tmp_path: Path) -> None:
        """Add with explicit project tag."""
        doc = tmp_path / "notes.txt"
        doc.write_text("Meeting notes about the project requirements and deadlines.")

        with Memory(db=tmp_path / "test.db") as mem:
            [result] = mem.add(doc, project="my_project")
            assert result.status == "ok"
            docs = mem.list(project="my_project")
            assert len(docs) >= 1

    @requires_sqlite_vec
    def test_add_duplicate_skipped(self, tmp_path: Path) -> None:
        """Adding the same file twice without force=True skips it."""
        doc = tmp_path / "dup.md"
        doc.write_text("Some content that should only be ingested once for testing.")

        with Memory(db=tmp_path / "test.db") as mem:
            [r1] = mem.add(doc)
            assert r1.status == "ok"
            [r2] = mem.add(doc)
            assert r2.status == "skipped"

    @requires_sqlite_vec
    def test_add_force_reingests(self, tmp_path: Path) -> None:
        """force=True re-ingests an existing document."""
        doc = tmp_path / "force.md"
        doc.write_text("Content that will be re-ingested with force flag enabled.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            [r2] = mem.add(doc, force=True)
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

    @requires_sqlite_vec
    def test_search_forwards_rrf_weights_to_store(self, tmp_path: Path) -> None:
        """Memory.search() must actually pass vec_weight/fts_weight to the store (#151).

        The most direct assertion that the plumbing works is to spy on
        `VstashStore.search` and verify the kwargs arrive with the
        exact values that were passed to `Memory.search`. A ranking-
        based test is flaky on small corpora because RRF tends to
        converge to the same top-1 regardless of weights when there
        is only one keyword-matching doc.
        """
        (tmp_path / "doc.md").write_text(
            "Rust is a systems programming language focused on memory "
            "safety and zero-cost abstractions."
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")

            captured: list[dict[str, object]] = []
            original_search = mem._store.search

            def spy_search(*args: object, **kwargs: object) -> list[SearchResult]:
                captured.append(dict(kwargs))
                return original_search(*args, **kwargs)

            mem._store.search = spy_search  # type: ignore[method-assign]

            mem.search("memory safety", vec_weight=0.9, fts_weight=0.1)
            mem.search("memory safety", vec_weight=0.1, fts_weight=0.9)
            mem.search("memory safety")  # adaptive default

            assert len(captured) == 3
            assert captured[0]["vec_weight"] == 0.9
            assert captured[0]["fts_weight"] == 0.1
            assert captured[1]["vec_weight"] == 0.1
            assert captured[1]["fts_weight"] == 0.9
            # Default path passes None for both so the store can run
            # adaptive RRF — this is the load-bearing assertion that
            # Memory.search does NOT silently pin a default.
            assert captured[2]["vec_weight"] is None
            assert captured[2]["fts_weight"] is None

    @requires_sqlite_vec
    def test_search_weights_rejects_out_of_range(self, tmp_path: Path) -> None:
        """Out-of-range RRF weights raise a typed error (#151)."""
        import pytest

        from vstash.validation import RRFWeightOutOfRangeError

        (tmp_path / "doc.md").write_text("Some content for the store so search can run at all.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", vec_weight=1.5)
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", fts_weight=-0.1)
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", vec_weight=0.7, fts_weight=2.0)

    @requires_sqlite_vec
    def test_ask_forwards_rrf_weights(self, tmp_path: Path) -> None:
        """Memory.ask() must forward vec_weight/fts_weight to the retrieval call."""
        import inspect

        sig = inspect.signature(Memory.ask)
        assert "vec_weight" in sig.parameters
        assert "fts_weight" in sig.parameters
        # Spy on the search call path via a subclass override.
        captured: dict[str, object] = {}

        class _SpyMemory(Memory):
            def search(self, query: str, **kwargs: object) -> list[SearchResult]:
                captured.update(kwargs)
                return []

        (tmp_path / "a.md").write_text("content for the spy test.")
        with _SpyMemory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "a.md")
            try:
                mem.ask("anything", vec_weight=0.3, fts_weight=0.7)
            except Exception:
                # Inference backend may not be configured in CI — that
                # is fine; we only care that search() was called with
                # the forwarded kwargs before the LLM step.
                pass
        assert captured.get("vec_weight") == 0.3
        assert captured.get("fts_weight") == 0.7


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


# ------------------------------------------------------------------ #
# SDK DB resolution consistency                                        #
# ------------------------------------------------------------------ #


class TestSdkDbResolution:
    """Verify Memory SDK uses the unified resolve_db_path chain."""

    def test_sdk_uses_resolve_db_path_with_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Memory() without explicit db= delegates to resolve_db_path."""
        from vstash.profile import resolve_db_path

        custom_db = tmp_path / "custom.db"
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        expected = str(resolve_db_path(config_db_path=str(custom_db)))

        with patch("vstash.memory.VstashStore") as mock_store:
            mock_store.return_value.close = lambda: None
            cfg = VstashConfig()
            # Simulate custom storage.db_path
            storage_cfg = cfg.storage.model_copy(update={"db_path": str(custom_db)})
            custom_cfg = cfg.model_copy(update={"storage": storage_cfg})
            with patch("vstash.memory._load_config_from", return_value=custom_cfg):
                mem = Memory()
                assert mock_store.call_args[0][0] == expected
                mem.close()

    def test_sdk_explicit_db_overrides_all(self, tmp_path: Path) -> None:
        """Memory(db=...) always uses that path, regardless of config."""
        explicit = str(tmp_path / "explicit.db")
        with patch("vstash.memory.VstashStore") as mock_store:
            mock_store.return_value.close = lambda: None
            mem = Memory(db=explicit)
            assert mock_store.call_args[0][0] == explicit
            mem.close()


# ------------------------------------------------------------------ #
# Dynamic chunk_size                                                   #
# ------------------------------------------------------------------ #


class TestDynamicChunkSize:
    """Test chunk_size/chunk_overlap parameters in Memory."""

    def test_constructor_chunk_size(self, tmp_db_path: str) -> None:
        """Constructor chunk_size is stored and used."""
        mem = Memory(db=tmp_db_path, chunk_size=2048, chunk_overlap=256)
        assert mem._chunk_size == 2048
        assert mem._chunk_overlap == 256
        mem.close()

    def test_default_chunk_size_is_none(self, tmp_db_path: str) -> None:
        mem = Memory(db=tmp_db_path)
        assert mem._chunk_size is None
        assert mem._chunk_overlap is None
        mem.close()

    def test_per_call_chunk_size_override(self, tmp_db_path: str) -> None:
        """Per-call chunk_size produces different chunk counts."""
        with Memory(db=tmp_db_path) as mem:
            # Small chunk_size → more chunks
            mem.remember("word " * 2000, title="small-chunks", chunk_size=256)
            stats1 = mem.stats()
            count_small = stats1.chunks

            mem.remember("word " * 2000, title="large-chunks", chunk_size=4096)
            stats2 = mem.stats()
            count_large = stats2.chunks - count_small

            assert count_small > count_large

    def test_constructor_chunk_size_used_by_add(self, tmp_db_path: str) -> None:
        """Constructor chunk_size flows through to add()."""
        with Memory(db=tmp_db_path, chunk_size=512) as mem:
            mem.add("tests/conftest.py")
            stats = mem.stats()
            chunk_count_512 = stats.chunks

        with Memory(db=tmp_db_path + "_large") as mem2:
            mem2.add("tests/conftest.py", chunk_size=4096)
            stats2 = mem2.stats()
            chunk_count_4096 = stats2.chunks
            mem2.close()

        # Smaller chunk_size should produce more chunks
        assert chunk_count_512 >= chunk_count_4096
