"""Tests for vstash.mcp — MCP server tool functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vstash.models import DocumentInfo, IngestResult, SearchResult, StoreStats
from vstash.mcp import (
    _error,
    _ok,
    vstash_add,
    vstash_ask,
    vstash_forget,
    vstash_list,
    vstash_search,
    vstash_stats,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _reset_singletons() -> None:
    """Reset module-level singletons between tests."""
    import vstash.mcp as mcp_mod

    mcp_mod._config = None
    mcp_mod._store = None


@pytest.fixture(autouse=True)
def _clean_singletons() -> None:
    """Ensure singletons are reset before each test."""
    _reset_singletons()
    yield  # type: ignore[misc]
    _reset_singletons()


def _make_search_result(text: str = "chunk text", title: str = "Doc") -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(text=text, title=title, path="/test/doc.md", chunk=0, score=0.5)


def _make_doc_info(path: str = "/test/doc.md", title: str = "Doc") -> DocumentInfo:
    """Helper to create a DocumentInfo for testing."""
    return DocumentInfo(
        path=path,
        title=title,
        source_type="markdown",
        chunk_count=3,
        char_count=500,
        added_at="2026-01-01T00:00:00+00:00",
    )


# ------------------------------------------------------------------ #
# Serialization helpers                                                #
# ------------------------------------------------------------------ #


class TestSerializationHelpers:
    """Test _ok() and _error() JSON wrappers."""

    def test_ok_with_dict(self) -> None:
        result = json.loads(_ok({"key": "value"}))
        assert result == {"key": "value"}

    def test_ok_with_pydantic_model(self) -> None:
        stats = StoreStats(documents=5, chunks=100, db_size_mb=1.5, db_path="/test/db")
        result = json.loads(_ok(stats))
        assert result["documents"] == 5
        assert result["chunks"] == 100

    def test_ok_with_list_of_models(self) -> None:
        docs = [_make_doc_info("/a.md", "A"), _make_doc_info("/b.md", "B")]
        result = json.loads(_ok(docs))
        assert len(result) == 2
        assert result[0]["title"] == "A"
        assert result[1]["title"] == "B"

    def test_error_returns_json_with_error_key(self) -> None:
        result = json.loads(_error("something broke"))
        assert result == {"error": "something broke"}


# ------------------------------------------------------------------ #
# vstash_list                                                          #
# ------------------------------------------------------------------ #


class TestVstashList:
    """Test vstash_list tool."""

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_list_returns_documents(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        docs = [_make_doc_info("/a.md", "A"), _make_doc_info("/b.md", "B")]
        mock_store.return_value.list_documents.return_value = docs

        result = json.loads(vstash_list())
        assert len(result) == 2
        assert result[0]["path"] == "/a.md"
        assert result[1]["path"] == "/b.md"

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_list_empty_store(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        mock_store.return_value.list_documents.return_value = []

        result = json.loads(vstash_list())
        assert result == []

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_list_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_list())
        assert "error" in result
        assert "not found" in result["error"]


# ------------------------------------------------------------------ #
# vstash_stats                                                         #
# ------------------------------------------------------------------ #


class TestVstashStats:
    """Test vstash_stats tool."""

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_stats_returns_valid_json(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        mock_store.return_value.stats.return_value = StoreStats(
            documents=10, chunks=250, db_size_mb=3.14, db_path="/test/memory.db"
        )

        result = json.loads(vstash_stats())
        assert result["documents"] == 10
        assert result["chunks"] == 250
        assert result["db_size_mb"] == 3.14
        assert result["db_path"] == "/test/memory.db"


# ------------------------------------------------------------------ #
# vstash_search                                                        #
# ------------------------------------------------------------------ #


class TestVstashSearch:
    """Test vstash_search tool."""

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_returns_chunks(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        chunks = [_make_search_result("relevant text", "Doc1")]
        mock_store.return_value.search.return_value = chunks
        mock_config.return_value.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = json.loads(vstash_search("test query"))
        assert len(result) == 1
        assert result[0]["text"] == "relevant text"
        assert result[0]["score"] == 0.5

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_empty_results(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        mock_store.return_value.search.return_value = []
        mock_config.return_value.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = json.loads(vstash_search("nothing here"))
        assert result == []

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_search_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_search("test"))
        assert "error" in result


# ------------------------------------------------------------------ #
# vstash_forget                                                        #
# ------------------------------------------------------------------ #


class TestVstashForget:
    """Test vstash_forget tool."""

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_forget_existing_document(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        mock_store.return_value.delete_document.return_value = True

        result = json.loads(vstash_forget("/test/doc.md"))
        assert result["status"] == "deleted"
        assert result["source"] == "/test/doc.md"

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_forget_nonexistent_document(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        mock_store.return_value.delete_document.return_value = False

        result = json.loads(vstash_forget("/test/nope.md"))
        assert result["status"] == "not_found"


# ------------------------------------------------------------------ #
# vstash_ask                                                           #
# ------------------------------------------------------------------ #


class TestVstashAsk:
    """Test vstash_ask tool."""

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_returns_answer_and_sources(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        chunks = [_make_search_result("context about Python", "PythonGuide")]
        mock_store.return_value.search.return_value = chunks
        mock_config.return_value.embeddings.model = "BAAI/bge-small-en-v1.5"

        with patch("vstash.chat.ask", return_value="Python is great."):
            result = json.loads(vstash_ask("What is Python?"))

        assert result["answer"] == "Python is great."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "PythonGuide"

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_no_chunks_returns_message(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        mock_store.return_value.search.return_value = []
        mock_config.return_value.embeddings.model = "BAAI/bge-small-en-v1.5"

        result = json.loads(vstash_ask("unknown topic"))
        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_deduplicates_sources(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        # Two chunks from same doc should produce one source
        chunks = [
            _make_search_result("chunk 1", "SameDoc"),
            _make_search_result("chunk 2", "SameDoc"),
        ]
        mock_store.return_value.search.return_value = chunks
        mock_config.return_value.embeddings.model = "BAAI/bge-small-en-v1.5"

        with patch("vstash.chat.ask", return_value="Answer."):
            result = json.loads(vstash_ask("query"))

        assert len(result["sources"]) == 1  # deduplicated

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_ask_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_ask("test"))
        assert "error" in result
        assert "not found" in result["error"]


# ------------------------------------------------------------------ #
# vstash_add                                                           #
# ------------------------------------------------------------------ #


class TestVstashAdd:
    """Test vstash_add tool."""

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_add_single_file(
        self, mock_config: MagicMock, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        ingest_result = IngestResult(
            status="ok",
            source=str(test_file),
            doc_id="abc123",
            title="Test",
            chunks=2,
            chars=13,
            elapsed_s=0.5,
        )

        with patch("vstash.ingest.ingest", return_value=ingest_result):
            result = json.loads(vstash_add(str(test_file)))

        assert result["status"] == "ok"
        assert result["chunks"] == 2

    def test_add_nonexistent_path_returns_error(self) -> None:
        with patch("vstash.mcp._get_config"), patch("vstash.mcp._get_store"):
            with patch(
                "vstash.ingest.ingest",
                side_effect=FileNotFoundError("not found"),
            ):
                result = json.loads(vstash_add("/nonexistent/file.txt"))

        assert "error" in result
        assert "not found" in result["error"].lower()


# ------------------------------------------------------------------ #
# Error handling                                                       #
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Test that all tools return JSON errors instead of raising."""

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_list_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_list())
        assert "error" in result

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_stats_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_stats())
        assert "error" in result

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_forget_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_forget("/test"))
        assert "error" in result

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_search_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_search("test"))
        assert "error" in result

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_ask_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_ask("test"))
        assert "error" in result

    @patch("vstash.mcp._get_config", side_effect=Exception("unexpected"))
    @patch("vstash.mcp._get_store")
    def test_add_unexpected_error(
        self, mock_store: MagicMock, mock_config: MagicMock
    ) -> None:
        result = json.loads(vstash_add("/test"))
        assert "error" in result
