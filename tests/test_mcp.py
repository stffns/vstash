"""Tests for vstash.mcp — MCP server tool functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vstash.models import ChunkInfo, DocumentInfo, IngestResult, SearchResult, StoreStats
from vstash.mcp import (
    _error,
    _ok,
    vstash_add,
    vstash_ask,
    vstash_forget,
    vstash_get_chunk,
    vstash_get_document_chunks,
    vstash_journal_log,
    vstash_journal_prune,
    vstash_journal_recall,
    vstash_journal_save,
    vstash_list,
    vstash_search,
    vstash_stats,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _reset_singletons() -> None:
    """Reset module-level singletons between tests.

    Closes any previously cached store before clearing it to avoid
    leaking sqlite connections across tests.
    """
    import vstash.mcp as mcp_mod

    if mcp_mod._store is not None:
        try:
            mcp_mod._store.close()
        except Exception:
            pass
    mcp_mod._config = None
    mcp_mod._store = None


@pytest.fixture(autouse=True)
def _clean_singletons() -> None:
    """Ensure singletons are reset before each test."""
    _reset_singletons()
    yield  # type: ignore[misc]
    _reset_singletons()


def _setup_mock_config(mock_config: MagicMock) -> None:
    """Populate the realistic limit fields the new validation layer needs.

    The MCP search/ask handlers route through ``vstash.services.search``
    (since the Sprint 2 services migration), which calls
    ``validate_search_input`` up front against ``cfg.limits``. With a
    bare MagicMock those comparisons crash with
    ``TypeError: '>' not supported between 'int' and 'MagicMock'``.
    Set the same defaults VstashConfig() ships so every search/ask
    test gets through validation cleanly.
    """
    cfg = mock_config.return_value
    cfg.embeddings.model = "BAAI/bge-small-en-v1.5"
    cfg.limits.max_query_chars = 100_000
    cfg.limits.max_top_k = 10_000
    cfg.limits.max_distance_cutoff = 10.0
    cfg.limits.max_recency_boost = 10.0


def _make_search_result(text: str = "chunk text", title: str = "Doc") -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(chunk_id=0, text=text, title=title, path="/test/doc.md", chunk=0, score=0.5)


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
    def test_list_returns_documents(self, mock_config: MagicMock, mock_store: MagicMock) -> None:
        docs = [_make_doc_info("/a.md", "A"), _make_doc_info("/b.md", "B")]
        mock_store.return_value.list_documents.return_value = docs

        result = json.loads(vstash_list())
        assert len(result) == 2
        assert result[0]["path"] == "/a.md"
        assert result[1]["path"] == "/b.md"

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_list_empty_store(self, mock_config: MagicMock, mock_store: MagicMock) -> None:
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
    def test_stats_returns_valid_json(self, mock_config: MagicMock, mock_store: MagicMock) -> None:
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
        mock_store.return_value.expand_context.return_value = chunks
        mock_store.return_value.last_best_distance = 0.5
        mock_store.return_value.record_search_event.return_value = 1
        _setup_mock_config(mock_config)

        result = json.loads(vstash_search("test query"))
        assert "chunks" in result
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["text"] == "relevant text"
        assert result["chunks"][0]["score"] == 0.5
        assert result["relevance"] in ("high", "medium", "low")

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_empty_results(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        mock_store.return_value.search.return_value = []
        _setup_mock_config(mock_config)

        result = json.loads(vstash_search("nothing here"))
        assert result["chunks"] == []
        assert result["relevance"] == "none"

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_search_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_search("test"))
        assert "error" in result

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_forwards_rrf_weights_to_store(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """vstash_search MCP tool must forward vec_weight/fts_weight/
        retrieval_mode kwargs all the way to VstashStore.search
        (#159, #281).

        Spy on the store.search call and assert the exact kwargs arrive
        with the coerced values.  Mirrors the SDK spy test pattern from
        test_search_forwards_rrf_weights_to_store in test_memory.py so
        the two surfaces stay symmetric.
        """
        chunks = [_make_search_result("hit", "Doc1")]
        mock_store.return_value.search.return_value = chunks
        mock_store.return_value.expand_context.return_value = chunks
        mock_store.return_value.last_best_distance = 0.5
        mock_store.return_value.record_search_event.return_value = 1
        _setup_mock_config(mock_config)

        vstash_search("test query", vec_weight=0.9, fts_weight=0.1)

        _, kwargs = mock_store.return_value.search.call_args
        assert kwargs["vec_weight"] == 0.9
        assert kwargs["fts_weight"] == 0.1
        # #275: MCP forwards the resolved retrieval_mode enum to the
        # store.  Default (no mode set) resolves to "hybrid".
        assert kwargs["retrieval_mode"] == "hybrid"
        assert "fts_only" not in kwargs

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_default_rrf_kwargs_are_none_and_false(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """Regression guard: omitting the new kwargs must forward None /
        False to the store, NOT silently pinned defaults."""
        chunks = [_make_search_result("hit", "Doc1")]
        mock_store.return_value.search.return_value = chunks
        mock_store.return_value.expand_context.return_value = chunks
        mock_store.return_value.last_best_distance = 0.5
        mock_store.return_value.record_search_event.return_value = 1
        _setup_mock_config(mock_config)

        vstash_search("test query")

        _, kwargs = mock_store.return_value.search.call_args
        assert kwargs["vec_weight"] is None
        assert kwargs["fts_weight"] is None
        assert kwargs["retrieval_mode"] == "hybrid"
        assert "fts_only" not in kwargs

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_coerces_string_kwargs_from_mcp_client(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """MCP clients can send strings where floats are expected.
        vstash_search must coerce them defensively instead of 422'ing.

        This is the defensive pattern documented in
        ``_coerce_optional_float`` / ``_coerce_retrieval_mode`` and
        mirrors the existing ``top_k = int(top_k)`` /
        ``recency_boost = float(recency_boost)`` coercions elsewhere
        in this module.
        """
        chunks = [_make_search_result("hit", "Doc1")]
        mock_store.return_value.search.return_value = chunks
        mock_store.return_value.expand_context.return_value = chunks
        mock_store.return_value.last_best_distance = 0.5
        mock_store.return_value.record_search_event.return_value = 1
        _setup_mock_config(mock_config)

        # Weights as strings, fts_only omitted (default False -> hybrid).
        vstash_search(
            "test",
            vec_weight="0.8",  # type: ignore[arg-type]
            fts_weight="0.2",  # type: ignore[arg-type]
        )
        _, kwargs = mock_store.return_value.search.call_args
        assert kwargs["vec_weight"] == 0.8
        assert kwargs["fts_weight"] == 0.2
        assert kwargs["retrieval_mode"] == "hybrid"

        # retrieval_mode as a string coerces via _coerce_retrieval_mode.
        mock_store.return_value.search.reset_mock()
        vstash_search("test", retrieval_mode="fts_only")
        _, kwargs = mock_store.return_value.search.call_args
        assert kwargs["retrieval_mode"] == "fts_only"

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_rejects_unparseable_string_kwargs(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """Unparseable strings must surface as a structured MCP error,
        not a 500 — the tool wraps the ValueError into _error()."""
        _setup_mock_config(mock_config)

        result = json.loads(
            vstash_search("test", vec_weight="not_a_number")  # type: ignore[arg-type]
        )
        assert "error" in result
        assert "vec_weight" in result["error"]

        # Unknown retrieval_mode string surfaces as structured error.
        result2 = json.loads(
            vstash_search("test", retrieval_mode="semantic")  # type: ignore[arg-type]
        )
        assert "error" in result2
        assert "retrieval_mode" in result2["error"]

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_search_rejects_nan_and_inf_weights(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """Review on #168: ``validate_search_input`` uses ``< 0.0`` /
        ``> 1.0`` bounds which do not catch NaN or ±Inf. The MCP
        coercion layer must reject non-finite floats before they reach
        the validator — a NaN weight would otherwise propagate into
        RRF scoring and produce NaN result scores.
        """
        _setup_mock_config(mock_config)

        for bad_value in ("nan", "NaN", "inf", "-inf", "Infinity"):
            result = json.loads(
                vstash_search("test", vec_weight=bad_value)  # type: ignore[arg-type]
            )
            assert "error" in result, f"expected error for vec_weight={bad_value!r}"
            assert "vec_weight" in result["error"]
            assert "non-finite" in result["error"], (
                f"error message should name non-finite: {result['error']}"
            )

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_retrieval_mode_fts_only_ignores_invalid_weights(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """Precedence rule (documented in docs/mcp-server.md): when
        ``retrieval_mode="fts_only"``, ``vec_weight`` and ``fts_weight``
        are dropped before coercion and validation.  A caller who sends
        an invalid or out-of-range weight together with fts_only mode
        should still get a successful FTS-only query.

        Flagged in the #168 review -- without this fix, the weight
        coercion would run first and raise ``ValueError`` before the
        mode was even looked at.
        """
        chunks = [_make_search_result("hit", "Doc1")]
        mock_store.return_value.search.return_value = chunks
        mock_store.return_value.expand_context.return_value = chunks
        mock_store.return_value.last_best_distance = 0.5
        mock_store.return_value.record_search_event.return_value = 1
        _setup_mock_config(mock_config)

        # Every one of these combinations would fail coercion on its
        # own but must succeed when paired with retrieval_mode="fts_only".
        for bad_vec, bad_fts in [
            ("not_a_number", None),
            ("nan", None),
            (None, "-inf"),
            ("10.0", "5.0"),  # out of [0, 1] -- but never reaches validator
        ]:
            mock_store.return_value.search.reset_mock()
            result = json.loads(
                vstash_search(
                    "test",
                    retrieval_mode="fts_only",
                    vec_weight=bad_vec,  # type: ignore[arg-type]
                    fts_weight=bad_fts,  # type: ignore[arg-type]
                )
            )
            assert "error" not in result, (
                f"retrieval_mode='fts_only' should have dropped invalid weights "
                f"vec_weight={bad_vec!r}, fts_weight={bad_fts!r}, "
                f"but got {result}"
            )
            _, kwargs = mock_store.return_value.search.call_args
            assert kwargs["retrieval_mode"] == "fts_only"
            assert kwargs["vec_weight"] is None
            assert kwargs["fts_weight"] is None


# ------------------------------------------------------------------ #
# vstash_forget                                                        #
# ------------------------------------------------------------------ #


class TestVstashForget:
    """Test vstash_forget tool."""

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_forget_existing_document(self, mock_config: MagicMock, mock_store: MagicMock) -> None:
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
        mock_store.return_value.find_document.return_value = None

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
        store_inst = mock_store.return_value
        store_inst.search.return_value = chunks
        store_inst.last_best_distance = 0.5
        store_inst.expand_context.return_value = chunks
        _setup_mock_config(mock_config)

        with patch("vstash.chat.ask", return_value="Python is great."):
            result = json.loads(vstash_ask("What is Python?"))

        assert result["answer"] == "Python is great."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "PythonGuide"
        store_inst.record_search_event.assert_called_once()
        store_inst.expand_context.assert_called_once()

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_no_chunks_returns_message(
        self, mock_config: MagicMock, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        mock_store.return_value.search.return_value = []
        _setup_mock_config(mock_config)

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
        store_inst = mock_store.return_value
        store_inst.search.return_value = chunks
        store_inst.last_best_distance = 0.5
        store_inst.expand_context.return_value = chunks
        _setup_mock_config(mock_config)

        with patch("vstash.chat.ask", return_value="Answer."):
            result = json.loads(vstash_ask("query"))

        assert len(result["sources"]) == 1  # deduplicated

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_ask_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_ask("test"))
        assert "error" in result
        assert "not found" in result["error"]

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_forwards_rrf_weights_and_mode(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """vstash_ask must forward vec_weight / fts_weight /
        retrieval_mode to the retrieval step (#159, #281).  Symmetric
        with the SDK ``Memory.ask()`` forwarding test.
        """
        chunks = [_make_search_result("context", "Doc1")]
        store_inst = mock_store.return_value
        store_inst.search.return_value = chunks
        store_inst.last_best_distance = 0.5
        store_inst.expand_context.return_value = chunks
        _setup_mock_config(mock_config)

        with patch("vstash.chat.ask", return_value="answer"):
            vstash_ask(
                "query",
                vec_weight=0.3,
                fts_weight=0.7,
            )

        _, kwargs = store_inst.search.call_args
        assert kwargs["vec_weight"] == 0.3
        assert kwargs["fts_weight"] == 0.7
        assert kwargs["retrieval_mode"] == "hybrid"
        assert "fts_only" not in kwargs

    @patch("vstash.mcp.embed_query", return_value=[0.1] * 384)
    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_ask_coerces_string_kwargs(
        self,
        mock_config: MagicMock,
        mock_store: MagicMock,
        mock_embed: MagicMock,
    ) -> None:
        """vstash_ask must defensively coerce string kwargs from MCP
        clients the same way vstash_search does (#159).
        """
        chunks = [_make_search_result("context", "Doc1")]
        store_inst = mock_store.return_value
        store_inst.search.return_value = chunks
        store_inst.last_best_distance = 0.5
        store_inst.expand_context.return_value = chunks
        _setup_mock_config(mock_config)

        # Weights as strings, no mode set (default hybrid).
        with patch("vstash.chat.ask", return_value="answer"):
            vstash_ask(
                "query",
                vec_weight="0.1",  # type: ignore[arg-type]
                fts_weight="0.9",  # type: ignore[arg-type]
            )
        _, kwargs = store_inst.search.call_args
        assert kwargs["vec_weight"] == 0.1
        assert kwargs["fts_weight"] == 0.9
        assert kwargs["retrieval_mode"] == "hybrid"

        # retrieval_mode as a string coerces via _coerce_retrieval_mode.
        store_inst.search.reset_mock()
        with patch("vstash.chat.ask", return_value="answer"):
            vstash_ask("query", retrieval_mode="fts_only")
        _, kwargs = store_inst.search.call_args
        assert kwargs["retrieval_mode"] == "fts_only"


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
        with (
            patch("vstash.mcp._get_config"),
            patch("vstash.mcp._get_store"),
            patch(
                "vstash.ingest.ingest",
                side_effect=FileNotFoundError("not found"),
            ),
        ):
            result = json.loads(vstash_add("/nonexistent/file.txt"))

        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_add_url_bypasses_path_resolve(
        self, mock_config: MagicMock, mock_store: MagicMock
    ) -> None:
        """URLs should go directly to ingest without Path().resolve()."""
        url = "https://example.com/article.html"
        ingest_result = IngestResult(
            status="ok",
            source=url,
            doc_id="url123",
            title="Article",
            chunks=5,
            chars=2000,
            elapsed_s=1.2,
        )

        with patch("vstash.ingest.ingest", return_value=ingest_result) as mock_ingest:
            result = json.loads(vstash_add(url))

        # Verify ingest received the raw URL, not a resolved path
        mock_ingest.assert_called_once()
        called_source = mock_ingest.call_args[0][0]
        assert called_source == url
        assert result["status"] == "ok"

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_add_tilde_path_gets_expanded(
        self, mock_config: MagicMock, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        """Tilde paths should be expanded to absolute paths before ingestion."""
        # Create a real file under a fake home to test expansion logic
        test_file = tmp_path / "notes.md"
        test_file.write_text("Some notes")

        ingest_result = IngestResult(
            status="ok",
            source=str(test_file),
            doc_id="tilde123",
            title="Notes",
            chunks=1,
            chars=10,
            elapsed_s=0.1,
        )

        with patch("vstash.ingest.ingest", return_value=ingest_result) as mock_ingest:
            json.loads(vstash_add(str(test_file)))

        # Verify ingest received an absolute path (no ~ remaining)
        called_source = mock_ingest.call_args[0][0]
        assert "~" not in called_source
        assert Path(called_source).is_absolute()

    @patch("vstash.mcp._get_store")
    @patch("vstash.mcp._get_config")
    def test_add_with_force_reingests(
        self, mock_config: MagicMock, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        """force=True should pass through to the ingest function."""
        test_file = tmp_path / "notes.md"
        test_file.write_text("Some notes")

        ingest_result = IngestResult(
            status="ok",
            source=str(test_file),
            doc_id="f123",
            title="Notes",
            chunks=1,
            chars=10,
            elapsed_s=0.1,
        )

        with patch("vstash.ingest.ingest", return_value=ingest_result) as mock_ingest:
            result = json.loads(vstash_add(str(test_file), force=True))

        assert result["status"] == "ok"
        # Verify force=True was passed to ingest
        assert mock_ingest.call_args.kwargs["force"] is True


# ------------------------------------------------------------------ #
# vstash_get_document_chunks                                           #
# ------------------------------------------------------------------ #


class TestVstashGetDocumentChunks:
    """Test vstash_get_document_chunks MCP tool."""

    @patch("vstash.mcp._get_store")
    def test_get_document_chunks_success(self, mock_store: MagicMock) -> None:
        mock_store.return_value.get_document_chunks.return_value = ["chunk 1", "chunk 2"]

        result = json.loads(vstash_get_document_chunks("/test/doc.md"))
        assert result["chunk_count"] == 2
        assert result["chunks"] == ["chunk 1", "chunk 2"]
        assert result["path"] == "/test/doc.md"

    @patch("vstash.mcp._get_store")
    def test_get_document_chunks_not_found(self, mock_store: MagicMock) -> None:
        mock_store.return_value.get_document_chunks.return_value = []

        result = json.loads(vstash_get_document_chunks("/nonexistent.md"))
        assert "error" in result

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_get_document_chunks_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_get_document_chunks("/test/doc.md"))
        assert "error" in result
        assert "not found" in result["error"]

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_get_document_chunks_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_get_document_chunks("/test/doc.md"))
        assert "error" in result

    @patch("vstash.mcp._get_store")
    def test_get_document_chunks_with_collection(self, mock_store: MagicMock) -> None:
        mock_store.return_value.get_document_chunks.return_value = ["text"]

        result = json.loads(vstash_get_document_chunks("/test/doc.md", collection="notes"))
        assert result["chunk_count"] == 1
        mock_store.return_value.get_document_chunks.assert_called_once_with(
            "/test/doc.md", collection="notes"
        )

    @patch("vstash.mcp._get_store")
    def test_get_document_chunks_url_no_normalization(self, mock_store: MagicMock) -> None:
        """URLs should not be path-normalized."""
        mock_store.return_value.get_document_chunks.return_value = ["content"]

        vstash_get_document_chunks("https://example.com/doc")
        mock_store.return_value.get_document_chunks.assert_called_once_with(
            "https://example.com/doc", collection=None
        )

    @patch("vstash.mcp._get_store")
    def test_get_document_chunks_text_path_no_normalization(self, mock_store: MagicMock) -> None:
        """text:// paths should not be path-normalized."""
        mock_store.return_value.get_document_chunks.return_value = ["note content"]

        result = json.loads(vstash_get_document_chunks("text://my note"))
        assert result["path"] == "text://my note"
        mock_store.return_value.get_document_chunks.assert_called_once_with(
            "text://my note", collection=None
        )


# ------------------------------------------------------------------ #
# vstash_get_chunk                                                     #
# ------------------------------------------------------------------ #


class TestVstashGetChunk:
    """Test vstash_get_chunk MCP tool."""

    @patch("vstash.mcp._get_store")
    def test_get_chunk_success(self, mock_store: MagicMock) -> None:
        chunk = ChunkInfo(
            chunk_id=42,
            doc_id="abc123",
            chunk=0,
            text="some text",
            title="Doc",
            path="/test/doc.md",
            collection="default",
        )
        mock_store.return_value.get_chunk.return_value = chunk

        result = json.loads(vstash_get_chunk(42))
        assert result["chunk_id"] == 42
        assert result["text"] == "some text"
        assert result["title"] == "Doc"

    @patch("vstash.mcp._get_store")
    def test_get_chunk_not_found(self, mock_store: MagicMock) -> None:
        mock_store.return_value.get_chunk.return_value = None

        result = json.loads(vstash_get_chunk(9999))
        assert "error" in result
        assert "9999" in result["error"]

    @patch("vstash.mcp._get_store", side_effect=FileNotFoundError("no db"))
    def test_get_chunk_db_not_found(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_get_chunk(1))
        assert "error" in result
        assert "not found" in result["error"]

    @patch("vstash.mcp._get_store", side_effect=Exception("unexpected"))
    def test_get_chunk_unexpected_error(self, mock_store: MagicMock) -> None:
        result = json.loads(vstash_get_chunk(1))
        assert "error" in result

    def test_get_chunk_invalid_type(self) -> None:
        result = json.loads(vstash_get_chunk("not_a_number"))  # type: ignore[arg-type]
        assert "error" in result
        assert "integer" in result["error"]


class TestVstashForgetFuzzy:
    """Test vstash_forget fuzzy matching."""

    @patch("vstash.mcp._get_store")
    def test_forget_fuzzy_match(self, mock_store: MagicMock) -> None:
        """If exact match fails, fuzzy match by partial path should work."""
        store_instance = mock_store.return_value
        store_instance.delete_document.side_effect = [False, True]
        store_instance.find_document.return_value = "/full/path/to/notes.md"

        result = json.loads(vstash_forget("notes.md"))

        assert result["status"] == "deleted"
        assert result["source"] == "/full/path/to/notes.md"
        assert result["matched_from"] == "notes.md"

    @patch("vstash.mcp._get_store")
    def test_forget_fuzzy_no_match(self, mock_store: MagicMock) -> None:
        """If neither exact nor fuzzy match, return not_found."""
        store_instance = mock_store.return_value
        store_instance.delete_document.return_value = False
        store_instance.find_document.return_value = None

        result = json.loads(vstash_forget("nonexistent.pdf"))

        assert result["status"] == "not_found"


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
    def test_add_unexpected_error(self, mock_store: MagicMock, mock_config: MagicMock) -> None:
        result = json.loads(vstash_add("/test"))
        assert "error" in result


# ------------------------------------------------------------------ #
# vstash_journal_save                                                  #
# ------------------------------------------------------------------ #


class TestVstashJournalSave:
    """Test vstash_journal_save tool."""

    @patch("vstash.journal.journal_save")
    def test_save_returns_metadata(self, mock_save: MagicMock) -> None:
        mock_save.return_value = {
            "title": "2026-04-02 — Test Entry",
            "chunks": 1,
            "tags": "journal",
            "added_at": "2026-04-02T12:00:00",
        }

        result = json.loads(vstash_journal_save("some text", title="Test Entry"))
        assert result["title"] == "2026-04-02 — Test Entry"
        assert result["chunks"] == 1
        mock_save.assert_called_once_with(
            "some text", title="Test Entry", project=None, tags=None, source="mcp"
        )

    @patch("vstash.journal.journal_save", side_effect=Exception("write failed"))
    def test_save_error(self, mock_save: MagicMock) -> None:
        result = json.loads(vstash_journal_save("text"))
        assert "error" in result
        assert "failed" in result["error"]


# ------------------------------------------------------------------ #
# vstash_journal_recall                                                #
# ------------------------------------------------------------------ #


class TestVstashJournalRecall:
    """Test vstash_journal_recall tool."""

    @patch("vstash.journal.journal_recall")
    def test_recall_with_query(self, mock_recall: MagicMock) -> None:
        mock_recall.return_value = [
            {"title": "Entry 1", "text": "some context", "score": 0.8},
        ]

        result = json.loads(vstash_journal_recall("past decisions"))
        assert len(result) == 1
        assert result[0]["title"] == "Entry 1"
        mock_recall.assert_called_once_with(
            query="past decisions",
            top_k=5,
            project=None,
            tags=None,
            added_after=None,
            added_before=None,
        )

    @patch("vstash.journal.journal_recall")
    def test_recall_no_query_returns_recent(self, mock_recall: MagicMock) -> None:
        mock_recall.return_value = []

        result = json.loads(vstash_journal_recall())
        assert result == []
        mock_recall.assert_called_once_with(
            query=None,
            top_k=5,
            project=None,
            tags=None,
            added_after=None,
            added_before=None,
        )

    @patch("vstash.journal.journal_recall", side_effect=Exception("search failed"))
    def test_recall_error(self, mock_recall: MagicMock) -> None:
        result = json.loads(vstash_journal_recall("query"))
        assert "error" in result


# ------------------------------------------------------------------ #
# vstash_journal_log                                                   #
# ------------------------------------------------------------------ #


class TestVstashJournalLog:
    """Test vstash_journal_log tool."""

    @patch("vstash.journal.journal_log")
    def test_log_returns_entries(self, mock_log: MagicMock) -> None:
        mock_log.return_value = [
            {"title": "Entry A", "added_at": "2026-04-02"},
            {"title": "Entry B", "added_at": "2026-04-01"},
        ]

        result = json.loads(vstash_journal_log(limit=10))
        assert len(result) == 2
        mock_log.assert_called_once_with(limit=10, recent=None, project=None)

    @patch("vstash.journal.journal_log", side_effect=Exception("read failed"))
    def test_log_error(self, mock_log: MagicMock) -> None:
        result = json.loads(vstash_journal_log())
        assert "error" in result


# ------------------------------------------------------------------ #
# vstash_journal_prune                                                 #
# ------------------------------------------------------------------ #


class TestVstashJournalPrune:
    """Test vstash_journal_prune tool."""

    @patch("vstash.journal.journal_prune")
    def test_prune_returns_count(self, mock_prune: MagicMock) -> None:
        mock_prune.return_value = {"deleted": 3, "titles": ["A", "B", "C"]}

        result = json.loads(vstash_journal_prune("30d"))
        assert result["deleted"] == 3
        mock_prune.assert_called_once_with("30d", project=None, dry_run=False)

    @patch("vstash.journal.journal_prune")
    def test_prune_dry_run(self, mock_prune: MagicMock) -> None:
        mock_prune.return_value = {"deleted": 0, "would_delete": 5}

        result = json.loads(vstash_journal_prune("7d", dry_run=True))
        assert result["would_delete"] == 5
        mock_prune.assert_called_once_with("7d", project=None, dry_run=True)

    @patch("vstash.journal.journal_prune", side_effect=ValueError("invalid age format"))
    def test_prune_invalid_age(self, mock_prune: MagicMock) -> None:
        result = json.loads(vstash_journal_prune("xyz"))
        assert "error" in result
        assert "invalid" in result["error"]

    @patch("vstash.journal.journal_prune", side_effect=Exception("unexpected"))
    def test_prune_unexpected_error(self, mock_prune: MagicMock) -> None:
        result = json.loads(vstash_journal_prune("30d"))
        assert "error" in result
