"""Tests for vstash.langchain — VstashRetriever integration with LangChain."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import requires_sqlite_vec

# Skip the entire module if langchain-core is not installed.
pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from vstash.langchain import VstashRetriever
from vstash.memory import Memory
from vstash.models import SearchResult


# ------------------------------------------------------------------ #
# Unit tests with mocked Memory                                       #
# ------------------------------------------------------------------ #


class TestVstashRetrieverUnit:
    """Test VstashRetriever with a mocked Memory — no DB required."""

    def _make_retriever(self, **kwargs) -> VstashRetriever:
        """Create a retriever with a mocked Memory."""
        mock_memory = MagicMock(spec=Memory)
        mock_memory.search.return_value = [
            SearchResult(
                chunk_id=0,
                text="Python is a programming language.",
                title="Python Guide",
                path="/docs/python.md",
                chunk=0,
                score=0.95,
            ),
            SearchResult(
                chunk_id=1,
                text="Python supports multiple paradigms.",
                title="Python Guide",
                path="/docs/python.md",
                chunk=1,
                score=0.82,
            ),
        ]
        return VstashRetriever(memory=mock_memory, **kwargs)

    def test_invoke_returns_documents(self) -> None:
        """invoke() returns a list of LangChain Document objects."""
        retriever = self._make_retriever()
        docs = retriever.invoke("programming language")

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_document_page_content(self) -> None:
        """Document page_content contains the chunk text."""
        retriever = self._make_retriever()
        docs = retriever.invoke("programming language")

        assert docs[0].page_content == "Python is a programming language."
        assert docs[1].page_content == "Python supports multiple paradigms."

    def test_document_metadata(self) -> None:
        """Document metadata contains source, title, score, and chunk."""
        retriever = self._make_retriever()
        docs = retriever.invoke("programming language")

        meta = docs[0].metadata
        assert meta["source"] == "/docs/python.md"
        assert meta["title"] == "Python Guide"
        assert meta["score"] == 0.95
        assert meta["chunk"] == 0

    def test_top_k_passed_to_search(self) -> None:
        """top_k parameter is forwarded to Memory.search()."""
        retriever = self._make_retriever(top_k=3)
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args
        assert call_kwargs.kwargs["top_k"] == 3

    def test_project_filter_forwarded(self) -> None:
        """project filter is passed to Memory.search() when set."""
        retriever = self._make_retriever(project="alpha")
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args
        assert call_kwargs.kwargs["project"] == "alpha"

    def test_collection_filter_forwarded(self) -> None:
        """collection filter is passed to Memory.search() when set."""
        retriever = self._make_retriever(collection="research")
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args
        assert call_kwargs.kwargs["collection"] == "research"

    def test_layer_filter_forwarded(self) -> None:
        """layer filter is passed to Memory.search() when set."""
        retriever = self._make_retriever(layer="summaries")
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args
        assert call_kwargs.kwargs["layer"] == "summaries"

    def test_no_filters_when_unset(self) -> None:
        """When no filters are set, only top_k is passed."""
        retriever = self._make_retriever()
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args
        assert "project" not in call_kwargs.kwargs
        assert "collection" not in call_kwargs.kwargs
        assert "layer" not in call_kwargs.kwargs

    def test_none_filter_overrides_memory_defaults(self) -> None:
        """Passing filter=None should be forwarded to search to clear scoping."""
        retriever = self._make_retriever(project=None, collection=None, layer=None)
        retriever.invoke("query")

        call_kwargs = retriever.memory.search.call_args.kwargs
        assert "project" in call_kwargs
        assert call_kwargs["project"] is None
        assert "collection" in call_kwargs
        assert call_kwargs["collection"] is None
        assert "layer" in call_kwargs
        assert call_kwargs["layer"] is None

    def test_empty_results(self) -> None:
        """Returns empty list when Memory.search returns no results."""
        retriever = self._make_retriever()
        retriever.memory.search.return_value = []
        docs = retriever.invoke("nonexistent topic")

        assert docs == []


# ------------------------------------------------------------------ #
# Integration tests with real Memory + sqlite-vec                      #
# ------------------------------------------------------------------ #


class TestVstashRetrieverIntegration:
    """Test VstashRetriever with a real Memory instance."""

    @requires_sqlite_vec
    def test_end_to_end_retrieval(self, tmp_path: Path) -> None:
        """Full pipeline: ingest → retrieve via LangChain retriever."""
        doc = tmp_path / "langchain_test.md"
        doc.write_text(
            "# Machine Learning\n\n"
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly "
            "programmed. It focuses on developing algorithms that can access data "
            "and use it to learn for themselves."
        )

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            retriever = VstashRetriever(memory=mem, top_k=3)
            docs = retriever.invoke("artificial intelligence algorithms")

            assert len(docs) > 0
            assert all(isinstance(d, Document) for d in docs)
            assert any("machine learning" in d.page_content.lower() for d in docs)
            # Verify metadata is populated
            assert docs[0].metadata["source"] is not None
            assert docs[0].metadata["score"] > 0

    @requires_sqlite_vec
    def test_retriever_with_project_scope(self, tmp_path: Path) -> None:
        """Retriever respects project scoping from Memory."""
        doc_a = tmp_path / "alpha.md"
        doc_a.write_text("Alpha project covers distributed systems and consensus protocols.")
        doc_b = tmp_path / "beta.md"
        doc_b.write_text("Beta project covers frontend design patterns and user interfaces.")

        db = tmp_path / "test.db"
        with Memory(db=db) as mem:
            mem.add(doc_a, project="alpha")
            mem.add(doc_b, project="beta")

        with Memory(db=db, project="alpha") as mem:
            retriever = VstashRetriever(memory=mem, top_k=5)
            docs = retriever.invoke("distributed systems")

            sources = {d.metadata["source"] for d in docs}
            assert any("alpha" in s for s in sources)
