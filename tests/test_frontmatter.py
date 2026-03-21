"""Tests for frontmatter extraction and filtered retrieval."""

from __future__ import annotations

import pytest

from vstash.ingest import _extract_frontmatter, _strip_frontmatter


class TestExtractFrontmatter:
    """Tests for _extract_frontmatter()."""

    def test_no_frontmatter(self) -> None:
        text = "Hello world\nThis is a document."
        assert _extract_frontmatter(text) == {}

    def test_simple_frontmatter(self) -> None:
        text = "---\nproject: vstash\nlayer: core\n---\nContent here"
        fm = _extract_frontmatter(text)
        assert fm["project"] == "vstash"
        assert fm["layer"] == "core"

    def test_frontmatter_with_tags_list(self) -> None:
        text = "---\ntags: [python, rag, search]\n---\nContent"
        fm = _extract_frontmatter(text)
        assert fm["tags"] == ["python", "rag", "search"]

    def test_frontmatter_with_date_field(self) -> None:
        text = "---\nproject: myproject\ndate: 2026-03-21\n---\nContent"
        fm = _extract_frontmatter(text)
        assert fm["project"] == "myproject"
        assert "date" in fm

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\nContent"
        fm = _extract_frontmatter(text)
        assert fm == {}

    def test_frontmatter_not_at_start(self) -> None:
        text = "Some text\n---\nproject: test\n---\nContent"
        fm = _extract_frontmatter(text)
        assert fm == {}

    def test_frontmatter_with_whitespace_before(self) -> None:
        text = "  \n---\nproject: vstash\n---\nContent"
        fm = _extract_frontmatter(text)
        assert fm.get("project") == "vstash"


class TestStripFrontmatter:
    """Tests for _strip_frontmatter()."""

    def test_strip_removes_block(self) -> None:
        text = "---\nproject: vstash\n---\nContent here"
        result = _strip_frontmatter(text)
        assert "---" not in result
        assert "project" not in result
        assert "Content here" in result

    def test_strip_no_frontmatter(self) -> None:
        text = "Just plain text"
        assert _strip_frontmatter(text) == text

    def test_strip_preserves_later_separators(self) -> None:
        text = "---\nproject: test\n---\nContent\n---\nMore content"
        result = _strip_frontmatter(text)
        assert "Content" in result
        assert "More content" in result


class TestStoreMetadataFilter:
    """Tests for store-level project/layer filtering."""

    @pytest.fixture()
    def store_with_metadata(self, tmp_path: object) -> object:
        """Create a store with documents in different projects/layers."""
        from vstash.store import VstashStore

        db_path = str(tmp_path) + "/test.db"  # type: ignore[operator]
        store = VstashStore(db_path, embedding_dim=4)
        dim = 4

        # Doc in project=alpha, layer=api
        store.add_document(
            path="/docs/alpha_api.md",
            title="Alpha API",
            chunks=["Alpha API docs chunk"],
            embeddings=[[0.1] * dim],
            project="alpha",
            layer="api",
            tags="python,rest",
        )
        # Doc in project=alpha, layer=core
        store.add_document(
            path="/docs/alpha_core.md",
            title="Alpha Core",
            chunks=["Alpha core module chunk"],
            embeddings=[[0.2] * dim],
            project="alpha",
            layer="core",
            tags="python",
        )
        # Doc in project=beta, layer=api
        store.add_document(
            path="/docs/beta_api.md",
            title="Beta API",
            chunks=["Beta API docs chunk"],
            embeddings=[[0.3] * dim],
            project="beta",
            layer="api",
            tags="go,grpc",
        )
        # Doc with no metadata
        store.add_document(
            path="/docs/readme.md",
            title="Readme",
            chunks=["General readme chunk"],
            embeddings=[[0.4] * dim],
        )
        return store

    def test_list_all(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents()
        assert len(docs) == 4

    def test_list_by_project(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents(project="alpha")
        assert len(docs) == 2
        assert all(d.project == "alpha" for d in docs)

    def test_list_by_layer(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents(layer="api")
        assert len(docs) == 2
        assert all(d.layer == "api" for d in docs)

    def test_list_by_project_and_layer(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents(project="alpha", layer="api")
        assert len(docs) == 1
        assert docs[0].title == "Alpha API"

    def test_find_by_project(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        path = store.find_document("API", project="beta")
        assert path is not None
        assert "beta" in path

    def test_search_by_project(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        results = store.search(
            query_embedding=[0.1] * 4,
            query_text="API docs",
            top_k=5,
            project="alpha",
        )
        # Should only return alpha project docs
        for r in results:
            assert "alpha" in r.path.lower() or "Alpha" in r.title

    def test_metadata_in_document_info(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents(project="alpha", layer="api")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.project == "alpha"
        assert doc.layer == "api"
        assert doc.tags == "python,rest"

    def test_no_metadata_returns_none(self, store_with_metadata: object) -> None:
        store = store_with_metadata  # type: ignore[assignment]
        docs = store.list_documents()
        readme = [d for d in docs if d.title == "Readme"][0]
        assert readme.project is None
        assert readme.layer is None
        assert readme.tags is None
