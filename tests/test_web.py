"""Tests for vstash.web -- Starlette web interface."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from vstash.models import DocumentInfo, IngestResult, SearchResult, StoreStats
from vstash.web import create_app


def _reset_singletons() -> None:
    """Reset module-level singletons between tests.

    Closes any previously cached store before clearing it to avoid
    leaking sqlite connections across tests.
    """
    import vstash.web as web_mod

    if web_mod._store is not None:
        try:
            web_mod._store.close()
        except Exception:
            pass
    web_mod._config = None
    web_mod._store = None
    web_mod._chat_history.clear()


@pytest.fixture(autouse=True)
def _clean_singletons() -> None:
    """Ensure web module singletons and chat history are reset around each test."""
    _reset_singletons()
    yield  # type: ignore[misc]
    _reset_singletons()


@pytest.fixture
def client() -> TestClient:
    """Starlette TestClient with a lifespan-less app for simple route testing."""
    app = create_app()
    with TestClient(app) as c:
        yield c  # type: ignore[misc]


def _make_search_result(text: str = "chunk", title: str = "Doc") -> SearchResult:
    return SearchResult(chunk_id=1, text=text, title=title, path="/test/doc.md", chunk=0, score=0.5)


def _make_doc_info(path: str = "/test/doc.md", title: str = "Doc") -> DocumentInfo:
    return DocumentInfo(
        path=path,
        title=title,
        source_type="markdown",
        chunk_count=3,
        char_count=500,
        added_at="2026-01-01T00:00:00+00:00",
    )


# ------------------------------------------------------------------ #
# /api/embed                                                           #
# ------------------------------------------------------------------ #


class TestApiEmbed:
    """Input-validation and happy-path tests for POST /api/embed."""

    def test_rejects_missing_text_and_texts(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={})
        assert resp.status_code == 400
        assert "provide 'text'" in resp.json()["error"]

    def test_rejects_empty_texts_list(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={"texts": []})
        assert resp.status_code == 400
        assert "non-empty list" in resp.json()["error"]

    def test_rejects_non_string_item_in_texts(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={"texts": ["ok", 123]})
        assert resp.status_code == 400

    def test_rejects_whitespace_only_item(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={"texts": ["ok", "   "]})
        assert resp.status_code == 400

    def test_rejects_batch_over_limit(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={"texts": ["t"] * 300})
        assert resp.status_code == 400
        assert "exceeds limit" in resp.json()["error"]

    def test_rejects_blank_single_text(self, client: TestClient) -> None:
        resp = client.post("/api/embed", json={"text": "   "})
        assert resp.status_code == 400

    def test_batch_embed_success(self, client: TestClient) -> None:
        with patch("vstash.web._do_embed") as mock_embed:
            mock_embed.return_value = {
                "embeddings": [[0.1, 0.2]],
                "model": "fake",
                "dim": 2,
            }
            resp = client.post("/api/embed", json={"texts": ["hello"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["dim"] == 2
        assert body["model"] == "fake"

    def test_single_embed_success(self, client: TestClient) -> None:
        with patch("vstash.web._do_embed_query") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2], "model": "fake"}
            resp = client.post("/api/embed", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["embedding"] == [0.1, 0.2]


# ------------------------------------------------------------------ #
# /api/search                                                          #
# ------------------------------------------------------------------ #


class TestApiSearch:
    def test_rejects_empty_query(self, client: TestClient) -> None:
        resp = client.post("/api/search", json={"query": "  "})
        assert resp.status_code == 400
        assert "query is required" in resp.json()["error"]

    def test_rejects_missing_query(self, client: TestClient) -> None:
        resp = client.post("/api/search", json={})
        assert resp.status_code == 400

    def test_search_success(self, client: TestClient) -> None:
        with patch("vstash.web._do_search") as mock_search:
            mock_search.return_value = {"results": [], "query": "foo"}
            resp = client.post("/api/search", json={"query": "foo", "top_k": 3})
        assert resp.status_code == 200
        mock_search.assert_called_once_with("foo", 3, 0.0)


# ------------------------------------------------------------------ #
# /api/chat and /api/chat/reset                                        #
# ------------------------------------------------------------------ #


class TestApiChat:
    def test_rejects_empty_query(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 400

    def test_chat_no_results_streams_fallback(self, client: TestClient) -> None:
        with patch("vstash.web._do_chat_retrieve") as mock_retrieve:
            mock_retrieve.return_value = ([], [])
            resp = client.post("/api/chat", json={"query": "what?"})
        assert resp.status_code == 200
        body = resp.text
        assert "No relevant documents found" in body
        assert '"done": true' in body

    def test_chat_reset_clears_history(self, client: TestClient) -> None:
        import vstash.web as web_mod

        web_mod._chat_history.extend(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )
        resp = client.post("/api/chat/reset")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert web_mod._chat_history == []


# ------------------------------------------------------------------ #
# /api/documents, /api/stats                                           #
# ------------------------------------------------------------------ #


class TestApiDocuments:
    def test_list_documents(self, client: TestClient) -> None:
        with patch("vstash.web._do_documents") as mock_docs:
            mock_docs.return_value = [_make_doc_info("/a.md", "A").model_dump()]
            resp = client.get("/api/documents")
        assert resp.status_code == 200
        assert resp.json()[0]["title"] == "A"


class TestApiStats:
    def test_stats_returns_snapshot(self, client: TestClient) -> None:
        stats = StoreStats(documents=5, chunks=100, db_size_mb=1.5, db_path="/tmp/db")
        with patch("vstash.web._do_stats") as mock_stats:
            mock_stats.return_value = stats
            resp = client.get("/api/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["documents"] == 5
        assert body["chunks"] == 100


# ------------------------------------------------------------------ #
# /api/upload                                                          #
# ------------------------------------------------------------------ #


class TestApiUpload:
    def test_rejects_missing_file(self, client: TestClient) -> None:
        resp = client.post("/api/upload", data={})
        assert resp.status_code == 400
        assert "No file provided" in resp.json()["error"]

    def test_upload_success(self, client: TestClient) -> None:
        fake_result = IngestResult(
            status="ok",
            source="/tmp/uploaded.txt",
            title="uploaded",
            chunks=1,
        )
        with patch("vstash.web._do_upload") as mock_upload:
            mock_upload.return_value = fake_result
            files = {"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
            resp = client.post("/api/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["title"] == "uploaded"
        assert resp.json()["status"] == "ok"


# ------------------------------------------------------------------ #
# HTML index + observability                                           #
# ------------------------------------------------------------------ #


class TestIndex:
    def test_index_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "<html" in resp.text.lower()


class TestHealthEndpoint:
    def test_health_ok_when_store_reachable(self, client: TestClient) -> None:
        with patch("vstash.web._do_health_check") as mock_check:
            mock_check.return_value = None
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "uptime_seconds" in body

    def test_health_returns_503_on_store_error(self, client: TestClient) -> None:
        with patch("vstash.web._do_health_check") as mock_check:
            mock_check.side_effect = RuntimeError("boom")
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "error"}


class TestMetricsEndpoint:
    def test_metrics_returns_snapshot(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Snapshot always includes uptime_seconds from the registry.
        assert "uptime_seconds" in resp.json()


# ------------------------------------------------------------------ #
# _json() helper                                                       #
# ------------------------------------------------------------------ #


class TestJsonHelper:
    """The _json() helper unwraps Pydantic models and lists of models."""

    def test_serializes_pydantic_model(self) -> None:
        from vstash.web import _json

        stats = StoreStats(documents=1, chunks=2, db_size_mb=0.1, db_path="/tmp/db")
        resp = _json(stats)
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body == stats.model_dump()

    def test_serializes_list_of_models(self) -> None:
        from vstash.web import _json

        docs = [_make_doc_info("/a.md", "A"), _make_doc_info("/b.md", "B")]
        resp = _json(docs)
        body = json.loads(resp.body)
        assert len(body) == 2
        assert body[0]["title"] == "A"

    def test_preserves_status_code(self) -> None:
        from vstash.web import _json

        resp = _json({"error": "bad"}, status=400)
        assert resp.status_code == 400
