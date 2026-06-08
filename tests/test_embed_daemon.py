"""Tests for the embedder daemon (vstash serve /api/embed endpoint)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import vstash.embed as embed_mod


class _FakeEmbedHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that mimics /api/embed and /health."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/embed":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            if "text" in body:
                resp = {"embedding": [0.1] * 384, "model": "BAAI/bge-small-en-v1.5"}
            elif "texts" in body:
                n = len(body["texts"])
                resp = {
                    "embeddings": [[0.1] * 384] * n,
                    "model": "BAAI/bge-small-en-v1.5",
                    "dim": 384,
                }
            else:
                self.send_response(400)
                self.end_headers()
                return

            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_daemon():
    """Start a fake embed daemon on a random port, yield the URL, stop on exit."""
    server = HTTPServer(("127.0.0.1", 0), _FakeEmbedHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture(autouse=True)
def _reset_daemon_state():
    """Reset daemon global state before and after each test."""
    _saved = (
        embed_mod._daemon_checked,
        embed_mod._daemon_available,
        embed_mod._daemon_url,
        embed_mod._is_daemon,
        embed_mod._DAEMON_DEFAULT_URL,
    )
    _saved_mismatch = set(embed_mod._daemon_model_mismatch)
    embed_mod._daemon_checked = False
    embed_mod._daemon_available = False
    embed_mod._daemon_url = None
    embed_mod._is_daemon = False
    embed_mod._daemon_model_mismatch.clear()
    yield
    # Restore the original cache contents (preserving object identity) so this
    # fixture is symmetric with the other globals it save/restores.
    embed_mod._daemon_model_mismatch.clear()
    embed_mod._daemon_model_mismatch.update(_saved_mismatch)
    (
        embed_mod._daemon_checked,
        embed_mod._daemon_available,
        embed_mod._daemon_url,
        embed_mod._is_daemon,
        embed_mod._DAEMON_DEFAULT_URL,
    ) = _saved


class TestDaemonClient:
    def test_daemon_embed_query(self, fake_daemon: str):
        result = embed_mod._daemon_embed_query("hello", fake_daemon, "BAAI/bge-small-en-v1.5")
        assert result is not None
        assert len(result) == 384

    def test_daemon_embed_texts(self, fake_daemon: str):
        result = embed_mod._daemon_embed_texts(
            ["hello", "world"], fake_daemon, "BAAI/bge-small-en-v1.5"
        )
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 384

    def test_daemon_embed_query_bad_url(self):
        result = embed_mod._daemon_embed_query(
            "hello", "http://127.0.0.1:1", "BAAI/bge-small-en-v1.5"
        )
        assert result is None

    def test_daemon_embed_texts_bad_url(self):
        result = embed_mod._daemon_embed_texts(
            ["hello"], "http://127.0.0.1:1", "BAAI/bge-small-en-v1.5"
        )
        assert result is None

    def test_model_mismatch_returns_none(self, fake_daemon: str):
        result = embed_mod._daemon_embed_query("hello", fake_daemon, "wrong-model")
        assert result is None

    def test_model_mismatch_batch_returns_none(self, fake_daemon: str):
        result = embed_mod._daemon_embed_texts(["hello"], fake_daemon, "wrong-model")
        assert result is None

    def test_model_mismatch_is_cached(self, fake_daemon: str):
        """A mismatch records the (url, model) pair so future calls can skip the
        daemon instead of re-paying the round-trip."""
        embed_mod._daemon_embed_texts(["hello"], fake_daemon, "wrong-model")
        assert (fake_daemon, "wrong-model") in embed_mod._daemon_model_mismatch

    def test_embed_texts_skips_daemon_for_cached_mismatch(self, fake_daemon: str, monkeypatch):
        """Once a model is known to mismatch, embed_texts must go straight to the
        local backend without calling the daemon again."""
        embed_mod._daemon_url = fake_daemon
        embed_mod._daemon_available = True
        embed_mod._daemon_checked = True
        embed_mod._daemon_model_mismatch.add((fake_daemon, "cached-mismatch-model"))

        called = {"daemon": 0}

        def _should_not_run(*_a, **_k):
            called["daemon"] += 1
            return None

        monkeypatch.setattr(embed_mod, "_daemon_embed_texts", _should_not_run)

        class _FakeBackend:
            def embed_batch(self, texts, model_name, backend):
                return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

        monkeypatch.setattr(embed_mod, "_resolve_builtin_backend", lambda _m: _FakeBackend())

        result = embed_mod.embed_texts(["x", "y"], "cached-mismatch-model")
        assert called["daemon"] == 0  # daemon skipped via the mismatch cache
        assert len(result) == 2

    def test_check_daemon_finds_server(self, fake_daemon: str):
        embed_mod._DAEMON_DEFAULT_URL = fake_daemon
        url = embed_mod._check_daemon()
        assert url == fake_daemon
        assert embed_mod._daemon_available is True

    def test_check_daemon_caches_miss(self):
        embed_mod._DAEMON_DEFAULT_URL = "http://127.0.0.1:1"
        url = embed_mod._check_daemon()
        assert url is None
        assert embed_mod._daemon_checked is True
        # Second call uses cache (no network)
        url2 = embed_mod._check_daemon()
        assert url2 is None

    def test_check_daemon_skipped_when_is_daemon(self, fake_daemon: str):
        embed_mod._is_daemon = True
        embed_mod._DAEMON_DEFAULT_URL = fake_daemon
        url = embed_mod._check_daemon()
        assert url is None

    def test_embed_query_uses_daemon(self, fake_daemon: str):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = fake_daemon
        result = embed_mod.embed_query("hello", "BAAI/bge-small-en-v1.5")
        assert len(result) == 384

    def test_embed_texts_uses_daemon(self, fake_daemon: str):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = fake_daemon
        result = embed_mod.embed_texts(["hello", "world"], "BAAI/bge-small-en-v1.5")
        assert len(result) == 2

    def test_fallback_to_local_on_daemon_failure(self):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = "http://127.0.0.1:1"
        result = embed_mod.embed_query("hello", "BAAI/bge-small-en-v1.5")
        assert len(result) == 384
