"""Tests for the embedder daemon (vstash serve /api/embed endpoint)."""

from __future__ import annotations

import json
import threading
import time
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
                resp = {"embedding": [0.1] * 384, "model": "test"}
            elif "texts" in body:
                n = len(body["texts"])
                resp = {"embeddings": [[0.1] * 384] * n, "model": "test", "dim": 384}
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
        pass  # suppress output


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


class TestDaemonClient:
    def test_daemon_embed_query(self, fake_daemon: str):
        result = embed_mod._daemon_embed_query("hello", fake_daemon)
        assert result is not None
        assert len(result) == 384

    def test_daemon_embed_texts(self, fake_daemon: str):
        result = embed_mod._daemon_embed_texts(["hello", "world"], fake_daemon)
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 384

    def test_daemon_embed_query_bad_url(self):
        result = embed_mod._daemon_embed_query("hello", "http://127.0.0.1:1")
        assert result is None

    def test_daemon_embed_texts_bad_url(self):
        result = embed_mod._daemon_embed_texts(["hello"], "http://127.0.0.1:1")
        assert result is None

    def test_check_daemon_finds_server(self, fake_daemon: str):
        # Reset global state
        embed_mod._daemon_checked = False
        embed_mod._daemon_available = False
        embed_mod._daemon_url = None
        embed_mod._DAEMON_DEFAULT_URL = fake_daemon
        try:
            url = embed_mod._check_daemon()
            assert url == fake_daemon
            assert embed_mod._daemon_available is True
        finally:
            # Restore defaults
            embed_mod._DAEMON_DEFAULT_URL = "http://127.0.0.1:8585"
            embed_mod._daemon_checked = False
            embed_mod._daemon_available = False
            embed_mod._daemon_url = None

    def test_check_daemon_caches_miss(self):
        embed_mod._daemon_checked = False
        embed_mod._daemon_available = False
        embed_mod._daemon_url = None
        old_url = embed_mod._DAEMON_DEFAULT_URL
        embed_mod._DAEMON_DEFAULT_URL = "http://127.0.0.1:1"
        try:
            url = embed_mod._check_daemon()
            assert url is None
            assert embed_mod._daemon_checked is True
            # Second call should be instant (cached)
            t0 = time.perf_counter()
            url2 = embed_mod._check_daemon()
            elapsed = time.perf_counter() - t0
            assert url2 is None
            assert elapsed < 0.01
        finally:
            embed_mod._DAEMON_DEFAULT_URL = old_url
            embed_mod._daemon_checked = False
            embed_mod._daemon_available = False
            embed_mod._daemon_url = None

    def test_embed_query_uses_daemon_when_available(self, fake_daemon: str):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = fake_daemon
        try:
            result = embed_mod.embed_query("hello", "BAAI/bge-small-en-v1.5")
            assert len(result) == 384
        finally:
            embed_mod._daemon_checked = False
            embed_mod._daemon_available = False
            embed_mod._daemon_url = None

    def test_embed_texts_uses_daemon_when_available(self, fake_daemon: str):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = fake_daemon
        try:
            result = embed_mod.embed_texts(["hello", "world"], "BAAI/bge-small-en-v1.5")
            assert len(result) == 2
        finally:
            embed_mod._daemon_checked = False
            embed_mod._daemon_available = False
            embed_mod._daemon_url = None

    def test_fallback_to_local_on_daemon_failure(self):
        embed_mod._daemon_checked = True
        embed_mod._daemon_available = True
        embed_mod._daemon_url = "http://127.0.0.1:1"
        try:
            # Should fall back to local embedding (model is cached from other tests)
            result = embed_mod.embed_query("hello", "BAAI/bge-small-en-v1.5")
            assert len(result) == 384
        finally:
            embed_mod._daemon_checked = False
            embed_mod._daemon_available = False
            embed_mod._daemon_url = None
