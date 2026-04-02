"""End-to-end tests for API retry with backoff in chat.py.

Tests that the retry mechanism integrates correctly with the backend
dispatch layer and that streaming functions are not broken.
"""

from __future__ import annotations

import pytest

from vstash.chat import _is_retryable, _retry_call, ask
from vstash.config import VstashConfig
from vstash.models import SearchResult


def _make_chunk(text: str = "test context") -> SearchResult:
    return SearchResult(chunk_id=0, text=text, title="TestDoc", path="/test.md", chunk=0, score=0.5)


class TestRetryE2E:
    """End-to-end retry behavior through the public ask() function."""

    def test_ask_raises_on_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ask() should raise ValueError for missing API key (not retried)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = VstashConfig.model_validate(
            {"inference": {"backend": "openai"}, "openai": {"api_key": ""}}
        )
        chunks = [_make_chunk()]
        with pytest.raises((ValueError, ImportError)):
            ask("test query", chunks, cfg)

    def test_invalid_backend_not_registered(self) -> None:
        """Ensure an 'invalid' backend is not registered in _BACKENDS."""
        from vstash.chat import _BACKENDS

        assert "invalid" not in _BACKENDS

    def test_retry_decorator_preserves_successful_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_retry_ask should return the result of a successful call."""
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.001)

        def success(*a, **kw):  # type: ignore[no-untyped-def]
            return "answer"

        success.__name__ = "success"
        assert _retry_call(success) == "answer"

    def test_retry_transient_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate: first call gets 429, second succeeds."""
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.001)
        attempts = []

        def backend(*a, **kw):  # type: ignore[no-untyped-def]
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("HTTP 429 rate limit exceeded")
            return "ok after retry"

        backend.__name__ = "backend"
        result = _retry_call(backend)
        assert result == "ok after retry"
        assert len(attempts) == 2

    def test_retry_all_transient_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All 3 retries hit 503 → should raise after max retries."""
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.001)
        attempts = []

        def backend(*a, **kw):  # type: ignore[no-untyped-def]
            attempts.append(1)
            raise ConnectionError("503 Service Unavailable")

        backend.__name__ = "backend"
        with pytest.raises(ConnectionError, match="503"):
            _retry_call(backend)
        assert len(attempts) == 3

    def test_non_retryable_error_fails_immediately(self) -> None:
        """Authentication errors should not be retried."""
        attempts = []

        def backend(*a, **kw):  # type: ignore[no-untyped-def]
            attempts.append(1)
            raise ConnectionError("401 Unauthorized: invalid API key")

        backend.__name__ = "backend"
        with pytest.raises(ConnectionError, match="401"):
            _retry_call(backend)
        assert len(attempts) == 1

    def test_timeout_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TimeoutError should be caught and retried."""
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.001)
        attempts = []

        def backend(*a, **kw):  # type: ignore[no-untyped-def]
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("connection timed out")
            return "recovered"

        backend.__name__ = "backend"
        result = _retry_call(backend)
        assert result == "recovered"
        assert len(attempts) == 2

    def test_os_error_connection_reset_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError (connection reset) should be retried."""
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.001)
        attempts = []

        def backend(*a, **kw):  # type: ignore[no-untyped-def]
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("connection reset by peer")
            return "reconnected"

        backend.__name__ = "backend"
        result = _retry_call(backend)
        assert result == "reconnected"


class TestRetryPatterns:
    """Verify the retryable pattern matching is correct."""

    @pytest.mark.parametrize(
        "msg",
        [
            "rate limit exceeded",
            "HTTP 429 Too Many Requests",
            "503 Service Unavailable",
            "502 Bad Gateway",
            "connection timed out",
            "Request timeout",
            "connection reset by peer",
            "connection refused",
            "temporarily unavailable",
            "Internal server error",
            "Model overloaded, try again",
        ],
    )
    def test_retryable_messages(self, msg: str) -> None:
        assert _is_retryable(ConnectionError(msg))

    @pytest.mark.parametrize(
        "msg",
        [
            "401 Unauthorized",
            "400 Bad Request",
            "404 Not Found",
            "Invalid API key",
            "Model not found: gpt-5",
            "Permission denied",
        ],
    )
    def test_non_retryable_messages(self, msg: str) -> None:
        assert not _is_retryable(ConnectionError(msg))
