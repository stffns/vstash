"""Tests for vstash.chat — prompt building and backend dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vstash.chat import _build_messages, _build_prompt, _is_retryable, _retry_call, ask, ask_full
from vstash.config import VstashConfig
from vstash.models import AskResult, SearchResult


def _make_chunk(text: str, title: str = "TestDoc") -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(chunk_id=0, text=text, title=title, path="/test/doc.md", chunk=0, score=0.5)


class TestBuildPrompt:
    """Test prompt construction from chunks."""

    def test_empty_chunks(self) -> None:
        prompt = _build_prompt("test question", [])
        assert "No relevant context" in prompt
        assert "test question" in prompt

    def test_single_chunk(self) -> None:
        chunks = [_make_chunk("This is context.")]
        prompt = _build_prompt("what is this?", chunks)
        assert "Context 1" in prompt
        assert "This is context." in prompt
        assert "what is this?" in prompt
        assert "[TestDoc]" in prompt

    def test_multiple_chunks(self) -> None:
        chunks = [
            _make_chunk("First context.", "Doc1"),
            _make_chunk("Second context.", "Doc2"),
        ]
        prompt = _build_prompt("query", chunks)
        assert "Context 1 [Doc1]" in prompt
        assert "Context 2 [Doc2]" in prompt


class TestBuildMessages:
    """Test message list construction."""

    def test_basic_messages(self) -> None:
        chunks = [_make_chunk("context")]
        messages = _build_messages("question", chunks)
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert len(messages) == 2

    def test_with_history(self) -> None:
        chunks = [_make_chunk("context")]
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        messages = _build_messages("second question", chunks, history=history)
        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "first question"
        assert messages[2]["role"] == "assistant"
        assert messages[-1]["role"] == "user"

    def test_no_history_is_default(self) -> None:
        chunks = [_make_chunk("context")]
        messages = _build_messages("question", chunks)
        assert len(messages) == 2


class TestBackendDispatch:
    """Test that unknown backends raise ValueError."""

    def test_invalid_backend_raises_validation_error(self) -> None:
        with pytest.raises(Exception):
            VstashConfig.model_validate({"inference": {"backend": "invalid"}})

    def test_valid_backends_have_dispatch_entries(self) -> None:
        from vstash.chat import _BACKENDS

        assert "cerebras" in _BACKENDS
        assert "ollama" in _BACKENDS
        assert "openai" in _BACKENDS


class TestRetryLogic:
    """Test retry with exponential backoff."""

    def test_is_retryable_rate_limit(self) -> None:
        assert _is_retryable(ConnectionError("rate limit exceeded"))

    def test_is_retryable_429(self) -> None:
        assert _is_retryable(ConnectionError("HTTP 429 Too Many Requests"))

    def test_is_retryable_503(self) -> None:
        assert _is_retryable(ConnectionError("503 Service Unavailable"))

    def test_is_retryable_timeout(self) -> None:
        assert _is_retryable(TimeoutError("connection timed out"))

    def test_is_retryable_timeout_empty_message(self) -> None:
        """TimeoutError with no message should still be retryable (by type)."""
        assert _is_retryable(TimeoutError())

    def test_not_retryable_auth_error(self) -> None:
        assert not _is_retryable(ConnectionError("401 Unauthorized"))

    def test_not_retryable_bad_request(self) -> None:
        assert not _is_retryable(ConnectionError("400 Bad Request"))

    def test_retry_succeeds_on_second_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.01)
        call_count = 0

        def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("503 Service Unavailable")
            return "success"

        flaky.__name__ = "flaky"
        result = _retry_call(flaky)
        assert result == "success"
        assert call_count == 2

    def test_retry_gives_up_after_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("vstash.chat._BASE_DELAY", 0.01)

        def always_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("503 Service Unavailable")

        always_fail.__name__ = "always_fail"
        with pytest.raises(ConnectionError, match="503"):
            _retry_call(always_fail)

    def test_no_retry_on_non_retryable(self) -> None:
        call_count = 0

        def auth_error(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            raise ConnectionError("401 Unauthorized")

        auth_error.__name__ = "auth_error"
        with pytest.raises(ConnectionError, match="401"):
            _retry_call(auth_error)
        assert call_count == 1


# ------------------------------------------------------------------ #
# ask_full() / AskResult (#303)                                        #
# ------------------------------------------------------------------ #


def _cerebras_response(content: str, *, reasoning: str | None = None, usage: dict | None = None):
    """Mimic the shape ``cerebras.cloud.sdk.Cerebras.chat.completions.create``
    returns: ``response.choices[0].message.{content,reasoning}`` plus
    ``response.usage``."""
    msg = SimpleNamespace(content=content)
    if reasoning is not None:
        msg.reasoning = reasoning
    choice = SimpleNamespace(message=msg)
    usage_obj = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(choices=[choice], usage=usage_obj)


def _cfg_cerebras(model: str = "gpt-oss-120b") -> VstashConfig:
    return VstashConfig.model_validate(
        {
            "inference": {"backend": "cerebras", "model": model},
            "cerebras": {"api_key": "test-key"},
        }
    )


class TestAskFullCerebras:
    """Cerebras is the priority backend (Merken Phase 2 driver)."""

    def test_populates_reasoning_and_usage_when_present(self) -> None:
        cfg = _cfg_cerebras("gpt-oss-120b")
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        client.chat.completions.create.return_value = _cerebras_response(
            "the answer",
            reasoning="step 1: ... step 2: ...",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        with patch("cerebras.cloud.sdk.Cerebras", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert isinstance(result, AskResult)
        assert result.content == "the answer"
        assert result.reasoning == "step 1: ... step 2: ..."
        assert result.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        assert result.backend == "cerebras"
        assert result.model == "gpt-oss-120b"

    def test_reasoning_none_when_model_omits_it(self) -> None:
        """llama3.1-8b on Cerebras has no reasoning channel; result.reasoning
        must be None, not empty string."""
        cfg = _cfg_cerebras("llama3.1-8b")
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        # No ``reasoning`` attribute on the mocked message at all.
        client.chat.completions.create.return_value = _cerebras_response("plain answer")
        with patch("cerebras.cloud.sdk.Cerebras", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert result.content == "plain answer"
        assert result.reasoning is None
        assert result.backend == "cerebras"
        assert result.model == "llama3.1-8b"

    def test_empty_string_reasoning_collapses_to_none(self) -> None:
        """``message.reasoning = ""`` (some SDK versions) must collapse to None,
        not become a valid empty trace."""
        cfg = _cfg_cerebras("gpt-oss-120b")
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        client.chat.completions.create.return_value = _cerebras_response("answer", reasoning="")
        with patch("cerebras.cloud.sdk.Cerebras", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert result.reasoning is None


class TestAskFullOpenAI:
    """OpenAI / OpenAI-compat. Reasoning channel is non-standard so the
    common case is ``reasoning=None``; usage is always populated."""

    def test_no_reasoning_for_vanilla_openai(self) -> None:
        cfg = VstashConfig.model_validate(
            {
                "inference": {"backend": "openai"},
                "openai": {"api_key": "k", "model": "gpt-4o-mini"},
            }
        )
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        msg = SimpleNamespace(content="hello")
        choice = SimpleNamespace(message=msg)
        usage_obj = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice], usage=usage_obj
        )
        with patch("openai.OpenAI", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert result.content == "hello"
        assert result.reasoning is None
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert result.backend == "openai"
        assert result.model == "gpt-4o-mini"


class TestAskFullOllama:
    """Ollama exposes thinking content for qwen3 thinking-enabled models."""

    def test_with_thinking_and_eval_counts(self) -> None:
        cfg = VstashConfig.model_validate(
            {
                "inference": {"backend": "ollama"},
                "ollama": {"host": "http://localhost:11434", "model": "qwen3:8b"},
            }
        )
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        client.chat.return_value = {
            "message": {"content": "answer", "thinking": "trace..."},
            "prompt_eval_count": 30,
            "eval_count": 12,
        }
        with patch("ollama.Client", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert result.content == "answer"
        assert result.reasoning == "trace..."
        assert result.usage == {
            "prompt_tokens": 30,
            "completion_tokens": 12,
            "total_tokens": 42,
        }
        assert result.backend == "ollama"
        assert result.model == "qwen3:8b"

    def test_without_thinking_returns_none(self) -> None:
        cfg = VstashConfig.model_validate(
            {
                "inference": {"backend": "ollama"},
                "ollama": {"host": "http://localhost:11434", "model": "llama3:8b"},
            }
        )
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        client.chat.return_value = {
            "message": {"content": "plain answer"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        with patch("ollama.Client", return_value=client):
            result = ask_full("q", chunks, cfg)
        assert result.reasoning is None
        assert result.content == "plain answer"


class TestAskBackwardCompat:
    """``ask()`` keeps its ``-> str`` contract via ``ask_full().content``."""

    def test_ask_returns_plain_string(self) -> None:
        cfg = _cfg_cerebras("gpt-oss-120b")
        chunks = [_make_chunk("ctx")]
        client = MagicMock()
        client.chat.completions.create.return_value = _cerebras_response(
            "the answer", reasoning="hidden cot"
        )
        with patch("cerebras.cloud.sdk.Cerebras", return_value=client):
            result = ask("q", chunks, cfg)
        assert isinstance(result, str)
        assert result == "the answer"
