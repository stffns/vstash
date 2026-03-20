"""Tests for vstash.chat — prompt building and backend dispatch."""

from __future__ import annotations

import pytest

from vstash.chat import _build_messages, _build_prompt
from vstash.config import VstashConfig
from vstash.models import SearchResult


def _make_chunk(text: str, title: str = "TestDoc") -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        text=text, title=title, path="/test/doc.md", chunk=0, score=0.5
    )


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
