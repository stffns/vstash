"""Tests for vstash.retrain_synth -- LLM-based query synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vstash.config import VstashConfig
from vstash.retrain_synth import (
    _build_prompt,
    _parse_queries,
    _prompt_hash,
    synthesize_queries,
)


# ------------------------------------------------------------------ #
# Prompt helpers                                                       #
# ------------------------------------------------------------------ #


class TestPromptBuilder:
    def test_embeds_n_and_passage(self) -> None:
        p = _build_prompt("some passage text", n=3)
        assert "3 short natural queries" in p
        assert "some passage text" in p

    def test_truncates_huge_passage(self) -> None:
        big = "x" * 10_000
        p = _build_prompt(big, n=2)
        # The passage is truncated to _MAX_PASSAGE_CHARS (1500) and
        # embedded verbatim between triple quotes. The template itself
        # contains a couple of x-letters in ordinary words ("exactly",
        # "explanation"), which is why we assert on the embedded run
        # length rather than a total count.
        assert "x" * 1500 in p
        assert "x" * 1501 not in p

    def test_prompt_hash_is_deterministic(self) -> None:
        h1 = _prompt_hash("abc")
        h2 = _prompt_hash("abc")
        h3 = _prompt_hash("abd")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16


# ------------------------------------------------------------------ #
# Parser                                                               #
# ------------------------------------------------------------------ #


class TestParseQueries:
    def test_clean_json_array(self) -> None:
        out = _parse_queries('["what is rrf", "how hybrid search works"]', n=5)
        assert out == ["what is rrf", "how hybrid search works"]

    def test_strips_markdown_fences(self) -> None:
        raw = '```json\n["q1", "q2"]\n```'
        assert _parse_queries(raw, n=5) == ["q1", "q2"]

    def test_extracts_array_from_preamble(self) -> None:
        raw = 'Sure, here are two queries:\n["first one", "second one"]\nHope that helps.'
        assert _parse_queries(raw, n=5) == ["first one", "second one"]

    def test_caps_at_n(self) -> None:
        raw = '["a", "b", "c", "d"]'
        assert _parse_queries(raw, n=2) == ["a", "b"]

    def test_drops_empty_strings(self) -> None:
        raw = '["", "real one", "   "]'
        assert _parse_queries(raw, n=5) == ["real one"]

    def test_garbage_returns_empty(self) -> None:
        assert _parse_queries("absolutely not json at all", n=3) == []

    def test_non_list_json_returns_empty(self) -> None:
        assert _parse_queries('{"queries": ["a"]}', n=3) == []


# ------------------------------------------------------------------ #
# End-to-end synthesize_queries with mocked LLM + cache                #
# ------------------------------------------------------------------ #


@pytest.fixture
def cfg() -> VstashConfig:
    """Vstash config with an explicit openai backend so the LLM path
    is deterministic in tests (no local auto-detection)."""
    return VstashConfig.model_validate(
        {
            "inference": {"backend": "openai"},
            "openai": {"api_key": "test-key", "model": "test-model"},
        }
    )


class TestSynthesizeQueries:
    def test_happy_path_produces_map(self, cfg: VstashConfig, tmp_path: Path) -> None:
        chunks = [
            {"id": 1, "text": "Reciprocal rank fusion combines two rankings."},
            {"id": 2, "text": "NDCG is a ranking metric based on DCG and IDCG."},
        ]
        fake = MagicMock()
        fake.choices = [MagicMock()]
        fake.choices[0].message.content = '["what is rrf", "how rrf combines rankings"]'

        with patch(
            "vstash.retrain_synth._llm_complete", return_value=fake.choices[0].message.content
        ):
            out = synthesize_queries(
                chunks,
                cfg,
                n_per_chunk=2,
                cache_path=tmp_path / "synth.jsonl",
            )

        assert set(out.keys()) == {1, 2}
        for qs in out.values():
            assert qs == ["what is rrf", "how rrf combines rankings"]

        # Cache file should have one record per chunk.
        lines = [
            json.loads(line) for line in (tmp_path / "synth.jsonl").read_text().splitlines() if line
        ]
        assert len(lines) == 2
        assert {r["chunk_id"] for r in lines} == {1, 2}

    def test_cache_hit_skips_llm(self, cfg: VstashConfig, tmp_path: Path) -> None:
        cache = tmp_path / "synth.jsonl"
        chunks = [{"id": 1, "text": "Some passage"}]
        with patch("vstash.retrain_synth._llm_complete", return_value='["q1", "q2"]') as mock_llm:
            synthesize_queries(chunks, cfg, n_per_chunk=2, cache_path=cache)
        # Second pass: should hit the cache, zero LLM calls.
        with patch("vstash.retrain_synth._llm_complete", return_value='["BAD"]') as mock_llm:
            out = synthesize_queries(chunks, cfg, n_per_chunk=2, cache_path=cache)
        assert mock_llm.call_count == 0
        assert out == {1: ["q1", "q2"]}

    def test_llm_failure_skips_chunk_continues(self, cfg: VstashConfig, tmp_path: Path) -> None:
        chunks = [
            {"id": 1, "text": "chunk one"},
            {"id": 2, "text": "chunk two"},
        ]
        responses = iter(
            [
                RuntimeError("boom"),  # first call raises
                '["ok1", "ok2"]',  # second call succeeds
            ]
        )

        def side_effect(*_a: Any, **_k: Any) -> str:
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("vstash.retrain_synth._llm_complete", side_effect=side_effect):
            out = synthesize_queries(chunks, cfg, n_per_chunk=2, cache_path=tmp_path / "c.jsonl")

        # Chunk 1 failed -> absent. Chunk 2 succeeded -> present.
        assert 1 not in out
        assert out[2] == ["ok1", "ok2"]

    def test_unparseable_response_skips_chunk(self, cfg: VstashConfig, tmp_path: Path) -> None:
        chunks = [{"id": 1, "text": "text"}, {"id": 2, "text": "text2"}]
        with patch(
            "vstash.retrain_synth._llm_complete",
            side_effect=["garbage nope", '["good1", "good2"]'],
        ):
            out = synthesize_queries(chunks, cfg, n_per_chunk=2, cache_path=tmp_path / "c.jsonl")
        assert 1 not in out
        assert out[2] == ["good1", "good2"]

    def test_progress_callback_invoked_for_every_chunk(
        self, cfg: VstashConfig, tmp_path: Path
    ) -> None:
        chunks = [{"id": i, "text": f"t{i}"} for i in range(3)]
        calls: list[tuple[int, int]] = []
        with patch("vstash.retrain_synth._llm_complete", return_value='["a", "b"]'):
            synthesize_queries(
                chunks,
                cfg,
                n_per_chunk=2,
                cache_path=tmp_path / "c.jsonl",
                on_progress=lambda done, total: calls.append((done, total)),
            )
        assert calls == [(1, 3), (2, 3), (3, 3)]
