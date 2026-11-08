"""The local judge around ask: claim splitting, gate, verify, and the service wiring (no model needed)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vstash import judge as judge_mod
from vstash.config import JudgeConfig, VstashConfig
from vstash.models import AskResult, SearchResult
from vstash.services.ask import judged_ask


def _chunk(text: str, i: int = 0) -> SearchResult:
    return SearchResult(
        chunk_id=i, text=text, title=f"doc{i}", path=f"/d/{i}.md", chunk=0, score=0.01
    )


class FakeJudge:
    """Answerable when the context mentions the query's last word; a claim is supported when its
    first word appears in the context."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def answerable(self, query: str, context: str) -> float:
        self.calls.append("answerable")
        return 0.9 if query.split()[-1].strip("?").lower() in context.lower() else 0.05

    def supported(self, claims: list[str], context: str) -> list[float]:
        self.calls.append(f"supported:{len(claims)}")
        return [0.95 if c.split()[0].lower() in context.lower() else 0.1 for c in claims]


def _cfg(**judge) -> VstashConfig:
    return VstashConfig(judge=JudgeConfig(enabled=True, **judge))


class TestSplitClaims:
    def test_sentences_and_list_items(self) -> None:
        text = "Paris is the capital of France. It has about two million people.\n- The Louvre is a museum in Paris."
        assert judge_mod.split_claims(text) == [
            "Paris is the capital of France.",
            "It has about two million people.",
            "The Louvre is a museum in Paris.",
        ]

    def test_drops_headings_code_short_lines_and_citations(self) -> None:
        text = "## Summary\nSure.\nIn short:\n```python\nx = 1\n```\nThe store uses SQLite in WAL mode [1]."
        assert judge_mod.split_claims(text) == ["The store uses SQLite in WAL mode."]

    def test_empty(self) -> None:
        assert judge_mod.split_claims("") == []


class TestGateAndVerify:
    def test_gate_below_threshold(self) -> None:
        j = judge_mod.gate(
            FakeJudge(), "who built the bridge?", [_chunk("A note about cats.")], JudgeConfig()
        )
        assert j.gated and j.answerable == pytest.approx(0.05)

    def test_gate_off_never_gates(self) -> None:
        j = judge_mod.gate(
            FakeJudge(), "who built the bridge?", [_chunk("cats")], JudgeConfig(gate=False)
        )
        assert not j.gated

    def test_no_chunks_is_unanswerable(self) -> None:
        fake = FakeJudge()
        j = judge_mod.gate(fake, "anything?", [], JudgeConfig())
        assert j.gated and j.answerable == 0.0 and fake.calls == []

    def test_verify_flags_unsupported(self) -> None:
        chunks = [_chunk("SQLite stores everything in one file.")]
        base = judge_mod.gate(FakeJudge(), "what does sqlite store?", chunks, JudgeConfig())
        j = judge_mod.verify(
            FakeJudge(),
            base,
            "SQLite keeps all data in a single file. Postgres would be faster for this workload.",
            chunks,
            JudgeConfig(),
        )
        assert [c.flagged for c in j.claims] == [False, True]
        assert j.supported_share == 0.5
        assert [c.text for c in j.flagged] == ["Postgres would be faster for this workload."]

    def test_max_claims(self) -> None:
        many = " ".join(f"Sentence number {i} is a claim here." for i in range(10))
        j = judge_mod.verify(
            FakeJudge(),
            judge_mod.Judgment(answerable=1.0),
            many,
            [_chunk("x")],
            JudgeConfig(max_claims=3),
        )
        assert len(j.claims) == 3


class TestJudgedAsk:
    def test_disabled_is_plain_ask_full(self) -> None:
        fake = AskResult(content="hi")
        with (
            patch("vstash.services.ask._chat_ask_full", return_value=fake) as ask,
            patch.object(judge_mod, "get_judge") as gj,
        ):
            out = judged_ask("q?", [_chunk("x")], VstashConfig())
        assert out is fake and ask.call_count == 1 and gj.call_count == 0

    def test_gated_skips_the_llm(self) -> None:
        with (
            patch.object(judge_mod, "get_judge", return_value=FakeJudge()),
            patch("vstash.services.ask._chat_ask_full") as ask,
        ):
            out = judged_ask("who built the bridge?", [_chunk("cats and dogs")], _cfg())
        ask.assert_not_called()
        assert out.backend is None and out.judgment is not None and out.judgment.gated
        assert "couldn't find this in your memory" in out.content

    def test_answerable_calls_llm_and_verifies(self) -> None:
        fake_judge = FakeJudge()
        reply = AskResult(
            content="SQLite keeps everything in one file. Redis is used for caching.",
            backend="ollama",
        )
        with (
            patch.object(judge_mod, "get_judge", return_value=fake_judge),
            patch("vstash.services.ask._chat_ask_full", return_value=reply),
        ):
            out = judged_ask(
                "what does it use? sqlite", [_chunk("vstash uses SQLite in one file.")], _cfg()
            )
        assert out.content == reply.content and out.backend == "ollama"
        assert out.judgment is not None and not out.judgment.gated
        assert [c.flagged for c in out.judgment.claims] == [False, True]
        assert fake_judge.calls == ["answerable", "supported:2"]

    def test_verify_off(self) -> None:
        reply = AskResult(content="SQLite keeps everything in one file.")
        with (
            patch.object(judge_mod, "get_judge", return_value=FakeJudge()),
            patch("vstash.services.ask._chat_ask_full", return_value=reply),
        ):
            out = judged_ask("what? sqlite", [_chunk("sqlite")], _cfg(verify=False))
        assert out.judgment is not None and out.judgment.claims == []


def test_config_defaults_off() -> None:
    cfg = VstashConfig()
    assert cfg.judge.enabled is False and cfg.judge.model == "LiquidAI/d1-2B"


def test_config_from_toml_section() -> None:
    cfg = VstashConfig.model_validate({"judge": {"enabled": True, "gate_threshold": 0.35}})
    assert cfg.judge.enabled and cfg.judge.gate_threshold == 0.35


def test_get_judge_keys_the_cache_on_the_verify_adapter():
    """A judge with a fact-checking adapter is a different model: it must not reuse the plain one."""
    built: list[tuple] = []

    class Spy:
        def __init__(self, *args) -> None:
            built.append(args)

    with patch.object(judge_mod, "D1Judge", Spy), patch.dict(judge_mod._cache, clear=True):
        plain = judge_mod.get_judge(JudgeConfig(enabled=True))
        again = judge_mod.get_judge(JudgeConfig(enabled=True))
        tuned = judge_mod.get_judge(JudgeConfig(enabled=True, verify_adapter="/a/jev"))
    assert plain is again
    assert tuned is not plain
    assert built == [("LiquidAI/d1-2B", "auto", None), ("LiquidAI/d1-2B", "auto", "/a/jev")]
