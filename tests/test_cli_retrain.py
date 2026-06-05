"""CLI tests for ``vstash retrain``'s --eval-queries / --training-queries flags.

These tests focus on the JSONL loader, the CLI surface area, and the
forwarding of parsed labeled queries into ``run_retrain`` -- they do not
run the actual fine-tune (which requires sentence-transformers + torch).
``run_retrain`` is monkeypatched to a recording stub so we can assert
exactly what arguments would have been passed in a real run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.conftest import requires_sqlite_vec
from vstash.cli import app
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec

runner = CliRunner()


@pytest.fixture
def patched_cli(populated_store: VstashStore, monkeypatch) -> dict[str, Any]:
    """Patch the CLI so retrain runs without sentence-transformers.

    Returns a dict ``{"calls": [...]}`` that captures every call made to
    the patched ``run_retrain`` so tests can assert the kwargs it would
    have received.  ``calls[-1]`` is the most recent invocation.
    """
    import vstash.cli as cli_mod
    from vstash.config import load_config

    # Wire the test store into the CLI. The retrain commands live in the
    # cli._retrain module since the #284 split and read _get_store from there.
    monkeypatch.setattr(
        cli_mod._retrain,
        "_get_store",
        lambda cfg=None, warm=False, profile=None: (load_config(), populated_store),
    )

    # ``vstash retrain`` short-circuits when the store has fewer than 10
    # chunks (the corpus-too-small guard).  populated_store has 4-6
    # chunks so we'd hit that branch on every test; stub stats() to
    # report a corpus large enough to pass.
    real_stats = populated_store.stats()
    monkeypatch.setattr(
        populated_store,
        "stats",
        lambda: real_stats.model_copy(update={"chunks": 100}),
    )

    calls: list[dict] = []

    # Stub ``run_retrain`` -- avoids needing sentence-transformers in CI
    # and lets us inspect the kwargs precisely.
    def fake_run_retrain(store, **kwargs):
        calls.append({"store": store, **kwargs})
        return SimpleNamespace(
            n_pairs=10,
            baseline=None,
            final=None,
            output_path=str(Path(kwargs["output_path"]).expanduser()),
            gated_out=False,
            min_gain=kwargs.get("min_gain", 0.0),
            delta_ndcg=0.0,
        )

    # The CLI imports ``run_retrain`` lazily inside the command body, so
    # we patch the source attribute on the module that supplies it.
    import vstash.retrain as retrain_mod

    monkeypatch.setattr(retrain_mod, "retrain", fake_run_retrain)
    return {"calls": calls}


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


class TestEvalQueriesFlag:
    """``--eval-queries`` parses, validates, and forwards to run_retrain."""

    def test_forwards_parsed_eval_queries_to_run_retrain(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        eval_path = _write_jsonl(
            tmp_path / "eval.jsonl",
            [
                {"query": "what is python", "relevant_paths": ["/test/python_guide.md"]},
                {"query": "ml basics", "relevant_paths": ["/test/ml_intro.md"]},
            ],
        )
        result = runner.invoke(
            app,
            [
                "retrain",
                "--eval-queries",
                str(eval_path),
                "--output",
                str(tmp_path / "out_model"),
                "--no-bulk-mine",
            ],
        )
        assert result.exit_code == 0, result.stdout
        # Rich may wrap the line printed by the CLI when the test
        # terminal is narrow; collapse newlines/spaces before
        # substring-matching the messages we care about.
        flat = " ".join(result.stdout.split())
        assert "2 labeled" in flat
        assert "gate uses these instead of internal split" in flat
        assert len(patched_cli["calls"]) == 1
        kwargs = patched_cli["calls"][0]
        assert kwargs["eval_queries"] is not None
        assert len(kwargs["eval_queries"]) == 2
        assert kwargs["eval_queries"][0]["query"] == "what is python"
        assert kwargs["eval_queries"][0]["relevant_paths"] == ["/test/python_guide.md"]

    def test_eval_queries_default_is_none(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        train_path = _write_jsonl(
            tmp_path / "train.jsonl",
            [{"query": "q", "relevant_paths": ["/test/python_guide.md"]}],
        )
        result = runner.invoke(
            app,
            [
                "retrain",
                "--training-queries",
                str(train_path),
                "--output",
                str(tmp_path / "out_model"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert patched_cli["calls"][0]["eval_queries"] is None

    def test_file_not_found_exits_with_clear_message(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        result = runner.invoke(
            app,
            [
                "retrain",
                "--eval-queries",
                str(tmp_path / "missing.jsonl"),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "--eval-queries" in flat
        assert "file not found" in flat
        assert patched_cli["calls"] == []

    def test_invalid_json_exits_with_clear_message(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not valid json\n")
        result = runner.invoke(
            app,
            ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "failed to parse" in flat
        assert "Expected JSONL" in flat
        assert patched_cli["calls"] == []

    def test_missing_required_keys_rejected_per_line(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = _write_jsonl(
            tmp_path / "missing_keys.jsonl",
            [{"query": "q", "relevant_paths": ["/test/python_guide.md"]}, {"query": "no rel"}],
        )
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "line 2" in flat
        assert patched_cli["calls"] == []

    def test_empty_relevant_paths_rejected(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = _write_jsonl(tmp_path / "empty_rel.jsonl", [{"query": "q", "relevant_paths": []}])
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "empty or non-list" in flat

    def test_empty_file_rejected(self, tmp_path: Path, patched_cli: dict[str, Any]) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(empty), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "no records" in flat

    def test_non_string_query_rejected(self, tmp_path: Path, patched_cli: dict[str, Any]) -> None:
        bad = _write_jsonl(
            tmp_path / "bad_query.jsonl",
            [{"query": 42, "relevant_paths": ["/test/python_guide.md"]}],
        )
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "line 1" in flat
        assert "non-empty string" in flat

    def test_empty_string_query_rejected(self, tmp_path: Path, patched_cli: dict[str, Any]) -> None:
        bad = _write_jsonl(
            tmp_path / "blank_query.jsonl",
            [{"query": "  ", "relevant_paths": ["/test/python_guide.md"]}],
        )
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "non-empty string" in flat

    def test_non_string_relevant_path_rejected(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = _write_jsonl(
            tmp_path / "bad_paths.jsonl",
            [{"query": "q", "relevant_paths": ["/ok.md", 7]}],
        )
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "Every relevant path must be a non-empty string" in flat

    def test_directory_path_rejected(self, tmp_path: Path, patched_cli: dict[str, Any]) -> None:
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(tmp_path), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "path is not a file" in flat

    def test_invalid_json_includes_line_number(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = tmp_path / "with_bad_line.jsonl"
        bad.write_text(
            json.dumps({"query": "ok", "relevant_paths": ["/a"]})
            + "\n{not valid json\n"
            + json.dumps({"query": "ok2", "relevant_paths": ["/b"]})
            + "\n"
        )
        result = runner.invoke(
            app, ["retrain", "--eval-queries", str(bad), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "line 2" in flat
        assert "failed to parse" in flat

    def test_mutually_exclusive_with_no_eval(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        eval_path = _write_jsonl(
            tmp_path / "eval.jsonl",
            [{"query": "q", "relevant_paths": ["/test/python_guide.md"]}],
        )
        result = runner.invoke(
            app,
            [
                "retrain",
                "--eval-queries",
                str(eval_path),
                "--no-eval",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "mutually exclusive" in flat
        assert patched_cli["calls"] == []

    def test_warns_on_relevant_paths_not_in_store(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        eval_path = _write_jsonl(
            tmp_path / "eval.jsonl",
            [{"query": "q", "relevant_paths": ["/missing/in_store.md"]}],
        )
        result = runner.invoke(
            app,
            [
                "retrain",
                "--eval-queries",
                str(eval_path),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        flat = " ".join(result.stdout.split())
        # The warning hits before run_retrain so the call still goes through.
        assert "1 relevant_paths reference" in flat
        assert "/missing/in_store.md" in flat
        assert patched_cli["calls"][0]["eval_queries"][0]["relevant_paths"] == [
            "/missing/in_store.md"
        ]


class TestTrainingQueriesUsesSharedLoader:
    """Refactor sanity: --training-queries still validates via the helper."""

    def test_training_queries_invalid_json_rejected(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n")
        result = runner.invoke(
            app,
            [
                "retrain",
                "--training-queries",
                str(bad),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "--training-queries" in flat
        assert "failed to parse" in flat

    def test_training_queries_missing_relevant_paths_rejected(
        self, tmp_path: Path, patched_cli: dict[str, Any]
    ) -> None:
        bad = _write_jsonl(tmp_path / "bad.jsonl", [{"query": "q"}])
        result = runner.invoke(
            app,
            [
                "retrain",
                "--training-queries",
                str(bad),
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "line 1" in flat
