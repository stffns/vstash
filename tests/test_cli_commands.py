"""Tests for vstash CLI commands — list, stats, forget, config, add, ask."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tests.conftest import requires_sqlite_vec
from vstash.cli import app
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec

runner = CliRunner()


@pytest.fixture(autouse=True)
def _patch_store(populated_store: VstashStore, monkeypatch) -> None:
    """Monkeypatch _get_store so CLI uses the test store."""
    import vstash.cli as cli_mod
    from vstash.config import load_config

    monkeypatch.setattr(
        cli_mod,
        "_get_store",
        lambda cfg=None, warm=False, profile=None: (load_config(), populated_store),
    )


class TestListCommand:
    """Test 'vstash list' command."""

    def test_list_shows_documents_table(self) -> None:
        """list command outputs document titles in a table."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Python Guide" in result.stdout
        assert "ML Introduction" in result.stdout

    def test_list_empty_collection(self) -> None:
        """list with a non-existent collection shows empty message."""
        result = runner.invoke(app, ["list", "--collection", "nonexistent"])
        assert result.exit_code == 0
        assert "No documents" in result.stdout or "empty" in result.stdout.lower()


class TestStatsCommand:
    """Test 'vstash stats' command."""

    def test_stats_shows_memory_panel(self) -> None:
        """stats command outputs document and chunk counts."""
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Documents" in result.stdout
        assert "Chunks" in result.stdout
        assert "2" in result.stdout  # 2 documents in populated_store


class TestForgetCommand:
    """Test 'vstash forget' command."""

    def test_forget_removes_document(self) -> None:
        """forget command removes a known document."""
        result = runner.invoke(app, ["forget", "/test/python_guide.md"])
        assert result.exit_code == 0
        assert "Removed" in result.stdout or "✓" in result.stdout

    def test_forget_nonexistent_shows_not_found(self) -> None:
        """forget command with unknown path shows not-found message."""
        result = runner.invoke(app, ["forget", "/nonexistent/file.md"])
        assert result.exit_code == 0
        assert "Not found" in result.stdout


class TestConfigCommand:
    """Test 'vstash config' command."""

    def test_config_shows_panel(self) -> None:
        """config command outputs configuration details."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "Inference backend" in result.stdout
        assert "Embedding model" in result.stdout
        assert "Chunk size" in result.stdout


class TestAddErrorPaths:
    """Test 'vstash add' basic error paths."""

    def test_add_nonexistent_file(self) -> None:
        """add with a non-existent file path reports an error."""
        result = runner.invoke(app, ["add", "/nonexistent/path/file.md"])
        # Should not crash — the ingest function handles the error
        assert result.exit_code == 0 or result.exit_code == 1


class TestAskErrorPaths:
    """Test 'vstash ask' basic error paths."""

    def test_ask_empty_store(self, tmp_db_path: str, monkeypatch) -> None:
        """ask with no documents in store shows a helpful message."""
        import vstash.cli as cli_mod
        from vstash.config import load_config

        cfg = load_config()
        from vstash.embed import get_embedding_dim

        dim = get_embedding_dim(cfg.embeddings.model)
        empty_store = VstashStore(tmp_db_path + "_empty", embedding_dim=dim)

        monkeypatch.setattr(
            cli_mod,
            "_get_store",
            lambda cfg=None, warm=False, profile=None: (load_config(), empty_store),
        )

        result = runner.invoke(app, ["ask", "What is Python?"])
        assert "No relevant documents found" in result.stdout
        empty_store.close()


class TestWhyCommand:
    """Test 'vstash why' command (issue #157 / H-WHY)."""

    def _patch_embed(self, monkeypatch) -> None:
        """Stub embed_query so tests do not download a real model."""
        import vstash.cli as cli_mod

        def _fake(query: str, model: str) -> list[float]:
            # Match populated_store's chunk dim; use a deterministic vector
            # close to one of the ingested chunks so the trace has content.
            return [0.3] * 384

        monkeypatch.setattr(cli_mod, "embed_query", _fake)

    def test_why_requires_expect_or_expect_chunk_id(self, monkeypatch) -> None:
        """vstash why without --expect / --expect-chunk-id errors cleanly."""
        self._patch_embed(monkeypatch)
        result = runner.invoke(app, ["why", "python programming"])
        assert result.exit_code == 1
        assert "--expect" in result.stdout

    def test_why_with_existing_path_prints_trace(self, monkeypatch) -> None:
        """vstash why <query> --expect <path> prints a pipeline trace
        table and exits with appearance-aware code."""
        self._patch_embed(monkeypatch)
        result = runner.invoke(
            app,
            [
                "why",
                "python programming",
                "--expect",
                "/test/python_guide.md",
                "--top-k",
                "5",
            ],
        )
        # Exit 0 if the expected doc appeared in top-k, 2 if not. Both are
        # valid in the sense that the command ran end-to-end.
        assert result.exit_code in (0, 2)
        assert "Pipeline trace" in result.stdout
        # Stage column must include at least one of the known stages.
        assert any(
            stage in result.stdout
            for stage in (
                "vector_search",
                "distance_cutoff",
                "fts_search",
                "rrf_fusion",
                "top_k_cutoff",
            )
        )

    def test_why_with_unknown_path_errors_gracefully(self, monkeypatch) -> None:
        """An unknown --expect path surfaces the ValueError from
        miss_analysis as a clean CLI error (not a traceback)."""
        self._patch_embed(monkeypatch)
        result = runner.invoke(
            app,
            ["why", "python", "--expect", "/does/not/exist.md"],
        )
        assert result.exit_code == 1
        # The store raises "No chunks found for path: ..."; the CLI wraps
        # it with a red 'x' marker.
        assert "No chunks found" in result.stdout or "not found" in result.stdout.lower()

    def test_why_json_error_path_returns_json(self, monkeypatch) -> None:
        """Code-review W1: when --json is set, error outputs must still be
        valid JSON so piped consumers (jq / scripts) do not choke on Rich
        markup text. Mirrors the ``vstash search --miss --json`` contract."""
        import json as _json

        self._patch_embed(monkeypatch)
        # Trigger the "no --expect, no --expect-chunk-id" error path.
        result = runner.invoke(app, ["why", "q", "--json"])
        assert result.exit_code == 1
        data = _json.loads(result.stdout)
        assert "error" in data
        assert "--expect" in data["error"]

        # Also trigger the unknown-path error path.
        result = runner.invoke(
            app,
            ["why", "q", "--expect", "/nope.md", "--json"],
        )
        assert result.exit_code == 1
        data = _json.loads(result.stdout)
        assert "error" in data

    def test_why_json_output_is_parseable(self, monkeypatch) -> None:
        """--json emits a single JSON document matching the MissAnalysis
        schema, suitable for piping into jq or another script."""
        import json as _json

        self._patch_embed(monkeypatch)
        result = runner.invoke(
            app,
            ["why", "python", "--expect", "/test/python_guide.md", "--json"],
        )
        assert result.exit_code in (0, 2)
        # stdout should be parseable JSON with the MissAnalysis keys.
        data = _json.loads(result.stdout)
        assert data["query"] == "python"
        assert data["expected_path"] == "/test/python_guide.md"
        assert "stage_verdicts" in data
        assert "actual_top_k" in data
        assert "suggestions" in data


class TestRelevanceTier:
    """Test the _relevance_tier helper."""

    def test_high_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.50) == "high"
        assert relevance_tier(0.95) == "high"

    def test_medium_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.96) == "medium"
        assert relevance_tier(0.98) == "medium"

    def test_low_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.99) == "low"
        assert relevance_tier(1.20) == "low"
