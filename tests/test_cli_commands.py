"""Tests for vstash CLI commands — list, stats, forget, config, add, ask."""

from __future__ import annotations

import pytest

from tests._cli_helpers import patch_cli_attr
from typer.testing import CliRunner

from tests.conftest import requires_sqlite_vec
from vstash.cli import app
from vstash.store import VstashStore

pytestmark = requires_sqlite_vec

runner = CliRunner()


@pytest.fixture(autouse=True)
def _patch_store(populated_store: VstashStore, monkeypatch) -> None:
    """Monkeypatch _get_store so CLI uses the test store."""
    from vstash.config import load_config

    patch_cli_attr(
        monkeypatch,
        "_get_store",
        lambda cfg=None, warm=False, profile=None: (load_config(), populated_store),
    )


class TestSearchValidation:
    """`vstash search` validates at the boundary (parity with SDK/MCP/web)."""

    def test_search_rejects_pathological_top_k(self) -> None:
        result = runner.invoke(app, ["search", "python", "--top-k", "999999"])
        # Clean LimitError (exit 2 + a readable message), not a deep SQLite crash.
        assert result.exit_code == 2
        assert "top_k" in result.output and "limit" in result.output.lower()

    def test_search_accepts_valid_top_k(self) -> None:
        result = runner.invoke(app, ["search", "python", "--top-k", "3"])
        assert result.exit_code == 0


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
        from vstash.config import load_config

        cfg = load_config()
        from vstash.embed import get_embedding_dim

        dim = get_embedding_dim(cfg.embeddings.model)
        empty_store = VstashStore(tmp_db_path + "_empty", embedding_dim=dim)

        patch_cli_attr(
            monkeypatch,
            "_get_store",
            lambda cfg=None, warm=False, profile=None: (load_config(), empty_store),
        )

        result = runner.invoke(app, ["ask", "What is Python?"])
        assert "No relevant documents found" in result.stdout
        empty_store.close()


class TestAutoMissHintHook:
    """#157 part 3 (2026-04-21): ``vstash ask`` and ``vstash search``
    auto-log a miss_hint to search_events when result_count=0 or tier=low
    and ``[observability] auto_miss_hint`` is enabled (default)."""

    def _patch_embed(self, monkeypatch, dim: int) -> None:

        def _fake(query: str, model: str) -> list[float]:
            # A vector that does NOT match any populated_store chunk so
            # the search returns empty / low-tier.
            return [0.99] * dim

        patch_cli_attr(monkeypatch, "embed_query", _fake)

    def test_record_search_event_persists_hint(self, populated_store: VstashStore) -> None:
        """Unit-level check: record_search_event + recent_miss_hints
        roundtrip a dict through the miss_hint column without data loss."""
        populated_store.record_search_event(
            query="unmatchable_xyz",
            best_distance=1.0,
            relevance_tier="low",
            result_count=0,
            miss_hint={
                "reason": "empty",
                "tier": "low",
                "best_distance": 1.0,
                "result_count": 0,
                "top_k_requested": 5,
            },
        )
        hints = populated_store.recent_miss_hints(limit=5)
        assert any(h["query"] == "unmatchable_xyz" for h in hints)
        match = next(h for h in hints if h["query"] == "unmatchable_xyz")
        assert match["miss_hint"]["reason"] == "empty"
        assert match["miss_hint"]["tier"] == "low"

    def test_record_search_event_none_hint_stays_null(self, populated_store: VstashStore) -> None:
        """When the caller passes miss_hint=None (e.g., auto_miss_hint
        is disabled) the row is written but recent_miss_hints skips it."""
        start_ids = {h["id"] for h in populated_store.recent_miss_hints(limit=50)}
        populated_store.record_search_event(
            query="no_hint_query",
            best_distance=0.5,
            relevance_tier="high",
            result_count=3,
            miss_hint=None,
        )
        end_hints = populated_store.recent_miss_hints(limit=50)
        new = [h for h in end_hints if h["id"] not in start_ids]
        assert new == [], "recent_miss_hints should skip rows with NULL miss_hint"

    def test_build_miss_hint_helper(self) -> None:
        """``_build_miss_hint`` returns None when auto_hint is off OR
        when tier is high with results; returns a structured dict for
        empty / all-low cases."""
        from vstash.cli._app import _build_miss_hint

        # Disabled -> None
        assert (
            _build_miss_hint(
                cfg_auto_hint=False,
                result_count=0,
                tier="low",
                best_distance=1.0,
                top_k_requested=5,
            )
            is None
        )
        # High tier with results -> None
        assert (
            _build_miss_hint(
                cfg_auto_hint=True,
                result_count=3,
                tier="high",
                best_distance=0.2,
                top_k_requested=5,
            )
            is None
        )
        # Empty -> populated
        hint = _build_miss_hint(
            cfg_auto_hint=True,
            result_count=0,
            tier="low",
            best_distance=1.0,
            top_k_requested=5,
        )
        assert hint is not None
        assert hint["reason"] == "empty"
        assert hint["top_k_requested"] == 5
        # All-low -> populated
        hint2 = _build_miss_hint(
            cfg_auto_hint=True,
            result_count=3,
            tier="low",
            best_distance=0.99,
            top_k_requested=5,
        )
        assert hint2 is not None
        assert hint2["reason"] == "all_low"


class TestWhyCommand:
    """Test 'vstash why' command (issue #157 / H-WHY)."""

    def _patch_embed(self, monkeypatch, populated_store: VstashStore) -> None:
        """Stub embed_query so tests do not download a real model.
        Dim is derived from the populated_store fixture instead of
        hardcoded, so the test remains stable if the default embedding
        model changes dim."""

        dim = populated_store.embedding_dim

        def _fake(query: str, model: str) -> list[float]:
            return [0.3] * dim

        patch_cli_attr(monkeypatch, "embed_query", _fake)

    def test_why_requires_expect_or_expect_chunk_id(
        self, monkeypatch, populated_store: VstashStore
    ) -> None:
        """vstash why without --expect / --expect-chunk-id errors cleanly."""
        self._patch_embed(monkeypatch, populated_store)
        result = runner.invoke(app, ["why", "python programming"])
        assert result.exit_code == 1
        assert "--expect" in result.stdout

    def test_why_with_existing_path_prints_trace(
        self, monkeypatch, populated_store: VstashStore
    ) -> None:
        """vstash why <query> --expect <path> prints a pipeline trace
        table and exits with appearance-aware code."""
        self._patch_embed(monkeypatch, populated_store)
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

    def test_why_with_unknown_path_errors_gracefully(
        self, monkeypatch, populated_store: VstashStore
    ) -> None:
        """An unknown --expect path surfaces the ValueError from
        miss_analysis as a clean CLI error (not a traceback)."""
        self._patch_embed(monkeypatch, populated_store)
        result = runner.invoke(
            app,
            ["why", "python", "--expect", "/does/not/exist.md"],
        )
        assert result.exit_code == 1
        # The store raises "No chunks found for path: ..."; the CLI wraps
        # it with a red 'x' marker.
        assert "No chunks found" in result.stdout or "not found" in result.stdout.lower()

    def test_why_json_error_path_returns_json(
        self, monkeypatch, populated_store: VstashStore
    ) -> None:
        """Code-review W1: when --json is set, error outputs must still be
        valid JSON so piped consumers (jq / scripts) do not choke on Rich
        markup text. Mirrors the ``vstash search --miss --json`` contract."""
        import json as _json

        self._patch_embed(monkeypatch, populated_store)
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

    def test_why_recent_rejects_negative(self, monkeypatch, populated_store: VstashStore) -> None:
        """Code-review: ``--recent`` must validate the int. Negative
        values previously would slip through (and SQLite's ``LIMIT -1``
        would silently return all rows). Now raises a clean error."""
        self._patch_embed(monkeypatch, populated_store)
        result = runner.invoke(app, ["why", "--recent", "-3"])
        assert result.exit_code == 1
        assert "--recent" in result.stdout

    def test_recent_miss_hints_rejects_negative_limit(self, populated_store: VstashStore) -> None:
        """``VstashStore.recent_miss_hints`` clamps against ``LIMIT -1``."""
        with pytest.raises(ValueError, match="limit must be"):
            populated_store.recent_miss_hints(limit=-1)
        with pytest.raises(ValueError, match="limit must be"):
            populated_store.recent_miss_hints(limit=0)

    def test_why_recent_lists_logged_hints(self, monkeypatch, populated_store: VstashStore) -> None:
        """#157 part 3 (2026-04-21): ``vstash why --recent N`` lists
        the N most recent search_events whose auto-logged miss_hint is
        populated. Seed one synthetic event and assert it surfaces."""
        # Seed a miss_hint row directly via the store API so the test
        # does not depend on the search pipeline firing auto-log.
        populated_store.record_search_event(
            query="unicorn discovery",
            best_distance=0.99,
            relevance_tier="low",
            result_count=0,
            miss_hint={
                "reason": "empty",
                "tier": "low",
                "best_distance": 0.99,
                "result_count": 0,
                "top_k_requested": 5,
            },
        )

        result = runner.invoke(app, ["why", "--recent", "5"])
        assert result.exit_code == 0
        assert "unicorn discovery" in result.stdout
        # Should surface the reason and tier.
        assert "empty" in result.stdout
        assert "low" in result.stdout

    def test_why_recent_json_output(self, monkeypatch, populated_store: VstashStore) -> None:
        """``vstash why --recent N --json`` emits a parseable JSON object."""
        import json as _json

        populated_store.record_search_event(
            query="another miss",
            best_distance=0.995,
            relevance_tier="low",
            result_count=0,
            miss_hint={
                "reason": "empty",
                "tier": "low",
                "best_distance": 0.995,
                "result_count": 0,
                "top_k_requested": 5,
            },
        )

        result = runner.invoke(app, ["why", "--recent", "3", "--json"])
        assert result.exit_code == 0
        data = _json.loads(result.stdout)
        assert "recent_miss_hints" in data
        hints = data["recent_miss_hints"]
        assert any(h["query"] == "another miss" for h in hints)
        # Each hint row has the structured blob we expect.
        for h in hints:
            assert "miss_hint" in h
            assert "reason" in h["miss_hint"]

    def test_why_recent_empty_is_friendly(self, monkeypatch, tmp_path, sample_config) -> None:
        """With no hints recorded, --recent prints a friendly message,
        not an error. Uses the pytest ``monkeypatch`` fixture directly
        so the teardown runs automatically."""
        from vstash.config import load_config as _lc
        from vstash.embed import get_embedding_dim as _gdim

        dim = _gdim(sample_config.embeddings.model)
        fresh = VstashStore(str(tmp_path / "fresh.db"), embedding_dim=dim)
        try:
            patch_cli_attr(
                monkeypatch,
                "_get_store",
                lambda cfg=None, warm=False, profile=None: (_lc(), fresh),
            )
            result = runner.invoke(app, ["why", "--recent", "5"])
            assert result.exit_code == 0
            assert "No recent miss_hints" in result.stdout
        finally:
            fresh.close()

    def test_why_json_output_is_parseable(self, monkeypatch, populated_store: VstashStore) -> None:
        """--json emits a single JSON document matching the MissAnalysis
        schema, suitable for piping into jq or another script."""
        import json as _json

        self._patch_embed(monkeypatch, populated_store)
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
    """Test the ``relevance_tier`` helper.

    Thresholds were recalibrated to cosine space in schema v2 (#272):
    ``high <= 0.4513`` (legacy L2 0.95 squared / 2) and
    ``medium <= 0.4802`` (legacy L2 0.98 squared / 2).  These values
    preserve tier assignments for BGE-small unit-normalized embeddings
    under the new cosine metric.
    """

    def test_high_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.0) == "high"
        assert relevance_tier(0.30) == "high"
        assert relevance_tier(0.4513) == "high"

    def test_medium_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.46) == "medium"
        assert relevance_tier(0.4802) == "medium"

    def test_low_relevance(self) -> None:
        from vstash.store import relevance_tier

        assert relevance_tier(0.49) == "low"
        assert relevance_tier(1.20) == "low"
        assert relevance_tier(2.0) == "low"
