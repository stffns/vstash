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
        lambda cfg=None, warm=False: (load_config(), populated_store),
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
            lambda cfg=None, warm=False: (load_config(), empty_store),
        )

        result = runner.invoke(app, ["ask", "What is Python?"])
        assert "No relevant documents found" in result.stdout
        empty_store.close()


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
