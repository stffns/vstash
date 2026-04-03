"""Tests for multi-profile: resolution, management, and federated search."""

from __future__ import annotations

from pathlib import Path

import pytest

from vstash.profile import (
    PROFILES_DIR,
    active_profile_info,
    create_profile,
    delete_profile,
    federated_search,
    list_profiles,
    resolve_db_path,
    validate_name,
)


# ------------------------------------------------------------------ #
# Name validation                                                     #
# ------------------------------------------------------------------ #


class TestValidateName:
    def test_valid_names(self) -> None:
        for name in ["work", "my-project", "agent_1", "A", "research2025"]:
            assert validate_name(name) == name

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid profile name"):
            validate_name("")

    def test_rejects_path_traversal(self) -> None:
        for bad in ["../evil", "foo/bar", ".hidden"]:
            with pytest.raises(ValueError, match="Invalid profile name"):
                validate_name(bad)

    def test_rejects_spaces_and_special(self) -> None:
        for bad in ["has space", "no@sign", "semi;colon"]:
            with pytest.raises(ValueError, match="Invalid profile name"):
                validate_name(bad)

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="Invalid profile name"):
            validate_name("a" * 65)

    def test_max_length_ok(self) -> None:
        assert validate_name("a" * 64) == "a" * 64


# ------------------------------------------------------------------ #
# Resolution chain                                                    #
# ------------------------------------------------------------------ #


class TestResolveDbPath:
    def test_default_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No profile, no env, no local dir => default ~/.vstash/memory.db."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path()
        assert result == Path.home() / ".vstash" / "memory.db"

    def test_explicit_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit profile resolves to profiles dir."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path("work")
        assert result == PROFILES_DIR / "work" / "memory.db"

    def test_env_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """VSTASH_PROFILE env var resolves to profiles dir."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "research")
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path()
        assert result == PROFILES_DIR / "research" / "memory.db"

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit profile takes priority over VSTASH_PROFILE."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "env-profile")
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path("explicit")
        assert result == PROFILES_DIR / "explicit" / "memory.db"

    def test_vstash_db_path_overrides_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VSTASH_DB_PATH env var takes highest priority."""
        custom = tmp_path / "custom.db"
        monkeypatch.setenv("VSTASH_DB_PATH", str(custom))
        monkeypatch.setenv("VSTASH_PROFILE", "should-not-use")
        result = resolve_db_path("also-ignored")
        assert result == custom.resolve()

    def test_local_vstash_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Project-local .vstash/memory.db is discovered."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        local_db = tmp_path / ".vstash" / "memory.db"
        local_db.parent.mkdir()
        local_db.touch()
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path()
        assert result == local_db

    def test_local_vstash_walks_parents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Walking upward finds .vstash/memory.db in parent dir."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        local_db = tmp_path / ".vstash" / "memory.db"
        local_db.parent.mkdir()
        local_db.touch()
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        result = resolve_db_path()
        assert result == local_db

    def test_env_overrides_local_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """VSTASH_PROFILE takes priority over local .vstash/ dir."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "from-env")
        local_db = tmp_path / ".vstash" / "memory.db"
        local_db.parent.mkdir()
        local_db.touch()
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path()
        assert result == PROFILES_DIR / "from-env" / "memory.db"

    def test_config_db_path_overrides_profile_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Custom storage.db_path in toml overrides VSTASH_PROFILE and local dir."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "should-not-use")
        monkeypatch.chdir(tmp_path)
        custom = str(tmp_path / "custom.db")
        result = resolve_db_path(config_db_path=custom)
        assert result == Path(custom).resolve()

    def test_config_db_path_default_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Default storage.db_path is treated as if not set."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path(config_db_path="~/.vstash/memory.db")
        assert result == Path.home() / ".vstash" / "memory.db"

    def test_explicit_profile_overrides_config_db_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Explicit profile name takes priority over config db_path."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        result = resolve_db_path("work", config_db_path="/custom/db.sqlite")
        assert result == PROFILES_DIR / "work" / "memory.db"

    def test_vstash_db_path_env_overrides_config_db_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VSTASH_DB_PATH env always wins over config db_path."""
        env_path = tmp_path / "env.db"
        monkeypatch.setenv("VSTASH_DB_PATH", str(env_path))
        result = resolve_db_path(config_db_path="/custom/db.sqlite")
        assert result == env_path.resolve()


# ------------------------------------------------------------------ #
# Profile management                                                  #
# ------------------------------------------------------------------ #


class TestProfileManagement:
    def test_create_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        db_path = create_profile("work")
        assert db_path == profiles_dir / "work" / "memory.db"
        assert (profiles_dir / "work").is_dir()

    def test_create_duplicate_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        create_profile("work")
        with pytest.raises(ValueError, match="already exists"):
            create_profile("work")

    def test_delete_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        create_profile("temp")
        assert (profiles_dir / "temp").is_dir()
        delete_profile("temp")
        assert not (profiles_dir / "temp").exists()

    def test_delete_nonexistent_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        with pytest.raises(ValueError, match="does not exist"):
            delete_profile("ghost")

    def test_list_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "nope")
        assert list_profiles() == []

    def test_list_with_profiles(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        for name in ["zebra", "alpha", "middle"]:
            create_profile(name)
            # Touch memory.db so list_profiles finds it
            (profiles_dir / name / "memory.db").touch()
        assert list_profiles() == ["alpha", "middle", "zebra"]

    def test_list_skips_dirs_without_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        # Create dir without memory.db
        (profiles_dir / "empty").mkdir(parents=True)
        create_profile("valid")
        (profiles_dir / "valid" / "memory.db").touch()
        assert list_profiles() == ["valid"]


# ------------------------------------------------------------------ #
# Active profile info                                                 #
# ------------------------------------------------------------------ #


class TestActiveProfileInfo:
    def test_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        name, reason = active_profile_info()
        assert name is None
        assert "default" in reason

    def test_explicit_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        name, reason = active_profile_info("work")
        assert name == "work"
        assert "--profile" in reason

    def test_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "research")
        monkeypatch.chdir(tmp_path)
        name, reason = active_profile_info()
        assert name == "research"
        assert "env var" in reason.lower() or "VSTASH_PROFILE" in reason

    def test_db_path_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VSTASH_DB_PATH", "/custom/path.db")
        name, reason = active_profile_info("ignored")
        assert name is None
        assert "VSTASH_DB_PATH" in reason


# ------------------------------------------------------------------ #
# Federated search                                                    #
# ------------------------------------------------------------------ #


def _sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _sqlite_vec_available(), reason="sqlite-vec not available")
class TestFederatedSearch:
    def test_merges_results_from_multiple_profiles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Federated search returns results from multiple profiles."""
        from vstash.store import VstashStore

        profiles_dir = tmp_path / "profiles"
        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384

        # Create default db with a doc
        store1 = VstashStore(str(default_db), embedding_dim=dim)
        store1.add_document(
            path="/doc1.md",
            title="Doc 1",
            chunks=["machine learning basics"],
            embeddings=[[0.1] * dim],
            source_type="markdown",
        )
        store1.close()

        # Create named profile with a different doc
        p_dir = profiles_dir / "work"
        p_dir.mkdir(parents=True)
        store2 = VstashStore(str(p_dir / "memory.db"), embedding_dim=dim)
        store2.add_document(
            path="/doc2.md",
            title="Doc 2",
            chunks=["deep learning neural networks"],
            embeddings=[[0.2] * dim],
            source_type="markdown",
        )
        store2.close()

        results = federated_search(
            query_embedding=[0.15] * dim,
            query_text="machine learning",
            embedding_dim=dim,
            top_k=5,
        )

        # Should have results from both profiles
        profile_names = {r[0] for r in results}
        assert len(results) >= 2
        assert "default" in profile_names
        assert "work" in profile_names

    def test_empty_profiles_return_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No profiles => empty results."""
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "nope")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", tmp_path / "nope.db")
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        results = federated_search(
            query_embedding=[0.1] * 384,
            query_text="test",
            embedding_dim=384,
        )
        assert results == []

    def test_single_profile_works(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Federated search with only default db still works."""
        from vstash.store import VstashStore

        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "empty_profiles")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384
        store = VstashStore(str(default_db), embedding_dim=dim)
        store.add_document(
            path="/only.md",
            title="Only Doc",
            chunks=["solo content here"],
            embeddings=[[0.5] * dim],
            source_type="markdown",
        )
        store.close()

        results = federated_search(
            query_embedding=[0.5] * dim,
            query_text="solo content",
            embedding_dim=dim,
        )
        assert len(results) == 1
        assert results[0][0] == "default"
        assert results[0][1].title == "Only Doc"

    def test_rrf_fuses_identical_chunks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same doc in two profiles should be fused with boosted RRF score."""
        from vstash.store import VstashStore

        profiles_dir = tmp_path / "profiles"
        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384
        embedding = [0.3] * dim

        # Same doc in both default and named profile
        for db_path_val in [default_db, profiles_dir / "mirror" / "memory.db"]:
            db_path_val.parent.mkdir(parents=True, exist_ok=True)
            store = VstashStore(str(db_path_val), embedding_dim=dim)
            store.add_document(
                path="/shared.md",
                title="Shared",
                chunks=["shared content"],
                embeddings=[embedding],
                source_type="markdown",
            )
            store.close()

        results = federated_search(
            query_embedding=embedding,
            query_text="shared",
            embedding_dim=dim,
            top_k=5,
        )

        # Should be fused into 1 result with boosted score
        paths = [(r[1].path, r[1].chunk) for r in results]
        assert paths.count(("/shared.md", 0)) == 1
        # Fused score should be higher than single-profile (2 contributions)
        k = 60
        single_score = 1.0 / k  # rank 0 in one profile
        assert results[0][1].score > single_score

    def test_rrf_score_written_back(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Returned SearchResult.score should reflect the fused RRF score."""
        from vstash.store import VstashStore

        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "empty")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384
        store = VstashStore(str(default_db), embedding_dim=dim)
        store.add_document(
            path="/doc.md",
            title="Doc",
            chunks=["content"],
            embeddings=[[0.5] * dim],
            source_type="markdown",
        )
        store.close()

        results = federated_search(
            query_embedding=[0.5] * dim,
            query_text="content",
            embedding_dim=dim,
        )
        assert len(results) == 1
        # Score should be the RRF score, not the original store score
        k = 60
        expected = 1.0 / k  # rank 0, single profile
        assert abs(results[0][1].score - expected) < 1e-6


# ------------------------------------------------------------------ #
# Active profile info — validation                                    #
# ------------------------------------------------------------------ #


class TestActiveProfileInfoValidation:
    def test_invalid_explicit_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        name, reason = active_profile_info("../bad")
        assert name is None
        assert "invalid" in reason.lower()

    def test_invalid_env_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "no spaces")
        monkeypatch.chdir(tmp_path)
        name, reason = active_profile_info()
        assert name is None
        assert "invalid" in reason.lower()


# ------------------------------------------------------------------ #
# CLI profile subcommands                                             #
# ------------------------------------------------------------------ #


class TestProfileCLI:
    def test_profile_create_and_list(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "create", "test-work"])
        assert result.exit_code == 0
        assert "test-work" in result.stdout

        result = runner.invoke(app, ["profile", "list"])
        assert result.exit_code == 0
        assert "test-work" in result.stdout

    def test_profile_delete(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)

        runner = CliRunner()
        runner.invoke(app, ["profile", "create", "temp"])
        result = runner.invoke(app, ["profile", "delete", "temp", "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout

    def test_profile_delete_nonexistent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "empty")
        runner = CliRunner()
        result = runner.invoke(app, ["profile", "delete", "ghost", "--yes"])
        assert result.exit_code == 1

    def test_profile_create_invalid_name(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "create", "../bad"])
        assert result.exit_code == 1

    def test_invalid_profile_flag(self) -> None:
        from typer.testing import CliRunner

        from vstash.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--profile", "../evil", "stats"])
        assert result.exit_code != 0


# ------------------------------------------------------------------ #
# SDK Memory(profile=...) integration                                 #
# ------------------------------------------------------------------ #


class TestMemoryProfile:
    def test_memory_profile_uses_profile_db(self, tmp_path, monkeypatch) -> None:
        """Memory(profile='x') resolves to profiles/x/memory.db."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("vstash.profile.VSTASH_HOME", fake_home)
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", fake_home / "profiles")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", fake_home / "memory.db")
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)

        expected = fake_home / "profiles" / "work" / "memory.db"

        from vstash.profile import resolve_db_path

        result = resolve_db_path("work")
        assert result == expected
        # Verify parent dir was created
        assert expected.parent.is_dir()

    def test_memory_db_overrides_profile(self, tmp_path, monkeypatch) -> None:
        """Memory(db=...) takes precedence over profile."""
        from unittest.mock import patch

        db_file = tmp_path / "custom.db"
        db_file.touch()

        # Mock the store and config to avoid real embedding initialization
        with (
            patch("vstash.memory.get_embedding_dim", return_value=384),
            patch("vstash.memory.VstashStore") as mock_store,
            patch("vstash.memory.load_config"),
        ):
            from vstash.memory import Memory

            mem = Memory(db=str(db_file), profile="ignored")
            # The store should have been called with the db path, not the profile path
            assert mock_store.call_args[0][0] == str(db_file)
            mem.close()


# ------------------------------------------------------------------ #
# Federated search — context expansion                                #
# ------------------------------------------------------------------ #


class TestFederatedContextExpansion:
    """Test that federated search can expand context per-store."""

    def test_expand_window_enriches_chunks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """expand_window=1 returns expanded text from adjacent chunks."""
        from vstash.store import VstashStore

        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "empty")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384
        store = VstashStore(str(default_db), embedding_dim=dim)
        store.add_document(
            path="/multi.md",
            title="Multi",
            chunks=["chapter one intro", "chapter two body", "chapter three end"],
            embeddings=[[0.1] * dim, [0.5] * dim, [0.1] * dim],
            source_type="markdown",
        )
        store.close()

        # Without expansion
        results_no_expand = federated_search(
            query_embedding=[0.5] * dim,
            query_text="chapter two",
            embedding_dim=dim,
            top_k=1,
            expand_window=0,
        )
        assert len(results_no_expand) == 1
        text_no = results_no_expand[0][1].text

        # With expansion
        results_expanded = federated_search(
            query_embedding=[0.5] * dim,
            query_text="chapter two",
            embedding_dim=dim,
            top_k=1,
            expand_window=1,
        )
        assert len(results_expanded) == 1
        text_exp = results_expanded[0][1].text

        # Expanded text should be longer (includes adjacent chunks)
        assert len(text_exp) > len(text_no)
        assert "chapter one" in text_exp
        assert "chapter two" in text_exp
        assert "chapter three" in text_exp

    def test_expand_window_zero_no_expansion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """expand_window=0 returns original text (no expansion)."""
        from vstash.store import VstashStore

        default_db = tmp_path / "memory.db"
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", tmp_path / "empty")
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        dim = 384
        store = VstashStore(str(default_db), embedding_dim=dim)
        store.add_document(
            path="/doc.md",
            title="Doc",
            chunks=["before", "target", "after"],
            embeddings=[[0.1] * dim, [0.9] * dim, [0.1] * dim],
            source_type="markdown",
        )
        store.close()

        results = federated_search(
            query_embedding=[0.9] * dim,
            query_text="target",
            embedding_dim=dim,
            top_k=1,
            expand_window=0,
        )
        assert len(results) == 1
        # Should NOT contain adjacent chunks
        assert "before" not in results[0][1].text
        assert "after" not in results[0][1].text


# ------------------------------------------------------------------ #
# Integration: CLI + MCP resolve same DB path                         #
# ------------------------------------------------------------------ #


class TestDbResolutionConsistency:
    """Verify CLI, reindex, and MCP all resolve to the same DB path."""

    def test_custom_config_db_path_consistency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """All entry points resolve to the same DB when storage.db_path is set."""
        custom_db = str(tmp_path / "custom" / "memory.db")
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        expected = Path(custom_db).resolve()

        # CLI path
        cli_path = resolve_db_path(config_db_path=custom_db)
        assert cli_path == expected

        # MCP would call resolve_db_path the same way
        mcp_path = resolve_db_path(config_db_path=custom_db)
        assert mcp_path == expected

        # Reindex would call resolve_db_path with profile + config_db_path
        reindex_path = resolve_db_path(profile=None, config_db_path=custom_db)
        assert reindex_path == expected

        assert cli_path == mcp_path == reindex_path

    def test_profile_override_consistency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Explicit profile yields same path regardless of config_db_path."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        expected = PROFILES_DIR / "work" / "memory.db"

        # CLI with profile
        cli_path = resolve_db_path("work", config_db_path="/some/other.db")
        assert cli_path == expected

        # Reindex with profile
        reindex_path = resolve_db_path("work", config_db_path="/some/other.db")
        assert reindex_path == expected

    def test_env_override_consistency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VSTASH_DB_PATH env wins for all entry points."""
        env_db = tmp_path / "env.db"
        monkeypatch.setenv("VSTASH_DB_PATH", str(env_db))

        expected = env_db.resolve()

        cli_path = resolve_db_path(config_db_path="/ignored.db")
        mcp_path = resolve_db_path(config_db_path="/also_ignored.db")
        reindex_path = resolve_db_path("work", config_db_path="/still_ignored.db")

        assert cli_path == mcp_path == reindex_path == expected

    def test_default_config_falls_through_to_profile_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Default config db_path doesn't block VSTASH_PROFILE resolution."""
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.setenv("VSTASH_PROFILE", "research")
        monkeypatch.chdir(tmp_path)

        expected = PROFILES_DIR / "research" / "memory.db"

        # Default config_db_path should be ignored
        result = resolve_db_path(config_db_path="~/.vstash/memory.db")
        assert result == expected
