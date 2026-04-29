"""Tests for ``open_store_for_config`` and the surfaces that use it.

Issue #297 closed: every public surface (CLI, MCP, web, SDK, journal,
federated search) used to duplicate the ``StorageConfig -> VstashStore``
wiring, and several silently dropped the IVFPQ tuning fields. Adding a
new ``StorageConfig`` knob meant touching 7+ files; missing one meant a
config that worked under the CLI did nothing under the SDK.

These tests:

1. Lock in that ``open_store_for_config`` forwards every storage knob
   plus observability/limits/cache.
2. Lock in that each surface routes through it (so a future knob added
   only to the helper reaches every entry point automatically).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vstash._store_open import open_store_for_config
from vstash.config import (
    CacheConfig,
    LimitsConfig,
    ObservabilityConfig,
    StorageConfig,
    VstashConfig,
)


# ------------------------------------------------------------------ #
# Helper construction                                                  #
# ------------------------------------------------------------------ #


def _ivfpq_config(db_path: str) -> VstashConfig:
    """Build a config with non-default IVFPQ + obs/limits/cache values.

    M=24 divides 384 (BGE dim), so the IVFPQ backend would actually
    accept this if a caller fit it -- but we only construct the store
    with vector_backend=sqlite-vec, so the IVFPQ knobs are stored on
    the instance without triggering a fit. That's the point: the
    fields must be wired all the way through even when the backend
    doesn't consume them.
    """
    return VstashConfig(
        storage=StorageConfig(
            db_path=db_path,
            vector_backend="sqlite-vec",
            snapvec_bits=3,
            ivfpq_nlist=16,
            ivfpq_M=24,
            ivfpq_K=128,
            ivfpq_rerank_candidates=32,
            ivfpq_nprobe=4,
        ),
        observability=ObservabilityConfig(slow_query_ms=42.0, auto_miss_hint=False),
        limits=LimitsConfig(max_top_k=42),
        cache=CacheConfig(query_cache_size=128),
    )


# ------------------------------------------------------------------ #
# 1. Helper forwards every field                                       #
# ------------------------------------------------------------------ #


class TestOpenStoreForConfig:
    def test_forwards_every_storage_knob(self, tmp_path: Path) -> None:
        cfg = _ivfpq_config(str(tmp_path / "store.db"))
        store = open_store_for_config(cfg)
        try:
            # Storage / IVFPQ
            assert store._vector_backend == "sqlite-vec"
            assert store._snapvec_bits == 3
            assert store._ivfpq_nlist == 16
            assert store._ivfpq_M == 24
            assert store._ivfpq_K == 128
            assert store._ivfpq_rerank_candidates == 32
            assert store._ivfpq_nprobe == 4
            # Observability / limits / cache
            assert store._observability.slow_query_ms == 42.0
            assert store._observability.auto_miss_hint is False
            assert store._limits.max_top_k == 42
            assert store._cache_config.query_cache_size == 128
        finally:
            store.close()

    def test_db_path_override_wins_over_resolve_chain(self, tmp_path: Path) -> None:
        cfg_db = tmp_path / "from_cfg.db"
        explicit = tmp_path / "explicit.db"
        cfg = _ivfpq_config(str(cfg_db))
        store = open_store_for_config(cfg, db_path=str(explicit))
        try:
            assert Path(store.db_path) == explicit
        finally:
            store.close()

    def test_embedding_dim_override(self, tmp_path: Path) -> None:
        """Reindex passes the *old* model's dim while migrating to a new one."""
        cfg = _ivfpq_config(str(tmp_path / "dim.db"))
        store = open_store_for_config(cfg, embedding_dim=128)
        try:
            assert store.embedding_dim == 128
        finally:
            store.close()


# ------------------------------------------------------------------ #
# 2. Every surface routes through the helper                          #
# ------------------------------------------------------------------ #


class TestSurfacesRouteThroughHelper:
    """Each surface must call ``open_store_for_config`` exactly once.

    Using the helper as a sentinel proves the surface has not gone back
    to bespoke ``VstashStore(...)`` wiring with a partial kwargs list.
    """

    def test_cli_get_store_uses_helper(self, monkeypatch, tmp_path: Path) -> None:
        import vstash.cli as cli_mod

        cfg = _ivfpq_config(str(tmp_path / "cli.db"))
        monkeypatch.setattr(cli_mod, "load_config", lambda: cfg)
        monkeypatch.setenv("VSTASH_DB_PATH", str(tmp_path / "cli.db"))
        monkeypatch.setattr(cli_mod.atexit, "register", lambda _cb: None)

        with patch("vstash.cli.open_store_for_config", wraps=open_store_for_config) as spy:
            _cfg, store = cli_mod._get_store()
            try:
                assert spy.call_count == 1
                assert spy.call_args.args[0] is cfg
            finally:
                store.close()

    def test_memory_uses_helper(self, tmp_path: Path) -> None:
        from vstash.memory import Memory

        with patch("vstash.memory.open_store_for_config", wraps=open_store_for_config) as spy:
            mem = Memory(db=str(tmp_path / "sdk.db"))
            try:
                assert spy.call_count == 1
                assert spy.call_args.kwargs["db_path"] == str(tmp_path / "sdk.db")
            finally:
                mem.close()

    def test_journal_uses_helper(self, monkeypatch, tmp_path: Path) -> None:
        import vstash.journal as journal_mod

        monkeypatch.setattr(journal_mod, "PROFILES_DIR", tmp_path / "profiles")
        cfg = _ivfpq_config(str(tmp_path / "journal_cfg.db"))

        with patch("vstash.journal.open_store_for_config", wraps=open_store_for_config) as spy:
            _cfg, store = journal_mod._get_journal_store(cfg)
            try:
                assert spy.call_count == 1
                # journal forces its own per-profile path, not the cfg-resolved one.
                journal_db = tmp_path / "profiles" / journal_mod.JOURNAL_PROFILE / "memory.db"
                assert spy.call_args.kwargs["db_path"] == str(journal_db)
            finally:
                store.close()

    def test_mcp_uses_helper(self, monkeypatch, tmp_path: Path) -> None:
        import vstash.mcp as mcp_mod

        cfg = _ivfpq_config(str(tmp_path / "mcp.db"))
        monkeypatch.setattr(mcp_mod, "_get_config", lambda: cfg)
        # Clear the lazy singleton so the test exercises a fresh open.
        monkeypatch.setattr(mcp_mod, "_store", None)
        monkeypatch.setattr(mcp_mod.atexit, "register", lambda _cb: None)

        with patch("vstash.mcp.open_store_for_config", wraps=open_store_for_config) as spy:
            store = mcp_mod._get_store()
            try:
                assert spy.call_count == 1
                assert spy.call_args.args[0] is cfg
            finally:
                store.close()
        # Reset the singleton so other tests don't pick up our cfg.
        mcp_mod._store = None

    def test_web_uses_helper(self, monkeypatch, tmp_path: Path) -> None:
        pytest.importorskip("starlette")
        import vstash.web as web_mod

        cfg = _ivfpq_config(str(tmp_path / "web.db"))
        monkeypatch.setattr(web_mod, "_get_config", lambda: cfg)
        monkeypatch.setattr(web_mod, "_store", None)

        with patch("vstash.web.open_store_for_config", wraps=open_store_for_config) as spy:
            store = web_mod._get_store()
            try:
                assert spy.call_count == 1
                assert spy.call_args.args[0] is cfg
            finally:
                store.close()
        web_mod._store = None

    def test_federated_search_uses_helper_per_profile(self, monkeypatch, tmp_path: Path) -> None:
        """Every per-profile store opened by federated_search must go through
        the helper -- otherwise IVFPQ tuning silently drops on the federated
        path even when it works on the primary store. Regression for #297.
        """
        from vstash.profile import federated_search
        from vstash.store import VstashStore

        default_db = tmp_path / "memory.db"
        profiles_dir = tmp_path / "profiles"
        monkeypatch.setattr("vstash.profile.DEFAULT_DB", default_db)
        monkeypatch.setattr("vstash.profile.PROFILES_DIR", profiles_dir)
        monkeypatch.setattr("vstash.profile._find_local_db", lambda: None)

        # Two profiles with content so federated_search has something to open.
        for db in [default_db, profiles_dir / "named" / "memory.db"]:
            db.parent.mkdir(parents=True, exist_ok=True)
            s = VstashStore(str(db), embedding_dim=384)
            s.add_document(
                path=f"/{db.parent.name}.md",
                title="t",
                chunks=["body"],
                embeddings=[[0.1] * 384],
                source_type="markdown",
            )
            s.close()

        cfg = _ivfpq_config(str(default_db))

        with patch("vstash._store_open.open_store_for_config", wraps=open_store_for_config) as spy:
            results = federated_search(
                query_embedding=[0.1] * 384,
                query_text="body",
                cfg=cfg,
            )
            assert results, "federated_search should find at least one result"
            # One call per profile -- both default and named.
            assert spy.call_count == 2
            for call in spy.call_args_list:
                assert call.args[0] is cfg
