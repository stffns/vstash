"""Regression tests for _get_store lifecycle (issue #268).

The CLI's ``_get_store`` constructs a ``VstashStore`` but delegates
cleanup to the interpreter shutdown. Before #268 no cleanup ran at all,
so snapvec stores were left with ``_snap_dirty=True`` and the next open
paid a full ``_rebuild_snapvec_from_vec_chunks``. These tests lock in
that the helper registers ``store.close`` via ``atexit`` and that the
callback flushes the deferred ``.snpv``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vstash.cli as cli_mod
from vstash.config import VstashConfig
from vstash.store import VstashStore


DIM = 32


def _hermetic_config_factory(db_path, *, vector_backend: str = "sqlite-vec"):
    """Build a load_config() replacement that ignores ~/.vstash/vstash.toml.

    Keeping these tests hermetic matters for two reasons:
    - They should never open or modify a developer's real ~/.vstash DB.
    - They must not trigger an ONNX/Sentence-Transformers download just
      because the developer's vstash.toml points at a non-default model.
    """

    def _factory() -> VstashConfig:
        base = VstashConfig()
        storage = base.storage.model_copy(
            update={
                "db_path": str(db_path),
                "vector_backend": vector_backend,
                "snapvec_bits": 4,
            }
        )
        return base.model_copy(update={"storage": storage})

    return _factory


def _patch_hermetic_config(monkeypatch, db_path, *, vector_backend: str = "sqlite-vec"):
    """Apply all monkeypatches needed for a hermetic _get_store call."""
    monkeypatch.setenv("VSTASH_DB_PATH", str(db_path))
    monkeypatch.setattr(
        cli_mod, "load_config", _hermetic_config_factory(db_path, vector_backend=vector_backend)
    )
    monkeypatch.setattr(cli_mod, "get_embedding_dim", lambda _model: DIM)


class TestGetStoreAtexit:
    """``_get_store`` must register ``store.close`` for shutdown flush."""

    def test_registers_close_via_atexit(self, tmp_path, monkeypatch):
        """The atexit callback is ``store.close`` (bound method)."""
        registered: list = []
        monkeypatch.setattr(cli_mod.atexit, "register", registered.append)

        db = tmp_path / "lifecycle.db"
        _patch_hermetic_config(monkeypatch, db)

        _cfg, store = cli_mod._get_store()
        try:
            assert store.close in registered, (
                "_get_store must register store.close via atexit so "
                "snapvec-backed DBs flush .snpv on CLI exit (see issue #268)"
            )
        finally:
            store.close()


class TestGetStoreFlushesSnapvec:
    """End-to-end: atexit callback flushes snapvec, next open skips rebuild."""

    def test_atexit_flush_prevents_rebuild_on_reopen(self, tmp_path, monkeypatch):
        pytest.importorskip("snapvec", reason="snapvec not installed")

        db = tmp_path / "cli_snap.db"

        # Capture what _get_store would register with atexit, but
        # don't actually register (we fire it manually below to simulate
        # process exit without relying on interpreter shutdown).
        registered: list = []
        monkeypatch.setattr(cli_mod.atexit, "register", registered.append)
        _patch_hermetic_config(monkeypatch, db, vector_backend="snapvec")

        _cfg, store = cli_mod._get_store()

        # Write path: add_document sets _snap_dirty=True and defers the
        # .snpv flush to close() / _checkpoint_snapvec.
        rng = np.random.default_rng(0)
        for i in range(8):
            store.add_document(
                path=f"/x/doc{i}.md",
                title=f"doc{i}",
                chunks=[f"chunk {i}"],
                embeddings=[rng.standard_normal(DIM).tolist()],
            )
        assert store._snap_dirty is True

        # Fire every callback _get_store registered (simulate exit).
        assert registered, "_get_store registered no atexit callbacks"
        for cb in registered:
            cb()
        assert store._snap_dirty is False

        # Reopen and verify no rebuild was triggered. We count calls to
        # _rebuild_snapvec_from_vec_chunks on the reopened handle's class
        # via a spy. _init_snapvec only calls rebuild when the .snpv on
        # disk is stale relative to vec_chunks; a clean shutdown must
        # leave them in sync.
        rebuild_calls = 0
        original_rebuild = VstashStore._rebuild_snapvec_from_vec_chunks

        def _spy(self, *args, **kwargs):
            nonlocal rebuild_calls
            rebuild_calls += 1
            return original_rebuild(self, *args, **kwargs)

        monkeypatch.setattr(VstashStore, "_rebuild_snapvec_from_vec_chunks", _spy)

        reopened = VstashStore(
            str(db),
            embedding_dim=DIM,
            vector_backend="snapvec",
            snapvec_bits=4,
        )
        try:
            assert rebuild_calls == 0, (
                "atexit flush must leave .snpv consistent with vec_chunks; "
                f"reopen triggered {rebuild_calls} rebuild(s) (see issue #268)"
            )
            assert len(reopened._snap) == 8
        finally:
            reopened.close()
