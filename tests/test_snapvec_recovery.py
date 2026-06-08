"""Snapvec durability: rollback recovery rebuilds from vec_chunks, and reindex
flushes the rebuilt index immediately (PR: fix/snapvec-rollback-recovery).

Flat snapvec defers its ``.snpv`` write to close()/checkpoint, so the on-disk
file can lag ``vec_chunks`` mid-session. Both tests use the real snapvec backend
and fail without the fix.
"""

from __future__ import annotations

import pytest

from vstash.store import VstashStore

DIM = 4


def _add(store: VstashStore, i: int) -> None:
    store.add_document(
        path=f"/d{i}.md",
        title=f"d{i}",
        chunks=[f"chunk {i}"],
        embeddings=[[0.1 * (i + 1), 0.2, 0.3, 0.4]],
    )


def test_reload_snapvec_rebuilds_from_vec_chunks_when_disk_is_stale(tmp_db_path: str) -> None:
    """After a rollback, _reload_snapvec must not restore an out-of-date .snpv
    that is missing committed-but-unflushed adds — it rebuilds from vec_chunks."""
    pytest.importorskip("snapvec")
    store = VstashStore(tmp_db_path, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
    try:
        # Flush a 2-vector .snpv to disk.
        _add(store, 0)
        _add(store, 1)
        store._checkpoint_snapvec()

        # Three more committed adds, deferred (not flushed): in-memory snap has 5,
        # the on-disk .snpv still has 2.
        _add(store, 2)
        _add(store, 3)
        _add(store, 4)
        assert len(store._snap) == 5
        assert store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 5

        # Simulate the post-rollback recovery. A bare SnapIndex.load would
        # restore the stale 2-vector .snpv; the fix re-runs the staleness-aware
        # load and rebuilds to match vec_chunks.
        store._reload_snapvec()
        assert len(store._snap) == 5, (
            "reload must rebuild from vec_chunks, not restore a stale .snpv"
        )
    finally:
        store.close()


def test_reindex_checkpoints_snapvec_immediately(tmp_db_path: str) -> None:
    """reindex persists the rebuilt snapvec index immediately instead of leaving
    it dirty until close() (minimises the crash-rebuild window)."""
    pytest.importorskip("snapvec")
    store = VstashStore(tmp_db_path, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
    try:
        for i in range(3):
            _add(store, i)

        def embed(texts: list[str]) -> list[list[float]]:
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        store.reindex(embed, new_dim=DIM)
        assert store._snap_dirty is False, "reindex should checkpoint snapvec immediately"
    finally:
        store.close()
