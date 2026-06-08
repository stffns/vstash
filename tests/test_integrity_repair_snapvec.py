"""#381: ``integrity_repair`` keeps the in-memory snapvec index in sync when it
deletes orphan chunks, so the next ``snapvec_parity`` check still passes.

Only exercises the opt-in snapvec backend (the default sqlite-vec path has no
``self._snap``), so it is skipped when snapvec is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

from pathlib import Path

from vstash.store import VstashStore

DIM = 8


def _snap_store(tmp_path: Path) -> VstashStore:
    return VstashStore(
        str(tmp_path / "snap.db"),
        embedding_dim=DIM,
        vector_backend="snapvec",
        snapvec_bits=4,
    )


def test_repair_orphan_keeps_snapvec_parity(tmp_path: Path) -> None:
    store = _snap_store(tmp_path)
    try:
        store.add_document(
            path="/doc.md", title="doc", chunks=["alpha beta"], embeddings=[[0.5] * DIM]
        )
        # Orphan the chunk: delete its document row WITHOUT cascading, leaving the
        # chunk in chunks + vec_chunks + the in-memory snapvec index.
        store._conn.execute("PRAGMA foreign_keys=OFF")
        store._conn.execute("DELETE FROM documents WHERE path = '/doc.md'")
        store._conn.execute("PRAGMA foreign_keys=ON")
        store._conn.commit()

        repairs = {r.name: r for r in store.integrity_repair()}
        assert repairs["no_orphan_chunks"].success

        # The fix: the snap index was updated alongside the SQLite delete, so the
        # parity invariant holds immediately after repair (it failed before #381).
        checks = {c.name: c for c in store.integrity_check()}
        assert checks["snapvec_parity"].passed, checks["snapvec_parity"].detail
    finally:
        store.close()
