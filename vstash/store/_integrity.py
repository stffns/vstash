"""Integrity invariants for ``vstash.store``: check and repair (#134).

Extracted as the ``_IntegrityMixin`` step of the #280 split. It is a mixin:
``VstashStore`` inherits it, and the methods rely on instance state and sibling
methods created on the concrete store (``self._conn``, ``self._snap``,
``self._bump_cache_epoch``, ``self._invalidate_idf_cache``), so the class is
never instantiated on its own.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..models import IntegrityCheck, IntegrityRepair
from ._common import _SQLITE_PARAM_BATCH

__all__ = ["_IntegrityMixin"]


class _IntegrityMixin:
    """The 5 integrity-check invariants and the profile-scoped repair pass."""

    # Attributes supplied by the host class (``VstashStore.__init__``).
    _conn: sqlite3.Connection
    _snap: Any
    _snap_dirty: bool

    def integrity_check(self) -> list[IntegrityCheck]:
        """Run a battery of database integrity checks.

        Each check is an isolated invariant — chunk count parity, FTS5
        index parity, vec_chunks parity, orphan chunks, SQLite-level
        ``PRAGMA integrity_check``.  Together they catch the corruption
        modes that survive a crashed ingest or a botched manual edit.

        Read-only.  Use :meth:`integrity_repair` for the safe-to-fix
        subset.
        """
        checks: list[IntegrityCheck] = []

        # 1. Chunk count parity: documents.chunk_count vs COUNT(chunks).
        rows = self._conn.execute(
            """
            SELECT d.id, d.path, d.chunk_count,
                   (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS actual
            FROM documents d
            WHERE d.chunk_count != (
                SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id
            )
            """
        ).fetchall()
        checks.append(
            IntegrityCheck(
                name="chunk_count_parity",
                description="documents.chunk_count matches COUNT(chunks)",
                passed=len(rows) == 0,
                affected_count=len(rows),
                detail=(
                    ""
                    if not rows
                    else "; ".join(f"{r[1]} (declared={r[2]}, actual={r[3]})" for r in rows[:3])
                ),
                repairable=True,
            )
        )

        # 2. Vec index parity (sqlite-vec only).  Snapvec lives outside
        # SQLite, so its parity is reported as a separate check below.
        if self._snap is None:
            missing_vec = self._conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                LEFT JOIN vec_chunks v ON v.rowid = c.id
                WHERE v.rowid IS NULL
                """
            ).fetchone()[0]
            checks.append(
                IntegrityCheck(
                    name="vec_index_parity",
                    description="every chunk has a vec_chunks row",
                    passed=missing_vec == 0,
                    affected_count=int(missing_vec),
                    detail=f"{missing_vec} chunk(s) missing from vec_chunks" if missing_vec else "",
                    repairable=False,  # vec rebuild needs the original embeddings
                )
            )
        else:
            # Snapvec parity: row count of the in-memory index vs chunks.
            chunk_total = int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            snap_total = len(self._snap)
            checks.append(
                IntegrityCheck(
                    name="snapvec_parity",
                    description="snapvec index has one entry per chunk",
                    passed=chunk_total == snap_total,
                    affected_count=abs(chunk_total - snap_total),
                    detail=f"chunks={chunk_total}, snapvec={snap_total}"
                    if chunk_total != snap_total
                    else "",
                    repairable=False,
                )
            )

        # 3. FTS5 index integrity.  ``fts_chunks`` is a content=chunks
        # virtual table, which means ``COUNT(*)`` reads from the
        # underlying chunks table — it cannot detect index drift.
        # The canonical way to check an FTS5 index is the built-in
        # ``'integrity-check'`` command, which scans the index and
        # raises ``sqlite3.DatabaseError`` if anything is amiss.
        try:
            self._conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('integrity-check')")
            checks.append(
                IntegrityCheck(
                    name="fts_index_parity",
                    description="FTS5 integrity-check passes",
                    passed=True,
                    affected_count=0,
                    detail="",
                    repairable=True,
                )
            )
        except sqlite3.DatabaseError as exc:
            checks.append(
                IntegrityCheck(
                    name="fts_index_parity",
                    description="FTS5 integrity-check passes",
                    passed=False,
                    affected_count=1,
                    detail=f"FTS5 integrity-check failed: {exc}",
                    repairable=True,
                )
            )

        # 4. Orphan chunks: rows in chunks whose doc_id has no document.
        # The FK is ON DELETE CASCADE so this should never happen, but
        # historic databases or manual edits can produce orphans.
        orphan_count = int(
            self._conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks c
                LEFT JOIN documents d ON d.id = c.doc_id
                WHERE d.id IS NULL
                """
            ).fetchone()[0]
        )
        checks.append(
            IntegrityCheck(
                name="no_orphan_chunks",
                description="every chunk's doc_id resolves to a document",
                passed=orphan_count == 0,
                affected_count=orphan_count,
                detail=f"{orphan_count} orphan chunk(s) with no parent document"
                if orphan_count
                else "",
                repairable=True,
            )
        )

        # 5. SQLite-level integrity_check.
        try:
            sqlite_result = self._conn.execute("PRAGMA integrity_check").fetchone()[0]
            sqlite_ok = sqlite_result == "ok"
            checks.append(
                IntegrityCheck(
                    name="sqlite_integrity",
                    description="SQLite PRAGMA integrity_check reports ok",
                    passed=sqlite_ok,
                    affected_count=0 if sqlite_ok else 1,
                    detail=("" if sqlite_ok else sqlite_result),
                    repairable=False,
                )
            )
        except sqlite3.DatabaseError as exc:
            checks.append(
                IntegrityCheck(
                    name="sqlite_integrity",
                    description="SQLite PRAGMA integrity_check reports ok",
                    passed=False,
                    affected_count=1,
                    detail=str(exc),
                    repairable=False,
                )
            )

        return checks

    def integrity_repair(self) -> list[IntegrityRepair]:
        """Apply the safe-to-fix subset of integrity repairs.

        For each :class:`IntegrityCheck` flagged as ``repairable``, run
        a focused fix:

        - ``chunk_count_parity``  → recompute ``documents.chunk_count``
          from ``COUNT(chunks)``.
        - ``fts_index_parity``    → ``INSERT INTO fts_chunks(fts_chunks)
          VALUES('rebuild')`` (the SQLite FTS5 rebuild incantation).
        - ``no_orphan_chunks``    → delete chunks whose ``doc_id``
          resolves to no document, plus their ``vec_chunks`` /
          ``fts_chunks`` companions.

        Repairs that need source data we no longer have (re-embedding,
        SQLite page recovery) are reported as not-repairable by
        :meth:`integrity_check` and skipped here.
        """
        repairs: list[IntegrityRepair] = []
        checks = self.integrity_check()
        check_by_name = {c.name: c for c in checks}

        with self._write_lock:
            # The fts_index_parity probe inside integrity_check() is a
            # DML statement (``INSERT INTO fts(fts) VALUES(...)``), so
            # the stdlib sqlite3 driver opens an implicit transaction
            # for it.  Commit any pending state before BEGIN IMMEDIATE
            # so the explicit transaction below isn't preempted.
            if self._conn.in_transaction:
                self._conn.commit()
            try:
                self._conn.execute("BEGIN IMMEDIATE")

                # 1. Recompute documents.chunk_count
                check = check_by_name.get("chunk_count_parity")
                if check is not None and not check.passed:
                    cursor = self._conn.execute(
                        """
                        UPDATE documents
                        SET chunk_count = (
                            SELECT COUNT(*) FROM chunks c WHERE c.doc_id = documents.id
                        )
                        WHERE chunk_count != (
                            SELECT COUNT(*) FROM chunks c WHERE c.doc_id = documents.id
                        )
                        """
                    )
                    repairs.append(
                        IntegrityRepair(
                            name="chunk_count_parity",
                            success=True,
                            affected_count=cursor.rowcount,
                            detail=f"recomputed chunk_count for {cursor.rowcount} document(s)",
                        )
                    )

                # 2. Delete orphan chunks (and their vec/fts companions
                # via the cascade trigger on chunks delete).
                check = check_by_name.get("no_orphan_chunks")
                if check is not None and not check.passed:
                    orphan_ids = [
                        row[0]
                        for row in self._conn.execute(
                            """
                            SELECT c.id
                            FROM chunks c
                            LEFT JOIN documents d ON d.id = c.doc_id
                            WHERE d.id IS NULL
                            """
                        ).fetchall()
                    ]
                    if orphan_ids:
                        # Manually clean up vec_chunks (no cascade for
                        # virtual table) before deleting chunks rows.
                        for batch_start in range(0, len(orphan_ids), _SQLITE_PARAM_BATCH):
                            batch = orphan_ids[batch_start : batch_start + _SQLITE_PARAM_BATCH]
                            placeholders = ",".join("?" * len(batch))
                            self._conn.execute(
                                f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})",
                                batch,
                            )
                            self._conn.execute(
                                f"DELETE FROM chunks WHERE id IN ({placeholders})",
                                batch,
                            )
                        # Keep the in-memory snapvec index in sync with the
                        # SQLite delete so the next snapvec_parity check passes
                        # (#381). Mirrors delete_document's snap handling; the
                        # flush is deferred to close() / _checkpoint_snapvec().
                        if self._snap is not None:
                            for cid in orphan_ids:
                                self._snap.delete(cid)
                            self._snap_dirty = True
                    repairs.append(
                        IntegrityRepair(
                            name="no_orphan_chunks",
                            success=True,
                            affected_count=len(orphan_ids),
                            detail=f"removed {len(orphan_ids)} orphan chunk(s)",
                        )
                    )

                # 3. Rebuild FTS5 index from chunks (the canonical
                # SQLite FTS5 rebuild command).
                check = check_by_name.get("fts_index_parity")
                if check is not None and not check.passed:
                    self._conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
                    repairs.append(
                        IntegrityRepair(
                            name="fts_index_parity",
                            success=True,
                            affected_count=1,
                            detail="rebuilt fts_chunks from chunks table",
                        )
                    )

                self._conn.commit()
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
            except Exception as exc:
                self._conn.rollback()
                # The orphan cleanup above may have mutated the in-memory snap
                # index before the failing statement. The SQLite rollback restored
                # those chunks, so reload the snap from disk to match — otherwise
                # the in-memory index would be missing chunks SQLite still has
                # (#381 review). No-op when self._snap is None / unmutated.
                self._reload_snapvec()
                repairs.append(
                    IntegrityRepair(
                        name="repair_transaction",
                        success=False,
                        detail=f"rolled back: {exc}",
                    )
                )

        return repairs
