"""Vector-index lifecycle for ``vstash.store``: snapvec (flat) and snapvec-ivfpq.

Extracted as the ``_IndexBackendMixin`` step of the #280 split. It is a mixin:
``VstashStore`` inherits it, and the methods rely on instance state created in
``VstashStore.__init__`` (``self._snap``, ``self._snap_dirty``,
``self._vector_backend``, the ``self._ivfpq_*`` tuning fields, ``self._conn``,
``self.db_path``, ``self.embedding_dim``, ``self._snapvec_bits``), so the class
is never instantiated on its own.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ._common import _HAS_SNAPVEC, SnapIndex

logger = logging.getLogger(__name__)


class _IndexBackendMixin:
    """snapvec / IVFPQ index construction, persistence, and reload lifecycle."""

    # Attributes supplied by the host class (``VstashStore.__init__``).
    _conn: sqlite3.Connection
    db_path: Path
    embedding_dim: int
    _snap: Any
    _snap_dirty: bool
    _vector_backend: str
    _snapvec_bits: int

    @property
    def _snapvec_path(self) -> Path:
        """Path to the flat snapvec index file (next to the .db file)."""
        return self.db_path.with_suffix(".snpv")

    @property
    def _ivfpq_path(self) -> Path:
        """Path to the IVFPQ snapvec index file (next to the .db file)."""
        return self.db_path.with_suffix(".snpi")

    def _init_ivfpq(self) -> None:
        """Load or create the IVFPQ backend.

        The backend starts unfit; sqlite-vec remains authoritative until
        ``fit_ivfpq()`` trains on the corpus. After that, searches prefer
        the IVFPQ path and new adds append directly.

        Crash recovery (issue #329): the on-disk ``.snpi`` is only flushed
        on ``close()`` / ``_checkpoint_snapvec``. A crash between an
        ``add_document`` (which updates the in-memory IVFPQ index)
        and ``close()`` can leave ``.snpi`` with fewer rows than
        ``vec_chunks``. On reopen, this method detects that staleness
        (``len(loaded_index) < COUNT(vec_chunks)``) and downgrades the
        backend to its unfitted state so ``VstashStore.search`` falls
        back to sqlite-vec rather than silently missing the rows
        ingested in the last write burst. The user is told via
        ``logger.warning`` to rerun ``vstash snapvec fit`` when
        convenient. We do not auto-refit because IVFPQ fitting is a
        ~50s operation that would block every reopen after a crash;
        sqlite-vec retrieval is correct in the interim.
        """
        if not _HAS_SNAPVEC:
            logger.warning(
                "snapvec-ivfpq backend requested but snapvec is not installed. "
                "Install with: pip install vstash[snapvec]. Falling back to sqlite-vec."
            )
            self._vector_backend = "sqlite-vec"
            return

        from ..vectorbackend.snapvec_ivfpq import IVFPQBackend

        nlist = self._ivfpq_nlist or self._derive_nlist()
        kwargs = dict(
            dim=self.embedding_dim,
            nlist=nlist,
            M=self._ivfpq_M,
            K=self._ivfpq_K,
            keep_full_precision=True,
            rerank_candidates=self._ivfpq_rerank_candidates,
            nprobe=self._ivfpq_nprobe,
        )
        try:
            self._snap = IVFPQBackend.load(str(self._ivfpq_path), **kwargs)
        except ValueError:
            # Config error (e.g. ivfpq_M does not divide embedding_dim).
            # Surface directly so the user gets the real message instead of
            # the generic "failed to load, run vstash snapvec fit".
            raise
        except Exception:
            logger.warning(
                "Failed to load IVFPQ index at %s — starting unfitted. "
                "Run 'vstash snapvec fit' to train from current vec_chunks.",
                self._ivfpq_path,
                exc_info=True,
            )
            self._snap = IVFPQBackend(**kwargs)
            return

        # Staleness check (issue #329): mirrors the FLAT precedent at
        # _init_snapvec. ANY count mismatch is stale, in either
        # direction:
        #
        # - sqlite_n > snap_n: the canonical add_document case. Crash
        #   after writing to vec_chunks but before .snpi flush; the
        #   index is missing the last write burst.
        # - sqlite_n < snap_n: the delete case. Crash after
        #   delete_document removed rows from vec_chunks but before
        #   .snpi flush; the index still carries the deleted ids,
        #   which then consume ANN candidate slots and reduce recall
        #   (search returns hits for tombstoned chunk_ids that the
        #   path map filters out, so top-K is partial).
        #
        # In both cases the right answer is to downgrade to unfitted
        # and let sqlite-vec answer until the user runs `vstash
        # snapvec fit` to rebuild. Skip when the index is unfitted
        # (len == 0 by contract) or the file did not exist (load
        # returns an unfitted instance in that case too).
        if self._snap.fitted:
            try:
                sqlite_n = int(self._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0])
            except sqlite3.Error:
                # The staleness probe itself failed (transient lock,
                # vec0 hiccup) — that says nothing about the index, so
                # do NOT downgrade a fitted index on it. A real parity
                # problem will surface via integrity_check(); a real DB
                # problem will surface loudly on first search.
                logger.warning(
                    "Could not verify IVFPQ staleness for %s (COUNT on "
                    "vec_chunks failed); keeping the fitted index.",
                    self._ivfpq_path,
                    exc_info=True,
                )
                return
            snap_n = len(self._snap)
            if sqlite_n != snap_n:
                logger.warning(
                    "IVFPQ index at %s is stale: %d routable vectors vs %d in "
                    "vec_chunks (likely a crash between add_document/delete and "
                    "close()). Downgrading to unfitted; search will fall back "
                    "to sqlite-vec until you run 'vstash snapvec fit' to "
                    "rebuild from current vec_chunks.",
                    self._ivfpq_path,
                    snap_n,
                    sqlite_n,
                )
                self._snap = IVFPQBackend(**kwargs)

    @staticmethod
    def _nlist_for(n: int) -> int:
        """FAISS rule of thumb: 4 * sqrt(N), clamped to [8, 1024]."""
        if n <= 0:
            return 256  # placeholder; real nlist is derived at fit() time
        return max(8, min(1024, int(4 * math.sqrt(n))))

    def _derive_nlist(self) -> int:
        """Pick a default nlist from the current corpus size."""
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self._nlist_for(n)

    def fit_ivfpq(self, *, training_sample: int = 50_000) -> dict[str, Any]:
        """Train the IVFPQ backend on current vec_chunks and index all rows.

        Requires ``vector_backend='snapvec-ivfpq'``. Reads every
        ``(rowid, embedding)`` out of vec_chunks, samples up to
        ``training_sample`` for fit(), calls add_batch on the full set,
        and persists the index.

        Returns a dict with stats (n_indexed, nlist, training_sample,
        build_seconds).
        """
        if self._vector_backend != "snapvec-ivfpq":
            raise RuntimeError(
                f"fit_ivfpq requires vector_backend='snapvec-ivfpq'; "
                f"current backend is {self._vector_backend!r}"
            )
        import time as _time

        import numpy as np

        from ..vectorbackend.snapvec_ivfpq import IVFPQBackend

        # Stream vec_chunks into a pre-allocated float32 matrix so peak
        # memory stays bounded at ~N * dim * 4 bytes (e.g. ~150 MB at
        # 100K x 384) instead of exploding via fetchall + list-of-lists.
        n_rows = self._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        if not n_rows:
            raise RuntimeError("vec_chunks is empty; ingest data before fit_ivfpq")
        ids = np.empty(n_rows, dtype=np.int64)
        matrix = np.empty((n_rows, self.embedding_dim), dtype=np.float32)
        cursor = self._conn.execute("SELECT rowid, embedding FROM vec_chunks")
        for i, row in enumerate(cursor):
            ids[i] = int(row["rowid"])
            matrix[i] = np.frombuffer(row["embedding"], dtype=np.float32)
        # IVFPQBackend normalizes internally, but the coarse k-means for
        # training benefits from pre-normalized input too.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms > 1e-9, norms, 1.0)

        nlist = self._ivfpq_nlist or self._nlist_for(n_rows)
        t0 = _time.perf_counter()
        self._snap = IVFPQBackend(
            dim=self.embedding_dim,
            nlist=nlist,
            M=self._ivfpq_M,
            K=self._ivfpq_K,
            keep_full_precision=True,
            rerank_candidates=self._ivfpq_rerank_candidates,
            nprobe=self._ivfpq_nprobe,
        )
        if n_rows <= training_sample:
            sample = matrix
        else:
            rng = np.random.default_rng(0)
            sel = rng.choice(n_rows, size=training_sample, replace=False)
            sample = matrix[sel]
        self._snap.fit(sample)
        self._snap.add_batch(ids.tolist(), matrix)
        self._snap.save(str(self._ivfpq_path))
        self._bump_cache_epoch()
        build_s = _time.perf_counter() - t0
        return {
            "n_indexed": int(n_rows),
            "nlist": nlist,
            "training_sample": len(sample),
            "build_seconds": round(build_s, 2),
            "path": str(self._ivfpq_path),
        }

    def _init_snapvec(self) -> None:
        """Load or create the SnapIndex for the snapvec backend.

        If the on-disk ``.snpv`` is out of sync with ``vec_chunks``
        (e.g. a process crashed between ``add_document`` and
        ``close()`` with deferred save active), rebuild the flat
        index from ``vec_chunks``, which is the source of truth.
        """
        if not _HAS_SNAPVEC:
            logger.warning(
                "snapvec backend requested but snapvec is not installed. "
                "Install with: pip install vstash[snapvec]. Falling back to sqlite-vec."
            )
            self._vector_backend = "sqlite-vec"
            return

        path = self._snapvec_path
        # ``loaded_clean`` == True when we have a SnapIndex whose on-disk
        # dim matches ``self.embedding_dim``. Only in that case is the
        # stale-vs-fresh comparison against ``vec_chunks`` meaningful. If
        # dim mismatched or the file was corrupt, we already reset to an
        # empty index and the user needs ``vstash reindex`` to rebuild
        # with the new model; shoving old-dim vectors through the flat
        # crash-recovery path would crash with a shape mismatch.
        loaded_clean = False
        if path.exists():
            try:
                self._snap = SnapIndex.load(str(path))
                if self._snap.dim != self.embedding_dim:
                    logger.warning(
                        "SnapIndex dim=%d != embedding_dim=%d. Rebuilding index.",
                        self._snap.dim,
                        self.embedding_dim,
                    )
                    self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
                    self._snap.save(str(self._snapvec_path))
                else:
                    loaded_clean = True
            except Exception:
                logger.warning(
                    "Failed to load SnapIndex from %s — creating empty. "
                    "Existing vectors lost; run 'vstash reindex' to rebuild.",
                    path,
                    exc_info=True,
                )
                self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
                self._snap.save(str(self._snapvec_path))
        else:
            self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
            # Fresh in-memory index with matching dim; safe to check
            # staleness and rebuild from vec_chunks if needed (e.g.
            # user deleted the .snpv companion but kept the .db).
            loaded_clean = True

        # Staleness check: flat snapvec defers save until close() /
        # checkpoint, so a crash mid-session can leave ``.snpv`` with
        # fewer rows than ``vec_chunks``. Detect and rebuild so users
        # do not silently lose vectors from the last write burst.
        if loaded_clean and self._vector_backend == "snapvec" and self._snap is not None:
            try:
                sqlite_n = int(self._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0])
            except sqlite3.Error:
                sqlite_n = 0
            snap_n = len(self._snap)
            if sqlite_n > snap_n:
                logger.info(
                    "SnapIndex has %d vectors but vec_chunks has %d. Rebuilding flat "
                    "snapvec index from SQLite (crash recovery).",
                    snap_n,
                    sqlite_n,
                )
                self._rebuild_snapvec_from_vec_chunks()

    def _rebuild_snapvec_from_vec_chunks(self, batch_size: int = 10_000) -> int:
        """Rebuild the flat SnapIndex from ``vec_chunks`` source-of-truth.

        Used for crash recovery when ``_init_snapvec`` detects staleness
        and as the ``fit_snapvec`` analogue to ``fit_ivfpq``. Returns the
        number of vectors indexed.

        Two O(N^2) patterns are avoided here (issue #252):

        1. ``LIMIT ? OFFSET ?`` pagination on ``vec_chunks`` rescans
           ``offset`` rows on every call. With N/batch_size pages the
           total scan cost is O(N^2/batch_size). Keyset pagination
           (``WHERE rowid > last``) is O(N) total.
        2. ``SnapIndex.add_batch`` does ``np.vstack([self._indices,
           batch_idx])`` internally (snapvec/_index.py:206), so calling
           it N/batch_size times copies a growing buffer on each call.
           Coalescing all (rowids, vectors) into a single final call
           keeps the total memcpy at O(N).

        Measured on 2026-04-22: N=100k rebuild dropped from 41.5s to
        4.0s (10.3x) after the fix (keyset + coalesce + frombuffer).
        Scaling went from super-linear (148x wall clock for 20x N) to
        near-linear (24x for 20x N). See experiments/results/
        snapvec_rebuild_scaling_{before,after}.json.
        """
        if self._snap is None:
            raise RuntimeError("snapvec backend not initialised")
        self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)

        # rowid >= 1 because chunks.id is INTEGER PRIMARY KEY AUTOINCREMENT
        # and vec_chunks.rowid is fed from chunks.id; starting at 0 matches
        # all real rows on the first page.
        last_rowid = 0
        rowid_parts: list[list[int]] = []
        vector_parts: list[np.ndarray] = []
        while True:
            rows = self._conn.execute(
                "SELECT rowid, embedding FROM vec_chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
                [last_rowid, batch_size],
            ).fetchall()
            if not rows:
                break
            rowid_parts.append([int(r["rowid"]) for r in rows])
            # np.frombuffer avoids the Python-list intermediate that
            # struct.unpack + list() pays inside ``_deserialize``. At
            # N=100k with dim=384 the fill path allocates ~15M Python
            # floats otherwise; reading the blob straight into a float32
            # array drops that to a single memcpy per row.
            vector_parts.append(
                np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            )
            last_rowid = int(rows[-1]["rowid"])

        total = sum(len(p) for p in rowid_parts)
        if total:
            all_rowids = [rid for part in rowid_parts for rid in part]
            all_vectors = (
                vector_parts[0] if len(vector_parts) == 1 else np.concatenate(vector_parts, axis=0)
            )
            # Drop the per-batch lists once coalesced so the peak RSS is
            # bounded by (concat buffer + SnapIndex internal copy) instead
            # of 3x the final index size. Matters at N >> 500k (f32 x 384
            # dim ~= 1.5 GB per copy). rowid_parts is tiny relative to
            # vectors, cleared for consistency.
            vector_parts.clear()
            rowid_parts.clear()
            self._snap.add_batch(all_rowids, all_vectors)

        # Immediately persist the rebuilt index so subsequent opens do
        # not re-run the rebuild unnecessarily.
        self._snap.save(str(self._snapvec_path))
        self._snap_dirty = False
        logger.info("Rebuilt flat snapvec index with %d vectors", total)
        return total

    def _checkpoint_snapvec(self) -> None:
        """Flush the in-memory snapvec index to disk if dirty."""
        if self._snap is None or not self._snap_dirty:
            return
        target = self._ivfpq_path if self._vector_backend == "snapvec-ivfpq" else self._snapvec_path
        self._snap.save(str(target))
        self._snap_dirty = False

    def _reload_snapvec(self) -> None:
        """Reload the snapvec index from disk after a failed transaction.

        Restores the on-disk state so the in-memory index matches SQLite
        after a rollback.
        """
        if self._snap is None:
            return
        if self._vector_backend == "snapvec-ivfpq":
            # Re-run the load path used at construction time.
            self._init_ivfpq()
            self._snap_dirty = False
            return
        # Re-run the flat construction load path, which loads the .snpv AND does
        # the staleness check (vec_chunks count > snap count -> rebuild from
        # SQLite). A bare SnapIndex.load would restore a .snpv that is missing
        # the session's committed-but-unflushed adds (flat snapvec defers its
        # save to close()/checkpoint), leaving the in-memory index behind
        # vec_chunks until the next open. Mirrors the ivfpq branch above, which
        # already re-runs _init_ivfpq() for the same reason.
        #
        # Guard it: this runs in the rollback recovery path, so an exception
        # here (e.g. disk-full while _rebuild_snapvec_from_vec_chunks saves)
        # must NOT propagate and mask the original transaction error the caller
        # is about to re-raise. Fall back to an empty index (rebuilt on next
        # open) and log, matching the pre-rewrite behaviour.
        try:
            self._init_snapvec()
        except Exception:
            logger.warning(
                "Failed to reload/rebuild SnapIndex after rollback — creating empty index. "
                "Run 'vstash reindex' to rebuild vector search.",
                exc_info=True,
            )
            self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
        self._snap_dirty = False
