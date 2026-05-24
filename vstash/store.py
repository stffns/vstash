"""
store.py — sqlite-vec + FTS5 hybrid store with Reciprocal Rank Fusion.

Pure vector search misses exact keyword matches (names, codes, error strings).
Pure FTS5 misses semantic similarity.
RRF combines both: score = Σ 1/(k + rank), k=60 is the standard constant.

Single .db file. WAL mode for safe concurrent reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import operator
import sqlite3
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Literal

import threading

import numpy as np
import sqlite_vec

from collections import OrderedDict

from .config import CacheConfig, LimitsConfig, ObservabilityConfig
from .errors import SchemaVersionError
from .validation import validate_document_input, validate_search_input
from .models import (
    ChunkInfo,
    DocumentInfo,
    ExplainInfo,
    IntegrityCheck,
    IntegrityRepair,
    MissAnalysis,
    MissAnalysisActualResult,
    SearchResult,
    StageVerdict,
    StoreStats,
)

# SQLite's SQLITE_LIMIT_VARIABLE_NUMBER default is 999; batch IN clauses below this.
_SQLITE_PARAM_BATCH = 900

# ------------------------------------------------------------------ #
# Schema versioning (#135)                                             #
# ------------------------------------------------------------------ #

#: Current schema version.  Bumped only when a change requires a
#: migration the runtime cannot perform automatically (column drop,
#: type change, semantics change).  Pure additive ALTER TABLE migrations
#: stay within the same version because they're handled in
#: ``_migrate_schema``.
#:
#: v2 (#272): ``vec_chunks`` uses ``distance_metric=cosine``.  Prior v1
#: DBs stored identical float bytes under sqlite-vec's default L2
#: metric; on-open migration rebuilds the virtual table (no
#: re-embedding).
SCHEMA_VERSION = "2"

#: Schema versions this build of vstash knows how to read.  Anything
#: not in this set raises :class:`SchemaVersionError` on open.  v1 is
#: accepted because on-open migration promotes it to v2 in-place.
KNOWN_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2"})


# SchemaVersionError moved to vstash.errors in v0.37 (single source of
# truth for the VstashError hierarchy). Imported at the top of this
# module; this comment is left as a breadcrumb for code archaeology.


# ------------------------------------------------------------------ #
# Miss-analysis tracing (#108)                                         #
# ------------------------------------------------------------------ #


class _PipelineTracer:
    """Caller-owned collector for per-stage verdicts during search().

    Used by miss_analysis() to record how a specific chunk fared at
    each stage of the search pipeline.  The tracer is created by the
    caller, passed into search(), and read back afterwards.  Because
    ownership is local to the caller, concurrent miss_analysis() calls
    on a shared VstashStore cannot stomp on each other.

    When tracking is not needed, search() receives ``None`` instead of
    a tracer instance — every method on the real tracer is short-
    circuited by an early ``if self.target is None: return`` check in
    the caller code, so there is zero hot-path cost.
    """

    __slots__ = ("target", "verdicts")

    def __init__(self, target_chunk_id: int) -> None:
        self.target: int = int(target_chunk_id)
        self.verdicts: list[dict[str, object]] = []

    def record(
        self,
        stage: str,
        passed: bool,
        rank: int | None = None,
        score: float | None = None,
        detail: str = "",
        counterfactual: str | None = None,
    ) -> None:
        """Append a StageVerdict-shaped dict to the caller's buffer."""
        self.verdicts.append(
            {
                "stage": stage,
                "passed": passed,
                "rank": rank,
                "score": score,
                "detail": detail,
                "counterfactual": counterfactual,
            }
        )


# vstash uses one in-memory FTS5 connection per thread for stemming
# (see VstashStore._stem_terms).  On shutdown, ``close()`` running on
# the main thread releases connections whose owner threads have
# already exited — that requires libsqlite to support cross-thread
# Connection.close(), which it does in any threadsafety mode > 0.
# Modern Python ships with serialized mode (level 3) by default; we
# fail loudly if someone is on an exotic single-threaded build so the
# breakage is obvious instead of a silent crash inside libsqlite.
if sqlite3.threadsafety == 0:  # pragma: no cover
    raise RuntimeError(
        "vstash requires sqlite3 built with threading support "
        f"(sqlite3.threadsafety > 0, got {sqlite3.threadsafety}). "
        "Most Python builds use threadsafety=3 (serialized) by default."
    )

# Probe for snapvec availability (optional dependency).
# snapvec >= 0.6.0 ships delete O(1) via swap-with-last upstream — no
# monkey-patch needed. The pin in pyproject.toml enforces the floor.
try:
    from snapvec import SnapIndex

    _HAS_SNAPVEC = True
except ImportError:
    _HAS_SNAPVEC = False

logger = logging.getLogger(__name__)


def _serialize(vector: list[float]) -> bytes:
    """Serialize a float vector into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize(data: bytes) -> list[float]:
    """Deserialize a sqlite-vec binary blob back to a float list.

    Raises:
        ValueError: If the blob length is not a multiple of float size.
    """
    item_size = struct.calcsize("f")
    if len(data) % item_size != 0:
        msg = f"Embedding blob length {len(data)} is not a multiple of {item_size}"
        raise ValueError(msg)
    count = len(data) // item_size
    return list(struct.unpack(f"{count}f", data))


# Pick the fastest pure-Python dot product available on this
# interpreter.  `math.sumprod` (Python 3.12+) is a single C-level loop
# tuned for dot products and is ~3x faster than `sum(map(operator.mul,
# a, b))` on 384-dim vectors.  For Python 3.10/3.11 (which we still
# support, per pyproject.toml requires-python = ">=3.10"), fall back to
# the map+operator path — still ~3x faster than the original generator
# expression.  The selection happens once at module load; the hot path
# pays zero overhead for the check.
try:
    _dot_product = math.sumprod  # Python 3.12+
except AttributeError:

    def _dot_product(a: list[float], b: list[float]) -> float:
        return sum(map(operator.mul, a, b))


def _cosine_sim(
    a: list[float],
    b: list[float],
    norm_a: float | None = None,
    norm_b: float | None = None,
) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1].

    Args:
        a: First vector.
        b: Second vector.
        norm_a: Precomputed L2 norm of *a* (``math.hypot(*a)``). If None,
            computed on the fly.
        norm_b: Precomputed L2 norm of *b* (``math.hypot(*b)``). If None,
            computed on the fly.

    Uses ``math.sumprod`` on Python 3.12+ and ``sum(map(operator.mul,
    ...))`` as a fallback, combined with ``math.hypot(*vec)`` for the
    L2 norm.  Both branches route through C-level stdlib loops and
    avoid the Python-bytecode overhead of generator expressions.

    Returns 0.0 when either input is an empty vector or a zero
    vector (the existing guard catches these via the ``norm < 1e-9``
    check).
    """
    dot = _dot_product(a, b)
    if norm_a is None:
        norm_a = math.hypot(*a)
    if norm_b is None:
        norm_b = math.hypot(*b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


#: High-confidence cosine distance cutoff for ``relevance_tier``.  Value
#: is the cosine equivalent of the legacy L2-on-unit-vec threshold 0.95
#: (``cos_dist = L2^2 / 2 = 0.4513``), so BGE-small unit-normalized
#: embeddings keep identical tier assignments across the v1 -> v2
#: metric change (#272).
RELEVANCE_TIER_HIGH_MAX = 0.4513

#: Medium-confidence cosine distance cutoff for ``relevance_tier``.
#: Cosine equivalent of the legacy L2 threshold 0.98
#: (``0.98^2 / 2 = 0.4802``).  Anything above is classified "low".
RELEVANCE_TIER_MEDIUM_MAX = 0.4802


def relevance_tier(distance: float) -> str:
    """Classify cosine distance into a relevance tier.

    Thresholds were recalibrated for cosine metric in schema v2 (#272).
    The old labels claimed "cosine distance" while sqlite-vec was
    actually returning L2 distance, which only worked by accident on
    unit-normalized BGE.  See the ``RELEVANCE_TIER_*`` constants above
    for how the new cutoffs were derived.

    Tiers:
        "high"   -- distance <= ``RELEVANCE_TIER_HIGH_MAX`` (0.4513):
            confident match.
        "medium" -- ``RELEVANCE_TIER_HIGH_MAX`` < distance <=
            ``RELEVANCE_TIER_MEDIUM_MAX`` (0.4802): uncertain.
        "low"    -- distance > ``RELEVANCE_TIER_MEDIUM_MAX``: likely
            off-topic.
    """
    if distance <= RELEVANCE_TIER_HIGH_MAX:
        return "high"
    if distance <= RELEVANCE_TIER_MEDIUM_MAX:
        return "medium"
    return "low"


# Standard RRF constant — balances precision vs recall
RRF_K = 60

# Adaptive RRF: query length threshold above which FTS weight is reduced.
# ArguAna (194 avg words) showed -38.4% vs dense; queries >50 words are
# typically semantic paraphrases where keywords add noise.
_ADAPTIVE_RRF_LONG_QUERY = 50

# Long-query distance_cutoff. 25.0 = 5.0^2: the squared cosine equivalent
# of the legacy v1 L2 5.0x cutoff (#272). Diffuse long-query embeddings
# compress distances; without this relaxation the default 1.3225 cutoff
# rejects nearly every candidate past rank 0.
_LONG_QUERY_DISTANCE_CUTOFF = 25.0


class VstashStore:
    """SQLite-backed vector + FTS5 hybrid store with RRF ranking.

    Supports context manager protocol for safe resource cleanup::

        with VstashStore(db_path, embedding_dim=384) as store:
            store.add_document(...)

    Args:
        db_path: Path to the SQLite database file.
        embedding_dim: Dimensionality of embedding vectors.
    """

    def __init__(
        self,
        db_path: str,
        embedding_dim: int = 384,
        vector_backend: str = "sqlite-vec",
        snapvec_bits: int = 4,
        ivfpq_nlist: int = 0,
        ivfpq_M: int = 96,
        ivfpq_K: int = 256,
        ivfpq_rerank_candidates: int = 100,
        ivfpq_nprobe: int = 0,
        observability: ObservabilityConfig | None = None,
        limits: LimitsConfig | None = None,
        cache: CacheConfig | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._write_lock = threading.Lock()
        # Per-thread state for the cosine distance of the best vector match.
        # Thread-local so concurrent searches on a shared instance don't race.
        self._thread_local = threading.local()
        self._thread_local.last_best_distance = 0.0
        # Per-thread in-memory FTS5 stemming connections.  We use a plain
        # dict keyed by thread id (instead of threading.local()) so that
        # close() can iterate and release connections from any thread —
        # otherwise stem conns created in worker threads (vstash serve,
        # MCP) would leak until process exit.
        self._stem_conns: dict[int, sqlite3.Connection] = {}
        self._stem_lock = threading.Lock()
        self._conn = self._connect()

        # --- Adaptive RRF cache ---
        self._idf_cache: tuple[dict[str, float], int] | None = None
        self._batch_depth: int = 0
        self._batch_dirty: bool = False

        # --- Deferred FTS indexing ---
        self._defer_fts: bool = False
        self._deferred_fts_rows: list[tuple[int, str]] = []

        # --- Observability ---
        # Store the whole ObservabilityConfig object (frozen Pydantic
        # model) so that future knobs can be added without touching
        # every VstashStore construction site.  Callers pass it via
        # ``observability=`` in the constructor; if omitted we use the
        # defaults, which is what single-process CLI use expects.
        self._observability: ObservabilityConfig = observability or ObservabilityConfig()

        # --- Limits / fail-safe validation (#133) ---
        # Same pattern as observability: store the whole frozen Pydantic
        # model so future knobs can be added without touching every
        # construction site.  Validators are applied at the public API
        # boundary (search, add_document) — never inside the hot path.
        self._limits: LimitsConfig = limits or LimitsConfig()

        # --- Query result cache (opt-in LRU) ---
        self._cache_config: CacheConfig = cache or CacheConfig()
        self._cache_epoch: int = 0
        self._query_cache: OrderedDict[int, list[SearchResult]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # --- SnapVec backend (optional) ---
        self._snap: Any = None
        self._vector_backend = vector_backend
        self._snapvec_bits = snapvec_bits
        self._ivfpq_nlist = ivfpq_nlist
        self._ivfpq_M = ivfpq_M
        self._ivfpq_K = ivfpq_K
        self._ivfpq_rerank_candidates = ivfpq_rerank_candidates
        self._ivfpq_nprobe = ivfpq_nprobe
        self._snap_dirty = False
        if vector_backend == "snapvec":
            self._init_snapvec()
        elif vector_backend == "snapvec-ivfpq":
            self._init_ivfpq()

    def _bump_cache_epoch(self) -> None:
        with self._cache_lock:
            self._cache_epoch += 1
            self._query_cache.clear()

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

        from .vectorbackend.snapvec_ivfpq import IVFPQBackend

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
                sqlite_n = 0
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

        from .vectorbackend.snapvec_ivfpq import IVFPQBackend

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

    def _save_snapvec(self) -> None:
        """Mark the in-memory snapvec index dirty; flush deferred to
        ``close()`` / ``_checkpoint_snapvec()``.

        Historically the flat backend wrote ``.snpv`` on every call
        (typically one per ``add_document``). That is a full-file
        rewrite and costs ~1.25 ms/MB on Apple Silicon (memory
        bandwidth). In tight per-doc ingest loops the OS page cache
        hides the cost at small N, but once the file grows past the
        kernel's dirty-page absorption budget (roughly a few tens of
        MB in practice), every save starts paying real disk I/O and
        the sum becomes quadratic. Observed on 2026-04-19: a 100k
        per-doc ingest on flat snapvec took 40+ minutes of pure
        ``.snpv`` rewrites.

        Fix: mirror the ivfpq pattern and defer the flush for both
        backends. ``vec_chunks`` is the SQLite source of truth, so a
        process crash before the deferred flush runs is recoverable
        via ``_rebuild_snapvec_from_vec_chunks`` (``_init_snapvec``
        detects the staleness on next open and rebuilds
        automatically).

        Callers that want a synchronous flush can invoke
        ``_checkpoint_snapvec()`` directly (``close()`` does this on
        teardown).
        """
        if self._snap is None:
            return
        self._snap_dirty = True

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
        path = self._snapvec_path
        if path.exists():
            try:
                self._snap = SnapIndex.load(str(path))
            except Exception:
                logger.warning(
                    "Failed to reload SnapIndex after rollback — creating empty index. "
                    "Run 'vstash reindex' to rebuild vector search."
                )
                self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
        else:
            self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
        self._snap_dirty = False

    @property
    def last_best_distance(self) -> float:
        """Cosine distance of the best vector match from the last search."""
        return getattr(self._thread_local, "last_best_distance", 0.0)

    @last_best_distance.setter
    def last_best_distance(self, value: float) -> None:
        self._thread_local.last_best_distance = value

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> VstashStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Connection setup                                                     #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        """Create and configure a database connection.

        Tries enable_load_extension + sqlite_vec.load() first (standard approach).
        Falls back to sqlite_vec.Connection if enable_load_extension is unavailable.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row

        try:
            # Standard approach: works when Python is compiled with
            # --enable-loadable-sqlite-extensions (or Homebrew SQLite)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except AttributeError:
            # Fallback: Python without enable_load_extension support
            # Try sqlite_vec.Connection which may handle loading internally
            conn.close()
            conn = sqlite_vec.Connection(str(self.db_path))
            conn.row_factory = sqlite3.Row

        # WAL mode — safe concurrent reads + single writer
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables(conn)
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema if not present."""
        # Create tables first (without indexes on new columns)
        conn.executescript(f"""
            -- Document metadata
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                title       TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'file',
                collection  TEXT NOT NULL DEFAULT 'default',
                project     TEXT,
                layer       TEXT,
                tags        TEXT,
                char_count  INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                added_at    TEXT NOT NULL
            );

            -- Chunk text + position
            CREATE TABLE IF NOT EXISTS chunks (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                seq     INTEGER NOT NULL,
                text    TEXT NOT NULL
            );

            -- Vector index (sqlite-vec).  distance_metric=cosine is
            -- required (#272): sqlite-vec defaults to L2 which is only
            -- monotonic with cosine on unit-normalized embeddings and
            -- exceeds [0, 2] otherwise, breaking non-normalized models.
            -- v1 DBs get rebuilt on open via ``_migrate_v1_to_v2``.
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(embedding float[{self.embedding_dim}] distance_metric=cosine);

            -- Full-text search index (FTS5)
            -- content= makes it a content table — no duplicate text stored
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
            USING fts5(text, content=chunks, content_rowid=id, tokenize='porter ascii');

            -- FTS5 vocabulary table for IDF computation (adaptive RRF weights)
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab
            USING fts5vocab(fts_chunks, row);

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);

            -- Search statistics for adaptive relevance threshold
            CREATE TABLE IF NOT EXISTS search_stats (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                spread     REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Search event telemetry for validating relevance signal.
            -- ``miss_hint`` (issue #157 part 3, 2026-04-21) is a small
            -- JSON blob populated by ``record_search_event`` when a
            -- query returned empty / all-low results; consumed by
            -- ``vstash why --recent`` for post-hoc diagnosis.
            CREATE TABLE IF NOT EXISTS search_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT NOT NULL,
                best_distance   REAL NOT NULL,
                relevance_tier  TEXT NOT NULL,
                result_count    INTEGER NOT NULL,
                dismissed       INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                miss_hint       TEXT
            );

            -- Store metadata: key-value table for tracking what the
            -- current DB was built with.  Used to detect embedding model
            -- drift (e.g. query embedded with one model, corpus embedded
            -- with another) which would silently degrade retrieval.
            CREATE TABLE IF NOT EXISTS store_meta (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_search_events_tier
            ON search_events(relevance_tier, created_at);

            -- Auto-sync FTS5 when chunks are deleted directly
            CREATE TRIGGER IF NOT EXISTS trg_chunks_delete
            AFTER DELETE ON chunks
            BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text)
                VALUES('delete', OLD.id, OLD.text);
            END;
        """)
        conn.commit()

        # Migrate first — columns must exist before creating indexes on them
        self._migrate_schema(conn)

        # Create indexes on potentially-new columns (safe after migration)
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_documents_collection
            ON documents(collection);
            CREATE INDEX IF NOT EXISTS idx_documents_project
            ON documents(project);
            CREATE INDEX IF NOT EXISTS idx_documents_layer
            ON documents(layer);
        """)
        conn.commit()

        # Schema version check (#135).  Stamps fresh DBs with the
        # current SCHEMA_VERSION and validates existing DBs against
        # the known set.  Legacy DBs created before this stamp existed
        # are treated as v1 — that's the only schema vstash has shipped.
        self._check_schema_version(conn)

    def _check_schema_version(self, conn: sqlite3.Connection) -> None:
        """Read or stamp the schema version in ``store_meta``.

        The vec_chunks DDL is the authoritative signal: if it already
        contains ``distance_metric=cosine`` the DB is effectively v2,
        regardless of whether the schema_version row was stamped.  This
        lets us safely re-open DBs created by any prior vstash build,
        including pre-v0.25 unstamped ones and empty pre-v0.25 DBs that
        have no embeddings yet.

        Flow:

        1. Always call :meth:`_migrate_v1_to_v2`; that method is a no-op
           if the ``vec_chunks`` DDL already declares cosine metric.
        2. Read the existing schema_version row.
        3. If we migrated, or the row is missing or stamped v1, promote
           it to the current SCHEMA_VERSION.
        4. Reject any stamped version we do not recognize with
           :class:`SchemaVersionError`.
        """
        from . import __version__ as _vstash_version

        migrated = self._migrate_v1_to_v2(conn)

        pre_row = conn.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone()
        stored_version = str(pre_row["value"]) if pre_row else None

        if stored_version is None or stored_version == "1" or migrated:
            existing = SCHEMA_VERSION
        else:
            existing = stored_version

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ["schema_version", existing, now_iso],
        )
        conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ["vstash_version", _vstash_version, now_iso],
        )
        conn.commit()

        if existing not in KNOWN_SCHEMA_VERSIONS:
            msg = (
                f"Database at {self.db_path} declares schema_version={existing!r}, "
                f"which this build of vstash ({_vstash_version}) does not recognize. "
                f"Known versions: {sorted(KNOWN_SCHEMA_VERSIONS)}. "
                f"Upgrade vstash or restore the DB from a compatible backup."
            )
            raise SchemaVersionError(msg)

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> bool:
        """In-place migrate a v1 DB's ``vec_chunks`` to cosine metric.

        v1 created ``vec_chunks`` as ``vec0(embedding float[N])`` which
        sqlite-vec treats as L2.  v2 requires ``distance_metric=cosine``.
        The stored float bytes are identical under both metrics, so the
        migration reads every ``(rowid, embedding)`` row, drops the old
        virtual table, creates a new one with cosine, and bulk-reinserts
        the same bytes.  No re-embedding required.

        Atomic: the whole rebuild runs inside a ``BEGIN IMMEDIATE``
        transaction so a crash between ``DROP`` and the bulk ``INSERT``
        rolls back both DDL and data, and a second process attempting
        the same migration concurrently hits ``SQLITE_BUSY`` rather than
        racing on the table drop.

        Idempotent: on re-entry the ``sqlite_master`` DDL is inspected
        first; if it already declares cosine metric we return ``False``
        without touching the table.

        Returns:
            ``True`` if the migration rebuilt the table, ``False`` if it
            was already cosine (no-op).
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        if row is None:
            # Defensive: ``_create_schema`` creates ``vec_chunks`` before
            # this runs, but the check keeps the function safe as a
            # standalone helper.
            return False
        if self._ddl_declares_cosine(row["sql"]):
            return False

        # BEGIN IMMEDIATE acquires the reserved lock up front, so a
        # concurrent process opening the same v1 DB blocks here instead
        # of racing on the DROP.  The transaction covers DDL + bulk
        # insert so a crash rolls back to the L2 state rather than
        # leaving an empty cosine table with no embeddings.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Re-check under the lock in case a concurrent process
            # completed the migration between our pre-check and BEGIN.
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
            ).fetchone()
            if row is None or self._ddl_declares_cosine(row["sql"]):
                conn.rollback()
                return False

            # Copy entirely in SQL so we never materialize the full
            # embedding blob in Python memory -- at 384-dim float32 a
            # million rows is ~1.5 GB which a naive ``fetchall`` +
            # ``executemany`` would paginate through Python lists.
            #
            # Two non-obvious choices:
            #
            # 1. The backup is a regular (not ``vec0``) table so the
            #    vec0 shadow tables don't collide with the real
            #    ``vec_chunks`` ones during the rebuild.  ``ALTER TABLE
            #    RENAME`` on vec0 is a trap because the shadow tables
            #    (``vec_chunks_chunks``, ``vec_chunks_rowids``, ...)
            #    keep the original name and every subsequent query
            #    fails with ``no such table: main.vec_chunks_chunks``.
            #
            # 2. The backup lives in ``TEMP`` so it does not grow the
            #    main DB file.  SQLite does not auto-shrink the main
            #    DB after ``DROP TABLE``; pages stay on the freelist
            #    and a migration on a large store would permanently
            #    ~double the file size until a manual ``VACUUM``.
            #    TEMP tables write to the per-connection temp DB and
            #    disappear on connection close.
            conn.execute(
                "CREATE TEMP TABLE _vec_chunks_v2_backup "
                "(rowid INTEGER PRIMARY KEY, embedding BLOB)"
            )
            # ``cursor.rowcount`` on an ``INSERT INTO ... SELECT``
            # returns the number of rows inserted without the extra
            # table scan that a follow-up ``SELECT COUNT(*)`` would
            # cost on large stores.
            cursor = conn.execute(
                "INSERT INTO _vec_chunks_v2_backup (rowid, embedding) "
                "SELECT rowid, embedding FROM vec_chunks"
            )
            row_count = cursor.rowcount
            conn.execute("DROP TABLE vec_chunks")
            conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks "
                f"USING vec0(embedding float[{self.embedding_dim}] distance_metric=cosine)"
            )
            conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) "
                "SELECT rowid, embedding FROM _vec_chunks_v2_backup"
            )
            conn.execute("DROP TABLE _vec_chunks_v2_backup")
            # Clear adaptive-threshold history: spread values recorded
            # under L2 are not comparable to cosine spreads.  Dropping
            # them lets the rolling window repopulate from v2 searches.
            conn.execute("DELETE FROM search_stats")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        logger.info(
            "Migrated vec_chunks from L2 to cosine metric (schema v1 -> v2, %d embeddings rebuilt)",
            row_count,
        )
        return True

    @staticmethod
    def _ddl_declares_cosine(ddl: str | None) -> bool:
        """Whitespace-insensitive check for ``distance_metric=cosine``.

        sqlite_master stores DDL verbatim, including whitespace, so a
        naive substring check would miss ``distance_metric = cosine`` or
        similar variants.  Strip whitespace and lower-case before the
        compare.
        """
        if not ddl:
            return False
        normalized = "".join(ddl.lower().split())
        return "distance_metric=cosine" in normalized

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to existing databases."""
        doc_columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        migrations: list[str] = []
        if "collection" not in doc_columns:
            migrations.append(
                "ALTER TABLE documents ADD COLUMN collection TEXT NOT NULL DEFAULT 'default'"
            )
        if "project" not in doc_columns:
            migrations.append("ALTER TABLE documents ADD COLUMN project TEXT")
        if "layer" not in doc_columns:
            migrations.append("ALTER TABLE documents ADD COLUMN layer TEXT")
        if "tags" not in doc_columns:
            migrations.append("ALTER TABLE documents ADD COLUMN tags TEXT")

        # miss_hint column on search_events (issue #157 part 3, 2026-04-21)
        event_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(search_events)").fetchall()
        }
        if event_columns and "miss_hint" not in event_columns:
            migrations.append("ALTER TABLE search_events ADD COLUMN miss_hint TEXT")

        # Frequency + decay scoring columns on chunks
        chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "access_count" not in chunk_columns:
            migrations.append("ALTER TABLE chunks ADD COLUMN access_count INTEGER DEFAULT 0")
        if "last_accessed_at" not in chunk_columns:
            migrations.append("ALTER TABLE chunks ADD COLUMN last_accessed_at TEXT")
        if "created_at" not in chunk_columns:
            migrations.append("ALTER TABLE chunks ADD COLUMN created_at TEXT")

        for sql in migrations:
            conn.execute(sql)
        if migrations:
            conn.commit()

        # Backfill created_at from parent document's added_at
        if "created_at" not in chunk_columns:
            conn.execute("""
                UPDATE chunks SET created_at = (
                    SELECT d.added_at FROM documents d WHERE d.id = chunks.doc_id
                ) WHERE created_at IS NULL
            """)
            conn.commit()

        # Fix v0.5.0 data: ingestion set access_count=1 for chunks that were
        # never actually searched. Detect by access_count=1 + no last_accessed_at.
        if "access_count" in chunk_columns:
            conn.execute("""
                UPDATE chunks SET access_count = 0
                WHERE access_count = 1 AND last_accessed_at IS NULL
            """)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Store metadata (key/value)                                            #
    # ------------------------------------------------------------------ #

    def get_meta(self, key: str) -> str | None:
        """Read a key from the store_meta table.  Returns None if unset."""
        row = self._conn.execute("SELECT value FROM store_meta WHERE key = ?", [key]).fetchone()
        return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a key into the store_meta table."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO store_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                [key, value, now_iso],
            )
            self._conn.commit()

    def check_embedding_drift(self, current_model: str) -> str | None:
        """Detect if the current embedding model differs from what the
        DB was built with.  Returns a warning message if drift is
        detected, or None if everything matches.

        This is the single most important check a substrate can do for
        a user: embedding model drift silently degrades retrieval
        quality without producing any visible error.

        On first use (empty store_meta), this sets the model name and
        returns None.  Subsequent opens compare and warn on mismatch.
        """
        stored_model = self.get_meta("embedding_model")
        stored_fastembed = self.get_meta("fastembed_version")

        # Detect current fastembed version (best-effort)
        try:
            import fastembed

            current_fastembed = fastembed.__version__
        except Exception:
            current_fastembed = "unknown"

        if stored_model is None:
            # First time seeing metadata — claim the DB with the current
            # model so future opens can detect drift.
            self.set_meta("embedding_model", current_model)
            self.set_meta("fastembed_version", current_fastembed)
            return None

        if stored_model != current_model:
            return (
                f"Embedding model mismatch: this database was built with "
                f"'{stored_model}' but the current config uses '{current_model}'. "
                f"Query embeddings and stored embeddings are incompatible. "
                f"Run `vstash reindex --model {current_model}` or revert the "
                f"config to '{stored_model}'."
            )

        # Same model name, but fastembed version changed AND the model
        # is one of the known-affected ones (multilingual variants that
        # switched from CLS pooling to mean pooling in fastembed 0.5.2+).
        _POOLING_DRIFT_MODELS = {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }
        if (
            current_model in _POOLING_DRIFT_MODELS
            and stored_fastembed is not None
            and stored_fastembed != current_fastembed
        ):
            return (
                f"fastembed version changed from {stored_fastembed} to "
                f"{current_fastembed} while using '{current_model}'. "
                "This model switched from CLS pooling to mean pooling "
                "in fastembed 0.5.2, producing incompatible embeddings. "
                "Run `vstash reindex` to re-embed the corpus with the "
                "current fastembed version, or pin fastembed to the "
                "version that created the DB."
            )

        return None

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def doc_exists(self, path: str) -> bool:
        """Check if a document with the given path is already ingested.

        Args:
            path: File path or URL to check.

        Returns:
            True if the document exists in the store.
        """
        row = self._conn.execute("SELECT 1 FROM documents WHERE path = ?", [path]).fetchone()
        return row is not None

    def add_document(
        self,
        path: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_type: str = "file",
        collection: str = "default",
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
    ) -> str:
        """Add a document and its chunks to the store.

        If the document already exists (same path hash), it is replaced.

        Args:
            path: Absolute file path or URL.
            title: Human-readable document title.
            chunks: List of text chunks.
            embeddings: Corresponding embedding vectors.
            source_type: Document type (pdf, code, url, etc.).
            collection: Named collection to group this document.
            project: Project identifier from frontmatter.
            layer: Layer/category from frontmatter.
            tags: Comma-separated tags from frontmatter.

        Returns:
            The generated document ID (32-char hex hash).
        """
        # Fail-safe validation at the public API boundary (#133).  Catches
        # the four common pathological ingestion cases (overlong path,
        # zero chunks, way too many chunks, an oversized chunk) before
        # we open a write transaction.
        validate_document_input(
            path=path,
            chunks=chunks,
            embeddings=embeddings,
            limits=self._limits,
        )

        doc_id = hashlib.sha256(f"{collection}:{path}".encode()).hexdigest()[:32]

        with self._write_lock:
            # Explicit transaction ensures atomicity — a crash mid-way
            # won't leave the database in an inconsistent state.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Remove existing version if re-ingesting
                self._delete_by_doc_id(doc_id)

                self._conn.execute(
                    """INSERT INTO documents
                       (id, path, title, source_type, collection,
                        project, layer, tags,
                        char_count, chunk_count, added_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        doc_id,
                        path,
                        title,
                        source_type,
                        collection,
                        project,
                        layer,
                        tags,
                        sum(len(c) for c in chunks),
                        len(chunks),
                        datetime.now(timezone.utc).isoformat(),
                    ],
                )

                now_iso = datetime.now(timezone.utc).isoformat()

                # Insert chunk — last_accessed_at is NULL until the chunk is actually accessed
                # via search, so the decay formula doesn't treat new chunks as "recently accessed".
                chunk_data = [(doc_id, seq, text, now_iso) for seq, text in enumerate(chunks)]
                self._conn.executemany(
                    "INSERT INTO chunks (doc_id, seq, text, access_count, created_at, last_accessed_at)"
                    " VALUES (?, ?, ?, 0, ?, NULL)",
                    chunk_data,
                )

                # Get rowids for linking vec + fts tables
                rowids = [
                    row[0]
                    for row in self._conn.execute(
                        "SELECT id FROM chunks WHERE doc_id = ? ORDER BY seq",
                        [doc_id],
                    ).fetchall()
                ]

                # Vector index entries. rowids comes from a SELECT-by-doc_id
                # right after the chunk inserts above, so it must align 1:1
                # with the input embeddings/chunks; strict=True surfaces any
                # invariant violation as a ValueError instead of silently
                # truncating to the shorter side.
                vec_data = [
                    (rowid, _serialize(emb)) for rowid, emb in zip(rowids, embeddings, strict=True)
                ]
                self._conn.executemany(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    vec_data,
                )

                # FTS5 entries (rowid must match chunks.id)
                fts_data = [(rowid, text) for rowid, text in zip(rowids, chunks, strict=True)]
                if not self._defer_fts:
                    self._conn.executemany(
                        "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                        fts_data,
                    )

                # Add to snapvec in-memory (persisted after successful commit)
                if self._snap is not None:
                    snap_vecs = np.array(embeddings, dtype=np.float32)
                    self._snap.add_batch(rowids, snap_vecs)
                    self._snap_dirty = True

                self._conn.commit()
                if self._defer_fts:
                    self._deferred_fts_rows.extend(fts_data)
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
                # Persist snapvec AFTER successful SQLite commit
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise
        return doc_id

    def add_documents_batch(
        self,
        documents: list[dict],
    ) -> list[str]:
        """Ingest multiple documents in a single transaction.

        Significantly faster than calling add_document() in a loop when
        ingesting many documents, because it amortises the transaction
        overhead (BEGIN/COMMIT) across all documents.

        Args:
            documents: List of dicts, each with keys:
                path, title, chunks, embeddings, source_type,
                and optionally: collection, project, layer, tags.

        Returns:
            List of generated document IDs (same order as input).
        """
        if not documents:
            return []

        from .validation import validate_document_input

        doc_ids: list[str] = []

        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            pending_fts: list[tuple[int, str]] = []
            # Coalesce every doc's vectors and rowids into one buffer so
            # we hand ``SnapIndex.add_batch`` a single (N, dim) array at
            # the end. Its internal ``np.vstack`` is O(total_size + new)
            # per call, so calling it N times with size-1 inputs during
            # the loop would be O(N^2) memory copies (measured: 500k
            # docs via the old path took ~11 minutes of pure RAM
            # memcpy). One final batch drops the cost to O(N).
            pending_snap_rowids: list[int] = []
            pending_snap_vecs: list[np.ndarray] = []
            try:
                now_iso = datetime.now(timezone.utc).isoformat()

                for doc in documents:
                    path = doc["path"]
                    title = doc["title"]
                    chunks = doc["chunks"]
                    embeddings = doc["embeddings"]
                    source_type = doc["source_type"]
                    collection = doc.get("collection", "default")
                    project = doc.get("project")
                    layer = doc.get("layer")
                    tags = doc.get("tags")

                    validate_document_input(
                        path=path,
                        chunks=chunks,
                        embeddings=embeddings,
                        limits=self._limits,
                    )

                    doc_id = hashlib.sha256(f"{collection}:{path}".encode()).hexdigest()[:32]
                    doc_ids.append(doc_id)

                    self._delete_by_doc_id(doc_id)

                    self._conn.execute(
                        """INSERT INTO documents
                           (id, path, title, source_type, collection,
                            project, layer, tags,
                            char_count, chunk_count, added_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            doc_id,
                            path,
                            title,
                            source_type,
                            collection,
                            project,
                            layer,
                            tags,
                            sum(len(c) for c in chunks),
                            len(chunks),
                            now_iso,
                        ],
                    )

                    chunk_data = [(doc_id, seq, text, now_iso) for seq, text in enumerate(chunks)]
                    self._conn.executemany(
                        "INSERT INTO chunks (doc_id, seq, text, access_count, "
                        "created_at, last_accessed_at) VALUES (?, ?, ?, 0, ?, NULL)",
                        chunk_data,
                    )

                    rowids = [
                        row[0]
                        for row in self._conn.execute(
                            "SELECT id FROM chunks WHERE doc_id = ? ORDER BY seq",
                            [doc_id],
                        ).fetchall()
                    ]

                    vec_data = [
                        (rowid, _serialize(emb))
                        for rowid, emb in zip(rowids, embeddings, strict=True)
                    ]
                    self._conn.executemany(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        vec_data,
                    )

                    fts_data = list(zip(rowids, chunks, strict=True))
                    if self._defer_fts:
                        pending_fts.extend(fts_data)
                    else:
                        self._conn.executemany(
                            "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                            fts_data,
                        )

                    if self._snap is not None:
                        pending_snap_rowids.extend(rowids)
                        pending_snap_vecs.append(np.asarray(embeddings, dtype=np.float32))

                # ONE coalesced add_batch for the whole transaction's
                # worth of vectors instead of one per document. Avoids
                # the quadratic np.vstack inside SnapIndex.add_batch.
                if self._snap is not None and pending_snap_rowids:
                    all_snap_vecs = (
                        pending_snap_vecs[0]
                        if len(pending_snap_vecs) == 1
                        else np.concatenate(pending_snap_vecs, axis=0)
                    )
                    self._snap.add_batch(pending_snap_rowids, all_snap_vecs)
                    self._snap_dirty = True

                self._conn.commit()
                if pending_fts:
                    self._deferred_fts_rows.extend(pending_fts)
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise

        return doc_ids

    def delete_by_path_prefix(self, prefix: str) -> int:
        """Remove all documents whose path starts with *prefix*.

        Useful for bulk-removing documents when a directory is deleted.

        Args:
            prefix: Path prefix (e.g. ``/home/user/docs/``).
                Must not be empty.

        Returns:
            Number of documents deleted.

        Raises:
            ValueError: If prefix is empty (would match all documents).
        """
        if not prefix:
            raise ValueError("prefix must not be empty")
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT id FROM documents WHERE path LIKE ? ESCAPE '\\'",
                [prefix.replace("%", "\\%").replace("_", "\\_") + "%"],
            ).fetchall()
            if not rows:
                return 0
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._delete_by_doc_ids([row[0] for row in rows])
                self._conn.commit()
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise
            return len(rows)

    def doc_completeness(
        self, path: str, collection: str = "default"
    ) -> Literal["missing", "partial", "complete"]:
        """Report whether a document at ``path`` is fully ingested (#134).

        ``doc_exists`` only checks the documents row; this method goes
        further and verifies that:
          - the documents row exists for ``(collection, path)``
          - ``COUNT(chunks) == documents.chunk_count``
          - every chunk has a matching ``vec_chunks`` row (sqlite-vec
            backend only — snapvec lives outside SQLite and is not
            checked here)

        ``collection`` is required for correctness: the same path can
        be ingested into multiple collections (each gets its own
        ``doc_id = sha256(collection:path)``), and a partial copy in
        one collection must not mark another collection's complete
        copy as healable.

        Used by idempotent ``ingest()`` to decide whether to skip
        (``"complete"``), re-ingest (``"partial"`` — delete the partial
        rows first), or freshly ingest (``"missing"``).
        """
        # Use the same hash recipe as add_document so we look up the
        # *exact* row that a fresh ingest would write to.  Looking up
        # by ``WHERE path = ?`` would conflate collections.
        doc_id = hashlib.sha256(f"{collection}:{path}".encode()).hexdigest()[:32]
        row = self._conn.execute(
            "SELECT chunk_count FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        if row is None:
            return "missing"
        declared_chunks = int(row[0] or 0)

        actual_chunks = int(
            self._conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", [doc_id]).fetchone()[
                0
            ]
        )
        if actual_chunks != declared_chunks or actual_chunks == 0:
            return "partial"

        # Vector index parity (sqlite-vec only).  Snapvec lives in a
        # sidecar file; integrity_check() handles it separately.
        if self._snap is None:
            missing_vec = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM chunks c
                    LEFT JOIN vec_chunks v ON v.rowid = c.id
                    WHERE c.doc_id = ? AND v.rowid IS NULL
                    """,
                    [doc_id],
                ).fetchone()[0]
            )
            if missing_vec > 0:
                return "partial"

        return "complete"

    def delete_document(self, path: str, collection: str | None = None) -> bool:
        """Remove a document and all its chunks from the store.

        By default, deletes every copy of ``path`` regardless of
        collection (the existing behavior, which several callers depend
        on).  When ``collection`` is provided, only the document for
        that exact ``(collection, path)`` pair is removed — used by
        idempotent ingest recovery so a partial copy in one collection
        doesn't wipe a complete copy in another.

        Args:
            path: File path or URL to remove.
            collection: If set, restrict deletion to this collection.

        Returns:
            True if at least one document was found and deleted.
        """
        with self._write_lock:
            if collection is None:
                doc_ids = [
                    row[0]
                    for row in self._conn.execute(
                        "SELECT id FROM documents WHERE path = ?", [path]
                    ).fetchall()
                ]
            else:
                doc_ids = [
                    row[0]
                    for row in self._conn.execute(
                        "SELECT id FROM documents WHERE path = ? AND collection = ?",
                        [path, collection],
                    ).fetchall()
                ]
            if not doc_ids:
                return False
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._delete_by_doc_ids(doc_ids)
                self._conn.commit()
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
                # Persist snapvec AFTER successful SQLite commit
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise
            return True

    def _delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete a document by its internal hash ID.

        Args:
            doc_id: 32-char hex document hash.

        Returns:
            True if the document existed and was deleted.
        """
        return self._delete_by_doc_ids([doc_id])

    def _delete_by_doc_ids(self, doc_ids: list[str]) -> bool:
        """Delete documents by their internal hash IDs.

        Args:
            doc_ids: List of 32-char hex document hashes.

        Returns:
            True if at least one document existed and was deleted.
        """
        if not doc_ids:
            return False

        total_deleted = 0
        for i in range(0, len(doc_ids), _SQLITE_PARAM_BATCH):
            batch_doc_ids = doc_ids[i : i + _SQLITE_PARAM_BATCH]
            doc_placeholders = ",".join("?" * len(batch_doc_ids))

            chunk_ids = [
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM chunks WHERE doc_id IN ({doc_placeholders})", batch_doc_ids
                ).fetchall()
            ]

            if chunk_ids:
                for j in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
                    batch_chunk_ids = chunk_ids[j : j + _SQLITE_PARAM_BATCH]
                    chunk_placeholders = ",".join("?" * len(batch_chunk_ids))

                    # Delete vec_chunks first (no trigger involved)
                    self._conn.execute(
                        f"DELETE FROM vec_chunks WHERE rowid IN ({chunk_placeholders})",
                        batch_chunk_ids,
                    )

                    # Delete from snapvec index (in-memory, persisted after commit).
                    # snapvec >= 0.6 makes .delete() O(1) via swap-with-last, so a
                    # per-id loop is the right API — no delete_batch needed.
                    if self._snap is not None:
                        for cid in batch_chunk_ids:
                            self._snap.delete(cid)
                        self._snap_dirty = True

                # Delete chunks — trg_chunks_delete trigger auto-syncs FTS5
                self._conn.execute(
                    f"DELETE FROM chunks WHERE doc_id IN ({doc_placeholders})", batch_doc_ids
                )

            cursor = self._conn.execute(
                f"DELETE FROM documents WHERE id IN ({doc_placeholders})", batch_doc_ids
            )
            total_deleted += cursor.rowcount

        return total_deleted > 0

    # ------------------------------------------------------------------ #
    # Search — Hybrid RRF                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_search_cache_key(
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        vec_weight: float | None,
        fts_weight: float | None,
        distance_cutoff: float,
        collection: str | None,
        project: str | None,
        layer: str | None,
        adaptive_rrf: bool,
        recency_boost: float,
        added_after: str | None,
        added_before: str | None,
        mmr_lambda: float,
        retrieval_mode: str,
        cache_epoch: int,
    ) -> int:
        """Build the search cache key from the full set of query parameters.

        ``cache_epoch`` is mixed in so a write invalidates every cached
        entry without having to scan the LRU. ``retrieval_mode`` is
        one of ``"hybrid" | "vec_only" | "fts_only"`` so queries that
        short-circuit one branch do not collide with hybrid queries of
        the same text.
        """
        emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        return hash(
            (
                emb_bytes,
                query_text,
                top_k,
                vec_weight,
                fts_weight,
                distance_cutoff,
                collection,
                project,
                layer,
                adaptive_rrf,
                recency_boost,
                added_after,
                added_before,
                mmr_lambda,
                retrieval_mode,
                cache_epoch,
            )
        )

    @staticmethod
    def _resolve_retrieval_mode(
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None,
    ) -> Literal["hybrid", "vec_only", "fts_only"]:
        """Normalize ``retrieval_mode`` to the canonical enum.

        ``None`` resolves to ``"hybrid"`` (the default pipeline).  Any
        other value must be one of the three known modes; everything
        else raises ``ValueError`` with the expected set.

        The legacy ``fts_only=True`` bool parameter was deprecated in
        v0.33.0 and removed in v0.35.0 (#281).  Callers still passing
        it will hit a ``TypeError`` from Python's argument binder
        instead of a soft deprecation warning -- an intentional hard
        break after two releases of advance notice.
        """
        if retrieval_mode is None:
            return "hybrid"
        if retrieval_mode not in ("hybrid", "vec_only", "fts_only"):
            raise ValueError(
                f"retrieval_mode must be one of 'hybrid', 'vec_only', 'fts_only'; "
                f"got {retrieval_mode!r}"
            )
        return retrieval_mode

    @staticmethod
    def _resolve_rrf_weights(
        vec_weight: float | None, fts_weight: float | None
    ) -> tuple[float, float]:
        """Fill in missing RRF weights with the default 0.6/0.4 split.

        If only one side is provided, the other is set so the pair sums
        to 1.0. If both are ``None``, return the historical defaults.
        """
        if vec_weight is None and fts_weight is None:
            return 0.6, 0.4
        if vec_weight is None:
            assert fts_weight is not None
            fts_w = float(fts_weight)
            return 1.0 - fts_w, fts_w
        if fts_weight is None:
            return float(vec_weight), 1.0 - float(vec_weight)
        return float(vec_weight), float(fts_weight)

    @staticmethod
    def _build_fts_match_query(
        query_text: str, words: list[str] | None = None
    ) -> tuple[str, list[str]]:
        """Build an injection-safe FTS5 MATCH string from raw query text.

        Returns ``(match_string, quoted_words)``. Words of length 1 are
        dropped (FTS5 tokenizer usually strips them anyway) and each
        token is double-quoted so FTS5 cannot interpret operators like
        NEAR/NOT/OR from user input. If nothing survives, the entire
        query is quoted as a single phrase.

        ``words`` may be passed by callers that already split the query
        (e.g. ``search()``) so we do not tokenize twice on the hot path.
        """
        if words is None:
            words = query_text.split()
        quoted_words = ['"' + w.replace('"', '""') + '"' for w in words if len(w) > 1]
        if quoted_words:
            return " OR ".join(quoted_words), quoted_words
        return '"' + query_text.replace('"', '""') + '"', quoted_words

    def _fuse_rrf_scores(
        self,
        vec_rows: list,
        fts_rows: list,
        vec_weight: float,
        fts_weight: float,
        relevant_chunk_ids: set[int],
        effective_k: int,
        fts_only: bool,
        explain: bool,
    ) -> tuple[
        dict[int, dict[str, str | int | float]],
        dict[int, float],
        dict[int, float],
        dict[int, int],
    ]:
        """Run Reciprocal Rank Fusion over vector and FTS result sets.

        Returns ``(scores, explain_rrf_vec, explain_rrf_fts, explain_fts_rank)``.
        The explain maps are populated only when ``explain=True``, otherwise
        they are empty.

        FTS rows that neither pass the vector distance filter nor land in
        the top ``effective_k * 2`` FTS hits are dropped, to keep weak
        keyword noise out of the score pool when vector search has a
        strong opinion. In ``fts_only`` mode this gate is disabled.
        """
        scores: dict[int, dict[str, str | int | float]] = {}
        explain_rrf_vec: dict[int, float] = {}
        explain_rrf_fts: dict[int, float] = {}
        explain_fts_rank: dict[int, int] = {}

        for rank, row in enumerate(vec_rows):
            chunk_id: int = row["id"]
            vec_contrib = vec_weight * (1.0 / (RRF_K + rank))
            scores[chunk_id] = {
                "id": chunk_id,
                "text": row["text"],
                "title": row["title"],
                "path": row["path"],
                "chunk": row["seq"],
                "rrf": vec_contrib,
                "added_at": row["added_at"],
                "collection": row["collection"],
                "tags": row["tags"],
                "layer": row["layer"],
            }
            if explain:
                explain_rrf_vec[chunk_id] = vec_contrib

        for rank, row in enumerate(fts_rows):
            chunk_id = row["id"]
            # is_fts_top is the gate that lets FTS-only hits into the
            # score pool when they are not also in the vector candidate
            # set. In fts_only mode the vector signal is absent, so we
            # let every FTS row through up to the candidate pool.
            is_fts_top = True if fts_only else rank < effective_k * 2
            fts_contribution = fts_weight * (1.0 / (RRF_K + rank))
            if chunk_id in scores:
                scores[chunk_id]["rrf"] = float(scores[chunk_id]["rrf"]) + fts_contribution
                if explain:
                    explain_rrf_fts[chunk_id] = fts_contribution
                    explain_fts_rank[chunk_id] = rank
            elif chunk_id in relevant_chunk_ids or is_fts_top:
                scores[chunk_id] = {
                    "id": chunk_id,
                    "text": row["text"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk": row["seq"],
                    "rrf": fts_contribution,
                    "added_at": row["added_at"],
                    "collection": row["collection"],
                    "tags": row["tags"],
                    "layer": row["layer"],
                }
                if explain:
                    explain_rrf_fts[chunk_id] = fts_contribution
                    explain_fts_rank[chunk_id] = rank

        return scores, explain_rrf_vec, explain_rrf_fts, explain_fts_rank

    def _apply_recency_boost(
        self,
        ranked: list[dict[str, str | int | float]],
        recency_boost: float,
    ) -> list[dict[str, str | int | float]]:
        """Multiply each chunk's RRF score by an exponential decay factor.

        ``recency_boost=0`` disables the boost. Chunks whose ``created_at``
        is unparseable are left untouched. Returns a newly sorted list
        so callers always see post-boost ordering.
        """
        if recency_boost <= 0 or not ranked:
            return ranked

        now = datetime.now(timezone.utc)
        chunk_ids = [int(r["id"]) for r in ranked]
        # Batch the IN clause so large top_k / candidate pools don't trip
        # SQLite's default SQLITE_LIMIT_VARIABLE_NUMBER (999 on most builds).
        created_map: dict[int, datetime] = {}
        for start in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
            batch = chunk_ids[start : start + _SQLITE_PARAM_BATCH]
            placeholders = ",".join("?" * len(batch))
            for row in self._conn.execute(
                f"SELECT id, created_at FROM chunks WHERE id IN ({placeholders})",
                batch,
            ).fetchall():
                try:
                    created_map[row["id"]] = datetime.fromisoformat(row["created_at"])
                except (TypeError, ValueError):
                    pass

        for r in ranked:
            cid = int(r["id"])
            if cid in created_map:
                days_ago = max(0.0, (now - created_map[cid]).total_seconds() / 86400)
                decay = math.exp(-0.05 * days_ago)
                r["rrf"] = float(r["rrf"]) * (1.0 + recency_boost * decay)

        return sorted(ranked, key=lambda x: float(x["rrf"]), reverse=True)

    @staticmethod
    def _build_search_results(
        ranked: list[dict[str, str | int | float]],
        *,
        explain: bool,
        explain_vec: dict[int, tuple[int, float]],
        explain_rrf_vec: dict[int, float],
        explain_rrf_fts: dict[int, float],
        explain_fts_rank: dict[int, int],
        quoted_words: list[str],
        vec_weight: float,
        fts_weight: float,
    ) -> list[SearchResult]:
        """Materialize ``SearchResult`` instances with optional ExplainInfo.

        All rounding is applied here so callers see stable floating-point
        values (scores to 6 decimals, distances to 4).
        """
        explain_map: dict[int, ExplainInfo] = {}
        if explain:
            fts_terms = [w.strip('"') for w in quoted_words] if quoted_words else []
            for r in ranked:
                cid = int(r["id"])
                vec_info = explain_vec.get(cid)
                rrf_vec_val = round(explain_rrf_vec.get(cid, 0.0), 6)
                rrf_fts_val = round(explain_rrf_fts.get(cid, 0.0), 6)
                explain_map[cid] = ExplainInfo(
                    vec_rank=vec_info[0] if vec_info else None,
                    vec_distance=round(vec_info[1], 4) if vec_info else None,
                    fts_rank=explain_fts_rank.get(cid),
                    rrf_vec=rrf_vec_val,
                    rrf_fts=rrf_fts_val,
                    rrf_total=round(rrf_vec_val + rrf_fts_val, 6),
                    mmr_penalty=round(float(r.get("_mmr_penalty", 0.0)), 4),
                    fts_terms=fts_terms,
                    rrf_vec_weight=vec_weight,
                    rrf_fts_weight=fts_weight,
                )

        return [
            SearchResult(
                chunk_id=int(r["id"]),
                text=str(r["text"]),
                title=str(r["title"]),
                path=str(r["path"]),
                chunk=int(r["chunk"]),
                score=round(float(r["rrf"]), 6),
                explain=explain_map.get(int(r["id"])) if explain else None,
                added_at=r.get("added_at"),
                collection=r.get("collection"),
                tags=r.get("tags"),
                layer=r.get("layer"),
            )
            for r in ranked
        ]

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        distance_cutoff: float = 1.3225,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        explain: bool = False,
        adaptive_rrf: bool = True,
        recency_boost: float = 0.0,
        added_after: str | None = None,
        added_before: str | None = None,
        mmr_lambda: float = 0.5,
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
        exact_match: str | None = None,
        exact_match_case_sensitive: bool = False,
        _tracer: _PipelineTracer | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: vector (semantic) + FTS5 (keyword) combined with RRF.

        RRF score = vec_weight * 1/(k+rank_vec) + fts_weight * 1/(k+rank_fts)

        Results are filtered by vector distance — chunks whose distance from
        the query is more than ``distance_cutoff`` times the best (closest)
        distance are discarded before RRF scoring.  This prevents irrelevant
        noise (e.g. Art of War appearing in deep learning queries).

        Args:
            query_embedding: Query vector from the embedding model.
            query_text: Raw query text for FTS5 keyword matching.
            top_k: Number of results to return.
            vec_weight: Weight for vector search contribution.
            fts_weight: Weight for keyword search contribution.
            distance_cutoff: Maximum allowed ratio of distance to best distance.
                Chunks with distance > best_distance * distance_cutoff are dropped.
                Default 1.3225 = 1.15^2; under BGE unit-normalized
                embeddings the cosine-distance ratio relates to the
                legacy v1 L2-distance ratio by a square (#272), so this
                value preserves v1 cutoff semantics.
            collection: If set, restrict search to documents in this collection.
            project: If set, restrict search to documents with this project tag.
            layer: If set, restrict search to documents with this layer tag.
            exact_match: Optional substring that each returned chunk's
                ``text`` must contain. Applied as a post-filter after
                the full pipeline so the upstream candidate pool does
                NOT know about this constraint -- callers who need a
                guaranteed top_k under a selective substring should
                request a larger ``top_k`` and accept that fewer
                results may come back. Bypasses FTS5 tokenization, so
                literal strings with punctuation / casing / code
                identifiers survive (unlike the FTS5 keyword path
                which stems and lowercases). #106, 2026-04-21.
            exact_match_case_sensitive: Toggle for the above. Default
                ``False`` does a casefold comparison which matches
                typical retrieval-filter UX expectations.
            retrieval_mode: Which search branches to run (default
                ``"hybrid"``). Three values:

                - ``"hybrid"`` (default): vector ANN + FTS5 + adaptive
                  RRF + distance cutoff + MMR. This is the pipeline the
                  paper and README benchmarks are measured against.
                - ``"fts_only"`` (#152): skip the vector ANN scan,
                  distance cutoff, and adaptive RRF. FTS5 hits still
                  flow through RRF scoring (``vec_weight=0.0``,
                  ``fts_weight=1.0``), MMR, recency boost, and context
                  expansion — not a raw BM25 dump. Useful for queries
                  with diffuse vector representations (cross-lingual,
                  highly technical) or as a fallback when the vector
                  pool is expected to be empty.
                - ``"vec_only"`` (#275): symmetric to ``"fts_only"``.
                  Skip the FTS5 keyword search; force
                  ``vec_weight=1.0``, ``fts_weight=0.0``. Useful when
                  the corpus has no meaningful keyword signal (tabular
                  data, code where identifiers are noise, cross-lingual
                  corpora where tokenization disagrees with the query)
                  or for benchmarking / ranking debug.

        Returns:
            Ranked list of SearchResult ordered by descending score.
        """
        _mode = self._resolve_retrieval_mode(retrieval_mode)
        _fts_only = _mode == "fts_only"
        _vec_only = _mode == "vec_only"
        # Fail-safe validation at the public API boundary (#133).  Runs
        # before any work — if the caller passed a 50k-token query or a
        # negative top_k, we reject it here instead of crashing inside
        # SQLite or the embedding model with a cryptic error.  Cost is
        # a handful of comparisons.
        validate_search_input(
            query_text=query_text,
            top_k=top_k,
            distance_cutoff=distance_cutoff,
            recency_boost=recency_boost,
            limits=self._limits,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
        )

        # --- Query cache key ---
        # Skip the cache when an exact_match filter is ACTIVE (non-empty).
        # An empty string and None are no-ops against the post-filter and
        # stay cacheable. Including the substring in the cache key would
        # be correct but blow up key cardinality and reduce reuse; the
        # filter is rare enough that the simpler skip is preferable.
        _cache_key: int | None = None
        _cache_max = self._cache_config.query_cache_size
        if _cache_max > 0 and _tracer is None and not explain and not exact_match:
            _cache_key = self._compute_search_cache_key(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
                distance_cutoff=distance_cutoff,
                collection=collection,
                project=project,
                layer=layer,
                adaptive_rrf=adaptive_rrf,
                recency_boost=recency_boost,
                added_after=added_after,
                added_before=added_before,
                mmr_lambda=mmr_lambda,
                retrieval_mode=_mode,
                cache_epoch=self._cache_epoch,
            )

        # Per-search miss-analysis tracker (#108).  The tracer is owned
        # by the caller (miss_analysis) and passed in; this keeps the
        # tracking state thread-local to the caller and zero-cost on
        # the regular search hot path (tracer is None).
        #
        # Internal: _tracer is a hook for miss_analysis() only.  Power
        # users of VstashStore should not pass it directly.
        track_target: int | None = _tracer.target if _tracer is not None else None

        # Observability: we wrap the entire body in try/finally so that
        # latency and slow_queries_total are recorded even when the body
        # raises.  An earlier version used Timer.__enter__/__exit__ with
        # no finally — exceptions silently dropped the histogram write,
        # causing searches_total to drift ahead of the histogram count
        # (exactly the metrics skew you do NOT want when debugging a
        # production incident).
        from .metrics import registry

        _search_start = time.perf_counter()
        _result_count = 0
        # vec_weight/fts_weight may still be None here — initialize to
        # 0.0 so the slow-query log in finally doesn't crash if we fail
        # before adaptive RRF has resolved them.
        _vec_weight_observed: float = 0.0
        _fts_weight_observed: float = 0.0
        registry.counter_inc("searches_total")
        try:
            # --- Cache hit (inside try/finally so observability is recorded) ---
            if _cache_key is not None:
                cached: list[SearchResult] | None = None
                with self._cache_lock:
                    if _cache_key in self._query_cache:
                        self._query_cache.move_to_end(_cache_key)
                        cached = list(self._query_cache[_cache_key])
                if cached is not None:
                    _result_count = len(cached)
                    registry.counter_inc("query_cache_hits_total")
                    return cached

            # Branch short-circuits for non-hybrid modes.
            #
            # FTS-only (#152): bypass vector search entirely. Force
            # weights to (0.0, 1.0) and disable adaptive RRF so the
            # pipeline cannot silently re-enable the vector path. The
            # vector search, distance cutoff, and relevance filter
            # blocks below are guarded on ``not _fts_only`` or on
            # ``vec_rows`` being non-empty, so downstream MMR / scoring
            # runs unchanged.
            #
            # Vec-only (#275): symmetric. Bypass the FTS5 query. Force
            # weights to (1.0, 0.0) and disable adaptive RRF so the
            # FTS path cannot re-enable itself via IDF signals. The
            # ``fts_rows`` block below is skipped when ``_vec_only``
            # so no FTS hits ever enter the RRF fusion.
            if _fts_only:
                vec_weight = 0.0
                fts_weight = 1.0
                adaptive_rrf = False
            elif _vec_only:
                vec_weight = 1.0
                fts_weight = 0.0
                # Long-query cutoff relaxation must still fire here.
                # _compute_adaptive_rrf_params owns the cutoff for
                # >50-word queries, but it's gated on adaptive_rrf which
                # we're about to turn off. Without this, ArguAna-style
                # long queries collapse to ~0 NDCG in vec_only mode
                # (best_dist ~0.2 in cosine, default cutoff 1.3225 yields
                # threshold ~0.265, rejecting nearly every candidate
                # past rank 0). Mirror the hybrid long-query relaxation
                # so vec_only ablations are honest, and gate it on
                # adaptive_rrf so an explicit adaptive_rrf=False from
                # the caller still opts out (parity with hybrid).
                if adaptive_rrf and len(query_text.split()) > _ADAPTIVE_RRF_LONG_QUERY:
                    distance_cutoff = _LONG_QUERY_DISTANCE_CUTOFF
                adaptive_rrf = False

            # Adaptive RRF: compute weights from query characteristics (IDF + length)
            # Skip if caller provided explicit weights
            if adaptive_rrf and vec_weight is None and fts_weight is None:
                vec_weight, fts_weight, distance_cutoff = self._compute_adaptive_rrf_params(
                    query_text, default_cutoff=distance_cutoff
                )
            vec_weight, fts_weight = self._resolve_rrf_weights(vec_weight, fts_weight)

            # Adaptive candidate pool — avoid pulling half the corpus on small DBs
            effective_k = top_k
            total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            candidate_pool = min(effective_k * 10, max(effective_k * 3, total_chunks // 3))

            # --- Build metadata filter ---
            vec_clause, col_clause, filter_params = self._build_doc_filter(
                collection=collection,
                project=project,
                layer=layer,
                added_after=added_after,
                added_before=added_before,
            )

            # --- Vector search ---
            vec_rows: list = []
            if _fts_only:
                # #152: skip vector ANN entirely. No candidates, no
                # distances, no distance-cutoff work downstream. Set
                # last_best_distance to the worst possible so the
                # distance-based relevance tier renders "low" — the
                # caller opted out of the vector signal.
                self.last_best_distance = 2.0
                if track_target is not None and _tracer is not None:
                    # Record vector_search as *passed* (not failed) so
                    # the downstream "both generators missed → invisible
                    # to RRF" logic in miss_analysis does not
                    # misattribute the drop to rrf_fusion.  The detail
                    # string makes the intent explicit: the stage was
                    # intentionally skipped by the caller, not that it
                    # ran and found nothing.
                    _tracer.record(
                        "vector_search",
                        passed=True,
                        detail="retrieval_mode='fts_only': vector search intentionally skipped by caller",
                    )
                    _tracer.record(
                        "distance_cutoff",
                        passed=True,
                        detail="retrieval_mode='fts_only': no distance cutoff applied",
                    )
            elif self._snap is not None and len(self._snap) > 0:
                # The two snapvec backends disagree on their return
                # semantics and this shared call site has to reconcile
                # them to the cosine-distance contract the rest of
                # the pipeline assumes (#289).
                #
                # - Flat ``snapvec`` (``SnapIndex``) returns
                #   ``(id, similarity)`` in ``[-1, 1]`` sorted
                #   descending and needs the ``1 - similarity``
                #   conversion here.
                # - ``snapvec-ivfpq`` (``IVFPQBackend``) already
                #   applies ``1 - similarity`` inside its own
                #   ``search()`` (see ``vectorbackend/snapvec_ivfpq.py``) and
                #   must not be double-inverted.
                #
                # ``min / max`` clamps keep the invariant that
                # ``last_best_distance`` and ``relevance_tier`` both
                # assume (``[0, 2]``), even if an implementation
                # quirk yields a value slightly outside ``[-1, 1]``.
                snap_results = self._snap.search(
                    np.array(query_embedding, dtype=np.float32), k=candidate_pool
                )
                snap_ids = [int(r[0]) for r in snap_results]
                if self._vector_backend == "snapvec":
                    snap_dists = {
                        int(r[0]): min(2.0, max(0.0, 1.0 - float(r[1]))) for r in snap_results
                    }
                else:
                    snap_dists = {int(r[0]): min(2.0, max(0.0, float(r[1]))) for r in snap_results}

                if snap_ids:
                    placeholders = ",".join("?" * len(snap_ids))
                    # Build filter clause adapted for snapvec (no v.rowid)
                    snap_filter = vec_clause.replace("v.rowid", "c.id") if vec_clause else ""
                    rows = self._conn.execute(
                        f"""
                        SELECT c.id, c.text, d.title, d.path, c.seq, d.added_at, d.collection, d.tags, d.layer
                        FROM chunks c
                        JOIN documents d ON d.id = c.doc_id
                        WHERE c.id IN ({placeholders})
                          {snap_filter}
                        """,
                        [*snap_ids, *filter_params],
                    ).fetchall()
                    # Build dict for fast lookup, attach distance, preserve snap order
                    row_by_id = {r["id"]: dict(r) for r in rows}
                    vec_rows = []
                    for sid in snap_ids:
                        if sid in row_by_id:
                            entry = row_by_id[sid]
                            entry["distance"] = snap_dists.get(sid, 2.0)
                            vec_rows.append(entry)
                else:
                    vec_rows = []
            else:
                vec_rows = self._conn.execute(
                    f"""
                    SELECT c.id, c.text, d.title, d.path, c.seq, v.distance, d.added_at, d.collection, d.tags, d.layer
                    FROM vec_chunks v
                    JOIN chunks c ON c.id = v.rowid
                    JOIN documents d ON d.id = c.doc_id
                    WHERE v.embedding MATCH ?
                      AND k = ?
                      {vec_clause}
                    ORDER BY v.distance
                    """,
                    [_serialize(query_embedding), candidate_pool, *filter_params],
                ).fetchall()

            # --- Track: vector search stage ---
            # In fts_only mode the "vector_search" and "distance_cutoff"
            # stages were already recorded as skipped above; don't
            # overwrite them with a generic "not found" verdict.
            if track_target is not None and not _fts_only:
                target_vec_rank: int | None = None
                target_vec_distance: float | None = None
                for i, row in enumerate(vec_rows):
                    if int(row["id"]) == track_target:
                        target_vec_rank = i
                        target_vec_distance = float(row["distance"])
                        break
                if target_vec_rank is not None:
                    _tracer.record(
                        "vector_search",
                        passed=True,
                        rank=target_vec_rank,
                        score=target_vec_distance,
                        detail=(
                            f"Found in vector candidate pool at rank {target_vec_rank + 1}/{len(vec_rows)} "
                            f"with distance {target_vec_distance:.4f}"
                        ),
                    )
                else:
                    _tracer.record(
                        "vector_search",
                        passed=False,
                        rank=None,
                        score=None,
                        detail=(
                            f"Not in vector candidate pool of size {candidate_pool}. "
                            "The chunk's embedding is too far from the query embedding."
                        ),
                        counterfactual="Would need candidate_pool > current rank to appear",
                    )

            # --- Filter by vector distance gap ---
            # The best (closest) result has the smallest distance.
            # Remove results that are semantically too far from the ideal match.
            if vec_rows:
                best_distance = float(vec_rows[0]["distance"])
                self.last_best_distance = best_distance
                # Capture target distance BEFORE filtering for tracking
                target_dist_before: float | None = None
                if track_target is not None:
                    for r in vec_rows:
                        if int(r["id"]) == track_target:
                            target_dist_before = float(r["distance"])
                            break
                threshold = best_distance * distance_cutoff
                cutoff_applied = best_distance > 0
                if cutoff_applied:
                    vec_rows = [r for r in vec_rows if float(r["distance"]) <= threshold]
                # Track distance cutoff verdict whenever we have a before-distance
                # (i.e., the target was in vec_rows pre-filter)
                if track_target is not None and target_dist_before is not None:
                    target_in_after = any(int(r["id"]) == track_target for r in vec_rows)
                    if not cutoff_applied:
                        # best_distance == 0: cutoff logic is skipped entirely.
                        # Surface the target's absolute distance so users know
                        # whether the bypass is a lucky rescue or a "carried by
                        # the loophole" situation (high-distance chunks are
                        # getting through because a perfect match exists).
                        _tracer.record(
                            "distance_cutoff",
                            passed=True,
                            score=target_dist_before,
                            detail=(
                                f"Distance cutoff bypassed (best_distance=0, perfect "
                                f"match exists). Target distance={target_dist_before:.4f}; "
                                f"this would normally require cutoff ratio > "
                                f"{target_dist_before * 100:.1f} to pass."
                            ),
                        )
                    elif target_in_after:
                        _tracer.record(
                            "distance_cutoff",
                            passed=True,
                            score=target_dist_before,
                            detail=(
                                f"Distance {target_dist_before:.4f} ≤ threshold "
                                f"{threshold:.4f} (best={best_distance:.4f} × cutoff={distance_cutoff:.2f})"
                            ),
                        )
                    else:
                        needed_cutoff = (
                            target_dist_before / best_distance
                            if best_distance > 0
                            else float("inf")
                        )
                        _tracer.record(
                            "distance_cutoff",
                            passed=False,
                            score=target_dist_before,
                            detail=(
                                f"Distance {target_dist_before:.4f} > threshold "
                                f"{threshold:.4f} (best={best_distance:.4f} × cutoff={distance_cutoff:.2f})"
                            ),
                            counterfactual=(
                                f"Would have passed with distance_cutoff ≥ {needed_cutoff:.2f}"
                            ),
                        )
            else:
                self.last_best_distance = 2.0  # max cosine distance = worst case

            # Track which chunk IDs passed the vector distance filter
            relevant_chunk_ids: set[int] = {row["id"] for row in vec_rows}

            # --- Explain: capture per-chunk vector rank and distance ---
            _explain_vec: dict[int, tuple[int, float]] = {}  # chunk_id -> (rank, distance)
            if explain:
                for rank, row in enumerate(vec_rows):
                    _explain_vec[int(row["id"])] = (rank, float(row["distance"]))

            # --- FTS5 search ---
            words = query_text.split()
            safe_query, quoted_words = self._build_fts_match_query(query_text, words)
            if _vec_only:
                # #275: caller asked to bypass keyword search entirely.
                # Produce no FTS rows; downstream RRF scoring uses
                # weights (1.0, 0.0) so the absence cannot re-weight
                # anything. Record the stage in the tracer as an
                # intentional skip (not a miss) for parity with the
                # fts_only path above.
                fts_rows = []
                if track_target is not None and _tracer is not None:
                    _tracer.record(
                        "fts_search",
                        passed=True,
                        detail="retrieval_mode='vec_only': FTS5 search intentionally skipped by caller",
                    )
            else:
                try:
                    fts_rows = self._conn.execute(
                        f"""
                        SELECT c.id, c.text, d.title, d.path, c.seq,
                               rank as fts_rank, d.added_at, d.collection, d.tags, d.layer
                        FROM fts_chunks f
                        JOIN chunks c ON c.id = f.rowid
                        JOIN documents d ON d.id = c.doc_id
                        WHERE fts_chunks MATCH ?
                          {col_clause}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        [safe_query, *filter_params, candidate_pool],
                    ).fetchall()
                except sqlite3.OperationalError:
                    # FTS5 query syntax error (e.g. single char) — fall back to no FTS
                    fts_rows = []

                # --- Track: FTS search stage ---
                if track_target is not None:
                    target_fts_rank: int | None = None
                    for i, row in enumerate(fts_rows):
                        if int(row["id"]) == track_target:
                            target_fts_rank = i
                            break
                    stemmed_terms = self._stem_terms(words) if words else []
                    if target_fts_rank is not None:
                        _tracer.record(
                            "fts_search",
                            passed=True,
                            rank=target_fts_rank,
                            detail=(
                                f"Matched FTS at rank {target_fts_rank + 1}/{len(fts_rows)}. "
                                f"Stemmed query terms: {stemmed_terms}"
                            ),
                        )
                    else:
                        _tracer.record(
                            "fts_search",
                            passed=False,
                            detail=(
                                f"Did not match FTS5. Stemmed query terms: {stemmed_terms}. "
                                "The chunk text does not contain any of these stems "
                                "(after porter stemming)."
                            ),
                            counterfactual="Would need a query containing words from the chunk's vocabulary",
                        )

            # --- Adaptive fallback: vector pool empty (#156) ---
            # If the vector pool is empty — either the initial ANN scan
            # returned nothing, the metadata filter eliminated everything,
            # or the distance cutoff was so tight that no chunk survived
            # — the pipeline would otherwise fuse an empty vec list with
            # the FTS list using the default or adaptive RRF weights
            # (e.g. 0.6/0.4). The FTS rows would then be scored with
            # only ``fts_weight * 1/(k+rank)``, which degrades the
            # absolute RRF score unnecessarily — a literal-match FTS
            # hit at rank 0 gets ~0.0067 instead of the 0.0167 it would
            # earn under pure FTS weighting.  Detect the condition and
            # collapse to FTS-only scoring explicitly, so downstream
            # consumers of ``SearchResult.score`` see a meaningful value
            # and the relevance-tier signal correctly reports "low"
            # instead of carrying a stale high-confidence distance.
            #
            # First, always reset ``last_best_distance`` when vec_rows
            # is empty — whether or not there are FTS results. Without
            # this, a query where both pools are empty (e.g. corpus
            # doesn't contain the query at all) would report whatever
            # ``last_best_distance`` held from a previous query,
            # lying about the current query's confidence.  Flagged by
            # Gemini review on #156.
            if not vec_rows:
                self.last_best_distance = 2.0

            # Then, if there are FTS results to boost, actually apply
            # the weight override and fire the observability signal.
            if not vec_rows and fts_rows:
                registry.counter_inc("adaptive_rrf_vector_empty_fallback_total")
                vec_weight = 0.0
                fts_weight = 1.0
                if track_target is not None and _tracer is not None:
                    _tracer.record(
                        "adaptive_fallback",
                        passed=True,
                        detail=(
                            "vector pool empty after distance cutoff: collapsed "
                            "to FTS-only scoring (vec_weight=0.0, fts_weight=1.0)"
                        ),
                    )

            # --- Reciprocal Rank Fusion ---
            scores, _explain_rrf_vec, _explain_rrf_fts, _explain_fts_rank = self._fuse_rrf_scores(
                vec_rows=vec_rows,
                fts_rows=fts_rows,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
                relevant_chunk_ids=relevant_chunk_ids,
                effective_k=effective_k,
                fts_only=_fts_only,
                explain=explain,
            )

            # Sort by RRF score descending
            ranked = sorted(scores.values(), key=lambda x: float(x["rrf"]), reverse=True)

            # --- Track: RRF fusion stage ---
            if track_target is not None:
                target_rrf_rank: int | None = None
                target_rrf_score: float | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_rrf_rank = i
                        target_rrf_score = float(r["rrf"])
                        break
                if target_rrf_rank is not None:
                    _tracer.record(
                        "rrf_fusion",
                        passed=True,
                        rank=target_rrf_rank,
                        score=target_rrf_score,
                        detail=(
                            f"After RRF fusion: rank {target_rrf_rank + 1}/{len(ranked)}, "
                            f"combined score {target_rrf_score:.6f}"
                        ),
                    )
                else:
                    _tracer.record(
                        "rrf_fusion",
                        passed=False,
                        detail=(
                            "Eliminated before RRF fusion (failed both vector and FTS, "
                            "or filtered by metadata)"
                        ),
                    )

            # --- Recency boost (temporal decay) ---
            # Biases scores toward recent content, useful for agentic
            # memory where latest context tends to be most relevant.
            ranked = self._apply_recency_boost(ranked, recency_boost)

            # --- Track: recency boost stage ---
            if track_target is not None and recency_boost > 0:
                target_after_boost: int | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_after_boost = i
                        break
                if target_after_boost is not None:
                    _tracer.record(
                        "recency_boost",
                        passed=True,
                        rank=target_after_boost,
                        detail=(
                            f"After recency_boost={recency_boost}: rank {target_after_boost + 1}"
                        ),
                    )

            # --- Pre-MMR ranks (for tracking what MMR removes) ---
            pre_mmr_rank_of_target: int | None = None
            if track_target is not None:
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        pre_mmr_rank_of_target = i
                        break

            # Intra-document MMR deduplication: allow multiple chunks from the
            # same document only when they are semantically diverse (e.g. different
            # chapters of a book).  Chunks from different documents compete purely
            # on score — no cross-document penalty.
            ranked = self._mmr_dedup(ranked, top_k, mmr_lambda=mmr_lambda, _explain=explain)

            # --- Track: MMR + top_k cutoff ---
            if track_target is not None:
                target_final_rank: int | None = None
                for i, r in enumerate(ranked):
                    if int(r["id"]) == track_target:
                        target_final_rank = i
                        break
                # MMR verdict
                if pre_mmr_rank_of_target is not None and target_final_rank is None:
                    _tracer.record(
                        "mmr_dedup",
                        passed=False,
                        rank=pre_mmr_rank_of_target,
                        detail=(
                            f"Was rank {pre_mmr_rank_of_target + 1} pre-MMR but eliminated by "
                            "intra-document MMR deduplication (another chunk from the same "
                            f"document was already selected and they're too similar). "
                            f"Current mmr_lambda={mmr_lambda:.2f}."
                        ),
                        counterfactual=(
                            f"Try a higher mmr_lambda (>{mmr_lambda:.2f}) for less "
                            "aggressive dedup, or mmr_lambda=1.0 to disable MMR entirely."
                        ),
                    )
                elif target_final_rank is not None:
                    _tracer.record(
                        "mmr_dedup",
                        passed=True,
                        rank=target_final_rank,
                        detail=f"Survived MMR dedup at rank {target_final_rank + 1}",
                    )
                # top_k cutoff verdict
                if target_final_rank is not None:
                    if target_final_rank < top_k:
                        _tracer.record(
                            "top_k_cutoff",
                            passed=True,
                            rank=target_final_rank,
                            detail=f"Final rank {target_final_rank + 1} ≤ top_k={top_k}",
                        )
                    else:
                        _tracer.record(
                            "top_k_cutoff",
                            passed=False,
                            rank=target_final_rank,
                            detail=f"Final rank {target_final_rank + 1} > top_k={top_k}",
                            counterfactual=f"Would appear with top_k ≥ {target_final_rank + 1}",
                        )

            results = self._build_search_results(
                ranked,
                explain=explain,
                explain_vec=_explain_vec,
                explain_rrf_vec=_explain_rrf_vec,
                explain_rrf_fts=_explain_rrf_fts,
                explain_fts_rank=_explain_fts_rank,
                quoted_words=quoted_words,
                vec_weight=vec_weight,
                fts_weight=fts_weight,
            )

            # Exact-match text filter (#106). Applied post-pipeline on
            # the final ranked list: a chunk that otherwise would have
            # made the top-k is dropped unless its ``text`` contains
            # ``exact_match`` as a substring. Bypasses FTS5 tokenization
            # so identifiers / phrases that stem or tokenize differently
            # than the user typed survive -- e.g. ``exact_match="rate-
            # limit"`` requires the literal string, not ``rate limit``.
            #
            # Case-sensitive opt-in via ``exact_match_case_sensitive``.
            # Default is case-insensitive which is the usual UX
            # expectation for retrieval filters.
            #
            # Post-filter means the candidate pool upstream was sized
            # without knowing about this filter, so callers who need a
            # guaranteed top_k under a selective substring should pass a
            # larger ``top_k`` (e.g. 3x) or accept a smaller result set.
            if exact_match:
                if exact_match_case_sensitive:
                    results = [r for r in results if exact_match in r.text]
                else:
                    needle = exact_match.casefold()
                    results = [r for r in results if needle in r.text.casefold()]

            # Stash values the finally block needs to log slow queries
            # with accurate data.
            _result_count = len(results)
            _vec_weight_observed = float(vec_weight)
            _fts_weight_observed = float(fts_weight)

            # _tracer.verdicts has been populated in-place by the record()
            # calls above when tracking is enabled.  No store-level state.

            if _cache_key is not None:
                with self._cache_lock:
                    self._query_cache[_cache_key] = list(results)
                    while len(self._query_cache) > _cache_max:
                        self._query_cache.popitem(last=False)

            return results
        finally:
            # Record latency and slow-query telemetry for every search,
            # including those that raised.  _result_count is 0 if we
            # failed before reaching the results-building block.
            elapsed_ms = (time.perf_counter() - _search_start) * 1000.0
            registry.histogram_observe("search_latency_ms", elapsed_ms)
            if elapsed_ms >= self._observability.slow_query_ms:
                registry.counter_inc("slow_queries_total")
                # Sanitize query preview: strip control characters and
                # non-printable codepoints so log shippers don't choke
                # on embedded NULs or escape sequences.
                preview = "".join(c if c.isprintable() else "?" for c in query_text[:60])
                if len(query_text) > 60:
                    preview += "…"
                logger.warning(
                    "slow query: %.1fms query=%s results=%d vec_w=%.2f fts_w=%.2f",
                    elapsed_ms,
                    preview,
                    _result_count,
                    _vec_weight_observed,
                    _fts_weight_observed,
                )

    # ------------------------------------------------------------------ #
    # Miss analysis (#108) — explain why a doc did NOT appear in top-k     #
    # ------------------------------------------------------------------ #

    def miss_analysis(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        expected_path: str | None = None,
        expected_chunk_id: int | None = None,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> MissAnalysis:
        """Diagnose why an expected document did not appear in search results.

        Runs the same search pipeline as ``search()`` but with per-stage
        tracking enabled, then builds a structured ``MissAnalysis`` with
        the trace, the actual top-k for context, and rule-based
        suggestions.

        Args:
            query_embedding: Query vector (same as ``search()``).
            query_text: Raw query text (same as ``search()``).
            expected_path: Path of the document the caller expected to see.
                Either this or ``expected_chunk_id`` must be provided.
            expected_chunk_id: Specific chunk id to track instead of
                resolving from a path.  Useful when the caller already has
                a chunk id from a previous search.
            top_k: Number of results to evaluate (same as ``search()``).
            collection/project/layer: Same metadata filters as ``search()``.

        Returns:
            ``MissAnalysis`` with stage verdicts, actual top-k, and
            suggestions.

        Raises:
            ValueError: if neither ``expected_path`` nor ``expected_chunk_id``
                is given, or the path/id resolves to nothing.
        """
        if expected_path is None and expected_chunk_id is None:
            raise ValueError("Provide either expected_path or expected_chunk_id")

        # Resolve target chunk id, tracking how the choice was made so the
        # caller knows whether the trace represents the whole document or
        # just its best-matching chunk.
        target_chunk_id: int
        resolved_path: str | None = expected_path
        target_resolution: str
        total_chunks_in_doc = 1

        if expected_chunk_id is not None:
            row = self._conn.execute(
                "SELECT c.id, d.path, d.id AS doc_id "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
                [int(expected_chunk_id)],
            ).fetchone()
            if row is None:
                raise ValueError(f"Chunk id {expected_chunk_id} not found")
            target_chunk_id = int(row["id"])
            resolved_path = str(row["path"])
            target_resolution = "explicit_id"
            # Count siblings in the same document for caller transparency
            sibling_count = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", [row["doc_id"]]
            ).fetchone()[0]
            total_chunks_in_doc = int(sibling_count)
        else:
            # Apply the same metadata filters used by the search pipeline
            # so that miss_analysis("rate limits", expected_path="/x.md",
            # collection="docs") resolves to the chunk(s) of /x.md that
            # belong to collection="docs", not a cross-collection copy.
            filter_conditions, filter_params = self._get_filter_conditions(
                "d",
                collection=collection,
                project=project,
                layer=layer,
            )
            where_extras = ""
            if filter_conditions:
                where_extras = " AND " + " AND ".join(filter_conditions)
            doc_chunks = self._conn.execute(
                f"""
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.path = ?
                  {where_extras}
                """,
                [expected_path, *filter_params],
            ).fetchall()
            if not doc_chunks:
                # Be helpful: differentiate "path exists but filtered out"
                # from "path truly not in the store".
                exists_elsewhere = self._conn.execute(
                    "SELECT 1 FROM documents WHERE path = ? LIMIT 1", [expected_path]
                ).fetchone()
                if exists_elsewhere is not None:
                    raise ValueError(
                        f"Path {expected_path!r} exists but is excluded by the "
                        "collection/project/layer filters passed to miss_analysis(). "
                        "Re-run without filters to diagnose."
                    )
                raise ValueError(f"No chunks found for path: {expected_path}")
            chunk_ids = [int(r["id"]) for r in doc_chunks]
            total_chunks_in_doc = len(chunk_ids)

            if total_chunks_in_doc == 1:
                target_chunk_id = chunk_ids[0]
                target_resolution = "only_chunk"
            else:
                # Multi-chunk document: pick the chunk with the smallest
                # distance to the query.  This biases the trace toward
                # the best-case chunk; target_resolution reflects that.
                #
                # Batch the IN clause to respect SQLITE_LIMIT_VARIABLE_NUMBER
                # (default 999).  Books / long manuals can have 1000+ chunks,
                # which would otherwise blow the limit.
                best_rowid: int | None = None
                best_dist: float = float("inf")
                q_ser = _serialize(query_embedding)
                try:
                    for start in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
                        batch = chunk_ids[start : start + _SQLITE_PARAM_BATCH]
                        placeholders = ",".join("?" * len(batch))
                        rows = self._conn.execute(
                            f"""
                            SELECT v.rowid, v.distance
                            FROM vec_chunks v
                            WHERE v.embedding MATCH ?
                              AND v.rowid IN ({placeholders})
                              AND k = ?
                            ORDER BY v.distance
                            LIMIT 1
                            """,
                            [q_ser, *batch, len(batch)],
                        ).fetchall()
                        if rows and float(rows[0]["distance"]) < best_dist:
                            best_dist = float(rows[0]["distance"])
                            best_rowid = int(rows[0]["rowid"])
                except sqlite3.Error:
                    best_rowid = None
                if best_rowid is not None:
                    target_chunk_id = best_rowid
                else:
                    logging.getLogger(__name__).warning(
                        "miss_analysis: vec lookup failed for multi-chunk doc "
                        "'%s'; falling back to first chunk id",
                        expected_path,
                    )
                    target_chunk_id = chunk_ids[0]
                target_resolution = "best_of_n"

        # Run search with a caller-owned tracer — the buffer is local
        # to this call, so concurrent miss_analysis() on the same store
        # cannot corrupt each other.
        tracer = _PipelineTracer(target_chunk_id)
        results = self.search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            _tracer=tracer,
        )

        # Determine if the expected doc appeared in results
        appeared = False
        final_rank: int | None = None
        for i, r in enumerate(results):
            if r.chunk_id == target_chunk_id or (
                resolved_path is not None and r.path == resolved_path
            ):
                appeared = True
                final_rank = i
                break

        # Build StageVerdict list from this caller's verdicts
        stage_verdicts = [
            StageVerdict(
                stage=v["stage"],  # type: ignore[arg-type]
                passed=bool(v["passed"]),
                rank=v["rank"],  # type: ignore[arg-type]
                score=v["score"],  # type: ignore[arg-type]
                detail=str(v["detail"]),
                counterfactual=v["counterfactual"],  # type: ignore[arg-type]
            )
            for v in tracer.verdicts
        ]

        # Find the stage that actually eliminated the chunk from the
        # pipeline.  Tricky: vector_search and fts_search are INDEPENDENT
        # candidate generators — failing one does NOT drop the chunk,
        # because the other modality can still surface it into RRF.
        # Only the gate stages (distance_cutoff, rrf_fusion, mmr_dedup,
        # top_k_cutoff) actually remove a chunk from the pipeline.
        #
        # Special case: if BOTH vector_search and fts_search failed, the
        # chunk never reached RRF and the functional drop_at is "rrf_fusion"
        # (it's invisible to the fusion layer).
        _GATE_STAGES = {"distance_cutoff", "rrf_fusion", "mmr_dedup", "top_k_cutoff"}
        dropped_at: str | None = None
        by_stage = {v.stage: v for v in stage_verdicts}
        vec_failed = "vector_search" in by_stage and not by_stage["vector_search"].passed
        fts_failed = "fts_search" in by_stage and not by_stage["fts_search"].passed
        if vec_failed and fts_failed:
            # Both generators missed — chunk is invisible to RRF.
            dropped_at = "rrf_fusion"
        else:
            # Otherwise, the first gate stage that failed is the drop point.
            for v in stage_verdicts:
                if v.stage in _GATE_STAGES and not v.passed:
                    dropped_at = v.stage
                    break

        # Actual top-k for context
        actual_top_k = [
            MissAnalysisActualResult(
                rank=i,
                chunk_id=r.chunk_id,
                path=r.path,
                title=r.title,
                score=r.score,
            )
            for i, r in enumerate(results)
        ]

        # Generate rule-based suggestions
        metadata_filtered = bool(collection or project or layer)
        suggestions = self._build_miss_suggestions(
            stage_verdicts,
            dropped_at,
            appeared,
            final_rank=final_rank,
            top_k=top_k,
            metadata_filtered=metadata_filtered,
        )

        return MissAnalysis(
            query=query_text,
            expected_path=resolved_path,
            expected_chunk_id=target_chunk_id,
            target_resolution=target_resolution,  # type: ignore[arg-type]
            total_chunks_in_doc=total_chunks_in_doc,
            top_k_requested=top_k,
            appeared_in_results=appeared,
            final_rank=final_rank,
            dropped_at=dropped_at,
            stage_verdicts=stage_verdicts,
            actual_top_k=actual_top_k,
            suggestions=suggestions,
        )

    @staticmethod
    def _build_miss_suggestions(
        stage_verdicts: list[StageVerdict],
        dropped_at: str | None,
        appeared: bool,
        final_rank: int | None = None,
        top_k: int = 5,
        metadata_filtered: bool = False,
    ) -> list[str]:
        """Map stage failure modes to actionable suggestion strings.

        Pure rule-based, no LLM calls.  Each rule corresponds to a
        specific reason a chunk could fall out of the pipeline.
        Note: the ``recency_boost`` stage is intentionally absent from
        this map because it cannot drop a chunk — it only re-ranks.
        """
        suggestions: list[str] = []
        by_stage = {v.stage: v for v in stage_verdicts}

        if appeared:
            # Near-miss: the doc appeared but in the bottom of top-k.
            # Users passing --miss for a doc at rank 4/5 are asking
            # "why is this not where I expected?", not "is it there?".
            if final_rank is not None and top_k > 0 and final_rank >= top_k * 0.6:
                return [
                    f"The expected document IS in top-k but at rank {final_rank + 1}/{top_k} "
                    "(near-miss tier). It's competitive but being outranked by other chunks. "
                    "Consider a more specific query, or raise top_k to get more headroom."
                ]
            return ["The expected document IS in the top-k. No miss to analyze."]

        if dropped_at == "vector_search":
            suggestions.append(
                "The chunk's embedding is semantically far from the query. "
                "Try reformulating using vocabulary that appears in the target document, "
                "or increase the candidate pool size."
            )
            if "fts_search" in by_stage and by_stage["fts_search"].passed:
                suggestions.append(
                    "FTS5 keyword search DID find this chunk — consider raising fts_weight "
                    "or relying on keyword matching for this query type."
                )

        if dropped_at == "distance_cutoff":
            cf = by_stage["distance_cutoff"].counterfactual
            cf_clause = f" {cf}." if cf else ""
            suggestions.append(
                "Distance cutoff dropped the chunk just past threshold."
                f"{cf_clause} Pass distance_cutoff=2.0 (or higher) to relax the cutoff."
            )
            # Rescue hint: if FTS already matched, raising fts_weight
            # works even without touching the cutoff.
            if "fts_search" in by_stage and by_stage["fts_search"].passed:
                suggestions.append(
                    "FTS5 already matched this chunk — raising fts_weight would rescue it "
                    "without needing to relax the distance cutoff."
                )

        if dropped_at == "fts_search":
            v = by_stage["fts_search"]
            suggestions.append(
                f"FTS5 keyword search did not match. {v.detail} "
                "Either reformulate the query with words that appear in the chunk, "
                "or rely entirely on vector search by setting fts_weight=0."
            )

        if dropped_at == "rrf_fusion":
            vec_failed = "vector_search" in by_stage and not by_stage["vector_search"].passed
            fts_failed = "fts_search" in by_stage and not by_stage["fts_search"].passed
            if vec_failed and fts_failed:
                if metadata_filtered:
                    suggestions.append(
                        "Most likely the metadata filter (collection/project/layer) excluded "
                        "this document.  Re-run miss_analysis without filters to confirm."
                    )
                else:
                    suggestions.append(
                        "The chunk is invisible to BOTH vector and FTS search. Possible causes: "
                        "(1) the document was indexed with a different embedding model than the "
                        "current one — try `vstash reindex`; (2) the chunk text is empty or "
                        "contains only stopwords; (3) the query is semantically and lexically "
                        "unrelated to the chunk."
                    )
            else:
                suggestions.append(
                    "The chunk was eliminated before RRF fusion.  "
                    "Inspect the vector_search and fts_search verdicts for detail."
                )

        if dropped_at == "mmr_dedup":
            cf = by_stage["mmr_dedup"].counterfactual
            cf_clause = f" {cf}" if cf else ""
            suggestions.append(
                "MMR deduplication eliminated this chunk because another chunk from the "
                f"same document was already selected and they're too similar.{cf_clause}"
            )

        if dropped_at == "top_k_cutoff":
            cf = by_stage["top_k_cutoff"].counterfactual
            cf_clause = f" {cf}" if cf else ""
            suggestions.append(
                f"The chunk survived all stages but was just below the top_k cutoff.{cf_clause}"
            )

        if not suggestions:
            suggestions.append(
                "Unable to localize the failure to a single stage. "
                "Inspect stage_verdicts manually for details."
            )

        return suggestions

    # ------------------------------------------------------------------ #
    # MMR intra-document deduplication                                      #
    # ------------------------------------------------------------------ #

    def _mmr_dedup(
        self,
        ranked: list[dict[str, str | int | float]],
        top_k: int,
        mmr_lambda: float,
        _explain: bool = False,
    ) -> list[dict[str, str | int | float]]:
        """Select top-k results using intra-document MMR diversity.

        Chunks from *different* documents compete purely on score.  When
        multiple chunks from the *same* document appear, the second (and
        subsequent) chunks are penalised by their cosine similarity to the
        already-selected chunk(s) from that document.

        This allows two distant chapters of a book to both appear in results,
        while still preventing near-duplicate chunks from flooding top-k.

        When ``mmr_lambda = 1.0`` this degrades to hard per-document dedup
        (at most one chunk per document, highest-scoring wins).  The method
        also falls back to hard dedup if embedding lookup fails.
        """
        if not ranked:
            return []

        # Fast path: if mmr_lambda == 1.0, no diversity penalty — just dedup.
        # Also fast-path when there are no same-doc duplicates.
        from collections import Counter

        doc_counts = Counter(str(r["path"]) for r in ranked)
        has_duplicates = any(c > 1 for c in doc_counts.values())

        if mmr_lambda >= 1.0 or not has_duplicates:
            # Hard dedup: keep first (highest-scoring) chunk per document.
            seen: set[str] = set()
            deduped: list[dict[str, str | int | float]] = []
            for r in ranked:
                doc_key = str(r["path"])
                if doc_key not in seen:
                    seen.add(doc_key)
                    deduped.append(r)
            return deduped[:top_k]

        # --- Fetch embeddings for chunks with same-doc duplicates ---
        dup_doc_paths = {p for p, c in doc_counts.items() if c > 1}
        dup_ids = [int(r["id"]) for r in ranked if str(r["path"]) in dup_doc_paths]

        embeddings: dict[int, list[float]] = {}
        if dup_ids:
            placeholders = ",".join("?" * len(dup_ids))
            try:
                rows = self._conn.execute(
                    f"SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({placeholders})",
                    dup_ids,
                ).fetchall()
                for row in rows:
                    embeddings[row["rowid"]] = _deserialize(row["embedding"])
            except sqlite3.Error:
                logging.getLogger(__name__).warning(
                    "MMR embedding fetch failed — falling back to hard dedup. "
                    "Results may be less diverse.",
                    exc_info=True,
                )
                # Fallback: hard dedup
                seen_fb: set[str] = set()
                deduped_fb: list[dict[str, str | int | float]] = []
                for r in ranked:
                    doc_key = str(r["path"])
                    if doc_key not in seen_fb:
                        seen_fb.add(doc_key)
                        deduped_fb.append(r)
                return deduped_fb[:top_k]

        # --- Greedy MMR selection ---
        # Normalise scores to [0, 1] for MMR balancing.
        #
        # All current RRF / fusion paths store the score under the
        # "rrf" key. The old scoring pipeline used "final_score" and
        # was removed in v0.18.0 (#109); the dead fallback it left
        # behind here was a maintenance liability that made this loop
        # harder to reason about during the #153 audit. Removed.
        scores = [float(r["rrf"]) for r in ranked]
        s_min, s_max = min(scores), max(scores)
        s_range = s_max - s_min if s_max > s_min else 1.0

        selected: list[dict[str, str | int | float]] = []
        remaining = list(range(len(ranked)))
        in_remaining = [True] * len(ranked)

        # Pre-compute normalized scores once (#167).  The value of
        # ``(score - s_min) / s_range`` is invariant across the outer
        # top_k loop — it depends only on each candidate's static score
        # — so recomputing it inside the inner ``for idx in remaining``
        # loop costs O(N * top_k) pure-Python ops for no benefit.
        # Hoisting it cuts the inner loop complexity to O(N) and
        # reuses the ``scores`` list already computed above, avoiding
        # the extra dict lookup and float conversion that a naive
        # hoist would do (flagged in the #167 review).
        norm_scores = [(s - s_min) / s_range for s in scores]

        # Precompute invariant MMR relevance terms to avoid O(K * N) recalculations
        relevance_terms = [mmr_lambda * ns for ns in norm_scores]
        penalty_multiplier = 1.0 - mmr_lambda

        # Extract values into fast lists for index-based access
        doc_keys = [str(r["path"]) for r in ranked]
        chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]

        # Pre-group chunk indices by doc_keys to avoid O(N) linear scans when updating max_sims
        doc_to_indices: dict[str, list[int]] = {}
        for i, doc_key in enumerate(doc_keys):
            if doc_key not in doc_to_indices:
                doc_to_indices[doc_key] = []
            doc_to_indices[doc_key].append(i)

        # Precompute L2 norms for cosine similarity to avoid O(K * N) recomputation.
        chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]

        # Track the maximum similarity to any selected chunk from the *same document*.
        # Replaces O(N * S) recomputation with O(1) lookup + O(N) update.
        max_sims = [0.0] * len(ranked)

        for _ in range(min(top_k, len(ranked))):
            best_idx = -1
            best_mmr = -float("inf")
            best_rem_idx = -1

            for rem_idx, idx in enumerate(remaining):
                max_sim = max_sims[idx]

                mmr_score = relevance_terms[idx] - penalty_multiplier * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
                    best_rem_idx = rem_idx

            if best_idx < 0 or best_mmr < 0:
                # Stop when the best remaining candidate has negative MMR,
                # meaning its redundancy penalty exceeds its relevance.
                break

            chosen = ranked[best_idx]
            if _explain:
                chosen["_mmr_penalty"] = (1 - mmr_lambda) * max_sims[best_idx]
            selected.append(chosen)

            # O(1) swap-with-last removal
            remaining[best_rem_idx] = remaining[-1]
            remaining.pop()
            in_remaining[best_idx] = False

            # Update max_sims for remaining chunks from the same document
            # by comparing against the newly selected embedding.
            new_doc_key = doc_keys[best_idx]
            new_emb = chunk_embs[best_idx]
            new_norm = chunk_norms[best_idx]
            if new_emb is not None:
                for idx in doc_to_indices[new_doc_key]:
                    if in_remaining[idx]:
                        idx_emb = chunk_embs[idx]
                        if idx_emb is not None:
                            sim = _cosine_sim(
                                idx_emb, new_emb, norm_a=chunk_norms[idx], norm_b=new_norm
                            )
                            if sim > max_sims[idx]:
                                max_sims[idx] = sim

        return selected

    # ------------------------------------------------------------------ #
    # Filter builder                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_filter_conditions(
        alias: str = "",
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
        added_after: str | None = None,
        added_before: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Build filter conditions for document metadata.

        Args:
            alias: Table alias prefix (e.g. ``'d.'``, ``'d2.'``, or ``''``).
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by tag (LIKE match within comma-separated tags).
            added_after: ISO date — only documents added on or after this date.
            added_before: ISO date — only documents added before this date.

        Returns:
            Tuple of (condition strings, bind parameters).
        """
        prefix = f"{alias}." if alias else ""
        conditions: list[str] = []
        params: list[str] = []
        if collection:
            conditions.append(f"{prefix}collection = ?")
            params.append(collection)
        if project:
            conditions.append(f"{prefix}project = ?")
            params.append(project)
        if layer:
            conditions.append(f"{prefix}layer = ?")
            params.append(layer)
        if tags:
            conditions.append(f"{prefix}tags LIKE ?")
            params.append(f"%{tags}%")
        if added_after:
            conditions.append(f"{prefix}added_at >= ?")
            params.append(added_after)
        if added_before:
            conditions.append(f"{prefix}added_at < ?")
            params.append(added_before)
        return conditions, params

    @staticmethod
    def _build_doc_filter(
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
        added_after: str | None = None,
        added_before: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Build SQL filter clauses for document metadata.

        Returns three items:
        - **vec_clause**: ``AND v.rowid IN (…)`` for vec0 pre-filtering
        - **col_clause**: ``AND d.collection = ? …`` for JOINed queries (FTS)
        - **params**: bind-parameter values (same order for both clauses)

        Args:
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by tag (LIKE match).
            added_after: ISO date — only documents added on or after this date.
            added_before: ISO date — only documents added before this date.
        """
        conditions_d2, params = VstashStore._get_filter_conditions(
            "d2",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            added_after=added_after,
            added_before=added_before,
        )

        if not conditions_d2:
            return "", "", []

        where = " AND ".join(conditions_d2)
        vec_clause = f"""
            AND v.rowid IN (
                SELECT c2.id
                FROM chunks c2
                JOIN documents d2 ON d2.id = c2.doc_id
                WHERE {where}
            )
        """
        # For JOINed queries where documents is aliased as 'd'
        conditions_d, _ = VstashStore._get_filter_conditions(
            "d",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            added_after=added_after,
            added_before=added_before,
        )
        col_clause = "AND " + " AND ".join(conditions_d)
        return vec_clause, col_clause, params

    # ------------------------------------------------------------------ #
    # Lookup                                                               #
    # ------------------------------------------------------------------ #

    def find_document(
        self,
        query: str,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> str | None:
        """Find a document by partial path or title match.

        Searches for documents where the path or title contains the query
        string (case-insensitive). Returns the path of the first match.

        Args:
            query: Partial filename, path, or title to search for.
            collection: If set, restrict search to this collection.
            project: If set, restrict search to this project.
            layer: If set, restrict search to this layer.

        Returns:
            The full path of the matching document, or None.
        """
        conditions, filter_params = self._get_filter_conditions(
            collection=collection,
            project=project,
            layer=layer,
        )
        extra = ("AND " + " AND ".join(conditions)) if conditions else ""
        row = self._conn.execute(
            f"""SELECT path FROM documents
               WHERE (path LIKE ? OR title LIKE ?) {extra}
               ORDER BY added_at DESC LIMIT 1""",
            [f"%{query}%", f"%{query}%", *filter_params],
        ).fetchone()
        return row["path"] if row else None

    def get_document_chunks(self, path: str, collection: str | None = None) -> list[str]:
        """Get all chunk texts for a document by path.

        Args:
            path: Document path as stored in the database.
            collection: Optional collection filter. If the same path exists in
                multiple collections and no collection is given, returns chunks
                from the most recently added document.

        Returns:
            List of chunk texts ordered by sequence number.
        """
        if collection is not None:
            doc_row = self._conn.execute(
                "SELECT id FROM documents WHERE path = ? AND collection = ? "
                "ORDER BY added_at DESC LIMIT 1",
                [path, collection],
            ).fetchone()
        else:
            doc_row = self._conn.execute(
                "SELECT id FROM documents WHERE path = ? ORDER BY added_at DESC LIMIT 1",
                [path],
            ).fetchone()
        if doc_row is None:
            return []
        rows = self._conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY seq",
            [doc_row["id"]],
        ).fetchall()
        return [row["text"] for row in rows]

    def get_document_added_at(self, paths: list[str]) -> dict[str, str | None]:
        """Look up added_at timestamps for documents by path.

        When the same path exists in multiple collections, returns the most
        recent added_at timestamp.

        Args:
            paths: List of document paths.

        Returns:
            Dict mapping path → added_at ISO string (or None if not found).
        """
        if not paths:
            return {}
        _BATCH = _SQLITE_PARAM_BATCH
        result: dict[str, str | None] = {p: None for p in paths}
        for i in range(0, len(paths), _BATCH):
            batch = paths[i : i + _BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT path, MAX(added_at) AS added_at "
                f"FROM documents WHERE path IN ({placeholders}) "
                f"GROUP BY path",
                batch,
            ).fetchall()
            for row in rows:
                result[row["path"]] = row["added_at"]
        return result

    def get_chunks_for_documents(self, paths: list[str]) -> dict[str, list[str]]:
        """Batch-fetch chunk texts for multiple documents (avoids N+1 queries).

        When the same path exists in multiple collections, returns chunks from
        the most recently added document for that path.

        Args:
            paths: List of document paths.

        Returns:
            Dict mapping path → list of chunk texts ordered by seq.
        """
        if not paths:
            return {}
        _BATCH = _SQLITE_PARAM_BATCH
        result: dict[str, list[str]] = {p: [] for p in paths}
        for i in range(0, len(paths), _BATCH):
            batch = paths[i : i + _BATCH]
            placeholders = ",".join("?" * len(batch))
            # Subquery picks the most recent doc_id per path.
            # MAX(added_at) in SELECT triggers SQLite's bare-column guarantee:
            # the `id` value comes from the row with the maximum added_at.
            rows = self._conn.execute(
                f"SELECT d.path, c.text FROM chunks c "
                f"JOIN documents d ON c.doc_id = d.id "
                f"JOIN ("
                f"  SELECT path, id, MAX(added_at) FROM documents "
                f"  WHERE path IN ({placeholders}) "
                f"  GROUP BY path"
                f") latest ON d.id = latest.id "
                f"ORDER BY d.path, c.seq",
                batch,
            ).fetchall()
            for row in rows:
                result[row["path"]].append(row["text"])
        return result

    def get_chunk(self, chunk_id: int) -> ChunkInfo | None:
        """Retrieve a single chunk by its database row ID.

        Args:
            chunk_id: The integer primary key of the chunk.

        Returns:
            ChunkInfo with chunk text and document metadata, or None if not found.
        """
        row = self._conn.execute(
            "SELECT c.id, c.doc_id, c.seq, c.text, d.title, d.path, d.collection "
            "FROM chunks c JOIN documents d ON c.doc_id = d.id "
            "WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return ChunkInfo(
            chunk_id=int(row["id"]),
            doc_id=row["doc_id"],
            chunk=int(row["seq"]),
            text=row["text"],
            title=row["title"],
            path=row["path"],
            collection=row["collection"],
        )

    def get_chunks(self, chunk_ids: list[int]) -> list[ChunkInfo]:
        """Retrieve multiple chunks by their database row IDs.

        Args:
            chunk_ids: List of integer primary keys.

        Returns:
            List of ChunkInfo in the same order as input IDs.
            Missing IDs are silently skipped.
        """
        if not chunk_ids:
            return []
        _BATCH = (
            _SQLITE_PARAM_BATCH  # stay under SQLite's SQLITE_LIMIT_VARIABLE_NUMBER (default 999)
        )
        lookup: dict[int, ChunkInfo] = {}
        for i in range(0, len(chunk_ids), _BATCH):
            batch = chunk_ids[i : i + _BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT c.id, c.doc_id, c.seq, c.text, d.title, d.path, d.collection "
                f"FROM chunks c JOIN documents d ON c.doc_id = d.id "
                f"WHERE c.id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                lookup[int(row["id"])] = ChunkInfo(
                    chunk_id=int(row["id"]),
                    doc_id=row["doc_id"],
                    chunk=int(row["seq"]),
                    text=row["text"],
                    title=row["title"],
                    path=row["path"],
                    collection=row["collection"],
                )
        return [lookup[cid] for cid in chunk_ids if cid in lookup]

    # ------------------------------------------------------------------ #
    # Inspect                                                              #
    # ------------------------------------------------------------------ #

    def list_documents(
        self,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        added_after: str | None = None,
    ) -> list[DocumentInfo]:
        """List all ingested documents.

        Args:
            collection: If set, filter to this collection only.
            project: If set, filter to this project only.
            layer: If set, filter to this layer only.
            added_after: If set, only return documents added after this ISO timestamp.

        Returns:
            List of DocumentInfo ordered by ingestion date (newest first).
        """
        conditions, filter_params = self._get_filter_conditions(
            collection=collection,
            project=project,
            layer=layer,
        )
        if added_after:
            conditions.append("added_at >= ?")
            filter_params.append(added_after)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"""SELECT path, title, source_type, collection,
                       project, layer, tags,
                       chunk_count, char_count, added_at
                FROM documents {where}
                ORDER BY added_at DESC""",
            filter_params,
        ).fetchall()
        return [DocumentInfo.model_validate(dict(r)) for r in rows]

    def list_collections(self) -> list[str]:
        """List distinct collection names.

        Returns:
            Sorted list of collection names.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT collection FROM documents ORDER BY collection"
        ).fetchall()
        return [row[0] for row in rows]

    def stats(self) -> StoreStats:
        """Get aggregate memory statistics.

        Returns:
            StoreStats with document count, chunk count, DB size, and path.

        Side effect: refreshes the store-related gauges in the metrics
        registry (``docs_total``, ``chunks_total``, ``collections_total``,
        ``db_size_bytes``, and ``stem_conn_count``) so that scrapers of
        ``vstash stats --detailed`` or ``GET /metrics`` reflect the
        current state without having to touch the registry manually.
        """
        from .metrics import registry

        doc_count: int = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count: int = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        col_count: int = self._conn.execute(
            "SELECT COUNT(DISTINCT collection) FROM documents"
        ).fetchone()[0]
        db_size: int = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Keep gauges in sync with what stats() just observed.
        registry.gauge_set("docs_total", doc_count)
        registry.gauge_set("chunks_total", chunk_count)
        registry.gauge_set("collections_total", col_count)
        registry.gauge_set("db_size_bytes", db_size)
        registry.gauge_set("stem_conn_count", len(self._stem_conns))

        return StoreStats(
            documents=doc_count,
            chunks=chunk_count,
            collections=col_count,
            db_size_mb=round(db_size / 1024 / 1024, 2),
            db_path=str(self.db_path),
        )

    # ------------------------------------------------------------------ #
    # Integrity (#134)                                                     #
    # ------------------------------------------------------------------ #

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
                repairs.append(
                    IntegrityRepair(
                        name="repair_transaction",
                        success=False,
                        detail=f"rolled back: {exc}",
                    )
                )

        return repairs

    # ------------------------------------------------------------------ #
    # Export                                                                #
    # ------------------------------------------------------------------ #

    def export_chunks(
        self,
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, object]]:
        """Export chunks with document metadata for training data curation.

        Each returned dict contains the chunk text plus its parent document
        metadata (title, path, project, layer, tags, collection).

        Args:
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by tag (LIKE match).
            include_embeddings: If True, include the raw embedding vector.

        Returns:
            List of dicts with keys: text, title, path, chunk, project,
            layer, tags, collection. If include_embeddings is True, also
            includes an 'embedding' key with the float vector.
        """
        conditions, params = self._get_filter_conditions(
            "d",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
        )
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        embed_col = ", v.embedding" if include_embeddings else ""
        embed_join = "LEFT JOIN vec_chunks v ON v.rowid = c.id" if include_embeddings else ""

        rows = self._conn.execute(
            f"""SELECT c.text, c.seq, d.title, d.path,
                       d.collection, d.project, d.layer, d.tags
                       {embed_col}
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                {embed_join}
                {where}
                ORDER BY d.added_at DESC, d.id ASC, c.seq ASC""",
            params,
        ).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            entry: dict[str, object] = {
                "text": row["text"],
                "title": row["title"],
                "path": row["path"],
                "chunk": row["seq"],
                "collection": row["collection"],
                "project": row["project"],
                "layer": row["layer"],
                "tags": row["tags"],
            }
            if include_embeddings:
                raw = row["embedding"]
                if raw is not None:
                    entry["embedding"] = _deserialize(raw)
            results.append(entry)

        return results

    # ------------------------------------------------------------------ #
    # Adaptive RRF weights                                                 #
    # ------------------------------------------------------------------ #

    def _stem_terms(self, words: list[str]) -> list[str]:
        """Stem words using the same FTS5 porter tokenizer as the index.

        Uses a per-thread in-memory FTS5 connection so stemming is
        identical to what fts5vocab reports.  ~0.02ms per call.

        **Threading model and the close-from-other-thread trick.**

        Each calling thread gets its own dedicated connection, registered
        in ``self._stem_conns`` keyed by thread id.  Connections are
        opened with ``check_same_thread=False`` *not* to enable sharing
        — they are still **only used from their owning thread** (the
        dict-by-tid lookup guarantees that) — but to allow ``close()``
        running on the main thread at shutdown to release connections
        whose owner threads have already exited.

        This is safe **only** when the underlying libsqlite is built
        with threading support, which Python's bundled sqlite3 always
        does in modern versions (``sqlite3.threadsafety >= 1``).  If
        someone were to run on an exotic build with
        ``sqlite3.threadsafety == 0`` (single-threaded mode), the close
        from another thread could crash.  We assert that at module
        import time so the failure mode is loud, not silent.
        """
        tid = threading.get_ident()
        with self._stem_lock:
            conn = self._stem_conns.get(tid)
            if conn is None:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.execute('CREATE VIRTUAL TABLE _stem USING fts5(x, tokenize="porter ascii")')
                conn.execute("CREATE VIRTUAL TABLE _stem_v USING fts5vocab(_stem, row)")
                self._stem_conns[tid] = conn
        # Use the connection outside the lock — each thread only ever
        # touches its own connection (lookup by tid above), so no
        # cross-thread access happens here.
        conn.execute("DELETE FROM _stem")
        conn.execute("INSERT INTO _stem VALUES (?)", [" ".join(words)])
        rows = conn.execute("SELECT term FROM _stem_v").fetchall()
        return [r[0] for r in rows]

    def _build_idf_cache(self) -> tuple[dict[str, float], int]:
        """Build a term → IDF dictionary from the FTS5 index.

        Computed once per store lifetime and cached.  Uses a single SQL
        query (no per-term lookups).  Terms are porter-stemmed (matching
        the FTS5 tokenizer) so lookups from ``_compute_adaptive_rrf_params``
        hit correctly.

        Returns:
            Tuple of (term_idf_dict, total_chunk_count).
        """
        from .metrics import registry

        if self._idf_cache is not None:
            registry.counter_inc("idf_cache_hits")
            return self._idf_cache
        registry.counter_inc("idf_cache_misses")

        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total_chunks == 0:
            self._idf_cache = ({}, 0)
            return self._idf_cache

        # Single query: get df for every distinct term in the corpus
        # fts5vocab(row) reports chunk-level df, matching total_chunks
        try:
            idf_dict = {
                row["term"]: math.log(total_chunks / (row["doc"] + 1))
                for row in self._conn.execute("SELECT term, doc FROM fts_chunks_vocab")
            }
        except Exception:
            # fts5vocab table may not exist — disable adaptive IDF
            logger.debug("fts_chunks_vocab unavailable; adaptive IDF disabled", exc_info=True)
            self._idf_cache = ({}, 0)
            return self._idf_cache

        self._idf_cache = (idf_dict, total_chunks)
        return self._idf_cache

    def _invalidate_idf_cache(self) -> None:
        """Clear the IDF cache after document changes.

        When inside a batch_mode() context, the actual invalidation is
        deferred until the batch exits so that bulk ingestion does not
        trigger repeated cache rebuilds.
        """
        if self._batch_depth > 0:
            self._batch_dirty = True
            return
        self._idf_cache = None

    def _flush_deferred_fts(self) -> None:
        """Bulk-insert all deferred FTS rows in a single transaction."""
        if not self._deferred_fts_rows:
            return
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                    self._deferred_fts_rows,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._deferred_fts_rows.clear()

    @contextmanager
    def batch_mode(self, *, defer_fts: bool = False) -> Iterator[None]:
        """Context manager that defers IDF cache invalidation and optionally FTS indexing.

        Use this when adding many documents in a loop to avoid redundant
        cache rebuilds and per-insert FTS overhead.  Supports re-entrant
        (nested) usage.

        Args:
            defer_fts: When True, FTS5 inserts are collected in memory
                and flushed in one bulk pass when the outermost batch
                exits.  This avoids updating the FTS B-tree per chunk,
                which is the dominant cost at >100 documents.

        Not thread-safe -- intended for single-threaded batch operations.
        Searches executed during a batch may use stale IDF weights and
        (when defer_fts=True) miss newly added documents in FTS results.

        Precondition when ``defer_fts=True``: do not ingest the same
        path twice within a single batch.  Re-ingesting a path deletes
        the old chunks (firing the FTS delete trigger), but deferred
        rows for the old rowids are already queued and would become
        orphaned FTS entries on flush.

        Example::

            with store.batch_mode(defer_fts=True):
                for path in paths:
                    store.add_document(...)
            # IDF cache invalidated + FTS populated in one pass
        """
        self._batch_depth += 1
        was_deferring = self._defer_fts
        if defer_fts:
            self._defer_fts = True
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                try:
                    if self._defer_fts:
                        self._flush_deferred_fts()
                except Exception:
                    logger.error(
                        "FTS flush failed; documents are stored but not FTS-indexed. "
                        "Run 'vstash reindex' to rebuild the FTS index."
                    )
                    raise
                finally:
                    self._defer_fts = was_deferring
                    if self._batch_dirty:
                        self._idf_cache = None
                        self._batch_dirty = False

    def _compute_adaptive_rrf_params(
        self, query_text: str, default_cutoff: float = 1.3225
    ) -> tuple[float, float, float]:
        """Compute adaptive vec/fts weights and distance cutoff.

        Uses mean IDF of query terms to determine whether keywords are
        informative (rare terms → boost FTS) or noisy (common terms →
        trust vectors).  Long queries (>50 words) reduce FTS weight and
        relax the distance cutoff (diffuse embeddings compress distances).

        IDF values are cached per store lifetime via ``_build_idf_cache()``.
        Per-query overhead is O(k) dict lookups for k query terms — microseconds.

        Returns:
            Tuple of (vec_weight, fts_weight, distance_cutoff).
        """
        words = query_text.split()
        n_words = len(words)

        # Long queries: favor vector + relax distance cutoff
        # (diffuse embeddings from long text compress distance range).
        # See ``_LONG_QUERY_DISTANCE_CUTOFF`` for the squaring rationale
        # (#272). The vec_only branch in ``search`` mirrors this same
        # relaxation; both must use the same constant.
        if n_words > _ADAPTIVE_RRF_LONG_QUERY:
            return 0.9, 0.1, _LONG_QUERY_DISTANCE_CUTOFF

        idf_dict, total_chunks = self._build_idf_cache()

        if total_chunks < 2:
            return 0.6, 0.4, default_cutoff  # default (IDF meaningless with <2 chunks)

        # Stem query terms using the same porter tokenizer as FTS5
        # so lookups match the fts5vocab keys exactly
        filtered_words = [w for w in words if len(w) > 1]
        if not filtered_words:
            return 0.6, 0.4, default_cutoff
        stemmed = self._stem_terms(filtered_words)

        # O(k) dict lookups — no SQL per term
        idfs: list[float] = []
        max_idf = math.log(total_chunks)  # IDF for terms not in corpus
        for term in stemmed:
            if term in idf_dict:
                idfs.append(idf_dict[term])
            else:
                # Stemmed term not in corpus → max rarity → high IDF
                idfs.append(max_idf)

        if not idfs:
            return 0.6, 0.4, default_cutoff  # default

        mean_idf = sum(idfs) / len(idfs)

        # Sigmoid: high IDF (rare terms) → boost FTS; low IDF → boost vector
        # Threshold = median IDF ≈ ln(N) / 2 (half the max IDF range)
        # Alpha = 2.0 gives smooth transition over ~1 IDF unit
        threshold = math.log(total_chunks) / 2
        alpha = 2.0
        fts_signal = 1.0 / (1.0 + math.exp(-alpha * (mean_idf - threshold)))

        # Map sigmoid [0,1] to fts_weight [0.1, 0.6]
        # Low signal (common words): fts=0.1, vec=0.9
        # High signal (rare terms): fts=0.6, vec=0.4
        fts_weight = 0.1 + fts_signal * 0.5
        vec_weight = 1.0 - fts_weight

        return round(vec_weight, 3), round(fts_weight, 3), default_cutoff

    def record_spread(self, spread: float) -> None:
        """Record a search spread value for adaptive threshold computation.

        Keeps only the last 50 entries to act as a sliding window.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO search_stats (spread, created_at) VALUES (?, ?)",
                [spread, now_iso],
            )
            # Prune to keep only the last 50 entries
            self._conn.execute(
                "DELETE FROM search_stats WHERE id NOT IN "
                "(SELECT id FROM search_stats ORDER BY id DESC LIMIT 50)"
            )
            self._conn.commit()

    def adaptive_relevance_threshold(self, fallback: float = 0.15) -> float:
        """Compute a per-corpus adaptive relevance threshold.

        Uses the mean and standard deviation of recent spreads to set a
        threshold at mean - 1 standard deviation. This adapts to the user's
        specific corpus: a corpus with naturally high spreads gets a higher
        threshold, and vice versa.

        Falls back to the fixed threshold when fewer than 10 data points exist.

        Returns:
            Adaptive threshold, or ``fallback`` if insufficient history.
        """
        rows = self._conn.execute(
            "SELECT spread FROM search_stats ORDER BY id DESC LIMIT 50"
        ).fetchall()

        if len(rows) < 10:
            return fallback

        spreads = [r["spread"] for r in rows]
        mean = sum(spreads) / len(spreads)
        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        std = variance**0.5

        # Threshold at mean - 1σ: spreads below this are unusually low
        threshold = max(0.01, mean - std)
        return threshold

    def record_search_event(
        self,
        query: str,
        best_distance: float,
        relevance_tier: str,
        result_count: int,
        miss_hint: dict | None = None,
    ) -> int:
        """Record a search event for discard telemetry.

        Args:
            query: Raw query text.
            best_distance: Distance of the top result (smaller = more relevant).
            relevance_tier: ``"high"`` | ``"medium"`` | ``"low"`` bucket.
            result_count: Number of rows the caller surfaced to the user.
            miss_hint: Optional lightweight diagnostic blob, persisted as
                JSON in the ``miss_hint`` column. Issue #157 part 3
                (2026-04-21). Typical shape:
                ``{"reason": "empty" | "all_low", "best_distance": ...,
                "tier": ..., "result_count": ..., "top_k_requested": ...}``.
                Consumed by ``vstash why --recent``.

        Returns the event ID so it can be marked as dismissed later.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        hint_json = json.dumps(miss_hint) if miss_hint is not None else None
        with self._write_lock:
            cursor = self._conn.execute(
                "INSERT INTO search_events (query, best_distance, relevance_tier, "
                "result_count, dismissed, created_at, miss_hint) "
                "VALUES (?, ?, ?, ?, 0, ?, ?)",
                [query, best_distance, relevance_tier, result_count, now_iso, hint_json],
            )
            # Prune to keep only the last 1000 entries
            self._conn.execute(
                "DELETE FROM search_events WHERE id NOT IN "
                "(SELECT id FROM search_events ORDER BY id DESC LIMIT 1000)"
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def recent_miss_hints(self, limit: int = 10) -> list[dict]:
        """Return the N most recent search_events that carry a miss_hint,
        newest first. Used by ``vstash why --recent``.

        Raises ``ValueError`` when ``limit < 1`` so a negative value
        does not accidentally turn into SQLite's "unlimited" sentinel
        (``LIMIT -1``)."""
        limit = int(limit)
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        rows = self._conn.execute(
            "SELECT id, query, best_distance, relevance_tier, result_count, "
            "miss_hint, created_at FROM search_events "
            "WHERE miss_hint IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            [limit],
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                hint = json.loads(r["miss_hint"]) if r["miss_hint"] else {}
            except (TypeError, ValueError):
                hint = {}
            out.append(
                {
                    "id": int(r["id"]),
                    "query": str(r["query"]),
                    "best_distance": float(r["best_distance"]),
                    "relevance_tier": str(r["relevance_tier"]),
                    "result_count": int(r["result_count"]),
                    "created_at": str(r["created_at"]),
                    "miss_hint": hint,
                }
            )
        return out

    def mark_search_dismissed(self, event_id: int) -> None:
        """Mark a search event as dismissed (user didn't engage with results)."""
        with self._write_lock:
            self._conn.execute(
                "UPDATE search_events SET dismissed = 1 WHERE id = ?",
                [event_id],
            )
            self._conn.commit()

    def search_telemetry_summary(self) -> dict[str, dict[str, int]]:
        """Return dismiss rates grouped by relevance tier.

        Returns:
            Dict mapping tier name to {"total": N, "dismissed": N}.
        """
        rows = self._conn.execute(
            "SELECT relevance_tier, COUNT(*) AS total, "
            "SUM(dismissed) AS dismissed FROM search_events "
            "GROUP BY relevance_tier"
        ).fetchall()
        return {
            row["relevance_tier"]: {
                "total": row["total"],
                "dismissed": row["dismissed"],
            }
            for row in rows
        }

    def expand_context(self, results: list[SearchResult], window: int = 1) -> list[SearchResult]:
        """Expand each search result with adjacent chunks from the same document.

        For each result, fetches up to ``window`` chunks before and after it
        (by sequence number), concatenates their text, and returns a new
        SearchResult with the expanded text. This gives the LLM more context
        without increasing the number of results.

        Args:
            results: Search results to expand.
            window: Number of adjacent chunks to include on each side.

        Returns:
            New list of SearchResult with expanded text.
        """
        if not results or window < 1:
            return results

        # -- Step 1: batch-resolve chunk_id → doc_id via PK lookup -------
        chunk_ids = [r.chunk_id for r in results]
        doc_id_map: dict[int, str] = {}
        for i in range(0, len(chunk_ids), _SQLITE_PARAM_BATCH):
            batch = chunk_ids[i : i + _SQLITE_PARAM_BATCH]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT id, doc_id FROM chunks WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                doc_id_map[row["id"]] = row["doc_id"]

        # Fallback for results whose chunk_id didn't resolve (stale IDs
        # from cached results or synthetic SearchResult objects in tests).
        for r in results:
            if r.chunk_id not in doc_id_map:
                row = self._conn.execute(
                    "SELECT c.doc_id FROM chunks c JOIN documents d ON d.id = c.doc_id "
                    "WHERE d.path = ? AND c.seq = ? AND c.text = ? LIMIT 1",
                    [r.path, r.chunk, r.text],
                ).fetchone()
                if row:
                    doc_id_map[r.chunk_id] = row["doc_id"]

        # -- Step 2: batch-fetch adjacent chunks grouped by doc_id --------
        # To avoid fetching massive unneeded intermediate rows when resolving
        # exact sparse matches against compound keys, we calculate exactly
        # which (doc_id, seq) pairs are needed, and query them in batches
        # via a Common Table Expression (CTE) with a VALUES clause.

        needed_targets = set()
        for r in results:
            did = doc_id_map.get(r.chunk_id)
            if did is None:
                continue
            for seq in range(r.chunk - window, r.chunk + window + 1):
                if seq >= 0:  # sequence numbers are 0-indexed and non-negative
                    needed_targets.add((did, seq))

        # Sort target pairs by (doc_id, seq) so the SQL lookup touches rows
        # in roughly sequential order within each document. Helps page-cache
        # locality for large stores without changing correctness.
        needed_targets_list = sorted(needed_targets)

        # Key: (doc_id, seq) → text
        chunk_text_map: dict[tuple[str, int], str] = {}

        # A pair (doc_id, seq) is 2 params. To stay under _SQLITE_PARAM_BATCH,
        # batch size for pairs is _SQLITE_PARAM_BATCH // 2
        batch_size = _SQLITE_PARAM_BATCH // 2

        for i in range(0, len(needed_targets_list), batch_size):
            batch = needed_targets_list[i : i + batch_size]
            values_clause = ", ".join(["(?, ?)"] * len(batch))
            flat_params = [item for sublist in batch for item in sublist]

            query = f"""
                WITH targets(doc_id, seq) AS (
                    VALUES {values_clause}
                )
                SELECT c.doc_id, c.seq, c.text
                FROM chunks c
                JOIN targets t ON c.doc_id = t.doc_id AND c.seq = t.seq
            """
            rows = self._conn.execute(query, flat_params).fetchall()
            for row in rows:
                chunk_text_map[(row["doc_id"], row["seq"])] = row["text"]

        # -- Step 3: assemble expanded results, preserving explain --------
        expanded = []
        for r in results:
            did = doc_id_map.get(r.chunk_id)
            if did is None:
                expanded.append(r)
                continue

            parts = []
            for seq in range(r.chunk - window, r.chunk + window + 1):
                text = chunk_text_map.get((did, seq))
                if text is not None:
                    parts.append(text)

            if parts:
                expanded.append(
                    SearchResult(
                        chunk_id=r.chunk_id,
                        text="\n".join(parts),
                        title=r.title,
                        path=r.path,
                        chunk=r.chunk,
                        score=r.score,
                        explain=r.explain,
                    )
                )
            else:
                expanded.append(r)

        return expanded

    # ------------------------------------------------------------------ #
    # Reindex                                                              #
    # ------------------------------------------------------------------ #

    def reindex(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        new_dim: int,
        batch_size: int = 256,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> int:
        """Re-embed all chunks with a new embedding model.

        Drops and recreates ``vec_chunks`` with the new dimensionality,
        then re-embeds all chunk text in batches.

        Args:
            embed_fn: Function that takes a list of texts and returns
                a list of embedding vectors.
            new_dim: Dimensionality of the new embedding model.
            batch_size: Number of chunks to embed per batch.
            progress_cb: Optional callback ``(processed, total)`` for
                progress reporting.

        Returns:
            Number of chunks re-embedded.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.
        """
        # Fail before we DROP vec_chunks. A non-positive batch_size would
        # make the keyset loop below terminate immediately after dropping
        # the table, silently wiping the vector index.
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        with self._write_lock:
            # Count total chunks
            total = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if total == 0:
                return 0

            try:
                # Drop and recreate vec_chunks with new dimensions
                self._conn.execute("DROP TABLE IF EXISTS vec_chunks")
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE vec_chunks "
                    f"USING vec0(embedding float[{new_dim}] distance_metric=cosine)"
                )

                # Rebuild snapvec index if active
                if self._snap is not None:
                    self._snap = SnapIndex(dim=new_dim, bits=self._snapvec_bits, seed=0)

                # Re-embed in batches. Two O(N^2) traps avoided here
                # (issue #265, sibling to #264):
                #
                # 1. Keyset pagination (WHERE id > last_id) instead of
                #    LIMIT ? OFFSET ?. SQLite rescans `offset` rows per
                #    page, so N/batch_size pages cost O(N^2/batch_size).
                # 2. Snapvec add_batch is coalesced into a single call
                #    after the loop. SnapIndex.add_batch does np.vstack
                #    on the growing buffer internally (snapvec/_index.py
                #    :206), so per-page calls copy O(current) each time.
                #
                # chunks.id is INTEGER PRIMARY KEY AUTOINCREMENT (>=1),
                # so seeding last_id=0 matches all rows on the first page.
                processed = 0
                last_id = 0
                snap_rowid_parts: list[list[int]] = []
                snap_vector_parts: list[np.ndarray] = []
                while True:
                    rows = self._conn.execute(
                        "SELECT id, text FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
                        [last_id, batch_size],
                    ).fetchall()
                    if not rows:
                        break

                    texts = [row["text"] for row in rows]
                    ids = [row["id"] for row in rows]
                    embeddings = embed_fn(texts)

                    # Fail fast if embed_fn returns the wrong count. Without
                    # this guard, executemany would silently truncate to the
                    # shorter zip and leave the store with a partial vec
                    # index while ``processed`` still counts all rows.
                    if len(embeddings) != len(ids):
                        raise ValueError(
                            f"embed_fn returned {len(embeddings)} embeddings "
                            f"for {len(ids)} texts; must match 1:1"
                        )

                    self._conn.executemany(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        [(cid, _serialize(emb)) for cid, emb in zip(ids, embeddings, strict=False)],
                    )

                    if self._snap is not None:
                        snap_rowid_parts.append(ids)
                        snap_vector_parts.append(np.asarray(embeddings, dtype=np.float32))

                    processed += len(rows)
                    last_id = ids[-1]
                    if progress_cb:
                        progress_cb(processed, total)

                if self._snap is not None and snap_rowid_parts:
                    all_rowids = (
                        snap_rowid_parts[0]
                        if len(snap_rowid_parts) == 1
                        else [rid for part in snap_rowid_parts for rid in part]
                    )
                    all_vectors = (
                        snap_vector_parts[0]
                        if len(snap_vector_parts) == 1
                        else np.concatenate(snap_vector_parts, axis=0)
                    )
                    # Drop per-batch lists once coalesced so peak RSS is
                    # bounded during the final add_batch allocation.
                    snap_vector_parts.clear()
                    snap_rowid_parts.clear()
                    self._snap.add_batch(all_rowids, all_vectors)

                self._conn.commit()
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise

            # Update stored dimension only after successful commit
            self.embedding_dim = new_dim
            # Persist snapvec AFTER successful SQLite commit; failures here
            # cannot be rolled back via SQLite, so we log them.
            if self._snap is not None:
                self._snap_dirty = True
                try:
                    self._save_snapvec()
                except Exception:
                    logger.exception(
                        "Failed to persist snapvec after successful reindex; "
                        "run 'vstash reindex' again to rebuild."
                    )

            return processed

    def close(self) -> None:
        """Close the database connection and all stem-conn resources.

        Releases every per-thread FTS5 stemming connection registered
        by ``_stem_terms()``, regardless of which thread created them.
        SQLite in-memory connections can be closed from any thread
        (unlike file connections opened with ``check_same_thread=True``),
        so this is safe to call from the main thread on shutdown of a
        multi-threaded server like ``vstash serve`` or the MCP server.
        """
        import contextlib

        # Flush any pending snapvec writes before tearing down. Both
        # the flat and ivfpq backends now defer ``.snpv`` / ivfpq
        # writes until here (or an explicit ``_checkpoint_snapvec``
        # call), so the file hits disk once per session instead of
        # once per ``add_document``. Crash recovery is handled on
        # next open by ``_init_snapvec`` comparing the stored index
        # length against ``vec_chunks`` and rebuilding if stale.
        with contextlib.suppress(Exception):
            self._checkpoint_snapvec()

        # Close all per-thread stem connections under the lock so we
        # don't race a worker thread that's just creating a new one.
        with self._stem_lock:
            for stem_conn in self._stem_conns.values():
                # sqlite3.Error covers the realistic failure modes
                # (already closed, locked, etc.).  Anything else is a
                # real bug worth surfacing rather than silently
                # swallowing during teardown.
                with contextlib.suppress(sqlite3.Error):
                    stem_conn.close()
            self._stem_conns.clear()
        self._conn.close()
