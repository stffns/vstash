"""
store.py — sqlite-vec + FTS5 hybrid store with Reciprocal Rank Fusion.

Pure vector search misses exact keyword matches (names, codes, error strings).
Pure FTS5 misses semantic similarity.
RRF combines both: score = Σ 1/(k + rank), k=60 is the standard constant.

Single .db file. WAL mode for safe concurrent reads.
"""

from __future__ import annotations

import hashlib
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
from typing import Literal

import threading

import numpy as np
import sqlite_vec

from .config import LimitsConfig, ObservabilityConfig
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
SCHEMA_VERSION = "1"

#: Schema versions this build of vstash knows how to read.  Anything
#: not in this set raises :class:`SchemaVersionError` on open.
KNOWN_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1"})


class SchemaVersionError(RuntimeError):
    """Raised when an existing DB declares a schema version this build
    of vstash does not recognize.

    The right remedy is to upgrade vstash, restore from backup, or run
    a future ``vstash migrate`` command (not yet implemented).
    """


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

# Probe for snapvec availability (optional dependency)
try:
    from snapvec import SnapIndex

    if not hasattr(SnapIndex, "delete_batch"):

        def _delete_batch(self: SnapIndex, ids: list[str | int]) -> int:
            import numpy as np

            to_delete_pos = []
            for i in ids:
                if i in self._id_to_pos:
                    to_delete_pos.append(self._id_to_pos.pop(i))

            if not to_delete_pos:
                return 0

            pos_set = set(to_delete_pos)

            self._ids = [i for p, i in enumerate(self._ids) if p not in pos_set]
            keep_mask = np.ones(len(self._indices), dtype=bool)
            keep_mask[to_delete_pos] = False

            self._indices = self._indices[keep_mask]
            self._norms = self._norms[keep_mask]
            if getattr(self, "_qjl", None) is not None:
                self._qjl = self._qjl[keep_mask]
                self._rnorms = self._rnorms[keep_mask]

            self._id_to_pos = {id_val: i for i, id_val in enumerate(self._ids)}

            self._cache = None
            return len(to_delete_pos)

        SnapIndex.delete_batch = _delete_batch  # type: ignore[attr-defined]

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


def relevance_tier(distance: float) -> str:
    """Classify vector distance into a relevance tier.

    Tiers:
        "high"   — distance <= 0.95: confident match.
        "medium" — 0.95 < distance <= 0.98: uncertain.
        "low"    — distance > 0.98: likely off-topic.
    """
    if distance <= 0.95:
        return "high"
    if distance <= 0.98:
        return "medium"
    return "low"


# Standard RRF constant — balances precision vs recall
RRF_K = 60

# Adaptive RRF: query length threshold above which FTS weight is reduced.
# ArguAna (194 avg words) showed -38.4% vs dense; queries >50 words are
# typically semantic paraphrases where keywords add noise.
_ADAPTIVE_RRF_LONG_QUERY = 50


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
        observability: ObservabilityConfig | None = None,
        limits: LimitsConfig | None = None,
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

        # --- SnapVec backend (optional) ---
        self._snap: SnapIndex | None = None  # type: ignore[name-defined]
        self._vector_backend = vector_backend
        self._snapvec_bits = snapvec_bits
        self._snap_dirty = False
        if vector_backend == "snapvec":
            self._init_snapvec()

    @property
    def _snapvec_path(self) -> Path:
        """Path to the snapvec index file (next to the .db file)."""
        return self.db_path.with_suffix(".snpv")

    def _init_snapvec(self) -> None:
        """Load or create the SnapIndex for the snapvec backend."""
        if not _HAS_SNAPVEC:
            logger.warning(
                "snapvec backend requested but snapvec is not installed. "
                "Install with: pip install vstash[snapvec]. Falling back to sqlite-vec."
            )
            self._vector_backend = "sqlite-vec"
            return

        path = self._snapvec_path
        if path.exists():
            try:
                self._snap = SnapIndex.load(str(path))
                # Verify dimension match
                if self._snap.dim != self.embedding_dim:
                    logger.warning(
                        "SnapIndex dim=%d != embedding_dim=%d. Rebuilding index.",
                        self._snap.dim,
                        self.embedding_dim,
                    )
                    self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
                    self._save_snapvec()
            except Exception:
                logger.warning(
                    "Failed to load SnapIndex from %s — creating empty. "
                    "Existing vectors lost; run 'vstash reindex' to rebuild.",
                    path,
                    exc_info=True,
                )
                self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)
                self._save_snapvec()
        else:
            self._snap = SnapIndex(dim=self.embedding_dim, bits=self._snapvec_bits, seed=0)

    def _save_snapvec(self) -> None:
        """Persist the SnapIndex to disk. Called after successful SQLite commit."""
        if self._snap is not None:
            self._snap.save(str(self._snapvec_path))
            self._snap_dirty = False

    def _reload_snapvec(self) -> None:
        """Reload SnapIndex from disk after a failed transaction.

        Restores the on-disk state so the in-memory index matches SQLite
        after a rollback.
        """
        if self._snap is None:
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

            -- Vector index (sqlite-vec)
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(embedding float[{self.embedding_dim}]);

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

            -- Search event telemetry for validating relevance signal
            CREATE TABLE IF NOT EXISTS search_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT NOT NULL,
                best_distance   REAL NOT NULL,
                relevance_tier  TEXT NOT NULL,
                result_count    INTEGER NOT NULL,
                dismissed       INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL
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

        On a fresh DB the version row does not exist yet — we write
        the current :data:`SCHEMA_VERSION`.  On an existing DB we read
        the row and compare against :data:`KNOWN_SCHEMA_VERSIONS`:

        - Match → no-op.
        - Missing → assume legacy v1 (the only schema ever shipped) and
          stamp it for next time.
        - Unknown → raise :class:`SchemaVersionError` so the caller
          knows the DB was written by a newer vstash than this build
          can safely read.
        """
        from . import __version__ as _vstash_version

        # Use INSERT OR IGNORE so two processes opening the same fresh
        # DB concurrently can't crash on a UNIQUE constraint — whichever
        # one wins the race stamps the version, the other no-ops.  We
        # always re-read after the (potentially-skipped) insert so the
        # value we validate is the one actually in the DB.
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ["schema_version", SCHEMA_VERSION, now_iso],
        )
        conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value, updated_at) VALUES (?, ?, ?)",
            ["vstash_version", _vstash_version, now_iso],
        )
        conn.commit()

        row = conn.execute("SELECT value FROM store_meta WHERE key = 'schema_version'").fetchone()
        # row cannot be None here — we just inserted (or ignored) it.
        existing = str(row["value"])
        if existing not in KNOWN_SCHEMA_VERSIONS:
            msg = (
                f"Database at {self.db_path} declares schema_version={existing!r}, "
                f"which this build of vstash ({_vstash_version}) does not recognize. "
                f"Known versions: {sorted(KNOWN_SCHEMA_VERSIONS)}. "
                f"Upgrade vstash or restore the DB from a compatible backup."
            )
            raise SchemaVersionError(msg)

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

        doc_id = hashlib.sha256(f"{collection}:{path}".encode("utf-8")).hexdigest()[:32]

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

                # Vector index entries
                vec_data = [(rowid, _serialize(emb)) for rowid, emb in zip(rowids, embeddings)]
                self._conn.executemany(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    vec_data,
                )

                # FTS5 entries (rowid must match chunks.id)
                fts_data = [(rowid, text) for rowid, text in zip(rowids, chunks)]
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
                self._invalidate_idf_cache()
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

                    doc_id = hashlib.sha256(f"{collection}:{path}".encode("utf-8")).hexdigest()[:32]
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

                    vec_data = [(rowid, _serialize(emb)) for rowid, emb in zip(rowids, embeddings)]
                    self._conn.executemany(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        vec_data,
                    )

                    fts_data = list(zip(rowids, chunks))
                    self._conn.executemany(
                        "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                        fts_data,
                    )

                    if self._snap is not None:
                        snap_vecs = np.array(embeddings, dtype=np.float32)
                        self._snap.add_batch(rowids, snap_vecs)
                        self._snap_dirty = True

                self._conn.commit()
                self._invalidate_idf_cache()
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
        doc_id = hashlib.sha256(f"{collection}:{path}".encode("utf-8")).hexdigest()[:32]
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

                    # Delete from snapvec index (in-memory, persisted after commit)
                    if self._snap is not None:
                        if hasattr(self._snap, "delete_batch"):
                            self._snap.delete_batch(batch_chunk_ids)  # type: ignore[attr-defined]
                        else:
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

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        distance_cutoff: float = 1.15,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        explain: bool = False,
        adaptive_rrf: bool = True,
        recency_boost: float = 0.0,
        added_after: str | None = None,
        added_before: str | None = None,
        mmr_lambda: float = 0.5,
        fts_only: bool = False,
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
            collection: If set, restrict search to documents in this collection.
            project: If set, restrict search to documents with this project tag.
            layer: If set, restrict search to documents with this layer tag.
            fts_only: If True, short-circuit the pipeline to FTS5 only (#152):
                no vector ANN scan, no distance cutoff, no adaptive
                RRF weight computation. FTS5 hits are still fed
                through the RRF scoring formula (with ``vec_weight=0.0``
                and ``fts_weight=1.0``, so each hit scores
                ``1/(60 + fts_rank)``), through MMR dedup, through the
                optional recency boost, and through context expansion
                — it is not a raw BM25 score dump.  Useful for
                debugging ranking, for queries where vector embeddings
                are known to be diffuse (cross-lingual, highly
                technical), or as a deliberate fallback when the
                vector pool is expected to be empty.

        Returns:
            Ranked list of SearchResult ordered by descending score.
        """
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
            # FTS-only short-circuit (#152): bypass vector search entirely.
            # This forces the weights to (0.0, 1.0) and disables adaptive
            # RRF so the pipeline cannot silently re-enable the vector
            # path.  The vector search, distance cutoff, and relevance
            # filter blocks below are all guarded on `not fts_only` or
            # on `vec_rows` being non-empty, so the downstream MMR /
            # scoring path runs unchanged on FTS-only input.
            if fts_only:
                vec_weight = 0.0
                fts_weight = 1.0
                adaptive_rrf = False

            # Adaptive RRF: compute weights from query characteristics (IDF + length)
            # Skip if caller provided explicit weights
            if adaptive_rrf and vec_weight is None and fts_weight is None:
                vec_weight, fts_weight, distance_cutoff = self._compute_adaptive_rrf_params(
                    query_text, default_cutoff=distance_cutoff
                )
            if vec_weight is None and fts_weight is None:
                vec_weight, fts_weight = 0.6, 0.4
            elif vec_weight is None:
                vec_weight = 1.0 - fts_weight
            elif fts_weight is None:
                fts_weight = 1.0 - vec_weight

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
            if fts_only:
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
                        detail="fts_only=True: vector search intentionally skipped by caller",
                    )
                    _tracer.record(
                        "distance_cutoff",
                        passed=True,
                        detail="fts_only=True: no distance cutoff applied",
                    )
            elif self._snap is not None and len(self._snap) > 0:
                # SnapVec ANN search — returns list[(id, distance)]
                snap_results = self._snap.search(
                    np.array(query_embedding, dtype=np.float32), k=candidate_pool
                )
                snap_ids = [int(r[0]) for r in snap_results]
                snap_dists = {int(r[0]): float(r[1]) for r in snap_results}

                if snap_ids:
                    placeholders = ",".join("?" * len(snap_ids))
                    # Build filter clause adapted for snapvec (no v.rowid)
                    snap_filter = vec_clause.replace("v.rowid", "c.id") if vec_clause else ""
                    rows = self._conn.execute(
                        f"""
                        SELECT c.id, c.text, d.title, d.path, c.seq, d.added_at
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
                    SELECT c.id, c.text, d.title, d.path, c.seq, v.distance, d.added_at
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
            if track_target is not None and not fts_only:
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
            # Quote each word individually and join with OR for keyword matching.
            # This preserves injection safety (each token is double-quoted to
            # prevent FTS5 syntax like NEAR, NOT, OR from being interpreted)
            # while allowing keyword-level matching instead of exact-phrase.
            words = query_text.split()
            quoted_words = ['"' + w.replace('"', '""') + '"' for w in words if len(w) > 1]
            safe_query = (
                " OR ".join(quoted_words)
                if quoted_words
                else '"' + query_text.replace('"', '""') + '"'
            )
            try:
                fts_rows = self._conn.execute(
                    f"""
                    SELECT c.id, c.text, d.title, d.path, c.seq,
                           rank as fts_rank, d.added_at
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
            scores: dict[int, dict[str, str | int | float]] = {}
            _explain_rrf_vec: dict[int, float] = {}  # chunk_id -> vec RRF contribution
            _explain_rrf_fts: dict[int, float] = {}  # chunk_id -> fts RRF contribution
            _explain_fts_rank: dict[int, int] = {}  # chunk_id -> fts rank

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
                }
                if explain:
                    _explain_rrf_vec[chunk_id] = vec_contrib

            for rank, row in enumerate(fts_rows):
                chunk_id = row["id"]
                # Only include FTS results that also passed vector relevance filter,
                # OR that are in the top FTS results (strong keyword match).
                # is_fts_top is the gate that lets FTS-only hits into
                # the score pool when they are not also in the vector
                # candidate set.  Normally we cap at top_k*2 to stop
                # weak keyword noise from polluting results when
                # vector search has a strong opinion.  In fts_only
                # mode there is no vector signal, so the cap becomes
                # a counter-productive artificial truncation — let
                # every FTS row through up to the candidate pool.
                if fts_only:
                    is_fts_top = True
                else:
                    is_fts_top = rank < effective_k * 2
                fts_contribution = fts_weight * (1.0 / (RRF_K + rank))
                if chunk_id in scores:
                    scores[chunk_id]["rrf"] = float(scores[chunk_id]["rrf"]) + fts_contribution
                    if explain:
                        _explain_rrf_fts[chunk_id] = fts_contribution
                        _explain_fts_rank[chunk_id] = rank
                elif chunk_id in relevant_chunk_ids or is_fts_top:
                    scores[chunk_id] = {
                        "id": chunk_id,
                        "text": row["text"],
                        "title": row["title"],
                        "path": row["path"],
                        "chunk": row["seq"],
                        "rrf": fts_contribution,
                        "added_at": row["added_at"],
                    }
                    if explain:
                        _explain_rrf_fts[chunk_id] = fts_contribution
                        _explain_fts_rank[chunk_id] = rank

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
            # When recency_boost > 0, multiply each chunk's RRF score by a decay
            # factor based on how recently it was created.  This biases results
            # toward recent content — useful for agentic memory where the latest
            # context is more likely to be relevant.
            if recency_boost > 0 and ranked:
                now = datetime.now(timezone.utc)
                chunk_ids = [int(r["id"]) for r in ranked]
                placeholders = ",".join("?" * len(chunk_ids))
                rows = self._conn.execute(
                    f"SELECT id, created_at FROM chunks WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                created_map = {}
                for row in rows:
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
                # Re-sort after boost
                ranked = sorted(ranked, key=lambda x: float(x["rrf"]), reverse=True)

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

            # --- Build ExplainInfo per chunk when requested ---
            _explain_map: dict[int, ExplainInfo] = {}
            if explain:
                fts_terms = [w.strip('"') for w in quoted_words] if quoted_words else []
                for r in ranked:
                    cid = int(r["id"])
                    vec_info = _explain_vec.get(cid)
                    rrf_vec_val = round(_explain_rrf_vec.get(cid, 0.0), 6)
                    rrf_fts_val = round(_explain_rrf_fts.get(cid, 0.0), 6)
                    _explain_map[cid] = ExplainInfo(
                        vec_rank=vec_info[0] if vec_info else None,
                        vec_distance=round(vec_info[1], 4) if vec_info else None,
                        fts_rank=_explain_fts_rank.get(cid),
                        rrf_vec=rrf_vec_val,
                        rrf_fts=rrf_fts_val,
                        rrf_total=round(rrf_vec_val + rrf_fts_val, 6),
                        mmr_penalty=round(float(r.get("_mmr_penalty", 0.0)), 4),
                        fts_terms=fts_terms,
                        rrf_vec_weight=vec_weight,
                        rrf_fts_weight=fts_weight,
                    )

            results = [
                SearchResult(
                    chunk_id=int(r["id"]),
                    text=str(r["text"]),
                    title=str(r["title"]),
                    path=str(r["path"]),
                    chunk=int(r["chunk"]),
                    score=round(float(r["rrf"]), 6),
                    explain=_explain_map.get(int(r["id"])) if explain else None,
                    added_at=r.get("added_at"),
                )
                for r in ranked
            ]

            # Stash values the finally block needs to log slow queries
            # with accurate data.
            _result_count = len(results)
            _vec_weight_observed = float(vec_weight)
            _fts_weight_observed = float(fts_weight)

            # _tracer.verdicts has been populated in-place by the record()
            # calls above when tracking is enabled.  No store-level state.
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

        # Extract values into fast lists for index-based access
        doc_keys = [str(r["path"]) for r in ranked]
        chunk_embs = [embeddings.get(int(r["id"])) for r in ranked]

        # Precompute L2 norms for cosine similarity to avoid O(K * N) recomputation.
        chunk_norms = [math.hypot(*emb) if emb is not None else 0.0 for emb in chunk_embs]

        # Track the maximum similarity to any selected chunk from the *same document*.
        # Replaces O(N * S) recomputation with O(1) lookup + O(N) update.
        max_sims = [0.0] * len(ranked)

        for _ in range(min(top_k, len(ranked))):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                norm_score = norm_scores[idx]

                # Diversity penalty: only against same-document selections.
                max_sim = max_sims[idx]

                mmr_score = mmr_lambda * norm_score - (1 - mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx < 0 or best_mmr < 0:
                # Stop when the best remaining candidate has negative MMR,
                # meaning its redundancy penalty exceeds its relevance.
                break

            chosen = ranked[best_idx]
            if _explain:
                chosen["_mmr_penalty"] = (1 - mmr_lambda) * max_sims[best_idx]
            selected.append(chosen)
            remaining.remove(best_idx)

            # Update max_sims for remaining chunks from the same document
            # by comparing against the newly selected embedding.
            new_doc_key = doc_keys[best_idx]
            new_emb = chunk_embs[best_idx]
            new_norm = chunk_norms[best_idx]
            if new_emb is not None:
                for idx in remaining:
                    if doc_keys[idx] == new_doc_key:
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

    @contextmanager
    def batch_mode(self) -> Iterator[None]:
        """Context manager that defers IDF cache invalidation.

        Use this when adding or deleting many documents in a loop to avoid
        redundant cache rebuilds.  Supports re-entrant (nested) usage.

        Not thread-safe — intended for single-threaded batch operations.
        Searches executed during a batch may use stale IDF weights; the
        cache is refreshed once the outermost batch exits.

        Example::

            with store.batch_mode():
                for path in paths:
                    store.add_document(...)
            # IDF cache is invalidated once here
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._idf_cache = None
                self._batch_dirty = False

    def _compute_adaptive_rrf_params(
        self, query_text: str, default_cutoff: float = 1.15
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
        # (diffuse embeddings from long text compress distance range)
        if n_words > _ADAPTIVE_RRF_LONG_QUERY:
            return 0.9, 0.1, 5.0

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
    ) -> int:
        """Record a search event for discard telemetry.

        Returns the event ID so it can be marked as dismissed later.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            cursor = self._conn.execute(
                "INSERT INTO search_events (query, best_distance, relevance_tier, "
                "result_count, dismissed, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                [query, best_distance, relevance_tier, result_count, now_iso],
            )
            # Prune to keep only the last 1000 entries
            self._conn.execute(
                "DELETE FROM search_events WHERE id NOT IN "
                "(SELECT id FROM search_events ORDER BY id DESC LIMIT 1000)"
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

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
        # Build (doc_id, seq_lo, seq_hi) ranges and group by doc_id to
        # minimise queries.  Each doc_id gets one range query that covers
        # the union of all windows for results in that document.
        from collections import defaultdict

        doc_ranges: dict[str, tuple[int, int]] = {}
        doc_result_indices: dict[str, list[int]] = defaultdict(list)
        for idx, r in enumerate(results):
            did = doc_id_map.get(r.chunk_id)
            if did is None:
                continue
            lo, hi = r.chunk - window, r.chunk + window
            if did in doc_ranges:
                old_lo, old_hi = doc_ranges[did]
                doc_ranges[did] = (min(old_lo, lo), max(old_hi, hi))
            else:
                doc_ranges[did] = (lo, hi)
            doc_result_indices[did].append(idx)

        # Fetch all needed chunks per doc_id in one query each.
        # Key: (doc_id, seq) → text
        chunk_text_map: dict[tuple[str, int], str] = {}
        for did, (lo, hi) in doc_ranges.items():
            rows = self._conn.execute(
                "SELECT seq, text FROM chunks WHERE doc_id = ? "
                "AND seq BETWEEN ? AND ? ORDER BY seq",
                [did, lo, hi],
            ).fetchall()
            for row in rows:
                chunk_text_map[(did, row["seq"])] = row["text"]

        # -- Step 3: assemble expanded results, preserving explain --------
        expanded = []
        for idx, r in enumerate(results):
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
        """
        with self._write_lock:
            # Count total chunks
            total = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if total == 0:
                return 0

            try:
                # Drop and recreate vec_chunks with new dimensions
                self._conn.execute("DROP TABLE IF EXISTS vec_chunks")
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{new_dim}])"
                )

                # Rebuild snapvec index if active
                if self._snap is not None:
                    self._snap = SnapIndex(dim=new_dim, bits=self._snapvec_bits, seed=0)

                # Re-embed in batches
                processed = 0
                offset = 0
                while offset < total:
                    rows = self._conn.execute(
                        "SELECT id, text FROM chunks ORDER BY id LIMIT ? OFFSET ?",
                        [batch_size, offset],
                    ).fetchall()
                    if not rows:
                        break

                    texts = [row["text"] for row in rows]
                    ids = [row["id"] for row in rows]
                    embeddings = embed_fn(texts)

                    for chunk_id, embedding in zip(ids, embeddings):
                        self._conn.execute(
                            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                            [chunk_id, _serialize(embedding)],
                        )

                    # Add batch to snapvec index
                    if self._snap is not None:
                        self._snap.add_batch(ids, np.array(embeddings, dtype=np.float32))

                    processed += len(rows)
                    offset += batch_size
                    if progress_cb:
                        progress_cb(processed, total)

                self._conn.commit()
                self._invalidate_idf_cache()
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
