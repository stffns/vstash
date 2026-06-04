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
import sqlite3
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

from ..config import CacheConfig, LimitsConfig, ObservabilityConfig
from ..errors import SchemaVersionError
from ..validation import validate_document_input
from ..models import (
    ChunkInfo,
    DocumentInfo,
    SearchResult,
    StoreStats,
)

from ._common import (
    _ADAPTIVE_RRF_LONG_QUERY,
    _HAS_SNAPVEC,
    _LONG_QUERY_DISTANCE_CUTOFF,
    _PipelineTracer,
    _SQLITE_PARAM_BATCH,
    _cosine_sim,
    _deserialize,
    _normalize_tags,
    _serialize,
    KNOWN_SCHEMA_VERSIONS,
    RELEVANCE_TIER_HIGH_MAX,
    RELEVANCE_TIER_MEDIUM_MAX,
    RRF_K,
    SCHEMA_VERSION,
    SnapIndex,
    relevance_tier,
)
from ._index import _IndexBackendMixin
from ._integrity import _IntegrityMixin
from ._schema import _SchemaManagerMixin
from ._search import _SearchEngineMixin

# Public + re-export surface of ``vstash.store``. Several modules and tests
# import these names directly from ``vstash.store``; the package __init__ is
# the single re-export hub (#280), so they are listed here explicitly. The
# underscore-prefixed entries are intentionally part of the stable internal
# surface (used by tests / sibling modules), not private to this module.
__all__ = [
    "KNOWN_SCHEMA_VERSIONS",
    "RELEVANCE_TIER_HIGH_MAX",
    "RELEVANCE_TIER_MEDIUM_MAX",
    "RRF_K",
    "SCHEMA_VERSION",
    "_ADAPTIVE_RRF_LONG_QUERY",
    "_HAS_SNAPVEC",
    "_LONG_QUERY_DISTANCE_CUTOFF",
    "_SQLITE_PARAM_BATCH",
    "SchemaVersionError",
    "VstashStore",
    "_PipelineTracer",
    "_cosine_sim",
    "_deserialize",
    "_normalize_tags",
    "_serialize",
    "relevance_tier",
]


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

logger = logging.getLogger(__name__)


class VstashStore(_IndexBackendMixin, _IntegrityMixin, _SchemaManagerMixin, _SearchEngineMixin):
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

        # Normalize the on-disk tag representation to match the form
        # the search-side comma-anchored ``LIKE`` expects. Writers
        # routinely hand us ``"alpha, beta"`` (frontmatter style); if
        # we stored it verbatim, a query for ``tag="beta"`` would
        # generate ``%,beta,%`` against ``,alpha, beta,`` and miss
        # because of the leading space. Round-tripping through
        # ``_normalize_tags`` strips that whitespace and dedupes
        # repeats once, at ingest time, so the storage invariant is
        # ``"tag1,tag2,..."`` with no spaces and no duplicates.
        stored_tags = ",".join(_normalize_tags(tags)) if tags else None

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
                        stored_tags,
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

        from ..validation import validate_document_input

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

                    # Normalize tags on write so the search-side
                    # comma-anchored ``LIKE`` matches reliably. Same
                    # invariant as ``add_document``: stored form is
                    # ``"tag1,tag2,..."`` with no whitespace and no
                    # duplicates.
                    stored_tags = ",".join(_normalize_tags(tags)) if tags else None

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
                            stored_tags,
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

    def update_metadata(
        self,
        path: str,
        *,
        title: str | None = None,
        tags: str | None = None,
        collection: str | None = None,
    ) -> bool:
        """Update document metadata (``title`` and/or ``tags``) in place.

        This is the in-place mutator for fields that do NOT require
        re-chunking or re-embedding. Use it for renames, retagging, or
        any metadata-only correction. To change the chunk text itself,
        use ``Memory.update(path, text=...)`` which re-runs the embed
        pipeline.

        Args:
            path: Document path (or URL) identifying the row.
            title: New title; pass ``None`` to leave unchanged.
            tags: New comma-separated tags string; pass ``None`` to
                leave unchanged. Pass ``""`` to clear the tag set.
            collection: If set, restrict the update to a specific
                ``(collection, path)`` pair. ``None`` (default) updates
                every matching row across collections, mirroring
                ``delete_document``'s default semantics.

        Returns:
            ``True`` if at least one row was updated.

        Raises:
            ValueError: Neither ``title`` nor ``tags`` was provided
                (the call would be a no-op and almost always indicates
                a caller bug).
        """
        if title is None and tags is None:
            raise ValueError(
                "update_metadata called with no field to update -- "
                "pass at least one of `title=` or `tags=`."
            )
        with self._write_lock:
            # Build the SET clause and bind params dynamically so the
            # statement only touches the fields the caller actually
            # provided. Mirrors the `_get_filter_conditions` style used
            # elsewhere in this module.
            set_parts: list[str] = []
            set_params: list[str] = []
            if title is not None:
                set_parts.append("title = ?")
                set_params.append(title)
            if tags is not None:
                # Normalize on write so the search-side comma-anchored
                # LIKE matches reliably (same invariant as
                # ``add_document`` post-#364). ``tags=""`` clears.
                set_parts.append("tags = ?")
                stored = ",".join(_normalize_tags(tags)) if tags else None
                set_params.append(stored)
            where_parts = ["path = ?"]
            where_params: list[str] = [path]
            if collection is not None:
                where_parts.append("collection = ?")
                where_params.append(collection)
            sql = f"UPDATE documents SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(sql, [*set_params, *where_params])
                rowcount = cur.rowcount
                self._conn.commit()
                if rowcount > 0:
                    # Tag/title changes are visible to search filters
                    # (tag-anchored LIKE in `_get_filter_conditions`) and
                    # to result formatting, so invalidate the LRU.
                    self._bump_cache_epoch()
            except Exception:
                self._conn.rollback()
                raise
            return rowcount > 0

    def prune_documents(
        self,
        *,
        before_iso: str | None = None,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | list[str] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Delete documents matching the given filters.

        At least one filter (``before_iso`` or any metadata) must be
        provided -- a fully-unfiltered prune would wipe the entire
        store, which is almost always a caller bug and is better done
        explicitly with ``delete_document`` in a loop.

        Args:
            before_iso: Delete documents whose ``added_at`` is strictly
                older than this ISO timestamp (``"2026-01-15"`` or full
                ``"2026-01-15T00:00:00+00:00"``).
            collection: Restrict to this collection.
            project: Restrict to this project.
            layer: Restrict to this layer.
            tags: Restrict to docs tagged with any of these tags (OR).
                Accepts the same forms as the search-side tag filter:
                comma-separated string (``"alpha,beta"``) or list.
            dry_run: If ``True``, report what would be deleted without
                touching the store.

        Returns:
            ``{"status": "ok"|"dry_run", "deleted": N, "paths": [...]}``.
        """
        if (
            before_iso is None
            and collection is None
            and project is None
            and layer is None
            and not _normalize_tags(tags)
        ):
            raise ValueError(
                "prune_documents called with no filter -- pass at least one "
                "of `before_iso`, `collection`, `project`, `layer`, or `tags` "
                "to scope the deletion."
            )
        conditions, params = self._get_filter_conditions(
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
        )
        if before_iso is not None:
            conditions.append("added_at < ?")
            params.append(before_iso)
        where = "WHERE " + " AND ".join(conditions)
        select_sql = f"SELECT id, path FROM documents {where}"

        # ``dry_run`` is informational and tolerates the small race
        # between SELECT and any concurrent writer -- it does not
        # delete anything. The real prune below puts the SELECT INSIDE
        # ``BEGIN IMMEDIATE`` so the rows we report deleted are exactly
        # the rows that satisfied the filter at the instant the write
        # lock was acquired. Otherwise a concurrent ``add_document``
        # between the unlocked SELECT and the locked delete would let
        # the reported ``paths`` / ``deleted`` count drift from what
        # ``_delete_by_doc_ids`` actually removed.
        if dry_run:
            rows = self._conn.execute(select_sql, params).fetchall()
            if not rows:
                return {"status": "dry_run", "deleted": 0, "paths": []}
            return {
                "status": "dry_run",
                "deleted": len(rows),
                "paths": [r["path"] for r in rows],
            }
        with self._write_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute(select_sql, params).fetchall()
                if not rows:
                    self._conn.rollback()
                    return {"status": "ok", "deleted": 0, "paths": []}
                doc_ids = [r["id"] for r in rows]
                paths = [r["path"] for r in rows]
                self._delete_by_doc_ids(doc_ids)
                self._conn.commit()
                self._invalidate_idf_cache()
                self._bump_cache_epoch()
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise
        return {"status": "ok", "deleted": len(paths), "paths": paths}

    def vacuum(self) -> None:
        """Run SQLite ``VACUUM`` to reclaim space and defragment the file.

        VACUUM cannot run inside a transaction. The store's normal
        write path uses ``BEGIN IMMEDIATE`` so callers must not invoke
        this from within another operation -- it is intended as the
        outermost step of ``Memory.compact()`` / ``vstash compact``,
        well after the prune phase has committed.

        VACUUM rebuilds the entire database file, so on large stores
        it can take seconds to minutes and momentarily doubles the disk
        usage (working copy + original). Both costs are unavoidable.
        """
        with self._write_lock:
            self._conn.execute("VACUUM")

    def optimize_fts(self) -> None:
        """Merge FTS5 segments in place and reclaim space.

        Uses the FTS5 ``'optimize'`` directive (``INSERT INTO fts_chunks
        (fts_chunks) VALUES ('optimize')``) which compacts the
        inverted index without altering any tokenization or content.
        Much cheaper than ``VACUUM`` and safe to run independently.
        """
        with self._write_lock:
            self._conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('optimize')")
            self._conn.commit()

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
        from ..metrics import registry

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
        from ..metrics import registry

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
