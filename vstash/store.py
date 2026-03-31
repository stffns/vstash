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
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from types import TracebackType

import threading

import numpy as np
import sqlite_vec

from .config import ScoringConfig
from .models import DocumentInfo, SearchResult, StoreStats

# Probe for snapvec availability (optional dependency)
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


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


# Standard RRF constant — balances precision vs recall
RRF_K = 60

# Frequency score saturation point — access counts above this
# produce diminishing returns in the log1p normalization.
FREQ_SATURATION = 100

# Adaptive scoring: minimum max/mean ratio (among accessed chunks) before
# memory scoring activates.  Empirically, scoring only helps when there are
# clear "favorite" chunks (benchmark-focused scenario needed ~30× differential
# for +16% NDCG).  8× is conservative: it activates only after genuine
# power-user patterns develop, not from uniform Zipf-like browsing.
SCORING_SIGNAL_RATIO = 8.0

# Adaptive scoring: γ reaches 1.0 at this max/mean ratio.
SCORING_SIGNAL_SATURATE = 15.0


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
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._write_lock = threading.Lock()
        # Per-thread state for the cosine distance of the best vector match.
        # Thread-local so concurrent searches on a shared instance don't race.
        self._thread_local = threading.local()
        self._thread_local.last_best_distance = 0.0
        self._conn = self._connect()

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
                    "Failed to load SnapIndex from %s, creating new.", path, exc_info=True
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
                logger.warning("Failed to reload SnapIndex after rollback, creating empty.")
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

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

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
                for seq, (text, embedding) in enumerate(zip(chunks, embeddings)):
                    # Insert chunk — get rowid for linking vec + fts tables
                    cursor = self._conn.execute(
                        "INSERT INTO chunks (doc_id, seq, text, access_count, created_at, last_accessed_at)"
                        " VALUES (?, ?, ?, 0, ?, ?)",
                        [doc_id, seq, text, now_iso, now_iso],
                    )
                    rowid = cursor.lastrowid

                    # Vector index entry
                    self._conn.execute(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        [rowid, _serialize(embedding)],
                    )

                    # FTS5 entry (rowid must match chunks.id)
                    self._conn.execute(
                        "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                        [rowid, text],
                    )

                # Add to snapvec in-memory (persisted after successful commit)
                if self._snap is not None:
                    snap_ids = [
                        row[0]
                        for row in self._conn.execute(
                            "SELECT id FROM chunks WHERE doc_id = ? ORDER BY seq",
                            [doc_id],
                        ).fetchall()
                    ]
                    snap_vecs = np.array(embeddings, dtype=np.float32)
                    self._snap.add_batch(snap_ids, snap_vecs)
                    self._snap_dirty = True

                self._conn.commit()
                # Persist snapvec AFTER successful SQLite commit
                if self._snap_dirty:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise
        return doc_id

    def delete_document(self, path: str) -> bool:
        """Remove a document and all its chunks from the store.

        Deletes all copies of the document regardless of collection.

        Args:
            path: File path or URL to remove.

        Returns:
            True if at least one document was found and deleted.
        """
        with self._write_lock:
            doc_ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT id FROM documents WHERE path = ?", [path]
                ).fetchall()
            ]
            if not doc_ids:
                return False
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for doc_id in doc_ids:
                    self._delete_by_doc_id(doc_id)
                self._conn.commit()
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
        chunk_ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", [doc_id]
            ).fetchall()
        ]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            # Delete vec_chunks first (no trigger involved)
            self._conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", chunk_ids)
            # Delete from snapvec index (in-memory, persisted after commit)
            if self._snap is not None:
                for cid in chunk_ids:
                    self._snap.delete(cid)
                self._snap_dirty = True
            # Delete chunks — trg_chunks_delete trigger auto-syncs FTS5
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
        cursor = self._conn.execute("DELETE FROM documents WHERE id = ?", [doc_id])
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    # Search — Hybrid RRF                                                  #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        vec_weight: float = 0.6,
        fts_weight: float = 0.4,
        distance_cutoff: float = 1.15,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        scoring: ScoringConfig | None = None,
        _gamma_override: float | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: vector (semantic) + FTS5 (keyword) combined with RRF.

        RRF score = vec_weight * 1/(k+rank_vec) + fts_weight * 1/(k+rank_fts)

        Results are filtered by vector distance — chunks whose distance from
        the query is more than ``distance_cutoff`` times the best (closest)
        distance are discarded before RRF scoring.  This prevents irrelevant
        noise (e.g. Art of War appearing in deep learning queries).

        When ``scoring`` is provided and enabled, results are over-fetched,
        re-ranked with frequency+decay, and truncated to ``top_k``.

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
            scoring: ScoringConfig instance. If enabled, applies frequency+decay re-ranking.

        Returns:
            Ranked list of SearchResult ordered by descending score.
        """
        # Determine effective pool size — over-fetch when scoring is enabled
        effective_k = top_k
        if scoring is not None and scoring.enabled:
            effective_k = max(top_k, scoring.over_fetch)

        # Adaptive candidate pool — avoid pulling half the corpus on small DBs
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        candidate_pool = min(effective_k * 10, max(effective_k * 3, total_chunks // 3))

        # --- Build metadata filter ---
        vec_clause, col_clause, filter_params = self._build_doc_filter(
            collection=collection,
            project=project,
            layer=layer,
        )

        # --- Vector search ---
        if self._snap is not None and len(self._snap) > 0:
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
                    SELECT c.id, c.text, d.title, d.path, c.seq
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
                SELECT c.id, c.text, d.title, d.path, c.seq, v.distance
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

        # --- Filter by vector distance gap ---
        # The best (closest) result has the smallest distance.
        # Remove results that are semantically too far from the ideal match.
        if vec_rows:
            best_distance = float(vec_rows[0]["distance"])
            self.last_best_distance = best_distance
            if best_distance > 0:
                threshold = best_distance * distance_cutoff
                vec_rows = [r for r in vec_rows if float(r["distance"]) <= threshold]
        else:
            self.last_best_distance = 2.0  # max cosine distance = worst case

        # Track which chunk IDs passed the vector distance filter
        relevant_chunk_ids: set[int] = {row["id"] for row in vec_rows}

        # --- FTS5 search ---
        # Quote each word individually and join with OR for keyword matching.
        # This preserves injection safety (each token is double-quoted to
        # prevent FTS5 syntax like NEAR, NOT, OR from being interpreted)
        # while allowing keyword-level matching instead of exact-phrase.
        words = query_text.split()
        quoted_words = ['"' + w.replace('"', '""') + '"' for w in words if len(w) > 1]
        safe_query = (
            " OR ".join(quoted_words) if quoted_words else '"' + query_text.replace('"', '""') + '"'
        )
        try:
            fts_rows = self._conn.execute(
                f"""
                SELECT c.id, c.text, d.title, d.path, c.seq,
                       rank as fts_rank
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

        # --- Reciprocal Rank Fusion ---
        scores: dict[int, dict[str, str | int | float]] = {}

        for rank, row in enumerate(vec_rows):
            chunk_id: int = row["id"]
            scores[chunk_id] = {
                "id": chunk_id,
                "text": row["text"],
                "title": row["title"],
                "path": row["path"],
                "chunk": row["seq"],
                "rrf": vec_weight * (1.0 / (RRF_K + rank)),
            }

        for rank, row in enumerate(fts_rows):
            chunk_id = row["id"]
            # Only include FTS results that also passed vector relevance filter,
            # OR that are in the top FTS results (strong keyword match).
            is_fts_top = rank < effective_k * 2
            fts_contribution = fts_weight * (1.0 / (RRF_K + rank))
            if chunk_id in scores:
                scores[chunk_id]["rrf"] = float(scores[chunk_id]["rrf"]) + fts_contribution
            elif chunk_id in relevant_chunk_ids or is_fts_top:
                scores[chunk_id] = {
                    "id": chunk_id,
                    "text": row["text"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk": row["seq"],
                    "rrf": fts_contribution,
                }

        # Sort by RRF score descending
        ranked = sorted(scores.values(), key=lambda x: float(x["rrf"]), reverse=True)

        # Apply frequency+decay re-ranking if scoring is enabled.
        # The effective beta is scaled by γ (scoring maturity) so that
        # scoring has zero influence until access patterns are meaningful.
        if scoring is not None and scoring.enabled:
            gamma = _gamma_override if _gamma_override is not None else self.scoring_maturity()
            if gamma > 0:
                effective_beta = scoring.beta * gamma
                ranked = ranked[:effective_k]
                ranked = self.rerank_with_decay(
                    ranked,
                    alpha=scoring.alpha,
                    beta=effective_beta,
                    decay_lambda=scoring.decay_lambda,
                )

        # Intra-document MMR deduplication: allow multiple chunks from the
        # same document only when they are semantically diverse (e.g. different
        # chapters of a book).  Chunks from different documents compete purely
        # on score — no cross-document penalty.
        mmr_lambda = scoring.mmr_lambda if scoring is not None else 0.5
        ranked = self._mmr_dedup(ranked, top_k, mmr_lambda)

        results = [
            SearchResult(
                text=str(r["text"]),
                title=str(r["title"]),
                path=str(r["path"]),
                chunk=int(r["chunk"]),
                score=round(float(r.get("final_score", r["rrf"])), 6),
            )
            for r in ranked
        ]

        # Track access for returned chunks (best-effort, failures don't affect results).
        # Always track when track_access is True, even if scoring is disabled —
        # this builds up usage history for future scoring enablement.
        if scoring is not None and scoring.track_access:
            try:
                result_ids = [int(r["id"]) for r in ranked if "id" in r]
                if result_ids:
                    self.track_access(result_ids)
            except Exception:
                logging.getLogger(__name__).debug("Access tracking failed", exc_info=True)

        return results

    # ------------------------------------------------------------------ #
    # MMR intra-document deduplication                                      #
    # ------------------------------------------------------------------ #

    def _mmr_dedup(
        self,
        ranked: list[dict[str, str | int | float]],
        top_k: int,
        mmr_lambda: float,
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
                logging.getLogger(__name__).debug(
                    "MMR embedding fetch failed, falling back to hard dedup",
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
        score_key = "final_score" if "final_score" in ranked[0] else "rrf"
        scores = [float(r[score_key]) for r in ranked]
        s_min, s_max = min(scores), max(scores)
        s_range = s_max - s_min if s_max > s_min else 1.0

        selected: list[dict[str, str | int | float]] = []
        # Track selected embeddings per document for similarity comparison.
        selected_embs_by_doc: dict[str, list[list[float]]] = {}
        remaining = list(range(len(ranked)))

        for _ in range(min(top_k, len(ranked))):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                r = ranked[idx]
                norm_score = (float(r[score_key]) - s_min) / s_range
                doc_key = str(r["path"])

                # Diversity penalty: only against same-document selections.
                max_sim = 0.0
                if doc_key in selected_embs_by_doc:
                    chunk_id = int(r["id"])
                    emb = embeddings.get(chunk_id)
                    if emb is not None:
                        for sel_emb in selected_embs_by_doc[doc_key]:
                            sim = _cosine_sim(emb, sel_emb)
                            if sim > max_sim:
                                max_sim = sim

                mmr_score = mmr_lambda * norm_score - (1 - mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx < 0 or best_mmr < 0:
                # Stop when the best remaining candidate has negative MMR,
                # meaning its redundancy penalty exceeds its relevance.
                break

            chosen = ranked[best_idx]
            selected.append(chosen)
            remaining.remove(best_idx)

            # Track embedding for future similarity checks.
            doc_key = str(chosen["path"])
            chunk_id = int(chosen["id"])
            emb = embeddings.get(chunk_id)
            if emb is not None:
                selected_embs_by_doc.setdefault(doc_key, []).append(emb)

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
    ) -> tuple[list[str], list[str]]:
        """Build filter conditions for document metadata.

        Args:
            alias: Table alias prefix (e.g. ``'d.'``, ``'d2.'``, or ``''``).
            collection: Filter by collection name.
            project: Filter by project tag.
            layer: Filter by layer tag.
            tags: Filter by tag (LIKE match within comma-separated tags).

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
        return conditions, params

    @staticmethod
    def _build_doc_filter(
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
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
        """
        conditions_d2, params = VstashStore._get_filter_conditions(
            "d2",
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
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

    # ------------------------------------------------------------------ #
    # Inspect                                                              #
    # ------------------------------------------------------------------ #

    def list_documents(
        self,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> list[DocumentInfo]:
        """List all ingested documents.

        Args:
            collection: If set, filter to this collection only.
            project: If set, filter to this project only.
            layer: If set, filter to this layer only.

        Returns:
            List of DocumentInfo ordered by ingestion date (newest first).
        """
        conditions, filter_params = self._get_filter_conditions(
            collection=collection,
            project=project,
            layer=layer,
        )
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
        """
        doc_count: int = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count: int = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        col_count: int = self._conn.execute(
            "SELECT COUNT(DISTINCT collection) FROM documents"
        ).fetchone()[0]
        db_size: int = self.db_path.stat().st_size if self.db_path.exists() else 0
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
    # Adaptive Scoring — maturity gate                                     #
    # ------------------------------------------------------------------ #

    def scoring_maturity(self) -> float:
        """Compute γ ∈ [0, 1] measuring whether access patterns have enough
        differential to make frequency+decay scoring useful.

        Uses the max/mean ratio of access counts among accessed chunks.
        When all chunks have similar access counts (ratio < SCORING_SIGNAL_RATIO),
        γ = 0 and scoring is effectively disabled.  As the ratio grows toward
        SCORING_SIGNAL_SATURATE, γ ramps linearly to 1.0.

        Returns:
            Float in [0, 1].  0 = no useful signal, 1 = full scoring weight.
        """
        row = self._conn.execute(
            "SELECT AVG(access_count) AS mean, MAX(access_count) AS max_val, "
            "COUNT(*) AS n "
            "FROM chunks WHERE access_count > 0"
        ).fetchone()

        if not row or not row["mean"] or row["n"] < 10:
            return 0.0

        ratio = row["max_val"] / row["mean"]
        if ratio < SCORING_SIGNAL_RATIO:
            return 0.0

        # Linear ramp from 0 → 1 between SIGNAL_RATIO and SIGNAL_SATURATE
        denominator = SCORING_SIGNAL_SATURATE - SCORING_SIGNAL_RATIO
        if denominator <= 1e-9:
            return 1.0
        return min(1.0, (ratio - SCORING_SIGNAL_RATIO) / denominator)

    # ------------------------------------------------------------------ #
    # Frequency + Decay Scoring                                            #
    # ------------------------------------------------------------------ #

    def rerank_with_decay(
        self,
        candidates: list[dict[str, object]],
        *,
        alpha: float = 0.8,
        beta: float = 0.2,
        decay_lambda: float = 0.05,
    ) -> list[dict[str, object]]:
        """Re-rank candidates post-RRF with frequency + temporal decay.

        Normalizes RRF scores to [0, 1] via min-max scaling so that
        the alpha/beta weights operate on comparable scales.

        Args:
            candidates: List of dicts with keys: id, rrf, text, title, path, chunk.
            alpha: Weight for semantic similarity (normalized RRF).
            beta: Weight for access history (frequency * decay).
            decay_lambda: Exponential decay rate.

        Returns:
            Same list, sorted by final_score descending, with final_score added.
        """
        if not candidates:
            return candidates

        # Short-circuit: if beta ≈ 0 (e.g. γ suppressed it), skip the
        # metadata DB lookup entirely — ranking is determined by RRF alone.
        # We still min-max normalize so that normalized_rrf is in [0, 1].
        if beta < 1e-9:
            rrf_scores = [float(c["rrf"]) for c in candidates]
            min_rrf = min(rrf_scores)
            rrf_range = max(rrf_scores) - min_rrf
            for c in candidates:
                normalized = (float(c["rrf"]) - min_rrf) / rrf_range if rrf_range > 0 else 1.0
                c["final_score"] = alpha * normalized
            candidates.sort(key=lambda c: float(c["final_score"]), reverse=True)
            return candidates

        # Fetch access metadata for all candidate chunk IDs
        chunk_ids = [int(c["id"]) for c in candidates]
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT id, access_count, last_accessed_at, created_at "
            f"FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        meta = {row["id"]: dict(row) for row in rows}

        # Min-max scaling of RRF scores
        rrf_scores = [float(c["rrf"]) for c in candidates]
        min_rrf = min(rrf_scores)
        max_rrf = max(rrf_scores)
        rrf_range = max_rrf - min_rrf

        now = datetime.now(timezone.utc)

        for c in candidates:
            chunk_id = int(c["id"])
            info = meta.get(chunk_id, {})
            access_count = info.get("access_count", 0) or 0
            last_accessed = info.get("last_accessed_at")
            created = info.get("created_at")

            # Normalize RRF to [0, 1]
            normalized_rrf = (float(c["rrf"]) - min_rrf) / rrf_range if rrf_range > 0 else 1.0

            # Temporal decay (clamp to 0 to guard against clock skew / future dates)
            ref_str = last_accessed or created
            if ref_str is None:
                days_ago = 0.0
            else:
                ref_dt = datetime.fromisoformat(ref_str)
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
                days_ago = max(0.0, (now - ref_dt).total_seconds() / 86400)

            # +1 baseline so zero-access chunks still get a small nonzero score
            freq_score = (1 + access_count) * math.exp(-decay_lambda * days_ago)
            # Normalize frequency component to [0, 1] via log1p, capped at 1.0
            freq_normalized = min(1.0, math.log1p(freq_score) / math.log1p(FREQ_SATURATION))
            c["final_score"] = alpha * normalized_rrf + beta * freq_normalized

        candidates.sort(key=lambda c: float(c["final_score"]), reverse=True)
        return candidates

    def track_access(self, chunk_ids: list[int]) -> None:
        """Record access for the given chunks (batch UPDATE).

        Increments access_count and sets last_accessed_at for each chunk.
        Called after search results are built so failures don't affect results.

        Args:
            chunk_ids: List of chunk IDs that were returned to the user.
        """
        if not chunk_ids:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(chunk_ids))
        with self._write_lock:
            self._conn.execute(
                f"UPDATE chunks SET access_count = COALESCE(access_count, 0) + 1, "
                f"last_accessed_at = ? WHERE id IN ({placeholders})",
                [now_iso, *chunk_ids],
            )
            self._conn.commit()

    def total_access_count(self) -> int:
        """Return the sum of all access_count values across chunks."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(access_count), 0) AS total FROM chunks"
        ).fetchone()
        return int(row["total"])

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

        expanded = []
        for r in results:
            row = self._conn.execute(
                "SELECT text FROM chunks WHERE doc_id = ("
                "  SELECT doc_id FROM chunks c JOIN documents d ON d.id = c.doc_id "
                "  WHERE d.path = ? AND c.seq = ? LIMIT 1"
                ") AND seq BETWEEN ? AND ? ORDER BY seq",
                [r.path, r.chunk, r.chunk - window, r.chunk + window],
            ).fetchall()

            if row:
                combined_text = "\n".join(chunk["text"] for chunk in row)
                expanded.append(
                    SearchResult(
                        text=combined_text,
                        title=r.title,
                        path=r.path,
                        chunk=r.chunk,
                        score=r.score,
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

                # Update stored dimension
                self.embedding_dim = new_dim
                self._conn.commit()
                # Persist snapvec AFTER successful SQLite commit
                if self._snap is not None:
                    self._save_snapvec()
            except Exception:
                self._conn.rollback()
                self._reload_snapvec()
                raise

            return processed

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
