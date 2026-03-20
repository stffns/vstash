"""
store.py — sqlite-vec + FTS5 hybrid store with Reciprocal Rank Fusion.

Pure vector search misses exact keyword matches (names, codes, error strings).
Pure FTS5 misses semantic similarity.
RRF combines both: score = Σ 1/(k + rank), k=60 is the standard constant.

Single .db file. WAL mode for safe concurrent reads.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import sqlite_vec

from .models import DocumentInfo, SearchResult, StoreStats


def _serialize(vector: list[float]) -> bytes:
    """Serialize a float vector into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


# Standard RRF constant — balances precision vs recall
RRF_K = 60


class VstashStore:
    """SQLite-backed vector + FTS5 hybrid store with RRF ranking.

    Supports context manager protocol for safe resource cleanup::

        with VstashStore(db_path, embedding_dim=384) as store:
            store.add_document(...)

    Args:
        db_path: Path to the SQLite database file.
        embedding_dim: Dimensionality of embedding vectors.
    """

    def __init__(self, db_path: str, embedding_dim: int = 384) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._conn = self._connect()

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
        conn.executescript(f"""
            -- Document metadata
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                title       TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'file',
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

            -- Auto-sync FTS5 when chunks are deleted directly
            CREATE TRIGGER IF NOT EXISTS trg_chunks_delete
            AFTER DELETE ON chunks
            BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text)
                VALUES('delete', OLD.id, OLD.text);
            END;
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
        doc_id = hashlib.sha256(path.encode()).hexdigest()[:32]
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        return row is not None

    def add_document(
        self,
        path: str,
        title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_type: str = "file",
    ) -> str:
        """Add a document and its chunks to the store.

        If the document already exists (same path hash), it is replaced.

        Args:
            path: Absolute file path or URL.
            title: Human-readable document title.
            chunks: List of text chunks.
            embeddings: Corresponding embedding vectors.
            source_type: Document type (pdf, code, url, etc.).

        Returns:
            The generated document ID (16-char hex hash).
        """
        doc_id = hashlib.sha256(path.encode()).hexdigest()[:32]

        # Remove existing version if re-ingesting
        self._delete_by_doc_id(doc_id)

        self._conn.execute(
            """INSERT INTO documents (id, path, title, source_type, char_count, chunk_count, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                doc_id, path, title, source_type,
                sum(len(c) for c in chunks),
                len(chunks),
                datetime.now(UTC).isoformat(),
            ],
        )

        for seq, (text, embedding) in enumerate(zip(chunks, embeddings)):
            # Insert chunk — get rowid for linking vec + fts tables
            cursor = self._conn.execute(
                "INSERT INTO chunks (doc_id, seq, text) VALUES (?, ?, ?)",
                [doc_id, seq, text],
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

        self._conn.commit()
        return doc_id

    def delete_document(self, path: str) -> bool:
        """Remove a document and all its chunks from the store.

        Args:
            path: File path or URL to remove.

        Returns:
            True if the document was found and deleted.
        """
        doc_id = hashlib.sha256(path.encode()).hexdigest()[:32]
        deleted = self._delete_by_doc_id(doc_id)
        self._conn.commit()
        return deleted

    def _delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete a document by its internal hash ID.

        Args:
            doc_id: 32-char hex document hash.

        Returns:
            True if the document existed and was deleted.
        """
        chunk_ids = [
            row[0] for row in
            self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", [doc_id]
            ).fetchall()
        ]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            # Delete vec_chunks first (no trigger involved)
            self._conn.execute(
                f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", chunk_ids
            )
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

        Returns:
            Ranked list of SearchResult ordered by descending RRF score.
        """
        # Adaptive candidate pool — avoid pulling half the corpus on small DBs
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        candidate_pool = min(top_k * 10, max(top_k * 3, total_chunks // 3))

        # --- Vector search ---
        vec_rows = self._conn.execute(
            """
            SELECT c.id, c.text, d.title, d.path, c.seq, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            JOIN documents d ON d.id = c.doc_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            [_serialize(query_embedding), candidate_pool],
        ).fetchall()

        # --- Filter by vector distance gap ---
        # The best (closest) result has the smallest distance.
        # Remove results that are semantically too far from the ideal match.
        if vec_rows:
            best_distance = float(vec_rows[0]["distance"])
            if best_distance > 0:
                threshold = best_distance * distance_cutoff
                vec_rows = [r for r in vec_rows if float(r["distance"]) <= threshold]

        # Track which chunk IDs passed the vector distance filter
        relevant_chunk_ids: set[int] = {row["id"] for row in vec_rows}

        # --- FTS5 search ---
        safe_query = query_text.replace('"', '""')
        try:
            fts_rows = self._conn.execute(
                """
                SELECT c.id, c.text, d.title, d.path, c.seq,
                       rank as fts_rank
                FROM fts_chunks f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE fts_chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                [safe_query, candidate_pool],
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 query syntax error (e.g. single char) — fall back to no FTS
            fts_rows = []

        # --- Reciprocal Rank Fusion ---
        scores: dict[int, dict[str, str | int | float]] = {}

        for rank, row in enumerate(vec_rows):
            chunk_id: int = row["id"]
            scores[chunk_id] = {
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
            is_fts_top = rank < top_k * 2
            fts_contribution = fts_weight * (1.0 / (RRF_K + rank))
            if chunk_id in scores:
                scores[chunk_id]["rrf"] = float(scores[chunk_id]["rrf"]) + fts_contribution
            elif chunk_id in relevant_chunk_ids or is_fts_top:
                scores[chunk_id] = {
                    "text": row["text"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk": row["seq"],
                    "rrf": fts_contribution,
                }

        # Sort by RRF score descending, return top_k
        ranked = sorted(scores.values(), key=lambda x: float(x["rrf"]), reverse=True)[:top_k]

        return [
            SearchResult(
                text=str(r["text"]),
                title=str(r["title"]),
                path=str(r["path"]),
                chunk=int(r["chunk"]),
                score=round(float(r["rrf"]), 6),
            )
            for r in ranked
        ]

    # ------------------------------------------------------------------ #
    # Lookup                                                               #
    # ------------------------------------------------------------------ #

    def find_document(self, query: str) -> str | None:
        """Find a document by partial path or title match.

        Searches for documents where the path or title contains the query
        string (case-insensitive). Returns the path of the first match.

        Args:
            query: Partial filename, path, or title to search for.

        Returns:
            The full path of the matching document, or None.
        """
        row = self._conn.execute(
            """SELECT path FROM documents
               WHERE path LIKE ? OR title LIKE ?
               ORDER BY added_at DESC LIMIT 1""",
            [f"%{query}%", f"%{query}%"],
        ).fetchone()
        return row["path"] if row else None

    # ------------------------------------------------------------------ #
    # Inspect                                                              #
    # ------------------------------------------------------------------ #

    def list_documents(self) -> list[DocumentInfo]:
        """List all ingested documents.

        Returns:
            List of DocumentInfo ordered by ingestion date (newest first).
        """
        rows = self._conn.execute(
            """SELECT path, title, source_type, chunk_count, char_count, added_at
               FROM documents ORDER BY added_at DESC"""
        ).fetchall()
        return [DocumentInfo.model_validate(dict(r)) for r in rows]

    def stats(self) -> StoreStats:
        """Get aggregate memory statistics.

        Returns:
            StoreStats with document count, chunk count, DB size, and path.
        """
        doc_count: int = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count: int = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db_size: int = self.db_path.stat().st_size if self.db_path.exists() else 0
        return StoreStats(
            documents=doc_count,
            chunks=chunk_count,
            db_size_mb=round(db_size / 1024 / 1024, 2),
            db_path=str(self.db_path),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
