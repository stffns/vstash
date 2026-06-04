"""Schema lifecycle for ``vstash.store``: DDL, migrations, and ``store_meta``.

Extracted from the store facade as the ``_SchemaManagerMixin`` step of the
#280 split. It is a mixin: ``VstashStore`` inherits it, and the methods rely
on instance state created in ``VstashStore.__init__`` (``self._conn``,
``self.embedding_dim``, ``self._write_lock``, ``self.db_path``), so the class
is never instantiated on its own.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from ..errors import SchemaVersionError
from ._common import KNOWN_SCHEMA_VERSIONS, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class _SchemaManagerMixin:
    """DDL creation, schema-version migration, and ``store_meta`` key/value access."""

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
        from .. import __version__ as _vstash_version

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
