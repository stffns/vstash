"""
memory.py — High-level Python SDK for vstash.

    from vstash import Memory

    mem = Memory(project="my_agent")
    mem.add("docs/spec.pdf")
    answer = mem.ask("What are the system requirements?")

This module wraps the low-level store, ingest, embed, and chat modules
into a single class with a 6-method public API: add, search, ask, remove,
list, stats.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from .chat import ask as _chat_ask
from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .ingest import ingest
from .models import (
    ChunkInfo,
    DocumentInfo,
    IngestResult,
    MissAnalysis,
    SearchResult,
    StoreStats,
)
from .store import VstashStore

# Sentinel for distinguishing "parameter not provided" from explicit None.
_UNSET = object()


class Memory:
    """High-level Python SDK for vstash.

    Drop any document. Ask anything. Get an answer in under a second.

    Args:
        config: Path to vstash.toml. Auto-detected if not provided.
        project: Default project tag for add/search operations.
        collection: Default collection name (default: "default").
        db: Override path to the SQLite database file.
        profile: Named profile to use (e.g. "work", "research").
            Resolves to ``~/.vstash/profiles/<name>/memory.db``.
            ``db`` takes priority over ``profile`` if both are given.

    Example::

        from vstash import Memory

        mem = Memory(project="my_agent")
        mem.add("docs/spec.pdf")
        answer = mem.ask("What are the system requirements?")
        chunks = mem.search("deployment strategy", top_k=3)

        # Named profile
        work = Memory(profile="work")
    """

    def __init__(
        self,
        config: str | Path | None = None,
        *,
        project: str | None = None,
        collection: str = "default",
        db: str | Path | None = None,
        profile: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._cfg = _load_config_from(config)

        # Validate user-supplied identifiers up front (#133).  An empty
        # or whitespace-only project / collection is a caller bug, not a
        # filter — reject it now instead of corrupting the WHERE clause.
        from .validation import validate_identifier

        validate_identifier(project, field="project")
        # collection has a default of "default", so only validate if the
        # caller actually passed something else.  None means "no filter".
        if collection != "default":
            validate_identifier(collection, field="collection")

        self._project = project
        self._collection = collection
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Resolution: explicit db param > unified resolve_db_path chain
        if db:
            db_path = str(db)
        else:
            from .profile import resolve_db_path

            db_path = str(resolve_db_path(profile, config_db_path=self._cfg.storage.db_path))
        dim = get_embedding_dim(self._cfg.embeddings.model)
        self._store = VstashStore(
            db_path,
            embedding_dim=dim,
            vector_backend=self._cfg.storage.vector_backend,
            snapvec_bits=self._cfg.storage.snapvec_bits,
            observability=self._cfg.observability,
            limits=self._cfg.limits,
        )
        # Silent killer defense: warn if the DB was built with a
        # different embedding model or a fastembed version that changed
        # pooling semantics (#132 follow-up).  Only warn for non-empty
        # stores — an empty DB has nothing to mismatch.  Wrapped in
        # try/except so mocked stores in tests (MagicMock) don't trip.
        import logging as _logging

        try:
            if self._store.stats().chunks > 0:
                drift_msg = self._store.check_embedding_drift(self._cfg.embeddings.model)
                if drift_msg:
                    _logging.getLogger("vstash").warning(drift_msg)
        except (TypeError, AttributeError):
            # Mocked store or unusual state — skip the check.
            pass

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> Memory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(
        self,
        source: str | Path,
        *,
        force: bool = False,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        tags: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[IngestResult]:
        """Ingest a file, URL, or directory into memory.

        For directories, recursively ingests all supported files while
        respecting top-level .gitignore patterns and excluding common
        non-content directories (__pycache__, node_modules, .venv, etc.).

        Args:
            source: File path, URL, or directory to ingest.
            force: Re-ingest even if the document already exists.
            collection: Override the default collection. Pass None for no collection.
            project: Override the default project tag. Pass None for no project.
            layer: Layer/category tag.
            tags: Comma-separated tags.

        Returns:
            List of IngestResult (one per file, even for single-file ingestion).
        """
        col = self._collection if collection is _UNSET else collection
        proj = self._project if project is _UNSET else project

        source_str = str(source)

        # URLs: skip Path.resolve() which can fail or be meaningless
        is_url = source_str.startswith(("http://", "https://"))

        cs = chunk_size if chunk_size is not None else self._chunk_size
        co = chunk_overlap if chunk_overlap is not None else self._chunk_overlap

        if not is_url:
            resolved = Path(source).expanduser().resolve()
            if resolved.is_dir():
                from .ingest import ingest_directory

                return ingest_directory(
                    str(resolved),
                    self._cfg,
                    self._store,
                    force=force,
                    collection=col,
                    project=proj,
                    layer=layer,
                    tags=tags,
                    chunk_size=cs,
                    chunk_overlap=co,
                )

        result = ingest(
            source_str,
            self._cfg,
            self._store,
            force=force,
            collection=col,
            project=proj,
            layer=layer,
            tags=tags,
            chunk_size=cs,
            chunk_overlap=co,
        )
        return [result]

    def remember(
        self,
        text: str,
        title: str | None = None,
        *,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        tags: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResult:
        """Ingest raw text directly — no file needed.

        Agent-friendly alternative to ``add()``. Chunks and embeds the text
        in-memory without writing a temporary file to disk.

        Args:
            text: The content to ingest.
            title: Human-readable title for the document. When *None*,
                a descriptive title is auto-generated from the text content.
            collection: Override the default collection. Pass None for no collection.
            project: Override the default project tag. Pass None for no project.
            layer: Layer/category tag.
            tags: Comma-separated tags.

        Returns:
            Single IngestResult.

        Example::

            mem = Memory(project="myproj")
            mem.remember("OAuth2 uses PKCE for public clients", title="auth-notes")
        """
        from .ingest import ingest_text

        col = self._collection if collection is _UNSET else collection
        proj = self._project if project is _UNSET else project

        return ingest_text(
            text,
            self._cfg,
            self._store,
            title=title,
            collection=col,
            project=proj,
            layer=layer,
            tags=tags,
            chunk_size=chunk_size if chunk_size is not None else self._chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else self._chunk_overlap,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        recency_boost: float = 0.0,
        added_after: str | None = None,
        added_before: str | None = None,
        mmr_lambda: float = 0.5,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        fts_only: bool = False,
    ) -> list[SearchResult]:
        """Semantic search without LLM inference.

        Returns ranked chunks from the most relevant documents.
        This method is free (no API calls) — only local embeddings + SQLite.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            collection: Override the default collection filter. Pass None for no filter.
            project: Override the default project filter. Pass None for no filter.
            layer: Filter by layer tag.
            recency_boost: Temporal decay multiplier (0.0 = off). When > 0,
                recent chunks score higher. Use ~0.5 for mild bias, ~1.0 for strong.
            added_after: ISO date (e.g. '2024-01-15') — only return results
                from documents added on or after this date.
            added_before: ISO date (e.g. '2024-06-01') — only return results
                from documents added before this date.
            vec_weight: Pin the RRF vector weight for this single call,
                overriding adaptive RRF. Valid range ``[0.0, 1.0]``.
                Pass ``None`` (default) to keep adaptive per-query
                weighting active. Passing only one of ``vec_weight`` /
                ``fts_weight`` disables adaptive weighting and sets the
                other to ``1.0 - provided`` so the pair sums to 1.0.
            fts_weight: Pin the RRF FTS weight for this single call.
                Same semantics and range as ``vec_weight``.
            fts_only: If True, run FTS5 keyword search only and skip the
                vector ANN scan, distance cutoff, and adaptive RRF
                entirely. Useful for debugging ranking, for queries
                where vector embeddings are known to be diffuse
                (cross-lingual, highly technical), or as a deliberate
                fallback when the vector pool is expected to be empty.
                MMR dedup still runs on the FTS-only result set. The
                vector embedding is still computed (it costs ~1ms and
                upstream code may need it), but it is never queried.
                Mutually exclusive with ``vec_weight`` / ``fts_weight``
                in spirit — passing both is allowed but the explicit
                weights are ignored when ``fts_only=True``.

        Returns:
            Ranked list of SearchResult ordered by relevance.
        """
        q_embedding = embed_query(query, self._cfg.embeddings.model)
        return self._store.search(
            query_embedding=q_embedding,
            query_text=query,
            top_k=top_k,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
            fts_only=fts_only,
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
            recency_boost=recency_boost,
            added_after=added_after,
            added_before=added_before,
            mmr_lambda=mmr_lambda,
        )

    def miss_analysis(
        self,
        query: str,
        *,
        expected_path: str | Path | None = None,
        expected_chunk_id: int | None = None,
        top_k: int = 5,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
    ) -> MissAnalysis:
        """Diagnose why an expected document did not appear in search results.

        Runs the same search pipeline as ``search()`` with per-stage
        instrumentation, then returns a structured trace explaining
        which stage eliminated the expected document and how to fix it.

        Args:
            query: Natural language search query.
            expected_path: Path of the document the caller expected to see.
                Either this or ``expected_chunk_id`` must be provided.
            expected_chunk_id: Specific chunk id to track instead.
            top_k: Number of results to evaluate (same as ``search()``).
            collection: Override default collection filter.
            project: Override default project filter.
            layer: Filter by layer tag.

        Returns:
            ``MissAnalysis`` with stage verdicts, the actual top-k for
            comparison, and rule-based suggestions.

        Raises:
            ValueError: if neither ``expected_path`` nor
                ``expected_chunk_id`` is provided, or the path/id resolves
                to nothing.
        """
        if expected_path is None and expected_chunk_id is None:
            raise ValueError("Provide either expected_path or expected_chunk_id")

        # Normalize the path the same way add() does so it matches what's
        # in the database (file paths get resolved, urls/text:// pass through)
        path_str: str | None = None
        if expected_path is not None:
            ps = str(expected_path)
            if ps.startswith(("http://", "https://", "text://")):
                path_str = ps
            else:
                path_str = str(Path(ps).resolve())

        q_embedding = embed_query(query, self._cfg.embeddings.model)
        return self._store.miss_analysis(
            query_embedding=q_embedding,
            query_text=query,
            expected_path=path_str,
            expected_chunk_id=expected_chunk_id,
            top_k=top_k,
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
        )

    def get_chunk(self, chunk_id: int) -> ChunkInfo | None:
        """Retrieve a single chunk by its database row ID.

        Args:
            chunk_id: Integer primary key of the chunk.

        Returns:
            ChunkInfo with text and document metadata, or None if not found.
        """
        return self._store.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: list[int]) -> list[ChunkInfo]:
        """Retrieve multiple chunks by their database row IDs.

        Args:
            chunk_ids: List of integer primary keys.

        Returns:
            List of ChunkInfo in input order. Missing IDs are skipped.
        """
        return self._store.get_chunks(chunk_ids)

    def get_document_chunks(self, path: str | Path, *, collection: object = _UNSET) -> list[str]:
        """Get all chunk texts for a document by path.

        Normalizes file paths the same way as ``add()`` (via ``Path.resolve()``)
        so that relative and absolute paths match the stored document path.

        Args:
            path: Document path (file path, URL, or text:// title).
            collection: Override the default collection filter. Omit to use the
                Memory instance default. Pass None for no filter.

        Returns:
            List of chunk texts ordered by sequence number.
        """
        path_str = str(path)
        if not path_str.startswith(("http://", "https://", "text://")):
            path_str = str(Path(path_str).resolve())
        return self._store.get_document_chunks(
            path_str, collection=self._resolve_collection(collection)
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        history: list[dict[str, str]] | None = None,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        fts_only: bool = False,
    ) -> str:
        """Search memory + generate an LLM answer.

        Retrieves the top-k relevant chunks, then sends them along with
        the query to the configured inference backend (Cerebras, Ollama,
        or OpenAI).

        Args:
            query: Natural language question.
            top_k: Number of context chunks to retrieve.
            collection: Override the default collection filter.
            project: Override the default project filter.
            layer: Filter by layer tag.
            history: Previous conversation turns for multi-turn chat.
            vec_weight: Pin the RRF vector weight on the retrieval
                call. See :meth:`search` for full semantics.
            fts_weight: Pin the RRF FTS weight on the retrieval call.
            fts_only: If True, retrieve LLM context using FTS5 only —
                no vector ANN, no distance cutoff. See :meth:`search`.

        Returns:
            Model response text.

        Raises:
            ValueError: If no inference backend is configured.
            ConnectionError: If the inference API fails.
        """
        chunks = self.search(
            query,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
            fts_only=fts_only,
        )
        return _chat_ask(query, chunks, self._cfg, history)

    def remove(self, source: str | Path) -> bool:
        """Remove a document from memory.

        Normalizes file paths the same way as ``add()`` (via ``Path.resolve()``)
        so that relative and absolute paths match the stored document path.
        URLs are passed through unchanged.

        Args:
            source: File path or URL to remove.

        Returns:
            True if the document was found and removed.
        """
        source_str = str(source)
        # Normalize file paths to match ingest() behavior
        # Skip normalization for URLs and text:// synthetic paths
        if not source_str.startswith(("http://", "https://", "text://")):
            source_str = str(Path(source_str).resolve())
        return self._store.delete_document(source_str)

    def list(
        self,
        *,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
    ) -> list[DocumentInfo]:
        """List ingested documents.

        When called without arguments, uses the constructor's project/collection
        defaults. If the constructor collection is ``"default"``, no collection
        filter is applied (returns documents from all collections). Pass an
        explicit value to override, or pass ``None`` to clear the filter.

        Args:
            collection: Filter by collection. Pass None for no filter.
            project: Filter by project. Pass None for no filter.
            layer: Filter by layer tag.

        Returns:
            List of DocumentInfo ordered by ingestion date (newest first).
        """
        return self._store.list_documents(
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
        )

    def stats(self) -> StoreStats:
        """Return aggregate memory statistics.

        Returns:
            StoreStats with document count, chunk count, DB size, and path.
        """
        return self._store.stats()

    def close(self) -> None:
        """Close the database connection.

        For long-lived processes (agents, servers), call this when done.
        When using ``Memory`` as a context manager, this is called
        automatically.
        """
        self._store.close()

    # ------------------------------------------------------------------ #
    # Journal — cross-session memory                                       #
    # ------------------------------------------------------------------ #

    def journal_save(
        self,
        text: str,
        *,
        title: str | None = None,
        tags: str | None = None,
        source: str | None = None,
    ) -> dict:
        """Save a journal entry for cross-session recall.

        Journal entries are stored in a dedicated 'journal' profile,
        separate from the main document memory.

        Args:
            text: Content to journal.
            title: Optional title (auto-generated with timestamp if None).
            tags: Comma-separated tags (auto-adds 'journal').
            source: Source identifier (e.g. 'agent', 'session', 'hook').

        Returns:
            Dict with entry metadata.

        Example::

            mem = Memory(project="my_agent")
            mem.journal_save("Decided to use OAuth2 PKCE", source="agent")
        """
        from .journal import journal_save

        return journal_save(
            text,
            title=title,
            project=self._project,
            tags=tags,
            source=source,
            cfg=self._cfg,
        )

    def journal_recall(
        self,
        query: str | None = None,
        *,
        top_k: int = 5,
    ) -> list[dict]:
        """Recall relevant journal entries from past sessions.

        When query is None, returns the most recent entries.
        When query is provided, performs semantic search.

        Args:
            query: Search query, or None for recent entries.
            top_k: Number of entries to return.

        Returns:
            List of dicts with text, title, score/added_at.

        Example::

            mem = Memory(project="my_agent")
            context = mem.journal_recall("authentication decisions")
        """
        from .journal import journal_recall

        return journal_recall(
            query=query,
            top_k=top_k,
            project=self._project,
            cfg=self._cfg,
        )

    def journal_log(
        self,
        *,
        limit: int = 20,
        recent: str | None = None,
    ) -> list[dict]:
        """Chronological view of journal entries (newest first).

        Args:
            limit: Max number of entries to return.
            recent: Time window filter (e.g. '7d', '24h', '2w').

        Returns:
            List of dicts with title, project, tags, chunks, chars, added_at.
        """
        from .journal import journal_log

        return journal_log(
            limit=limit,
            recent=recent,
            project=self._project,
            cfg=self._cfg,
        )

    def journal_prune(
        self,
        age: str,
        *,
        dry_run: bool = False,
    ) -> dict:
        """Remove journal entries older than the specified age.

        Args:
            age: Age threshold like '30d', '2w', '24h'.
            dry_run: If True, report what would be deleted without deleting.

        Returns:
            Dict with count of deleted entries and their titles.
        """
        from .journal import journal_prune

        return journal_prune(
            age,
            project=self._project,
            dry_run=dry_run,
            cfg=self._cfg,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_collection(self, override: object) -> str | None:
        """Resolve collection filter: explicit override > constructor default.

        Uses the _UNSET sentinel to distinguish "not provided" from explicit None.
        Passing None explicitly clears the filter (cross-collection search).
        """
        if override is not _UNSET:
            return override  # type: ignore[return-value]
        # Return None (no filter) if default is "default" — search everywhere
        return self._collection if self._collection != "default" else None

    def _resolve_project(self, override: object) -> str | None:
        """Resolve project filter: explicit override > constructor default.

        Uses the _UNSET sentinel to distinguish "not provided" from explicit None.
        Passing None explicitly clears the filter (cross-project search).
        """
        if override is not _UNSET:
            return override  # type: ignore[return-value]
        return self._project


# ------------------------------------------------------------------ #
# Config loader                                                        #
# ------------------------------------------------------------------ #


def _load_config_from(config: str | Path | None) -> VstashConfig:
    """Load config from a specific path or use the standard resolution.

    Args:
        config: Explicit path to vstash.toml, or None for auto-detection.

    Returns:
        Parsed VstashConfig.

    Raises:
        FileNotFoundError: If an explicit config path is provided but doesn't exist.
    """
    if config is not None:
        path = Path(config).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. Pass config=None to use auto-detection."
            )
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return VstashConfig.model_validate(raw)
    return load_config()
