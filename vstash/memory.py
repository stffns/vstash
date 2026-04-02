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

import os
from pathlib import Path
from types import TracebackType

from .chat import ask as _chat_ask
from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .ingest import ingest
from .models import ChunkInfo, DocumentInfo, IngestResult, SearchResult, StoreStats
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
    ) -> None:
        self._cfg = _load_config_from(config)
        self._project = project
        self._collection = collection

        # Resolution: db > VSTASH_DB_PATH env > storage.db_path in toml > profile chain
        _DEFAULT_DB = "~/.vstash/memory.db"
        if db:
            db_path = str(db)
        elif os.getenv("VSTASH_DB_PATH"):
            from .profile import resolve_db_path

            db_path = str(resolve_db_path(profile))  # resolve_db_path handles env
        elif self._cfg.storage.db_path != _DEFAULT_DB:
            # User explicitly set storage.db_path in vstash.toml
            db_path = str(Path(self._cfg.storage.db_path).expanduser().resolve())
        else:
            from .profile import resolve_db_path

            db_path = str(resolve_db_path(profile))
        dim = get_embedding_dim(self._cfg.embeddings.model)
        self._store = VstashStore(
            db_path,
            embedding_dim=dim,
            vector_backend=self._cfg.storage.vector_backend,
            snapvec_bits=self._cfg.storage.snapvec_bits,
        )

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
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
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

        Returns:
            Ranked list of SearchResult ordered by relevance.
        """
        q_embedding = embed_query(query, self._cfg.embeddings.model)
        return self._store.search(
            query_embedding=q_embedding,
            query_text=query,
            top_k=top_k,
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
            scoring=self._cfg.scoring,
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
