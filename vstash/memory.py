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
from .chat import stream as _chat_stream
from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim
from .ingest import ingest
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats
from .store import VstashStore


class Memory:
    """High-level Python SDK for vstash.

    Drop any document. Ask anything. Get an answer in under a second.

    Args:
        config: Path to vstash.toml. Auto-detected if not provided.
        project: Default project tag for add/search operations.
        collection: Default collection name (default: "default").
        db: Override path to the SQLite database file.

    Example::

        from vstash import Memory

        mem = Memory(project="my_agent")
        mem.add("docs/spec.pdf")
        answer = mem.ask("What are the system requirements?")
        chunks = mem.search("deployment strategy", top_k=3)
    """

    def __init__(
        self,
        config: str | Path | None = None,
        *,
        project: str | None = None,
        collection: str = "default",
        db: str | Path | None = None,
    ) -> None:
        self._cfg = _load_config_from(config)
        self._project = project
        self._collection = collection

        # Allow db override (useful for tests and isolated agents)
        db_path = str(db) if db else self._cfg.db_path
        dim = get_embedding_dim(self._cfg.embeddings.model)
        self._store = VstashStore(db_path, embedding_dim=dim)

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
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
    ) -> IngestResult:
        """Ingest a file or URL into memory.

        Args:
            source: File path or URL to ingest.
            force: Re-ingest even if the document already exists.
            collection: Override the default collection.
            project: Override the default project tag.
            layer: Layer/category tag.
            tags: Comma-separated tags.

        Returns:
            IngestResult with status, chunk count, timing, etc.
        """
        return ingest(
            str(source),
            self._cfg,
            self._store,
            force=force,
            collection=collection or self._collection,
            project=project or self._project,
            layer=layer,
            tags=tags,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search without LLM inference.

        Returns ranked chunks from the most relevant documents.
        This method is free (no API calls) — only local embeddings + SQLite.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            collection: Override the default collection filter.
            project: Override the default project filter.
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
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
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

        Args:
            source: File path or URL to remove.

        Returns:
            True if the document was found and removed.
        """
        return self._store.delete_document(str(source))

    def list(
        self,
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> list[DocumentInfo]:
        """List ingested documents.

        Args:
            collection: Filter by collection. Uses default if not provided.
            project: Filter by project. Uses default if not provided.
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
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_collection(self, override: str | None) -> str | None:
        """Resolve collection filter: explicit override > constructor default."""
        if override is not None:
            return override
        # Return None (no filter) if default is "default" — search everywhere
        return self._collection if self._collection != "default" else None

    def _resolve_project(self, override: str | None) -> str | None:
        """Resolve project filter: explicit override > constructor default."""
        if override is not None:
            return override
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
    """
    if config is not None:
        path = Path(config).expanduser()
        if path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            return VstashConfig.model_validate(raw)
    return load_config()
