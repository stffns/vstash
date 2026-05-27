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
from typing import Literal

from ._store_open import open_store_for_config
from .chat import ask_full as _chat_ask_full
from .config import VstashConfig, load_config
from .embed import embed_query
from .ingest import ingest
from .models import (
    AskResult,
    ChunkInfo,
    DocumentInfo,
    IngestResult,
    MissAnalysis,
    SearchResult,
    StoreStats,
)
from .services import search_with_embedding

# Sentinel for distinguishing "parameter not provided" from explicit None.
_UNSET = object()


def _resolve_before(before: str) -> str:
    """Normalize a ``before=`` filter to an ISO timestamp string.

    Accepts two surface forms and resolves them to a single canonical
    representation that ``VstashStore.prune_documents`` can compare
    directly against the ``added_at`` column:

      * **Age string** (``"30d"``, ``"2w"``, ``"24h"``): parsed the
        same way as ``vstash journal prune`` and converted to
        ``now(UTC) - delta``. The ``"now"`` snapshot is taken at call
        time so back-to-back invocations of the same age cut at the
        same instant.
      * **ISO date / timestamp** (``"2026-01-15"`` or
        ``"2026-01-15T00:00:00+00:00"``): returned unchanged after
        being validated against ``datetime.fromisoformat`` so the
        store comparison uses lexical ``<`` on the column, which is
        correct under the ``YYYY-MM-DDTHH:MM:SS+00:00`` form used by
        ingest's ``datetime.now(timezone.utc).isoformat()``.

    Anything that is neither a recognised age nor a parseable ISO
    timestamp raises ``ValueError``. This is the load-bearing safety
    rail for ``prune_documents``: SQLite compares text columns
    lexically, so a free-form string like ``"invalid"`` whose first
    character ``'i'`` (ASCII 105) is greater than any plausible
    ISO-prefix character ``'2'`` (ASCII 50) would let ``added_at <
    'invalid'`` evaluate true for **every** row and silently wipe the
    store. Validating upfront makes the failure mode loud and local.

    Reusing the journal's age parser keeps the two CLIs
    (``vstash compact`` and ``vstash journal prune``) accepting the
    same vocabulary.
    """
    from datetime import datetime, timezone

    from .journal import _parse_age

    stripped = before.strip()
    if not stripped:
        raise ValueError("`before` must be a non-empty age or ISO date string")
    # Heuristic: any plain-digit-then-unit suffix (``30d``, ``2w``,
    # ``24h``) is an age. Lowercase the suffix so ``30D`` / ``2W`` /
    # ``24H`` parse too -- ``_parse_age`` is case-sensitive on the
    # unit, and rejecting the upper-case form here would be more
    # surprising than just normalising it. No length cap: any number
    # of leading digits is fine (``"100000h"`` is ~11 years, a
    # perfectly valid age cutoff), and the digit+suffix shape is
    # unambiguously distinct from any ISO date / timestamp.
    if stripped[-1].lower() in "dwh" and stripped[:-1].isdigit():
        delta = _parse_age(stripped.lower())
        return (datetime.now(timezone.utc) - delta).isoformat()
    # Not an age: must be a parseable ISO datetime. ``fromisoformat``
    # accepts both ``YYYY-MM-DD`` and the full ``YYYY-MM-DDTHH:MM:SS``
    # forms, and rejects everything else with ``ValueError``. We pass
    # the bare exception through with a friendlier message so callers
    # see WHY their input was refused without losing the original
    # cause.
    try:
        parsed = datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise ValueError(
            f"`before={before!r}` is neither a recognised age "
            f"('30d' / '2w' / '24h') nor a parseable ISO date / timestamp "
            f"('2026-01-15' / '2026-01-15T00:00:00+00:00'). "
            f"Refusing to pass through to the store because SQLite would "
            f"compare lexically and silently delete every document."
        ) from exc
    # Canonicalise to UTC. ``added_at`` is stored as
    # ``datetime.now(timezone.utc).isoformat()`` (always ``+00:00``),
    # and SQLite compares the column lexically. A naive cutoff like
    # ``2026-01-15T00:00:00-05:00`` would sort lexically AFTER
    # ``2026-01-15T00:00:00+00:00`` even though it represents an
    # earlier instant, so prune would mis-delete around timezone
    # boundaries. Treat naive inputs as UTC; convert aware inputs.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


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
        self._store = open_store_for_config(
            self._cfg,
            db_path=str(db) if db else None,
            profile=profile,
        )
        # Silent killer defense: warn if the DB was built with a
        # different embedding model or a fastembed version that changed
        # pooling semantics (#132 follow-up).  Always run — on fresh
        # stores this stamps embedding_model into store_meta so future
        # opens can detect drift.  Wrapped in try/except so mocked
        # stores in tests (MagicMock) don't trip.
        import logging as _logging

        try:
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
            collection: Override the default collection. ``documents.collection``
                is NOT NULL with default ``"default"``; passing ``None`` falls back
                to that schema default rather than storing NULL (#296).
            project: Override the default project tag. Pass None for no project.
            layer: Layer/category tag.
            tags: Comma-separated tags.

        Returns:
            List of IngestResult (one per file, even for single-file ingestion).
        """
        col = self._collection if collection is _UNSET else collection
        if col is None:
            col = "default"
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
            collection: Override the default collection. ``documents.collection``
                is NOT NULL with default ``"default"``; passing ``None`` falls back
                to that schema default rather than storing NULL (#296).
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
        if col is None:
            col = "default"
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
        tags: str | list[str] | None = None,
        mmr_lambda: float = 0.5,
        vec_weight: float | None = None,
        fts_weight: float | None = None,
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
        exact_match: str | None = None,
        exact_match_case_sensitive: bool = False,
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
            tags: Restrict to documents tagged with one or more of these
                tags (OR semantics). Accepts a comma-separated string
                (``"alpha,beta"``) or a list (``["alpha", "beta"]``).
                Comma-anchored LIKE match so ``alpha`` does not
                false-match ``alphabet``.
            vec_weight: Pin the RRF vector weight for this single call,
                overriding adaptive RRF. Valid range ``[0.0, 1.0]``.
                Pass ``None`` (default) to keep adaptive per-query
                weighting active. Passing only one of ``vec_weight`` /
                ``fts_weight`` disables adaptive weighting and sets the
                other to ``1.0 - provided`` so the pair sums to 1.0.
            fts_weight: Pin the RRF FTS weight for this single call.
                Same semantics and range as ``vec_weight``.
            exact_match: Optional literal substring the returned chunk's
                ``text`` must contain. Bypasses FTS5 tokenization
                (stemming, lowercasing) so identifiers and punctuation
                survive. #106, 2026-04-21.
            exact_match_case_sensitive: Whether ``exact_match`` is
                case-sensitive. Default False (casefold compare).
            retrieval_mode: Which search branches to run (default
                ``"hybrid"``). Three values:

                - ``"hybrid"`` (default): vector ANN + FTS5 + adaptive
                  RRF + distance cutoff + MMR. This is the pipeline the
                  paper and README benchmarks are measured against.
                - ``"fts_only"`` (#152): skip the vector ANN scan,
                  distance cutoff, and adaptive RRF. Useful for queries
                  with diffuse vector representations (cross-lingual,
                  highly technical) or as a fallback when the vector
                  pool is expected to be empty.
                - ``"vec_only"`` (#275): skip the FTS5 keyword search;
                  force ``vec_weight=1.0``, ``fts_weight=0.0``. Useful
                  when the corpus has no meaningful keyword signal
                  (tabular data, code where identifiers are noise) or
                  for benchmarking / ranking debug.

                Mutually exclusive with ``vec_weight`` / ``fts_weight``:
                non-``"hybrid"`` modes drop any explicit weight arguments
                *before* reaching ``VstashStore.search`` so a nonsensical
                pair like ``retrieval_mode="fts_only", vec_weight=1.5``
                does not raise. The mode is a stronger statement of
                intent than pinned weights.

        Returns:
            Ranked list of SearchResult ordered by relevance.
        """
        # Resolve the retrieval mode through the store helper first --
        # it handles legacy aliases (like ``fts_only=True`` bool that
        # mapped to ``"fts_only"`` pre-v0.33). After that the services
        # layer takes over: validates inputs, embeds the query, and
        # runs store.search with the right weights for the resolved
        # mode. expand_window=0 keeps Memory.search's existing
        # contract of returning raw chunks (no context expansion).
        _mode = self._store._resolve_retrieval_mode(retrieval_mode)
        return search_with_embedding(
            cfg=self._cfg,
            store=self._store,
            query=query,
            top_k=top_k,
            expand_window=0,
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
            recency_boost=recency_boost,
            added_after=added_after,
            added_before=added_before,
            tags=tags,
            mmr_lambda=mmr_lambda,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
            retrieval_mode=_mode,
            exact_match=exact_match,
            exact_match_case_sensitive=exact_match_case_sensitive,
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
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
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
            retrieval_mode: ``"hybrid"`` (default), ``"vec_only"``, or
                ``"fts_only"``. See :meth:`search` for full semantics.

        Returns:
            Model response text.

        Raises:
            ValueError: If no inference backend is configured.
            ConnectionError: If the inference API fails.
        """
        return self.ask_full(
            query,
            top_k=top_k,
            collection=collection,
            project=project,
            layer=layer,
            history=history,
            vec_weight=vec_weight,
            fts_weight=fts_weight,
            retrieval_mode=retrieval_mode,
        ).content

    def ask_full(
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
        retrieval_mode: Literal["hybrid", "vec_only", "fts_only"] | None = None,
    ) -> AskResult:
        """Like :meth:`ask`, but returns content + reasoning + usage.

        Same retrieval and prompt construction as :meth:`ask`; the only
        difference is that the LLM call goes through
        :func:`vstash.chat.ask_full` so the reasoning channel and token
        counts are preserved (issue #303). Drives Merken Phase 2
        distillation (``Q -> reasoning_trace -> A`` shape).

        Args:
            See :meth:`ask` -- arguments are identical.

        Returns:
            AskResult. ``reasoning`` is populated when the backend
            surfaces it (Cerebras gpt-oss-*, Ollama qwen3 thinking).

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
            retrieval_mode=retrieval_mode,
        )
        return _chat_ask_full(query, chunks, self._cfg, history)

    def prune(
        self,
        *,
        before: str | None = None,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        tags: str | list[str] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Delete documents matching a date / metadata filter.

        ``before`` accepts either an age string (``"30d"``, ``"2w"``,
        ``"24h"``) which is converted to an absolute UTC cutoff at the
        moment of the call, OR a full ISO date / timestamp
        (``"2026-01-15"``, ``"2026-01-15T00:00:00+00:00"``) which is
        passed straight through. At least one filter must be provided;
        an unfiltered prune raises ``ValueError`` rather than wiping
        the store silently.

        Args:
            before: Age string or ISO date. Documents whose
                ``added_at`` is strictly older are deleted.
            collection: Restrict to a single collection. Defaults to
                the ``Memory``-instance collection. Pass ``None`` to
                cover every collection.
            project: Restrict to a project. Same UNSET / None rules
                as the constructor.
            layer: Restrict to a layer.
            tags: Restrict to docs tagged with any of these tags
                (OR semantics). Same shape as the search-side tag
                filter -- comma-separated string or list, comma-anchored
                match so ``alpha`` does not false-match ``alphabet``.
            dry_run: If ``True``, return what would be deleted without
                touching the store.

        Returns:
            ``{"status": "ok"|"dry_run", "deleted": N, "paths": [...]}``.

        Example::

            mem = Memory(project="my_agent")
            mem.prune(before="90d", dry_run=True)        # preview
            mem.prune(before="90d")                       # apply
            mem.prune(layer="scratch")                    # by layer
            mem.prune(tags="archive")                     # by tag
        """
        before_iso: str | None = None
        if before is not None:
            before_iso = _resolve_before(before)

        # Use the shared resolver helpers so prune obeys the same
        # scope rules as every other read/write on this class. The
        # previous ``col_filter = None if col == "default" else col``
        # silently widened a ``Memory(collection="default")`` prune
        # to *every* collection -- exactly the #165 asymmetry
        # ``_resolve_collection`` is designed to prevent. Pass
        # ``collection=None`` explicitly when you want "all
        # collections", same as ``Memory.update`` and the CLI does.
        return self._store.prune_documents(
            before_iso=before_iso,
            collection=self._resolve_collection(collection),
            project=self._resolve_project(project),
            layer=layer,
            tags=tags,
            dry_run=dry_run,
        )

    def compact(
        self,
        *,
        before: str | None = None,
        vacuum: bool = True,
        optimize_fts: bool = True,
        collection: object = _UNSET,
        project: object = _UNSET,
        layer: str | None = None,
        tags: str | list[str] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Full housekeeping pass: prune (optional) + VACUUM + FTS optimize.

        Composes ``prune`` with the two store-level cleanup primitives
        in a single call. Each phase is independent; passing
        ``before=None`` skips the prune entirely and ``compact`` reduces
        to a pure VACUUM + optimize.

        Args:
            before: Age or ISO date for the prune phase. ``None`` skips
                pruning. See ``Memory.prune`` for the accepted formats.
            vacuum: Run ``VACUUM`` after pruning. Set ``False`` on very
                large stores where you want to defer the file rebuild
                (VACUUM can take seconds-to-minutes on multi-GB DBs).
            optimize_fts: Run the FTS5 ``'optimize'`` directive to
                compact the inverted index. Much cheaper than VACUUM;
                independent of it.
            collection / project / layer: Scope for the prune phase.
                Defaults follow ``Memory.prune``.
            dry_run: If ``True``, run the prune in dry-run mode and
                skip VACUUM / optimize entirely. Lets callers preview
                without modifying the file.

        Returns:
            ``{"prune": {...} | None, "vacuum": bool, "optimize_fts": bool}``
            describing each phase. The ``prune`` field is the same dict
            ``Memory.prune`` returns, or ``None`` when ``before`` was
            not provided.
        """
        report: dict = {"prune": None, "vacuum": False, "optimize_fts": False}
        if before is not None:
            report["prune"] = self.prune(
                before=before,
                collection=collection,
                project=project,
                layer=layer,
                tags=tags,
                dry_run=dry_run,
            )
        if dry_run:
            return report
        if vacuum:
            self._store.vacuum()
            report["vacuum"] = True
        if optimize_fts:
            self._store.optimize_fts()
            report["optimize_fts"] = True
        return report

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

    def update(
        self,
        source: str | Path,
        *,
        text: str | None = None,
        title: str | None = None,
        tags: str | None = None,
        collection: object = _UNSET,
    ) -> dict:
        """Update an existing document in place.

        Three update modes, picked from the kwargs provided:

        - **metadata-only** (``title`` and/or ``tags`` set, ``text`` is
          ``None``): runs a single SQL ``UPDATE`` against the
          ``documents`` row. No re-chunking, no re-embedding, no
          embedding model invocation. Tags pass through verbatim;
          pass ``tags=""`` to clear all tags.
        - **content** (``text`` set): re-runs the full ingest pipeline
          for the new text -- chunk + embed + store -- replacing every
          chunk of the matching document. ``title`` and ``tags`` may be
          updated in the same call; they ride on the freshly-stored
          row. The previous document row is removed first via
          ``Memory.remove`` so the new content cannot collide with stale
          chunk ids.
        - **noop** (no field provided): raises ``ValueError`` -- a
          caller that does not pass any field to update is almost
          always a bug.

        File-path normalization matches ``add()``/``remove()`` so
        relative paths resolve consistently. URLs and ``text://``
        synthetic paths pass through unchanged.

        Args:
            source: File path, URL, or ``text://`` identifier of the
                document to mutate.
            text: New content. When provided, every chunk of the doc is
                re-embedded; ``title`` falls back to ``add``'s
                title-extraction logic if not overridden.
            title: New title. Optional in both modes.
            tags: New comma-separated tag string. Optional in both modes.
                Pass an empty string to clear tags.
            collection: Override the default collection filter when
                multiple ``(collection, path)`` rows could match the
                same ``source``. Default uses the ``Memory``-instance
                collection; pass ``None`` to update every collection's
                copy of ``source``.

        Returns:
            Dict describing the update::

                {
                  "status": "updated" | "not_found" | "noop",
                  "mode":   "metadata" | "content",
                  "fields": ["title", "tags"],   # what was changed
                  "chunks": <int>,               # for content mode
                }

            ``status="not_found"`` means no row matched ``source``; the
            store was not modified. ``status="noop"`` means a content
            update produced an empty result (text too short to chunk).

        Raises:
            ValueError: No field was provided to update.

        Example::

            mem = Memory(project="my_agent")
            mem.update("docs/spec.md", title="Spec v2")
            mem.update("docs/spec.md", tags="archive,spec")
            mem.update("docs/spec.md", text=open("docs/spec.md").read())
        """
        if text is None and title is None and tags is None:
            raise ValueError(
                "Memory.update called with no field to update -- "
                "pass at least one of `text=`, `title=`, or `tags=`."
            )

        source_str = str(source)
        if not source_str.startswith(("http://", "https://", "text://")):
            source_str = str(Path(source_str).resolve())

        col = self._collection if collection is _UNSET else collection

        if text is None:
            # Metadata-only path. Use the store's atomic UPDATE
            # primitive; do not invoke the embed pipeline at all.
            #
            # ``col`` is the Memory-instance default OR the caller's
            # explicit override. We pass it through *verbatim*: the
            # ``"default"``->``None`` mapping the previous version did
            # re-introduced the #165 read/write asymmetry (a
            # ``Memory(collection="default")`` user who called
            # ``mem.update`` would have their change silently applied
            # across every collection). To update every collection
            # holding ``source``, the caller passes ``collection=None``
            # explicitly.
            updated = self._store.update_metadata(
                source_str,
                title=title,
                tags=tags,
                collection=col,
            )
            fields = [k for k, v in (("title", title), ("tags", tags)) if v is not None]
            return {
                "status": "updated" if updated else "not_found",
                "mode": "metadata",
                "fields": fields,
                "chunks": 0,
            }

        # Content update: replace the chunks in place with the
        # caller-supplied text. We deliberately do NOT re-read the
        # path from disk (the caller is overriding the content), and
        # we preserve every metadata field the caller did NOT pass so
        # tags / project / layer / source_type / collection survive a
        # content-only refresh.
        #
        # The scope is whatever the caller asked for: ``col`` is
        # ``"default"`` / explicit string / ``None`` (=all collections).
        # We do NOT collapse ``"default"`` to ``None`` -- doing so
        # would silently widen the scope and re-introduce #165.
        existing_rows = self._store._conn.execute(
            "SELECT path, title, source_type, collection, "
            "project, layer, tags, chunk_count, char_count, added_at "
            "FROM documents WHERE path = ?" + (" AND collection = ?" if col is not None else ""),
            [source_str, col] if col is not None else [source_str],
        ).fetchall()
        if not existing_rows:
            return {
                "status": "not_found",
                "mode": "content",
                "fields": [],
                "chunks": 0,
            }
        existing_docs = [DocumentInfo(**dict(row)) for row in existing_rows]

        # Chunk + embed the new text once and reuse for every
        # collection that holds this ``source``. ``chunk_text`` and
        # ``embed_texts`` are the same primitives ``ingest_text`` uses,
        # so chunking semantics match ``Memory.add`` exactly.
        from .embed import embed_texts
        from .ingest import chunk_text

        cs = self._chunk_size if self._chunk_size is not None else self._cfg.chunking.size
        co = self._chunk_overlap if self._chunk_overlap is not None else self._cfg.chunking.overlap
        new_chunks = chunk_text(text, cs, co)
        if not new_chunks:
            return {
                "status": "noop",
                "mode": "content",
                "fields": [],
                "chunks": 0,
            }
        new_embeddings = embed_texts(new_chunks, self._cfg.embeddings.model)

        # Delete-then-add, scoped to the same set of collections we
        # just listed. Re-add into *each* existing collection so a
        # multi-collection update does not silently collapse the doc
        # into a single collection. Not strictly atomic (the doc is
        # briefly missing between the two steps), but the same window
        # already exists in the ``force=True`` re-ingest path used by
        # ``add()`` and the watch worker.
        self._store.delete_document(source_str, collection=col)
        for existing in existing_docs:
            self._store.add_document(
                path=source_str,
                title=title if title is not None else existing.title,
                chunks=new_chunks,
                embeddings=new_embeddings,
                source_type=existing.source_type,
                collection=existing.collection,
                project=existing.project,
                layer=existing.layer,
                tags=tags if tags is not None else existing.tags,
            )
        fields = [k for k, v in (("text", text), ("title", title), ("tags", tags)) if v is not None]
        return {
            "status": "updated",
            "mode": "content",
            "fields": fields,
            "chunks": len(new_chunks) * len(existing_docs),
        }

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
        tags: str | list[str] | None = None,
        added_after: str | None = None,
        added_before: str | None = None,
    ) -> list[dict]:
        """Recall relevant journal entries from past sessions.

        When query is None, returns the most recent entries.
        When query is provided, performs semantic search.

        Args:
            query: Search query, or None for recent entries.
            top_k: Number of entries to return.
            tags: Restrict to journal entries tagged with one or more of
                these tags (OR semantics). Accepts a comma-separated
                string (``"work,decisions"``) or a list. Comma-anchored
                LIKE match so ``work`` does not false-match ``workflow``.
            added_after: ISO date (e.g. ``"2026-01-15"``) — only return
                entries logged on or after this date.
            added_before: ISO date — only return entries logged strictly
                before this date.

        Returns:
            List of dicts with text, title, score/added_at.

        Example::

            mem = Memory(project="my_agent")
            context = mem.journal_recall("authentication decisions")
            recent_work = mem.journal_recall(tags="work", added_after="2026-05-20")
        """
        from .journal import journal_recall

        return journal_recall(
            query=query,
            top_k=top_k,
            project=self._project,
            tags=tags,
            added_after=added_after,
            added_before=added_before,
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

        Uses the ``_UNSET`` sentinel to distinguish "not provided" from
        explicit ``None``.  Passing ``None`` explicitly clears the
        filter and searches across every collection in the database.
        Omitting the argument honors ``self._collection`` — the same
        behavior writes get.

        See #165 for the bug history: prior to v0.27 this method had
        a ``!= "default"`` shortcut that silently converted the
        instance collection to ``None`` for reads, breaking symmetry
        with writes.  That shortcut is removed.
        """
        if override is not _UNSET:
            return override  # type: ignore[return-value]
        return self._collection

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
