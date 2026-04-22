"""Tests for vstash.memory — the Python SDK (Memory class)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vstash.config import VstashConfig
from vstash.memory import Memory, _load_config_from
from vstash.models import DocumentInfo, IngestResult, SearchResult, StoreStats

# All tests that touch the store need sqlite-vec.
# Import the skip marker from conftest.
from tests.conftest import requires_sqlite_vec


# ------------------------------------------------------------------ #
# Config loader                                                        #
# ------------------------------------------------------------------ #


class TestLoadConfig:
    """Test config resolution for the Memory class."""

    def test_load_default_config(self) -> None:
        """No path → falls back to load_config() defaults."""
        cfg = _load_config_from(None)
        assert isinstance(cfg, VstashConfig)

    def test_load_explicit_path(self, tmp_path: Path) -> None:
        """Explicit path to a vstash.toml file loads correctly."""
        toml = tmp_path / "vstash.toml"
        toml.write_text('[inference]\nbackend = "ollama"\nmodel = "test-model"\n')
        cfg = _load_config_from(str(toml))
        assert cfg.inference.backend == "ollama"
        assert cfg.inference.model == "test-model"

    def test_load_nonexistent_path_raises(self) -> None:
        """Non-existent explicit config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_config_from("/nonexistent/vstash.toml")


# ------------------------------------------------------------------ #
# Memory constructor                                                   #
# ------------------------------------------------------------------ #


class TestMemoryInit:
    """Test Memory class initialization."""

    @requires_sqlite_vec
    def test_basic_init(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        mem = Memory(db=db)
        try:
            assert mem._store is not None
            assert mem._project is None
            assert mem._collection == "default"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_init_with_project(self, tmp_path: Path) -> None:
        mem = Memory(db=tmp_path / "test.db", project="agent_x")
        try:
            assert mem._project == "agent_x"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_init_with_collection(self, tmp_path: Path) -> None:
        mem = Memory(db=tmp_path / "test.db", collection="research")
        try:
            assert mem._collection == "research"
        finally:
            mem.close()

    @requires_sqlite_vec
    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            s = mem.stats()
            assert isinstance(s, StoreStats)
        # After exiting, store should be closed
        # (accessing it would raise, but we just verify no error on exit)


# ------------------------------------------------------------------ #
# Memory.add                                                           #
# ------------------------------------------------------------------ #


class TestMemoryAdd:
    """Test document ingestion via Memory.add()."""

    @requires_sqlite_vec
    def test_add_file(self, tmp_path: Path) -> None:
        """Add a text file and verify it's ingested."""
        doc = tmp_path / "test.md"
        doc.write_text("# Hello\n\nThis is a test document with enough content to chunk.")

        with Memory(db=tmp_path / "test.db") as mem:
            results = mem.add(doc)
            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], IngestResult)
            assert results[0].status == "ok"
            assert results[0].chunks > 0

    @requires_sqlite_vec
    def test_add_with_project(self, tmp_path: Path) -> None:
        """Add with explicit project tag."""
        doc = tmp_path / "notes.txt"
        doc.write_text("Meeting notes about the project requirements and deadlines.")

        with Memory(db=tmp_path / "test.db") as mem:
            [result] = mem.add(doc, project="my_project")
            assert result.status == "ok"
            docs = mem.list(project="my_project")
            assert len(docs) >= 1

    @requires_sqlite_vec
    def test_add_duplicate_skipped(self, tmp_path: Path) -> None:
        """Adding the same file twice without force=True skips it."""
        doc = tmp_path / "dup.md"
        doc.write_text("Some content that should only be ingested once for testing.")

        with Memory(db=tmp_path / "test.db") as mem:
            [r1] = mem.add(doc)
            assert r1.status == "ok"
            [r2] = mem.add(doc)
            assert r2.status == "skipped"

    @requires_sqlite_vec
    def test_add_force_reingests(self, tmp_path: Path) -> None:
        """force=True re-ingests an existing document."""
        doc = tmp_path / "force.md"
        doc.write_text("Content that will be re-ingested with force flag enabled.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            [r2] = mem.add(doc, force=True)
            assert r2.status == "ok"


# ------------------------------------------------------------------ #
# Memory.search                                                        #
# ------------------------------------------------------------------ #


class TestMemorySearch:
    """Test semantic search via Memory.search()."""

    @requires_sqlite_vec
    def test_search_returns_results(self, tmp_path: Path) -> None:
        doc = tmp_path / "python.md"
        doc.write_text(
            "# Python Programming\n\n"
            "Python is a high-level programming language known for simplicity and readability. "
            "It supports object-oriented, functional, and procedural programming paradigms."
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            results = mem.search("programming language")
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].score > 0

    @requires_sqlite_vec
    def test_search_empty_store(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            results = mem.search("anything")
            assert results == []

    @requires_sqlite_vec
    def test_search_top_k(self, tmp_path: Path) -> None:
        doc = tmp_path / "multi.md"
        doc.write_text(
            "\n\n".join(
                f"## Section {i}\n\nContent about topic {i} with enough detail." * 5
                for i in range(10)
            )
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            results = mem.search("topic", top_k=3)
            assert len(results) <= 3

    @requires_sqlite_vec
    def test_search_forwards_rrf_weights_to_store(self, tmp_path: Path) -> None:
        """Memory.search() must actually pass vec_weight/fts_weight to the store (#151).

        The most direct assertion that the plumbing works is to spy on
        `VstashStore.search` and verify the kwargs arrive with the
        exact values that were passed to `Memory.search`. A ranking-
        based test is flaky on small corpora because RRF tends to
        converge to the same top-1 regardless of weights when there
        is only one keyword-matching doc.
        """
        (tmp_path / "doc.md").write_text(
            "Rust is a systems programming language focused on memory "
            "safety and zero-cost abstractions."
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")

            captured: list[dict[str, object]] = []
            original_search = mem._store.search

            def spy_search(*args: object, **kwargs: object) -> list[SearchResult]:
                captured.append(dict(kwargs))
                return original_search(*args, **kwargs)

            mem._store.search = spy_search  # type: ignore[method-assign]

            mem.search("memory safety", vec_weight=0.9, fts_weight=0.1)
            mem.search("memory safety", vec_weight=0.1, fts_weight=0.9)
            mem.search("memory safety")  # adaptive default

            assert len(captured) == 3
            assert captured[0]["vec_weight"] == 0.9
            assert captured[0]["fts_weight"] == 0.1
            assert captured[1]["vec_weight"] == 0.1
            assert captured[1]["fts_weight"] == 0.9
            # Default path passes None for both so the store can run
            # adaptive RRF — this is the load-bearing assertion that
            # Memory.search does NOT silently pin a default.
            assert captured[2]["vec_weight"] is None
            assert captured[2]["fts_weight"] is None

    @requires_sqlite_vec
    def test_search_weights_rejects_out_of_range(self, tmp_path: Path) -> None:
        """Out-of-range RRF weights raise a typed error (#151)."""
        import pytest

        from vstash.validation import RRFWeightOutOfRangeError

        (tmp_path / "doc.md").write_text("Some content for the store so search can run at all.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", vec_weight=1.5)
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", fts_weight=-0.1)
            with pytest.raises(RRFWeightOutOfRangeError):
                mem.search("content", vec_weight=0.7, fts_weight=2.0)

    @requires_sqlite_vec
    def test_ask_forwards_rrf_weights(self, tmp_path: Path) -> None:
        """Memory.ask() must forward vec_weight/fts_weight to the retrieval call."""
        import inspect

        sig = inspect.signature(Memory.ask)
        assert "vec_weight" in sig.parameters
        assert "fts_weight" in sig.parameters
        # Spy on the search call path via a subclass override.
        captured: dict[str, object] = {}

        class _SpyMemory(Memory):
            def search(self, query: str, **kwargs: object) -> list[SearchResult]:
                captured.update(kwargs)
                return []

        (tmp_path / "a.md").write_text("content for the spy test.")
        with _SpyMemory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "a.md")
            try:
                mem.ask("anything", vec_weight=0.3, fts_weight=0.7)
            except Exception:
                # Inference backend may not be configured in CI — that
                # is fine; we only care that search() was called with
                # the forwarded kwargs before the LLM step.
                pass
        assert captured.get("vec_weight") == 0.3
        assert captured.get("fts_weight") == 0.7

    @requires_sqlite_vec
    def test_search_fts_only_skips_vector_path(self, tmp_path: Path) -> None:
        """fts_only=True must bypass the vector ANN entirely (#152).

        The direct observable signal: a chunk that is keyword-matchable
        via FTS5 but whose embedding is orthogonal to the query MUST
        still surface under fts_only=True even with the strictest
        vector distance cutoff, because the vector path is not run.
        """
        # Doc A: contains the literal query term "kubernetes" once but
        # otherwise talks about something unrelated.
        (tmp_path / "infra.md").write_text(
            "# Personal Notes\n\n"
            "Bought a new kettle. Also read an old forum post about "
            "kubernetes, which I never use but the word stuck in my head."
        )
        # Doc B: vector-aligned distractor with no keyword overlap.
        (tmp_path / "distractor.md").write_text(
            "# Cooking Journal\n\n"
            "Made pasta today with fresh tomatoes and garlic. Recipe "
            "came out well. Planning to try a risotto tomorrow."
        )
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "infra.md")
            mem.add(tmp_path / "distractor.md")

            # fts_only MUST find the kubernetes-bearing doc regardless
            # of vector distance, because the vector path is skipped.
            # Assert on the chunk text, not the auto-generated title —
            # vstash derives titles from filenames when no frontmatter
            # is present, which is not what this test is about.
            results = mem.search("kubernetes", fts_only=True)
            assert len(results) > 0
            assert any("kubernetes" in r.text.lower() for r in results), (
                f"fts_only=True failed to surface the FTS-matching doc; "
                f"got {[(r.title, r.text[:60]) for r in results]}"
            )
            # The distractor doc has no keyword match, so only one doc
            # should appear in the FTS-only result set.
            assert all("kubernetes" in r.text.lower() for r in results), (
                "fts_only=True leaked a non-matching chunk into results"
            )

    @requires_sqlite_vec
    def test_search_forwards_retrieval_mode_to_store(self, tmp_path: Path) -> None:
        """Memory.search resolves fts_only/retrieval_mode and forwards the
        enum value to VstashStore.search (#152, #275).

        The legacy ``fts_only=True`` path is expected to emit a
        DeprecationWarning, which we filter here -- we only care that
        the resolved enum lands on the store side.
        """
        import warnings

        (tmp_path / "doc.md").write_text("Spy test content.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")

            captured: list[dict[str, object]] = []
            original_search = mem._store.search

            def spy_search(*args: object, **kwargs: object) -> list[SearchResult]:
                captured.append(dict(kwargs))
                return original_search(*args, **kwargs)

            mem._store.search = spy_search  # type: ignore[method-assign]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                mem.search("content", fts_only=True)
                mem.search("content", retrieval_mode="vec_only")
                mem.search("content")  # default -> hybrid

            assert len(captured) == 3
            assert captured[0]["retrieval_mode"] == "fts_only"
            assert captured[1]["retrieval_mode"] == "vec_only"
            assert captured[2]["retrieval_mode"] == "hybrid"
            # fts_only bool must NOT be forwarded to the store any more:
            # the mode enum is the single source of truth at the store
            # boundary.
            for call in captured:
                assert "fts_only" not in call

    @requires_sqlite_vec
    def test_search_fts_only_skips_distance_signal(self, tmp_path: Path) -> None:
        """fts_only=True must set last_best_distance to the worst value.

        The vector-search branch normally records the best cosine
        distance into ``self._store.last_best_distance`` so the
        relevance-tier signal can render "high / medium / low". In
        fts_only mode that signal is meaningless, so the store
        explicitly sets ``last_best_distance = 2.0`` (the worst
        possible cosine distance). This is an observable contract
        that proves the vector block was skipped without relying on
        monkey-patching the read-only sqlite3 execute method.
        """
        (tmp_path / "doc.md").write_text("Content that mentions rust once for fts matching.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")

            # Hybrid first: last_best_distance should be < 2.0 after
            # a query that actually finds something.
            mem.search("rust", top_k=3)
            hybrid_distance = mem._store.last_best_distance
            assert hybrid_distance < 2.0, (
                f"hybrid search should populate last_best_distance < 2.0, got {hybrid_distance}"
            )

            # FTS-only on the same query: last_best_distance must
            # be exactly 2.0 (worst case), proving the vector path
            # was bypassed.
            mem.search("rust", top_k=3, fts_only=True)
            fts_distance = mem._store.last_best_distance
            assert fts_distance == 2.0, (
                f"fts_only=True should leave last_best_distance == 2.0, got {fts_distance}"
            )

    @requires_sqlite_vec
    def test_search_fts_only_overrides_explicit_weights(self, tmp_path: Path) -> None:
        """fts_only must strip explicit vec_weight/fts_weight before validation.

        Passing ``fts_only=True, vec_weight=1.5`` is semantically
        ``fts_only=True`` — the weights are ignored. It must NOT raise
        ``RRFWeightOutOfRangeError`` because of a stale value the
        caller did not intend to validate.
        """
        (tmp_path / "doc.md").write_text("Some searchable text here.")
        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "doc.md")
            # Would raise RRFWeightOutOfRangeError without the
            # Memory.search override.
            results = mem.search(
                "text",
                fts_only=True,
                vec_weight=1.5,  # out of range — would normally raise
                fts_weight=-0.2,  # ditto
            )
            assert isinstance(results, list)

    @requires_sqlite_vec
    def test_ask_forwards_fts_only(self, tmp_path: Path) -> None:
        """Memory.ask() must forward fts_only to its internal search call (#152)."""
        captured: dict[str, object] = {}

        class _SpyMemory(Memory):
            def search(self, query: str, **kwargs: object) -> list[SearchResult]:
                captured.update(kwargs)
                return []

        (tmp_path / "a.md").write_text("content for spy.")
        with _SpyMemory(db=tmp_path / "test.db") as mem:
            mem.add(tmp_path / "a.md")
            try:
                mem.ask("anything", fts_only=True)
            except Exception:
                pass  # inference backend optional
        assert captured.get("fts_only") is True


# ------------------------------------------------------------------ #
# Memory.remove                                                        #
# ------------------------------------------------------------------ #


class TestMemoryRemove:
    """Test document removal via Memory.remove()."""

    @requires_sqlite_vec
    def test_remove_existing(self, tmp_path: Path) -> None:
        doc = tmp_path / "remove_me.md"
        doc.write_text("This document will be removed after ingestion for testing.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            assert len(mem.list()) >= 1
            removed = mem.remove(str(doc.resolve()))
            assert removed is True

    @requires_sqlite_vec
    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            removed = mem.remove("/nonexistent/file.pdf")
            assert removed is False


# ------------------------------------------------------------------ #
# Memory.list and Memory.stats                                         #
# ------------------------------------------------------------------ #


class TestMemoryListAndStats:
    """Test listing and statistics."""

    @requires_sqlite_vec
    def test_list_empty(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            docs = mem.list()
            assert docs == []

    @requires_sqlite_vec
    def test_list_returns_document_info(self, tmp_path: Path) -> None:
        doc = tmp_path / "listed.md"
        doc.write_text("A document that should appear in the list of ingested files.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            docs = mem.list()
            assert len(docs) >= 1
            assert isinstance(docs[0], DocumentInfo)

    @requires_sqlite_vec
    def test_stats_returns_store_stats(self, tmp_path: Path) -> None:
        with Memory(db=tmp_path / "test.db") as mem:
            s = mem.stats()
            assert isinstance(s, StoreStats)
            assert s.documents == 0
            assert s.chunks == 0


# ------------------------------------------------------------------ #
# Memory with project scoping                                          #
# ------------------------------------------------------------------ #


class TestMemoryProjectScoping:
    """Test that constructor-level project/collection filters work."""

    @requires_sqlite_vec
    def test_project_filter_on_search(self, tmp_path: Path) -> None:
        """Search with project scoping only returns matching docs."""
        doc_a = tmp_path / "doc_a.md"
        doc_a.write_text("Alpha project documentation about the API design and architecture.")
        doc_b = tmp_path / "doc_b.md"
        doc_b.write_text("Beta project documentation about the deployment and infrastructure.")

        db = tmp_path / "test.db"

        # Ingest with different projects using a base memory
        with Memory(db=db) as mem:
            mem.add(doc_a, project="alpha")
            mem.add(doc_b, project="beta")

        # Search scoped to alpha
        with Memory(db=db, project="alpha") as mem_alpha:
            results = mem_alpha.search("documentation")
            paths = {r.path for r in results}
            # Should find alpha, not beta
            assert any("doc_a" in p for p in paths)

    @requires_sqlite_vec
    def test_db_override(self, tmp_path: Path) -> None:
        """db= parameter overrides the config db_path."""
        custom_db = tmp_path / "custom.db"
        with Memory(db=custom_db) as mem:
            s = mem.stats()
            assert str(custom_db) in s.db_path

    @requires_sqlite_vec
    def test_explicit_none_overrides_project(self, tmp_path: Path) -> None:
        """Passing project=None explicitly clears the constructor default."""
        doc = tmp_path / "doc.md"
        doc.write_text("Content for sentinel override testing across projects.")

        db = tmp_path / "test.db"
        with Memory(db=db) as mem:
            mem.add(doc, project="alpha")

        # Memory scoped to alpha, but explicit None should search all projects
        with Memory(db=db, project="alpha") as mem:
            scoped = mem.search("content")
            unscoped = mem.search("content", project=None)
            # Both should find the doc (only one project here),
            # but the key is that project=None doesn't raise
            assert len(unscoped) >= len(scoped)

    @requires_sqlite_vec
    def test_search_auto_scopes_to_default_collection(self, tmp_path: Path) -> None:
        """Regression guard for #165: instance collection='default' must be
        honored on search(), not silently dropped.

        Prior to the fix, `_resolve_collection` returned None whenever
        the instance collection was the literal string "default",
        treating it as a sentinel for "no filter". This created a
        read/write asymmetry where writes scoped to "default" but
        reads leaked across every collection. Downstream consumers
        (engram's audit log) hit this as a silent data-leak bug.
        """
        db = tmp_path / "test.db"
        with Memory(db=db, collection="default") as mem:
            mem.remember("fact in default collection about wombats", title="t1")
            mem.remember(
                "audit row mentioning wombats elsewhere",
                title="t2",
                collection="other_coll",
                layer="audit",
            )

            # search() with no arg must honor the instance default.
            results = mem.search("wombats", top_k=5)
            titles = sorted(r.title or "" for r in results)
            assert titles == ["t1"], (
                f"Memory(collection='default').search() leaked cross-collection "
                f"rows: got titles={titles}, expected ['t1']. "
                f"This is #165 — _resolve_collection no longer shortcuts 'default' to None."
            )

            # Explicit collection=None still gives cross-collection search.
            unscoped = mem.search("wombats", top_k=5, collection=None)
            unscoped_titles = sorted(r.title or "" for r in unscoped)
            assert unscoped_titles == ["t1", "t2"], (
                f"collection=None should return both rows; got {unscoped_titles}"
            )

            # Explicit collection='other_coll' scopes to just the audit row.
            other = mem.search("wombats", top_k=5, collection="other_coll")
            other_titles = sorted(r.title or "" for r in other)
            assert other_titles == ["t2"], (
                f"collection='other_coll' should return only t2; got {other_titles}"
            )

    @requires_sqlite_vec
    def test_search_auto_scopes_to_custom_collection(self, tmp_path: Path) -> None:
        """Memory(collection='my_coll').search() scopes to my_coll (#165).

        This case worked before the #165 fix — the bug only triggered
        when the instance collection was the literal 'default'. This
        test documents the full symmetry so any future refactor of
        _resolve_collection keeps both cases consistent.
        """
        db = tmp_path / "test.db"
        with Memory(db=db, collection="my_coll") as mem:
            mem.remember("fact in my_coll about cats", title="c1")
            mem.remember("fact in other", title="c2", collection="other")

            results = mem.search("cats", top_k=5)
            titles = sorted(r.title or "" for r in results)
            assert titles == ["c1"], (
                f"Memory(collection='my_coll').search() scope failed; got {titles}"
            )

    @requires_sqlite_vec
    def test_remember_and_search_use_same_collection_resolution(self, tmp_path: Path) -> None:
        """Symmetry guard for #165: whatever collection remember() writes
        to, search() without an explicit arg must read from the same one.

        This is the deeper invariant that #165 exposed. Before the fix
        it held for custom collections but broke for 'default'. Now it
        holds for both.
        """
        for coll in ("default", "my_coll", "audit", "episodic"):
            db = tmp_path / f"sym_{coll}.db"
            with Memory(db=db, collection=coll) as mem:
                # Use longer text — vstash drops sub-chunk-size fragments
                # as "empty", which would mask the collection assertion.
                mem.remember(
                    f"distinctive marker text about kiwifruit biology "
                    f"written in the {coll} collection for symmetry testing",
                    title=f"m_{coll}",
                )
                # Also write to a sibling collection to rule out false positives.
                mem.remember(
                    f"unrelated sibling row about mountain hiking gear "
                    f"written to other_{coll} as a cross-collection distractor",
                    title=f"s_{coll}",
                    collection=f"other_{coll}",
                )

                results = mem.search("kiwifruit biology", top_k=3)
                titles = [r.title for r in results]
                assert titles == [f"m_{coll}"], (
                    f"Symmetry broken for collection={coll!r}: remember() "
                    f"wrote to {coll!r} but search() returned {titles}."
                )

    @requires_sqlite_vec
    def test_list_auto_scopes_to_default_collection(self, tmp_path: Path) -> None:
        """#166 review follow-up: the #165 fix touches every method that
        calls ``_resolve_collection``, not only ``search()``. ``list()``
        must also honor ``self._collection`` when it is the literal
        ``"default"``.

        Without this regression test, a future refactor that re-
        introduces a ``!= "default"`` shortcut in ``_resolve_collection``
        could silently break ``list()`` / ``get_document_chunks()`` /
        ``miss_analysis()`` without any existing test catching it —
        because all the search-centric tests go through ``search()``.
        """
        db = tmp_path / "test.db"
        with Memory(db=db, collection="default") as mem:
            mem.remember(
                "fact stored in the default collection about kiwifruit biology",
                title="default_row",
            )
            mem.remember(
                "audit row stored in a separate collection about kiwifruit",
                title="audit_row",
                collection="audit",
                layer="audit",
            )

            # list() without collection override must only return the
            # document in the instance's default collection. Before the
            # #165 fix, it would have leaked the audit row.
            docs = mem.list()
            titles = sorted(d.title or "" for d in docs)
            assert titles == ["default_row"], (
                f"Memory(collection='default').list() leaked cross-collection "
                f"rows: got {titles}, expected ['default_row']. The #165 fix "
                f"must apply to list() and all other _resolve_collection "
                f"callers, not just search()."
            )

            # Explicit collection=None still returns everything — regression
            # guard that the escape hatch is preserved for list() too.
            all_docs = mem.list(collection=None)
            all_titles = sorted(d.title or "" for d in all_docs)
            assert all_titles == ["audit_row", "default_row"], (
                f"list(collection=None) should return both rows; got {all_titles}"
            )


# ------------------------------------------------------------------ #
# Memory.ask                                                           #
# ------------------------------------------------------------------ #


class TestMemoryAsk:
    """Test LLM-backed ask via Memory.ask()."""

    @requires_sqlite_vec
    def test_ask_calls_chat(self, tmp_path: Path) -> None:
        """ask() retrieves chunks and passes them to chat.ask."""
        doc = tmp_path / "askable.md"
        doc.write_text("Python is a high-level language for general purpose programming.")

        with Memory(db=tmp_path / "test.db") as mem:
            mem.add(doc)
            with patch("vstash.memory._chat_ask", return_value="mocked answer") as mock:
                answer = mem.ask("what is python?")
                assert answer == "mocked answer"
                mock.assert_called_once()
                # First arg is the query
                assert mock.call_args[0][0] == "what is python?"
                # Second arg is the chunks list
                assert isinstance(mock.call_args[0][1], list)


# ------------------------------------------------------------------ #
# SDK DB resolution consistency                                        #
# ------------------------------------------------------------------ #


class TestSdkDbResolution:
    """Verify Memory SDK uses the unified resolve_db_path chain."""

    def test_sdk_uses_resolve_db_path_with_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Memory() without explicit db= delegates to resolve_db_path."""
        from vstash.profile import resolve_db_path

        custom_db = tmp_path / "custom.db"
        monkeypatch.delenv("VSTASH_DB_PATH", raising=False)
        monkeypatch.delenv("VSTASH_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        expected = str(resolve_db_path(config_db_path=str(custom_db)))

        with patch("vstash.memory.VstashStore") as mock_store:
            mock_store.return_value.close = lambda: None
            cfg = VstashConfig()
            # Simulate custom storage.db_path
            storage_cfg = cfg.storage.model_copy(update={"db_path": str(custom_db)})
            custom_cfg = cfg.model_copy(update={"storage": storage_cfg})
            with patch("vstash.memory._load_config_from", return_value=custom_cfg):
                mem = Memory()
                assert mock_store.call_args[0][0] == expected
                mem.close()

    def test_sdk_explicit_db_overrides_all(self, tmp_path: Path) -> None:
        """Memory(db=...) always uses that path, regardless of config."""
        explicit = str(tmp_path / "explicit.db")
        with patch("vstash.memory.VstashStore") as mock_store:
            mock_store.return_value.close = lambda: None
            mem = Memory(db=explicit)
            assert mock_store.call_args[0][0] == explicit
            mem.close()


# ------------------------------------------------------------------ #
# Dynamic chunk_size                                                   #
# ------------------------------------------------------------------ #


class TestDynamicChunkSize:
    """Test chunk_size/chunk_overlap parameters in Memory."""

    def test_constructor_chunk_size(self, tmp_db_path: str) -> None:
        """Constructor chunk_size is stored and used."""
        mem = Memory(db=tmp_db_path, chunk_size=2048, chunk_overlap=256)
        assert mem._chunk_size == 2048
        assert mem._chunk_overlap == 256
        mem.close()

    def test_default_chunk_size_is_none(self, tmp_db_path: str) -> None:
        mem = Memory(db=tmp_db_path)
        assert mem._chunk_size is None
        assert mem._chunk_overlap is None
        mem.close()

    def test_per_call_chunk_size_override(self, tmp_db_path: str) -> None:
        """Per-call chunk_size produces different chunk counts."""
        with Memory(db=tmp_db_path) as mem:
            # Small chunk_size → more chunks
            mem.remember("word " * 2000, title="small-chunks", chunk_size=256)
            stats1 = mem.stats()
            count_small = stats1.chunks

            mem.remember("word " * 2000, title="large-chunks", chunk_size=4096)
            stats2 = mem.stats()
            count_large = stats2.chunks - count_small

            assert count_small > count_large

    def test_constructor_chunk_size_used_by_add(self, tmp_db_path: str) -> None:
        """Constructor chunk_size flows through to add()."""
        with Memory(db=tmp_db_path, chunk_size=512) as mem:
            mem.add("tests/conftest.py")
            stats = mem.stats()
            chunk_count_512 = stats.chunks

        with Memory(db=tmp_db_path + "_large") as mem2:
            mem2.add("tests/conftest.py", chunk_size=4096)
            stats2 = mem2.stats()
            chunk_count_4096 = stats2.chunks
            mem2.close()

        # Smaller chunk_size should produce more chunks
        assert chunk_count_512 >= chunk_count_4096


class TestMemorySearchExactMatch:
    """#106 (2026-04-21): Memory.search passes exact_match through to
    the store and respects the case-sensitivity toggle."""

    def test_memory_search_forwards_exact_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text(
            "# Doc A\n\nThis paragraph mentions RateLimit as a specific identifier."
        )
        (tmp_path / "b.md").write_text(
            "# Doc B\n\nThis paragraph mentions rate limits but nothing code-ish."
        )
        with Memory(db=tmp_path / "mem.db") as mem:
            mem.add(tmp_path / "a.md")
            mem.add(tmp_path / "b.md")

            # Include the exact literal in the query so doc A is reliably
            # retrieved by FTS / vector before the post-filter runs. Without
            # this, the test could pass vacuously if the candidate pool
            # happens to exclude doc A upstream.
            strict = mem.search(
                "RateLimit rate limits",
                top_k=10,
                exact_match="RateLimit",
                exact_match_case_sensitive=True,
            )
            assert strict, "passthrough test must surface at least one hit"
            for r in strict:
                assert "RateLimit" in r.text
            paths = {r.path for r in strict}
            assert paths == {str(tmp_path / "a.md")}
