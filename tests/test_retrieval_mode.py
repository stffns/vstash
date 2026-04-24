"""Tests for the retrieval_mode enum (issue #275, #281).

The three modes (``hybrid``, ``vec_only``, ``fts_only``) are mutually
exclusive switches on ``VstashStore.search``. Contracts under test:

- Default behaviour (mode unset / ``"hybrid"``) is unchanged -- both
  branches run, adaptive RRF is enabled.
- ``"vec_only"`` skips the FTS5 SQL entirely and forces
  ``(vec_weight, fts_weight) = (1.0, 0.0)``.
- ``"fts_only"`` keeps its historical short-circuit semantics (#152).
- An unknown string for ``retrieval_mode`` raises ``ValueError``.
- The legacy ``fts_only=True`` bool kwarg was removed in v0.35.0
  (#281); callers still passing it hit a ``TypeError`` from
  Python's argument binder instead of a soft deprecation warning.
- The query cache distinguishes between modes (same query text in
  different modes must not collide).
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_sqlite_vec
from vstash.store import VstashStore


pytestmark = requires_sqlite_vec


DIM = 16


def _seed_store(tmp_path, dim: int = DIM) -> VstashStore:
    """Small store with two docs sharing a keyword but distinct embeddings."""
    store = VstashStore(str(tmp_path / "rm.db"), embedding_dim=dim)
    store.add_document(
        path="/rm/alpha.md",
        title="Alpha",
        chunks=["alpha mentions rust in passing"],
        embeddings=[[1.0] + [0.0] * (dim - 1)],
    )
    store.add_document(
        path="/rm/beta.md",
        title="Beta",
        chunks=["beta also mentions rust briefly"],
        embeddings=[[0.0, 1.0] + [0.0] * (dim - 2)],
    )
    return store


class TestResolver:
    """_resolve_retrieval_mode is the single source of truth for mode."""

    def test_default_is_hybrid(self) -> None:
        assert VstashStore._resolve_retrieval_mode(None) == "hybrid"

    def test_retrieval_mode_is_passthrough(self) -> None:
        assert VstashStore._resolve_retrieval_mode("vec_only") == "vec_only"
        assert VstashStore._resolve_retrieval_mode("fts_only") == "fts_only"
        assert VstashStore._resolve_retrieval_mode("hybrid") == "hybrid"

    def test_unknown_mode_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            VstashStore._resolve_retrieval_mode("semantic")

    def test_legacy_fts_only_kwarg_raises_type_error(self, tmp_path) -> None:
        """Removal of the legacy ``fts_only=True`` bool is a hard
        break in v0.35.0 (#281).  Passing it to the **public** API
        hits Python's argument binder with "unexpected keyword
        argument 'fts_only'", not a soft deprecation warning.

        Asserting on ``VstashStore.search`` rather than the internal
        resolver means this test fails if the kwarg is accidentally
        reintroduced on the public surface (the thing users actually
        call), which is the regression we care about blocking.
        """
        store = _seed_store(tmp_path)
        try:
            with pytest.raises(
                TypeError,
                match=r"unexpected keyword argument 'fts_only'",
            ):
                store.search(
                    query_embedding=[1.0] + [0.0] * (DIM - 1),
                    query_text="rust",
                    top_k=1,
                    fts_only=True,  # type: ignore[call-arg]
                )
        finally:
            store.close()


class TestVecOnlyBranch:
    """retrieval_mode='vec_only' must skip the FTS5 path.

    The observable contract: ``_fuse_rrf_scores`` receives ``fts_rows=[]``
    and ``fts_weight=0.0`` when the caller asks for ``"vec_only"``. We
    spy on the private helper rather than ``sqlite3.Connection.execute``
    (which is read-only in CPython and cannot be monkeypatched).
    """

    def test_vec_only_passes_empty_fts_rows_to_rrf(self, tmp_path, monkeypatch) -> None:
        store = _seed_store(tmp_path)
        try:
            original_fuse = store._fuse_rrf_scores
            captured: dict = {}

            def _spy_fuse(**kwargs):
                captured.update(kwargs)
                return original_fuse(**kwargs)

            monkeypatch.setattr(store, "_fuse_rrf_scores", _spy_fuse)

            store.search(
                query_embedding=[1.0] + [0.0] * (DIM - 1),
                query_text="rust",
                top_k=3,
                retrieval_mode="vec_only",
            )
            assert captured.get("fts_rows") == [], (
                f"vec_only must feed empty FTS rows to RRF; got {captured.get('fts_rows')}"
            )
            assert captured.get("fts_weight") == 0.0, (
                f"vec_only must force fts_weight=0.0; got {captured.get('fts_weight')}"
            )
            assert captured.get("vec_weight") == 1.0, (
                f"vec_only must force vec_weight=1.0; got {captured.get('vec_weight')}"
            )
        finally:
            store.close()

    def test_hybrid_feeds_nonempty_fts_rows_when_text_matches(self, tmp_path, monkeypatch) -> None:
        """Default hybrid must still populate fts_rows (non-regression)."""
        store = _seed_store(tmp_path)
        try:
            original_fuse = store._fuse_rrf_scores
            captured: dict = {}

            def _spy_fuse(**kwargs):
                captured.update(kwargs)
                return original_fuse(**kwargs)

            monkeypatch.setattr(store, "_fuse_rrf_scores", _spy_fuse)

            store.search(
                query_embedding=[1.0] + [0.0] * (DIM - 1),
                query_text="rust",
                top_k=3,
            )
            fts_rows = captured.get("fts_rows", [])
            assert len(fts_rows) >= 1, (
                "hybrid default must pass non-empty FTS rows when the corpus matches"
            )
        finally:
            store.close()


class TestCacheKeySeparation:
    """The query cache key must include retrieval_mode so different modes
    for the same query text do not collide."""

    def test_mode_separates_cache_key(self) -> None:
        kwargs = dict(
            query_embedding=[0.1] * DIM,
            query_text="rust",
            top_k=3,
            vec_weight=None,
            fts_weight=None,
            distance_cutoff=1.15,
            collection=None,
            project=None,
            layer=None,
            adaptive_rrf=True,
            recency_boost=0.0,
            added_after=None,
            added_before=None,
            mmr_lambda=0.5,
            cache_epoch=0,
        )
        h_hybrid = VstashStore._compute_search_cache_key(retrieval_mode="hybrid", **kwargs)
        h_vec = VstashStore._compute_search_cache_key(retrieval_mode="vec_only", **kwargs)
        h_fts = VstashStore._compute_search_cache_key(retrieval_mode="fts_only", **kwargs)
        assert h_hybrid != h_vec, "hybrid and vec_only keys must not collide"
        assert h_hybrid != h_fts, "hybrid and fts_only keys must not collide"
        assert h_vec != h_fts, "vec_only and fts_only keys must not collide"
