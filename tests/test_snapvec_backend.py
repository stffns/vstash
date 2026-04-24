"""Tests for the optional snapvec vector search backend."""

from __future__ import annotations

import numpy as np  # noqa: E402
import pytest

pytest.importorskip("snapvec", reason="snapvec not installed")

from vstash.store import VstashStore, _HAS_SNAPVEC  # noqa: E402


DIM = 32  # small dim for fast tests


@pytest.fixture
def snap_store(tmp_path):
    """Create a VstashStore with snapvec backend."""
    db = str(tmp_path / "snap_test.db")
    store = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
    yield store
    store.close()


@pytest.fixture
def populated_snap_store(snap_store):
    """Snap store with two documents."""
    # Document 1
    emb1 = [np.random.default_rng(1).standard_normal(DIM).tolist() for _ in range(3)]
    snap_store.add_document(
        path="/test/doc1.md",
        title="Document One",
        chunks=["chunk one alpha", "chunk one beta", "chunk one gamma"],
        embeddings=emb1,
        source_type="markdown",
    )
    # Document 2
    emb2 = [np.random.default_rng(2).standard_normal(DIM).tolist() for _ in range(2)]
    snap_store.add_document(
        path="/test/doc2.md",
        title="Document Two",
        chunks=["chunk two delta", "chunk two epsilon"],
        embeddings=emb2,
        source_type="markdown",
    )
    return snap_store


class TestSnapvecDetection:
    def test_has_snapvec(self):
        assert _HAS_SNAPVEC is True

    def test_snap_index_created(self, snap_store):
        assert snap_store._snap is not None
        assert snap_store._vector_backend == "snapvec"

    def test_fallback_when_sqlite_vec(self, tmp_path):
        """Default backend should not create a SnapIndex."""
        db = str(tmp_path / "fallback.db")
        store = VstashStore(db, embedding_dim=DIM)
        assert store._snap is None
        assert store._vector_backend == "sqlite-vec"
        store.close()


class TestSnapvecAdd:
    def test_add_populates_snap_index(self, snap_store):
        emb = [np.random.default_rng(42).standard_normal(DIM).tolist()]
        snap_store.add_document(
            path="/test/single.md",
            title="Single",
            chunks=["hello world"],
            embeddings=emb,
        )
        assert len(snap_store._snap) == 1

    def test_add_multiple_chunks(self, populated_snap_store):
        assert len(populated_snap_store._snap) == 5  # 3 + 2


class TestSnapvecSearch:
    def test_search_returns_results(self, populated_snap_store):
        query_emb = np.random.default_rng(1).standard_normal(DIM).tolist()
        results = populated_snap_store.search(
            query_embedding=query_emb,
            query_text="chunk alpha",
            top_k=3,
        )
        assert len(results) > 0
        assert all(r.text for r in results)

    def test_search_respects_top_k(self, populated_snap_store):
        query_emb = np.random.default_rng(1).standard_normal(DIM).tolist()
        results = populated_snap_store.search(
            query_embedding=query_emb,
            query_text="chunk",
            top_k=2,
        )
        assert len(results) <= 2


class TestSnapvecDistanceSemantics:
    """Regression tests for #289.

    ``snapvec.SnapIndex.search`` returns (id, similarity) in [-1, 1],
    but downstream ``distance_cutoff``, ``relevance_tier``, and
    ``last_best_distance`` all live in cosine-distance space [0, 2].
    The store must convert on read.  Before the fix, these invariants
    all failed: a perfect match reported distance ~0.94 (similarity),
    classified ``"low"`` by ``relevance_tier``, and the distance
    cutoff never fired.
    """

    def test_last_best_distance_stays_in_cosine_range(self, snap_store):
        """``last_best_distance`` must be in [0, 2] after a snapvec
        search, matching the sqlite-vec backend contract.  A same-
        vector query should land in the ``"high"`` relevance tier,
        not ``"low"`` (which would mean similarity was leaking
        through and ~0.94 was treated as distance).
        """
        from vstash.store import RELEVANCE_TIER_HIGH_MAX

        emb = np.random.default_rng(0).standard_normal(DIM).tolist()
        snap_store.add_document(
            path="/s/a.md",
            title="a",
            chunks=["hello"],
            embeddings=[emb],
        )
        snap_store.search(query_embedding=emb, query_text="hello", top_k=1)
        assert 0.0 <= snap_store.last_best_distance <= 2.0
        # Contract-based upper bound: a same-vector query must land
        # in the ``"high"`` tier regardless of the bit-depth /
        # quantization quirks of the snapvec backend.  Hardcoding a
        # specific delta (e.g. 0.15) would couple the test to the
        # 4-bit quantization error.
        assert snap_store.last_best_distance <= RELEVANCE_TIER_HIGH_MAX

    def test_relevance_tier_high_for_near_match(self, snap_store):
        """A near-identical query produces ``"high"`` confidence under
        cosine-space thresholds (``RELEVANCE_TIER_HIGH_MAX = 0.4513``).
        Without the similarity->distance conversion, the store would
        report the cosine similarity (~0.94) which falls in ``"low"``.
        """
        from vstash.store import relevance_tier

        emb = np.random.default_rng(7).standard_normal(DIM).tolist()
        snap_store.add_document(
            path="/s/h.md",
            title="h",
            chunks=["topic"],
            embeddings=[emb],
        )
        snap_store.search(query_embedding=emb, query_text="topic", top_k=1)
        assert relevance_tier(snap_store.last_best_distance) == "high"

    def test_distance_cutoff_drops_far_chunks(self, snap_store):
        """The distance cutoff must actually gate results: an embedding
        that points away from the query should be filterable.  Under
        the pre-fix behaviour the similarity-as-distance inversion made
        the cutoff a no-op for snapvec.
        """
        # Two clearly different directions in 32-dim space.
        close_emb = np.zeros(DIM, dtype=np.float32)
        close_emb[0] = 1.0
        far_emb = np.zeros(DIM, dtype=np.float32)
        far_emb[16] = 1.0  # orthogonal to close_emb
        snap_store.add_document(
            path="/s/close.md",
            title="close",
            chunks=["near topic"],
            embeddings=[close_emb.tolist()],
        )
        snap_store.add_document(
            path="/s/far.md",
            title="far",
            chunks=["distant topic"],
            embeddings=[far_emb.tolist()],
        )
        # Query is ``close_emb`` with a small perturbation on a
        # different index.  That guarantees cos_dist to the close
        # chunk is > 0 (tiny, but nonzero), so we do not accidentally
        # trigger the ``best_distance == 0`` cutoff-bypass in
        # ``VstashStore.search`` (store.py:2368).  Without the
        # perturbation, a snapvec build that returned an exact
        # similarity of 1.0 on the identical vector would skip the
        # cutoff entirely and leave the far chunk in the results,
        # masking a real regression as a pass.
        query_emb = close_emb.copy()
        query_emb[1] = 0.05

        # Tight cutoff: with correct distances the orthogonal chunk
        # (cos_dist ~1.0) must be dropped when the near chunk is at
        # cos_dist ~0.  ``retrieval_mode="vec_only"`` isolates the
        # vec path so the FTS hit on the shared word "topic" cannot
        # reintroduce the far chunk after the cutoff.
        results = snap_store.search(
            query_embedding=query_emb.tolist(),
            query_text="topic",
            top_k=5,
            distance_cutoff=1.15,
            adaptive_rrf=False,
            retrieval_mode="vec_only",
        )
        paths = {r.path for r in results}
        assert "/s/close.md" in paths
        assert "/s/far.md" not in paths, (
            f"distance_cutoff failed to drop the orthogonal chunk under snapvec; "
            f"paths={paths}, last_best_distance={snap_store.last_best_distance}"
        )


class TestSnapvecDelete:
    def test_delete_removes_from_snap_index(self, populated_snap_store):
        initial_count = len(populated_snap_store._snap)
        populated_snap_store.delete_document("/test/doc1.md")
        assert len(populated_snap_store._snap) == initial_count - 3

    def test_delete_nonexistent(self, populated_snap_store):
        count_before = len(populated_snap_store._snap)
        result = populated_snap_store.delete_document("/nonexistent.md")
        assert result is False
        assert len(populated_snap_store._snap) == count_before


class TestSnapvecPersistence:
    def test_save_and_reload(self, tmp_path):
        """SnapIndex should persist and reload across store instances."""
        db = str(tmp_path / "persist.db")

        # Create and populate
        store1 = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        emb = [np.random.default_rng(99).standard_normal(DIM).tolist() for _ in range(3)]
        store1.add_document(
            path="/test/persist.md",
            title="Persist",
            chunks=["a", "b", "c"],
            embeddings=emb,
        )
        store1.close()

        # Reload
        store2 = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        assert store2._snap is not None
        assert len(store2._snap) == 3

        # Search should still work
        results = store2.search(
            query_embedding=emb[0],
            query_text="a",
            top_k=3,
        )
        assert len(results) > 0
        store2.close()


class TestSnapvecReindex:
    def test_reindex_rebuilds_snap_index(self, populated_snap_store):
        def fake_embed(texts):
            rng = np.random.default_rng(0)
            return [rng.standard_normal(DIM).tolist() for _ in texts]

        count = populated_snap_store.reindex(fake_embed, new_dim=DIM)
        assert count == 5
        assert len(populated_snap_store._snap) == 5

    def test_reindex_handles_more_than_one_batch(self, tmp_path):
        """Reindex must paginate past a single batch AND avoid the two
        O(N^2) patterns fixed in issue #265.

        Counting chunks alone would not catch a regression to the old
        OFFSET + per-page behaviour because the final state is the same.
        We observe the SQL stream to assert keyset pagination and wrap
        ``snap.add_batch`` to assert coalescing.
        """
        db = str(tmp_path / "multibatch.db")
        with VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4) as store:
            n_chunks = 53
            rng = np.random.default_rng(0xBEEF)
            for i in range(n_chunks):
                store.add_document(
                    path=f"/test/doc_{i}.md",
                    title=f"doc {i}",
                    chunks=[f"chunk body {i}"],
                    embeddings=[rng.standard_normal(DIM).tolist()],
                )
            assert len(store._snap) == n_chunks

            def fake_embed(texts):
                inner = np.random.default_rng(0xCAFE)
                return [inner.standard_normal(DIM).tolist() for _ in texts]

            statements: list[str] = []
            store._conn.set_trace_callback(statements.append)

            # reindex re-creates ``self._snap`` with a fresh SnapIndex on
            # entry (store.py around line 4276), so a monkey-patch on the
            # pre-reindex instance would be discarded. Patch the class
            # method for the duration of the call instead.
            from snapvec import SnapIndex

            original_add_batch = SnapIndex.add_batch
            add_batch_calls: list[int] = []

            def counting_add_batch(self, ids, vectors):
                add_batch_calls.append(len(ids))
                return original_add_batch(self, ids, vectors)

            SnapIndex.add_batch = counting_add_batch  # type: ignore[method-assign]

            try:
                count = store.reindex(fake_embed, new_dim=DIM, batch_size=10)
            finally:
                SnapIndex.add_batch = original_add_batch  # type: ignore[method-assign]
                store._conn.set_trace_callback(None)

            assert count == n_chunks
            assert len(store._snap) == n_chunks

            import re

            # ``chunk_offset`` is a column in sqlite-vec's internal
            # vec_chunks_rowids table and would false-positive a plain
            # substring match. Look for OFFSET as a SQL keyword instead.
            offset_kw = re.compile(r"\bOFFSET\b", re.IGNORECASE)
            chunks_selects = [
                s for s in statements if re.search(r"FROM\s+chunks\b", s, re.IGNORECASE)
            ]
            assert not any(offset_kw.search(s) for s in chunks_selects), (
                "reindex must use keyset pagination, not LIMIT ? OFFSET ?"
            )
            assert any("WHERE id >" in s for s in chunks_selects), (
                "expected keyset pagination with `WHERE id > ?`"
            )

            assert len(add_batch_calls) == 1, (
                f"snap.add_batch should be coalesced into one call, got {len(add_batch_calls)}"
            )
            assert add_batch_calls[0] == n_chunks

    def test_reindex_rejects_nonpositive_batch_size(self, tmp_path):
        """``batch_size <= 0`` must raise BEFORE dropping vec_chunks.

        Regression guard: the keyset loop terminates immediately on
        ``batch_size=0``, and if we had already dropped vec_chunks the
        store would be silently wiped. The validation lives at the top
        of reindex() precisely to prevent that class of data loss.
        """
        with VstashStore(
            str(tmp_path / "bad_batch.db"),
            embedding_dim=DIM,
            vector_backend="snapvec",
            snapvec_bits=4,
        ) as store:
            store.add_document(
                path="/x.md",
                title="x",
                chunks=["x"],
                embeddings=[np.random.default_rng(0).standard_normal(DIM).tolist()],
            )

            def embed_fn(texts):
                return [[0.0] * DIM for _ in texts]

            with pytest.raises(ValueError, match="batch_size must be >= 1"):
                store.reindex(embed_fn, new_dim=DIM, batch_size=0)

            # vec_chunks must still contain the original row.
            assert len(store._snap) == 1

    def test_reindex_rejects_embed_fn_length_mismatch(self, tmp_path):
        """``embed_fn`` that returns wrong count must fail fast.

        Without the length check, ``executemany`` would silently truncate
        against the shorter zip; the store would end up with a partial
        vector index while ``reindex`` reports success. The validation
        inside the per-batch loop raises instead.
        """
        with VstashStore(
            str(tmp_path / "bad_embed.db"),
            embedding_dim=DIM,
            vector_backend="snapvec",
            snapvec_bits=4,
        ) as store:
            for i in range(3):
                store.add_document(
                    path=f"/doc_{i}.md",
                    title=f"doc {i}",
                    chunks=[f"text {i}"],
                    embeddings=[np.random.default_rng(i).standard_normal(DIM).tolist()],
                )

            def short_embed(texts):
                # Drop one: simulate a buggy embed_fn that silently
                # loses an item.
                return [[0.0] * DIM for _ in texts[:-1]]

            with pytest.raises(ValueError, match="embed_fn returned"):
                store.reindex(short_embed, new_dim=DIM, batch_size=256)


class TestSnapvecDimMismatch:
    def test_dim_mismatch_rebuilds(self, tmp_path):
        """If saved index has wrong dim, it should rebuild gracefully."""
        db = str(tmp_path / "mismatch.db")

        # Create with dim=32
        store1 = VstashStore(db, embedding_dim=32, vector_backend="snapvec", snapvec_bits=4)
        emb = [[0.1] * 32]
        store1.add_document(path="/test/x.md", title="X", chunks=["x"], embeddings=emb)
        store1.close()

        # Reopen with dim=64 — should warn and rebuild, not crash
        store2 = VstashStore(db, embedding_dim=64, vector_backend="snapvec", snapvec_bits=4)
        assert store2._snap is not None
        assert store2._snap.dim == 64
        assert len(store2._snap) == 0  # rebuilt empty
        store2.close()


class TestSnapvecDeferredSave:
    """Flat snapvec now defers ``_save_snapvec`` to ``close()``. These
    tests lock in the contract and the crash-recovery path that
    rebuilds ``.snpv`` from ``vec_chunks`` when the deferred write
    never happened."""

    def test_add_document_does_not_write_snpv_before_close(self, snap_store, tmp_path):
        """Writing through ``add_document`` does not touch the .snpv
        file on disk until ``close()``. Regression guard for the
        O(N^2) per-doc-ingest bug (2026-04-19)."""
        db_path = tmp_path / "snap_test.db"
        snpv_path = db_path.with_suffix(".snpv")
        # Empty store just got its initial empty .snpv written.
        initial_mtime = snpv_path.stat().st_mtime if snpv_path.exists() else None

        for i in range(20):
            snap_store.add_document(
                path=f"/defer/doc{i}.md",
                title=f"doc{i}",
                chunks=[f"chunk text {i}"],
                embeddings=[np.random.default_rng(i).standard_normal(DIM).tolist()],
            )

        # The ONLY expected write to .snpv during add_document is the
        # initial empty-index save inside ``_init_snapvec``. After that
        # every add_document must set ``_snap_dirty`` and not touch disk.
        assert snap_store._snap_dirty is True
        if initial_mtime is not None:
            assert snpv_path.stat().st_mtime == initial_mtime, (
                "flat snapvec must not rewrite .snpv on every add_document; "
                "expected a single deferred flush via _checkpoint_snapvec / close()"
            )

        # Explicit checkpoint flushes dirty state to disk.
        snap_store._checkpoint_snapvec()
        assert snap_store._snap_dirty is False
        assert snpv_path.exists()

    def test_close_flushes_pending_writes(self, tmp_path):
        """``close()`` must call ``_checkpoint_snapvec`` so that a
        clean shutdown persists every in-memory vector."""
        db = str(tmp_path / "close_flush.db")
        store = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        for i in range(10):
            store.add_document(
                path=f"/flush/doc{i}.md",
                title=f"doc{i}",
                chunks=[f"flush chunk {i}"],
                embeddings=[np.random.default_rng(i).standard_normal(DIM).tolist()],
            )
        assert store._snap_dirty is True
        store.close()

        # Reopen and verify all 10 docs survived.
        reopened = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        assert len(reopened._snap) == 10
        reopened.close()

    def test_crash_recovery_rebuilds_from_vec_chunks(self, tmp_path):
        """Simulate a crash between ``add_document`` and ``close()`` by
        intentionally skipping ``_checkpoint_snapvec`` on the first
        handle. On next open, ``_init_snapvec`` must detect the
        staleness (snap len < vec_chunks count) and rebuild the flat
        index from ``vec_chunks`` so no vectors are silently lost."""
        db = str(tmp_path / "crash_recovery.db")
        store = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        for i in range(15):
            store.add_document(
                path=f"/crash/doc{i}.md",
                title=f"doc{i}",
                chunks=[f"crash chunk {i}"],
                embeddings=[np.random.default_rng(i).standard_normal(DIM).tolist()],
            )
        # "Crash": close the sqlite connection but NEVER flush the
        # deferred .snpv. The on-disk .snpv is the initial empty one
        # that got written at _init_snapvec time.
        store._conn.close()
        # Avoid any further finalisation on this handle.
        store._snap = None

        # Reopen and verify recovery.
        recovered = VstashStore(db, embedding_dim=DIM, vector_backend="snapvec", snapvec_bits=4)
        assert recovered._snap is not None
        # All 15 vectors must be back in the in-memory index, populated
        # by _rebuild_snapvec_from_vec_chunks.
        assert len(recovered._snap) == 15
        recovered.close()

    def test_rebuild_helper_is_idempotent(self, populated_snap_store, tmp_path):
        """Calling ``_rebuild_snapvec_from_vec_chunks`` twice in a row
        produces the same state."""
        populated_snap_store._checkpoint_snapvec()
        first = populated_snap_store._rebuild_snapvec_from_vec_chunks()
        second = populated_snap_store._rebuild_snapvec_from_vec_chunks()
        assert first == second
        assert first > 0
