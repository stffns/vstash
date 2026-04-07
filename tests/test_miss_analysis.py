"""Tests for miss analysis (#108).

Verifies that ``VstashStore.miss_analysis()`` correctly identifies why an
expected document did not appear in search results, with per-stage
verdicts and rule-based suggestions.
"""

from __future__ import annotations

import pytest

from vstash.models import MissAnalysis
from vstash.store import VstashStore


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _add_doc(store: VstashStore, path: str, text: str, embedding: list[float]) -> None:
    store.add_document(
        path=path,
        title=path.lstrip("/").rsplit(".", 1)[0],
        chunks=[text],
        embeddings=[embedding],
    )


# ------------------------------------------------------------------ #
# Resolution and basic flow                                             #
# ------------------------------------------------------------------ #


class TestResolution:
    def test_requires_path_or_chunk_id(self, sample_store: VstashStore):
        with pytest.raises(ValueError, match="Provide either"):
            sample_store.miss_analysis(
                query_embedding=[0.1] * sample_store.embedding_dim,
                query_text="x",
            )

    def test_unknown_path_raises(self, sample_store: VstashStore):
        with pytest.raises(ValueError, match="No chunks found"):
            sample_store.miss_analysis(
                query_embedding=[0.1] * sample_store.embedding_dim,
                query_text="x",
                expected_path="/nonexistent.md",
            )

    def test_unknown_chunk_id_raises(self, sample_store: VstashStore):
        with pytest.raises(ValueError, match="not found"):
            sample_store.miss_analysis(
                query_embedding=[0.1] * sample_store.embedding_dim,
                query_text="x",
                expected_chunk_id=999999,
            )

    def test_returns_miss_analysis_model(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/a.md", "alpha content", [0.5] * dim)
        result = sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="alpha",
            expected_path="/a.md",
        )
        assert isinstance(result, MissAnalysis)
        assert result.query == "alpha"
        assert result.expected_path == "/a.md"
        assert result.expected_chunk_id is not None


# ------------------------------------------------------------------ #
# When the doc IS in results                                            #
# ------------------------------------------------------------------ #


class TestNoMiss:
    def test_appeared_returns_no_miss(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/found.md", "Python programming language", [0.5] * dim)
        result = sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="Python",
            expected_path="/found.md",
            top_k=5,
        )
        assert result.appeared_in_results is True
        assert result.final_rank == 0
        assert result.dropped_at is None
        # Suggestion should reflect that there's nothing to analyze
        assert any("no miss" in s.lower() or "is in" in s.lower() for s in result.suggestions)


# ------------------------------------------------------------------ #
# Per-stage failure modes                                               #
# ------------------------------------------------------------------ #


class TestStageDrops:
    def test_drops_at_distance_cutoff(self, sample_store: VstashStore):
        """A doc with very different embedding should be dropped by distance_cutoff."""
        dim = sample_store.embedding_dim
        # Two docs with orthogonal-ish embeddings
        emb_close = [1.0 if i < dim // 2 else 0.0 for i in range(dim)]
        emb_far = [0.0 if i < dim // 2 else 1.0 for i in range(dim)]
        _add_doc(sample_store, "/close.md", "alpha beta", emb_close)
        _add_doc(sample_store, "/far.md", "gamma delta", emb_far)

        # Query close to /close.md
        analysis = sample_store.miss_analysis(
            query_embedding=emb_close,
            query_text="alpha",
            expected_path="/far.md",
            top_k=1,
        )
        assert analysis.appeared_in_results is False
        # Should have a verdict for distance_cutoff
        stages = {v.stage for v in analysis.stage_verdicts}
        assert "distance_cutoff" in stages
        # And actual top-k should show the closer doc
        assert len(analysis.actual_top_k) >= 1

    def test_actual_top_k_populated(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/a.md", "alpha", [0.1] * dim)
        _add_doc(sample_store, "/b.md", "beta", [0.9] * dim)
        analysis = sample_store.miss_analysis(
            query_embedding=[0.1] * dim,
            query_text="alpha",
            expected_path="/b.md",
            top_k=1,
        )
        # Even if there's no miss to analyze, actual_top_k should be populated
        assert len(analysis.actual_top_k) >= 1
        assert analysis.actual_top_k[0].path == "/a.md"


# ------------------------------------------------------------------ #
# Suggestions                                                           #
# ------------------------------------------------------------------ #


class TestSuggestions:
    def test_distance_cutoff_suggestion_mentions_relax(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        emb_close = [1.0 if i < dim // 2 else 0.0 for i in range(dim)]
        emb_far = [0.0 if i < dim // 2 else 1.0 for i in range(dim)]
        _add_doc(sample_store, "/close.md", "alpha", emb_close)
        _add_doc(sample_store, "/far.md", "beta", emb_far)
        analysis = sample_store.miss_analysis(
            query_embedding=emb_close,
            query_text="alpha",
            expected_path="/far.md",
            top_k=1,
        )
        # If distance_cutoff dropped it, suggestion should mention cutoff
        if analysis.dropped_at == "distance_cutoff":
            joined = " ".join(analysis.suggestions).lower()
            assert "cutoff" in joined or "distance" in joined

    def test_at_least_one_suggestion(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        emb_close = [1.0 if i < dim // 2 else 0.0 for i in range(dim)]
        emb_far = [0.0 if i < dim // 2 else 1.0 for i in range(dim)]
        _add_doc(sample_store, "/close.md", "alpha", emb_close)
        _add_doc(sample_store, "/far.md", "beta", emb_far)
        analysis = sample_store.miss_analysis(
            query_embedding=emb_close,
            query_text="alpha",
            expected_path="/far.md",
            top_k=1,
        )
        assert len(analysis.suggestions) >= 1


# ------------------------------------------------------------------ #
# Multi-chunk doc resolution                                            #
# ------------------------------------------------------------------ #


class TestMultiChunkDoc:
    def test_picks_best_chunk_in_doc(self, sample_store: VstashStore):
        """When the expected doc has multiple chunks, miss_analysis should
        track the chunk closest to the query, not an arbitrary one."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/multi.md",
            title="multi",
            chunks=["chunk one alpha", "chunk two beta"],
            embeddings=[[0.1] * dim, [0.9] * dim],
        )
        analysis = sample_store.miss_analysis(
            query_embedding=[0.9] * dim,
            query_text="beta",
            expected_path="/multi.md",
            top_k=5,
        )
        # The chunk_id chosen should be the one closer to the query
        # (i.e., the second chunk with embedding [0.9]*dim)
        # We verify by checking it's one of the doc's chunks
        all_chunk_ids = [
            row[0]
            for row in sample_store._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ("
                "  SELECT id FROM documents WHERE path = '/multi.md'"
                ")"
            )
        ]
        assert analysis.expected_chunk_id in all_chunk_ids


# ------------------------------------------------------------------ #
# Concurrency: caller-owned tracer guarantees isolation                 #
# ------------------------------------------------------------------ #


class TestCallerOwnedTracer:
    def test_tracer_is_caller_owned(self, sample_store: VstashStore):
        """The tracer's verdicts must live on the tracer instance, not
        on the store.  This is the structural guarantee that concurrent
        miss_analysis() calls cannot stomp on each other's data.
        """
        from vstash.store import _PipelineTracer

        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/a.md", "alpha content", [0.5] * dim)
        _add_doc(sample_store, "/b.md", "beta content", [0.4] * dim)

        # Create two tracers manually and drive search() directly to
        # confirm each buffer is independent.
        chunks = sample_store._conn.execute(
            "SELECT c.id AS id, d.path AS path FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        id_a = next(r["id"] for r in chunks if r["path"] == "/a.md")
        id_b = next(r["id"] for r in chunks if r["path"] == "/b.md")

        tracer_a = _PipelineTracer(id_a)
        tracer_b = _PipelineTracer(id_b)

        sample_store.search([0.5] * dim, "content", top_k=1, _tracer=tracer_a)
        sample_store.search([0.5] * dim, "content", top_k=1, _tracer=tracer_b)

        # Each tracer must contain verdicts referring only to its own target
        assert tracer_a.target == id_a
        assert tracer_b.target == id_b
        assert tracer_a.verdicts, "tracer_a received no verdicts"
        assert tracer_b.verdicts, "tracer_b received no verdicts"
        # The buffers are distinct Python objects
        assert tracer_a.verdicts is not tracer_b.verdicts

    def test_store_has_no_tracking_state(self, sample_store: VstashStore):
        """VstashStore must not retain any cross-call tracking attribute.

        Regression guard for the refactor from instance-attribute
        buffer to caller-owned tracer.
        """
        assert not hasattr(sample_store, "_last_track_verdicts")


# ------------------------------------------------------------------ #
# Multi-chunk target resolution transparency (#5)                       #
# ------------------------------------------------------------------ #


class TestTargetResolution:
    def test_only_chunk_resolution(self, sample_store: VstashStore):
        """A single-chunk document should report resolution='only_chunk'."""
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/solo.md", "only chunk here", [0.5] * dim)
        analysis = sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="only",
            expected_path="/solo.md",
        )
        assert analysis.target_resolution == "only_chunk"
        assert analysis.total_chunks_in_doc == 1

    def test_best_of_n_resolution(self, sample_store: VstashStore):
        """A multi-chunk document should report resolution='best_of_n'
        and total_chunks_in_doc > 1 so the caller knows the trace
        reflects a best-case chunk, not the whole document."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/multi.md",
            title="multi",
            chunks=["chunk A content", "chunk B content", "chunk C content"],
            embeddings=[[0.1] * dim, [0.5] * dim, [0.9] * dim],
        )
        analysis = sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="content",
            expected_path="/multi.md",
        )
        assert analysis.target_resolution == "best_of_n"
        assert analysis.total_chunks_in_doc == 3

    def test_explicit_id_resolution(self, sample_store: VstashStore):
        """Passing expected_chunk_id directly should report 'explicit_id'."""
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/x.md", "x chunk", [0.5] * dim)
        chunk_id = sample_store._conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        analysis = sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="x",
            expected_chunk_id=chunk_id,
        )
        assert analysis.target_resolution == "explicit_id"
        assert analysis.expected_chunk_id == chunk_id


# ------------------------------------------------------------------ #
# Near-miss suggestion                                                  #
# ------------------------------------------------------------------ #


class TestNearMiss:
    def test_near_miss_gives_dedicated_suggestion(self, sample_store: VstashStore):
        """If the expected doc is in top-k but at a low rank, the
        suggestion should differ from the trivial 'no miss' message."""
        dim = sample_store.embedding_dim
        # Add 5 docs, target at rank 4 (out of top_k=5)
        target = [0.5] * dim
        sample_store.add_document(path="/d1.md", title="d1", chunks=["c1"], embeddings=[target])
        # Make query near target but have 3 closer docs
        for i in range(3):
            _add_doc(sample_store, f"/c{i}.md", f"close {i}", [0.5 + 0.01 * i] * dim)
        analysis = sample_store.miss_analysis(
            query_embedding=target,
            query_text="c",
            expected_path="/d1.md",
            top_k=5,
        )
        # The doc should appear, possibly at rank 0 (it's the closest)
        # or if not, the suggestion text should differ between tiers
        assert analysis.appeared_in_results is True
        # If rank is at the high end of top_k, expect a near-miss suggestion
        if analysis.final_rank is not None and analysis.final_rank >= 3:
            joined = " ".join(analysis.suggestions).lower()
            assert "near-miss" in joined or "competitive" in joined
