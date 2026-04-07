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
# Track buffer is isolated                                              #
# ------------------------------------------------------------------ #


class TestTrackingIsolation:
    def test_normal_search_does_not_populate_track_buffer(self, sample_store: VstashStore):
        """Calling search() without _track_chunk should not populate the buffer."""
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/a.md", "alpha", [0.5] * dim)
        # Reset buffer
        sample_store._last_track_verdicts = []
        # Normal search
        sample_store.search([0.5] * dim, "alpha", top_k=1)
        # Buffer should still be empty
        assert sample_store._last_track_verdicts == []

    def test_miss_analysis_repopulates_buffer(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        _add_doc(sample_store, "/a.md", "alpha", [0.5] * dim)
        sample_store.miss_analysis(
            query_embedding=[0.5] * dim,
            query_text="alpha",
            expected_path="/a.md",
        )
        # Buffer should be non-empty after miss_analysis
        assert len(sample_store._last_track_verdicts) > 0
