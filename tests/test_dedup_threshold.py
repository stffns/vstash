"""Cross-document near-duplicate collapse (``dedup_threshold``).

``_mmr_dedup`` penalises only same-document siblings, so a corpus holding
many near-identical *documents* -- audit logs, mirrored notes, re-ingested
revisions -- can fill every result slot with restatements of one answer.
``dedup_threshold`` is the opt-in cross-document counterpart.

Vectors are crafted rather than embedded so these run in milliseconds and
assert on exact geometry. Each vector is ``c * e0 + sqrt(1 - c^2) * u``,
which puts its cosine similarity to the query ``e0`` at exactly ``c`` while
``u`` (a unit vector orthogonal to ``e0``) controls similarity to siblings.
"""

from __future__ import annotations

import math

import pytest

from vstash.config import CacheConfig, VstashConfig
from vstash.errors import DedupThresholdOutOfRangeError
from vstash.profile import federated_search
from vstash.store import VstashStore
from vstash.validation import validate_dedup_threshold

from .conftest import requires_sqlite_vec

DIM = 384

# Neutralises the relative distance cutoff so it cannot silently drop the
# spread-out filler documents these tests depend on. The cutoff is
# best_distance-relative, and the corpus here deliberately spans distances
# 0.1 to 0.7 to keep sibling similarities controllable.
NO_CUTOFF = 100.0


def _basis(i: int) -> list[float]:
    vec = [0.0] * DIM
    vec[i] = 1.0
    return vec


def _at_similarity(c: float, u: list[float]) -> list[float]:
    """Unit vector whose cosine similarity to ``e0`` is exactly ``c``."""
    scale = math.sqrt(1.0 - c * c)
    return [c * a + scale * b for a, b in zip(_basis(0), u, strict=True)]


def _perturbed(base_idx: int, nudge_idx: int, eps: float = 1e-3) -> list[float]:
    """Unit vector ~``eps`` away from ``e[base_idx]``: a near-duplicate axis."""
    vec = [0.0] * DIM
    vec[base_idx] = 1.0
    vec[nudge_idx] = eps
    norm = math.hypot(*vec)
    return [v / norm for v in vec]


@requires_sqlite_vec
class TestDedupThreshold:
    """Flooding without the knob, distinct results with it."""

    @pytest.fixture
    def flooded_store(self, tmp_db_path: str) -> VstashStore:
        """One target, an 8-document near-duplicate cluster, 5 distinct fillers.

        By construction: target is closest to the query (cos 0.9), every
        cluster member sits at cos 0.5 and is >0.999 similar to its siblings,
        and the fillers sit at cos 0.3 on mutually orthogonal axes (pairwise
        similarity 0.09), so only the cluster should ever collapse.
        """
        store = VstashStore(tmp_db_path, embedding_dim=DIM)

        store.add_document(
            path="/target.md",
            title="Target",
            chunks=["the one distinctive answer"],
            embeddings=[_at_similarity(0.9, _basis(1))],
        )
        for i in range(8):
            store.add_document(
                path=f"/audit_{i}.md",
                title=f"Audit {i}",
                chunks=[f"restatement number {i} of the same answer"],
                embeddings=[_at_similarity(0.5, _perturbed(2, 20 + i))],
            )
        for i in range(5):
            store.add_document(
                path=f"/filler_{i}.md",
                title=f"Filler {i}",
                chunks=[f"unrelated note number {i}"],
                embeddings=[_at_similarity(0.3, _basis(3 + i))],
            )
        yield store
        store.close()

    @staticmethod
    def _search(store: VstashStore, **kwargs) -> list:
        return store.search(
            _basis(0),
            "distinctive answer",
            top_k=10,
            retrieval_mode="vec_only",
            distance_cutoff=NO_CUTOFF,
            **kwargs,
        )

    def test_cluster_floods_results_without_threshold(self, flooded_store: VstashStore):
        """Baseline: today's pipeline hands 8 of 10 slots to one answer."""
        paths = [r.path for r in self._search(flooded_store)]

        assert paths[0] == "/target.md"
        assert sum(1 for p in paths if p.startswith("/audit_")) == 8

    def test_threshold_collapses_cluster_to_one_representative(self, flooded_store: VstashStore):
        paths = [r.path for r in self._search(flooded_store, dedup_threshold=0.99)]

        assert sum(1 for p in paths if p.startswith("/audit_")) == 1

    def test_collapsed_slots_are_refilled_with_distinct_documents(self, flooded_store: VstashStore):
        """The collapse runs before the top_k cut, so freed slots get reused."""
        without = [r.path for r in self._search(flooded_store)]
        with_dedup = [r.path for r in self._search(flooded_store, dedup_threshold=0.99)]

        assert sum(1 for p in without if p.startswith("/filler_")) == 1
        assert sum(1 for p in with_dedup if p.startswith("/filler_")) == 5
        assert len(set(with_dedup)) == len(with_dedup)

    def test_highest_ranked_member_of_a_cluster_survives(self, flooded_store: VstashStore):
        """Keep-first: the collapse never promotes a worse-scoring duplicate."""
        without = [r.path for r in self._search(flooded_store)]
        with_dedup = [r.path for r in self._search(flooded_store, dedup_threshold=0.99)]

        first_audit = next(p for p in without if p.startswith("/audit_"))
        assert with_dedup[0] == "/target.md"
        assert first_audit in with_dedup

    def test_results_never_exceed_top_k(self, flooded_store: VstashStore):
        results = self._search(flooded_store, dedup_threshold=0.99)
        assert len(results) <= 10

    def test_scores_stay_descending(self, flooded_store: VstashStore):
        scores = [r.score for r in self._search(flooded_store, dedup_threshold=0.99)]
        assert scores == sorted(scores, reverse=True)

    def test_dissimilar_documents_survive_a_low_threshold(self, flooded_store: VstashStore):
        """The fillers sit at pairwise 0.09 -- a 0.5 threshold must spare them."""
        paths = [r.path for r in self._search(flooded_store, dedup_threshold=0.5)]
        assert sum(1 for p in paths if p.startswith("/filler_")) == 5

    def test_default_is_off(self, flooded_store: VstashStore):
        """Omitting the knob must reproduce the pre-feature ranking exactly."""
        explicit_none = [r.path for r in self._search(flooded_store, dedup_threshold=None)]
        omitted = [r.path for r in self._search(flooded_store)]
        assert explicit_none == omitted

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
    def test_out_of_range_threshold_rejected(self, flooded_store: VstashStore, bad: float):
        with pytest.raises(ValueError, match="dedup_threshold"):
            self._search(flooded_store, dedup_threshold=bad)

    @pytest.mark.parametrize("ok", [0.01, 0.5, 1.0])
    def test_in_range_threshold_accepted(self, flooded_store: VstashStore, ok: float):
        self._search(flooded_store, dedup_threshold=ok)


@requires_sqlite_vec
class TestSameDocumentBehaviour:
    """Chunks of ONE document: distinct passages survive, repetition collapses.

    The collapse deliberately does not exempt same-document pairs, so this is
    where it could damage the guarantee MMR exists to provide -- two distant
    chapters of a book must both stay reachable. Measured over four repo
    documents at the default 1024/128 chunking, adjacent chunks land at
    0.59-0.87 cosine and no pair anywhere reached 0.95, so a 0.95 threshold
    has real headroom over natural intra-document similarity.

    These assert on ``_collapse_near_duplicates`` directly. End to end, MMR's
    same-document penalty usually suppresses such a duplicate first, so a
    pipeline-level assertion would be measuring MMR rather than the collapse.
    """

    @pytest.fixture
    def manual_store(self, tmp_db_path: str) -> VstashStore:
        """One document: chunk 0 and 1 restate each other (~0.9999), chunks 2
        and 3 are genuinely different passages, and chunk 4 sits at 0.87 --
        the top of the band real chunking produces between adjacent chunks."""
        store = VstashStore(tmp_db_path, embedding_dim=DIM)
        near_dup_a = _perturbed(2, 30)
        near_dup_b = _perturbed(2, 31)
        adjacent_like = [
            0.87 * a + math.sqrt(1 - 0.87**2) * b for a, b in zip(_basis(4), _basis(6), strict=True)
        ]
        store.add_document(
            path="/manual.md",
            title="Manual",
            chunks=[
                "chapter one, the opening argument",
                "the summary repeats the opening argument",
                "chapter seven, an unrelated appendix",
                "chapter nine, a different appendix",
                "chapter ten, overlapping the previous page",
            ],
            embeddings=[
                near_dup_a,
                near_dup_b,
                _basis(4),
                _basis(5),
                adjacent_like,
            ],
        )
        yield store
        store.close()

    @staticmethod
    def _ranked(store: VstashStore) -> list[dict]:
        """Candidate list in seq order, shaped like the pipeline's."""
        ids = [row[0] for row in store._conn.execute("SELECT id FROM chunks ORDER BY seq")]
        return [
            {"id": cid, "path": "/manual.md", "rrf": 1.0 / (n + 1)} for n, cid in enumerate(ids)
        ]

    def _kept_seqs(self, store: VstashStore, threshold: float) -> list[int]:
        ranked = self._ranked(store)
        first_id = ranked[0]["id"]
        kept = store._collapse_near_duplicates(ranked, threshold)
        return [int(r["id"]) - first_id for r in kept]

    def test_repeated_passage_within_one_document_collapses(self, manual_store: VstashStore):
        """Chunks 0 and 1 restate each other -- exactly one survives."""
        seqs = self._kept_seqs(manual_store, 0.95)
        assert [s for s in seqs if s in (0, 1)] == [0]

    def test_distinct_passages_of_one_document_survive(self, manual_store: VstashStore):
        """The MMR guarantee: two distant chapters both stay reachable."""
        seqs = self._kept_seqs(manual_store, 0.95)
        assert 2 in seqs and 3 in seqs

    def test_natural_adjacent_similarity_is_left_alone(self, manual_store: VstashStore):
        """Chunk 4 sits at 0.87 to chunk 2 -- the top of the band measured on
        real documents. A 0.95 threshold must not touch it."""
        assert 4 in self._kept_seqs(manual_store, 0.95)

    def test_a_threshold_below_the_natural_band_does_eat_passages(self, manual_store: VstashStore):
        """Why the recommended floor is not lower: at 0.85 the same document
        loses a legitimate passage."""
        assert 4 not in self._kept_seqs(manual_store, 0.85)

    def test_keep_first_within_a_document(self, manual_store: VstashStore):
        """The survivor of an intra-document cluster is the higher-ranked one."""
        assert self._kept_seqs(manual_store, 0.95)[0] == 0

    def test_knob_removes_nothing_distinct_end_to_end(self, manual_store: VstashStore):
        """Whatever the pipeline returns without the knob, enabling it must not
        remove a passage that is not a near-duplicate of a kept one."""
        common = dict(
            top_k=10,
            retrieval_mode="vec_only",
            distance_cutoff=NO_CUTOFF,
        )
        without = {r.chunk for r in manual_store.search(_basis(0), "chapter", **common)}
        with_dedup = {
            r.chunk
            for r in manual_store.search(_basis(0), "chapter", **common, dedup_threshold=0.95)
        }
        assert with_dedup <= without
        assert without - with_dedup <= {1}


@requires_sqlite_vec
class TestExactDuplicateThreshold:
    """``dedup_threshold=1.0`` must actually collapse byte-identical vectors.

    Regression: the comparison ran without a float32 tolerance, and the dot
    product of a float32 unit vector with an identical copy lands on
    0.99999994 often enough that a bare ``>= 1.0`` missed roughly half the
    pairs the documented "collapses exact duplicates" behaviour promises.
    """

    @pytest.fixture
    def identical_pairs_store(self, tmp_db_path: str) -> VstashStore:
        import numpy as np

        store = VstashStore(tmp_db_path, embedding_dim=DIM)
        rng = np.random.default_rng(0)
        for i in range(20):
            vec = rng.normal(size=DIM).astype("float32")
            vec = (vec / np.linalg.norm(vec)).tolist()
            store.add_document(
                path=f"/copy_a_{i}.md", title=f"A{i}", chunks=[f"copy a {i}"], embeddings=[vec]
            )
            store.add_document(
                path=f"/copy_b_{i}.md", title=f"B{i}", chunks=[f"copy b {i}"], embeddings=[vec]
            )
        yield store
        store.close()

    def _search(self, store: VstashStore, **kwargs) -> list:
        return store.search(
            _basis(0),
            "copy",
            top_k=100,
            retrieval_mode="vec_only",
            distance_cutoff=NO_CUTOFF,
            **kwargs,
        )

    def test_threshold_one_collapses_every_identical_pair(self, identical_pairs_store: VstashStore):
        assert len(self._search(identical_pairs_store)) == 40
        assert len(self._search(identical_pairs_store, dedup_threshold=1.0)) == 20

    def test_threshold_one_keeps_merely_similar_documents(self, identical_pairs_store: VstashStore):
        """The tolerance is float32 noise, not a licence to collapse
        genuinely different vectors."""
        identical_pairs_store.add_document(
            path="/distinct.md",
            title="Distinct",
            chunks=["nothing like the others"],
            embeddings=[_at_similarity(0.9, _basis(1))],
        )
        paths = [r.path for r in self._search(identical_pairs_store, dedup_threshold=1.0)]
        assert "/distinct.md" in paths


class TestDedupThresholdValidation:
    """One boundary check shared by every adapter."""

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
    def test_out_of_range_rejected(self, bad: float):
        with pytest.raises(DedupThresholdOutOfRangeError, match="dedup_threshold"):
            validate_dedup_threshold(bad)

    @pytest.mark.parametrize("ok", [None, 0.01, 0.5, 1.0])
    def test_in_range_accepted(self, ok):
        validate_dedup_threshold(ok)

    def test_bool_rejected(self):
        """bool is an int subclass -- True must not silently become 1.0."""
        with pytest.raises(DedupThresholdOutOfRangeError):
            validate_dedup_threshold(True)

    def test_limit_error_stays_a_value_error(self):
        """Legacy ``except ValueError`` callers must keep working (#326)."""
        with pytest.raises(ValueError):
            validate_dedup_threshold(1.5)

    def test_federated_search_validates_before_the_per_profile_guard(self):
        """federated_search runs each profile inside ``except Exception``, so an
        unvalidated threshold would surface as an empty result set and a zero
        exit code instead of an error."""
        with pytest.raises(DedupThresholdOutOfRangeError):
            federated_search([0.0] * DIM, "query", cfg=VstashConfig(), top_k=5, dedup_threshold=1.5)


@requires_sqlite_vec
class TestDedupCacheKey:
    """The query cache must not serve one call's results to a different call."""

    @pytest.fixture
    def cached_store(self, tmp_db_path: str) -> VstashStore:
        store = VstashStore(
            tmp_db_path,
            embedding_dim=DIM,
            cache=CacheConfig(query_cache_size=8),
        )
        store.add_document(
            path="/keep.md",
            title="Keep",
            chunks=["kept content"],
            embeddings=[_at_similarity(0.9, _basis(1))],
            collection="keep",
        )
        for i in range(4):
            store.add_document(
                path=f"/drop_{i}.md",
                title=f"Drop {i}",
                chunks=[f"dropped content {i}"],
                embeddings=[_at_similarity(0.5, _perturbed(2, 20 + i))],
                collection="drop",
            )
        yield store
        store.close()

    def _search(self, store: VstashStore, **kwargs) -> list:
        return store.search(
            _basis(0),
            "content",
            top_k=5,
            retrieval_mode="vec_only",
            distance_cutoff=NO_CUTOFF,
            **kwargs,
        )

    def test_dedup_threshold_is_part_of_the_key(self, cached_store: VstashStore):
        plain = [r.path for r in self._search(cached_store)]
        deduped = [r.path for r in self._search(cached_store, dedup_threshold=0.99)]

        assert sum(1 for p in plain if p.startswith("/drop_")) == 4
        assert sum(1 for p in deduped if p.startswith("/drop_")) == 1

    def test_filters_are_part_of_the_key(self, cached_store: VstashStore):
        """A #106 filter tree used to be absent from the key entirely, so the
        unfiltered call's results were served back for the filtered one."""
        unfiltered = [r.path for r in self._search(cached_store)]
        filtered = [
            r.path for r in self._search(cached_store, filters={"not": {"collection": "drop"}})
        ]

        assert any(p.startswith("/drop_") for p in unfiltered)
        assert filtered == ["/keep.md"]

    def test_filter_key_is_order_insensitive_for_dict_siblings(self):
        """Sibling dict order must not fragment the cache."""
        common = {
            "query_embedding": [0.1] * DIM,
            "query_text": "q",
            "top_k": 5,
            "vec_weight": None,
            "fts_weight": None,
            "distance_cutoff": 1.3225,
            "collection": None,
            "project": None,
            "layer": None,
            "adaptive_rrf": True,
            "recency_boost": 0.0,
            "added_after": None,
            "added_before": None,
            "tags": None,
            "mmr_lambda": 0.5,
            "retrieval_mode": "hybrid",
            "cache_epoch": 0,
        }
        key_a = VstashStore._compute_search_cache_key(
            **common, filters={"collection": "a", "layer": "b"}
        )
        key_b = VstashStore._compute_search_cache_key(
            **common, filters={"layer": "b", "collection": "a"}
        )
        key_other = VstashStore._compute_search_cache_key(
            **common, filters={"collection": "z", "layer": "b"}
        )

        assert key_a == key_b
        assert key_a != key_other
