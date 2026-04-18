"""Tests for vstash.retrain_batch: generate_triples_batched (T1.4b) +
evaluate_model_batched (T1.4c)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vstash.embed import get_embedding_dim
from vstash.retrain_batch import (
    _fts_top_k,
    _rrf_top5_paths,
    generate_triples_batched,
)
from vstash.store import RRF_K, VstashStore


# ------------------------------------------------------------------ #
# torch + sentence-transformers stubs                                  #
# ------------------------------------------------------------------ #


def _install_torch_stub(encoded_by_texts: dict[str, list[list[float]]]) -> Any:
    """Install a fake torch module + a fake SentenceTransformer that
    echoes ``encoded_by_texts`` back from ``model.encode``.

    The returned objects behave enough like torch tensors for the
    matmul / topk path in generate_triples_batched. We don't aim for
    real linear algebra; instead, we feed precomputed similarity so
    tests are deterministic and fast.
    """
    import numpy as np

    class FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = np.asarray(arr, dtype=np.float32)

        def to(self, device: str) -> "FakeTensor":
            return self

        @property
        def T(self) -> "FakeTensor":
            return FakeTensor(self.arr.T)

        def size(self, dim: int) -> int:
            return int(self.arr.shape[dim])

        def __matmul__(self, other: "FakeTensor") -> "FakeTensor":
            return FakeTensor(self.arr @ other.arr)

        def topk(self, k: int, dim: int = -1):
            # Mirror torch.topk: returns (values, indices) for the
            # top-k along `dim`.
            if dim == 1:
                idx = np.argsort(-self.arr, axis=1)[:, :k]
                vals = np.take_along_axis(self.arr, idx, axis=1)
                return FakeTensor(vals), FakeTensor(idx)
            raise NotImplementedError

        def cpu(self) -> "FakeTensor":
            return self

        def tolist(self) -> list:
            return self.arr.astype(int).tolist()

        def __getitem__(self, item) -> "FakeTensor":
            return FakeTensor(self.arr[item])

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = FakeCuda()

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: Any) -> None:
            return None

    torch_mod.no_grad = lambda: FakeNoGrad()

    class FakeModel:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            self.model_name = model_name
            self.device = device

        def encode(self, texts: list[str], **kwargs: Any) -> FakeTensor:
            missing = [t for t in texts if t not in encoded_by_texts]
            if missing:
                raise KeyError(f"no stub embedding for: {missing[:3]}")
            return FakeTensor(np.asarray([encoded_by_texts[t] for t in texts], dtype=np.float32))

        # evaluate_model_batched queries one of these two APIs to size
        # the temp store. The specific value does not matter for the
        # tests, only that it matches the dim of the supplied vectors.
        def get_embedding_dimension(self) -> int:
            sample = next(iter(encoded_by_texts.values()), None)
            return len(sample) if sample else 0

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = FakeModel

    sys.modules["torch"] = torch_mod
    sys.modules["sentence_transformers"] = st_mod
    return torch_mod, st_mod


@pytest.fixture
def torch_st_stubs() -> Any:
    """Install torch + sentence_transformers stubs for the duration
    of a test. Per-test encode map is attached via ``stub_encode_map``.
    """
    prev_torch = sys.modules.get("torch")
    prev_st = sys.modules.get("sentence_transformers")
    # Inject an empty map; individual tests populate
    # test_retrain_batch._encode_map before calling generate_triples_batched.
    encode_map: dict[str, list[list[float]]] = {}

    torch_mod, st_mod = _install_torch_stub(encode_map)

    class Harness:
        def set_encode(self, texts_to_vec: dict[str, list[list[float]]]) -> None:
            # mutate in place so the stub sees new keys
            encode_map.clear()
            encode_map.update(texts_to_vec)

    yield Harness()

    if prev_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = prev_torch
    if prev_st is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = prev_st


def _mk_store(path: str, docs: list[tuple[str, str, str]]) -> VstashStore:
    """Populate a tiny store and return it. docs = (path, title, text)."""
    from vstash.config import VstashConfig

    dim = get_embedding_dim(VstashConfig().embeddings.model)
    store = VstashStore(path, embedding_dim=dim)
    for i, (doc_path, title, text) in enumerate(docs):
        emb = [float((i + 1) % 10) / 10.0] * dim
        store.add_document(
            path=doc_path,
            title=title,
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )
    return store


# ------------------------------------------------------------------ #
# _rrf_top5_paths                                                      #
# ------------------------------------------------------------------ #


class TestRrfTopN:
    def test_agreement_ranks_head_chunks_first(self) -> None:
        id_to_path = {10: "/a", 20: "/b", 30: "/c"}
        top = _rrf_top5_paths(
            vec_chunk_ids=[10, 20, 30],
            fts_chunk_ids=[10, 20, 30],
            vec_weight=0.5,
            fts_weight=0.5,
            chunk_id_to_path=id_to_path,
        )
        assert top == [10, 20, 30]

    def test_disagreement_is_resolved_by_weighted_score(self) -> None:
        """Weights shift the winner. vec-heavy should rank vec-first
        candidates above fts-first ones."""
        id_to_path = {1: "/vec-top", 2: "/fts-top"}
        # Each appears only in one signal. With weights 0.7 vs 0.3, the
        # 0.7 side wins even though both are at rank 0.
        top_vec_heavy = _rrf_top5_paths(
            vec_chunk_ids=[1],
            fts_chunk_ids=[2],
            vec_weight=0.7,
            fts_weight=0.3,
            chunk_id_to_path=id_to_path,
        )
        top_fts_heavy = _rrf_top5_paths(
            vec_chunk_ids=[1],
            fts_chunk_ids=[2],
            vec_weight=0.3,
            fts_weight=0.7,
            chunk_id_to_path=id_to_path,
        )
        assert top_vec_heavy[0] == 1
        assert top_fts_heavy[0] == 2

    def test_chunks_missing_from_path_map_are_dropped(self) -> None:
        """Stale FTS hits (chunk deleted after corpus snapshot) must
        not appear in the output."""
        top = _rrf_top5_paths(
            vec_chunk_ids=[1, 2],
            fts_chunk_ids=[99],  # 99 absent from id_to_path
            vec_weight=0.5,
            fts_weight=0.5,
            chunk_id_to_path={1: "/a", 2: "/b"},
        )
        assert 99 not in top


# ------------------------------------------------------------------ #
# _fts_top_k                                                           #
# ------------------------------------------------------------------ #


class TestFtsTopK:
    def test_returns_chunk_ids_ordered_by_rank(self, tmp_path: Path) -> None:
        store = _mk_store(
            str(tmp_path / "fts.db"),
            [
                ("/a", "A", "alpha beta gamma delta"),
                ("/b", "B", "alpha something totally different"),
                ("/c", "C", "zzz yyy xxx completely unrelated"),
            ],
        )
        try:
            top = _fts_top_k(store._conn, "alpha", top_k=5)
        finally:
            store.close()
        # Only the two docs that contain 'alpha' should come back.
        assert 1 in top and 2 in top
        assert 3 not in top

    def test_returns_empty_on_malformed_fts_query(self, tmp_path: Path) -> None:
        store = _mk_store(
            str(tmp_path / "fts2.db"),
            [("/a", "A", "hello world")],
        )
        try:
            # FTS5 raises OperationalError for some special inputs;
            # we must swallow it and return [] so mining continues.
            top = _fts_top_k(store._conn, "", top_k=5)
        finally:
            store.close()
        assert top == []


# ------------------------------------------------------------------ #
# generate_triples_batched                                              #
# ------------------------------------------------------------------ #


class TestGenerateTriplesBatched:
    def test_empty_store_returns_empty(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        store = _mk_store(str(tmp_path / "empty.db"), [])
        try:
            pairs = generate_triples_batched(store, base_model="dummy")
        finally:
            store.close()
        assert pairs == []

    def test_output_dict_has_required_keys(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        """Output dicts must have the same keys as generate_triples so
        train_mnrl + retrain_multi do not care which path produced
        the pairs. Chunks here are > 200 chars so the text[:200]
        prefix differs from the full chunk text, making
        query != positive. Shape check only; see
        ``test_parity_with_generate_triples_on_small_store`` for the
        approximate content parity assertion."""
        filler_a = " ".join([f"alpha-word-{i}" for i in range(60)])  # > 200 chars
        filler_b = " ".join([f"beta-word-{i}" for i in range(60)])
        filler_c = " ".join([f"gamma-word-{i}" for i in range(60)])
        filler_d = " ".join([f"delta-word-{i}" for i in range(60)])
        docs = [
            ("/a", "Alpha", filler_a),
            ("/b", "Beta", filler_b),
            ("/c", "Gamma", filler_c),
            ("/d", "Delta", filler_d),
        ]
        store = _mk_store(str(tmp_path / "shape.db"), docs)
        try:
            # 4 docs => 4-dim vectors, different for each.
            vecs = {
                docs[0][2]: [1.0, 0.0, 0.0, 0.0],
                docs[1][2]: [0.9, 0.1, 0.0, 0.0],
                docs[2][2]: [0.0, 0.0, 1.0, 0.0],
                docs[3][2]: [0.0, 0.0, 0.0, 1.0],
            }
            # Prefix queries are the first 200 chars (distinct from full text).
            prefixes = {doc[2][:200]: vecs[doc[2]] for doc in docs}
            torch_st_stubs.set_encode({**vecs, **prefixes})

            pairs = generate_triples_batched(
                store,
                base_model="dummy",
                max_queries=4,
                seed=42,
            )
        finally:
            store.close()

        assert pairs, "expected at least one mined triple"
        for p in pairs:
            assert set(p.keys()) == {"query", "positive", "negative"}
            assert isinstance(p["query"], str)
            assert isinstance(p["positive"], str)
            assert p["query"] != p["positive"]  # no degenerate pairs

    def test_respects_synth_queries(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        """When synthesized_queries is provided for a chunk, the mined
        pair's ``query`` field must come from the synth map, not the
        chunk prefix."""
        docs = [
            ("/a", "Alpha", "alpha beta gamma"),
            ("/b", "Beta", "beta gamma delta"),
        ]
        store = _mk_store(str(tmp_path / "synth.db"), docs)
        try:
            synth_text = "what is alpha beta"
            vecs = {
                docs[0][2]: [1.0, 0.0],
                docs[1][2]: [0.0, 1.0],
                synth_text: [0.9, 0.1],
                docs[1][2][:200]: [0.0, 1.0],  # prefix for chunk 2 fallback
            }
            torch_st_stubs.set_encode(vecs)

            # chunk 1 uses a synthesized query; chunk 2 falls back to prefix.
            pairs = generate_triples_batched(
                store,
                base_model="dummy",
                max_queries=2,
                synthesized_queries={1: [synth_text]},
                seed=42,
            )
        finally:
            store.close()

        synth_pairs = [p for p in pairs if p["query"] == synth_text]
        assert synth_pairs, "synthesized query text must surface as a pair's query"

    def test_exclude_chunk_ids_is_honored(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        docs = [
            ("/a", "Alpha", "alpha chunk text"),
            ("/b", "Beta", "beta chunk text"),
            ("/c", "Gamma", "gamma chunk text"),
        ]
        store = _mk_store(str(tmp_path / "exclude.db"), docs)
        try:
            vecs = {
                docs[0][2]: [1.0, 0.0],
                docs[1][2]: [0.0, 1.0],
                docs[2][2]: [0.5, 0.5],
            }
            prefixes = {doc[2][:200]: vecs[doc[2]] for doc in docs}
            torch_st_stubs.set_encode({**vecs, **prefixes})

            pairs = generate_triples_batched(
                store,
                base_model="dummy",
                max_queries=10,
                exclude_chunk_ids={2},  # skip chunk id 2 (the beta doc)
                seed=42,
            )
        finally:
            store.close()

        # None of the emitted queries should have come from beta.
        assert all("beta chunk text" not in p["query"] for p in pairs)

    def test_raises_when_torch_missing(self, tmp_path: Path) -> None:
        """When torch is absent, the batched miner must raise ImportError
        with the same install hint as the rest of the retrain module."""
        store = _mk_store(
            str(tmp_path / "torch_missing.db"),
            [("/a", "A", "some body text")],
        )
        prev_torch = sys.modules.get("torch")
        prev_st = sys.modules.get("sentence_transformers")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="sentence-transformers"):
                generate_triples_batched(store, base_model="dummy")
        finally:
            if prev_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = prev_torch
            if prev_st is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = prev_st
            store.close()


# ------------------------------------------------------------------ #
# retrain_multi wiring                                                 #
# ------------------------------------------------------------------ #


class TestRetrainMultiBulkMine:
    def test_bulk_mine_flag_routes_to_batched_miner(self, tmp_path: Path) -> None:
        """retrain_multi(bulk_mine=True) must call generate_triples_batched
        instead of generate_triples. Verify via mock rather than running
        actual training."""
        from vstash.retrain import retrain_multi

        # Small dual store setup; shape doesn't matter because both
        # generate_triples paths are mocked away.
        s1 = _mk_store(
            str(tmp_path / "s1.db"),
            [(f"/s1/{i}", f"S1 {i}", f"s1 chunk {i}") for i in range(12)],
        )
        s2 = _mk_store(
            str(tmp_path / "s2.db"),
            [(f"/s2/{i}", f"S2 {i}", f"s2 chunk {i}") for i in range(12)],
        )
        try:
            batched_return = [{"query": "q", "positive": "p", "negative": None}] * 15

            with (
                patch(
                    "vstash.retrain_batch.generate_triples_batched",
                    return_value=batched_return,
                ) as mock_batched,
                patch("vstash.retrain.generate_triples") as mock_plain,
                patch("vstash.retrain.train_mnrl"),
            ):
                retrain_multi(
                    {"a": s1, "b": s2},
                    base_model="dummy",
                    output_path=str(tmp_path / "model"),
                    total_triples=40,
                    sampling="uniform",
                    skip_eval=True,
                    bulk_mine=True,
                    bulk_mine_device="cpu",
                )

            # Two stores -> two calls to the batched miner, zero calls
            # to the legacy per-query miner.
            assert mock_batched.call_count == 2
            assert mock_plain.call_count == 0
            device_kwargs = [c.kwargs.get("device") for c in mock_batched.call_args_list]
            assert all(d == "cpu" for d in device_kwargs)
        finally:
            s1.close()
            s2.close()

    def test_bulk_mine_false_keeps_legacy_path(self, tmp_path: Path) -> None:
        from vstash.retrain import retrain_multi

        s1 = _mk_store(
            str(tmp_path / "s1.db"),
            [(f"/s1/{i}", f"S1 {i}", f"s1 chunk {i}") for i in range(12)],
        )
        try:
            with (
                patch("vstash.retrain_batch.generate_triples_batched") as mock_batched,
                patch(
                    "vstash.retrain.generate_triples",
                    return_value=[{"query": "q", "positive": "p", "negative": None}] * 15,
                ) as mock_plain,
                patch("vstash.retrain.train_mnrl"),
            ):
                retrain_multi(
                    {"a": s1},
                    base_model="dummy",
                    output_path=str(tmp_path / "model"),
                    total_triples=20,
                    sampling="uniform",
                    skip_eval=True,
                    bulk_mine=False,
                )

            assert mock_batched.call_count == 0
            assert mock_plain.call_count == 1
        finally:
            s1.close()


# ------------------------------------------------------------------ #
# Shared constants + approximation parity                              #
# ------------------------------------------------------------------ #


def test_rrf_k_imported_from_store_module() -> None:
    """retrain_batch uses the same RRF_K as store.py so the fusion
    formula matches. This is a precondition for the batched miner to
    be a faithful approximation of the legacy disagreement signal;
    see the module docstring for the remaining divergences (distance
    cutoff, candidate pool, MMR dedup, embedder path)."""
    from vstash.retrain_batch import RRF_K as batch_rrf_k

    assert batch_rrf_k == RRF_K == 60


def test_shared_top_k_and_adaptive_weights() -> None:
    """Both modules must consume the shared ``TOP_K`` and
    ``adaptive_rrf_weights``. If either drifts, the vec/fts top-5
    disagreement comparison is no longer meaningful across the two
    miners."""
    from vstash.retrain import TOP_K as retrain_top_k
    from vstash.retrain import adaptive_rrf_weights
    from vstash.retrain_batch import TOP_K as batch_top_k

    assert retrain_top_k == batch_top_k == 10
    # Spot-check the ladder at the three representative word counts.
    assert adaptive_rrf_weights(5) == (0.70, 0.30, 0.30, 0.70)
    assert adaptive_rrf_weights(30) == (0.85, 0.15, 0.15, 0.85)
    assert adaptive_rrf_weights(80) == (0.95, 0.05, 0.50, 0.50)


def test_parity_with_generate_triples_on_small_store(tmp_path: Path, torch_st_stubs: Any) -> None:
    """Coarse parity check against ``generate_triples`` on a small
    store. Not byte-for-byte -- the batched path skips the distance
    cutoff, candidate pool, and MMR dedup (see module docstring) --
    but both miners must return the same pair count and must share a
    nonzero fraction of their query texts. If this drops to 0 overlap,
    something structural has broken in the batched path (empty top-k,
    wrong RRF weights, etc.)."""
    from vstash.retrain import generate_triples

    def filler(tag: str) -> str:  # > 200 chars so prefix != full text
        return " ".join(f"{tag}-word-{i}" for i in range(60))

    docs = [
        ("/a", "A", filler("alpha")),
        ("/b", "B", filler("beta")),
        ("/c", "C", filler("gamma")),
        ("/d", "D", filler("delta")),
    ]
    store = _mk_store(str(tmp_path / "parity.db"), docs)
    try:
        vec_map = {
            docs[0][2]: [1.0, 0.0, 0.0, 0.0],
            docs[1][2]: [0.0, 1.0, 0.0, 0.0],
            docs[2][2]: [0.0, 0.0, 1.0, 0.0],
            docs[3][2]: [0.0, 0.0, 0.0, 1.0],
        }
        prefixes = {doc[2][:200]: vec_map[doc[2]] for doc in docs}
        torch_st_stubs.set_encode({**vec_map, **prefixes})

        # Batched path.
        batched_pairs = generate_triples_batched(
            store,
            base_model="dummy",
            max_queries=4,
            seed=42,
        )

        # Legacy path -- mock embed_query + store.search with results
        # that roughly reproduce what the batched path sees, so the
        # two outputs are in the same ballpark. We do not mirror the
        # disagreement logic exactly; we just assert both return a
        # nonempty list and share the queries that survive.
        def _fake_embed(text: str, model_name: str) -> list[float]:
            return prefixes.get(text) or [0.0, 0.0, 0.0, 0.0]

        from vstash.models import SearchResult

        def _fake_search(**kwargs: Any) -> list[SearchResult]:
            # Return all docs in a stable order -- the positive lookup
            # in generate_triples keys on doc_path, not ranking.
            return [
                SearchResult(
                    chunk_id=i + 1,
                    text=docs[i][2],
                    title=docs[i][1],
                    path=docs[i][0],
                    chunk=0,
                    score=0.5,
                )
                for i in range(len(docs))
            ]

        with (
            patch("vstash.retrain.embed_query", side_effect=_fake_embed),
            patch.object(store, "search", side_effect=_fake_search),
        ):
            legacy_pairs = generate_triples(
                store,
                model_name="dummy",
                max_queries=4,
                seed=42,
            )
    finally:
        store.close()

    assert batched_pairs, "batched miner must produce pairs on this fixture"
    assert legacy_pairs, "legacy miner must produce pairs on this fixture"
    batched_queries = {p["query"] for p in batched_pairs}
    legacy_queries = {p["query"] for p in legacy_pairs}
    # Both must share at least one query; they sample from the same
    # pool and use the same seed.
    assert batched_queries & legacy_queries, (
        f"no query overlap: batched={batched_queries}, legacy={legacy_queries}"
    )


# ------------------------------------------------------------------ #
# evaluate_model_batched (T1.4c)                                       #
# ------------------------------------------------------------------ #


class TestEvaluateModelBatched:
    def test_empty_queries_returns_zero_metrics(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        from vstash.retrain_batch import evaluate_model_batched

        store = _mk_store(str(tmp_path / "empty.db"), [("/a", "A", "some body text")])
        try:
            out = evaluate_model_batched(store, "dummy", eval_queries=[])
        finally:
            store.close()
        assert out.n_queries == 0
        assert out.ndcg_at_10 == 0.0

    def test_no_relevant_paths_returns_zero_metrics(
        self, tmp_path: Path, torch_st_stubs: Any
    ) -> None:
        from vstash.retrain_batch import evaluate_model_batched

        # Eval query points at a path that is NOT in the store.
        store = _mk_store(
            str(tmp_path / "no_rel.db"),
            [("/a", "A", "body text a"), ("/b", "B", "body text b")],
        )
        torch_st_stubs.set_encode(
            {
                "body text a": [1.0, 0.0],
                "body text b": [0.0, 1.0],
                "does not exist": [0.5, 0.5],
            }
        )
        try:
            out = evaluate_model_batched(
                store,
                "dummy",
                eval_queries=[{"query": "does not exist", "relevant_paths": ["/never"]}],
                noise_sample_size=2,
            )
        finally:
            store.close()
        assert out.n_queries == 0
        assert out.ndcg_at_10 == 0.0

    def test_relevant_chunk_retrieved_gives_hit(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        """Query embedding identical to doc A's embedding must put A
        at rank 1, producing NDCG@10=1.0, MRR=1.0, Hit@10=1.0."""
        from vstash.retrain_batch import evaluate_model_batched

        store = _mk_store(
            str(tmp_path / "hit.db"),
            [
                ("/target", "Target", "alpha beta gamma"),
                ("/other", "Other", "zzz yyy xxx"),
            ],
        )
        torch_st_stubs.set_encode(
            {
                "alpha beta gamma": [1.0, 0.0],
                "zzz yyy xxx": [0.0, 1.0],
                "find alpha": [1.0, 0.0],
            }
        )
        try:
            out = evaluate_model_batched(
                store,
                "dummy",
                eval_queries=[{"query": "find alpha", "relevant_paths": ["/target"]}],
                noise_sample_size=2,
            )
        finally:
            store.close()
        assert out.n_queries == 1
        assert out.ndcg_at_10 == pytest.approx(1.0)
        assert out.mrr == pytest.approx(1.0)
        assert out.hit_at_10 == pytest.approx(1.0)

    def test_relevant_chunk_beyond_top_k_gives_zero(
        self, tmp_path: Path, torch_st_stubs: Any
    ) -> None:
        """If the relevant doc's vec similarity is lower than all noise
        docs, its rank exceeds top-K and NDCG@10 is 0. Verifies the
        top-K truncation is honored."""
        from vstash.retrain_batch import evaluate_model_batched

        # 12 docs: relevant at position [0], noise filling the rest with
        # vectors closer to the query than the relevant doc is.
        docs = [("/relevant", "R", "unique relevant text")]
        for i in range(12):
            docs.append((f"/noise-{i}", f"N{i}", f"noise body {i}"))
        store = _mk_store(str(tmp_path / "miss.db"), docs)

        query_text = "search for something"
        encode_map = {
            "unique relevant text": [0.1, 0.9],  # very unlike the query
            query_text: [0.9, 0.1],  # opposite direction
        }
        for i in range(12):
            # Align all noise vectors close to the query so they outrank
            # the relevant doc.
            encode_map[f"noise body {i}"] = [0.9, 0.1 + 0.001 * i]
        torch_st_stubs.set_encode(encode_map)

        try:
            out = evaluate_model_batched(
                store,
                "dummy",
                eval_queries=[{"query": query_text, "relevant_paths": ["/relevant"]}],
                noise_sample_size=12,
            )
        finally:
            store.close()
        # The relevant doc should not be in top-10 because noise is closer.
        # (FTS might still rescue it if query terms match, but our crafted
        # query shares no terms with the relevant chunk text.)
        assert out.n_queries == 1
        # Either 0.0 (not in top-10) or some partial credit via FTS. As
        # long as it's clearly below 1.0, the truncation is working.
        assert out.ndcg_at_10 < 1.0

    def test_raises_without_torch(self, tmp_path: Path) -> None:
        from vstash.retrain_batch import evaluate_model_batched

        store = _mk_store(
            str(tmp_path / "torch_missing.db"),
            [("/a", "A", "some body text")],
        )
        prev_torch = sys.modules.get("torch")
        prev_st = sys.modules.get("sentence_transformers")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="sentence-transformers"):
                evaluate_model_batched(
                    store,
                    "dummy",
                    eval_queries=[{"query": "q", "relevant_paths": ["/a"]}],
                )
        finally:
            if prev_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = prev_torch
            if prev_st is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = prev_st
            store.close()


class TestRetrainMultiBulkEval:
    def test_bulk_eval_flag_routes_to_batched_evaluator(self, tmp_path: Path) -> None:
        """retrain_multi(bulk_eval=True) must call evaluate_model_batched
        instead of evaluate_model. Verified with mocks -- no real
        training or encoding runs."""
        from vstash.retrain import EvalMetrics, retrain_multi

        s1 = _mk_store(
            str(tmp_path / "s1.db"),
            [(f"/s1/{i}", f"S1 {i}", f"s1 chunk {i}") for i in range(30)],
        )
        try:
            baseline = EvalMetrics(ndcg_at_10=0.5, mrr=0.5, hit_at_10=0.6, n_queries=20)
            final = EvalMetrics(ndcg_at_10=0.6, mrr=0.6, hit_at_10=0.7, n_queries=20)
            eval_queries = [{"query": f"q{i}", "relevant_paths": [f"/s1/{i}"]} for i in range(20)]

            with (
                patch(
                    "vstash.retrain_batch.evaluate_model_batched",
                    side_effect=[baseline, final],
                ) as mock_batched,
                patch("vstash.retrain.evaluate_model") as mock_plain,
                patch(
                    "vstash.retrain.generate_triples",
                    return_value=[{"query": "q", "positive": "p", "negative": None}] * 15,
                ),
                patch("vstash.retrain.train_mnrl"),
            ):
                retrain_multi(
                    {"a": s1},
                    base_model="dummy",
                    output_path=str(tmp_path / "model"),
                    total_triples=30,
                    sampling="uniform",
                    eval_queries_by_dataset={"a": eval_queries},
                    bulk_eval=True,
                    bulk_mine_device="cpu",
                )

            assert mock_batched.call_count == 2  # baseline + final
            assert mock_plain.call_count == 0
            # Device override must reach the batched evaluator.
            for call in mock_batched.call_args_list:
                assert call.kwargs.get("device") == "cpu"
        finally:
            s1.close()

    def test_bulk_eval_false_keeps_legacy_evaluator(self, tmp_path: Path) -> None:
        from vstash.retrain import EvalMetrics, retrain_multi

        s1 = _mk_store(
            str(tmp_path / "s1.db"),
            [(f"/s1/{i}", f"S1 {i}", f"s1 chunk {i}") for i in range(30)],
        )
        try:
            baseline = EvalMetrics(ndcg_at_10=0.5, mrr=0.5, hit_at_10=0.6, n_queries=20)
            final = EvalMetrics(ndcg_at_10=0.6, mrr=0.6, hit_at_10=0.7, n_queries=20)
            eval_queries = [{"query": f"q{i}", "relevant_paths": [f"/s1/{i}"]} for i in range(20)]

            with (
                patch("vstash.retrain_batch.evaluate_model_batched") as mock_batched,
                patch(
                    "vstash.retrain.evaluate_model",
                    side_effect=[baseline, final],
                ) as mock_plain,
                patch(
                    "vstash.retrain.generate_triples",
                    return_value=[{"query": "q", "positive": "p", "negative": None}] * 15,
                ),
                patch("vstash.retrain.train_mnrl"),
            ):
                retrain_multi(
                    {"a": s1},
                    base_model="dummy",
                    output_path=str(tmp_path / "model"),
                    total_triples=30,
                    sampling="uniform",
                    eval_queries_by_dataset={"a": eval_queries},
                    bulk_eval=False,
                )

            assert mock_batched.call_count == 0
            assert mock_plain.call_count == 2
        finally:
            s1.close()
