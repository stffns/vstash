"""Tests for vstash.retrain_batch.generate_triples_batched (T1.4b)."""

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

    def test_matches_generate_triples_shape(self, tmp_path: Path, torch_st_stubs: Any) -> None:
        """Output dicts must have the same keys as generate_triples so
        train_mnrl + retrain_multi do not care which path produced the
        pairs. Chunks here are >200 chars so the text[:200] prefix
        differs from the full chunk text, making query != positive."""
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
# RRF_K export surface                                                 #
# ------------------------------------------------------------------ #


def test_rrf_k_imported_from_store_module() -> None:
    """retrain_batch must use the same RRF_K as store.py so disagreement
    mining is byte-for-byte comparable to the legacy per-query path."""
    from vstash.retrain_batch import RRF_K as batch_rrf_k

    assert batch_rrf_k == RRF_K == 60
