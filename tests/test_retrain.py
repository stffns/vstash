"""Tests for vstash.retrain -- self-supervised embedding fine-tuning."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vstash.models import SearchResult
from vstash.retrain import (
    EvalMetrics,
    _ndcg_at_k,
    _ndcg_from_ranks,
    evaluate_model,
    generate_triples,
    qrels_to_eval_queries,
    retrain,
    split_corpus_for_eval,
    train_mnrl,
)
from vstash.store import VstashStore


def _install_st_torch_stubs() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Install fake ``sentence_transformers`` + ``torch.utils.data`` modules in
    ``sys.modules`` so ``train_mnrl`` can import them even when they are
    not really installed in the test environment.

    The stubs expose the exact attributes ``train_mnrl`` touches:
    ``SentenceTransformer``, ``InputExample``, ``losses.MultipleNegativesRankingLoss``
    and ``DataLoader``. Tests can then assert against these mocks.
    Returns the installed stub modules so tests can restore inner mocks.
    """
    st_mod = types.ModuleType("sentence_transformers")
    st_losses = types.ModuleType("sentence_transformers.losses")
    st_mod.SentenceTransformer = MagicMock()
    st_mod.InputExample = MagicMock(side_effect=lambda texts: MagicMock(texts=texts))
    st_losses.MultipleNegativesRankingLoss = MagicMock()
    st_mod.losses = st_losses

    torch_mod = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")
    torch_data.DataLoader = MagicMock(return_value=[MagicMock()] * 5)
    torch_utils.data = torch_data
    torch_mod.utils = torch_utils

    sys.modules["sentence_transformers"] = st_mod
    sys.modules["sentence_transformers.losses"] = st_losses
    sys.modules["torch"] = torch_mod
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_data
    return st_mod, st_losses, torch_data


@pytest.fixture
def st_stubs() -> Any:
    """Install sentence_transformers + torch stubs for the duration of a
    test, and remove them afterwards so other tests see a clean state.
    """
    installed = [
        "sentence_transformers",
        "sentence_transformers.losses",
        "torch",
        "torch.utils",
        "torch.utils.data",
    ]
    previous = {k: sys.modules.get(k) for k in installed}
    stubs = _install_st_torch_stubs()
    yield stubs
    for k in installed:
        if previous[k] is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = previous[k]


# ------------------------------------------------------------------ #
# generate_triples                                                     #
# ------------------------------------------------------------------ #


class TestGenerateTriples:
    """Unit tests for the disagreement-triple sampler."""

    def test_empty_store_returns_empty_list(self, sample_store: VstashStore) -> None:
        pairs = generate_triples(sample_store, model_name="BAAI/bge-small-en-v1.5")
        assert pairs == []

    def test_respects_max_queries(self, populated_store: VstashStore) -> None:
        """max_queries caps the number of sampled chunks.

        The populated_store fixture creates 5 chunks total. Asking for
        2 pseudo-queries must not touch more than 2 rows, regardless of
        how many pairs end up being kept.
        """
        with patch("vstash.retrain.embed_query") as mock_embed:
            mock_embed.return_value = [0.0] * populated_store.embedding_dim
            with patch.object(populated_store, "search") as mock_search:
                mock_search.return_value = []
                pairs = generate_triples(populated_store, model_name="m", max_queries=2)
        assert mock_embed.call_count <= 2
        assert pairs == []

    def test_builds_pair_when_signals_agree_without_hard_neg(
        self, populated_store: VstashStore
    ) -> None:
        """When vec and FTS both put the source chunk at the top, no hard
        negative exists but the pair is still emitted (MNRL can fall back
        to in-batch negatives).
        """
        stub_result = SearchResult(
            chunk_id=1,
            text="Python is a high-level programming language known for its simplicity.",
            title="Python Guide",
            path="/test/python_guide.md",
            chunk=0,
            score=0.5,
        )
        with (
            patch("vstash.retrain.embed_query") as mock_embed,
            patch.object(populated_store, "search") as mock_search,
        ):
            mock_embed.return_value = [0.0] * populated_store.embedding_dim
            mock_search.return_value = [stub_result]
            pairs = generate_triples(populated_store, model_name="m", max_queries=5)

        assert pairs, "should emit at least one pair"
        for p in pairs:
            assert set(p.keys()) == {"query", "positive", "negative"}
            # agreement on top-5 means no hard negative is discoverable
            assert p["negative"] is None

    def test_builds_hard_negative_when_signals_disagree(self, populated_store: VstashStore) -> None:
        """When the vector and FTS top-5 paths differ, the first unique
        chunk in vec_results becomes the hard negative.
        """
        # The positive must match the chunk's own path so the function
        # treats it as the positive and skips the "positive == query"
        # early-return. We craft the vec branch to include the own
        # document path plus an extra path that FTS does not see.
        own_path = "/test/python_guide.md"
        vec_results = [
            SearchResult(
                chunk_id=1,
                text="positive text for the query",
                title="Python Guide",
                path=own_path,
                chunk=0,
                score=0.9,
            ),
            SearchResult(
                chunk_id=99,
                text="hard negative text",
                title="Other",
                path="/test/other.md",
                chunk=0,
                score=0.4,
            ),
        ]
        fts_results = [
            SearchResult(
                chunk_id=1,
                text="positive text for the query",
                title="Python Guide",
                path=own_path,
                chunk=0,
                score=0.3,
            ),
        ]
        with (
            patch("vstash.retrain.embed_query") as mock_embed,
            patch.object(populated_store, "search") as mock_search,
        ):
            mock_embed.return_value = [0.0] * populated_store.embedding_dim

            def search_side_effect(**kwargs: Any) -> list[SearchResult]:
                # vec-heavy call has vec_weight > fts_weight
                return vec_results if kwargs["vec_weight"] > kwargs["fts_weight"] else fts_results

            mock_search.side_effect = search_side_effect
            pairs = generate_triples(populated_store, model_name="m", max_queries=5)

        assert pairs, "should emit pairs when signals disagree"
        emitted_with_neg = [p for p in pairs if p.get("negative") == "hard negative text"]
        assert emitted_with_neg, "expected at least one pair with the crafted hard negative"

    def test_skips_when_positive_equals_query(self, populated_store: VstashStore) -> None:
        """If the only retrieval result matching the source chunk path
        is textually identical to the pseudo-query (sliced from the same
        chunk), the function should drop the pair rather than emit a
        degenerate (q, q) pair.
        """
        # Craft a result whose text equals the first 200 chars of the
        # chunk text in the fixture.
        row = populated_store._conn.execute(
            "SELECT c.text, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id LIMIT 1"
        ).fetchone()
        own_path = row["path"]
        own_text = row["text"]
        query_text = own_text[:200]

        identical = SearchResult(
            chunk_id=1, text=query_text, title="t", path=own_path, chunk=0, score=0.9
        )
        with (
            patch("vstash.retrain.embed_query") as mock_embed,
            patch.object(populated_store, "search") as mock_search,
        ):
            mock_embed.return_value = [0.0] * populated_store.embedding_dim
            mock_search.return_value = [identical]
            pairs = generate_triples(populated_store, model_name="m", max_queries=1)
        # No valid pair can be emitted because positive == query.
        assert all(p["query"] != p["positive"] for p in pairs)

    def test_search_exception_skips_iteration_but_continues(
        self, populated_store: VstashStore
    ) -> None:
        """A failing ``search()`` call on one pseudo-query must not abort
        the whole run -- the function should skip that iteration and
        carry on with the rest.
        """
        calls = {"n": 0}

        def flaky_search(**kwargs: Any) -> list[SearchResult]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return []

        with (
            patch("vstash.retrain.embed_query") as mock_embed,
            patch.object(populated_store, "search") as mock_search,
        ):
            mock_embed.return_value = [0.0] * populated_store.embedding_dim
            mock_search.side_effect = flaky_search
            # Should not raise.
            pairs = generate_triples(populated_store, model_name="m", max_queries=3)
        assert pairs == []
        assert calls["n"] > 1  # continued after the first failure


# ------------------------------------------------------------------ #
# train_mnrl                                                           #
# ------------------------------------------------------------------ #


class TestTrainMNRL:
    """Unit tests for the MNRL fine-tuning wrapper."""

    def test_raises_importerror_without_sentence_transformers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When sentence-transformers is not installed the helper must
        raise ImportError with an actionable install hint.

        The real environment has sentence-transformers installed, so we
        simulate absence by blocking the import.
        """
        import builtins

        real_import = builtins.__import__

        def block_st(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sentence_transformers":
                raise ImportError("simulated missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_st)
        with pytest.raises(ImportError, match="sentence-transformers"):
            train_mnrl(pairs=[{"query": "q", "positive": "p"}], output_path=str(tmp_path / "m"))

    def test_writes_training_metadata(self, tmp_path: Path, st_stubs: Any) -> None:
        """On successful training the helper persists a training_meta.json
        with the hyperparameters the caller used.
        """
        st_mod, _, _ = st_stubs
        mock_model = MagicMock()
        st_mod.SentenceTransformer.return_value = mock_model

        pairs = [{"query": "q", "positive": "p", "negative": "n"}] * 10
        out = tmp_path / "model"

        saved_to = train_mnrl(
            pairs=pairs,
            base_model="dummy",
            output_path=str(out),
            epochs=3,
            lr=1e-5,
            batch_size=16,
        )

        assert saved_to == str(out)
        meta = json.loads((out / "training_meta.json").read_text())
        assert meta["base_model"] == "dummy"
        assert meta["n_pairs"] == 10
        assert meta["epochs"] == 3
        assert meta["batch_size"] == 16
        assert meta["lr"] == 1e-5
        assert isinstance(meta["training_time_s"], (int, float))
        mock_model.fit.assert_called_once()
        mock_model.save.assert_called_once_with(str(out))

    def test_builds_triplet_examples_when_negative_present(
        self, tmp_path: Path, st_stubs: Any
    ) -> None:
        """Pairs that include a 'negative' key must be wrapped as
        three-element InputExamples so MNRL uses them as explicit hard
        negatives.
        """
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        pairs = [
            {"query": "q1", "positive": "p1", "negative": "n1"},
            {"query": "q2", "positive": "p2"},  # no explicit negative
        ]
        train_mnrl(pairs=pairs, base_model="d", output_path=str(tmp_path / "m"))

        def _texts_len(call: Any) -> int:
            # retrain.py uses InputExample(texts=[...]) so the list is
            # in call.kwargs, with a positional-arg fallback just in case.
            texts = call.kwargs.get("texts")
            if texts is None and call.args:
                texts = call.args[0]
            return len(texts)

        shapes = sorted(_texts_len(c) for c in st_mod.InputExample.call_args_list)
        assert shapes == [2, 3], (
            f"expected one 2-text and one 3-text InputExample, got shapes {shapes}"
        )

    def test_creates_output_directory(self, tmp_path: Path, st_stubs: Any) -> None:
        """train_mnrl must create the output directory if it does not exist."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        out = tmp_path / "nested" / "dir" / "model"
        train_mnrl(
            pairs=[{"query": "q", "positive": "p"}],
            base_model="d",
            output_path=str(out),
        )
        assert out.is_dir()
        assert (out / "training_meta.json").is_file()

    def test_expands_tilde_in_output_path(self, tmp_path: Path, st_stubs: Any) -> None:
        """``output_path`` with a leading ``~`` expands to the user's home."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        # Redirect ~ to tmp_path for the test.
        with patch.object(Path, "expanduser", lambda self: tmp_path / str(self).lstrip("~/")):
            saved_to = train_mnrl(
                pairs=[{"query": "q", "positive": "p"}],
                base_model="d",
                output_path="~/my_model",
            )
        assert Path(saved_to).is_dir()
        assert (Path(saved_to) / "training_meta.json").is_file()


# ------------------------------------------------------------------ #
# NDCG math                                                            #
# ------------------------------------------------------------------ #


class TestNdcgAtK:
    """Direct unit tests for the NDCG@k helper."""

    def test_rank_1_is_perfect(self) -> None:
        assert _ndcg_at_k(1) == pytest.approx(1.0)

    def test_rank_above_k_is_zero(self) -> None:
        assert _ndcg_at_k(11, k=10) == 0.0

    def test_none_rank_is_zero(self) -> None:
        assert _ndcg_at_k(None) == 0.0

    def test_rank_2_matches_formula(self) -> None:
        # 1 / log2(3) ~= 0.6309
        assert _ndcg_at_k(2) == pytest.approx(1.0 / 1.584962500721156, rel=1e-9)


class TestNdcgFromRanks:
    """Multi-relevant NDCG@k handles BEIR-style qrels correctly."""

    def test_single_relevant_at_rank_1_scores_one(self) -> None:
        assert _ndcg_from_ranks([1], num_relevant=1) == pytest.approx(1.0)

    def test_zero_num_relevant_is_zero(self) -> None:
        assert _ndcg_from_ranks([], num_relevant=0) == 0.0

    def test_no_hits_is_zero(self) -> None:
        assert _ndcg_from_ranks([], num_relevant=3) == 0.0

    def test_all_relevant_at_top_is_perfect(self) -> None:
        # 3 relevant docs at ranks 1,2,3 -> NDCG = 1.0 exactly.
        assert _ndcg_from_ranks([1, 2, 3], num_relevant=3) == pytest.approx(1.0)

    def test_partial_hits_below_one(self) -> None:
        # 1 of 3 relevant found at rank 1: ideal would place 3 at ranks 1,2,3.
        # DCG = 1/log2(2) = 1; IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4).
        dcg = 1.0
        idcg = 1.0 + 1.0 / 1.584962500721156 + 1.0 / 2.0
        score = _ndcg_from_ranks([1], num_relevant=3)
        assert score == pytest.approx(dcg / idcg, rel=1e-9)
        assert 0.0 < score < 1.0

    def test_caps_idcg_at_k(self) -> None:
        # 20 relevant total but k=10 -> IDCG counts only the first 10 slots.
        perfect_at_k = _ndcg_from_ranks(list(range(1, 11)), num_relevant=20, k=10)
        assert perfect_at_k == pytest.approx(1.0)


class TestQrelsToEvalQueries:
    """BEIR-style qrels must convert cleanly to the eval_queries format."""

    def test_basic_mapping(self) -> None:
        queries = {"q1": "What is RRF?", "q2": "Multilingual retrieval"}
        qrels = {
            "q1": {"doc_a": 1, "doc_b": 1, "doc_c": 0},
            "q2": {"doc_d": 1},
        }
        out = qrels_to_eval_queries(queries, qrels)
        assert len(out) == 2
        q1 = next(q for q in out if q["query_id"] == "q1")
        assert set(q1["relevant_paths"]) == {"doc_a", "doc_b"}
        assert q1["query"] == "What is RRF?"
        q2 = next(q for q in out if q["query_id"] == "q2")
        assert q2["relevant_paths"] == ["doc_d"]

    def test_custom_path_resolver(self) -> None:
        queries = {"q1": "test"}
        qrels = {"q1": {"42": 1}}
        out = qrels_to_eval_queries(queries, qrels, path_for_doc_id=lambda d: f"scifact://{d}")
        assert out[0]["relevant_paths"] == ["scifact://42"]

    def test_drops_queries_with_no_relevant_docs(self) -> None:
        queries = {"q1": "ok", "q2": "drop me"}
        qrels = {"q1": {"a": 1}, "q2": {"b": 0}}
        out = qrels_to_eval_queries(queries, qrels)
        assert {q["query_id"] for q in out} == {"q1"}

    def test_min_relevance_threshold(self) -> None:
        queries = {"q1": "q"}
        qrels = {"q1": {"a": 1, "b": 2, "c": 0}}
        out_strict = qrels_to_eval_queries(queries, qrels, min_relevance=2)
        assert out_strict[0]["relevant_paths"] == ["b"]


# ------------------------------------------------------------------ #
# Held-out split                                                       #
# ------------------------------------------------------------------ #


class TestSplitCorpusForEval:
    """The train/eval split must be deterministic, disjoint, and capped."""

    def test_empty_store_returns_empty(self, sample_store: VstashStore) -> None:
        reserved, queries = split_corpus_for_eval(sample_store)
        assert reserved == set()
        assert queries == []

    def test_too_few_chunks_returns_empty(self, populated_store: VstashStore) -> None:
        # populated_store has 5 chunks, below the default min of 20.
        reserved, queries = split_corpus_for_eval(populated_store)
        assert reserved == set()
        assert queries == []

    def test_deterministic(self, populated_store: VstashStore) -> None:
        """Same seed -> same split. min_queries lowered for fixture size."""
        a_ids, a_queries = split_corpus_for_eval(populated_store, min_queries=2, max_queries=3)
        b_ids, b_queries = split_corpus_for_eval(populated_store, min_queries=2, max_queries=3)
        assert a_ids == b_ids
        assert [q["source_chunk_id"] for q in a_queries] == [
            q["source_chunk_id"] for q in b_queries
        ]

    def test_different_seeds_produce_different_splits(self, populated_store: VstashStore) -> None:
        a_ids, _ = split_corpus_for_eval(populated_store, min_queries=2, max_queries=3, seed=1)
        b_ids, _ = split_corpus_for_eval(populated_store, min_queries=2, max_queries=3, seed=99)
        # Not strictly guaranteed on tiny data, but overwhelmingly likely.
        assert a_ids != b_ids

    def test_queries_carry_required_fields(self, populated_store: VstashStore) -> None:
        _, queries = split_corpus_for_eval(populated_store, min_queries=2, max_queries=3)
        assert queries
        for q in queries:
            assert set(q.keys()) >= {"query", "relevant_paths", "source_chunk_id"}
            assert isinstance(q["relevant_paths"], list)
            assert len(q["relevant_paths"]) == 1
            # Legacy field kept for backward compat with any external caller.
            assert q.get("relevant_path") == q["relevant_paths"][0]
            assert len(q["query"]) <= 200


# ------------------------------------------------------------------ #
# generate_triples: exclude_chunk_ids                                  #
# ------------------------------------------------------------------ #


class TestGenerateTriplesExcludesHeldOut:
    """Chunks reserved for eval must never appear as pseudo-queries."""

    def test_exclude_ids_skipped(self, populated_store: VstashStore) -> None:
        all_ids = [
            r["id"] for r in populated_store._conn.execute("SELECT id FROM chunks").fetchall()
        ]
        # Reserve all but one chunk. Only that one should ever become a
        # pseudo-query.
        keep_id = all_ids[0]
        excluded = set(all_ids[1:])

        seen_ids: list[int] = []

        def capture_embed(text: str, *_a: Any, **_k: Any) -> list[float]:
            row = populated_store._conn.execute(
                "SELECT id FROM chunks WHERE text = ? OR text LIKE ? LIMIT 1",
                [text, text + "%"],
            ).fetchone()
            if row is not None:
                seen_ids.append(row["id"])
            return [0.0] * populated_store.embedding_dim

        with (
            patch("vstash.retrain.embed_query", side_effect=capture_embed),
            patch.object(populated_store, "search") as mock_search,
        ):
            mock_search.return_value = []
            generate_triples(
                populated_store,
                model_name="m",
                max_queries=10,
                exclude_chunk_ids=excluded,
            )

        assert all(sid == keep_id for sid in seen_ids), (
            f"excluded chunk ids leaked into pseudo-queries: {seen_ids}"
        )


# ------------------------------------------------------------------ #
# evaluate_model                                                       #
# ------------------------------------------------------------------ #


def _install_fake_st_model(dim: int, mapping: dict[str, list[float]]) -> Any:
    """Return a fake SentenceTransformer whose encode() reads from a table."""
    fake = MagicMock()
    fake.get_sentence_embedding_dimension = MagicMock(return_value=dim)
    # sentence-transformers v5.x renamed this method; support both so the
    # fake matches whichever attribute the production code probes first.
    fake.get_embedding_dimension = MagicMock(return_value=dim)

    def encode(
        texts: list[str] | str,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            key = t[:60]  # prefix-match to be robust to truncation
            v = None
            for k, vec in mapping.items():
                if key.startswith(k[:60]) or k.startswith(key):
                    v = vec
                    break
            if v is None:
                v = [0.0] * dim
            vecs.append(v)
        return np.asarray(vecs, dtype=float)

    fake.encode = encode
    return fake


class TestEvaluateModel:
    """evaluate_model reindexes with the target model and reports metrics."""

    def test_empty_queries_returns_zero(self, populated_store: VstashStore) -> None:
        metrics = evaluate_model(populated_store, model_name_or_path="m", eval_queries=[])
        assert metrics.n_queries == 0
        assert metrics.ndcg_at_10 == 0.0

    def test_perfect_retrieval_scores_one(self, populated_store: VstashStore) -> None:
        """If the target model puts the relevant chunk at rank 1 for every
        query, NDCG@10, MRR, and Hit@10 must all equal 1.0.
        """
        dim = populated_store.embedding_dim
        rows = populated_store._conn.execute(
            "SELECT c.text, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        assert rows, "populated_store should have chunks"

        # Give each distinct path its own dense basis vector.
        paths = sorted({r["path"] for r in rows})
        assert len(paths) <= dim, "test assumes n_paths <= embedding dim"
        path_vec: dict[str, list[float]] = {}
        for i, p in enumerate(paths):
            v = [0.0] * dim
            v[i] = 1.0
            path_vec[p] = v

        # mapping from text-prefix -> embedding. Chunks and queries derived
        # from the same chunk share the same relevant_path, so they share the
        # same vector -> perfect retrieval.
        mapping = {r["text"][:60]: path_vec[r["path"]] for r in rows}

        eval_queries = [
            {
                "query": r["text"][:200],
                "relevant_path": r["path"],
                "source_chunk_id": 0,
            }
            for r in rows
        ]

        fake_model = _install_fake_st_model(dim, mapping)
        with patch("vstash.retrain._load_sentence_transformer", return_value=fake_model):
            metrics = evaluate_model(
                populated_store,
                model_name_or_path="fake",
                eval_queries=eval_queries,
                noise_sample_size=0,
            )

        assert metrics.n_queries == len(rows)
        assert metrics.hit_at_10 == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.ndcg_at_10 == pytest.approx(1.0)

    def test_multi_relevant_query_counts_all_hits(self, populated_store: VstashStore) -> None:
        """When a single query has multiple relevant docs, NDCG credits
        hits against all of them up to k."""
        dim = populated_store.embedding_dim
        rows = populated_store._conn.execute(
            "SELECT c.text, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        paths = sorted({r["path"] for r in rows})
        # Use one basis vector per path, plus an extra dim for the query.
        path_vec: dict[str, list[float]] = {}
        for i, p in enumerate(paths):
            v = [0.0] * dim
            v[i] = 1.0
            path_vec[p] = v
        # Query vector sits halfway between the first two path vectors so
        # both come out near the top under cosine.
        q_vec = [0.0] * dim
        q_vec[0] = 1.0
        q_vec[1] = 1.0

        # Chunks keyed by 60-char prefix -> path's basis vector.
        mapping = {r["text"][:60]: path_vec[r["path"]] for r in rows}
        mapping["MULTIRELEVANT_QUERY"] = q_vec

        eval_queries = [
            {
                "query": "MULTIRELEVANT_QUERY",
                "relevant_paths": paths[:2],  # both top-2 paths are relevant
            }
        ]

        fake_model = _install_fake_st_model(dim, mapping)
        with patch("vstash.retrain._load_sentence_transformer", return_value=fake_model):
            metrics = evaluate_model(
                populated_store,
                model_name_or_path="fake",
                eval_queries=eval_queries,
                noise_sample_size=0,
            )

        # With both relevant docs retrieved in the top 2, NDCG must be 1.0.
        assert metrics.n_queries == 1
        assert metrics.ndcg_at_10 == pytest.approx(1.0)
        assert metrics.hit_at_10 == pytest.approx(1.0)

    def test_accepts_legacy_relevant_path_field(self, populated_store: VstashStore) -> None:
        """Queries with the legacy 'relevant_path' (singular) key still work."""
        dim = populated_store.embedding_dim
        rows = populated_store._conn.execute(
            "SELECT c.text, d.path FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        paths = sorted({r["path"] for r in rows})
        path_vec = {p: [0.0] * dim for p in paths}
        for i, p in enumerate(paths):
            path_vec[p][i] = 1.0
        mapping = {r["text"][:60]: path_vec[r["path"]] for r in rows}

        legacy_queries = [{"query": r["text"][:200], "relevant_path": r["path"]} for r in rows]
        fake_model = _install_fake_st_model(dim, mapping)
        with patch("vstash.retrain._load_sentence_transformer", return_value=fake_model):
            metrics = evaluate_model(
                populated_store,
                model_name_or_path="fake",
                eval_queries=legacy_queries,
                noise_sample_size=0,
            )
        assert metrics.n_queries == len(rows)
        assert metrics.ndcg_at_10 == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# retrain() orchestration                                              #
# ------------------------------------------------------------------ #


class TestRetrainOrchestration:
    """The composed retrain() entry gates on NDCG delta."""

    def test_skip_eval_saves_unconditionally(
        self, populated_store: VstashStore, tmp_path: Path, st_stubs: Any
    ) -> None:
        """--no-eval path: no baseline/final computed, model is saved."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        fake_pairs = [{"query": "q", "positive": "p", "negative": "n"}] * 20
        with patch("vstash.retrain.generate_triples", return_value=fake_pairs):
            result = retrain(
                populated_store,
                base_model="dummy",
                output_path=str(tmp_path / "m"),
                skip_eval=True,
                eval_fraction=0.15,
            )

        assert result.output_path == str(tmp_path / "m")
        assert result.baseline is None
        assert result.final is None
        assert result.n_pairs == 20
        assert result.gated_out is False
        assert (tmp_path / "m" / "training_meta.json").is_file()

    def test_gates_out_on_regression(
        self, populated_store: VstashStore, tmp_path: Path, st_stubs: Any
    ) -> None:
        """When the final model is worse than baseline and min_gain=0,
        retrain must NOT promote the candidate to ``output_path``."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        baseline_metrics = EvalMetrics(ndcg_at_10=0.80, mrr=0.80, hit_at_10=1.0, n_queries=25)
        final_metrics = EvalMetrics(ndcg_at_10=0.70, mrr=0.70, hit_at_10=0.9, n_queries=25)
        fake_eval_queries = [
            {"query": f"q{i}", "relevant_path": f"/p{i}", "source_chunk_id": i} for i in range(25)
        ]
        fake_pairs = [{"query": "q", "positive": "p", "negative": "n"}] * 20

        with (
            patch(
                "vstash.retrain.split_corpus_for_eval",
                return_value=(set(), fake_eval_queries),
            ),
            patch(
                "vstash.retrain.evaluate_model",
                side_effect=[baseline_metrics, final_metrics],
            ),
            patch("vstash.retrain.generate_triples", return_value=fake_pairs),
        ):
            result = retrain(
                populated_store,
                base_model="dummy",
                output_path=str(tmp_path / "m"),
                min_gain=0.0,
            )

        assert result.gated_out is True
        assert result.output_path is None
        assert result.delta_ndcg == pytest.approx(-0.10)
        # Candidate directory must be retained for inspection.
        assert (tmp_path / "m.candidate").is_dir()
        # Final model directory must NOT exist.
        assert not (tmp_path / "m").exists()
        # Eval block must be persisted into candidate meta.
        meta = json.loads((tmp_path / "m.candidate" / "training_meta.json").read_text())
        assert meta["eval"]["gated_out"] is True
        assert meta["eval"]["delta_ndcg_at_10"] == pytest.approx(-0.10, abs=1e-5)

    def test_promotes_candidate_on_gain(
        self, populated_store: VstashStore, tmp_path: Path, st_stubs: Any
    ) -> None:
        """When the final model improves NDCG, the candidate is promoted
        to ``output_path`` and the .candidate directory disappears."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        baseline_metrics = EvalMetrics(ndcg_at_10=0.70, mrr=0.70, hit_at_10=0.9, n_queries=25)
        final_metrics = EvalMetrics(ndcg_at_10=0.80, mrr=0.80, hit_at_10=1.0, n_queries=25)
        fake_eval_queries = [
            {"query": f"q{i}", "relevant_path": f"/p{i}", "source_chunk_id": i} for i in range(25)
        ]
        fake_pairs = [{"query": "q", "positive": "p"}] * 20

        with (
            patch(
                "vstash.retrain.split_corpus_for_eval",
                return_value=(set(), fake_eval_queries),
            ),
            patch(
                "vstash.retrain.evaluate_model",
                side_effect=[baseline_metrics, final_metrics],
            ),
            patch("vstash.retrain.generate_triples", return_value=fake_pairs),
        ):
            result = retrain(
                populated_store,
                base_model="dummy",
                output_path=str(tmp_path / "m"),
                min_gain=0.0,
            )

        assert result.gated_out is False
        assert result.output_path == str(tmp_path / "m")
        assert result.delta_ndcg == pytest.approx(0.10)
        assert (tmp_path / "m" / "training_meta.json").is_file()
        assert not (tmp_path / "m.candidate").exists()

        meta = json.loads((tmp_path / "m" / "training_meta.json").read_text())
        assert meta["eval"]["gated_out"] is False
        assert meta["eval"]["baseline"]["ndcg_at_10"] == pytest.approx(0.70)
        assert meta["eval"]["final"]["ndcg_at_10"] == pytest.approx(0.80)

    def test_insufficient_eval_queries_falls_back_to_no_eval(
        self, populated_store: VstashStore, tmp_path: Path, st_stubs: Any
    ) -> None:
        """When the corpus is too small to build a valid eval set, retrain
        must fall back to skip_eval=True instead of crashing."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        fake_pairs = [{"query": "q", "positive": "p"}] * 20
        with (
            patch("vstash.retrain.split_corpus_for_eval", return_value=(set(), [])),
            patch("vstash.retrain.generate_triples", return_value=fake_pairs),
            patch("vstash.retrain.evaluate_model") as mock_eval,
        ):
            result = retrain(
                populated_store,
                base_model="dummy",
                output_path=str(tmp_path / "m"),
            )

        assert mock_eval.call_count == 0, "evaluate_model must not be called when eval set is empty"
        assert result.output_path == str(tmp_path / "m")
        assert result.baseline is None
        assert result.final is None

    def test_external_eval_queries_override_split(
        self, populated_store: VstashStore, tmp_path: Path, st_stubs: Any
    ) -> None:
        """When the caller passes eval_queries explicitly, the internal
        pseudo-query split is skipped and the supplied queries are used
        verbatim for both baseline and final eval."""
        st_mod, _, _ = st_stubs
        st_mod.SentenceTransformer.return_value = MagicMock()

        baseline_metrics = EvalMetrics(ndcg_at_10=0.55, mrr=0.55, hit_at_10=0.7, n_queries=30)
        final_metrics = EvalMetrics(ndcg_at_10=0.60, mrr=0.60, hit_at_10=0.75, n_queries=30)

        external = [
            {"query": f"labeled q{i}", "relevant_paths": [f"/p{i}"], "query_id": f"q{i}"}
            for i in range(30)
        ]
        fake_pairs = [{"query": "q", "positive": "p"}] * 20

        with (
            patch("vstash.retrain.split_corpus_for_eval") as mock_split,
            patch(
                "vstash.retrain.evaluate_model",
                side_effect=[baseline_metrics, final_metrics],
            ) as mock_eval,
            patch("vstash.retrain.generate_triples", return_value=fake_pairs),
        ):
            result = retrain(
                populated_store,
                base_model="dummy",
                output_path=str(tmp_path / "m"),
                eval_queries=external,
            )

        # split_corpus_for_eval must be bypassed when external queries supplied.
        assert mock_split.call_count == 0
        # Both baseline and final must receive the same external query set.
        assert mock_eval.call_count == 2
        for call in mock_eval.call_args_list:
            assert call.kwargs["eval_queries"] is external
        assert result.gated_out is False
        assert result.delta_ndcg == pytest.approx(0.05)
