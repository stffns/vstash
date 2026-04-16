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
from vstash.retrain import generate_triples, train_mnrl
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
