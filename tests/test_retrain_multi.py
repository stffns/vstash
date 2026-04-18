"""Tests for vstash.retrain.retrain_multi -- multi-corpus training (T1.4)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vstash.embed import get_embedding_dim
from vstash.retrain import (
    EvalMetrics,
    MultiRetrainResult,
    compute_triple_budget,
    retrain_multi,
)
from vstash.store import VstashStore


# ------------------------------------------------------------------ #
# compute_triple_budget                                                #
# ------------------------------------------------------------------ #


class TestComputeTripleBudget:
    def test_empty_sizes_returns_empty(self) -> None:
        assert compute_triple_budget({}, 1000) == {}

    def test_zero_total_returns_zero_per_dataset(self) -> None:
        out = compute_triple_budget({"a": 100, "b": 200}, 0)
        assert out == {"a": 0, "b": 0}

    def test_uniform_distributes_evenly(self) -> None:
        # 9000 / 3 = 3000 each, no rounding.
        out = compute_triple_budget({"a": 100, "b": 1000, "c": 50}, 9000, strategy="uniform")
        assert out == {"a": 3000, "b": 3000, "c": 3000}
        assert sum(out.values()) == 9000

    def test_uniform_respects_total_with_rounding(self) -> None:
        # 10000 / 3 leaves remainder 1, distributed by largest-remainder.
        out = compute_triple_budget({"a": 100, "b": 1000, "c": 50}, 10000, strategy="uniform")
        assert sum(out.values()) == 10000
        for v in out.values():
            assert abs(v - 10000 / 3) <= 1

    def test_proportional_matches_sizes(self) -> None:
        sizes = {"a": 1000, "b": 3000, "c": 6000}  # 10k total -> 10%/30%/60%
        out = compute_triple_budget(sizes, 10000, strategy="proportional")
        assert sum(out.values()) == 10000
        assert out["a"] == pytest.approx(1000, abs=1)
        assert out["b"] == pytest.approx(3000, abs=1)
        assert out["c"] == pytest.approx(6000, abs=1)

    def test_temperature_zero_equals_uniform(self) -> None:
        sizes = {"a": 100, "b": 1000, "c": 50}
        temp = compute_triple_budget(sizes, 3000, strategy="temperature", temperature=0.0)
        uni = compute_triple_budget(sizes, 3000, strategy="uniform")
        assert temp == uni

    def test_temperature_one_equals_proportional(self) -> None:
        sizes = {"a": 1000, "b": 3000, "c": 6000}
        temp = compute_triple_budget(sizes, 10000, strategy="temperature", temperature=1.0)
        prop = compute_triple_budget(sizes, 10000, strategy="proportional")
        assert temp == prop

    def test_temperature_half_favours_small_datasets_vs_proportional(self) -> None:
        """With alpha=0.5, NFCorpus (3.6k) gets a bigger share than its raw
        size fraction in a scifact+nfcorpus+fiqa mix. This is the whole
        point of temperature sampling for T1.4."""
        sizes = {"scifact": 5000, "nfcorpus": 3600, "fiqa": 57000}
        total = 10000
        prop = compute_triple_budget(sizes, total, strategy="proportional")
        temp = compute_triple_budget(sizes, total, strategy="temperature", temperature=0.5)

        # Small corpora get more under temperature sampling.
        assert temp["nfcorpus"] > prop["nfcorpus"]
        assert temp["scifact"] > prop["scifact"]
        # FiQA gets less than its raw-size share (which was 85%+).
        assert temp["fiqa"] < prop["fiqa"]
        # But FiQA is still the largest share overall (alpha<1 damps, not inverts).
        assert temp["fiqa"] > temp["scifact"] > temp["nfcorpus"]
        assert sum(temp.values()) == total

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be"):
            compute_triple_budget({"a": 100}, 1000, strategy="temperature", temperature=-1.0)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            compute_triple_budget({"a": 100}, 1000, strategy="bogus")  # type: ignore[arg-type]

    def test_all_zero_sizes_returns_zero_budgets(self) -> None:
        """Empty corpora produce zero weight under proportional/temperature;
        distribution must not divide-by-zero."""
        out = compute_triple_budget({"a": 0, "b": 0}, 100, strategy="temperature")
        assert out == {"a": 0, "b": 0}

    def test_sum_equals_total_for_many_datasets(self) -> None:
        # Stress test largest-remainder math.
        sizes = {f"d{i}": 100 + i * 37 for i in range(20)}
        for strategy in ("uniform", "proportional", "temperature"):
            out = compute_triple_budget(sizes, 12345, strategy=strategy)  # type: ignore[arg-type]
            assert sum(out.values()) == 12345


# ------------------------------------------------------------------ #
# retrain_multi integration (with mocks for the ML parts)              #
# ------------------------------------------------------------------ #


def _install_st_torch_stubs() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
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


def _mk_store(path: str, docs: list[tuple[str, str, str]]) -> VstashStore:
    """Create and populate a small VstashStore.

    docs: list of (path, title, chunk_text). Embeddings are deterministic
    per chunk so the stores do not share content or vectors.
    """
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


@pytest.fixture
def triple_stores(tmp_path: Path) -> dict[str, VstashStore]:
    """Three distinct stores of different sizes, so sampling math is
    observable in the budget/pairs assertions."""
    small = _mk_store(
        str(tmp_path / "small.db"),
        [(f"/small/{i}", f"Small {i}", f"small doc {i} text body") for i in range(12)],
    )
    medium = _mk_store(
        str(tmp_path / "medium.db"),
        [(f"/medium/{i}", f"Medium {i}", f"medium doc {i} text body") for i in range(30)],
    )
    big = _mk_store(
        str(tmp_path / "big.db"),
        [(f"/big/{i}", f"Big {i}", f"big doc {i} text body") for i in range(60)],
    )
    stores = {"small": small, "medium": medium, "big": big}
    yield stores
    for s in stores.values():
        s.close()


def test_raises_when_stores_empty_dict() -> None:
    with pytest.raises(ValueError, match="at least one store"):
        retrain_multi({}, base_model="m")


def test_raises_when_stores_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one store"):
        retrain_multi([], base_model="m")


def test_raises_when_all_stores_are_empty(tmp_path: Path) -> None:
    empty_a = _mk_store(str(tmp_path / "a.db"), [])
    empty_b = _mk_store(str(tmp_path / "b.db"), [])
    try:
        with pytest.raises(ValueError, match="All provided stores are empty"):
            retrain_multi({"a": empty_a, "b": empty_b}, base_model="m")
    finally:
        empty_a.close()
        empty_b.close()


def test_list_input_gets_auto_aliased(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """A plain list of stores must be normalized into corpus_0 / corpus_1 / ..."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    baseline_metrics = EvalMetrics(ndcg_at_10=0.50, mrr=0.50, hit_at_10=0.6, n_queries=25)
    final_metrics = EvalMetrics(ndcg_at_10=0.55, mrr=0.55, hit_at_10=0.65, n_queries=25)

    with (
        patch(
            "vstash.retrain.split_corpus_for_eval",
            return_value=(
                set(),
                [{"query": f"q{i}", "relevant_paths": [f"/p{i}"]} for i in range(25)],
            ),
        ),
        patch(
            "vstash.retrain.evaluate_model",
            side_effect=[baseline_metrics, baseline_metrics, baseline_metrics]
            + [final_metrics, final_metrics, final_metrics],
        ),
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        result = retrain_multi(
            list(triple_stores.values()),
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            min_gain=0.0,
        )

    assert set(result.per_dataset_pairs.keys()) == {"corpus_0", "corpus_1", "corpus_2"}
    assert result.output_path == str(tmp_path / "m")


def test_happy_path_macro_gate_passes_and_promotes(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """Baseline and final deltas differ per dataset; the macro average
    clears min_gain so the candidate is promoted to output_path."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    baseline_small = EvalMetrics(ndcg_at_10=0.50, mrr=0.5, hit_at_10=0.6, n_queries=20)
    baseline_medium = EvalMetrics(ndcg_at_10=0.60, mrr=0.6, hit_at_10=0.7, n_queries=25)
    baseline_big = EvalMetrics(ndcg_at_10=0.70, mrr=0.7, hit_at_10=0.8, n_queries=30)
    final_small = EvalMetrics(ndcg_at_10=0.60, mrr=0.6, hit_at_10=0.7, n_queries=20)
    final_medium = EvalMetrics(ndcg_at_10=0.55, mrr=0.55, hit_at_10=0.65, n_queries=25)
    final_big = EvalMetrics(ndcg_at_10=0.80, mrr=0.8, hit_at_10=0.85, n_queries=30)
    # macro baseline = 0.60, macro final = 0.65, delta = +0.05 -> passes min_gain=0.0

    eval_payload = [{"query": f"q{i}", "relevant_paths": [f"/p{i}"]} for i in range(25)]

    eval_sequence = [
        baseline_small,
        baseline_medium,
        baseline_big,
        final_small,
        final_medium,
        final_big,
    ]

    with (
        patch("vstash.retrain.split_corpus_for_eval", return_value=(set(), eval_payload)),
        patch("vstash.retrain.evaluate_model", side_effect=eval_sequence) as mock_eval,
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        result = retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=600,
            sampling="temperature",
            temperature=0.5,
            min_gain=0.0,
        )

    assert isinstance(result, MultiRetrainResult)
    assert result.gated_out is False
    assert result.output_path == str(tmp_path / "m")
    assert result.macro_baseline_ndcg == pytest.approx(0.60)
    assert result.macro_final_ndcg == pytest.approx(0.65)
    assert result.macro_delta_ndcg == pytest.approx(0.05)
    # Per-dataset deltas must be visible so callers can display them.
    assert result.per_dataset_delta["small"] == pytest.approx(0.10)
    assert result.per_dataset_delta["medium"] == pytest.approx(-0.05)
    assert result.per_dataset_delta["big"] == pytest.approx(0.10)
    # evaluate_model called 6 times: baseline + final, 3 datasets each.
    assert mock_eval.call_count == 6

    # Meta persists per-dataset eval + budget on the final model dir.
    meta = json.loads((tmp_path / "m" / "training_meta.json").read_text())
    assert meta["multi_eval"]["sampling"] == "temperature"
    assert meta["multi_eval"]["temperature"] == 0.5
    assert meta["multi_eval"]["macro_delta_ndcg"] == pytest.approx(0.05)
    assert set(meta["multi_eval"]["per_dataset_baseline"].keys()) == {"small", "medium", "big"}
    # Budget must sum to total_triples even though small corpus is only 12 chunks.
    assert sum(meta["multi_eval"]["per_dataset_budget"].values()) == 600


def test_macro_gate_blocks_promotion_on_regression(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """Macro delta below min_gain leaves the candidate on disk but does
    NOT promote it to output_path."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    baseline = EvalMetrics(ndcg_at_10=0.70, mrr=0.7, hit_at_10=0.8, n_queries=25)
    worse = EvalMetrics(ndcg_at_10=0.65, mrr=0.65, hit_at_10=0.75, n_queries=25)
    eval_payload = [{"query": f"q{i}", "relevant_paths": [f"/p{i}"]} for i in range(25)]

    with (
        patch("vstash.retrain.split_corpus_for_eval", return_value=(set(), eval_payload)),
        patch(
            "vstash.retrain.evaluate_model",
            side_effect=[baseline, baseline, baseline, worse, worse, worse],
        ),
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        result = retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            min_gain=0.0,
        )

    assert result.gated_out is True
    assert result.output_path is None
    # Candidate dir preserved for inspection; final path does not exist.
    assert (tmp_path / "m.candidate").is_dir()
    assert not (tmp_path / "m").exists()


def test_per_dataset_gate_rejects_any_single_regression(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """With per_dataset_gate=True, a single dataset regression (even
    when the macro is positive) must block promotion."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    baseline_small = EvalMetrics(ndcg_at_10=0.50, mrr=0.5, hit_at_10=0.6, n_queries=20)
    baseline_medium = EvalMetrics(ndcg_at_10=0.60, mrr=0.6, hit_at_10=0.7, n_queries=25)
    baseline_big = EvalMetrics(ndcg_at_10=0.70, mrr=0.7, hit_at_10=0.8, n_queries=30)
    final_small = EvalMetrics(ndcg_at_10=0.60, mrr=0.6, hit_at_10=0.7, n_queries=20)
    final_medium = EvalMetrics(
        ndcg_at_10=0.55, mrr=0.55, hit_at_10=0.65, n_queries=25
    )  # regression
    final_big = EvalMetrics(ndcg_at_10=0.80, mrr=0.8, hit_at_10=0.85, n_queries=30)

    eval_payload = [{"query": f"q{i}", "relevant_paths": [f"/p{i}"]} for i in range(25)]
    seq = [baseline_small, baseline_medium, baseline_big, final_small, final_medium, final_big]

    with (
        patch("vstash.retrain.split_corpus_for_eval", return_value=(set(), eval_payload)),
        patch("vstash.retrain.evaluate_model", side_effect=seq),
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        result = retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            min_gain=0.0,
            per_dataset_gate=True,
        )

    assert result.gated_out is True
    assert result.output_path is None
    # Macro delta is still positive, but per-dataset gate blocks it.
    assert result.macro_delta_ndcg > 0


def test_eval_queries_by_dataset_skips_internal_split(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """When per-dataset labeled queries are passed, split_corpus_for_eval
    must not be called for those datasets."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    labeled_small = [
        {"query": f"small-q{i}", "relevant_paths": [f"/small/{i}"], "query_id": f"s{i}"}
        for i in range(25)
    ]
    labeled_medium = [
        {"query": f"medium-q{i}", "relevant_paths": [f"/medium/{i}"], "query_id": f"m{i}"}
        for i in range(25)
    ]

    baseline = EvalMetrics(ndcg_at_10=0.5, mrr=0.5, hit_at_10=0.6, n_queries=25)
    final = EvalMetrics(ndcg_at_10=0.6, mrr=0.6, hit_at_10=0.7, n_queries=25)
    fallback_queries = [{"query": f"q{i}", "relevant_paths": [f"/big/{i}"]} for i in range(25)]

    with (
        patch(
            "vstash.retrain.split_corpus_for_eval", return_value=(set(), fallback_queries)
        ) as mock_split,
        patch(
            "vstash.retrain.evaluate_model",
            side_effect=[baseline, baseline, baseline, final, final, final],
        ) as mock_eval,
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            eval_queries_by_dataset={"small": labeled_small, "medium": labeled_medium},
            min_gain=0.0,
        )

    # Only the "big" dataset (not passed) should trigger split_corpus_for_eval.
    assert mock_split.call_count == 1
    # Baseline eval received the labeled queries for small + medium, the
    # fallback for big. Check the first three calls (baseline pass).
    baseline_calls = mock_eval.call_args_list[:3]
    baseline_query_sets = [call.kwargs["eval_queries"] for call in baseline_calls]
    assert labeled_small in baseline_query_sets
    assert labeled_medium in baseline_query_sets
    assert fallback_queries in baseline_query_sets


def test_skip_eval_always_saves(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """skip_eval=True must train + promote without calling evaluate_model."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    with (
        patch("vstash.retrain.evaluate_model") as mock_eval,
        patch("vstash.retrain.split_corpus_for_eval") as mock_split,
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        result = retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            skip_eval=True,
        )

    assert mock_eval.call_count == 0
    assert mock_split.call_count == 0
    assert result.gated_out is False
    assert result.output_path == str(tmp_path / "m")


def test_too_few_pairs_gated_out(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """Total pairs across all corpora below the 10-pair minimum must
    flag gated_out=True and never save a model."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    baseline = EvalMetrics(ndcg_at_10=0.5, mrr=0.5, hit_at_10=0.6, n_queries=25)
    eval_payload = [{"query": f"q{i}", "relevant_paths": [f"/p{i}"]} for i in range(25)]

    with (
        patch("vstash.retrain.split_corpus_for_eval", return_value=(set(), eval_payload)),
        patch("vstash.retrain.evaluate_model", return_value=baseline),
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 2,
        ),
    ):
        result = retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            min_gain=0.0,
        )

    assert result.gated_out is True
    assert result.output_path is None
    assert result.total_pairs < 10
    assert not (tmp_path / "m").exists()


def test_use_amp_and_max_seq_length_forwarded_to_train_mnrl(
    triple_stores: dict[str, VstashStore], tmp_path: Path, st_stubs: Any
) -> None:
    """use_amp + max_seq_length must reach train_mnrl so the T4-safe
    defaults in the CLI and notebook actually take effect during
    training."""
    st_mod, _, _ = st_stubs
    st_mod.SentenceTransformer.return_value = MagicMock()

    with (
        patch("vstash.retrain.train_mnrl") as mock_train,
        patch(
            "vstash.retrain.generate_triples",
            return_value=[{"query": "q", "positive": "p", "negative": None}] * 10,
        ),
    ):
        retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=300,
            sampling="uniform",
            skip_eval=True,
            batch_size=16,
            use_amp=True,
            max_seq_length=128,
        )

    assert mock_train.call_count == 1
    kwargs = mock_train.call_args.kwargs
    assert kwargs["use_amp"] is True
    assert kwargs["max_seq_length"] == 128
    assert kwargs["batch_size"] == 16


def test_synthesize_queries_requires_cfg(
    triple_stores: dict[str, VstashStore], tmp_path: Path
) -> None:
    """Mirrors the single-store retrain() contract: synthesize_queries=True
    without a cfg argument is a usage error."""
    with pytest.raises(ValueError, match="cfg"):
        retrain_multi(
            triple_stores,
            base_model="dummy",
            output_path=str(tmp_path / "m"),
            total_triples=30,
            synthesize_queries=True,
            skip_eval=True,
        )
