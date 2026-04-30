"""Benchmark regression tests for adaptive RRF on BEIR datasets.

These tests re-run the BEIR evaluation pipeline and assert that NDCG@10
does not regress below the published baseline. They protect the claims
made in the paper and README about adaptive RRF improving all 5 BEIR
datasets vs fixed weights.

Marked with @pytest.mark.benchmark — skipped in normal CI.
Run explicitly with:
    pytest tests/test_beir_regression.py -m benchmark
    pytest tests/test_beir_regression.py -m benchmark -k scifact   # single dataset
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# These tests require BEIR data and embedding model — slow + heavy
pytestmark = pytest.mark.benchmark

BASELINE_PATH = (
    Path(__file__).parent.parent / "experiments" / "results" / "beir_adaptive_baseline.json"
)
BEIR_CACHE = Path(__file__).parent.parent / "experiments" / "data"

# Maximum allowed regression from baseline (absolute NDCG points)
# 0.005 = half a percentage point — catches real regressions but allows
# minor floating-point / library-version variance
REGRESSION_TOLERANCE = 0.005


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _beir_data_available(dataset: str) -> bool:
    return (BEIR_CACHE / f"beir_{dataset}").exists()


def _skip_if_no_data(dataset: str) -> None:
    if not _beir_data_available(dataset):
        pytest.skip(
            f"BEIR {dataset} data not cached — run: python -m experiments.beir_benchmark --datasets {dataset}"
        )


def _evaluate_dataset(dataset: str) -> dict:
    """Run vstash evaluation on a single BEIR dataset, return metrics dict."""
    from experiments.beir_benchmark import (
        download_beir,
        evaluate_vstash,
        load_beir,
    )
    from vstash.embed import embed_texts, get_embedding_dim
    from vstash.store import VstashStore

    model_id = "BAAI/bge-small-en-v1.5"
    dim = get_embedding_dim(model_id)

    cache = download_beir(dataset)
    corpus, queries, qrels = load_beir(cache)

    # Embed corpus
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    all_embeddings: list[list[float]] = []
    for bs in range(0, len(doc_texts), 64):
        all_embeddings.extend(embed_texts(doc_texts[bs : bs + 64], model_id))

    # Ingest into vstash
    db_path = tempfile.mktemp(suffix=".db")
    try:
        store = VstashStore(db_path, embedding_dim=dim)
        vstash_id_map: dict[str, str] = {}
        for doc_id, text, emb in zip(doc_ids, doc_texts, all_embeddings, strict=False):
            path = f"/beir/{doc_id}"
            store.add_document(
                path=path,
                title=corpus[doc_id].get("title", ""),
                chunks=[text],
                embeddings=[emb],
                source_type="text",
            )
            vstash_id_map[path] = doc_id

        metrics = evaluate_vstash(store, vstash_id_map, queries, qrels, model_id)
        store.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

    return metrics


class TestBEIRRegression:
    """Assert adaptive RRF NDCG@10 does not regress below published baseline."""

    def test_scifact_ndcg(self) -> None:
        _skip_if_no_data("scifact")
        baseline = _load_baseline()
        expected = baseline["results"]["scifact"]["ndcg_10"]
        actual = _evaluate_dataset("scifact")
        assert actual["ndcg_10"] >= expected - REGRESSION_TOLERANCE, (
            f"SciFact NDCG@10 regressed: {actual['ndcg_10']:.4f} < "
            f"baseline {expected:.4f} - {REGRESSION_TOLERANCE}"
        )

    def test_nfcorpus_ndcg(self) -> None:
        _skip_if_no_data("nfcorpus")
        baseline = _load_baseline()
        expected = baseline["results"]["nfcorpus"]["ndcg_10"]
        actual = _evaluate_dataset("nfcorpus")
        assert actual["ndcg_10"] >= expected - REGRESSION_TOLERANCE, (
            f"NFCorpus NDCG@10 regressed: {actual['ndcg_10']:.4f} < "
            f"baseline {expected:.4f} - {REGRESSION_TOLERANCE}"
        )

    def test_scidocs_ndcg(self) -> None:
        _skip_if_no_data("scidocs")
        baseline = _load_baseline()
        expected = baseline["results"]["scidocs"]["ndcg_10"]
        actual = _evaluate_dataset("scidocs")
        assert actual["ndcg_10"] >= expected - REGRESSION_TOLERANCE, (
            f"SciDocs NDCG@10 regressed: {actual['ndcg_10']:.4f} < "
            f"baseline {expected:.4f} - {REGRESSION_TOLERANCE}"
        )

    def test_fiqa_ndcg(self) -> None:
        _skip_if_no_data("fiqa")
        baseline = _load_baseline()
        expected = baseline["results"]["fiqa"]["ndcg_10"]
        actual = _evaluate_dataset("fiqa")
        assert actual["ndcg_10"] >= expected - REGRESSION_TOLERANCE, (
            f"FiQA NDCG@10 regressed: {actual['ndcg_10']:.4f} < "
            f"baseline {expected:.4f} - {REGRESSION_TOLERANCE}"
        )

    def test_arguana_ndcg(self) -> None:
        _skip_if_no_data("arguana")
        baseline = _load_baseline()
        expected = baseline["results"]["arguana"]["ndcg_10"]
        actual = _evaluate_dataset("arguana")
        assert actual["ndcg_10"] >= expected - REGRESSION_TOLERANCE, (
            f"ArguAna NDCG@10 regressed: {actual['ndcg_10']:.4f} < "
            f"baseline {expected:.4f} - {REGRESSION_TOLERANCE}"
        )

    def test_adaptive_beats_fixed_on_all_datasets(self) -> None:
        """Verify that adaptive RRF still beats fixed RRF on every dataset
        where we have both baseline numbers."""
        baseline = _load_baseline()
        for dataset, fixed_data in baseline.get("fixed_rrf_comparison", {}).items():
            _skip_if_no_data(dataset)
            actual = _evaluate_dataset(dataset)
            fixed_ndcg = fixed_data["ndcg_10"]
            assert actual["ndcg_10"] >= fixed_ndcg, (
                f"Adaptive RRF no longer beats fixed on {dataset}: "
                f"adaptive={actual['ndcg_10']:.4f} < fixed={fixed_ndcg:.4f}"
            )
