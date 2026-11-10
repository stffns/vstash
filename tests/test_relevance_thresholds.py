"""Per-model relevance-tier thresholds (``vstash.relevance`` + ``[relevance]``).

BGE-small keeps the historical 0.4513 / 0.4802 cutoffs; calibrated models get
their own (so a MiniLM good answer at distance ~0.55 is no longer "low"); a
``[relevance]`` override wins over both; adapters label searches with the
active model's thresholds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import EmbeddingsConfig, RelevanceConfig, VstashConfig
from vstash.relevance import (
    DEFAULT_RELEVANCE_THRESHOLDS,
    MODEL_RELEVANCE_THRESHOLDS,
    RelevanceThresholds,
    relevance_thresholds,
    relevance_tier,
    thresholds_for,
)

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _cfg(model: str = "BAAI/bge-small-en-v1.5", **rel) -> VstashConfig:
    return VstashConfig(embeddings=EmbeddingsConfig(model=model), relevance=RelevanceConfig(**rel))


class TestRegistry:
    def test_bge_defaults_unchanged(self) -> None:
        t = _cfg().relevance_thresholds()
        assert t == DEFAULT_RELEVANCE_THRESHOLDS == (0.4513, 0.4802)
        assert [relevance_tier(d, t) for d in (0.45, 0.47, 0.49)] == ["high", "medium", "low"]

    def test_default_arg_is_bge(self) -> None:
        assert relevance_tier(0.47) == "medium"

    def test_unknown_model_falls_back(self) -> None:
        assert relevance_thresholds("someone/unknown-model") == DEFAULT_RELEVANCE_THRESHOLDS

    def test_minilm_good_answers_not_low(self) -> None:
        # KakesuI: correct MiniLM answers at 0.49-0.56 were all "low" under BGE's cutoffs.
        t = _cfg(MINILM).relevance_thresholds()
        assert t == MODEL_RELEVANCE_THRESHOLDS[MINILM]
        assert all(relevance_tier(d, t) == "high" for d in (0.49, 0.52, 0.56))
        assert relevance_tier(0.56) == "low"

    def test_registry_entries_are_ordered(self) -> None:
        for model, t in MODEL_RELEVANCE_THRESHOLDS.items():
            assert 0 < t.high_max <= t.medium_max <= 2, model

    def test_back_compat_imports(self) -> None:
        from vstash.store import RELEVANCE_TIER_HIGH_MAX, RELEVANCE_TIER_MEDIUM_MAX
        from vstash.store import relevance_tier as store_tier

        assert (RELEVANCE_TIER_HIGH_MAX, RELEVANCE_TIER_MEDIUM_MAX) == (0.4513, 0.4802)
        assert store_tier is relevance_tier


class TestOverride:
    def test_full_override(self) -> None:
        t = _cfg(MINILM, high_max=0.5, medium_max=0.58).relevance_thresholds()
        assert t == RelevanceThresholds(0.5, 0.58)

    def test_partial_override_keeps_model_value(self) -> None:
        t = _cfg(MINILM, medium_max=0.75).relevance_thresholds()
        assert t == RelevanceThresholds(MODEL_RELEVANCE_THRESHOLDS[MINILM].high_max, 0.75)

    def test_inconsistent_override_rejected_at_load(self) -> None:
        with pytest.raises(ValueError, match="high_max <= medium_max"):
            _cfg(high_max=0.6)  # above BGE's medium_max 0.4802

    def test_lone_medium_below_model_high_rejected_with_hint(self) -> None:
        # MiniLM's default high_max (0.6786) would exceed a lone medium_max=0.58.
        with pytest.raises(ValueError, match="set both"):
            _cfg(MINILM, medium_max=0.58)
        assert _cfg(MINILM, high_max=0.5, medium_max=0.58).relevance_thresholds() == (0.5, 0.58)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 2.5])
    def test_out_of_range_rejected(self, bad) -> None:
        with pytest.raises(ValueError):
            RelevanceConfig(high_max=bad)

    def test_toml_section(self, tmp_path: Path, monkeypatch) -> None:
        from vstash.config import load_config

        f = tmp_path / "vstash.toml"
        f.write_text("[relevance]\nhigh_max = 0.4\nmedium_max = 0.45\n")
        monkeypatch.setenv("VSTASH_CONFIG", str(f))
        assert load_config().relevance_thresholds() == RelevanceThresholds(0.4, 0.45)


class TestThresholdsFor:
    def test_real_config(self) -> None:
        assert thresholds_for(_cfg(MINILM)) == MODEL_RELEVANCE_THRESHOLDS[MINILM]

    def test_mock_or_missing_falls_back(self) -> None:
        assert thresholds_for(MagicMock()) is None
        assert thresholds_for(object()) is None


DIM = 384


def _vec(cos: float) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[1] = cos, math.sqrt(1 - cos * cos)
    return v.tolist()


@requires_sqlite_vec
class TestAdapters:
    @pytest.fixture
    def store(self, tmp_path: Path):
        from vstash.store import VstashStore

        s = VstashStore(str(tmp_path / "r.db"), embedding_dim=DIM)
        s.add_document(
            path="text://nota",
            title="nota",
            chunks=["nota sobre el modelo de la memoria"],
            embeddings=[_vec(1.0)],
            source_type="text",
        )
        yield s
        s.close()

    def test_mcp_and_log_use_model_thresholds(self, store) -> None:
        from vstash.mcp import vstash_search

        q = _vec(0.45)  # cosine distance 0.55: "low" for BGE, "high" for MiniLM
        out = {}
        for model in ("BAAI/bge-small-en-v1.5", MINILM):
            with (
                patch("vstash.mcp._get_store", return_value=store),
                patch("vstash.mcp._get_config", return_value=_cfg(model)),
                patch("vstash.services.search.embed_query", return_value=q),
            ):
                res = json.loads(vstash_search("modelo memoria"))
            logged = store._conn.execute(
                "SELECT relevance_tier FROM search_events WHERE id = ?", [res["event_id"]]
            ).fetchone()[0]
            out[model] = (res["relevance"], logged)
        assert out["BAAI/bge-small-en-v1.5"] == ("low", "low")
        assert out[MINILM] == ("high", "high")
