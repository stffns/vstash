"""Relevance tiers: classify a search's best cosine distance as high / medium / low.

The tiers are an **off-topic detector**, not an answerability detector:
"low" means the best match is about as far from the query as a typical
question from an unrelated domain. A question about something the store
covers but never recorded ("on-topic but not stored") usually still lands in
"high" -- measured in ``experiments/results/relevance_tiers_calibration.md``.

Cosine distances live on a different scale for every embedding model, so the
thresholds are per model. :data:`MODEL_RELEVANCE_THRESHOLDS` holds calibrated
values; any other model falls back to the BGE-small defaults, and the
``[relevance]`` config section overrides either.

This module has no vstash imports so config, store and adapters can all use it.
"""

from __future__ import annotations

from typing import NamedTuple

#: High-confidence cosine distance cutoff for BGE-small. Cosine equivalent of
#: the legacy L2-on-unit-vectors threshold 0.95 (``0.95^2 / 2 = 0.4513``), so
#: BGE kept identical tier assignments across the v1 -> v2 metric change (#272).
RELEVANCE_TIER_HIGH_MAX = 0.4513

#: Medium-confidence cutoff for BGE-small: cosine equivalent of the legacy L2
#: threshold 0.98 (``0.98^2 / 2 = 0.4802``). Anything above is "low".
RELEVANCE_TIER_MEDIUM_MAX = 0.4802


class RelevanceThresholds(NamedTuple):
    """Upper cosine-distance bounds of the "high" and "medium" tiers."""

    high_max: float
    medium_max: float


DEFAULT_RELEVANCE_THRESHOLDS = RelevanceThresholds(
    RELEVANCE_TIER_HIGH_MAX, RELEVANCE_TIER_MEDIUM_MAX
)

#: Calibrated per-model thresholds (``experiments/calibrate_relevance_tiers.py``).
#: Each places "high" / "medium" at the same relative position between the
#: model's typical good match and its typical off-domain match as BGE's
#: constants do. Multilingual models use the Spanish calibration (the lower,
#: stricter of the two measured; English values are in the write-up).
#: BGE and its fine-tunes are not listed: they use the defaults.
MODEL_RELEVANCE_THRESHOLDS: dict[str, RelevanceThresholds] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": RelevanceThresholds(
        0.6786, 0.7216
    ),
    "jinaai/jina-embeddings-v5-text-nano-retrieval": RelevanceThresholds(0.8031, 0.8572),
    "embeddinggemma-300m": RelevanceThresholds(0.8037, 0.8491),
}


def relevance_thresholds(
    model: str | None = None,
    *,
    high_max: float | None = None,
    medium_max: float | None = None,
) -> RelevanceThresholds:
    """Thresholds for ``model``, with optional explicit overrides.

    Unknown models (and ``None``) get :data:`DEFAULT_RELEVANCE_THRESHOLDS`.
    ``high_max`` / ``medium_max`` replace the corresponding bound.

    Raises:
        ValueError: The resulting bounds are not ``0 < high_max <= medium_max <= 2``.
    """
    base = MODEL_RELEVANCE_THRESHOLDS.get(model or "", DEFAULT_RELEVANCE_THRESHOLDS)
    out = RelevanceThresholds(
        base.high_max if high_max is None else float(high_max),
        base.medium_max if medium_max is None else float(medium_max),
    )
    if not (0.0 < out.high_max <= out.medium_max <= 2.0):
        raise ValueError(
            f"relevance thresholds must satisfy 0 < high_max <= medium_max <= 2, got {out} "
            f"(model {model!r}); when lowering one bound below the model default "
            "of the other, set both in [relevance]"
        )
    return out


def relevance_tier(distance: float, thresholds: RelevanceThresholds | None = None) -> str:
    """Classify a best-match cosine distance into ``"high"`` / ``"medium"`` / ``"low"``.

    ``thresholds`` defaults to the BGE-small constants; pass
    :func:`relevance_thresholds` (or ``VstashConfig.relevance_thresholds()``)
    for the active model.

    Tiers:
        "high"   -- ``distance <= high_max``: on-topic match.
        "medium" -- ``high_max < distance <= medium_max``: uncertain.
        "low"    -- ``distance > medium_max``: likely off-topic for this store.
    """
    t = thresholds or DEFAULT_RELEVANCE_THRESHOLDS
    if distance <= t.high_max:
        return "high"
    if distance <= t.medium_max:
        return "medium"
    return "low"


def thresholds_for(cfg: object) -> RelevanceThresholds | None:
    """``cfg.relevance_thresholds()`` when ``cfg`` is a real config, else ``None``.

    Adapters call this so a partial or mocked config falls back to the
    defaults instead of failing the search.
    """
    getter = getattr(cfg, "relevance_thresholds", None)
    if getter is None:
        return None
    try:
        t = getter()
    except Exception:
        return None
    return t if isinstance(t, RelevanceThresholds) else None
