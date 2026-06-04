"""Pure helpers, constants, and the search tracer for ``vstash.store``.

Extracted from the monolithic store module as the first step of the #280
split.  This is the *leaf* of the ``vstash/store/`` package: it imports
nothing from the other store submodules, so it can never participate in an
import cycle.  Everything here is either a module-level constant or a
side-effect-free function/class.  ``vstash/store/__init__.py`` re-exports
every public name below so ``from vstash.store import _cosine_sim`` etc.
keep working unchanged.
"""

from __future__ import annotations

import math
import operator
import struct

# SQLite's SQLITE_LIMIT_VARIABLE_NUMBER default is 999; batch IN clauses below this.
_SQLITE_PARAM_BATCH = 900

# ------------------------------------------------------------------ #
# Schema versioning (#135)                                             #
# ------------------------------------------------------------------ #

#: Current schema version.  Bumped only when a change requires a
#: migration the runtime cannot perform automatically (column drop,
#: type change, semantics change).  Pure additive ALTER TABLE migrations
#: stay within the same version because they're handled in
#: ``_migrate_schema``.
#:
#: v2 (#272): ``vec_chunks`` uses ``distance_metric=cosine``.  Prior v1
#: DBs stored identical float bytes under sqlite-vec's default L2
#: metric; on-open migration rebuilds the virtual table (no
#: re-embedding).
SCHEMA_VERSION = "2"

#: Schema versions this build of vstash knows how to read.  Anything
#: not in this set raises :class:`SchemaVersionError` on open.  v1 is
#: accepted because on-open migration promotes it to v2 in-place.
KNOWN_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2"})


# ------------------------------------------------------------------ #
# Miss-analysis tracing (#108)                                         #
# ------------------------------------------------------------------ #


class _PipelineTracer:
    """Caller-owned collector for per-stage verdicts during search().

    Used by miss_analysis() to record how a specific chunk fared at
    each stage of the search pipeline.  The tracer is created by the
    caller, passed into search(), and read back afterwards.  Because
    ownership is local to the caller, concurrent miss_analysis() calls
    on a shared VstashStore cannot stomp on each other.

    When tracking is not needed, search() receives ``None`` instead of
    a tracer instance — every method on the real tracer is short-
    circuited by an early ``if self.target is None: return`` check in
    the caller code, so there is zero hot-path cost.
    """

    __slots__ = ("target", "verdicts")

    def __init__(self, target_chunk_id: int) -> None:
        self.target: int = int(target_chunk_id)
        self.verdicts: list[dict[str, object]] = []

    def record(
        self,
        stage: str,
        passed: bool,
        rank: int | None = None,
        score: float | None = None,
        detail: str = "",
        counterfactual: str | None = None,
    ) -> None:
        """Append a StageVerdict-shaped dict to the caller's buffer."""
        self.verdicts.append(
            {
                "stage": stage,
                "passed": passed,
                "rank": rank,
                "score": score,
                "detail": detail,
                "counterfactual": counterfactual,
            }
        )


def _normalize_tags(tags: str | list[str] | None) -> list[str]:
    """Normalize the ``tags`` filter input to a deduped list of tag strings.

    Accepts:
      - ``None`` / empty string / empty list -> ``[]`` (filter disabled).
      - A comma-separated string (``"alpha, beta"``) -> ``["alpha", "beta"]``.
      - A list of strings (``["alpha", "beta"]``) -> ``["alpha", "beta"]``.

    Whitespace around each tag is stripped, empty entries are dropped, and
    insertion order is preserved (no sort) so callers can keep meaningful
    ordering when they care. Duplicates are removed.
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = tags.split(",")
    else:
        # Each list element may itself contain commas (Typer's
        # ``--tag "a,b"`` pattern, or callers who mix repeated flags
        # with comma-joined strings). Split each element so the
        # caller surface accepts ``["alpha,beta"]`` and
        # ``["alpha", "beta"]`` interchangeably -- otherwise the
        # comma-anchored ``LIKE`` match would look for a literal tag
        # ``"alpha,beta"`` which never exists in storage.
        parts = []
        for t in tags:
            parts.extend(str(t).split(","))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _serialize(vector: list[float]) -> bytes:
    """Serialize a float vector into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize(data: bytes) -> list[float]:
    """Deserialize a sqlite-vec binary blob back to a float list.

    Raises:
        ValueError: If the blob length is not a multiple of float size.
    """
    item_size = struct.calcsize("f")
    if len(data) % item_size != 0:
        msg = f"Embedding blob length {len(data)} is not a multiple of {item_size}"
        raise ValueError(msg)
    count = len(data) // item_size
    return list(struct.unpack(f"{count}f", data))


# Pick the fastest pure-Python dot product available on this
# interpreter.  `math.sumprod` (Python 3.12+) is a single C-level loop
# tuned for dot products and is ~3x faster than `sum(map(operator.mul,
# a, b))` on 384-dim vectors.  For Python 3.10/3.11 (which we still
# support, per pyproject.toml requires-python = ">=3.10"), fall back to
# the map+operator path — still ~3x faster than the original generator
# expression.  The selection happens once at module load; the hot path
# pays zero overhead for the check.
try:
    _dot_product = math.sumprod  # Python 3.12+
except AttributeError:

    def _dot_product(a: list[float], b: list[float]) -> float:
        return sum(map(operator.mul, a, b))


def _cosine_sim(
    a: list[float],
    b: list[float],
    norm_a: float | None = None,
    norm_b: float | None = None,
) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1].

    Args:
        a: First vector.
        b: Second vector.
        norm_a: Precomputed L2 norm of *a* (``math.hypot(*a)``). If None,
            computed on the fly.
        norm_b: Precomputed L2 norm of *b* (``math.hypot(*b)``). If None,
            computed on the fly.

    Uses ``math.sumprod`` on Python 3.12+ and ``sum(map(operator.mul,
    ...))`` as a fallback, combined with ``math.hypot(*vec)`` for the
    L2 norm.  Both branches route through C-level stdlib loops and
    avoid the Python-bytecode overhead of generator expressions.

    Returns 0.0 when either input is an empty vector or a zero
    vector (the existing guard catches these via the ``norm < 1e-9``
    check).
    """
    dot = _dot_product(a, b)
    if norm_a is None:
        norm_a = math.hypot(*a)
    if norm_b is None:
        norm_b = math.hypot(*b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


#: High-confidence cosine distance cutoff for ``relevance_tier``.  Value
#: is the cosine equivalent of the legacy L2-on-unit-vec threshold 0.95
#: (``cos_dist = L2^2 / 2 = 0.4513``), so BGE-small unit-normalized
#: embeddings keep identical tier assignments across the v1 -> v2
#: metric change (#272).
RELEVANCE_TIER_HIGH_MAX = 0.4513

#: Medium-confidence cosine distance cutoff for ``relevance_tier``.
#: Cosine equivalent of the legacy L2 threshold 0.98
#: (``0.98^2 / 2 = 0.4802``).  Anything above is classified "low".
RELEVANCE_TIER_MEDIUM_MAX = 0.4802


def relevance_tier(distance: float) -> str:
    """Classify cosine distance into a relevance tier.

    Thresholds were recalibrated for cosine metric in schema v2 (#272).
    The old labels claimed "cosine distance" while sqlite-vec was
    actually returning L2 distance, which only worked by accident on
    unit-normalized BGE.  See the ``RELEVANCE_TIER_*`` constants above
    for how the new cutoffs were derived.

    Tiers:
        "high"   -- distance <= ``RELEVANCE_TIER_HIGH_MAX`` (0.4513):
            confident match.
        "medium" -- ``RELEVANCE_TIER_HIGH_MAX`` < distance <=
            ``RELEVANCE_TIER_MEDIUM_MAX`` (0.4802): uncertain.
        "low"    -- distance > ``RELEVANCE_TIER_MEDIUM_MAX``: likely
            off-topic.
    """
    if distance <= RELEVANCE_TIER_HIGH_MAX:
        return "high"
    if distance <= RELEVANCE_TIER_MEDIUM_MAX:
        return "medium"
    return "low"


# Standard RRF constant — balances precision vs recall
RRF_K = 60

# Adaptive RRF: query length threshold above which FTS weight is reduced.
# ArguAna (194 avg words) showed -38.4% vs dense; queries >50 words are
# typically semantic paraphrases where keywords add noise.
_ADAPTIVE_RRF_LONG_QUERY = 50

# Long-query distance_cutoff. 25.0 = 5.0^2: the squared cosine equivalent
# of the legacy v1 L2 5.0x cutoff (#272). Diffuse long-query embeddings
# compress distances; without this relaxation the default 1.3225 cutoff
# rejects nearly every candidate past rank 0.
_LONG_QUERY_DISTANCE_CUTOFF = 25.0
