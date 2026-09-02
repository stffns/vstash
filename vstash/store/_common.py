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

# Probe for snapvec availability (optional dependency).
# snapvec >= 0.6.0 ships delete O(1) via swap-with-last upstream — no
# monkey-patch needed. The pin in pyproject.toml enforces the floor.
# Kept here (the package leaf) so both the facade and _index can import
# SnapIndex / _HAS_SNAPVEC without an import cycle. SnapIndex is bound to
# None when snapvec is absent; every use site is guarded by _HAS_SNAPVEC.
try:
    from snapvec import SnapIndex

    _HAS_SNAPVEC = True
except ImportError:  # pragma: no cover - exercised only when snapvec is absent
    SnapIndex = None  # type: ignore[assignment, misc]
    _HAS_SNAPVEC = False

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


# Float32 tolerance for the near-duplicate collapse comparison. The dot
# product of a float32 unit vector with an identical copy frequently
# lands on 0.99999994 rather than 1.0, so a bare ``>= 1.0`` would miss
# about half of the byte-identical pairs a threshold of 1.0 promises.
_DEDUP_SIM_EPS = 1e-6


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


def _canonicalize_added_filter(value: str, *, label: str) -> str:
    """Validate + UTC-canonicalise an ISO date/timestamp ``added_at`` filter.

    ``added_at`` is stored as a UTC ISO string (``...+00:00``, from
    ``datetime.now(timezone.utc).isoformat()``) and SQLite compares the column
    **lexically**. A non-UTC offset (e.g. ``2026-01-15T00:00:00-05:00``) sorts
    lexically out of chronological order versus the stored ``+00:00`` form, so a
    raw `added_after` / `added_before` would return the wrong rows around
    timezone boundaries (#384). Naive inputs are treated as UTC; aware inputs are
    converted with ``astimezone``. Non-ISO input is rejected with a clear
    ``ValueError`` rather than passed through to a lexical comparison that would
    silently match the wrong rows. Mirrors the prune/compact ``before=`` handling.
    """
    from datetime import datetime, timezone

    try:
        stripped = value.strip()
        # Python 3.10's datetime.fromisoformat rejects the ``Z`` / ``z`` UTC
        # suffix (only 3.11+ accepts it); normalise it so the project's 3.10
        # floor parses the common ``...Z`` form. Mirrors journal.py's handling.
        if stripped.endswith(("Z", "z")):
            stripped = stripped[:-1] + "+00:00"
        parsed = datetime.fromisoformat(stripped)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"{label}={value!r} is not a parseable ISO date / timestamp "
            f"('2026-01-15' or '2026-01-15T00:00:00+00:00')."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


#: Document-metadata fields a filter leaf may constrain. Kept in one place so
#: the flat-kwarg path and the boolean ``filters`` tree accept the same set.
_FILTER_FIELDS = frozenset(
    {"collection", "project", "layer", "tags", "added_after", "added_before"}
)


def _field_condition(field: str, value: object, prefix: str) -> tuple[str | None, list[str]]:
    """Build the SQL condition + bind params for a single ``field == value`` leaf.

    Shared by the flat-kwarg filters and the boolean ``filters`` tree so tag
    anchoring and date canonicalisation behave identically on both paths.
    A ``None`` value compiles to an ``IS NULL`` / "unset" match (so a tree can
    ask for documents with no project / layer / tags); ``None`` is rejected for
    the date fields, which need a value. Returns ``(None, [])`` for an empty
    /no-op leaf (e.g. ``tags`` that normalises to nothing). Raises ``ValueError``
    on an unknown field.
    """
    if field not in _FILTER_FIELDS:
        raise ValueError(
            f"unknown filter field {field!r}; expected one of {sorted(_FILTER_FIELDS)}"
        )
    if field in ("collection", "project", "layer"):
        if value is None:
            return f"{prefix}{field} IS NULL", []
        return f"{prefix}{field} = ?", [str(value)]
    if field == "tags":
        if value is None:
            # "untagged": the column is either NULL or the empty string.
            return f"({prefix}tags IS NULL OR {prefix}tags = '')", []
        tag_list = _normalize_tags(value if isinstance(value, (str, list)) else str(value))
        if not tag_list:
            return None, []
        wrapped = f"','||{prefix}tags||','"
        ors = " OR ".join(f"{wrapped} LIKE ?" for _ in tag_list)
        return f"({ors})", [f"%,{t},%" for t in tag_list]
    # Date fields: a None bound is meaningless (use the absence of the key to
    # mean "no bound"); reject it loudly rather than comparing against 'None'.
    if value is None:
        raise ValueError(f"{field} filter requires a date value, not None")
    if field == "added_after":
        return f"{prefix}added_at >= ?", [
            _canonicalize_added_filter(str(value), label="added_after")
        ]
    # added_before
    return f"{prefix}added_at < ?", [_canonicalize_added_filter(str(value), label="added_before")]


def _compile_filter_tree(node: object, prefix: str) -> tuple[str | None, list[str]]:
    """Compile a boolean filter expression into parameterised SQL (#106).

    The expression is a nested ``dict``:

      * **Operator node** — exactly one of ``{"and": [...]}``, ``{"or": [...]}``
        (non-empty list of child nodes), or ``{"not": <node>}``.
      * **Leaf node** — a ``{field: value}`` dict over the metadata fields in
        :data:`_FILTER_FIELDS`; multiple fields in one leaf are AND-ed.

    Values are always bound as SQL parameters (never interpolated), so the field
    name — validated against :data:`_FILTER_FIELDS` — is the only thing that ever
    reaches the query text. Returns ``(None, [])`` for a node that contributes no
    constraint. Raises ``ValueError`` on a malformed tree.
    """
    if not isinstance(node, dict):
        raise ValueError(f"filter node must be a dict, got {type(node).__name__}")
    operators = {"and", "or", "not"} & node.keys()
    if operators:
        if len(node) != 1:
            raise ValueError("a filter operator node must hold exactly one of 'and' / 'or' / 'not'")
        op = next(iter(operators))
        if op == "not":
            sub_sql, sub_params = _compile_filter_tree(node["not"], prefix)
            if sub_sql is None:
                return None, []
            return f"(NOT {sub_sql})", sub_params
        children = node[op]
        if not isinstance(children, list) or not children:
            raise ValueError(f"'{op}' must be a non-empty list of filter nodes")
        parts: list[str] = []
        params: list[str] = []
        for child in children:
            sql, child_params = _compile_filter_tree(child, prefix)
            if sql is not None:
                parts.append(sql)
                params.extend(child_params)
        if not parts:
            return None, []
        joiner = " AND " if op == "and" else " OR "
        return "(" + joiner.join(parts) + ")", params
    # Leaf: one or more field conditions, AND-ed together.
    leaf_parts: list[str] = []
    leaf_params: list[str] = []
    for field, value in node.items():
        sql, field_params = _field_condition(field, value, prefix)
        if sql is not None:
            leaf_parts.append(sql)
            leaf_params.extend(field_params)
    if not leaf_parts:
        return None, []
    if len(leaf_parts) == 1:
        return leaf_parts[0], leaf_params
    return "(" + " AND ".join(leaf_parts) + ")", leaf_params
