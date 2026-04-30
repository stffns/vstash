"""
validation.py — Explicit limits and fail-safe input validation (#133).

A good substrate is honest about its boundaries.  This module rejects
inputs that would otherwise crash inside SQLite, sqlite-vec, or the
embedding model with cryptic errors — a 50_000-token query, a chunk
the size of a JPEG, a top_k that would pull every chunk into Python.

Validation lives at the **public API boundary** (``VstashStore`` and
``Memory``) and never inside the hot path.  Each validator is a few
comparisons; if your inputs are reasonable, the cost is unmeasurable.

All validators raise a subclass of :class:`LimitError`, which itself
inherits from :class:`vstash.errors.VstashError` and
:class:`ValueError`.  Catching either continues to work; the
``VstashError`` ancestor (added in v0.37) is the unified handle for
any vstash-domain error.

Limits are configured via the ``[limits]`` section in ``vstash.toml``
(see :class:`vstash.config.LimitsConfig`).  Defaults are intentionally
generous -- they catch pathological inputs without inconveniencing
realistic ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export the validation error hierarchy from `vstash.errors` so
# existing `from vstash.validation import LimitError, QueryTooLongError`
# imports keep working. The single source of truth is `errors.py`.
from .errors import (
    ChunkTooLargeError,
    DistanceCutoffOutOfRangeError,
    EmbeddingMismatchError,
    EmptyDocumentError,
    InvalidIdentifierError,
    LimitError,
    PathTooLongError,
    QueryInvalidError,
    QueryTooLongError,
    RecencyBoostOutOfRangeError,
    RRFWeightOutOfRangeError,
    TooManyChunksError,
    TopKOutOfRangeError,
)

if TYPE_CHECKING:
    from .config import LimitsConfig

__all__ = [
    "ChunkTooLargeError",
    "DistanceCutoffOutOfRangeError",
    "EmbeddingMismatchError",
    "EmptyDocumentError",
    "InvalidIdentifierError",
    "LimitError",
    "PathTooLongError",
    "QueryInvalidError",
    "QueryTooLongError",
    "RRFWeightOutOfRangeError",
    "RecencyBoostOutOfRangeError",
    "TooManyChunksError",
    "TopKOutOfRangeError",
    "validate_document_input",
    "validate_identifier",
    "validate_search_input",
]


# ------------------------------------------------------------------ #
# Validators                                                           #
# ------------------------------------------------------------------ #


def validate_identifier(
    value: str | None,
    *,
    field: str,
    max_length: int = 256,
) -> None:
    """Validate a project / collection / layer identifier.

    Accepts ``None`` (means "no filter") and any non-empty string up to
    ``max_length`` characters that contains no ASCII control characters.
    Whitespace-only strings are rejected because they almost always
    indicate a bug in the caller.
    """
    if value is None:
        return
    if not isinstance(value, str):
        msg = f"{field} must be a string, got {type(value).__name__}"
        raise InvalidIdentifierError(msg)
    if len(value) > max_length:
        msg = f"{field} length {len(value)} exceeds max {max_length}"
        raise InvalidIdentifierError(msg)
    if not value.strip():
        msg = f"{field} must not be empty or whitespace-only"
        raise InvalidIdentifierError(msg)
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        msg = f"{field} contains control characters (e.g. NUL, tab, newline)"
        raise InvalidIdentifierError(msg)


def validate_search_input(
    *,
    query_text: str,
    top_k: int,
    distance_cutoff: float,
    recency_boost: float,
    limits: LimitsConfig,
    vec_weight: float | None = None,
    fts_weight: float | None = None,
) -> None:
    """Validate the public arguments to :meth:`VstashStore.search`.

    Raises a :class:`LimitError` subclass if any argument is out of
    range.  Empty queries are *allowed* (caller may have a legitimate
    reason — e.g. embedding-only retrieval) but ``None`` and non-string
    values are rejected.

    ``vec_weight`` / ``fts_weight`` may be ``None`` (adaptive RRF, the
    default) or a float in ``[0.0, 1.0]``.  Out-of-range floats raise
    :class:`RRFWeightOutOfRangeError` — this is the first PR that exposes
    these knobs to end users via ``Memory.search`` (#151), so the
    boundary check lives here rather than silently clamping downstream.
    """
    if not isinstance(query_text, str):
        msg = f"query must be a string, got {type(query_text).__name__}"
        raise QueryInvalidError(msg)
    if len(query_text) > limits.max_query_chars:
        msg = (
            f"query length {len(query_text)} exceeds limit "
            f"{limits.max_query_chars} (set [limits] max_query_chars to override)"
        )
        raise QueryTooLongError(msg)

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        msg = f"top_k must be an int, got {type(top_k).__name__}"
        raise TopKOutOfRangeError(msg)
    if top_k < 1:
        msg = f"top_k must be >= 1, got {top_k}"
        raise TopKOutOfRangeError(msg)
    if top_k > limits.max_top_k:
        msg = f"top_k {top_k} exceeds limit {limits.max_top_k} (set [limits] max_top_k to override)"
        raise TopKOutOfRangeError(msg)

    if distance_cutoff < 0 or distance_cutoff > limits.max_distance_cutoff:
        msg = f"distance_cutoff {distance_cutoff} out of range [0, {limits.max_distance_cutoff}]"
        raise DistanceCutoffOutOfRangeError(msg)

    if recency_boost < 0 or recency_boost > limits.max_recency_boost:
        msg = f"recency_boost {recency_boost} out of range [0, {limits.max_recency_boost}]"
        raise RecencyBoostOutOfRangeError(msg)

    for name, value in (("vec_weight", vec_weight), ("fts_weight", fts_weight)):
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            msg = f"{name} must be a float in [0.0, 1.0] or None, got {type(value).__name__}"
            raise RRFWeightOutOfRangeError(msg)
        if value < 0.0 or value > 1.0:
            msg = f"{name} {value} out of range [0.0, 1.0]"
            raise RRFWeightOutOfRangeError(msg)


def validate_document_input(
    *,
    path: str,
    chunks: list[str],
    embeddings: list[list[float]],
    limits: LimitsConfig,
) -> None:
    """Validate the inputs to :meth:`VstashStore.add_document`.

    Catches the four most common pathological ingestion cases:
      - path longer than the OS file system supports
      - document with zero chunks (always a caller bug)
      - document with absurdly many chunks (would lock the DB)
      - a single chunk larger than ``max_chunk_chars`` (would crash the
        embedding model and/or SQLite)

    Also asserts that ``len(chunks) == len(embeddings)`` — a mismatch
    is silently corruption-prone if it slips through.
    """
    if not isinstance(path, str):
        msg = f"path must be a string, got {type(path).__name__}"
        raise PathTooLongError(msg)
    if len(path) > limits.max_path_chars:
        msg = (
            f"path length {len(path)} exceeds limit {limits.max_path_chars} "
            f"(POSIX PATH_MAX is typically 4096)"
        )
        raise PathTooLongError(msg)

    n_chunks = len(chunks)
    if n_chunks == 0:
        raise EmptyDocumentError("document has zero chunks")
    if n_chunks > limits.max_chunks_per_document:
        msg = f"document has {n_chunks} chunks, exceeds limit {limits.max_chunks_per_document}"
        raise TooManyChunksError(msg)
    if len(embeddings) != n_chunks:
        msg = f"chunks ({n_chunks}) and embeddings ({len(embeddings)}) length mismatch"
        raise EmbeddingMismatchError(msg)

    cap = limits.max_chunk_chars
    for i, c in enumerate(chunks):
        if not isinstance(c, str):
            msg = f"chunks[{i}] must be str, got {type(c).__name__}"
            raise ChunkTooLargeError(msg)
        if len(c) > cap:
            msg = f"chunks[{i}] length {len(c)} exceeds limit {cap}"
            raise ChunkTooLargeError(msg)
