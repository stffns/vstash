"""vstash.errors -- single source of truth for the vstash error hierarchy.

This module is the canonical home of vstash's *domain* exception
tree: input validation (:class:`LimitError`), schema mismatch
(:class:`SchemaVersionError`), and vector backend failure
(:class:`BackendError`). Every error in that tree descends from
:class:`VstashError`, so ``except VstashError`` is the catch-all
for vstash domain failures.

Scope caveat: not every exception raised from a vstash function is a
domain error. Some public guard paths still raise plain
:class:`ValueError` / :class:`RuntimeError` (for example
``Memory.miss_analysis`` rejects callers that pass neither
``expected_path`` nor ``expected_chunk_id``), and lower-level
failures bubble up as their stdlib types (``sqlite3.OperationalError``,
``FileNotFoundError``, ``KeyError``). Tracking the migration of
those sites to ``VstashError`` subclasses is a follow-up; until
that lands, ``except VstashError`` covers the *domain hierarchy*
defined here, not literally every error vstash can ever raise.

Backwards compatibility: the validation errors in this module also
inherit from :class:`ValueError`, and :class:`SchemaVersionError`
also inherits from :class:`RuntimeError`. Existing user code that
catches the stdlib base classes continues to work unchanged.

Re-exports: :mod:`vstash.validation` and :mod:`vstash.store` both
re-export their respective error classes from here, so existing
``from vstash.validation import LimitError`` and
``from vstash.store import SchemaVersionError`` imports keep working.
"""

from __future__ import annotations


class VstashError(Exception):
    """Base class for vstash's domain exception tree.

    Catch this to handle any vstash *domain* failure (input
    validation, schema mismatch, backend failure) without also
    catching unrelated stdlib exceptions. Direct subclasses split
    the failure space by category:

    - :class:`LimitError` -- input validation rejected at the API
      boundary (queries too long, top_k out of range, chunks too big).
    - :class:`SchemaVersionError` -- on-disk database has a schema
      version this build does not know how to read.
    - :class:`BackendError` -- a vector backend (sqlite-vec, snapvec,
      ivfpq) failed to fit, save, or query.

    Some legacy guard paths still raise plain :class:`ValueError` or
    :class:`RuntimeError`; migrating those to ``VstashError``
    subclasses is a follow-up. ``except VstashError`` covers the
    domain tree above, not every exception vstash can possibly raise.
    """


# ------------------------------------------------------------------ #
# Input validation errors (formerly defined in validation.py)         #
# ------------------------------------------------------------------ #


class LimitError(VstashError, ValueError):
    """Base class for all vstash input validation errors.

    Multi-inherits from :class:`VstashError` (so ``except VstashError``
    catches it) and :class:`ValueError` (so existing
    ``except ValueError`` handlers continue to work). Catch this
    directly to handle any vstash-imposed limit; catch a specific
    subclass to react to one category.
    """


class QueryInvalidError(LimitError):
    """Query text is missing, the wrong type, or otherwise unusable."""


class QueryTooLongError(LimitError):
    """Query text exceeds the configured maximum length."""


class TopKOutOfRangeError(LimitError):
    """``top_k`` is < 1 or above the configured maximum."""


class DistanceCutoffOutOfRangeError(LimitError):
    """``distance_cutoff`` is negative or above the configured maximum."""


class RecencyBoostOutOfRangeError(LimitError):
    """``recency_boost`` is negative or above the configured maximum."""


class DedupThresholdOutOfRangeError(LimitError):
    """``dedup_threshold`` outside ``(0.0, 1.0]``."""


class RRFWeightOutOfRangeError(LimitError):
    """``vec_weight`` or ``fts_weight`` is outside the ``[0.0, 1.0]`` range."""


class PathTooLongError(LimitError):
    """Document path exceeds the configured maximum length."""


class EmptyDocumentError(LimitError):
    """Document has zero chunks."""


class TooManyChunksError(LimitError):
    """Document has more chunks than the configured maximum."""


class ChunkTooLargeError(LimitError):
    """A single chunk exceeds the configured character limit."""


class EmbeddingMismatchError(LimitError):
    """``chunks`` and ``embeddings`` lists have different lengths."""


class InvalidIdentifierError(LimitError):
    """``project`` / ``collection`` identifier is empty, too long, or contains
    control characters."""


# ------------------------------------------------------------------ #
# Schema / storage errors                                              #
# ------------------------------------------------------------------ #


class SchemaVersionError(VstashError, RuntimeError):
    """Raised when an existing DB declares a schema version this build
    of vstash does not recognize.

    Multi-inherits from :class:`RuntimeError` for backwards
    compatibility with callers that ``except RuntimeError`` around
    store construction. The right remedy is to upgrade vstash, restore
    from backup, or run a future ``vstash migrate`` command (not yet
    implemented).
    """


# ------------------------------------------------------------------ #
# Backend errors (used by vector backends, retrieval pipeline)        #
# ------------------------------------------------------------------ #


class BackendError(VstashError):
    """A vector backend (sqlite-vec, snapvec, snapvec-ivfpq) failed.

    Distinct from :class:`LimitError` (caller-supplied input is bad)
    and :class:`SchemaVersionError` (database is from a different
    vstash version). Use this for backend-internal failures such as a
    snapvec index file failing to load, an IVFPQ fit step receiving
    too few training vectors, or a sqlite-vec extension load error.
    """


__all__ = [
    "BackendError",
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
    "SchemaVersionError",
    "TooManyChunksError",
    "TopKOutOfRangeError",
    "VstashError",
]
