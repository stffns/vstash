"""vstash.errors -- single source of truth for the exception hierarchy.

Every error vstash raises descends from :class:`VstashError`. Catching
``VstashError`` is the right way to handle any vstash-originated
failure without also swallowing unrelated exceptions.

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
    """Base class for all vstash-raised exceptions.

    Catch this to handle any vstash failure mode without also catching
    unrelated stdlib exceptions. Direct subclasses split the failure
    space by category:

    - :class:`LimitError` -- input validation rejected at the API
      boundary (queries too long, top_k out of range, chunks too big).
    - :class:`SchemaVersionError` -- on-disk database has a schema
      version this build does not know how to read.
    - :class:`BackendError` -- a vector backend (sqlite-vec, snapvec,
      ivfpq) failed to fit, save, or query.
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
