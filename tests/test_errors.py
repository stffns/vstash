"""Tests for the unified vstash exception hierarchy (vstash.errors).

The hierarchy was introduced in v0.37 and centralises errors that
were previously defined in vstash.validation (LimitError tree) and
vstash.store (SchemaVersionError) under a common VstashError ancestor.
These tests pin the contract:

- VstashError is an Exception (not a ValueError).
- LimitError multi-inherits from VstashError + ValueError.
- SchemaVersionError multi-inherits from VstashError + RuntimeError.
- BackendError is a direct VstashError subclass.
- Backwards-compat re-exports keep working: vstash.validation.LimitError
  is the same class as vstash.errors.LimitError, and vstash.store
  re-exports SchemaVersionError identically.
"""

from __future__ import annotations

import pytest

from vstash import (
    BackendError,
    LimitError,
    SchemaVersionError,
    VstashError,
)
from vstash.errors import (
    ChunkTooLargeError,
    EmbeddingMismatchError,
    PathTooLongError,
    QueryInvalidError,
    QueryTooLongError,
    TopKOutOfRangeError,
)


class TestAncestry:
    """The class hierarchy is the user-visible contract."""

    def test_vstash_error_is_exception(self) -> None:
        assert issubclass(VstashError, Exception)
        # Must NOT be a ValueError or RuntimeError so consumers can
        # distinguish "vstash failure" from "input bad" / "runtime bad".
        assert not issubclass(VstashError, ValueError)
        assert not issubclass(VstashError, RuntimeError)

    def test_limit_error_multi_inherits(self) -> None:
        assert issubclass(LimitError, VstashError)
        assert issubclass(LimitError, ValueError)

    def test_schema_version_error_multi_inherits(self) -> None:
        assert issubclass(SchemaVersionError, VstashError)
        assert issubclass(SchemaVersionError, RuntimeError)

    def test_backend_error_is_vstash_error_only(self) -> None:
        assert issubclass(BackendError, VstashError)
        # No stdlib parallel ancestor; backend failures are vstash-only.
        assert not issubclass(BackendError, ValueError)
        assert not issubclass(BackendError, RuntimeError)

    @pytest.mark.parametrize(
        "subclass",
        [
            QueryInvalidError,
            QueryTooLongError,
            TopKOutOfRangeError,
            PathTooLongError,
            ChunkTooLargeError,
            EmbeddingMismatchError,
        ],
    )
    def test_validation_subclasses_inherit_chain(self, subclass: type) -> None:
        # Every LimitError subclass must reach VstashError AND ValueError
        # transitively. This is what makes ``except VstashError`` and
        # ``except ValueError`` both correct catch clauses for any
        # validation failure.
        assert issubclass(subclass, LimitError)
        assert issubclass(subclass, VstashError)
        assert issubclass(subclass, ValueError)


class TestBackwardsCompat:
    """Existing imports must keep resolving to the same class objects."""

    def test_validation_reexports_limit_error(self) -> None:
        from vstash.validation import LimitError as VLE

        assert VLE is LimitError

    def test_validation_reexports_subclass(self) -> None:
        from vstash.validation import QueryTooLongError as VQTL

        assert VQTL is QueryTooLongError

    def test_store_reexports_schema_version_error(self) -> None:
        from vstash.store import SchemaVersionError as SVE

        assert SVE is SchemaVersionError

    def test_value_error_catches_limit_error(self) -> None:
        # Pre-v0.37 user code does ``except ValueError`` around store
        # ops. That must keep working.
        with pytest.raises(ValueError):
            raise QueryTooLongError("test")

    def test_runtime_error_catches_schema_version_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise SchemaVersionError("test")


class TestVstashErrorCatchAll:
    """The headline value of v0.37: one ``except`` for any vstash failure."""

    def test_catches_limit_error(self) -> None:
        with pytest.raises(VstashError):
            raise QueryTooLongError("query too long")

    def test_catches_schema_version_error(self) -> None:
        with pytest.raises(VstashError):
            raise SchemaVersionError("unknown schema version")

    def test_catches_backend_error(self) -> None:
        with pytest.raises(VstashError):
            raise BackendError("snapvec fit failed")

    def test_does_not_catch_unrelated_exceptions(self) -> None:
        # ``except VstashError`` must not swallow stdlib exceptions
        # raised from third-party code (e.g. sqlite3.OperationalError,
        # FileNotFoundError, KeyError) -- those should bubble up.
        with pytest.raises(KeyError):
            try:
                raise KeyError("missing")
            except VstashError:
                pytest.fail("VstashError caught a KeyError -- ancestry leak")
