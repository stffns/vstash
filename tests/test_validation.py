"""Tests for explicit limits and fail-safe input validation (#133)."""

from __future__ import annotations

import pytest

from vstash.config import LimitsConfig
from vstash.store import VstashStore
from vstash.validation import (
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
    TooManyChunksError,
    TopKOutOfRangeError,
    validate_document_input,
    validate_identifier,
    validate_search_input,
)


# ------------------------------------------------------------------ #
# Hierarchy                                                            #
# ------------------------------------------------------------------ #


class TestExceptionHierarchy:
    """All limit errors should subclass LimitError and ValueError."""

    @pytest.mark.parametrize(
        "exc",
        [
            QueryInvalidError,
            QueryTooLongError,
            TopKOutOfRangeError,
            DistanceCutoffOutOfRangeError,
            RecencyBoostOutOfRangeError,
            PathTooLongError,
            EmptyDocumentError,
            TooManyChunksError,
            ChunkTooLargeError,
            EmbeddingMismatchError,
            InvalidIdentifierError,
        ],
    )
    def test_subclass_of_limit_and_value_error(self, exc: type[Exception]) -> None:
        assert issubclass(exc, LimitError)
        assert issubclass(exc, ValueError)


# ------------------------------------------------------------------ #
# validate_identifier                                                  #
# ------------------------------------------------------------------ #


class TestValidateIdentifier:
    def test_none_is_allowed(self) -> None:
        validate_identifier(None, field="project")  # no raise

    def test_normal_string_is_allowed(self) -> None:
        validate_identifier("my_project", field="project")
        validate_identifier("work-2026", field="project")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(InvalidIdentifierError, match="empty"):
            validate_identifier("", field="project")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(InvalidIdentifierError, match="empty"):
            validate_identifier("   ", field="project")

    def test_overlong_rejected(self) -> None:
        with pytest.raises(InvalidIdentifierError, match="exceeds max"):
            validate_identifier("x" * 300, field="project")

    def test_control_chars_rejected(self) -> None:
        with pytest.raises(InvalidIdentifierError, match="control characters"):
            validate_identifier("foo\x00bar", field="project")
        with pytest.raises(InvalidIdentifierError, match="control characters"):
            validate_identifier("foo\nbar", field="project")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(InvalidIdentifierError, match="must be a string"):
            validate_identifier(42, field="project")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# validate_search_input                                                #
# ------------------------------------------------------------------ #


class TestValidateSearchInput:
    @pytest.fixture
    def limits(self) -> LimitsConfig:
        return LimitsConfig()

    def test_normal_input_passes(self, limits: LimitsConfig) -> None:
        validate_search_input(
            query_text="hello world",
            top_k=5,
            distance_cutoff=1.15,
            recency_boost=0.0,
            limits=limits,
        )

    def test_empty_query_is_allowed(self, limits: LimitsConfig) -> None:
        # Empty queries are unusual but not nonsensical (vec-only flow).
        validate_search_input(
            query_text="",
            top_k=5,
            distance_cutoff=1.15,
            recency_boost=0.0,
            limits=limits,
        )

    def test_query_too_long(self, limits: LimitsConfig) -> None:
        with pytest.raises(QueryTooLongError, match="exceeds limit"):
            validate_search_input(
                query_text="x" * (limits.max_query_chars + 1),
                top_k=5,
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_query_non_string(self, limits: LimitsConfig) -> None:
        with pytest.raises(QueryInvalidError, match="must be a string"):
            validate_search_input(
                query_text=None,  # type: ignore[arg-type]
                top_k=5,
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_top_k_zero_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(TopKOutOfRangeError, match=">= 1"):
            validate_search_input(
                query_text="x",
                top_k=0,
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_top_k_negative_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(TopKOutOfRangeError, match=">= 1"):
            validate_search_input(
                query_text="x",
                top_k=-3,
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_top_k_above_max_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(TopKOutOfRangeError, match="exceeds limit"):
            validate_search_input(
                query_text="x",
                top_k=limits.max_top_k + 1,
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_top_k_must_be_int(self, limits: LimitsConfig) -> None:
        with pytest.raises(TopKOutOfRangeError, match="must be an int"):
            validate_search_input(
                query_text="x",
                top_k=5.5,  # type: ignore[arg-type]
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_top_k_bool_rejected(self, limits: LimitsConfig) -> None:
        # bool is technically int in Python; reject it explicitly.
        with pytest.raises(TopKOutOfRangeError, match="must be an int"):
            validate_search_input(
                query_text="x",
                top_k=True,  # type: ignore[arg-type]
                distance_cutoff=1.15,
                recency_boost=0.0,
                limits=limits,
            )

    def test_negative_distance_cutoff_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(DistanceCutoffOutOfRangeError):
            validate_search_input(
                query_text="x",
                top_k=5,
                distance_cutoff=-0.1,
                recency_boost=0.0,
                limits=limits,
            )

    def test_huge_distance_cutoff_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(DistanceCutoffOutOfRangeError):
            validate_search_input(
                query_text="x",
                top_k=5,
                distance_cutoff=limits.max_distance_cutoff + 1,
                recency_boost=0.0,
                limits=limits,
            )

    def test_negative_recency_boost_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(RecencyBoostOutOfRangeError):
            validate_search_input(
                query_text="x",
                top_k=5,
                distance_cutoff=1.15,
                recency_boost=-0.5,
                limits=limits,
            )

    def test_huge_recency_boost_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(RecencyBoostOutOfRangeError):
            validate_search_input(
                query_text="x",
                top_k=5,
                distance_cutoff=1.15,
                recency_boost=limits.max_recency_boost + 1,
                limits=limits,
            )


# ------------------------------------------------------------------ #
# validate_document_input                                              #
# ------------------------------------------------------------------ #


class TestValidateDocumentInput:
    @pytest.fixture
    def limits(self) -> LimitsConfig:
        return LimitsConfig()

    def test_normal_doc_passes(self, limits: LimitsConfig) -> None:
        validate_document_input(
            path="/tmp/doc.md",
            chunks=["a", "b"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            limits=limits,
        )

    def test_overlong_path_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(PathTooLongError):
            validate_document_input(
                path="/" + "x" * limits.max_path_chars,
                chunks=["a"],
                embeddings=[[0.1]],
                limits=limits,
            )

    def test_zero_chunks_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(EmptyDocumentError):
            validate_document_input(
                path="/tmp/doc.md",
                chunks=[],
                embeddings=[],
                limits=limits,
            )

    def test_too_many_chunks_rejected(self, limits: LimitsConfig) -> None:
        n = limits.max_chunks_per_document + 1
        with pytest.raises(TooManyChunksError):
            validate_document_input(
                path="/tmp/doc.md",
                chunks=["x"] * n,
                embeddings=[[0.0]] * n,
                limits=limits,
            )

    def test_oversized_chunk_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(ChunkTooLargeError, match="exceeds limit"):
            validate_document_input(
                path="/tmp/doc.md",
                chunks=["x" * (limits.max_chunk_chars + 1)],
                embeddings=[[0.1]],
                limits=limits,
            )

    def test_embedding_mismatch_rejected(self, limits: LimitsConfig) -> None:
        with pytest.raises(EmbeddingMismatchError):
            validate_document_input(
                path="/tmp/doc.md",
                chunks=["a", "b"],
                embeddings=[[0.1]],
                limits=limits,
            )

    def test_chunk_must_be_string(self, limits: LimitsConfig) -> None:
        with pytest.raises(ChunkTooLargeError, match="must be str"):
            validate_document_input(
                path="/tmp/doc.md",
                chunks=[42],  # type: ignore[list-item]
                embeddings=[[0.1]],
                limits=limits,
            )


# ------------------------------------------------------------------ #
# Integration: store-level enforcement                                 #
# ------------------------------------------------------------------ #


class TestStoreLevelEnforcement:
    """Validators must run at the VstashStore public API boundary."""

    def test_search_rejects_overlong_query(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello"],
            embeddings=[[0.5] * dim],
        )
        with pytest.raises(QueryTooLongError):
            sample_store.search(
                query_embedding=[0.5] * dim,
                query_text="x" * 20_001,
                top_k=1,
            )

    def test_search_rejects_zero_top_k(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(TopKOutOfRangeError):
            sample_store.search(
                query_embedding=[0.5] * dim,
                query_text="hello",
                top_k=0,
            )

    def test_search_rejects_huge_top_k(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(TopKOutOfRangeError):
            sample_store.search(
                query_embedding=[0.5] * dim,
                query_text="hello",
                top_k=10_000,
            )

    def test_search_rejects_negative_recency_boost(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(RecencyBoostOutOfRangeError):
            sample_store.search(
                query_embedding=[0.5] * dim,
                query_text="hello",
                top_k=5,
                recency_boost=-0.1,
            )

    def test_add_document_rejects_zero_chunks(self, sample_store: VstashStore) -> None:
        with pytest.raises(EmptyDocumentError):
            sample_store.add_document(
                path="/empty.md",
                title="Empty",
                chunks=[],
                embeddings=[],
            )

    def test_add_document_rejects_oversized_chunk(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(ChunkTooLargeError):
            sample_store.add_document(
                path="/big.md",
                title="Big",
                chunks=["x" * 2_000_000],
                embeddings=[[0.1] * dim],
            )

    def test_add_document_rejects_embedding_mismatch(self, sample_store: VstashStore) -> None:
        dim = sample_store.embedding_dim
        with pytest.raises(EmbeddingMismatchError):
            sample_store.add_document(
                path="/mismatch.md",
                title="Mismatch",
                chunks=["a", "b"],
                embeddings=[[0.1] * dim],
            )

    def test_overridden_limits_take_effect(self, tmp_db_path: str) -> None:
        """A custom LimitsConfig passed to the store should be enforced."""
        tight = LimitsConfig(max_top_k=3)
        store = VstashStore(tmp_db_path, embedding_dim=4, limits=tight)
        try:
            store.add_document(
                path="/a.md",
                title="A",
                chunks=["hello"],
                embeddings=[[0.5] * 4],
            )
            with pytest.raises(TopKOutOfRangeError):
                store.search(
                    query_embedding=[0.5] * 4,
                    query_text="hello",
                    top_k=4,
                )
        finally:
            store.close()


# ------------------------------------------------------------------ #
# Integration: Memory-level enforcement                                #
# ------------------------------------------------------------------ #


class TestMemoryLevelEnforcement:
    """Memory.__init__ should validate identifier strings up front."""

    def test_empty_project_rejected(self, tmp_db_path: str) -> None:
        from vstash.memory import Memory

        with pytest.raises(InvalidIdentifierError):
            Memory(project="   ", db=tmp_db_path)

    def test_control_char_collection_rejected(self, tmp_db_path: str) -> None:
        from vstash.memory import Memory

        with pytest.raises(InvalidIdentifierError):
            Memory(collection="bad\x00name", db=tmp_db_path)

    def test_normal_identifiers_accepted(self, tmp_db_path: str) -> None:
        from vstash.memory import Memory

        mem = Memory(project="research", collection="papers", db=tmp_db_path)
        mem.close()
