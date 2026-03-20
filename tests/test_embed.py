"""Tests for vstash.embed — FastEmbed wrapper."""

from __future__ import annotations

from vstash.embed import KNOWN_DIMS, get_embedding_dim


class TestGetEmbeddingDim:
    """Test embedding dimension lookup."""

    def test_known_model_bge_small(self) -> None:
        assert get_embedding_dim("BAAI/bge-small-en-v1.5") == 384

    def test_known_model_bge_base(self) -> None:
        assert get_embedding_dim("BAAI/bge-base-en-v1.5") == 768

    def test_known_model_bge_large(self) -> None:
        assert get_embedding_dim("BAAI/bge-large-en-v1.5") == 1024

    def test_known_model_nomic(self) -> None:
        assert get_embedding_dim("nomic-ai/nomic-embed-text-v1.5") == 768

    def test_unknown_model_defaults_to_384(self) -> None:
        assert get_embedding_dim("unknown/model-name") == 384

    def test_known_dims_dict_is_populated(self) -> None:
        assert len(KNOWN_DIMS) >= 4
