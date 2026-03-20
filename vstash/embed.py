"""
embed.py — FastEmbed ONNX wrapper.

Always local. No server. No HTTP.
ONNX Runtime runs in-process at ~700 chunks/s on CPU.

Models:
  BAAI/bge-small-en-v1.5         → 384 dims, fastest, great quality/speed ratio
  nomic-ai/nomic-embed-text-v1.5 → 768 dims, higher quality, slower
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding


# Lazy init — don't load the model until first use
_model_cache: dict[str, TextEmbedding] = {}

KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimensionality for a known model.

    Args:
        model_name: FastEmbed model identifier.

    Returns:
        Embedding dimension (defaults to 384 for unknown models).
    """
    return KNOWN_DIMS.get(model_name, 384)


def _get_model(model_name: str) -> TextEmbedding:
    """Get or lazily initialize a FastEmbed model.

    Args:
        model_name: FastEmbed model identifier.

    Returns:
        Cached TextEmbedding instance.
    """
    if model_name not in _model_cache:
        from fastembed import TextEmbedding
        _model_cache[model_name] = TextEmbedding(model_name=model_name)
    return _model_cache[model_name]


def warmup(model_name: str) -> None:
    """Pre-load the embedding model to eliminate cold start latency.

    Runs a single dummy embedding so the ONNX runtime is fully
    initialised before the first real query.

    Args:
        model_name: FastEmbed model identifier.
    """
    model = _get_model(model_name)
    # Trigger ONNX session initialisation with a throwaway input
    list(model.embed(["warmup"]))


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a batch of texts.

    Args:
        texts: List of text strings to embed.
        model_name: FastEmbed model identifier.

    Returns:
        List of float vectors, one per input text.
    """
    model = _get_model(model_name)
    return [emb.tolist() for emb in model.embed(texts)]


def embed_query(text: str, model_name: str) -> list[float]:
    """Embed a single query string. Optimized path for search.

    Uses query-specific instruction prefix where supported by the model.

    Args:
        text: Query string to embed.
        model_name: FastEmbed model identifier.

    Returns:
        Float vector for the query.
    """
    model = _get_model(model_name)
    # query_embed uses query-specific instruction prefix where supported
    if hasattr(model, "query_embed"):
        return next(model.query_embed([text])).tolist()
    return next(model.embed([text])).tolist()
