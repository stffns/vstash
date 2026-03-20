"""
embed.py — Embedding backend with ONNX (FastEmbed) and MLX support.

Supports two backends:
  - **onnx** (default): FastEmbed ONNX Runtime, portable across all platforms
  - **mlx**: Apple MLX framework, ~25% faster on Apple Silicon GPUs

Auto mode detects Apple Silicon and picks MLX when available.

Models:
  BAAI/bge-small-en-v1.5         → 384 dims, fastest, great quality/speed ratio
  nomic-ai/nomic-embed-text-v1.5 → 768 dims, higher quality, slower
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fastembed import TextEmbedding


# ------------------------------------------------------------------ #
# Model registry                                                       #
# ------------------------------------------------------------------ #

KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}

# Mapping from standard model names to MLX community variants
_MLX_MODEL_MAP: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "mlx-community/bge-small-en-v1.5-bf16",
    "BAAI/bge-base-en-v1.5": "mlx-community/bge-base-en-v1.5-bf16",
    "BAAI/bge-large-en-v1.5": "mlx-community/bge-large-en-v1.5-bf16",
}

BackendType = Literal["onnx", "mlx", "auto"]


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimensionality for a known model.

    Args:
        model_name: FastEmbed model identifier.

    Returns:
        Embedding dimension (defaults to 384 for unknown models).
    """
    return KNOWN_DIMS.get(model_name, 384)


def _is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (arm64 macOS)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    """Check if MLX framework is installed."""
    try:
        import mlx  # noqa: F401
        import mlx_embeddings  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend(backend: BackendType) -> Literal["onnx", "mlx"]:
    """Resolve 'auto' to a concrete backend.

    Args:
        backend: Requested backend — 'onnx', 'mlx', or 'auto'.

    Returns:
        Concrete backend name.
    """
    if backend == "auto":
        if _is_apple_silicon() and _mlx_available():
            return "mlx"
        return "onnx"
    return backend


# ------------------------------------------------------------------ #
# ONNX backend (FastEmbed)                                             #
# ------------------------------------------------------------------ #

_onnx_cache: dict[str, TextEmbedding] = {}


def _get_onnx_model(model_name: str) -> TextEmbedding:
    """Get or lazily initialize a FastEmbed ONNX model.

    Args:
        model_name: FastEmbed model identifier.

    Returns:
        Cached TextEmbedding instance.
    """
    if model_name not in _onnx_cache:
        from fastembed import TextEmbedding
        _onnx_cache[model_name] = TextEmbedding(model_name=model_name)
    return _onnx_cache[model_name]


def _embed_onnx(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts using ONNX backend."""
    model = _get_onnx_model(model_name)
    return [emb.tolist() for emb in model.embed(texts)]


def _query_onnx(text: str, model_name: str) -> list[float]:
    """Embed a single query using ONNX backend."""
    model = _get_onnx_model(model_name)
    if hasattr(model, "query_embed"):
        return next(model.query_embed([text])).tolist()
    return next(model.embed([text])).tolist()


def _warmup_onnx(model_name: str) -> None:
    """Pre-load ONNX model."""
    model = _get_onnx_model(model_name)
    list(model.embed(["warmup"]))


# ------------------------------------------------------------------ #
# MLX backend (Apple Silicon GPU)                                      #
# ------------------------------------------------------------------ #

_mlx_cache: dict[str, tuple] = {}  # (model, tokenizer)


def _get_mlx_model(model_name: str) -> tuple:
    """Get or lazily initialize an MLX model.

    Args:
        model_name: Standard model identifier (mapped to MLX variant).

    Returns:
        Tuple of (model, tokenizer).
    """
    if model_name not in _mlx_cache:
        from mlx_embeddings.utils import load
        mlx_name = _MLX_MODEL_MAP.get(model_name, model_name)
        _mlx_cache[model_name] = load(mlx_name)
    return _mlx_cache[model_name]


def _embed_mlx(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts using MLX backend (Apple Silicon GPU)."""
    import mlx.core as mx
    import numpy as np

    model, tokenizer = _get_mlx_model(model_name)
    tok = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
    encoded = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])
    token_type_ids = mx.array(
        encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"]))
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    # Mean pooling over token embeddings
    last_hidden = output.last_hidden_state
    mask_expanded = mx.expand_dims(attention_mask, axis=-1)
    sum_embeddings = mx.sum(last_hidden * mask_expanded, axis=1)
    sum_mask = mx.sum(mask_expanded, axis=1)
    embeddings = sum_embeddings / mx.maximum(sum_mask, mx.array(1e-9))

    # L2 normalize
    norms = mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True))
    embeddings = embeddings / mx.maximum(norms, mx.array(1e-9))
    mx.eval(embeddings)

    return [row.tolist() for row in np.array(embeddings)]


def _query_mlx(text: str, model_name: str) -> list[float]:
    """Embed a single query using MLX backend."""
    return _embed_mlx([text], model_name)[0]


def _warmup_mlx(model_name: str) -> None:
    """Pre-load MLX model and run a dummy embed."""
    _embed_mlx(["warmup"], model_name)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

# Active backend — set once by warmup() or first call
_active_backend: Literal["onnx", "mlx"] | None = None


def _resolve(backend: BackendType = "auto") -> Literal["onnx", "mlx"]:
    """Resolve and cache the active backend."""
    global _active_backend  # noqa: PLW0603
    if _active_backend is None:
        _active_backend = resolve_backend(backend)
    return _active_backend


def warmup(model_name: str, backend: BackendType = "auto") -> None:
    """Pre-load the embedding model to eliminate cold start latency.

    Args:
        model_name: Model identifier.
        backend: Backend to use — 'onnx', 'mlx', or 'auto'.
    """
    resolved = _resolve(backend)
    if resolved == "mlx":
        _warmup_mlx(model_name)
    else:
        _warmup_onnx(model_name)


def embed_texts(
    texts: list[str],
    model_name: str,
    backend: BackendType = "auto",
) -> list[list[float]]:
    """Embed a batch of texts.

    Args:
        texts: List of text strings to embed.
        model_name: Model identifier.
        backend: Backend to use — 'onnx', 'mlx', or 'auto'.

    Returns:
        List of float vectors, one per input text.
    """
    resolved = _resolve(backend)
    if resolved == "mlx":
        return _embed_mlx(texts, model_name)
    return _embed_onnx(texts, model_name)


def embed_query(
    text: str,
    model_name: str,
    backend: BackendType = "auto",
) -> list[float]:
    """Embed a single query string. Optimized path for search.

    Args:
        text: Query string to embed.
        model_name: Model identifier.
        backend: Backend to use — 'onnx', 'mlx', or 'auto'.

    Returns:
        Float vector for the query.
    """
    resolved = _resolve(backend)
    if resolved == "mlx":
        return _query_mlx(text, model_name)
    return _query_onnx(text, model_name)


def get_active_backend() -> str:
    """Return the name of the currently active embedding backend.

    Returns:
        'onnx', 'mlx', or 'not initialized'.
    """
    return _active_backend or "not initialized"
