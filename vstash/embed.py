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

import json
import logging
import os
import platform
import threading
import urllib.request
from typing import TYPE_CHECKING, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Custom encoder plug-in hook (#278)                                   #
# ------------------------------------------------------------------ #


@runtime_checkable
class Encoder(Protocol):
    """Shape a plugged-in encoder must satisfy.

    ``embedding_dim`` lets callers of :func:`get_embedding_dim` discover
    the output size without running the model.  ``encode`` returns a
    matrix of shape ``(len(texts), embedding_dim)``; lists of floats or
    numpy arrays both work.

    SentenceTransformer instances expose
    ``get_sentence_embedding_dimension()`` rather than the
    ``embedding_dim`` attribute this protocol expects, so a small
    adapter is needed::

        from sentence_transformers import SentenceTransformer

        class STEncoder:
            def __init__(self, model: SentenceTransformer) -> None:
                self._m = model
                self.embedding_dim = model.get_sentence_embedding_dimension()

            def encode(self, texts: list[str]):
                return self._m.encode(texts, normalize_embeddings=True)

    PEFT-wrapped ST instances, custom pooling heads, and any other
    ``.encode``-speaking object follow the same shape.  See #278 for
    motivation.
    """

    embedding_dim: int

    def encode(self, texts: list[str]):  # -> list[list[float]] | np.ndarray
        ...


#: Global list of custom encoder resolvers, consulted in registration
#: order.  Each resolver receives a ``model_name`` string and returns
#: an :class:`Encoder` if it handles that name, otherwise ``None`` to
#: defer to the built-in backends.  See :func:`register_encoder_resolver`.
_encoder_resolvers: list[Callable[[str], "Encoder | None"]] = []
_encoder_resolvers_lock = threading.Lock()


def register_encoder_resolver(resolver: Callable[[str], "Encoder | None"]) -> None:
    """Register a custom encoder resolver.

    Resolvers are consulted **before** every built-in embedding path
    (daemon delegation, Gemma, HF ONNX, MLX, FastEmbed).  The first
    resolver that returns a non-``None`` encoder wins; its ``encode``
    method is then used for the call.  This lets callers plug in
    LoRA-adapted, locally-fine-tuned, or otherwise-unnamable encoders
    without monkey-patching :func:`embed_texts`.

    Resolvers are process-global.  Callers are responsible for
    registering them before any ingest or search uses the matching
    model name.  Registrations are idempotent by **object identity**
    (``is`` comparison), not value equality: a callable that defines
    ``__eq__`` is still registered distinctly from its peers, which
    matches what resolvers usually are (closures or bound methods).

    Args:
        resolver: Callable taking a ``model_name`` and returning an
            object that satisfies the :class:`Encoder` protocol, or
            ``None`` to defer to the next resolver / built-in.

    See :func:`unregister_encoder_resolver`.
    """
    with _encoder_resolvers_lock:
        if not any(r is resolver for r in _encoder_resolvers):
            _encoder_resolvers.append(resolver)


def unregister_encoder_resolver(resolver: Callable[[str], "Encoder | None"]) -> None:
    """Remove a previously-registered resolver.

    Identity-based match: removes the entry ``r`` where ``r is resolver``.
    No-op if the resolver was never registered.  Primarily useful in
    tests that want process-level hygiene between cases.
    """
    with _encoder_resolvers_lock:
        for i, r in enumerate(_encoder_resolvers):
            if r is resolver:
                del _encoder_resolvers[i]
                return


def _resolve_custom_encoder(model_name: str) -> "Encoder | None":
    """Walk the resolver list and return the first valid encoder that
    claims ``model_name``.

    Never raises -- a resolver that throws is logged and treated as
    ``None`` so one bad resolver does not brick the others.  A resolver
    that returns something that does not satisfy the :class:`Encoder`
    protocol (missing ``encode`` or ``embedding_dim``) is also logged
    and skipped, so a malformed encoder does not surface as a
    hard-to-trace ``AttributeError`` deep in the embedding call.
    """
    # Snapshot under the lock so concurrent (un)register does not mutate
    # the list mid-iteration; but call the resolver callbacks outside
    # the lock to avoid holding it across user code.
    with _encoder_resolvers_lock:
        snapshot = list(_encoder_resolvers)
    for resolver in snapshot:
        try:
            encoder = resolver(model_name)
        except Exception:
            _logger.exception(
                "custom encoder resolver %r raised for model_name=%r; skipping",
                resolver,
                model_name,
            )
            continue
        if encoder is None:
            continue
        if not isinstance(encoder, Encoder):
            _logger.warning(
                "custom encoder resolver %r returned %r for model_name=%r, "
                "which does not satisfy the Encoder protocol (needs "
                "``embedding_dim`` and ``encode``); skipping",
                resolver,
                encoder,
                model_name,
            )
            continue
        return encoder
    return None


def _embed_via_custom(texts: list[str], encoder: "Encoder") -> list[list[float]]:
    """Run a custom encoder and normalise its output to ``list[list[float]]``.

    Accepts numpy arrays, nested lists, or any object with ``tolist``.
    Validates shape: raises :class:`ValueError` if the returned batch
    has the wrong row count or a vector has the wrong dimension, so
    malformed custom encoders surface with an actionable message
    instead of cascading into an obscure sqlite-vec shape error.

    ``.tolist()`` on a numpy float array already returns Python floats,
    so we can skip the ``map(float, ...)`` fallback on the hot path
    and use it only for untyped nested lists.
    """
    vecs = encoder.encode(texts)
    tolist = getattr(vecs, "tolist", None)
    if tolist is not None:
        normalized = tolist()
    else:
        # Untyped nested iterable (e.g. a plain Python list of lists).
        # ``float()`` here rejects non-numeric elements with a clear
        # ``TypeError`` before the vectors reach the store.
        normalized = [list(map(float, v)) for v in vecs]

    if len(normalized) != len(texts):
        raise ValueError(
            f"Custom encoder returned {len(normalized)} vectors for {len(texts)} texts"
        )
    expected_dim = int(encoder.embedding_dim)
    for i, v in enumerate(normalized):
        if len(v) != expected_dim:
            raise ValueError(
                f"Custom encoder returned vector at index {i} with "
                f"dimension {len(v)}; expected {expected_dim}"
            )
    return normalized


# ------------------------------------------------------------------ #
# Model registry                                                       #
# ------------------------------------------------------------------ #

KNOWN_DIMS: dict[str, int] = {
    # English-only models
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    # RRF-tuned: BGE-small fine-tuned with hybrid retrieval disagreement signal.
    # v3 (2026-04-19): H-R9 winning config, temperature=0.5 + total_triples=60000.
    # +5.35% macro NDCG@10 on SciFact+NFCorpus+FiQA vs base bge-small.
    # v2: MNRL + explicit hard negatives. +7.4% SciFact, +19.5% NFCorpus vs base.
    # Loaded via SentenceTransformer (v3) / custom HF ONNX backend (v1/v2).
    # Use via ``vstash reindex --model Stffens/bge-small-rrf-v3``.
    "Stffens/bge-small-rrf-v3": 384,
    "Stffens/bge-small-rrf-v2": 384,
    "stffens/bge-small-rrf-v1": 384,  # legacy v1
    # Multilingual models (FastEmbed-supported)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
    # EmbeddingGemma (Google, ONNX via onnx-community)
    # +7.4% NDCG vs BGE-small on SciFact, 100+ languages
    "embeddinggemma-300m": 768,
}

# EmbeddingGemma ONNX config
_GEMMA_ONNX_REPO = "onnx-community/embeddinggemma-300m-ONNX"
_GEMMA_MAX_TOKENS = 2048

# Mapping from standard model names to MLX community variants
_MLX_MODEL_MAP: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "mlx-community/bge-small-en-v1.5-bf16",
    "BAAI/bge-base-en-v1.5": "mlx-community/bge-base-en-v1.5-bf16",
    "BAAI/bge-large-en-v1.5": "mlx-community/bge-large-en-v1.5-bf16",
    "intfloat/multilingual-e5-large": "mlx-community/multilingual-e5-large-bf16",
}

BackendType = Literal["onnx", "mlx", "auto"]


def get_embedding_dim(model_name: str) -> int:
    """Get the embedding dimensionality for a known model.

    Custom encoders registered via :func:`register_encoder_resolver`
    are consulted first -- their ``embedding_dim`` attribute is
    authoritative for names they handle.  Built-in FastEmbed models
    fall back to :data:`KNOWN_DIMS`.

    Args:
        model_name: Model identifier (FastEmbed name or a custom name
            handled by a registered resolver).

    Returns:
        Embedding dimension.

    Raises:
        ValueError: If the model is not in the known registry and no
            registered resolver claims the name.
    """
    custom = _resolve_custom_encoder(model_name)
    if custom is not None:
        return int(custom.embedding_dim)
    dim = KNOWN_DIMS.get(model_name)
    if dim is None:
        raise ValueError(
            f"Unknown embedding model: '{model_name}'. "
            f"Known models: {', '.join(sorted(KNOWN_DIMS))}. "
            "Add it to KNOWN_DIMS in embed.py with its dimension, or "
            "register a custom resolver via register_encoder_resolver()."
        )
    return dim


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
_onnx_lock = threading.Lock()


def _get_onnx_model(model_name: str) -> TextEmbedding:
    """Get or lazily initialize a FastEmbed ONNX model.

    Args:
        model_name: FastEmbed model identifier.

    Returns:
        Cached TextEmbedding instance.
    """
    if model_name not in _onnx_cache:
        with _onnx_lock:
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
# EmbeddingGemma backend (ONNX Runtime direct)                        #
# ------------------------------------------------------------------ #

_gemma_session: object | None = None
_gemma_tokenizer: object | None = None
_gemma_lock = threading.Lock()


def _is_gemma_model(model_name: str) -> bool:
    """Check if the model name refers to EmbeddingGemma."""
    return "embeddinggemma" in model_name.lower().replace("-", "").replace("_", "")


def _init_gemma() -> tuple:
    """Lazily initialize EmbeddingGemma ONNX session and tokenizer."""
    global _gemma_session, _gemma_tokenizer  # noqa: PLW0603
    if _gemma_session is None:
        with _gemma_lock:
            if _gemma_session is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                model_path = hf_hub_download(_GEMMA_ONNX_REPO, "onnx/model.onnx")
                hf_hub_download(_GEMMA_ONNX_REPO, "onnx/model.onnx_data")
                tokenizer_path = hf_hub_download(_GEMMA_ONNX_REPO, "tokenizer.json")
                _gemma_session = ort.InferenceSession(model_path)
                _gemma_tokenizer = Tokenizer.from_file(tokenizer_path)
                _gemma_tokenizer.enable_truncation(max_length=_GEMMA_MAX_TOKENS)
    return _gemma_session, _gemma_tokenizer


def _embed_gemma_one(text: str) -> list[float]:
    """Embed a single text with EmbeddingGemma."""
    import numpy as np

    session, tokenizer = _init_gemma()
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids[:_GEMMA_MAX_TOKENS]], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask[:_GEMMA_MAX_TOKENS]], dtype=np.int64)
    outputs = session.run(
        ["sentence_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    emb = outputs[0]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0].tolist()


_GEMMA_BATCH_SIZE = 16


def _embed_gemma(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a batch of texts with EmbeddingGemma using batched inference."""
    import numpy as np

    if not texts:
        return []

    session, tokenizer = _init_gemma()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _GEMMA_BATCH_SIZE):
        batch = texts[i : i + _GEMMA_BATCH_SIZE]
        encodings = tokenizer.encode_batch(batch)
        # Pad to max length in this batch
        max_len = max(len(e.ids) for e in encodings)
        batch_ids = np.array(
            [e.ids[:max_len] + [0] * (max_len - len(e.ids)) for e in encodings],
            dtype=np.int64,
        )
        batch_mask = np.array(
            [
                e.attention_mask[:max_len] + [0] * (max_len - len(e.attention_mask))
                for e in encodings
            ],
            dtype=np.int64,
        )
        outputs = session.run(
            ["sentence_embedding"],
            {"input_ids": batch_ids, "attention_mask": batch_mask},
        )
        embs = outputs[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embs = embs / norms
        all_embeddings.extend(embs.tolist())

    return all_embeddings


def _query_gemma(text: str, model_name: str) -> list[float]:
    """Embed a single query with EmbeddingGemma."""
    return _embed_gemma_one(text)


def _warmup_gemma(model_name: str) -> None:
    """Pre-load EmbeddingGemma ONNX model."""
    _embed_gemma_one("warmup")


# ------------------------------------------------------------------ #
# Custom HuggingFace ONNX backend (for models not in FastEmbed)        #
# ------------------------------------------------------------------ #

# Models that ship an onnx/ folder on HuggingFace but are not in
# FastEmbed's whitelist. Loaded via onnxruntime + tokenizers directly.
_HF_ONNX_MODELS: set[str] = {
    "stffens/bge-small-rrf-v1",
    "Stffens/bge-small-rrf-v2",
}

_hf_onnx_cache: dict[str, tuple] = {}  # model_name -> (session, tokenizer, max_len)
_hf_onnx_lock = threading.Lock()
_hf_st_cache: dict[str, object] = {}  # model_name -> SentenceTransformer
_hf_st_lock = threading.Lock()
_hf_onnx_unavailable: set[str] = set()


def _is_hf_onnx_model(model_name: str) -> bool:
    """Check if this model should use the custom HF ONNX backend."""
    return model_name in _HF_ONNX_MODELS


def _init_hf_onnx(model_name: str) -> tuple:
    """Lazily initialize a custom HF ONNX model."""
    if model_name not in _hf_onnx_cache:
        with _hf_onnx_lock:
            if model_name not in _hf_onnx_cache:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                # Try onnx/ subfolder first, then root (varies by repo layout)
                from huggingface_hub.errors import EntryNotFoundError

                try:
                    model_path = hf_hub_download(model_name, "onnx/model.onnx")
                    onnx_prefix = "onnx/"
                except (EntryNotFoundError, OSError):
                    model_path = hf_hub_download(model_name, "model.onnx")
                    onnx_prefix = ""
                # Large ONNX exports store weights in an external data file
                # (`model.onnx.data` with a dot, or `model.onnx_data` with
                # an underscore depending on the exporter version). ONNX
                # Runtime expects that file to sit next to the .onnx. Fetch
                # it alongside; best-effort, not all models have it.
                for data_name in (
                    f"{onnx_prefix}model.onnx.data",
                    f"{onnx_prefix}model.onnx_data",
                ):
                    try:
                        hf_hub_download(model_name, data_name)
                        break
                    except (EntryNotFoundError, OSError):
                        continue
                try:
                    tokenizer_path = hf_hub_download(model_name, "onnx/tokenizer.json")
                except (EntryNotFoundError, OSError):
                    tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
                session = ort.InferenceSession(model_path)
                tokenizer = Tokenizer.from_file(tokenizer_path)
                tokenizer.enable_truncation(max_length=512)
                _hf_onnx_cache[model_name] = (session, tokenizer, 512)
    return _hf_onnx_cache[model_name]


def _init_hf_st(model_name: str):
    """Lazily load a SentenceTransformer for an HF model. Used as the
    fallback path when the ONNX export is broken (missing external data
    file) or when sentence-transformers is the only published format."""
    if model_name not in _hf_st_cache:
        with _hf_st_lock:
            if model_name not in _hf_st_cache:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        f"Cannot load {model_name}: ONNX init failed AND "
                        "sentence-transformers is not installed. Install with: "
                        "pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'"
                    ) from exc
                _logger.info("Loading %s via SentenceTransformer fallback", model_name)
                _hf_st_cache[model_name] = SentenceTransformer(model_name)
    return _hf_st_cache[model_name]


def _embed_hf_st(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts via the SentenceTransformer fallback path."""
    if not texts:
        return []
    model = _init_hf_st(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vecs]


def _embed_hf_onnx(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts using a custom HF ONNX model with mean pooling.

    Falls back to ``_embed_hf_st`` if the ONNX init fails for this model
    (e.g., the HF repo publishes a stub ``model.onnx`` whose external
    ``.data`` file was never uploaded).
    """
    import numpy as np

    if not texts:
        return []

    if model_name in _hf_onnx_unavailable:
        return _embed_hf_st(texts, model_name)

    try:
        session, tokenizer, max_len = _init_hf_onnx(model_name)
    except Exception as exc:  # includes onnxruntime RuntimeException
        _logger.warning(
            "HF ONNX init failed for %s (%s); falling back to sentence-transformers.",
            model_name,
            exc.__class__.__name__,
        )
        _hf_onnx_unavailable.add(model_name)
        return _embed_hf_st(texts, model_name)
    all_embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch)
        seq_len = min(max(len(e.ids) for e in encodings), max_len)
        input_ids = np.array(
            [e.ids[:seq_len] + [0] * (seq_len - min(len(e.ids), seq_len)) for e in encodings],
            dtype=np.int64,
        )
        attention_mask = np.array(
            [
                e.attention_mask[:seq_len] + [0] * (seq_len - min(len(e.attention_mask), seq_len))
                for e in encodings
            ],
            dtype=np.int64,
        )
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Some ONNX exports include token_type_ids, others don't
        input_names = {inp.name for inp in session.get_inputs()}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = session.run(None, feed)
        # Mean pooling over token embeddings (output 0 is last_hidden_state)
        hidden = outputs[0]  # (batch, seq_len, dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (hidden * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        embs = summed / counts
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
        embs = embs / norms
        all_embeddings.extend(embs.tolist())

    return all_embeddings


def _query_hf_onnx(text: str, model_name: str) -> list[float]:
    """Embed a single query with a custom HF ONNX model."""
    return _embed_hf_onnx([text], model_name)[0]


def _warmup_hf_onnx(model_name: str) -> None:
    """Pre-load a custom HF ONNX model."""
    _embed_hf_onnx(["warmup"], model_name)


# ------------------------------------------------------------------ #
# MLX backend (Apple Silicon GPU)                                      #
# ------------------------------------------------------------------ #

_mlx_cache: dict[str, tuple] = {}  # (model, tokenizer)
_mlx_lock = threading.Lock()


def _get_mlx_model(model_name: str) -> tuple:
    """Get or lazily initialize an MLX model.

    Args:
        model_name: Standard model identifier (mapped to MLX variant).

    Returns:
        Tuple of (model, tokenizer).
    """
    if model_name not in _mlx_cache:
        with _mlx_lock:
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
    token_type_ids = mx.array(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])))

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
    """Pre-load MLX model and JIT-compile GPU kernels.

    Runs multiple passes with varying sequence lengths so the MLX
    JIT compiler has cached all common kernel shapes before the
    first real query.
    """
    _embed_mlx(["warmup"], model_name)
    _embed_mlx(["This is a longer warmup sentence to trigger JIT for varied lengths."], model_name)
    _embed_mlx(["short", "another sentence of medium length for the JIT pass"], model_name)


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
    if _is_gemma_model(model_name):
        _warmup_gemma(model_name)
        return
    if _is_hf_onnx_model(model_name):
        _warmup_hf_onnx(model_name)
        return
    resolved = _resolve(backend)
    if resolved == "mlx":
        _warmup_mlx(model_name)
    else:
        _warmup_onnx(model_name)


# ------------------------------------------------------------------ #
# Daemon client -- use a running `vstash serve` for embedding           #
# ------------------------------------------------------------------ #

# Set VSTASH_EMBED_URL to force daemon usage (e.g. "http://127.0.0.1:8585").
# When unset, the client probes localhost:8585 once and caches the result.
#
# _VSTASH_IS_DAEMON is set by `vstash serve` to prevent the daemon
# process from delegating to itself (infinite recursion).
_daemon_url: str | None = os.getenv("VSTASH_EMBED_URL")
_daemon_checked: bool = False
_daemon_available: bool = False
_daemon_lock = threading.Lock()
_is_daemon: bool = os.getenv("_VSTASH_IS_DAEMON") == "1"

_DAEMON_DEFAULT_URL = "http://127.0.0.1:8585"
_DAEMON_TIMEOUT = 0.3  # seconds -- fast fail for localhost


def _check_daemon() -> str | None:
    """Probe the daemon URL once. Returns URL if reachable, else None.

    Returns None immediately when running inside the daemon process
    to prevent infinite recursion. Probes both explicit VSTASH_EMBED_URL
    and the default localhost:8585, caching the result either way.
    """
    if _is_daemon:
        return None
    global _daemon_checked, _daemon_available, _daemon_url  # noqa: PLW0603
    if _daemon_checked:
        return _daemon_url if _daemon_available else None
    with _daemon_lock:
        if _daemon_checked:
            return _daemon_url if _daemon_available else None
        url = (_daemon_url or _DAEMON_DEFAULT_URL).rstrip("/")
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=_DAEMON_TIMEOUT):
                _daemon_url = url
                _daemon_available = True
                _logger.debug("vstash daemon detected at %s", _daemon_url)
        except Exception:
            _daemon_available = False
        _daemon_checked = True
        return _daemon_url if _daemon_available else None


def _mark_daemon_unavailable() -> None:
    """Mark daemon as unavailable after a failed request.

    Resets _daemon_checked so the next call re-probes. This handles
    daemon crashes gracefully: one failed request triggers a re-probe
    instead of paying the full HTTP timeout on every subsequent call.
    """
    global _daemon_checked, _daemon_available  # noqa: PLW0603
    _daemon_available = False
    _daemon_checked = False


def _daemon_embed_query(text: str, url: str, model_name: str) -> list[float] | None:
    """Embed a single query via the daemon. Returns None on failure or model mismatch."""
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"{url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            result = json.loads(resp.read())
            if result.get("model") != model_name:
                _logger.warning(
                    "Daemon model mismatch: requested %s, got %s. Falling back to local.",
                    model_name,
                    result.get("model"),
                )
                return None
            return result.get("embedding")
    except Exception:
        _mark_daemon_unavailable()
        return None


def _daemon_embed_texts(texts: list[str], url: str, model_name: str) -> list[list[float]] | None:
    """Embed a batch of texts via the daemon. Returns None on failure or model mismatch."""
    try:
        data = json.dumps({"texts": texts}).encode()
        req = urllib.request.Request(
            f"{url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            result = json.loads(resp.read())
            if result.get("model") != model_name:
                _logger.warning(
                    "Daemon model mismatch: requested %s, got %s. Falling back to local.",
                    model_name,
                    result.get("model"),
                )
                return None
            return result.get("embeddings")
    except Exception:
        _mark_daemon_unavailable()
        return None


def embed_texts(
    texts: list[str],
    model_name: str,
    backend: BackendType = "auto",
) -> list[list[float]]:
    """Embed a batch of texts.

    When a ``vstash serve`` daemon is running on localhost:8585
    (or at ``VSTASH_EMBED_URL``), the embedding is delegated to
    the daemon's warm model.  Falls back to local embedding on
    any failure.

    Args:
        texts: List of text strings to embed.
        model_name: Model identifier.
        backend: Backend to use — 'onnx', 'mlx', or 'auto'.

    Returns:
        List of float vectors, one per input text.
    """
    if not texts:
        return []

    # Custom resolvers win over every built-in path -- the daemon only
    # has the default model loaded, and name-based backend detection
    # (_is_gemma_model, _is_hf_onnx_model) would false-match a custom
    # name that happens to share a prefix.
    custom = _resolve_custom_encoder(model_name)
    if custom is not None:
        return _embed_via_custom(texts, custom)

    url = _check_daemon()
    if url is not None:
        result = _daemon_embed_texts(texts, url, model_name)
        if result is not None and len(result) == len(texts):
            return result

    if _is_gemma_model(model_name):
        return _embed_gemma(texts, model_name)
    if _is_hf_onnx_model(model_name):
        return _embed_hf_onnx(texts, model_name)
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

    When a ``vstash serve`` daemon is running, delegates to the
    daemon's warm model.  Falls back to local embedding on failure.

    Args:
        text: Query string to embed.
        model_name: Model identifier.
        backend: Backend to use — 'onnx', 'mlx', or 'auto'.

    Returns:
        Float vector for the query.
    """
    # Custom resolvers win over every built-in path (see embed_texts).
    custom = _resolve_custom_encoder(model_name)
    if custom is not None:
        vecs = _embed_via_custom([text], custom)
        return vecs[0]

    url = _check_daemon()
    if url is not None:
        result = _daemon_embed_query(text, url, model_name)
        if result is not None:
            return result

    if _is_gemma_model(model_name):
        return _query_gemma(text, model_name)
    if _is_hf_onnx_model(model_name):
        return _query_hf_onnx(text, model_name)
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
