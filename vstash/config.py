"""
config.py — vstash.toml loader with sane defaults.

Resolution order:
  1. vstash.toml in current directory
  2. ~/.vstash/vstash.toml (global)
  3. Built-in defaults

All config sections are Pydantic v2 BaseModel with validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InferenceConfig(BaseModel):
    """Inference backend selection."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["local", "cerebras", "ollama", "openai"] = Field(
        default="local",
        description="Inference backend: 'local' (auto-detect, default), 'ollama', 'cerebras', or 'openai'",
    )
    model: str = Field(
        default="auto",
        description="Model name for the selected backend ('auto' = use first available local model)",
    )


class CerebrasConfig(BaseModel):
    """Cerebras API configuration.

    **Security note**: Prefer setting the ``CEREBRAS_API_KEY`` environment
    variable instead of storing the key in ``vstash.toml``, which may be
    accidentally committed to version control.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(
        default="", description="Cerebras API key (prefer CEREBRAS_API_KEY env var)"
    )


class OllamaConfig(BaseModel):
    """Ollama local inference configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="qwen3.5:9b", description="Ollama model name")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration.

    **Security note**: Prefer setting the ``OPENAI_API_KEY`` environment
    variable instead of storing the key in ``vstash.toml``.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(default="", description="OpenAI API key (prefer OPENAI_API_KEY env var)")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible APIs",
    )
    extra_body: dict[str, object] | None = Field(
        default=None,
        description="Extra JSON body fields passed to chat completions (e.g., chat_template_kwargs for Qwen thinking mode)",
    )


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration.

    Supported models:
      - ``BAAI/bge-small-en-v1.5`` (384 dims, English, fastest — default)
      - ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` (384 dims, 50+ languages)
      - ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2`` (768 dims, 50+ languages)
      - ``intfloat/multilingual-e5-large`` (1024 dims, 100+ languages, highest quality)

    **Tip**: For multilingual corpora, set
    ``model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"``
    in ``vstash.toml`` and run ``vstash reindex`` to re-embed existing chunks.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model name (use paraphrase-multilingual-MiniLM-L12-v2 for multilingual)",
    )
    backend: Literal["onnx", "mlx", "auto"] = Field(
        default="auto",
        description="Embedding backend: 'onnx' (portable), 'mlx' (Apple Silicon), or 'auto'",
    )


class ChunkingConfig(BaseModel):
    """Text chunking parameters."""

    model_config = ConfigDict(frozen=True)

    size: int = Field(default=1024, gt=0, description="Tokens per chunk")
    overlap: int = Field(default=128, ge=0, description="Token overlap between chunks")
    top_k: int = Field(default=5, gt=0, description="Chunks retrieved per query")
    code_aware: bool = Field(
        default=True,
        description="Use syntax-aware splitting for code files (functions, classes)",
    )


class StorageConfig(BaseModel):
    """Database storage configuration."""

    model_config = ConfigDict(frozen=True)

    db_path: str = Field(
        default_factory=lambda: os.getenv("VSTASH_DB_PATH") or "~/.vstash/memory.db",
        description="Path to SQLite database file",
    )
    vector_backend: Literal["sqlite-vec", "snapvec", "snapvec-ivfpq"] = Field(
        default="sqlite-vec",
        description=(
            "Vector search backend: 'sqlite-vec' (default, exact), "
            "'snapvec' (flat compressed ANN), "
            "'snapvec-ivfpq' (IVF + residual PQ with fp16 rerank, >=50K chunks)"
        ),
    )
    snapvec_bits: int = Field(
        default=4,
        ge=2,
        le=4,
        description="Quantization bits for snapvec flat backend (2, 3, or 4)",
    )
    ivfpq_nlist: int = Field(
        default=0,
        ge=0,
        le=1024,
        description=(
            "IVF coarse clusters for snapvec-ivfpq. 0 = auto (4 * sqrt(N)). "
            "Explicit values are clamped to [8, 1024] to match the internal "
            "_nlist_for clamp. FAISS rule: need at least 30 training vectors "
            "per cluster."
        ),
    )
    ivfpq_M: int = Field(
        default=96,
        ge=1,
        description="PQ subspaces for snapvec-ivfpq. Must divide embedding_dim.",
    )
    ivfpq_K: int = Field(
        default=256,
        ge=2,
        le=256,
        description="PQ centroids per subspace for snapvec-ivfpq.",
    )
    ivfpq_rerank_candidates: int = Field(
        default=100,
        ge=0,
        description=(
            "Candidates to rerank with fp16 full-precision cache for "
            "snapvec-ivfpq. 0 disables rerank; 100 typically saturates recall."
        ),
    )
    ivfpq_nprobe: int = Field(
        default=0,
        ge=0,
        description=(
            "Clusters to visit per query. 0 = snapvec default (nlist // 16). "
            "Higher = better recall at linear latency cost."
        ),
    )

    @field_validator("ivfpq_nlist")
    @classmethod
    def _check_ivfpq_nlist(cls, value: int) -> int:
        """Reject explicit nlist values below the FAISS-sane lower bound.

        ``0`` means "auto-derive" and is passed through unchanged. Any
        other value must satisfy the same ``[8, 1024]`` range enforced
        at runtime by ``VstashStore._nlist_for``. Catching this at config
        load time turns a late, cryptic FAISS failure into an early
        validation error that names the offending field.
        """
        if value == 0:
            return value
        if value < 8:
            raise ValueError(
                f"ivfpq_nlist={value} is below the FAISS sane minimum of 8. "
                "Use 0 for auto-derive or a value in [8, 1024]."
            )
        return value


class ScoringConfig(BaseModel):
    """Frequency + decay memory scoring configuration.

    When enabled, search results are re-ranked post-RRF using a formula
    that combines semantic relevance with access frequency and temporal decay:

        final_score = alpha * normalized_rrf + beta * log(1 + access_count * e^(-lambda * days_ago))
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable frequency+decay re-ranking")
    alpha: float = Field(
        default=0.8, ge=0, le=1, description="Weight for semantic similarity (RRF)"
    )
    beta: float = Field(default=0.2, ge=0, le=1, description="Weight for access history")
    decay_lambda: float = Field(default=0.05, gt=0, description="Decay rate (0.05=weeks, 0.1=days)")
    over_fetch: int = Field(
        default=50, gt=0, description="Candidates to retrieve before re-ranking"
    )
    track_access: bool = Field(
        default=True,
        description="Record access counts on search (enabled by default when scoring is on)",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "MMR diversity parameter for intra-document dedup. "
            "1.0 = hard dedup (at most one chunk per document), "
            "0.0 = maximum diversity (no relevance weight). "
            "Default 0.5 balances relevance and diversity."
        ),
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> ScoringConfig:
        if self.alpha + self.beta > 1.0:
            msg = f"alpha ({self.alpha}) + beta ({self.beta}) must be <= 1.0"
            raise ValueError(msg)
        return self


class RecencyConfig(BaseModel):
    """Temporal recency boost for agentic memory.

    When enabled, search results are boosted by a temporal decay factor
    that favors recently created chunks.  This is independent of the
    retrieval pipeline (vector + FTS5 + RRF) — it only biases the final
    ranking toward recency.

    Designed for agentic memory where recent context matters more than
    old context.  Off by default for pure retrieval; the SDK's
    ``Memory.search()`` can enable it per call.
    """

    model_config = ConfigDict(frozen=True)

    boost: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Recency boost multiplier (0.0 = off, 0.5 = mild, 1.0 = strong). "
            "Applied as score *= (1 + boost * exp(-0.05 * days_ago))."
        ),
    )


class ObservabilityConfig(BaseModel):
    """Observability and operational metrics settings.

    Controls the built-in metrics registry (counters, gauges,
    histograms) and the slow-query logger.  The metrics registry
    itself is always on and cheap; these settings only control
    the slow-query log threshold and related operational hints.
    """

    model_config = ConfigDict(frozen=True)

    slow_query_ms: float = Field(
        default=100.0,
        ge=0,
        description=(
            "Search queries taking longer than this (in ms) are logged "
            "to stderr with their query text, latency, and result count. "
            "Set to 0 to log every query (debug), or a very high number "
            "to effectively disable."
        ),
    )
    auto_miss_hint: bool = Field(
        default=True,
        description=(
            "Record a lightweight ``miss_hint`` JSON on search_events "
            "rows when a query returns zero results or a result set "
            "entirely in the 'low' relevance tier (distance > 0.4802 "
            "cosine; see ``RELEVANCE_TIER_MEDIUM_MAX`` in store.py). "
            "Set to false to skip the extra JSON column write per miss. "
            "The hint is consumed by ``vstash why --recent``. Added in "
            "issue #157 part 3."
        ),
    )


class LimitsConfig(BaseModel):
    """Explicit limits for the public store/search APIs (#133).

    A good substrate is honest about its boundaries.  These limits gate
    inputs that would otherwise crash inside SQLite, sqlite-vec, or the
    embedding model with cryptic errors.  Defaults are intentionally
    generous — they catch pathological inputs without inconveniencing
    realistic ones.

    Each limit is enforced once at the public API boundary in
    :mod:`vstash.validation`; the hot path is unaffected.
    """

    model_config = ConfigDict(frozen=True)

    max_query_chars: int = Field(
        default=10_000,
        gt=0,
        description=(
            "Maximum number of characters allowed in a search query. "
            "Above this, queries are rejected before reaching the embedding "
            "model.  10 000 chars ≈ 2 000 tokens for most tokenizers."
        ),
    )
    max_top_k: int = Field(
        default=1_000,
        gt=0,
        description=(
            "Maximum value of ``top_k`` accepted by ``search()``.  Larger "
            "values would pull a substantial fraction of the corpus into "
            "Python and sort it, with quadratic memory cost."
        ),
    )
    max_distance_cutoff: float = Field(
        default=1_000.0,
        gt=0,
        description=(
            "Maximum value of ``distance_cutoff``.  The default cutoff "
            "in ``search()`` is ~1.15; values above 1 000 are almost "
            "certainly a caller bug."
        ),
    )
    max_recency_boost: float = Field(
        default=100.0,
        gt=0,
        description=(
            "Maximum value of ``recency_boost``.  Sane values are 0.0 to "
            "1.0; the cap exists to prevent float overflow in the decay "
            "formula when callers pass huge numbers."
        ),
    )
    max_path_chars: int = Field(
        default=4_096,
        gt=0,
        description=(
            "Maximum length of a document path.  POSIX ``PATH_MAX`` is "
            "typically 4096; longer paths cannot be opened on most "
            "filesystems anyway."
        ),
    )
    max_chunks_per_document: int = Field(
        default=50_000,
        gt=0,
        description=(
            "Maximum number of chunks a single ``add_document`` call may "
            "ingest.  Above this, the single transaction lock-steps the "
            "whole DB and the right answer is to split the document."
        ),
    )
    max_chunk_chars: int = Field(
        default=1_048_576,  # 1 MiB
        gt=0,
        description=(
            "Maximum size of an individual chunk in characters (default "
            "1 MiB).  A chunk this large is almost always a chunking bug "
            "and would crash the embedding model."
        ),
    )


class CacheConfig(BaseModel):
    """Query result caching configuration."""

    model_config = ConfigDict(frozen=True)

    query_cache_size: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description=(
            "Maximum number of query results to cache in memory (LRU eviction). "
            "0 = disabled (default). 128-512 is a good range for interactive use."
        ),
    )


class VstashConfig(BaseModel):
    """Root configuration for vstash.

    **Forward compatibility (#135).** Extra top-level keys in
    ``vstash.toml`` are accepted with a one-time warning instead of
    raising a hard error, so a user on an older vstash with a
    newer-format config file is not blocked from running.  This is the
    one place we deliberately relax Pydantic strictness — every nested
    section keeps its strict schema.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_top_level_keys(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        known = set(cls.model_fields.keys())
        unknown = sorted(k for k in values if k not in known)
        if unknown:
            import logging as _logging

            _logging.getLogger("vstash").warning(
                "vstash.toml contains unknown top-level section(s): %s. "
                "They are ignored by this build of vstash. "
                "If you are on an older vstash, upgrade; if these were "
                "removed intentionally, delete them from vstash.toml.",
                ", ".join(unknown),
            )
        return values

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cerebras: CerebrasConfig = Field(default_factory=CerebrasConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    recency: RecencyConfig = Field(default_factory=RecencyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @property
    def cerebras_api_key(self) -> str:
        """Resolve Cerebras API key from config or environment."""
        return self.cerebras.api_key or os.getenv("CEREBRAS_API_KEY", "")

    @property
    def openai_api_key(self) -> str:
        """Resolve OpenAI API key from config or environment."""
        return self.openai.api_key or os.getenv("OPENAI_API_KEY", "")

    @property
    def db_path(self) -> str:
        """Shortcut to storage.db_path."""
        return self.storage.db_path


# Canonical set of file extensions supported for ingestion.
# Used by ingest_directory() and watch mode — defined here to avoid duplication.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".md",
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".html",
        ".htm",
        ".csv",
    }
)

# Directories always excluded from recursive ingestion and watch mode.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)

# Safety limits for directory ingestion.
MAX_DIR_FILES = 500
MAX_DIR_BYTES = 200 * 1024 * 1024  # 200 MB


def load_config() -> VstashConfig:
    """Load config from vstash.toml, falling back to defaults.

    Resolution order:
        0. ``VSTASH_CONFIG`` environment variable (if set)
        1. ``./vstash.toml`` in the current directory
        2. ``~/.vstash/vstash.toml`` (global)
        3. Built-in defaults
    """
    candidates: list[Path] = []

    # Allow explicit config path via env var (useful for MCP server)
    env_config = os.getenv("VSTASH_CONFIG")
    if env_config:
        candidates.append(Path(env_config).expanduser())

    candidates.extend(
        [
            Path.cwd() / "vstash.toml",
            Path.home() / ".vstash" / "vstash.toml",
        ]
    )

    raw: dict = {}  # type: ignore[type-arg]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            break

    return VstashConfig.model_validate(raw)
