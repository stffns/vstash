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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    vector_backend: Literal["sqlite-vec", "snapvec"] = Field(
        default="sqlite-vec",
        description="Vector search backend: 'sqlite-vec' (default) or 'snapvec' (compressed ANN)",
    )
    snapvec_bits: int = Field(
        default=4,
        ge=2,
        le=4,
        description="Quantization bits for snapvec backend (2, 3, or 4)",
    )


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


class VstashConfig(BaseModel):
    """Root configuration for vstash."""

    model_config = ConfigDict(frozen=True)

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
