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

from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    """Inference backend selection."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["cerebras", "ollama", "openai"] = Field(
        default="cerebras",
        description="Inference backend: 'cerebras', 'ollama', or 'openai'",
    )
    model: str = Field(
        default="llama3.1-8b",
        description="Model name for the selected backend",
    )


class CerebrasConfig(BaseModel):
    """Cerebras API configuration."""

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(default="", description="Cerebras API key (or set CEREBRAS_API_KEY)")


class OllamaConfig(BaseModel):
    """Ollama local inference configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="llama3.2", description="Ollama model name")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(default="", description="OpenAI API key (or set OPENAI_API_KEY)")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible APIs",
    )


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model name",
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


class StorageConfig(BaseModel):
    """Database storage configuration."""

    model_config = ConfigDict(frozen=True)

    db_path: str = Field(default="~/.vstash/memory.db", description="Path to SQLite database file")


class VstashConfig(BaseModel):
    """Root configuration for vstash."""

    model_config = ConfigDict(frozen=False)  # Allow mutation during load

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cerebras: CerebrasConfig = Field(default_factory=CerebrasConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

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


def load_config() -> VstashConfig:
    """Load config from vstash.toml, falling back to defaults.

    Resolution order:
        1. ``./vstash.toml`` in the current directory
        2. ``~/.vstash/vstash.toml`` (global)
        3. Built-in defaults
    """
    candidates: list[Path] = [
        Path.cwd() / "vstash.toml",
        Path.home() / ".vstash" / "vstash.toml",
    ]

    raw: dict = {}  # type: ignore[type-arg]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            break

    return VstashConfig.model_validate(raw)
