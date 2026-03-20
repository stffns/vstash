"""Tests for vstash.config — Pydantic config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vstash.config import (
    ChunkingConfig,
    InferenceConfig,
    VstashConfig,
    load_config,
)


class TestConfigDefaults:
    """Test that default config values are correct."""

    def test_default_inference_backend(self) -> None:
        cfg = VstashConfig()
        assert cfg.inference.backend == "cerebras"

    def test_default_inference_model(self) -> None:
        cfg = VstashConfig()
        assert cfg.inference.model == "llama3.1-8b"

    def test_default_embedding_model(self) -> None:
        cfg = VstashConfig()
        assert cfg.embeddings.model == "BAAI/bge-small-en-v1.5"

    def test_default_chunk_size(self) -> None:
        cfg = VstashConfig()
        assert cfg.chunking.size == 1024
        assert cfg.chunking.overlap == 128
        assert cfg.chunking.top_k == 5

    def test_default_db_path(self) -> None:
        cfg = VstashConfig()
        assert cfg.storage.db_path == "~/.vstash/memory.db"

    def test_default_openai_model(self) -> None:
        cfg = VstashConfig()
        assert cfg.openai.model == "gpt-4o-mini"


class TestConfigValidation:
    """Test Pydantic validation on config fields."""

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(Exception):
            InferenceConfig(backend="invalid_backend")

    def test_valid_backends_accepted(self) -> None:
        for backend in ("cerebras", "ollama", "openai"):
            cfg = InferenceConfig(backend=backend)
            assert cfg.backend == backend

    def test_chunk_size_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ChunkingConfig(size=0)

    def test_chunk_overlap_cannot_be_negative(self) -> None:
        with pytest.raises(Exception):
            ChunkingConfig(overlap=-1)

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ChunkingConfig(top_k=0)


class TestConfigProperties:
    """Test derived properties on VstashConfig."""

    def test_cerebras_api_key_from_config(self) -> None:
        cfg = VstashConfig.model_validate({"cerebras": {"api_key": "test-key"}})
        assert cfg.cerebras_api_key == "test-key"

    def test_cerebras_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CEREBRAS_API_KEY", "env-key")
        cfg = VstashConfig()
        assert cfg.cerebras_api_key == "env-key"

    def test_openai_api_key_from_config(self) -> None:
        cfg = VstashConfig.model_validate({"openai": {"api_key": "oai-key"}})
        assert cfg.openai_api_key == "oai-key"

    def test_openai_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "oai-env-key")
        cfg = VstashConfig()
        assert cfg.openai_api_key == "oai-env-key"

    def test_db_path_shortcut(self) -> None:
        cfg = VstashConfig.model_validate({"storage": {"db_path": "/custom/path.db"}})
        assert cfg.db_path == "/custom/path.db"


class TestConfigModelValidate:
    """Test VstashConfig.model_validate with TOML-like dicts."""

    def test_partial_override(self) -> None:
        raw = {
            "inference": {"backend": "ollama", "model": "mistral"},
            "chunking": {"size": 512},
        }
        cfg = VstashConfig.model_validate(raw)
        assert cfg.inference.backend == "ollama"
        assert cfg.inference.model == "mistral"
        assert cfg.chunking.size == 512
        # defaults preserved
        assert cfg.chunking.overlap == 128
        assert cfg.embeddings.model == "BAAI/bge-small-en-v1.5"

    def test_empty_dict_gives_defaults(self) -> None:
        cfg = VstashConfig.model_validate({})
        assert cfg.inference.backend == "cerebras"
        assert cfg.chunking.size == 1024


class TestLoadConfig:
    """Test load_config file resolution."""

    def test_loads_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        toml_content = b'[inference]\nbackend = "ollama"\nmodel = "custom-model"\n'
        (tmp_path / "vstash.toml").write_bytes(toml_content)

        cfg = load_config()
        assert cfg.inference.backend == "ollama"
        assert cfg.inference.model == "custom-model"

    def test_falls_back_to_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.inference.backend == "cerebras"
