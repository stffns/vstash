"""Tests for vstash.embed — FastEmbed wrapper."""

from __future__ import annotations

import pytest

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

    def test_unknown_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_embedding_dim("unknown/model-name")

    def test_known_dims_dict_is_populated(self) -> None:
        assert len(KNOWN_DIMS) >= 4


class TestHFOnnxUnavailableLock:
    """Regression coverage for the threading lock on the shared
    ``_hf_onnx_unavailable`` set.

    Protects against multiple threads each retriggering a slow ONNX
    init for a model we already know is broken.  The lock makes the
    check-then-add pattern observably single-writer; this test just
    pins that invariant so a future refactor does not silently drop
    the lock and leave a race behind.
    """

    def test_concurrent_init_fallback_is_serialised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Spawn N threads that all hit the fallback path for the same
        model_name.  The set must end with exactly one entry -- the
        lock serialises the ``.add`` even though each thread sees the
        check-before-add race individually.
        """
        import threading

        import vstash.embed as embed_mod

        # Start clean so we don't leak state into other tests.
        with embed_mod._hf_onnx_unavailable_lock:
            embed_mod._hf_onnx_unavailable.discard("broken/model")

        def always_fail(model_name: str) -> tuple:
            raise RuntimeError("simulated ONNX init failure")

        monkeypatch.setattr(embed_mod, "_init_hf_onnx", always_fail)
        # Make the fallback path a no-op so we don't download ST weights.
        monkeypatch.setattr(
            embed_mod,
            "_embed_hf_st",
            lambda texts, model_name: [[0.0] for _ in texts],
        )

        n_threads = 16
        start = threading.Event()

        def worker():
            start.wait()
            embed_mod._embed_hf_onnx(["hello"], "broken/model")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=5)

        try:
            assert "broken/model" in embed_mod._hf_onnx_unavailable
            assert sum(1 for m in embed_mod._hf_onnx_unavailable if m == "broken/model") == 1
        finally:
            with embed_mod._hf_onnx_unavailable_lock:
                embed_mod._hf_onnx_unavailable.discard("broken/model")
