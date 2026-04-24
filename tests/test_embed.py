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


class TestHFOnnxUnavailableConcurrency:
    """Pin the chosen concurrency semantics of ``_hf_onnx_unavailable``.

    The set is deliberately unlocked (set ops are atomic under the
    GIL; a real guard would have to span the slow ``_init_hf_onnx``
    call and serialize every fallback).  This test exercises the
    concurrent fallback path and asserts the two contracts we care
    about: no thread deadlocks, and the failure marker sticks.

    It does **not** assert "``_init_hf_onnx`` was called exactly
    once" -- accepting up to ``n_threads`` init attempts is the
    explicit trade-off for keeping the hot path lock-free.
    """

    def test_concurrent_init_fallback_is_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import threading

        import vstash.embed as embed_mod

        # Start clean so we don't leak state into other tests.
        embed_mod._hf_onnx_unavailable.discard("broken/model")
        init_calls = 0
        init_calls_lock = threading.Lock()

        def always_fail(model_name: str) -> tuple:
            nonlocal init_calls
            with init_calls_lock:
                init_calls += 1
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
            # (1) No thread is stuck -- a future refactor that
            # introduces a lock around the slow path would show up
            # here as a join timeout.
            for t in threads:
                assert not t.is_alive(), "worker thread did not finish within timeout"
            # (2) The failure marker is set after the storm.
            assert "broken/model" in embed_mod._hf_onnx_unavailable
            # (3) Each thread independently tried to init at least
            # once; we accept up to n_threads attempts (the
            # deliberately-benign race).  Fail only if zero attempts
            # happened, which would mean the fallback path never ran.
            assert 1 <= init_calls <= n_threads
        finally:
            embed_mod._hf_onnx_unavailable.discard("broken/model")
