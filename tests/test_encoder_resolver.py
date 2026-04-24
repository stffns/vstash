"""Tests for the custom encoder resolver hook (#278).

Covers registration / unregistration semantics, priority over every
built-in backend, fall-through when no resolver claims the name, fault
isolation between resolvers, dim discovery, and the ``get_embedding_dim``
integration that callers need for store construction.
"""

from __future__ import annotations

from typing import Any

import pytest

import vstash.embed as embed_mod
from vstash.embed import (
    Encoder,
    embed_query,
    embed_texts,
    get_embedding_dim,
    register_encoder_resolver,
    unregister_encoder_resolver,
)


class _StubEncoder:
    """Minimal fake that satisfies the :class:`Encoder` protocol.

    Returns deterministic one-hot vectors per text so tests can assert
    which encoder actually produced an output without inspecting
    floating-point equality.
    """

    def __init__(self, dim: int = 8, tag: float = 1.0) -> None:
        self.embedding_dim = dim
        self._tag = tag
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[self._tag] * self.embedding_dim for _ in texts]


class _NumpyStubEncoder:
    """Alternative stub returning a numpy array, to cover the
    ``.tolist()`` branch in :func:`_embed_via_custom`.
    """

    def __init__(self, dim: int = 4) -> None:
        self.embedding_dim = dim

    def encode(self, texts: list[str]):
        import numpy as np

        return np.full((len(texts), self.embedding_dim), 2.0, dtype=np.float32)


@pytest.fixture(autouse=True)
def _clean_resolvers():
    """Clear the global resolver list around every test.

    The resolver registry is process-global by design; tests must not
    leak registrations across cases, so we snapshot before and restore
    after.  This fixture is the contract that keeps the registry safe
    to touch in unit tests.
    """
    before = list(embed_mod._encoder_resolvers)
    embed_mod._encoder_resolvers.clear()
    try:
        yield
    finally:
        embed_mod._encoder_resolvers.clear()
        embed_mod._encoder_resolvers.extend(before)


# ------------------------------------------------------------------ #
# Protocol conformance                                                 #
# ------------------------------------------------------------------ #


class TestEncoderProtocol:
    def test_stub_satisfies_protocol_at_runtime(self) -> None:
        enc = _StubEncoder()
        assert isinstance(enc, Encoder)

    def test_missing_embedding_dim_fails_isinstance(self) -> None:
        class Bad:
            def encode(self, texts: list[str]):
                return []

        assert not isinstance(Bad(), Encoder)


# ------------------------------------------------------------------ #
# Registration and unregistration                                      #
# ------------------------------------------------------------------ #


class TestResolverRegistration:
    def test_register_adds_to_list(self) -> None:
        def resolver(name):
            return None

        register_encoder_resolver(resolver)
        assert resolver in embed_mod._encoder_resolvers

    def test_register_is_idempotent(self) -> None:
        def resolver(name):
            return None

        register_encoder_resolver(resolver)
        register_encoder_resolver(resolver)
        assert embed_mod._encoder_resolvers.count(resolver) == 1

    def test_unregister_removes(self) -> None:
        def resolver(name):
            return None

        register_encoder_resolver(resolver)
        unregister_encoder_resolver(resolver)
        assert resolver not in embed_mod._encoder_resolvers

    def test_unregister_unknown_is_noop(self) -> None:
        def never_registered(name):
            return None

        # Must not raise.
        unregister_encoder_resolver(never_registered)


# ------------------------------------------------------------------ #
# Resolver priority over built-ins                                     #
# ------------------------------------------------------------------ #


class TestResolverPriority:
    def test_resolver_wins_over_onnx_path(self) -> None:
        """A registered resolver for a well-known FastEmbed model must
        take priority over the default ONNX backend.  Proves the hook
        runs before any backend detection.
        """
        stub = _StubEncoder(dim=4, tag=7.0)

        def resolver(name: str) -> Any:
            return stub if name == "BAAI/bge-small-en-v1.5" else None

        register_encoder_resolver(resolver)

        vecs = embed_texts(["hello", "world"], "BAAI/bge-small-en-v1.5")

        assert stub.calls == [["hello", "world"]]
        assert vecs == [[7.0] * 4, [7.0] * 4]

    def test_resolver_wins_over_hf_onnx_detection(self) -> None:
        """Names that look like HF repos (``owner/repo``) would normally
        route through ``_embed_hf_onnx``.  The resolver must intercept
        first, so a custom fine-tune that happens to share the naming
        convention does not download the HF model behind the caller's
        back.
        """
        stub = _StubEncoder(dim=6, tag=3.0)

        def resolver(name: str) -> Any:
            return stub if name == "me/private-lora-v5" else None

        register_encoder_resolver(resolver)

        vec = embed_query("q", "me/private-lora-v5")

        assert stub.calls == [["q"]]
        assert vec == [3.0] * 6

    def test_first_resolver_wins(self) -> None:
        """Registration order is priority order.  The first resolver
        that returns a non-None encoder is used; later resolvers never
        see the call.
        """
        first = _StubEncoder(dim=4, tag=1.0)
        second = _StubEncoder(dim=4, tag=2.0)

        def resolver_first(name):
            return first if name == "custom" else None

        def resolver_second(name):
            return second if name == "custom" else None

        register_encoder_resolver(resolver_first)
        register_encoder_resolver(resolver_second)

        vecs = embed_texts(["x"], "custom")

        assert first.calls == [["x"]]
        assert second.calls == []
        assert vecs == [[1.0] * 4]


# ------------------------------------------------------------------ #
# Fall-through behaviour                                               #
# ------------------------------------------------------------------ #


class TestFallThrough:
    def test_empty_texts_shortcircuits_before_resolver(self) -> None:
        """``embed_texts([])`` returns ``[]`` without invoking any
        resolver.  This keeps the empty-input contract cheap and avoids
        surprising resolver authors with empty-list calls.
        """
        stub = _StubEncoder()

        def resolver(name):
            return stub

        register_encoder_resolver(resolver)

        assert embed_texts([], "anything") == []
        assert stub.calls == []

    def test_no_resolver_match_falls_through_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every resolver returns None, the built-in ONNX path
        fires.  We monkeypatch ``_embed_onnx`` so the test does not
        need the fastembed model download.
        """
        called: list[tuple[list[str], str]] = []

        def fake_onnx(texts, model_name):
            called.append((list(texts), model_name))
            return [[0.0] * 3 for _ in texts]

        monkeypatch.setattr(embed_mod, "_embed_onnx", fake_onnx)
        monkeypatch.setattr(embed_mod, "_check_daemon", lambda: None)
        monkeypatch.setattr(embed_mod, "_resolve", lambda backend: "onnx")
        monkeypatch.setattr(embed_mod, "_is_gemma_model", lambda name: False)
        monkeypatch.setattr(embed_mod, "_is_hf_onnx_model", lambda name: False)

        def resolver(name):
            return None  # always defers

        register_encoder_resolver(resolver)

        embed_texts(["a", "b"], "BAAI/bge-small-en-v1.5")

        assert called == [(["a", "b"], "BAAI/bge-small-en-v1.5")]


# ------------------------------------------------------------------ #
# Fault isolation                                                      #
# ------------------------------------------------------------------ #


class TestResolverFaultIsolation:
    def test_raising_resolver_is_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A resolver that raises must not poison the registry -- later
        resolvers still get their turn.  One bad resolver should not
        silently disable custom encoders for the whole process.
        """
        good = _StubEncoder(dim=2, tag=9.0)

        def resolver_bad(name):
            raise RuntimeError("resolver is broken")

        def resolver_good(name):
            return good if name == "custom" else None

        register_encoder_resolver(resolver_bad)
        register_encoder_resolver(resolver_good)

        import logging

        with caplog.at_level(logging.ERROR, logger="vstash.embed"):
            vecs = embed_texts(["z"], "custom")

        assert vecs == [[9.0, 9.0]]
        assert any("resolver is broken" in rec.message or rec.exc_info for rec in caplog.records)


# ------------------------------------------------------------------ #
# get_embedding_dim integration                                        #
# ------------------------------------------------------------------ #


class TestGetEmbeddingDimIntegration:
    def test_custom_resolver_claims_unknown_name(self) -> None:
        """``get_embedding_dim`` must honour the resolver for names
        absent from ``KNOWN_DIMS`` -- otherwise callers cannot size a
        new ``VstashStore`` for a plugged-in encoder.
        """
        stub = _StubEncoder(dim=512)

        def resolver(name: str) -> Any:
            return stub if name == "my-lora" else None

        register_encoder_resolver(resolver)

        assert get_embedding_dim("my-lora") == 512

    def test_custom_resolver_overrides_known_dim(self) -> None:
        """If a resolver registers a custom ``my/fork`` it may carry a
        different dim than any built-in default; the resolver wins.
        """
        stub = _StubEncoder(dim=123)

        def resolver(name: str) -> Any:
            return stub if name == "BAAI/bge-small-en-v1.5" else None

        register_encoder_resolver(resolver)

        assert get_embedding_dim("BAAI/bge-small-en-v1.5") == 123

    def test_unknown_name_still_raises(self) -> None:
        """Without a resolver match, an unknown model still raises with
        the hint to register one or update ``KNOWN_DIMS``.
        """
        with pytest.raises(ValueError, match="register_encoder_resolver"):
            get_embedding_dim("totally-unknown-model-xyz")


# ------------------------------------------------------------------ #
# Output normalisation                                                 #
# ------------------------------------------------------------------ #


class TestOutputNormalisation:
    def test_numpy_output_is_converted_to_lists(self) -> None:
        """Encoders that return numpy arrays (the SentenceTransformer
        default) must end up as ``list[list[float]]`` so downstream
        ``_serialize`` in the store can handle them uniformly.
        """
        np = pytest.importorskip("numpy")  # noqa: F841

        stub = _NumpyStubEncoder(dim=4)

        def resolver(name: str) -> Any:
            return stub if name == "np-stub" else None

        register_encoder_resolver(resolver)

        vecs = embed_texts(["a", "b"], "np-stub")

        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)
        assert all(all(isinstance(x, float) for x in v) for v in vecs)
        assert vecs == [[2.0] * 4, [2.0] * 4]
