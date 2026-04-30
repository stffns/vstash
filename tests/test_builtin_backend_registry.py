"""Tests for the built-in backend registry in :mod:`vstash.embed`.

The registry was introduced in v0.37 to replace the hardcoded
if/elif chain in ``embed_texts`` / ``embed_query`` / ``warmup``.
These tests pin the contract:

- The registry is a non-empty list of ``_BuiltinBackend`` entries.
- The fallback (``name == "default"``) is always last and its
  matcher returns ``True`` for any model name.
- ``_resolve_builtin_backend`` walks the list in order and returns
  the first match.
- ``embed_texts`` / ``embed_query`` / ``warmup`` all dispatch through
  the registry rather than re-implementing the if/elif chain.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vstash.embed import (
    _BUILTIN_BACKENDS,
    _BuiltinBackend,
    _resolve_builtin_backend,
)


class TestRegistryShape:
    """The registry is the contract; pin its structural invariants."""

    def test_registry_is_non_empty(self) -> None:
        assert len(_BUILTIN_BACKENDS) > 0

    def test_every_entry_is_builtin_backend(self) -> None:
        for entry in _BUILTIN_BACKENDS:
            assert isinstance(entry, _BuiltinBackend)

    def test_fallback_is_last_and_always_matches(self) -> None:
        # The fallback contract is what makes _resolve_builtin_backend
        # never raise. If the fallback is ever moved out of last
        # position, dispatch becomes order-dependent in a way that
        # breaks the catch-all guarantee.
        last = _BUILTIN_BACKENDS[-1]
        assert last.name == "default"
        assert last.matches("anything")
        assert last.matches("")
        assert last.matches("BAAI/bge-small-en-v1.5")
        assert last.matches("google/embeddinggemma-300m")

    def test_no_duplicate_names(self) -> None:
        # Names are used in tests and could be used as registry keys
        # in a future version. Duplicates would be a bug regardless.
        names = [b.name for b in _BUILTIN_BACKENDS]
        assert len(names) == len(set(names))

    def test_every_entry_exposes_three_callables(self) -> None:
        # Every backend must implement all three dispatch surfaces
        # so the registry walker never needs to None-check.
        for entry in _BUILTIN_BACKENDS:
            assert callable(entry.matches)
            assert callable(entry.embed_batch)
            assert callable(entry.embed_query)
            assert callable(entry.warmup)


class TestResolverDispatch:
    """`_resolve_builtin_backend` returns the first matching entry."""

    def test_gemma_model_routes_to_gemma_backend(self) -> None:
        backend = _resolve_builtin_backend("google/embeddinggemma-300m")
        assert backend.name == "gemma"

    def test_unknown_model_routes_to_default(self) -> None:
        # Anything that does not match a specific matcher must fall
        # through to the catch-all rather than raise.
        backend = _resolve_builtin_backend("not-a-real-model-name")
        assert backend.name == "default"

    def test_empty_string_routes_to_default(self) -> None:
        # Defensive: the matchers should not crash on empty string.
        # Empty matches no specific backend but must hit the fallback.
        backend = _resolve_builtin_backend("")
        assert backend.name == "default"

    def test_dispatch_order_first_match_wins(self) -> None:
        # If two matchers would both accept a name (currently they do
        # not, but the contract should hold), the first registered
        # entry wins. We verify by inspecting the iteration order.
        from vstash.embed import _is_gemma_model, _is_hf_onnx_model

        # Find each named entry's index.
        names = [b.name for b in _BUILTIN_BACKENDS]
        gemma_idx = names.index("gemma")
        hf_idx = names.index("hf_onnx")
        default_idx = names.index("default")

        # Strict ordering: gemma is the most specific matcher and
        # must run first; hf_onnx is the next most specific; default
        # is always last. Loosening any of these ordering relations
        # would either let a broader matcher steal a gemma name or
        # break the catch-all guarantee.
        assert gemma_idx < hf_idx
        assert hf_idx < default_idx
        assert default_idx == len(names) - 1

        # Sanity: matchers are pure functions of model_name and are
        # mutually exclusive on the model names this project ships.
        assert _is_gemma_model("google/embeddinggemma-300m")
        assert not _is_hf_onnx_model("google/embeddinggemma-300m")

    def test_first_match_wins_when_two_matchers_accept(self) -> None:
        # Direct test of first-match-wins semantics: insert a fake
        # broad matcher that also accepts a gemma name and verify
        # the gemma backend still wins because of registration
        # order. This catches regressions where a future contributor
        # inserts a broader matcher ahead of a specific one.
        from vstash.embed import _BUILTIN_BACKENDS as registry, _resolve_builtin_backend

        spy_calls: list[str] = []

        def _spy_matcher(name: str) -> bool:
            spy_calls.append(name)
            return True  # would steal everything if placed first

        spy_entry = _BuiltinBackend(
            name="spy_broad",
            matches=_spy_matcher,
            embed_batch=lambda *a, **kw: [],
            embed_query=lambda *a, **kw: [],
            warmup=lambda *a, **kw: None,
        )

        # Insert AFTER gemma -- gemma should still win for a gemma name.
        registry.insert(1, spy_entry)
        try:
            backend = _resolve_builtin_backend("google/embeddinggemma-300m")
            assert backend.name == "gemma", (
                "first-match-wins is broken: spy entry stole a gemma name"
            )
            # The spy should never have been consulted because gemma
            # matched first and the loop short-circuited.
            assert spy_calls == []
        finally:
            registry.remove(spy_entry)


class TestPublicDispatch:
    """`embed_texts` / `embed_query` / `warmup` all walk the registry."""

    def test_embed_texts_dispatches_through_registry(self) -> None:
        # Patch the resolver and assert embed_texts calls it. The
        # body returns whatever the matched backend's embed_batch
        # returns -- here a sentinel list.
        from vstash.embed import embed_texts

        sentinel = [[1.0, 2.0]]
        fake_backend = _BuiltinBackend(
            name="test",
            matches=lambda _: True,
            embed_batch=lambda *a, **kw: sentinel,
            embed_query=lambda *a, **kw: [0.0],
            warmup=lambda *a, **kw: None,
        )
        with (
            patch("vstash.embed._resolve_builtin_backend", return_value=fake_backend),
            patch("vstash.embed._resolve_custom_encoder", return_value=None),
            patch("vstash.embed._check_daemon", return_value=None),
        ):
            result = embed_texts(["hello"], "any-model")
        assert result is sentinel

    def test_embed_query_dispatches_through_registry(self) -> None:
        from vstash.embed import embed_query

        sentinel = [3.14, 2.71]
        fake_backend = _BuiltinBackend(
            name="test",
            matches=lambda _: True,
            embed_batch=lambda *a, **kw: [[0.0]],
            embed_query=lambda *a, **kw: sentinel,
            warmup=lambda *a, **kw: None,
        )
        with (
            patch("vstash.embed._resolve_builtin_backend", return_value=fake_backend),
            patch("vstash.embed._resolve_custom_encoder", return_value=None),
            patch("vstash.embed._check_daemon", return_value=None),
        ):
            result = embed_query("hello", "any-model")
        assert result is sentinel

    def test_warmup_dispatches_through_registry(self) -> None:
        from vstash.embed import warmup

        warmup_calls: list[tuple] = []

        def _fake_warmup(name: str, backend: str) -> None:
            warmup_calls.append((name, backend))

        fake_backend = _BuiltinBackend(
            name="test",
            matches=lambda _: True,
            embed_batch=lambda *a, **kw: [[0.0]],
            embed_query=lambda *a, **kw: [0.0],
            warmup=_fake_warmup,
        )
        with patch("vstash.embed._resolve_builtin_backend", return_value=fake_backend):
            warmup("any-model", backend="onnx")
        assert warmup_calls == [("any-model", "onnx")]

    def test_warmup_short_circuits_on_custom_resolver(self) -> None:
        # warmup() must mirror embed_texts/embed_query: if a custom
        # encoder resolver claims the model name, the built-in
        # warmup chain must NOT run -- otherwise we warm onnx/mlx
        # against a name they do not recognise. Regression test for
        # a bug introduced when warmup migrated to the registry.
        from vstash.embed import (
            register_encoder_resolver,
            unregister_encoder_resolver,
            warmup,
        )

        class _FakeEncoder:
            embedding_dim = 2

            def encode(self, texts):
                return [[0.0, 0.0] for _ in texts]

        encoder = _FakeEncoder()

        def resolver(name: str):
            return encoder if name == "custom-warmup-test" else None

        register_encoder_resolver(resolver)
        try:
            with patch(
                "vstash.embed._resolve_builtin_backend",
                side_effect=AssertionError("built-in warmup should not run for custom name"),
            ):
                # Should not raise: the custom resolver short-circuits
                # before _resolve_builtin_backend is ever called.
                warmup("custom-warmup-test", backend="onnx")
        finally:
            unregister_encoder_resolver(resolver)

    def test_custom_encoder_resolver_wins_over_builtins(self) -> None:
        # Ensure the registry change did not regress the public
        # extension hook: a registered custom resolver still runs
        # before any built-in dispatch happens.
        from vstash.embed import (
            embed_texts,
            register_encoder_resolver,
            unregister_encoder_resolver,
        )

        class _FakeEncoder:
            embedding_dim = 2

            def encode(self, texts):
                return [[1.0, 2.0] for _ in texts]

        encoder = _FakeEncoder()

        def resolver(name: str):
            return encoder if name == "claimed-by-custom" else None

        register_encoder_resolver(resolver)
        try:
            with (
                patch(
                    "vstash.embed._resolve_builtin_backend",
                    side_effect=AssertionError("built-in dispatch should not run"),
                ),
                patch("vstash.embed._check_daemon", return_value=None),
            ):
                result = embed_texts(["x", "y"], "claimed-by-custom")
            assert result == [[1.0, 2.0], [1.0, 2.0]]
        finally:
            unregister_encoder_resolver(resolver)


class TestRegressionsAgainstOldDispatch:
    """The new dispatch must produce the same routing as the old chain.

    This locks the v0.36 -> v0.37 migration: gemma names go to the
    gemma backend, the rest go through the onnx/mlx fallback. If a
    future contributor accidentally swaps the registry order or
    registers a new backend with too-loose a matcher, these tests
    catch the regression before BEIR results drift.
    """

    @pytest.mark.parametrize(
        ("model_name", "expected_name"),
        [
            ("google/embeddinggemma-300m", "gemma"),
            # bge models go through the default (onnx/mlx) backend,
            # not hf_onnx -- the hf_onnx heuristic is reserved for the
            # short-lived HF ONNX adapter we shipped in v0.32.
            ("BAAI/bge-small-en-v1.5", "default"),
            ("BAAI/bge-base-en-v1.5", "default"),
            ("nomic-ai/nomic-embed-text-v1.5", "default"),
            ("Stffens/bge-small-rrf-v3", "default"),
            ("intfloat/multilingual-e5-small", "default"),
        ],
    )
    def test_known_model_routing(self, model_name: str, expected_name: str) -> None:
        backend = _resolve_builtin_backend(model_name)
        assert backend.name == expected_name, (
            f"{model_name} resolved to {backend.name!r}, expected {expected_name!r}"
        )
