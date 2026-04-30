"""Tests for the vstash.services layer.

The services package was introduced in v0.37 to centralise the
embed -> search -> expand triplet that the CLI, MCP, web API, and
SDK previously inlined separately. These tests pin the new
contract: search_with_embedding validates inputs, embeds the
query, runs the hybrid pipeline, and optionally expands context.
ask_with_context delegates to the search service then sends the
chunks to chat.ask_full.

The tests use mock embeddings (the populated_store fixture) so
they run in <2s without depending on the FastEmbed daemon.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vstash.models import AskResult, SearchResult
from vstash.services import ask_with_context, search_with_embedding
from vstash.validation import (
    LimitError,
    QueryTooLongError,
    TopKOutOfRangeError,
)


@pytest.fixture
def patched_embed_query():
    """Replace embed_query with a deterministic constant vector.

    The populated_store fixture uses 384-dim mock embeddings; we
    return a vector that lives in the same space so the SQL search
    actually returns rows. The vector is the average of the two
    document clusters in populated_store, which puts vstash's
    relevance ranking in a determinate state.
    """
    with patch("vstash.services.search.embed_query") as mock:
        mock.return_value = [0.3] * 384
        yield mock


class TestSearchService:
    """search_with_embedding behaviour."""

    def test_returns_search_results(self, populated_store, sample_config, patched_embed_query):
        results = search_with_embedding(
            cfg=sample_config,
            store=populated_store,
            query="python programming",
            top_k=3,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_calls_embed_query_once(self, populated_store, sample_config, patched_embed_query):
        search_with_embedding(
            cfg=sample_config,
            store=populated_store,
            query="any question",
            top_k=3,
        )
        # The whole point of the service is to centralise this call.
        # If a future refactor accidentally embeds twice, this test
        # catches it before the daemon bill doubles.
        assert patched_embed_query.call_count == 1

    def test_uses_configured_embedding_model(
        self, populated_store, sample_config, patched_embed_query
    ):
        search_with_embedding(
            cfg=sample_config,
            store=populated_store,
            query="anything",
            top_k=1,
        )
        # Second positional arg of embed_query is the model name.
        called_model = patched_embed_query.call_args[0][1]
        assert called_model == sample_config.embeddings.model

    def test_expand_window_zero_skips_expansion(
        self, populated_store, sample_config, patched_embed_query
    ):
        # When window=0 the service must not call expand_context. We
        # verify by spying on the store's method.
        with patch.object(
            populated_store, "expand_context", wraps=populated_store.expand_context
        ) as spy:
            results = search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                expand_window=0,
            )
            spy.assert_not_called()
        assert len(results) > 0

    def test_expand_window_default_one_calls_expansion(
        self, populated_store, sample_config, patched_embed_query
    ):
        with patch.object(
            populated_store, "expand_context", wraps=populated_store.expand_context
        ) as spy:
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
            )
            spy.assert_called_once()
            # Default window is 1 per the service signature.
            assert spy.call_args.kwargs.get("window", 1) == 1

    def test_validates_query_too_long(self, populated_store, sample_config, patched_embed_query):
        # Build a query that exceeds the configured max_query_chars.
        too_long = "a" * (sample_config.limits.max_query_chars + 1)
        with pytest.raises(QueryTooLongError):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query=too_long,
                top_k=3,
            )
        # Validation must fire BEFORE embed_query, otherwise the
        # daemon does work for nothing and bills the user.
        patched_embed_query.assert_not_called()

    def test_validates_top_k_out_of_range(
        self, populated_store, sample_config, patched_embed_query
    ):
        with pytest.raises(TopKOutOfRangeError):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=sample_config.limits.max_top_k + 1,
            )
        patched_embed_query.assert_not_called()

    def test_validation_errors_inherit_limit_error(
        self, populated_store, sample_config, patched_embed_query
    ):
        # Sanity: callers can write `except LimitError` for any
        # validation failure. This is the contract the v0.37 errors
        # refactor pinned, but the services layer is the boundary
        # where it most commonly fires for end users.
        with pytest.raises(LimitError):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="x" * (sample_config.limits.max_query_chars + 1),
                top_k=3,
            )

    def test_non_hybrid_mode_drops_caller_weights(
        self, populated_store, sample_config, patched_embed_query
    ):
        # In vec_only / fts_only the store forces the weights
        # internally; the service must NOT forward caller weights or
        # the validator-allowed pair would conflict with the store's
        # forced (1.0, 0.0) / (0.0, 1.0).
        with patch.object(populated_store, "search", wraps=populated_store.search) as spy:
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                retrieval_mode="vec_only",
                vec_weight=0.7,
                fts_weight=0.3,
            )
            kwargs = spy.call_args.kwargs
            assert kwargs["retrieval_mode"] == "vec_only"
            assert kwargs["vec_weight"] is None
            assert kwargs["fts_weight"] is None

    def test_distance_cutoff_pinned_forwards_to_store(
        self, populated_store, sample_config, patched_embed_query
    ):
        # When the caller pins a cutoff, the service must forward
        # the same value to store.search rather than letting the
        # store fall back to its hardcoded default. Pin a value
        # well inside cfg.limits.max_distance_cutoff so validation
        # passes.
        with patch.object(populated_store, "search", wraps=populated_store.search) as spy:
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                distance_cutoff=0.9,
            )
            kwargs = spy.call_args.kwargs
            assert kwargs.get("distance_cutoff") == 0.9

    def test_distance_cutoff_none_does_not_forward(
        self, populated_store, sample_config, patched_embed_query
    ):
        # When the caller passes None (the default), the service must
        # NOT forward distance_cutoff to store.search -- the store
        # should apply its own default. Forwarding None would crash
        # the store (its parameter is typed `float`, not `float | None`).
        with patch.object(populated_store, "search", wraps=populated_store.search) as spy:
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
            )
            kwargs = spy.call_args.kwargs
            assert "distance_cutoff" not in kwargs

    def test_distance_cutoff_above_limit_raises(
        self, populated_store, sample_config, patched_embed_query
    ):
        # Caller-pinned cutoff above cfg.limits.max_distance_cutoff
        # is rejected at the boundary. This is the contract Gemini
        # asked for: validate the value the caller actually passed,
        # not the configured ceiling.
        from vstash.validation import DistanceCutoffOutOfRangeError

        with pytest.raises(DistanceCutoffOutOfRangeError):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                distance_cutoff=sample_config.limits.max_distance_cutoff + 1,
            )
        # The validation must fire before embed_query, same as the
        # other validation tests.
        patched_embed_query.assert_not_called()

    @pytest.mark.parametrize("bad_window", [-1, -100, "1", 1.5, True])
    def test_invalid_expand_window_raises_pre_embed(
        self, populated_store, sample_config, patched_embed_query, bad_window
    ):
        # `expand_window` is a cheap public knob; it should be
        # rejected before the daemon round-trip just like top_k.
        # bool is a tricky case because `isinstance(True, int)` is
        # True in Python; we have to special-case it.
        with pytest.raises(ValueError, match="expand_window"):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                expand_window=bad_window,
            )
        patched_embed_query.assert_not_called()

    @pytest.mark.parametrize("bad_mode", ["fuzzy", "VEC_ONLY", "", "hybrid_only"])
    def test_invalid_retrieval_mode_raises_pre_embed(
        self, populated_store, sample_config, patched_embed_query, bad_mode
    ):
        # retrieval_mode is the second cheap-to-validate public knob.
        # The Literal type alias does not enforce at runtime, so the
        # service has to.
        with pytest.raises(ValueError, match="retrieval_mode"):
            search_with_embedding(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
                retrieval_mode=bad_mode,
            )
        patched_embed_query.assert_not_called()


class TestAskService:
    """ask_with_context delegates to search service then chat.ask_full."""

    def test_returns_ask_result(self, populated_store, sample_config, patched_embed_query):
        # Mock the LLM call so the test does not need a backend.
        fake = AskResult(
            content="42",
            reasoning=None,
            usage=None,
            backend="cerebras",
            model="gpt-oss-120b",
        )
        with patch("vstash.services.ask._chat_ask_full", return_value=fake) as mock_ask:
            result = ask_with_context(
                cfg=sample_config,
                store=populated_store,
                query="what is the meaning of life",
                top_k=2,
            )
        assert isinstance(result, AskResult)
        assert result.content == "42"
        # The service should have called chat.ask_full exactly once,
        # with the retrieved chunks.
        assert mock_ask.call_count == 1
        # Arg shape is (query, chunks, cfg, history).
        args = mock_ask.call_args[0]
        assert args[0] == "what is the meaning of life"
        assert isinstance(args[1], list)  # chunks
        assert args[2] is sample_config

    def test_uses_search_service_for_retrieval(
        self, populated_store, sample_config, patched_embed_query
    ):
        # Confirm that ask delegates to the SAME search service, not
        # a copy-paste of its body. If a future refactor splits them,
        # this test forces a second mock to keep them in sync.
        fake = AskResult(content="ok", backend="cerebras", model="x")
        with (
            patch("vstash.services.ask._chat_ask_full", return_value=fake),
            patch(
                "vstash.services.ask.search_with_embedding",
                wraps=__import__(
                    "vstash.services.search", fromlist=["search_with_embedding"]
                ).search_with_embedding,
            ) as spy_search,
        ):
            ask_with_context(
                cfg=sample_config,
                store=populated_store,
                query="anything",
                top_k=2,
            )
            spy_search.assert_called_once()

    def test_propagates_validation_error(self, populated_store, sample_config, patched_embed_query):
        # Validation lives in the search service; ask must not
        # swallow it (otherwise an over-long query would silently
        # become a chat call with no context).
        with patch("vstash.services.ask._chat_ask_full") as mock_ask:
            with pytest.raises(QueryTooLongError):
                ask_with_context(
                    cfg=sample_config,
                    store=populated_store,
                    query="x" * (sample_config.limits.max_query_chars + 1),
                    top_k=2,
                )
            mock_ask.assert_not_called()
