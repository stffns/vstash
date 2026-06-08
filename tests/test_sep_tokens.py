"""The chunking helpers must not depend on a prior _token_count having set the
_SEPARATOR_TOKENS global as a side effect.

_SEPARATOR_TOKENS was only populated inside _get_enc(), and the helpers read it
directly. In the live call paths _token_count happens to run first, but that is
a fragile implicit ordering. _sep_tokens() makes the helpers self-contained:
they load the encoder themselves. Each test monkeypatches _token_count so it
does NOT load the encoder, so without _sep_tokens() the separator math hits
``None + int`` -> TypeError.
"""

from __future__ import annotations

import vstash.ingest as ing


def test_sep_tokens_loads_encoder(monkeypatch) -> None:
    monkeypatch.setattr(ing, "_enc", None)
    monkeypatch.setattr(ing, "_SEPARATOR_TOKENS", None)
    n = ing._sep_tokens()
    assert isinstance(n, int) and n > 0


def test_split_by_paragraphs_self_contained(monkeypatch) -> None:
    monkeypatch.setattr(ing, "_enc", None)
    monkeypatch.setattr(ing, "_SEPARATOR_TOKENS", None)
    # A _token_count that never loads the encoder, so only _sep_tokens() can.
    monkeypatch.setattr(ing, "_token_count", lambda text: len(text.split()))
    result = ing._split_by_paragraphs("alpha beta\n\ngamma delta", chunk_size=100, overlap=0)
    assert result


def test_merge_small_chunks_self_contained(monkeypatch) -> None:
    monkeypatch.setattr(ing, "_enc", None)
    monkeypatch.setattr(ing, "_SEPARATOR_TOKENS", None)
    monkeypatch.setattr(ing, "_token_count", lambda text: len(text.split()))
    result = ing._merge_small_chunks(["one two", "three four"], chunk_size=100)
    assert result
