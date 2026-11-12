"""Built-in Jina v5 text backend (``jinaai/jina-embeddings-v5-text-nano-retrieval``).

The model is asymmetric (``Query: `` / ``Document: `` prompts) and its ONNX
export ignores ``attention_mask`` for padding, so the backend must (a) prefix
queries and documents differently and (b) only batch texts of identical token
length. Both are checked here against a fake session -- no download.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vstash import embed as E

MODEL = "jinaai/jina-embeddings-v5-text-nano-retrieval"


class _Enc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """One token per whitespace-separated word; id = word length."""

    def encode_batch(self, texts: list[str]) -> list[_Enc]:
        return [_Enc([len(w) for w in t.split()]) for t in texts]


class _FakeSession:
    """Records every run; embeds a row as [len, sum(ids), 1, 0, ...]."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def run(self, outputs, feeds):
        assert outputs == ["sentence_embedding"]
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        self.calls.append((ids, mask))
        out = np.zeros((ids.shape[0], 768), dtype=np.float32)
        out[:, 0] = ids.shape[1]
        out[:, 1] = ids.sum(axis=1)
        out[:, 2] = 1.0
        return [out]


@pytest.fixture
def fake():
    session, tokenizer = _FakeSession(), _FakeTokenizer()
    seen: list[list[str]] = []
    real_encode = tokenizer.encode_batch

    def spy(texts):
        seen.append(list(texts))
        return real_encode(texts)

    tokenizer.encode_batch = spy  # type: ignore[method-assign]
    with patch.object(E, "_init_jina_v5", return_value=(session, tokenizer)):
        yield session, seen


def _no_daemon():
    return patch.object(E, "_check_daemon", return_value=None)


class TestRegistration:
    def test_known_dim(self) -> None:
        assert E.get_embedding_dim(MODEL) == 768

    def test_routing(self) -> None:
        assert E._resolve_builtin_backend(MODEL).name == "jina_v5"

    def test_revision_is_pinned(self) -> None:
        rev = E._JINA_V5_MODELS[MODEL]
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)

    def test_model_dir_override(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("VSTASH_MODELS_DIR", str(tmp_path))
        d = E._local_model_dir(MODEL, E._JINA_V5_MODELS[MODEL])
        assert d.startswith(str(tmp_path))
        assert "jinaai--jina-embeddings-v5-text-nano-retrieval@" in d


class TestPrompts:
    def test_query_gets_query_prefix(self, fake) -> None:
        _, seen = fake
        with _no_daemon():
            E.embed_query("hola mundo", MODEL)
        assert seen == [["Query: hola mundo"]]

    def test_documents_get_document_prefix(self, fake) -> None:
        _, seen = fake
        with _no_daemon():
            E.embed_texts(["uno", "dos tres"], MODEL)
        assert seen == [["Document: uno", "Document: dos tres"]]


class TestNoPadding:
    def test_batches_only_equal_lengths_and_never_masks(self, fake) -> None:
        session, _ = fake
        texts = ["a", "b c d", "e", "f g h", "i j"]
        with _no_daemon():
            vecs = E.embed_texts(texts, MODEL)
        for ids, mask in session.calls:
            assert mask.shape == ids.shape
            assert mask.min() == 1  # no padded positions
        # 3 distinct token lengths (with the prefix): 2, 3, 4 tokens
        assert sorted(ids.shape[1] for ids, _ in session.calls) == [2, 3, 4]
        # Output order follows the input order, not the length grouping:
        # component 0 / component 2 recovers each row's token length.
        expected = [len(("Document: " + t).split()) for t in texts]
        assert len(vecs) == len(texts)
        for v, n in zip(vecs, expected, strict=True):
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
            assert v[0] / v[2] == pytest.approx(n)

    def test_large_group_is_chunked(self, fake, monkeypatch) -> None:
        session, _ = fake
        monkeypatch.setattr(E, "_JINA_V5_BATCH_SIZE", 2)
        with _no_daemon():
            E.embed_texts(["x"] * 5, MODEL)
        assert [ids.shape[0] for ids, _ in session.calls] == [2, 2, 1]

    def test_empty(self, fake) -> None:
        session, _ = fake
        assert E.embed_texts([], MODEL) == []
        assert session.calls == []


@pytest.mark.benchmark
class TestRealModel:
    """Downloads ~850 MB on first run; deselected by default (``-m benchmark``)."""

    def test_spanish_relevance_and_batch_parity(self) -> None:
        docs = [
            "Kakesu guarda las decisiones como registros separados en su colección de vstash.",
            "El clima en Bogotá hoy es lluvioso, con 14 grados.",
            "La receta de arepas lleva harina de maíz precocida, agua y sal.",
        ]
        with _no_daemon():
            batch = np.array(E.embed_texts(docs, MODEL))
            single = np.array([E.embed_texts([d], MODEL)[0] for d in docs])
            q = np.array(E.embed_query("¿dónde guarda Kakesu sus decisiones?", MODEL))
        np.testing.assert_allclose(batch, single, atol=1e-5)
        sims = batch @ q
        assert int(np.argmax(sims)) == 0
        assert sims[0] - sims[1:].max() > 0.3
