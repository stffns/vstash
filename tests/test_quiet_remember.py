"""Programmatic ingest stays off stderr: no Rich "Embedding" bar on remember.

``ingest_text`` drew a progress bar for every call, so each SDK / MCP /
journal ``remember`` printed one to stderr with no way to turn it off.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import requires_sqlite_vec
from vstash import Memory
from vstash.config import VstashConfig
from vstash.ingest import ingest_text
from vstash.store import VstashStore

DIM = 384


def _fake_embed(texts, _model):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


@pytest.fixture
def fake_embed():
    with patch("vstash.ingest.embed_texts", side_effect=_fake_embed):
        yield


@requires_sqlite_vec
class TestQuietRemember:
    def test_memory_remember_prints_no_bar(self, tmp_path: Path, fake_embed, capsys) -> None:
        mem = Memory(db=tmp_path / "q.db")
        try:
            mem.remember("una nota corta para la memoria", title="nota")
        finally:
            mem.close()
        assert "Embedding" not in capsys.readouterr().err

    def test_ingest_text_keeps_bar_by_default(self, tmp_path: Path, fake_embed, capsys) -> None:
        store = VstashStore(str(tmp_path / "b.db"), embedding_dim=DIM)
        try:
            ingest_text("un texto de prueba algo más largo para la memoria", VstashConfig(), store, title="b")
        finally:
            store.close()
        assert "Embedding" in capsys.readouterr().err

    def test_ingest_text_quiet(self, tmp_path: Path, fake_embed, capsys) -> None:
        store = VstashStore(str(tmp_path / "c.db"), embedding_dim=DIM)
        try:
            res = ingest_text("un texto de prueba algo más largo para la memoria", VstashConfig(), store, title="c", quiet=True)
        finally:
            store.close()
        assert res.status == "ok"
        assert "Embedding" not in capsys.readouterr().err
