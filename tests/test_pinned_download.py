"""ONNX models with external data are fetched as real files into a pinned dir.

onnxruntime >= 1.31 rejects ``model.onnx_data`` reached through the HF cache's
``snapshots/ -> blobs/`` symlinks ("External data path escapes model
directory"), which broke EmbeddingGemma. ``_download_pinned`` must pass
``local_dir`` and a pinned ``revision`` for every file.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vstash import embed as E


def test_download_pinned_uses_local_dir_and_revision(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VSTASH_MODELS_DIR", str(tmp_path))
    with patch("huggingface_hub.hf_hub_download") as dl:
        out = E._download_pinned("org/model", "a" * 40, ("x.onnx", "x.onnx_data"))
    assert out == str(tmp_path / ("org--model@" + "a" * 12))
    assert dl.call_count == 2
    for call in dl.call_args_list:
        assert call.kwargs["revision"] == "a" * 40
        assert call.kwargs["local_dir"] == out


def test_gemma_loads_from_pinned_local_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VSTASH_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(E, "_gemma_session", None)
    monkeypatch.setattr(E, "_gemma_tokenizer", None)
    session = MagicMock()
    with (
        patch.object(E, "_download_pinned", return_value=str(tmp_path)) as dl,
        patch("onnxruntime.InferenceSession", return_value=session) as sess,
        patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()),
    ):
        E._init_gemma()
    repo, revision, files = dl.call_args.args
    assert repo == E._GEMMA_ONNX_REPO
    assert revision == E._GEMMA_ONNX_REVISION
    assert "onnx/model.onnx_data" in files
    assert sess.call_args.args[0] == str(tmp_path / "onnx" / "model.onnx")
