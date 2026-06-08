"""Regression test for _promote_candidate cross-device rollback.

shutil.move across filesystems is copy-then-delete, so a mid-copy failure leaves
final_path as a partial directory. The rollback must remove that partial and
restore the previous model from backup instead of stranding it in .old.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vstash.retrain import _promote_candidate


def test_promote_rolls_back_partial_cross_device_move(tmp_path: Path, monkeypatch) -> None:
    final = tmp_path / "model"
    final.mkdir()
    (final / "weights.txt").write_text("ORIGINAL")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "weights.txt").write_text("NEW")

    def fake_move(src: str, dst: str) -> None:
        # Simulate a cross-device copy that fails partway, leaving a partial dst.
        d = Path(dst)
        d.mkdir(exist_ok=True)
        (d / "weights.txt.partial").write_text("HALF")
        raise OSError("cross-device link failed mid-copy")

    monkeypatch.setattr("vstash.retrain.shutil.move", fake_move)

    with pytest.raises(OSError, match="cross-device"):
        _promote_candidate(candidate, final)

    # The previous model must be restored intact, not left as the partial move,
    # and the .old backup must be gone (rolled back into place).
    assert final.exists()
    assert (final / "weights.txt").read_text() == "ORIGINAL"
    assert not (final / "weights.txt.partial").exists()
    assert not final.with_name("model.old").exists()


def test_promote_first_time_train_cleans_partial(tmp_path: Path, monkeypatch) -> None:
    """No prior model: a failed move must not leave a partial final behind, and
    must not crash trying to restore a nonexistent backup."""
    final = tmp_path / "model"  # does not exist (first train)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "weights.txt").write_text("NEW")

    def fake_move(src: str, dst: str) -> None:
        Path(dst).mkdir(exist_ok=True)
        raise OSError("cross-device link failed mid-copy")

    monkeypatch.setattr("vstash.retrain.shutil.move", fake_move)

    with pytest.raises(OSError, match="cross-device"):
        _promote_candidate(candidate, final)
    # No backup existed, so there's nothing to restore; the rollback must not
    # raise a second (masking) exception, and the partial garbage is removed.
    assert not final.exists()
    assert not final.with_name("model.old").exists()
