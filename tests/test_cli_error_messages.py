"""Regression tests for CLI error message rendering.

These exist because of an e2e bug found against the v0.25.0 wheel on
PyPI: error messages containing ``pip install vstash[ingest]`` were
silently truncated to ``pip install vstash`` because rich's console
markup parser interpreted ``[ingest]`` as an opening tag.
"""

from __future__ import annotations

from rich.console import Console

from vstash.cli import _safe_exc


class TestSafeExc:
    def test_brackets_are_escaped(self) -> None:
        """``[ingest]`` must survive a rich console render."""
        exc = ImportError(
            "File ingestion requires markitdown. Install it with: pip install vstash[ingest]"
        )
        c = Console(record=True, force_terminal=False, color_system=None, width=200)
        c.print(f"err: {_safe_exc(exc)}")
        out = c.export_text()
        assert "vstash[ingest]" in out
        # And the literal "[ingest]" must be preserved as text, not eaten.
        assert "[ingest]" in out

    def test_multiple_brackets(self) -> None:
        """A message with two extras both survive."""
        exc = RuntimeError("try pip install vstash[serve] or vstash[ingest]")
        c = Console(record=True, force_terminal=False, color_system=None, width=200)
        c.print(f"err: {_safe_exc(exc)}")
        out = c.export_text()
        assert "vstash[serve]" in out
        assert "vstash[ingest]" in out

    def test_plain_message_unchanged(self) -> None:
        """Messages without markup are unchanged byte-for-byte."""
        exc = ValueError("plain error message")
        assert _safe_exc(exc) == "plain error message"
