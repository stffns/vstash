"""Tests for vstash.watch — _DebounceTimer and _should_process."""

from __future__ import annotations

import threading
import time

from vstash.config import SUPPORTED_EXTENSIONS
from vstash.watch import _DebounceTimer, _should_process


class TestDebounceTimer:
    """Test _DebounceTimer fires callback after delay with debounce semantics."""

    def test_trigger_fires_callback_after_delay(self) -> None:
        """Callback is invoked once after the debounce delay."""
        result: list[str] = []
        event = threading.Event()

        def callback(path: str) -> None:
            result.append(path)
            event.set()

        timer = _DebounceTimer(delay=0.1)
        timer.trigger("/tmp/test.md", callback)

        event.wait(timeout=2.0)
        assert result == ["/tmp/test.md"]

    def test_cancel_prevents_pending_fire(self) -> None:
        """cancel() stops a pending timer so its callback never fires.

        This is the watch delete-race fix: a delete cancels a pending
        create/modify debounce so the stale timer cannot re-enqueue the file
        after it was removed from the store.
        """
        fired: list[str] = []
        timer = _DebounceTimer(delay=0.05)
        timer.trigger("/tmp/a.md", fired.append)
        timer.cancel("/tmp/a.md")
        time.sleep(0.12)  # well past the delay
        assert fired == []
        assert "/tmp/a.md" not in timer._timers

    def test_cancel_unknown_path_is_noop(self) -> None:
        timer = _DebounceTimer(delay=0.1)
        timer.cancel("/tmp/nonexistent.md")  # must not raise

    def test_cancel_prefix_cancels_files_under_dir(self) -> None:
        """A directory delete cancels pending debounces for files inside it,
        but leaves a sibling whose path merely shares the prefix string."""
        fired: list[str] = []
        timer = _DebounceTimer(delay=0.05)
        timer.trigger("/tmp/proj/a.md", fired.append)
        timer.trigger("/tmp/proj/sub/b.md", fired.append)
        timer.trigger("/tmp/project/c.md", fired.append)  # sibling, NOT under /tmp/proj
        timer.cancel_prefix("/tmp/proj")
        time.sleep(0.12)
        assert fired == ["/tmp/project/c.md"]
        assert "/tmp/project/c.md" not in timer._timers  # fired + cleaned up

    def test_re_trigger_resets_timer(self) -> None:
        """A second trigger for the same path resets the delay; callback fires once."""
        result: list[str] = []
        event = threading.Event()

        def callback(path: str) -> None:
            result.append(path)
            event.set()

        timer = _DebounceTimer(delay=0.2)
        timer.trigger("/tmp/test.md", callback)

        # Re-trigger before first one fires
        time.sleep(0.05)
        timer.trigger("/tmp/test.md", callback)

        event.wait(timeout=2.0)
        # Only one invocation should have occurred
        assert result == ["/tmp/test.md"]

    def test_cancel_all_stops_pending_timers(self) -> None:
        """cancel_all prevents pending callbacks from firing."""
        result: list[str] = []

        def callback(path: str) -> None:
            result.append(path)

        timer = _DebounceTimer(delay=0.3)
        timer.trigger("/tmp/a.md", callback)
        timer.trigger("/tmp/b.md", callback)

        timer.cancel_all()
        time.sleep(0.5)

        assert result == []

    def test_multiple_paths_fire_independently(self) -> None:
        """Triggers for different paths fire independently."""
        result: list[str] = []
        event = threading.Event()

        def callback(path: str) -> None:
            result.append(path)
            if len(result) >= 2:
                event.set()

        timer = _DebounceTimer(delay=0.1)
        timer.trigger("/tmp/a.md", callback)
        timer.trigger("/tmp/b.md", callback)

        event.wait(timeout=2.0)
        assert sorted(result) == ["/tmp/a.md", "/tmp/b.md"]


class TestShouldProcess:
    """Test _should_process file-filtering logic."""

    def test_supported_extension_returns_true(self) -> None:
        """Files with supported extensions pass the filter."""
        assert _should_process("/home/user/docs/readme.md", SUPPORTED_EXTENSIONS) is True
        assert _should_process("/home/user/code/app.py", SUPPORTED_EXTENSIONS) is True
        assert _should_process("/home/user/data/report.pdf", SUPPORTED_EXTENSIONS) is True

    def test_hidden_file_returns_false(self) -> None:
        """Hidden files (dotfiles) are rejected."""
        assert _should_process("/home/user/.hidden/file.md", SUPPORTED_EXTENSIONS) is False
        assert _should_process("/home/user/docs/.secret.md", SUPPORTED_EXTENSIONS) is False

    def test_unsupported_extension_returns_false(self) -> None:
        """Files with unsupported extensions are rejected."""
        assert _should_process("/home/user/image.png", SUPPORTED_EXTENSIONS) is False
        assert _should_process("/home/user/data.zip", SUPPORTED_EXTENSIONS) is False
        assert _should_process("/home/user/binary.exe", SUPPORTED_EXTENSIONS) is False

    def test_case_insensitive_extension(self) -> None:
        """Extension matching is case-insensitive."""
        assert _should_process("/home/user/README.MD", SUPPORTED_EXTENSIONS) is True
        assert _should_process("/home/user/script.PY", SUPPORTED_EXTENSIONS) is True

    def test_no_extension_returns_false(self) -> None:
        """Files without an extension are rejected."""
        assert _should_process("/home/user/Makefile", SUPPORTED_EXTENSIONS) is False
