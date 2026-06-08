"""
watch.py — File system watcher for auto-ingestion.

Monitors directories for file changes and automatically ingests
modified or created files into vstash memory.

Requires: watchdog >= 4.0.0  (install via `pip install vstash[watch]`)
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape as _rich_escape

from .config import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from .config import VstashConfig
    from .store import VstashStore

console = Console(stderr=True)


class _DebounceTimer:
    """Debounce rapid file-change events into a single ingestion call.

    Fires the callback after *delay* seconds of inactivity per path.
    Automatically cleans up timer entries after they fire.

    Args:
        delay: Seconds to wait before firing the callback.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def trigger(self, path: str, callback: Callable[[str], None]) -> None:
        """Schedule or reschedule ingestion for *path*.

        Args:
            path: Absolute file path that changed.
            callback: Callable that accepts a single ``str`` path argument.
        """
        with self._lock:
            existing = self._timers.get(path)
            if existing is not None:
                existing.cancel()

            def _fire_and_cleanup() -> None:
                try:
                    callback(path)
                finally:
                    with self._lock:
                        current = self._timers.get(path)
                        if current is timer:
                            self._timers.pop(path, None)

            timer = threading.Timer(self._delay, _fire_and_cleanup)
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    def cancel(self, path: str) -> None:
        """Cancel a pending timer for a single path, if any.

        Called when a delete event arrives for a path that has a pending
        create/modify debounce: without this, that timer would fire *after* the
        delete and re-enqueue a path that was just removed from the store,
        reordering the delete behind a stale ingest (e.g. an editor's
        delete-then-recreate atomic save).
        """
        with self._lock:
            timer = self._timers.pop(path, None)
        if timer is not None:
            timer.cancel()

    def cancel_prefix(self, prefix: str) -> None:
        """Cancel all pending timers for paths under directory ``prefix``.

        Used when a directory is deleted: any pending create/modify debounces
        for files inside it would otherwise fire and try to re-enqueue files
        that were just removed with the directory.
        """
        prefix_path = Path(prefix)
        with self._lock:
            matched = [p for p in self._timers if Path(p).is_relative_to(prefix_path)]
            timers = [self._timers.pop(p) for p in matched]
        for timer in timers:
            timer.cancel()

    def cancel_all(self) -> None:
        """Cancel all pending timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


def _should_process(path: str, extensions: frozenset[str]) -> bool:
    """Check if a file path should be ingested.

    Args:
        path: Absolute file path.
        extensions: Set of allowed file extensions (with leading dot).

    Returns:
        True if the file should be processed.
    """
    p = Path(path)
    # Skip hidden files/directories
    if any(part.startswith(".") for part in p.parts):
        return False
    # Check extension
    return p.suffix.lower() in extensions


# Burst-drain tuning. The watcher blocks for one queued file, then
# tries to accumulate a small burst before flushing through the batch
# ingest route. Both bounds are defensive:
#   - DRAIN_MAX_BATCH caps peak memory: a burst of 100k files would
#     otherwise hold every path + embedding in Python land before the
#     single batch transaction commits.
#   - DRAIN_MAX_WINDOW_S caps the tail latency a user sees: even if
#     more files keep arriving, we flush what we have so the first
#     dropped file does not wait indefinitely.
DRAIN_MAX_BATCH = 64
DRAIN_MAX_WINDOW_S = 0.25


def _drain_burst(
    q: queue.Queue[str | None],
    first: str,
    *,
    max_batch: int = DRAIN_MAX_BATCH,
    max_window_s: float = DRAIN_MAX_WINDOW_S,
    now: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Collect ``first`` plus any extra queued paths up to the window.

    Returns an ordered list of paths ready to feed to ``ingest_batch``.
    If a poison-pill (``None``) is drained mid-burst, it is re-enqueued
    so the main loop still sees it and exits cleanly.

    Exposed top-level so the drain semantics are unit-testable without
    spinning up watchdog or a real ``start_watch``.
    """
    batch: list[str] = [first]
    deadline = now() + max_window_s
    while len(batch) < max_batch:
        remaining = deadline - now()
        if remaining <= 0:
            break
        try:
            extra = q.get(timeout=remaining)
        except queue.Empty:
            break
        if extra is None:
            q.put(None)
            break
        batch.append(extra)
    return batch


def start_watch(
    paths: list[str],
    cfg: VstashConfig,
    store: VstashStore,
    *,
    collection: str = "default",
    extensions: frozenset[str] | None = None,
    debounce_s: float = 2.0,
) -> None:
    """Watch directories for file changes and auto-ingest.

    Blocks until interrupted with Ctrl+C.  Ingestion is serialised
    through a single worker thread to avoid concurrent SQLite writes.

    Args:
        paths: List of directory paths to watch.
        cfg: vstash configuration.
        store: Vector store instance.
        collection: Collection to tag ingested files.
        extensions: File extensions to watch (default: all supported).
        debounce_s: Seconds to wait after last change before ingesting.

    Raises:
        ImportError: If ``watchdog`` is not installed.
    """
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise ImportError(
            "watchdog is required for watch mode. Install it with: pip install vstash[watch]"
        ) from exc

    from .ingest import ingest_batch

    exts = extensions or SUPPORTED_EXTENSIONS

    # --- Serialised ingestion via queue (thread-safe for SQLite) ---
    ingest_queue: queue.Queue[str | None] = queue.Queue()
    stop_event = threading.Event()

    def _process_batch(paths: list[str]) -> None:
        """Route a drained burst through ingest_batch and report."""
        try:
            # Path.exists() can raise OSError (permission denied,
            # stale NFS mount, dead symlink target on some FSes).
            # Keep the check inside the try so a transient FS error on
            # one path does not kill the long-running worker thread.
            existing = [p for p in paths if Path(p).exists()]
            if not existing:
                return
            results = ingest_batch(
                existing,
                cfg,
                store,
                force=True,
                collection=collection,
                quiet=True,
            )
        except Exception as exc:
            console.print(f"[red]✗ Watch ingest error: {_rich_escape(str(exc))}[/red]")
            return
        ts = time.strftime("%H:%M:%S")
        for result in results:
            src_name = Path(result.source).name
            if result.status == "ok":
                console.print(
                    f"[dim]{ts}[/dim] [green]✓[/green] "
                    f"[bold]{result.title}[/bold] — "
                    f"{result.chunks} chunks → [cyan]{collection}[/cyan]"
                )
            elif result.status == "empty":
                console.print(f"[dim]{ts}[/dim] [yellow]⚠[/yellow] No content: {src_name}")
            elif result.status == "error":
                console.print(f"[dim]{ts}[/dim] [red]✗[/red] Error: {result.error}")

    def _worker() -> None:
        """Block for the first file, then drain the burst window and flush."""
        while not stop_event.is_set():
            try:
                first = ingest_queue.get(timeout=1)
            except queue.Empty:
                continue
            if first is None:  # Poison pill → exit
                break
            batch = _drain_burst(ingest_queue, first)
            _process_batch(batch)

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    debounce = _DebounceTimer(debounce_s)

    def _enqueue_file(file_path: str) -> None:
        """Push a file path into the ingestion queue."""
        ingest_queue.put(file_path)

    def _handle_delete(file_path: str) -> None:
        """Remove a deleted file from the store."""
        try:
            resolved = str(Path(file_path).resolve())
            deleted = store.delete_document(resolved)
            if deleted:
                ts = time.strftime("%H:%M:%S")
                console.print(f"[dim]{ts}[/dim] [red]✗[/red] Removed: {Path(file_path).name}")
        except Exception as exc:
            console.print(f"[red]✗ Watch delete error: {_rich_escape(str(exc))}[/red]")

    def _handle_dir_delete(dir_path: str) -> None:
        """Remove all documents under a deleted directory from the store."""
        try:
            resolved = str(Path(dir_path).resolve())
            # Use SQL LIKE prefix match — thread-safe and O(index) via idx_documents_path
            removed = store.delete_by_path_prefix(resolved + "/")
            if removed:
                ts = time.strftime("%H:%M:%S")
                console.print(
                    f"[dim]{ts}[/dim] [red]✗[/red] Removed {removed} docs from: "
                    f"{Path(dir_path).name}/"
                )
        except Exception as exc:
            console.print(f"[red]✗ Watch dir delete error: {_rich_escape(str(exc))}[/red]")

    class Handler(FileSystemEventHandler):
        """React to filesystem create/modify/delete events."""

        def on_created(self, event: FileSystemEvent) -> None:
            """Handle file creation."""
            if event.is_directory:
                return
            if _should_process(event.src_path, exts):
                debounce.trigger(event.src_path, _enqueue_file)

        def on_modified(self, event: FileSystemEvent) -> None:
            """Handle file modification."""
            if event.is_directory:
                return
            if _should_process(event.src_path, exts):
                debounce.trigger(event.src_path, _enqueue_file)

        def on_deleted(self, event: FileSystemEvent) -> None:
            """Handle file or directory deletion — remove from store."""
            if event.is_directory:
                # Cancel pending debounces for files inside the deleted dir so
                # they cannot fire and re-enqueue files removed with it.
                debounce.cancel_prefix(event.src_path)
                _handle_dir_delete(event.src_path)
            elif _should_process(event.src_path, exts):
                # Cancel any pending create/modify debounce for this path first,
                # so the delete is the last word and a stale timer can't
                # re-enqueue the just-removed file after it fires.
                debounce.cancel(event.src_path)
                _handle_delete(event.src_path)

    observer = Observer()
    handler = Handler()

    for watch_path in paths:
        resolved = str(Path(watch_path).expanduser().resolve())
        if not Path(resolved).is_dir():
            console.print(f"[yellow]⚠ Not a directory, skipping: {watch_path}[/yellow]")
            continue
        observer.schedule(handler, resolved, recursive=True)
        console.print(f"[cyan]👁 Watching[/cyan] {resolved}")

    console.print(
        f"[dim]Collection: {collection} | "
        f"Debounce: {debounce_s}s | "
        f"Extensions: {', '.join(sorted(exts))}[/dim]"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        debounce.cancel_all()
        stop_event.set()  # Signal worker to check exit condition
        # Drain pending items so worker doesn't process stale files
        while not ingest_queue.empty():
            try:
                ingest_queue.get_nowait()
            except queue.Empty:
                break
        ingest_queue.put(None)  # Poison pill for clean exit
        worker_thread.join(timeout=10)
        if worker_thread.is_alive():
            console.print("[yellow]⚠ Worker still processing — forcing stop.[/yellow]")
        observer.stop()
        console.print("\n[dim]Watcher stopped.[/dim]")
    observer.join()
