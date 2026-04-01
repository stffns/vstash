"""End-to-end integration tests for vstash watch.

Spins up a real watchdog observer + worker thread against a temporary
directory and verifies that file create, modify, and delete events
flow through to the VstashStore.

Requires: watchdog, sqlite-vec
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import pytest

from vstash.config import VstashConfig
from vstash.embed import get_embedding_dim
from vstash.store import VstashStore
from vstash.watch import _DebounceTimer, _should_process

# Shorter debounce for faster tests
_DEBOUNCE = 0.3
# Max wait for events to propagate (debounce + embed + store)
_WAIT = 15.0
# Poll interval
_POLL = 0.2


def _sqlite_vec_available() -> bool:
    try:
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except Exception:
        return False


def _watchdog_available() -> bool:
    try:
        from watchdog.observers import Observer  # noqa: F401

        return True
    except ImportError:
        return False


requires_deps = pytest.mark.skipif(
    not (_sqlite_vec_available() and _watchdog_available()),
    reason="sqlite-vec and/or watchdog not available",
)


def _wait_for(predicate, timeout=_WAIT, poll=_POLL, msg="condition"):
    """Poll until predicate() is truthy or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    pytest.fail(f"Timed out waiting for: {msg}")


def _make_watch_env(tmp_path: Path):
    """Set up store, config, observer, worker — everything start_watch does,
    but non-blocking so we can control it from the test."""
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    from vstash.ingest import ingest

    cfg = VstashConfig()
    dim = get_embedding_dim(cfg.embeddings.model)
    db_path = str(tmp_path / "watch_test.db")
    store = VstashStore(db_path, embedding_dim=dim)

    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()

    exts = frozenset({".md", ".txt", ".py"})
    ingest_queue: queue.Queue[str | None] = queue.Queue()
    stop_event = threading.Event()
    collection = "watch-test"

    def _worker() -> None:
        while not stop_event.is_set():
            try:
                file_path = ingest_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if file_path is None:
                break
            if not Path(file_path).exists():
                continue
            try:
                ingest(file_path, cfg, store, force=True, collection=collection)
            except Exception:
                pass  # swallow for test stability

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    debounce = _DebounceTimer(delay=_DEBOUNCE)

    def _enqueue(path: str) -> None:
        ingest_queue.put(path)

    def _handle_delete(file_path: str) -> None:
        resolved = str(Path(file_path).resolve())
        store.delete_document(resolved)

    class Handler(FileSystemEventHandler):
        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _should_process(event.src_path, exts):
                debounce.trigger(event.src_path, _enqueue)

        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _should_process(event.src_path, exts):
                debounce.trigger(event.src_path, _enqueue)

        def on_deleted(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _should_process(event.src_path, exts):
                _handle_delete(event.src_path)

    observer = Observer()
    observer.schedule(Handler(), str(watch_dir), recursive=True)
    observer.start()

    class Env:
        pass

    env = Env()
    env.store = store
    env.cfg = cfg
    env.watch_dir = watch_dir
    env.observer = observer
    env.worker = worker
    env.stop_event = stop_event
    env.ingest_queue = ingest_queue
    env.debounce = debounce
    env.collection = collection
    return env


def _teardown(env):
    """Clean shutdown."""
    env.debounce.cancel_all()
    env.stop_event.set()
    env.ingest_queue.put(None)
    env.worker.join(timeout=5)
    env.observer.stop()
    env.observer.join(timeout=5)
    env.store.close()


# ------------------------------------------------------------------ #
# Tests                                                               #
# ------------------------------------------------------------------ #


@requires_deps
class TestWatchE2E:
    """End-to-end watch tests with real filesystem events."""

    def test_create_file_is_ingested(self, tmp_path: Path) -> None:
        """Creating a .md file in the watched dir should ingest it."""
        env = _make_watch_env(tmp_path)
        try:
            # Create a markdown file
            test_file = env.watch_dir / "hello.md"
            test_file.write_text("# Hello World\n\nThis is a test document about watch mode.")

            # Wait for ingestion
            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="document to appear in store after create",
            )

            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 1
            assert "hello.md" in docs[0].path
        finally:
            _teardown(env)

    def test_modify_file_updates_store(self, tmp_path: Path) -> None:
        """Modifying a watched file should re-ingest it with new content."""
        env = _make_watch_env(tmp_path)
        try:
            test_file = env.watch_dir / "notes.txt"
            test_file.write_text("Original content about Python.")

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="initial document to appear",
            )

            # Modify the file
            time.sleep(0.5)  # ensure filesystem sees it as a new event
            test_file.write_text("Updated content about Rust programming language.")

            # Wait for re-ingestion — char_count should change
            def content_updated():
                docs = env.store.list_documents(collection=env.collection)
                if not docs:
                    return False
                # The updated file should have "Rust" in its chunks
                results = env.store._conn.execute(
                    "SELECT text FROM chunks c JOIN documents d ON c.doc_id = d.id "
                    "WHERE d.path LIKE '%notes.txt'",
                ).fetchall()
                return any("Rust" in r["text"] for r in results)

            _wait_for(content_updated, msg="file content to update in store")
        finally:
            _teardown(env)

    def test_delete_file_removes_from_store(self, tmp_path: Path) -> None:
        """Deleting a watched file should remove it from the store."""
        env = _make_watch_env(tmp_path)
        try:
            test_file = env.watch_dir / "temporary.md"
            test_file.write_text("# Temporary\n\nThis file will be deleted shortly.")

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="document to appear before delete",
            )
            assert len(env.store.list_documents(collection=env.collection)) == 1

            # Delete the file
            test_file.unlink()

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) == 0,
                msg="document to be removed after delete",
            )
        finally:
            _teardown(env)

    def test_hidden_files_ignored(self, tmp_path: Path) -> None:
        """Hidden files should not be ingested."""
        env = _make_watch_env(tmp_path)
        try:
            # Create a hidden file
            hidden = env.watch_dir / ".secret.md"
            hidden.write_text("# Secret\n\nThis should not be ingested.")

            # Create a visible file to confirm watch is working
            visible = env.watch_dir / "visible.md"
            visible.write_text("# Visible\n\nThis should be ingested.")

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="visible document to appear",
            )

            # Only the visible file should be ingested
            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 1
            assert "visible.md" in docs[0].path
        finally:
            _teardown(env)

    def test_unsupported_extension_ignored(self, tmp_path: Path) -> None:
        """Files with unsupported extensions should be ignored."""
        env = _make_watch_env(tmp_path)
        try:
            # Create unsupported file
            img = env.watch_dir / "photo.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0fake jpeg")

            # Create supported file to confirm watch works
            doc = env.watch_dir / "readme.md"
            doc.write_text("# Readme\n\nA real document.")

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="markdown document to appear",
            )

            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 1
            assert "readme.md" in docs[0].path
        finally:
            _teardown(env)

    def test_multiple_files_ingested(self, tmp_path: Path) -> None:
        """Multiple files created in quick succession should all be ingested."""
        env = _make_watch_env(tmp_path)
        try:
            for i in range(3):
                f = env.watch_dir / f"doc_{i}.md"
                f.write_text(f"# Document {i}\n\nContent for document number {i}.")
                time.sleep(0.1)  # small gap so events are distinct

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) >= 3,
                msg="all 3 documents to appear",
            )

            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 3
        finally:
            _teardown(env)

    def test_subdirectory_files_detected(self, tmp_path: Path) -> None:
        """Files in subdirectories should be detected (recursive watch)."""
        env = _make_watch_env(tmp_path)
        try:
            subdir = env.watch_dir / "subdir" / "nested"
            subdir.mkdir(parents=True)

            deep_file = subdir / "deep.md"
            deep_file.write_text("# Deep File\n\nNested in subdirectories.")

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="nested file to appear",
            )

            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 1
            assert "deep.md" in docs[0].path
        finally:
            _teardown(env)

    def test_debounce_groups_rapid_saves(self, tmp_path: Path) -> None:
        """Rapid consecutive writes should be debounced into one ingestion."""
        env = _make_watch_env(tmp_path)
        try:
            test_file = env.watch_dir / "rapid.md"

            # Simulate rapid saves (like an editor auto-saving)
            for i in range(5):
                test_file.write_text(f"# Version {i}\n\nRapid save number {i}.")
                time.sleep(0.05)

            _wait_for(
                lambda: len(env.store.list_documents(collection=env.collection)) > 0,
                msg="debounced document to appear",
            )

            # Should be exactly 1 document (debounced)
            docs = env.store.list_documents(collection=env.collection)
            assert len(docs) == 1
        finally:
            _teardown(env)

    def test_clean_shutdown(self, tmp_path: Path) -> None:
        """Shutdown should terminate observer and worker cleanly."""
        env = _make_watch_env(tmp_path)

        # Verify threads are alive
        assert env.worker.is_alive()
        assert env.observer.is_alive()

        _teardown(env)

        # Both should be dead after teardown
        assert not env.worker.is_alive()
        assert not env.observer.is_alive()
