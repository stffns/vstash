"""Tests for cross-thread stem connection cleanup (#125).

Verifies that ``VstashStore.close()`` releases per-thread FTS5
stemming connections registered by ``_stem_terms()`` regardless of
which thread created them.

Before #125, ``_stem_terms()`` stored its connection in
``threading.local()``, so connections created in worker threads
(e.g., a vstash serve request handler) were invisible to ``close()``
running on the main thread and would leak until process exit.
"""

from __future__ import annotations

import sqlite3
import threading

from vstash.store import VstashStore


class TestThreadSafetyAssumption:
    """Verify the libsqlite threading assumption that the fix relies on."""

    def test_sqlite3_threadsafety_supports_cross_thread_close(self):
        """vstash relies on sqlite3.threadsafety > 0 so close() can run
        on a different thread than the one that created a connection.
        Modern Python ships with threadsafety=3 (serialized).  This test
        documents the assumption and fails loudly if the runtime breaks it.
        """
        assert sqlite3.threadsafety > 0, (
            f"vstash assumes sqlite3.threadsafety > 0 (got {sqlite3.threadsafety}). "
            "Cross-thread Connection.close() in _stem_terms cleanup will crash."
        )


class TestStemConnCleanup:
    def test_main_thread_stem_conn_released(self, sample_store: VstashStore):
        """A stem conn created on the main thread should be released by close()."""
        # Trigger creation of the stem conn
        sample_store._stem_terms(["running", "tests"])
        assert len(sample_store._stem_conns) == 1
        sample_store.close()
        # close() should have cleared the dict
        assert sample_store._stem_conns == {}

    def test_worker_thread_stem_conn_released(self, sample_store: VstashStore):
        """A stem conn created on a worker thread should be released by close()."""
        worker_started = threading.Event()
        worker_done = threading.Event()

        def worker():
            sample_store._stem_terms(["worker", "thread", "stems"])
            worker_started.set()
            # Wait until the main thread has verified the conn exists
            worker_done.wait(timeout=5)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        worker_started.wait(timeout=5)

        # Main thread sees the worker's conn registered
        assert len(sample_store._stem_conns) == 1
        worker_tid = next(iter(sample_store._stem_conns.keys()))
        assert worker_tid != threading.get_ident()

        # Let the worker exit and main thread closes the store
        worker_done.set()
        t.join(timeout=5)

        sample_store.close()
        # The worker's conn should now be released, even though we're
        # calling close() from the main thread
        assert sample_store._stem_conns == {}

    def test_multiple_threads_each_get_own_conn(self, sample_store: VstashStore):
        """Each thread should get a distinct stem connection."""
        n_threads = 4
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            sample_store._stem_terms(["multi", "thread", "test"])

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # n_threads distinct connections should be registered
        assert len(sample_store._stem_conns) == n_threads
        # All keys should be unique thread ids
        assert len(set(sample_store._stem_conns.keys())) == n_threads

        # close() releases all of them
        sample_store.close()
        assert sample_store._stem_conns == {}

    def test_double_close_is_safe(self, sample_store: VstashStore):
        """Calling close() twice should not raise."""
        sample_store._stem_terms(["one", "two"])
        sample_store.close()
        # Second close should be a no-op (sqlite3 tolerates double-close)
        sample_store.close()
        assert sample_store._stem_conns == {}

    def test_no_stem_conn_no_close_failure(self, sample_store: VstashStore):
        """close() should work even if _stem_terms() was never called."""
        # No stem conn created — _stem_conns is empty
        assert sample_store._stem_conns == {}
        sample_store.close()
        # Still empty, no exception
        assert sample_store._stem_conns == {}
