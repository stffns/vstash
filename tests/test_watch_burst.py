"""Unit tests for the watch.py burst-drain logic (issue #266).

The drain helper takes a queue-head path plus the queue itself and
returns an ordered batch of up to ``max_batch`` paths, capped by a
``max_window_s`` deadline. The watcher worker then feeds that batch to
``ingest_batch`` so the whole burst shares one transaction, one FTS5
flush, and one snapvec vstack.

These tests exercise the drain contract in isolation from watchdog
and SQLite so we can assert semantics (ordering, poison-pill
re-enqueue, batch cap, window cap) deterministically.
"""

from __future__ import annotations

import queue


from vstash.watch import DRAIN_MAX_BATCH, _drain_burst


class TestDrainBurstOrdering:
    def test_returns_first_then_drained_in_fifo_order(self) -> None:
        q: queue.Queue[str | None] = queue.Queue()
        for p in ["b", "c", "d"]:
            q.put(p)

        batch = _drain_burst(q, "a")

        assert batch == ["a", "b", "c", "d"]
        assert q.empty()


class TestDrainBurstCaps:
    def test_max_batch_caps_drain(self) -> None:
        q: queue.Queue[str | None] = queue.Queue()
        # Queue holds way more than max_batch so we can see the cap fire.
        for i in range(200):
            q.put(f"p{i}")

        batch = _drain_burst(q, "first", max_batch=4)

        assert len(batch) == 4
        assert batch[0] == "first"
        assert batch[1:] == ["p0", "p1", "p2"]
        # Remaining items must still be on the queue for the next drain.
        remaining = []
        while not q.empty():
            remaining.append(q.get_nowait())
        assert remaining[0] == "p3"
        assert len(remaining) == 200 - 3

    def test_window_cap_stops_drain_when_empty(self) -> None:
        """Empty queue + zero window must not hang."""
        q: queue.Queue[str | None] = queue.Queue()

        # Frozen clock: time never advances, so the very first check of
        # "remaining = deadline - now()" returns exactly 0 after the
        # deadline is computed (deadline = t0 + 0 = t0). The drain must
        # return the single head path without blocking.
        frozen = [0.0]

        def _now() -> float:
            return frozen[0]

        batch = _drain_burst(q, "only", max_window_s=0.0, now=_now)
        assert batch == ["only"]


class TestDrainBurstPoisonPill:
    def test_poison_pill_mid_burst_is_requeued(self) -> None:
        """If None arrives mid-burst, the drain stops and re-enqueues it
        so the main worker loop still sees the shutdown signal.

        The re-queued poison pill lands at the tail, not back at the
        head, so the main loop will drain any already-queued paths
        ahead of it (one more burst window) before shutting down. That
        is the intended behaviour: shutdown is cooperative, not pre-
        emptive, and the cost of one extra burst is bounded by
        ``max_window_s``.
        """
        q: queue.Queue[str | None] = queue.Queue()
        q.put("b")
        q.put(None)
        q.put("c")

        batch = _drain_burst(q, "a", max_batch=10)

        # "a" is the head arg, "b" was taken from the queue before the
        # poison pill stopped the drain.
        assert batch == ["a", "b"]
        # Remaining queue: items after the None in the original stream
        # plus the re-queued None at the tail.
        remaining = []
        while not q.empty():
            remaining.append(q.get_nowait())
        assert None in remaining, (
            "poison pill must be re-enqueued so the main loop sees the shutdown signal"
        )
        assert "c" in remaining, "untouched items after the poison pill must survive"


class TestDrainBurstDefaults:
    def test_default_cap_matches_module_constant(self) -> None:
        """Cheap regression: the public DRAIN_MAX_BATCH is the one wired
        into the default keyword so changing one requires changing the
        other."""
        q: queue.Queue[str | None] = queue.Queue()
        # Fill the queue with max-1 items. Together with the head path
        # that totals DRAIN_MAX_BATCH and must not overflow.
        for i in range(DRAIN_MAX_BATCH + 10):
            q.put(f"p{i}")
        batch = _drain_burst(q, "head")
        assert len(batch) == DRAIN_MAX_BATCH
