"""
metrics.py — Lightweight in-process metrics for vstash.

A thread-safe registry of counters, gauges, and histograms used to
expose internal state for observability.  Intentionally minimal — no
Prometheus text format, no external deps, no network.  Consumers
(CLI, web endpoint) scrape the registry via ``snapshot()`` which
returns a JSON-serializable dict.

Design:
  - Thread-safe via a single ``threading.Lock`` (contention is rare
    since metrics updates are cheap).
  - Histograms use fixed log-scale buckets in milliseconds (1, 2, 5,
    10, 20, 50, 100, 200, 500, 1000, 2000, 5000, +Inf) — no percentile
    estimation, callers compute p50/p95/p99 from the snapshot if needed.
  - The module exposes a single module-level ``registry`` singleton so
    instrumentation can reach it without passing it everywhere.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import defaultdict
from typing import Any


class Counter:
    """Monotonically increasing integer counter.  Thread-safe via registry lock."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: int = 0

    def inc(self, amount: int = 1) -> None:
        self._value += amount

    def value(self) -> int:
        return self._value


class Gauge:
    """Arbitrary integer gauge that can go up or down."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: int = 0

    def set(self, value: int) -> None:
        self._value = value

    def inc(self, amount: int = 1) -> None:
        self._value += amount

    def dec(self, amount: int = 1) -> None:
        self._value -= amount

    def value(self) -> int:
        return self._value


# Fixed histogram buckets in milliseconds.  Chosen to cover vstash's
# realistic latency range (sub-ms to multi-second) with log-scale
# spacing so mid-range resolution is fine-grained where it matters.
_DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    float("inf"),
)


class Histogram:
    """Simple bucketed histogram with running sum and count.

    Used for latency distributions.  Buckets are fixed (not HDR) to keep
    the implementation trivial and cheap.  Each ``observe()`` increments
    **exactly one** bucket (O(buckets) lookup, not O(buckets²) as earlier
    iterations of this class did).  The cumulative Prometheus-style view
    is computed lazily in ``snapshot()``.
    """

    __slots__ = ("_bucket_counts", "_buckets_upper", "_count", "_sum")

    def __init__(self, buckets_ms: tuple[float, ...] = _DEFAULT_BUCKETS_MS) -> None:
        self._buckets_upper: tuple[float, ...] = buckets_ms
        # Per-bucket counts (not cumulative — cumulation happens in snapshot).
        self._bucket_counts: list[int] = [0] * len(buckets_ms)
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value_ms: float) -> None:
        """Record a single observation in milliseconds.

        Only the bucket corresponding to the value is incremented; the
        cumulative histogram view is produced at snapshot time.
        """
        self._sum += value_ms
        self._count += 1
        # Buckets are sorted ascending with a final +Inf, so the index of the
        # first upper bound >= value_ms (i.e. value_ms <= upper) is bisect_left.
        # The +Inf sentinel guarantees the index is always in range. O(log b),
        # in C, vs the previous O(b) Python scan -- this runs under the registry
        # lock on every store.search().
        self._bucket_counts[bisect.bisect_left(self._buckets_upper, value_ms)] += 1

    def count(self) -> int:
        return self._count

    def sum_ms(self) -> float:
        return self._sum

    def mean_ms(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the histogram.

        The ``buckets_ms`` field uses cumulative counts (Prometheus
        ``_bucket`` semantics): each bucket's count is the number of
        observations whose value was <= its upper bound.
        """
        cumulative = 0
        buckets: list[dict[str, Any]] = []
        for upper, count in zip(self._buckets_upper, self._bucket_counts, strict=True):
            cumulative += count
            buckets.append(
                {
                    "le": upper if upper != float("inf") else "+Inf",
                    "count": cumulative,
                }
            )
        return {
            "count": self._count,
            "sum_ms": round(self._sum, 3),
            "mean_ms": round(self.mean_ms(), 3),
            "buckets_ms": buckets,
        }


class MetricsRegistry:
    """Thread-safe registry of named metrics.

    A single module-level instance (``registry``) is used by the
    instrumentation points in ``store.py`` and read by the
    observability endpoints in ``cli.py`` and ``web.py``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = defaultdict(Counter)
        self._gauges: dict[str, Gauge] = defaultdict(Gauge)
        self._histograms: dict[str, Histogram] = defaultdict(Histogram)
        self._started_at: float = time.time()

    # ------------------------------------------------------------------ #
    # Write paths                                                          #
    # ------------------------------------------------------------------ #

    def counter_inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name].inc(amount)

    def gauge_set(self, name: str, value: int) -> None:
        with self._lock:
            self._gauges[name].set(value)

    def gauge_inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._gauges[name].inc(amount)

    def gauge_dec(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._gauges[name].dec(amount)

    def histogram_observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._histograms[name].observe(value_ms)

    # ------------------------------------------------------------------ #
    # Read path                                                             #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics.

        Structure::

            {
              "uptime_seconds": 123.4,
              "counters": {"searches_total": 42, ...},
              "gauges": {"docs_total": 312, ...},
              "histograms": {
                "search_latency_ms": {"count": 42, "mean_ms": 12.3, ...}
              }
            }
        """
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "counters": {name: c.value() for name, c in self._counters.items()},
                "gauges": {name: g.value() for name, g in self._gauges.items()},
                "histograms": {name: h.snapshot() for name, h in self._histograms.items()},
            }

    def reset(self) -> None:
        """Reset all metrics.  Primarily used by tests.

        Does NOT reset the ``started_at`` timestamp.
        """
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Module-level singleton used by the rest of the codebase.
registry = MetricsRegistry()


# ------------------------------------------------------------------ #
# Timer context manager — convenience for histogram observations       #
# ------------------------------------------------------------------ #


class Timer:
    """Context manager that records elapsed milliseconds into a histogram.

    Usage::

        with Timer("search_latency_ms"):
            run_expensive_thing()

    For code paths where the body is too long to fit in a ``with`` block
    (e.g. a 400-line ``search()`` method), prefer plain ``time.perf_counter``
    with ``try/finally`` so that exceptions still record the latency and
    increment counters.
    """

    __slots__ = ("_name", "_start")

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        registry.histogram_observe(self._name, elapsed_ms)

    def elapsed_ms(self) -> float:
        """Return elapsed time since entering the context (before __exit__)."""
        return (time.perf_counter() - self._start) * 1000.0
