"""Tests for the metrics module and observability instrumentation (#132)."""

from __future__ import annotations

import time

import pytest

from vstash.metrics import Counter, Gauge, Histogram, MetricsRegistry, Timer, registry
from vstash.store import VstashStore


# ------------------------------------------------------------------ #
# Primitive metric types                                                #
# ------------------------------------------------------------------ #


class TestCounter:
    def test_starts_at_zero(self):
        c = Counter()
        assert c.value() == 0

    def test_inc_default_by_one(self):
        c = Counter()
        c.inc()
        c.inc()
        assert c.value() == 2

    def test_inc_by_amount(self):
        c = Counter()
        c.inc(5)
        c.inc(3)
        assert c.value() == 8


class TestGauge:
    def test_set_and_value(self):
        g = Gauge()
        g.set(42)
        assert g.value() == 42

    def test_inc_dec(self):
        g = Gauge()
        g.inc(10)
        g.dec(3)
        assert g.value() == 7

    def test_can_go_negative(self):
        g = Gauge()
        g.dec(5)
        assert g.value() == -5


class TestHistogram:
    def test_starts_empty(self):
        h = Histogram()
        assert h.count() == 0
        assert h.sum_ms() == 0.0
        assert h.mean_ms() == 0.0

    def test_observe_records_sum_and_count(self):
        h = Histogram()
        h.observe(5.0)
        h.observe(15.0)
        h.observe(100.0)
        assert h.count() == 3
        assert h.sum_ms() == 120.0
        assert h.mean_ms() == 40.0

    def test_buckets_are_cumulative(self):
        h = Histogram()
        h.observe(3.0)  # falls in 5.0 bucket
        h.observe(8.0)  # falls in 10.0 bucket
        snap = h.snapshot()
        # Buckets are cumulative: 5.0 bucket has 1, 10.0 has 2, etc.
        buckets = {b["le"]: b["count"] for b in snap["buckets_ms"]}
        assert buckets[5.0] == 1
        assert buckets[10.0] == 2
        assert buckets[1000.0] == 2
        assert buckets["+Inf"] == 2

    def test_snapshot_shape(self):
        h = Histogram()
        h.observe(1.0)
        snap = h.snapshot()
        assert set(snap.keys()) == {"count", "sum_ms", "mean_ms", "buckets_ms"}
        assert isinstance(snap["buckets_ms"], list)


# ------------------------------------------------------------------ #
# MetricsRegistry                                                        #
# ------------------------------------------------------------------ #


class TestMetricsRegistry:
    def test_counter_operations(self):
        r = MetricsRegistry()
        r.counter_inc("test_counter")
        r.counter_inc("test_counter", 5)
        snap = r.snapshot()
        assert snap["counters"]["test_counter"] == 6

    def test_gauge_operations(self):
        r = MetricsRegistry()
        r.gauge_set("test_gauge", 100)
        r.gauge_inc("test_gauge", 5)
        r.gauge_dec("test_gauge", 2)
        snap = r.snapshot()
        assert snap["gauges"]["test_gauge"] == 103

    def test_histogram_operations(self):
        r = MetricsRegistry()
        r.histogram_observe("test_hist", 10.0)
        r.histogram_observe("test_hist", 20.0)
        snap = r.snapshot()
        assert snap["histograms"]["test_hist"]["count"] == 2
        assert snap["histograms"]["test_hist"]["sum_ms"] == 30.0

    def test_snapshot_has_uptime(self):
        r = MetricsRegistry()
        time.sleep(0.01)
        snap = r.snapshot()
        assert snap["uptime_seconds"] >= 0
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap

    def test_reset_clears_metrics_but_not_uptime(self):
        r = MetricsRegistry()
        r.counter_inc("x")
        r.gauge_set("y", 5)
        original_start = r._started_at
        r.reset()
        snap = r.snapshot()
        assert snap["counters"] == {}
        assert snap["gauges"] == {}
        assert r._started_at == original_start


# ------------------------------------------------------------------ #
# Timer context manager                                                  #
# ------------------------------------------------------------------ #


class TestTimer:
    def test_records_elapsed_into_registry(self):
        registry.reset()
        with Timer("timed_operation"):
            time.sleep(0.01)
        snap = registry.snapshot()
        assert "timed_operation" in snap["histograms"]
        h = snap["histograms"]["timed_operation"]
        assert h["count"] == 1
        # Should be roughly 10ms but allow a wide band
        assert 1.0 <= h["sum_ms"] <= 1000.0

    def test_elapsed_ms_before_exit(self):
        t = Timer("not_yet_committed")
        t.__enter__()
        time.sleep(0.005)
        assert t.elapsed_ms() >= 0
        t.__exit__(None, None, None)


# ------------------------------------------------------------------ #
# Store instrumentation                                                  #
# ------------------------------------------------------------------ #


class TestStoreInstrumentation:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Reset the registry before each test so counts are clean."""
        registry.reset()
        yield
        registry.reset()

    def test_search_increments_searches_total(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello world"],
            embeddings=[[0.5] * dim],
        )
        sample_store.search([0.5] * dim, "hello", top_k=1)
        sample_store.search([0.5] * dim, "world", top_k=1)
        snap = registry.snapshot()
        assert snap["counters"]["searches_total"] == 2

    def test_search_records_latency_histogram(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello world"],
            embeddings=[[0.5] * dim],
        )
        sample_store.search([0.5] * dim, "hello", top_k=1)
        snap = registry.snapshot()
        assert "search_latency_ms" in snap["histograms"]
        assert snap["histograms"]["search_latency_ms"]["count"] == 1

    def test_idf_cache_hits_and_misses(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha content"],
            embeddings=[[0.5] * dim],
        )
        # First search: miss (first build)
        sample_store.search([0.5] * dim, "alpha", top_k=1)
        # Second search: hit (cache primed)
        sample_store.search([0.5] * dim, "content", top_k=1)
        snap = registry.snapshot()
        assert snap["counters"].get("idf_cache_misses", 0) >= 1
        assert snap["counters"].get("idf_cache_hits", 0) >= 1

    def test_slow_query_log_triggers_counter(self, sample_store: VstashStore):
        """With threshold=0, every query is logged as slow."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello"],
            embeddings=[[0.5] * dim],
        )
        sample_store._slow_query_ms_threshold = 0.0
        sample_store.search([0.5] * dim, "hello", top_k=1)
        sample_store.search([0.5] * dim, "hello", top_k=1)
        snap = registry.snapshot()
        assert snap["counters"].get("slow_queries_total", 0) == 2

    def test_slow_query_log_respects_high_threshold(self, sample_store: VstashStore):
        """With a very high threshold, no query is slow."""
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello"],
            embeddings=[[0.5] * dim],
        )
        sample_store._slow_query_ms_threshold = 999_999.0
        sample_store.search([0.5] * dim, "hello", top_k=1)
        snap = registry.snapshot()
        assert snap["counters"].get("slow_queries_total", 0) == 0

    def test_stats_updates_gauges(self, sample_store: VstashStore):
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["one", "two"],
            embeddings=[[0.1] * dim, [0.2] * dim],
        )
        sample_store.stats()
        snap = registry.snapshot()
        assert snap["gauges"]["docs_total"] == 1
        assert snap["gauges"]["chunks_total"] == 2
        assert snap["gauges"]["collections_total"] == 1
