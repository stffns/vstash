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

    def test_observe_boundary_values_land_in_le_bucket(self):
        """A value exactly on a bucket boundary lands in that bucket (value <=
        upper) -- the discriminating case for the bisect_left lookup vs the old
        linear `<=` scan."""
        h = Histogram()
        for v in (1.0, 2.0, 5.0, 0.5, 7000.0):
            h.observe(v)
        buckets = {b["le"]: b["count"] for b in h.snapshot()["buckets_ms"]}
        # cumulative: <=1.0 -> {1.0, 0.5}; +2.0; +5.0; the 7000.0 only at +Inf
        assert buckets[1.0] == 2
        assert buckets[2.0] == 3
        assert buckets[5.0] == 4
        assert buckets[5000.0] == 4
        assert buckets["+Inf"] == 5

    def test_observe_custom_buckets_without_inf_does_not_crash(self):
        """A value above all bounds in a custom (no +Inf) bucket set is counted
        in sum/count but increments no bucket -- preserving the old loop's
        fall-through (the bisect index would otherwise be out of range)."""
        h = Histogram(buckets_ms=(1.0, 5.0))  # deliberately no +Inf sentinel
        h.observe(100.0)  # above every bound
        assert h.count() == 1
        assert h.sum_ms() == 100.0
        buckets = {b["le"]: b["count"] for b in h.snapshot()["buckets_ms"]}
        assert buckets[5.0] == 0  # no bucket incremented, no IndexError


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

    def test_record_spread_prunes_to_last_50(self, sample_store: VstashStore):
        """record_spread keeps a sliding window of the last 50 entries -- the
        cheap range-delete that replaced the per-search NOT IN (subquery) prune
        must keep the newest and drop the oldest."""
        for i in range(60):
            sample_store.record_spread(float(i))
        spreads = [
            r[0]
            for r in sample_store._conn.execute(
                "SELECT spread FROM search_stats ORDER BY id"
            ).fetchall()
        ]
        assert len(spreads) == 50  # capped at the window size
        assert spreads[-1] == 59.0  # newest kept
        assert min(spreads) == 10.0  # oldest (0..9) pruned

    def test_idf_cache_hits_and_misses(self, sample_store: VstashStore):
        """The IDF cache should record exactly one miss per invalidation
        and one hit per subsequent access.  add_document invalidates, so
        we reset both the cache and the registry before measuring to get
        deterministic counts.
        """
        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["alpha content"],
            embeddings=[[0.5] * dim],
        )
        # Reset everything so we have a clean baseline: cache cleared,
        # counters at zero.  The first search rebuilds (miss), subsequent
        # searches hit the cache.
        sample_store._invalidate_idf_cache()
        registry.reset()

        sample_store.search([0.5] * dim, "alpha", top_k=1)
        sample_store.search([0.5] * dim, "content", top_k=1)

        snap = registry.snapshot()
        assert snap["counters"].get("idf_cache_misses", 0) == 1
        assert snap["counters"].get("idf_cache_hits", 0) == 1

    def test_slow_query_log_triggers_counter(self, sample_store: VstashStore):
        """With threshold=0, every query is logged as slow."""
        from vstash.config import ObservabilityConfig

        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello"],
            embeddings=[[0.5] * dim],
        )
        # Inject a new observability config; frozen Pydantic model so we
        # replace the whole object instead of mutating a field.
        sample_store._observability = ObservabilityConfig(slow_query_ms=0.0)
        sample_store.search([0.5] * dim, "hello", top_k=1)
        sample_store.search([0.5] * dim, "hello", top_k=1)
        snap = registry.snapshot()
        assert snap["counters"].get("slow_queries_total", 0) == 2

    def test_slow_query_log_respects_high_threshold(self, sample_store: VstashStore):
        """With a very high threshold, no query is slow."""
        from vstash.config import ObservabilityConfig

        dim = sample_store.embedding_dim
        sample_store.add_document(
            path="/a.md",
            title="A",
            chunks=["hello"],
            embeddings=[[0.5] * dim],
        )
        sample_store._observability = ObservabilityConfig(slow_query_ms=999_999.0)
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


# ------------------------------------------------------------------ #
# Embedding drift detection (silent killer defense)                     #
# ------------------------------------------------------------------ #


class TestEmbeddingDrift:
    def test_first_open_sets_metadata(self, sample_store: VstashStore):
        """First call to check_embedding_drift records the model."""
        assert sample_store.get_meta("embedding_model") is None
        result = sample_store.check_embedding_drift("BAAI/bge-small-en-v1.5")
        assert result is None
        assert sample_store.get_meta("embedding_model") == "BAAI/bge-small-en-v1.5"
        assert sample_store.get_meta("fastembed_version") is not None

    def test_matching_model_returns_none(self, sample_store: VstashStore):
        """Opening with the same model twice does not warn."""
        sample_store.check_embedding_drift("BAAI/bge-small-en-v1.5")
        result = sample_store.check_embedding_drift("BAAI/bge-small-en-v1.5")
        assert result is None

    def test_different_model_returns_warning(self, sample_store: VstashStore):
        """Switching embedding models triggers a warning message."""
        sample_store.check_embedding_drift("BAAI/bge-small-en-v1.5")
        result = sample_store.check_embedding_drift("BAAI/bge-base-en-v1.5")
        assert result is not None
        assert "mismatch" in result.lower()
        assert "bge-small" in result
        assert "bge-base" in result

    def test_fastembed_version_drift_on_affected_model(self, sample_store: VstashStore):
        """Changing fastembed version while using a pooling-affected
        multilingual model triggers a dedicated warning."""
        affected_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # Seed the store as if it were built with an old fastembed
        sample_store.set_meta("embedding_model", affected_model)
        sample_store.set_meta("fastembed_version", "0.4.9")  # pre-pooling-change

        result = sample_store.check_embedding_drift(affected_model)
        assert result is not None
        assert "pooling" in result.lower()
        assert "0.4.9" in result

    def test_fastembed_drift_ignored_for_unaffected_model(self, sample_store: VstashStore):
        """BGE models use CLS pooling throughout fastembed history, so a
        version change does NOT need a warning."""
        sample_store.set_meta("embedding_model", "BAAI/bge-small-en-v1.5")
        sample_store.set_meta("fastembed_version", "0.4.9")
        result = sample_store.check_embedding_drift("BAAI/bge-small-en-v1.5")
        assert result is None

    def test_meta_roundtrip(self, sample_store: VstashStore):
        """get_meta/set_meta are a plain key/value store."""
        assert sample_store.get_meta("custom_key") is None
        sample_store.set_meta("custom_key", "custom_value")
        assert sample_store.get_meta("custom_key") == "custom_value"
        # Upsert semantics
        sample_store.set_meta("custom_key", "updated")
        assert sample_store.get_meta("custom_key") == "updated"
