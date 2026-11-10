"""End-to-end ``service_search_latency_ms`` and the sub-millisecond buckets.

``search_latency_ms`` is recorded inside ``store.search`` and misses the
query embedding, validation, usage log and context expansion; the service
histogram covers the whole call the adapters make.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import requires_sqlite_vec
from vstash.config import VstashConfig
from vstash.metrics import Histogram, registry
from vstash.services import search_with_embedding
from vstash.services.search import SERVICE_SEARCH_LATENCY_METRIC
from vstash.store import VstashStore
from vstash.validation import LimitError

DIM = 384


def _vec() -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v.tolist()


@pytest.fixture
def store(tmp_path: Path):
    s = VstashStore(str(tmp_path / "m.db"), embedding_dim=DIM)
    s.add_document(
        path="/t/a.md", title="a", chunks=["alpha"], embeddings=[_vec()], source_type="markdown"
    )
    yield s
    s.close()


def _hist(name: str) -> dict:
    return registry.snapshot()["histograms"].get(name, {"count": 0, "sum_ms": 0.0})


class TestSubMillisecondBuckets:
    def test_sub_ms_observations_are_resolved(self):
        h = Histogram()
        for v in (0.05, 0.2, 0.4, 0.9):
            h.observe(v)
        buckets = {b["le"]: b["count"] for b in h.snapshot()["buckets_ms"]}
        assert buckets[0.1] == 1
        assert buckets[0.25] == 2
        assert buckets[0.5] == 3
        assert buckets[1.0] == 4


@requires_sqlite_vec
class TestServiceLatencyMetric:
    def test_covers_the_embedding(self, store):
        def slow_embed(*_a, **_k):
            time.sleep(0.02)
            return _vec()

        before_svc = _hist(SERVICE_SEARCH_LATENCY_METRIC)
        before_store = _hist("search_latency_ms")
        with patch("vstash.services.search.embed_query", side_effect=slow_embed):
            search_with_embedding(cfg=VstashConfig(), store=store, query="alpha", top_k=1)
        svc = _hist(SERVICE_SEARCH_LATENCY_METRIC)
        st = _hist("search_latency_ms")
        assert svc["count"] == before_svc["count"] + 1
        svc_ms = svc["sum_ms"] - before_svc["sum_ms"]
        store_ms = st["sum_ms"] - before_store["sum_ms"]
        assert svc_ms >= 20.0
        assert svc_ms > store_ms

    def test_recorded_when_the_call_raises(self, store):
        before = _hist(SERVICE_SEARCH_LATENCY_METRIC)["count"]
        with pytest.raises(LimitError):
            search_with_embedding(cfg=VstashConfig(), store=store, query="alpha", top_k=0)
        assert _hist(SERVICE_SEARCH_LATENCY_METRIC)["count"] == before + 1
