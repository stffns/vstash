# Observability

*Added in v0.22.0 — [issue #132](https://github.com/stffns/vstash/issues/132).*

vstash ships with an in-process metrics registry that exposes counters, gauges, and histograms for operational visibility. No external dependencies, no Prometheus text format — just a JSON dict that operators can scrape from the CLI or the web server.

## Why

Running vstash in production (`vstash serve`, MCP server, long-lived agent loops) without observability is flying blind. Frameworks built on top of vector databases (Mem0, Zep, LangChain memory) are black boxes — vstash is explicitly a **glass box**. This is a substrate-level differentiator, not a feature.

## What's tracked

### Counters

| Name | Meaning |
|---|---|
| `searches_total` | Total number of `search()` calls since process start |
| `slow_queries_total` | Total queries that exceeded `[observability] slow_query_ms` |
| `idf_cache_hits` | Adaptive RRF IDF cache hits (good — means the cache is warm) |
| `idf_cache_misses` | IDF cache rebuilds (happens after document mutations) |

### Gauges

| Name | Meaning |
|---|---|
| `docs_total` | Number of ingested documents (updated on `stats()`) |
| `chunks_total` | Number of chunks (updated on `stats()`) |
| `collections_total` | Number of distinct collections |
| `db_size_bytes` | SQLite file size in bytes |
| `stem_conn_count` | Number of live per-thread FTS5 stemming connections |

### Histograms

| Name | Meaning |
|---|---|
| `search_latency_ms` | End-to-end search latency including RRF, MMR, and formatting |

Histogram buckets use fixed log-scale boundaries in milliseconds: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, +Inf. For percentile estimation, compute from the cumulative counts in the snapshot.

## How to scrape

### CLI

```bash
# Human-readable summary
vstash stats --detailed

# Machine-readable JSON for scrapers
vstash stats --detailed --json
```

Example `--detailed` output:

```
┌─────────────── vstash Memory ───────────────┐
│ Documents: 314                              │
│ Chunks: 1324                                │
│ Collections: 42                             │
│ Database: ~/.vstash/memory.db               │
│ Size: 8.1 MB                                │
└─────────────────────────────────────────────┘

Observability metrics (uptime 1834.2s)

Counters:
  idf_cache_hits = 47
  idf_cache_misses = 2
  searches_total = 52
  slow_queries_total = 1

Gauges:
  chunks_total = 1324
  collections_total = 42
  db_size_bytes = 8488960
  docs_total = 314
  stem_conn_count = 1

Histograms:
  search_latency_ms: count=52 mean=14.3ms sum=743.6ms
```

### HTTP endpoints (via `vstash serve`)

When running the web server, two additional endpoints are exposed:

```bash
# Health check — returns 200 if the store is reachable
curl http://localhost:8585/health
# → {"status": "ok"}

# Metrics snapshot
curl http://localhost:8585/metrics | jq .
```

The `/health` endpoint is designed for load balancers, Docker health probes, and uptime monitoring. It runs a lightweight `stats()` call and returns 503 if the SQLite connection is unreachable.

The `/metrics` endpoint returns the full registry snapshot as JSON. Suitable for scraping with any pull-based monitoring system. There is no Prometheus text format endpoint — users who need Prometheus can write a small translator in their scraper configuration.

### Python SDK

```python
from vstash.metrics import registry

snap = registry.snapshot()
print(snap["counters"]["searches_total"])
print(snap["histograms"]["search_latency_ms"]["mean_ms"])
```

## Slow query log

Queries exceeding `[observability] slow_query_ms` (default 100ms) are logged to stderr with their query text (truncated to 60 chars), latency, result count, and adaptive RRF weights:

```
WARNING vstash.store: slow query: 145.3ms query='complex multi-paragraph query about …' results=5 vec_w=0.70 fts_w=0.30
```

### Configuring

```toml
# vstash.toml
[observability]
slow_query_ms = 50    # log anything over 50ms
# slow_query_ms = 0    # log every query (debug mode)
# slow_query_ms = 9999 # effectively disable
```

## Thread safety

The metrics registry uses a single module-level `threading.Lock`. All counter/gauge/histogram updates are thread-safe. The lock is rarely contended because updates are cheap (no I/O).

This matters in two deployments:

1. **`vstash serve`** — Starlette worker threads update the registry concurrently. All fine.
2. **MCP server** — tool calls arrive on worker threads. Same registry, same thread safety.

## What's NOT here (by design)

- **No Prometheus text format** — kept JSON to avoid a prometheus_client dependency. Users can translate JSON → Prom format in their scraper.
- **No distributed tracing** — vstash is single-process by design.
- **No external APM integration** — Datadog, New Relic, etc. are out of scope.
- **No persistent metrics** — the registry is in-memory and resets on process restart.
- **No rate limiting** — that's an upper-layer concern.

## Integration examples

### Docker healthcheck

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8585/health || exit 1
```

### Prometheus pull via simple translator

```python
# scrape.py — runs as a sidecar, translates JSON → Prom text
import requests

def scrape():
    data = requests.get("http://localhost:8585/metrics").json()
    lines = []

    # Counters and gauges map 1:1 to Prometheus scalars.
    for name, value in data["counters"].items():
        lines.append(f"# TYPE vstash_{name} counter")
        lines.append(f"vstash_{name} {value}")
    for name, value in data["gauges"].items():
        lines.append(f"# TYPE vstash_{name} gauge")
        lines.append(f"vstash_{name} {value}")

    # Histograms expand to _bucket{le="..."}, _sum, and _count series.
    # vstash already emits cumulative bucket counts, so they map directly
    # onto Prometheus histogram semantics.
    for name, hist in data["histograms"].items():
        lines.append(f"# TYPE vstash_{name} histogram")
        for bucket in hist["buckets_ms"]:
            le = bucket["le"]  # float or "+Inf"
            lines.append(f'vstash_{name}_bucket{{le="{le}"}} {bucket["count"]}')
        lines.append(f"vstash_{name}_sum {hist['sum_ms']}")
        lines.append(f"vstash_{name}_count {hist['count']}")

    return "\n".join(lines)
```

### Slow query alerting

Parse stderr with any log aggregator (Loki, Elastic, etc.) filtering on `"slow query:"`. No structured logging yet — the current format is pragmatic.
