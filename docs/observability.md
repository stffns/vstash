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
| `adaptive_rrf_vector_empty_fallback_total` | Queries where the vector candidate pool was empty after the distance cutoff and the pipeline collapsed to FTS-only scoring (see below). A sustained rate > 0 suggests the embedding model is a bad fit for your corpus — see `docs/embedding-models.md` for mitigations. |

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

### Vector-empty fallback (`adaptive_rrf_vector_empty_fallback_total`)

*Added in v0.26.0 — [issue #156](https://github.com/stffns/vstash/issues/156).*

A sustained non-zero rate on this counter means the vector candidate pool is consistently being eliminated by the distance cutoff before RRF fusion can combine it with FTS5 results. The pipeline automatically collapses to FTS-only scoring when this happens — search does not fail, but the system is telling you that the vector signal is not contributing to your ranking.

**Common causes:**

1. **Embedding model mismatch** — the model cannot discriminate your domain vocabulary. The most frequent culprit is `paraphrase-multilingual-MiniLM-L12-v2` on clinical / legal / heavily-jargoned corpora. See `docs/embedding-models.md` for the mitigation ladder.
2. **Distance cutoff too tight** — `distance_cutoff` (default 1.15) is rejecting too many candidates. Try relaxing to 1.5 or 2.0.
3. **Long queries** — `paragraph_embedding` style queries of >50 words produce diffuse embeddings. Adaptive RRF already relaxes the cutoff for these, but the signal may still degrade.
4. **Sparse metadata filter** — a `collection` / `project` / `layer` filter eliminates all candidates in the local neighborhood. Rare but possible.

**Diagnostic drill-down:**

```bash
# How often has this fired since process start?
vstash stats --detailed --json | jq '.metrics.counters.adaptive_rrf_vector_empty_fallback_total'

# For a specific query that missed an expected doc, use `vstash why`:
vstash why "my query" --expect path/to/expected/doc.md

# Prints a stage-by-stage pipeline trace (vector_search -> distance_cutoff
# -> fts_search -> rrf_fusion -> recency_boost -> mmr_dedup -> top_k_cutoff)
# with the stage that dropped the expected chunk highlighted, plus the
# actual top-k for contrast and rule-based suggestions. Issue #157 /
# 2026-04-21.
#
# --json emits the raw MissAnalysis for piping:
vstash why "my query" --expect path/to/doc.md --json | jq .suggestions

# The Python SDK ``VstashStore.miss_analysis()`` is still available for
# programmatic use.
```

**Auto-logged miss hints** (#157 part 3):

```bash
# List the most recent auto-logged miss hints (empty / all-low search
# results) persisted in the DB. ``search_events`` keeps the last 1000
# rows across runs. Each row is a query that returned nothing or only
# low-relevance chunks; drill into any of them with
# `vstash why "<query>" --expect <path>` for the full trace.
vstash why --recent 10

# JSON for scripting / dashboards:
vstash why --recent 10 --json | jq '.recent_miss_hints[] | .query'
```

The hook is on by default. Disable via ``vstash.toml``:

```toml
[observability]
auto_miss_hint = false
```

**Planned follow-up for #157** (not yet shipped):
- ``/debug/why`` JSON/HTML route on ``vstash serve`` for a browser-native
  debug surface.

**Prometheus alerting suggestion:**

```yaml
- alert: VstashVectorEmptyFallbackElevated
  expr: rate(vstash_adaptive_rrf_vector_empty_fallback_total[5m]) > 0.1
  for: 10m
  annotations:
    summary: "vstash vector-empty fallback firing > 10% of queries"
    description: "Embedding model may be mismatched to corpus; see docs/embedding-models.md"
```
