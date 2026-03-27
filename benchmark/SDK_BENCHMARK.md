# SDK Benchmark: Memory class overhead

**Date:** 2026-03-27 10:18


## Memory() Init

| Metric | Value |
|--------|-------|
| Average | 1.26ms |
| Min | 0.93ms |
| Max | 2.58ms |

## Memory.add() vs ingest()

| File | Direct | SDK | Overhead |
|------|--------|-----|----------|
| climate_change_report.md | 1711.1ms | 40.9ms | -97.6% |
| fastapi_patterns.md | 74.0ms | 69.4ms | -6.2% |
| neural_architecture_search.md | 77.8ms | 74.0ms | -4.9% |

## Memory.search() vs store.search()

| Query | Direct | SDK | Overhead |
|-------|--------|-----|----------|
| What strategies exist for defending against an inv... | 4.5ms | 4.0ms | -11.0% |
| How do neural networks learn from data?... | 3.1ms | 3.3ms | 4.6% |
| What is the economic impact of rising temperatures... | 3.1ms | 2.9ms | -5.4% |
| How should APIs handle authentication securely?... | 2.9ms | 2.9ms | 0.2% |
| What are the best practices for database migration... | 2.9ms | 3.0ms | 4.2% |

## Verdict

- Memory() init: **1.26ms** average
- Search overhead: **-1.5%** average vs direct calls
- **Negligible overhead** — the SDK wrapper adds no measurable latency
