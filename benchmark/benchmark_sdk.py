#!/usr/bin/env python3
"""
benchmark_sdk.py — Measure Memory SDK overhead vs direct module usage.

Compares:
  1. Memory.add() vs ingest() directly
  2. Memory.search() vs store.search() + embed_query() directly
  3. Memory initialization time

Ensures the SDK wrapper adds negligible overhead.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vstash.config import load_config
from vstash.embed import embed_query, get_embedding_dim
from vstash.ingest import ingest
from vstash.memory import Memory
from vstash.store import VstashStore

REPORT_PATH = Path(__file__).parent / "SDK_BENCHMARK.md"
CORPUS_DIR = Path(__file__).parent / "corpus"

N_SEARCH_ITERATIONS = 20
QUERIES = [
    "What strategies exist for defending against an invasion?",
    "How do neural networks learn from data?",
    "What is the economic impact of rising temperatures?",
    "How should APIs handle authentication securely?",
    "What are the best practices for database migrations?",
]


def benchmark_init(db_path: str, n: int = 10) -> dict:
    """Measure Memory() constructor time."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mem = Memory(db=db_path)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mem.close()
    return {
        "avg_ms": round(sum(times) / len(times) * 1000, 2),
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


def benchmark_add(corpus_dir: Path, db_direct: str, db_sdk: str) -> dict:
    """Compare Memory.add() vs ingest() on corpus files."""
    cfg = load_config()
    dim = get_embedding_dim(cfg.embeddings.model)
    files = sorted(corpus_dir.glob("*.md"))[:3]

    # Direct
    store_direct = VstashStore(db_direct, embedding_dim=dim)
    direct_times = []
    for f in files:
        t0 = time.perf_counter()
        ingest(str(f), cfg, store_direct, force=True)
        direct_times.append(time.perf_counter() - t0)
    store_direct.close()

    # SDK
    mem = Memory(db=db_sdk)
    sdk_times = []
    for f in files:
        t0 = time.perf_counter()
        mem.add(f, force=True)
        sdk_times.append(time.perf_counter() - t0)
    mem.close()

    return {
        "files": [f.name for f in files],
        "direct_ms": [round(t * 1000, 1) for t in direct_times],
        "sdk_ms": [round(t * 1000, 1) for t in sdk_times],
        "overhead_pct": [
            round((s / d - 1) * 100, 1) if d > 0 else 0 for d, s in zip(direct_times, sdk_times)
        ],
    }


def benchmark_search(db_path: str) -> dict:
    """Compare Memory.search() vs direct store.search() + embed_query()."""
    cfg = load_config()
    dim = get_embedding_dim(cfg.embeddings.model)

    # Direct
    store = VstashStore(db_path, embedding_dim=dim)
    direct_times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        emb = embed_query(q, cfg.embeddings.model)
        store.search(emb, q, top_k=5)
        direct_times.append(time.perf_counter() - t0)
    store.close()

    # SDK
    mem = Memory(db=db_path)
    sdk_times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        mem.search(q, top_k=5)
        sdk_times.append(time.perf_counter() - t0)
    mem.close()

    return {
        "queries": QUERIES,
        "direct_ms": [round(t * 1000, 1) for t in direct_times],
        "sdk_ms": [round(t * 1000, 1) for t in sdk_times],
        "overhead_pct": [
            round((s / d - 1) * 100, 1) if d > 0 else 0 for d, s in zip(direct_times, sdk_times)
        ],
    }


def main() -> None:
    print("=" * 70)
    print("  SDK Benchmark: Memory class overhead")
    print("=" * 70)

    tmp = Path(__file__).parent / ".sdk_bench"
    tmp.mkdir(exist_ok=True)
    db_direct = str(tmp / "direct.db")
    db_sdk = str(tmp / "sdk.db")

    # --- Init ---
    print("\n⏱  Memory() init time...")
    init_result = benchmark_init(str(tmp / "init.db"))
    print(
        f"   avg: {init_result['avg_ms']}ms, "
        f"min: {init_result['min_ms']}ms, max: {init_result['max_ms']}ms"
    )

    # --- Add ---
    if not CORPUS_DIR.exists() or not list(CORPUS_DIR.glob("*.md")):
        print("\n⚠  No corpus files. Run benchmark.py first. Skipping add benchmark.")
        add_result = None
    else:
        print("\n⏱  Memory.add() vs ingest()...")
        add_result = benchmark_add(CORPUS_DIR, db_direct, db_sdk)
        for i, f in enumerate(add_result["files"]):
            print(
                f"   {f}: direct={add_result['direct_ms'][i]}ms, "
                f"sdk={add_result['sdk_ms'][i]}ms, "
                f"overhead={add_result['overhead_pct'][i]}%"
            )

    # --- Search (use SDK db which has docs) ---
    search_db = db_sdk if add_result else db_direct
    print("\n⏱  Memory.search() vs store.search()...")
    search_result = benchmark_search(search_db)
    for i, q in enumerate(search_result["queries"]):
        print(
            f'   "{q[:50]}...": '
            f"direct={search_result['direct_ms'][i]}ms, "
            f"sdk={search_result['sdk_ms'][i]}ms, "
            f"overhead={search_result['overhead_pct'][i]}%"
        )

    # --- Report ---
    lines = [
        "# SDK Benchmark: Memory class overhead\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n",
        "\n## Memory() Init\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average | {init_result['avg_ms']}ms |",
        f"| Min | {init_result['min_ms']}ms |",
        f"| Max | {init_result['max_ms']}ms |",
    ]

    if add_result:
        lines.append("\n## Memory.add() vs ingest()\n")
        lines.append("| File | Direct | SDK | Overhead |")
        lines.append("|------|--------|-----|----------|")
        for i, f in enumerate(add_result["files"]):
            lines.append(
                f"| {f} | {add_result['direct_ms'][i]}ms | "
                f"{add_result['sdk_ms'][i]}ms | {add_result['overhead_pct'][i]}% |"
            )

    lines.append("\n## Memory.search() vs store.search()\n")
    lines.append("| Query | Direct | SDK | Overhead |")
    lines.append("|-------|--------|-----|----------|")
    for i, q in enumerate(search_result["queries"]):
        short_q = q[:50] + "..."
        lines.append(
            f"| {short_q} | {search_result['direct_ms'][i]}ms | "
            f"{search_result['sdk_ms'][i]}ms | {search_result['overhead_pct'][i]}% |"
        )

    # Verdict
    avg_search_overhead = sum(search_result["overhead_pct"]) / len(search_result["overhead_pct"])
    lines.append("\n## Verdict\n")
    lines.append(f"- Memory() init: **{init_result['avg_ms']}ms** average")
    lines.append(f"- Search overhead: **{avg_search_overhead:.1f}%** average vs direct calls")
    if abs(avg_search_overhead) < 5:
        lines.append("- **Negligible overhead** — the SDK wrapper adds no measurable latency")
    elif avg_search_overhead < 15:
        lines.append("- **Minimal overhead** — the SDK wrapper adds minor latency")

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)
    print(f"\n📊 Report: {REPORT_PATH}")

    # Cleanup
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)


if __name__ == "__main__":
    main()
