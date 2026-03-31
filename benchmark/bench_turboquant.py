"""
bench_turboquant.py — sqlite-vec vs TurboQuantIndex head-to-head.

Measures:
  - Index build time
  - Query latency (p50, p95, p99)
  - Storage size
  - Search quality (Recall@K vs exact cosine ground truth)

Run:
    python benchmark/bench_turboquant.py
"""
from __future__ import annotations

import os
import sqlite3
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import sqlite_vec

sys_path = str(Path(__file__).parent.parent)
import sys; sys.path.insert(0, sys_path)

from turboquant.index import TurboQuantIndex

# ------------------------------------------------------------------ #
# Config                                                              #
# ------------------------------------------------------------------ #
DIM = 384          # BGE-small / paraphrase-MiniLM dimension
N_INDEX = 5_000    # vectors to index
N_QUERIES = 200    # queries to run
K = 10             # neighbors to retrieve
BITS = 4           # TurboQuant bits per coordinate
RNG_SEED = 42

print(f"Config: DIM={DIM}, N={N_INDEX}, queries={N_QUERIES}, K={K}, bits={BITS}\n")


# ------------------------------------------------------------------ #
# Synthetic data                                                      #
# ------------------------------------------------------------------ #
rng = np.random.default_rng(RNG_SEED)
vectors = rng.standard_normal((N_INDEX, DIM)).astype(np.float32)
# Normalize to unit sphere (typical for sentence embeddings)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / (norms + 1e-10)

queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
queries = queries / (q_norms + 1e-10)


# ------------------------------------------------------------------ #
# Ground truth (exact cosine, brute force)                           #
# ------------------------------------------------------------------ #
print("Computing exact ground truth…")
scores_matrix = queries @ vectors.T   # (N_QUERIES, N_INDEX)
ground_truth = [
    set(np.argsort(scores_matrix[i])[-K:])
    for i in range(N_QUERIES)
]
print("Done.\n")


# ------------------------------------------------------------------ #
# Helper: sqlite-vec                                                  #
# ------------------------------------------------------------------ #
def _serialize(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


def build_sqlite_vec(vecs: np.ndarray, db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        conn.close()
        conn = sqlite_vec.Connection(db_path)

    conn.execute(f"CREATE VIRTUAL TABLE vec USING vec0(embedding float[{DIM}])")
    conn.execute("BEGIN")
    for i, v in enumerate(vecs):
        conn.execute("INSERT INTO vec (rowid, embedding) VALUES (?, ?)", [i, _serialize(v)])
    conn.execute("COMMIT")
    return conn


def search_sqlite_vec(conn: sqlite3.Connection, query: np.ndarray, k: int) -> list[int]:
    rows = conn.execute(
        "SELECT rowid FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        [_serialize(query), k],
    ).fetchall()
    return [r[0] for r in rows]


# ------------------------------------------------------------------ #
# Benchmark: sqlite-vec                                              #
# ------------------------------------------------------------------ #
print("─" * 50)
print("BASELINE: sqlite-vec (float32, cosine)")
print("─" * 50)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    sv_path = f.name

t0 = time.monotonic()
sv_conn = build_sqlite_vec(vectors, sv_path)
sv_build_s = time.monotonic() - t0
sv_size_kb = os.path.getsize(sv_path) / 1024
print(f"  Build time : {sv_build_s:.3f}s")
print(f"  Index size : {sv_size_kb:.1f} KB  ({sv_size_kb/N_INDEX*1000:.0f} bytes/vector)")

latencies_sv = []
sv_results = []
for q in queries:
    t0 = time.monotonic()
    ids = search_sqlite_vec(sv_conn, q, K)
    latencies_sv.append((time.monotonic() - t0) * 1000)
    sv_results.append(set(ids))

p = np.percentile(latencies_sv, [50, 95, 99])
print(f"  Latency    : p50={p[0]:.2f}ms  p95={p[1]:.2f}ms  p99={p[2]:.2f}ms")

sv_recall = np.mean([
    len(sv_results[i] & ground_truth[i]) / K
    for i in range(N_QUERIES)
])
print(f"  Recall@{K}   : {sv_recall:.4f}")

sv_conn.close()
os.unlink(sv_path)


# ------------------------------------------------------------------ #
# Benchmark: TurboQuantIndex                                         #
# ------------------------------------------------------------------ #
print()
print("─" * 50)
print(f"TurboQuantIndex ({BITS}-bit PolarQuant)")
print("─" * 50)

with tempfile.NamedTemporaryFile(suffix=".tqvs", delete=False) as f:
    tq_path = f.name

t0 = time.monotonic()
tq_idx = TurboQuantIndex(dim=DIM, bits=BITS, seed=0)
tq_idx.add_batch(list(range(N_INDEX)), vectors)
tq_build_s = time.monotonic() - t0

tq_idx.save(tq_path)
tq_size_kb = os.path.getsize(tq_path) / 1024
print(f"  Build time : {tq_build_s:.3f}s")
print(f"  Index size : {tq_size_kb:.1f} KB  ({tq_size_kb/N_INDEX*1000:.0f} bytes/vector)")

latencies_tq = []
tq_results = []
for q in queries:
    t0 = time.monotonic()
    hits = tq_idx.search(q, k=K)
    latencies_tq.append((time.monotonic() - t0) * 1000)
    tq_results.append(set(id for id, _ in hits))

p = np.percentile(latencies_tq, [50, 95, 99])
print(f"  Latency    : p50={p[0]:.2f}ms  p95={p[1]:.2f}ms  p99={p[2]:.2f}ms")

tq_recall = np.mean([
    len(tq_results[i] & ground_truth[i]) / K
    for i in range(N_QUERIES)
])
print(f"  Recall@{K}   : {tq_recall:.4f}")
tq_idx2 = TurboQuantIndex.load(tq_path)
os.unlink(tq_path)

s = tq_idx.stats()
print(f"  Compression: {s['compression_ratio']:.2f}x vs float32")


# ------------------------------------------------------------------ #
# Summary                                                            #
# ------------------------------------------------------------------ #
print()
print("═" * 50)
print("SUMMARY")
print("═" * 50)
print(f"{'Metric':<22} {'sqlite-vec':>12} {'TurboQuant':>12} {'Δ':>8}")
print("─" * 56)
print(f"{'Build time (s)':<22} {sv_build_s:>12.3f} {tq_build_s:>12.3f} {(tq_build_s/sv_build_s):>7.1f}x")
print(f"{'Index size (KB)':<22} {sv_size_kb:>12.1f} {tq_size_kb:>12.1f} {(tq_size_kb/sv_size_kb):>7.1f}x")
print(f"{'Bytes/vector':<22} {sv_size_kb*1024/N_INDEX:>12.0f} {tq_size_kb*1024/N_INDEX:>12.0f}")
print(f"{'p50 latency (ms)':<22} {np.median(latencies_sv):>12.2f} {np.median(latencies_tq):>12.2f}")
print(f"{'p99 latency (ms)':<22} {np.percentile(latencies_sv,99):>12.2f} {np.percentile(latencies_tq,99):>12.2f}")
print(f"{'Recall@10':<22} {sv_recall:>12.4f} {tq_recall:>12.4f} {(sv_recall-tq_recall)*100:>+6.2f}pp")
print()
