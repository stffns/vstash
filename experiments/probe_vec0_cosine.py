"""Probe: verify sqlite-vec supports distance=cosine and returns cosine distance.

Expected behavior:
    - Same unit vector  -> 0.0
    - Orthogonal unit   -> 1.0   (cos_sim=0 -> 1 - 0 = 1)
    - Opposite unit     -> 2.0   (cos_sim=-1 -> 1 - -1 = 2)
    - Non-normalized    -> cosine still in [0, 2] (sqlite-vec normalizes internally)

Run: python experiments/probe_vec0_cosine.py
"""

from __future__ import annotations

import sqlite3
import struct

import sqlite_vec


def _ser(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def run(metric: str) -> dict[str, float]:
    db = ":memory:"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    ddl = f"CREATE VIRTUAL TABLE vec_t USING vec0(embedding float[3] distance_metric={metric})"
    if metric == "l2":
        ddl = "CREATE VIRTUAL TABLE vec_t USING vec0(embedding float[3])"

    conn.execute(ddl)

    probes = {
        "same":        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        "orthogonal": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        "opposite":   ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        "scaled":     ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),      # same direction, diff magnitude
        "multi_big":  ([0.5, 0.5, 0.5], [3.0, 2.0, 1.0]),       # arbitrary non-unit
    }

    out: dict[str, float] = {}
    for name, (q, d) in probes.items():
        conn.execute("DELETE FROM vec_t")
        conn.execute("INSERT INTO vec_t(rowid, embedding) VALUES (1, ?)", [_ser(d)])
        row = conn.execute(
            "SELECT distance FROM vec_t WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
            [_ser(q)],
        ).fetchone()
        out[name] = row[0] if row else float("nan")
    conn.close()
    return out


if __name__ == "__main__":
    print(f"sqlite-vec loadable_path: {sqlite_vec.loadable_path()}")
    for metric in ("l2", "cosine"):
        try:
            res = run(metric)
        except Exception as e:
            print(f"metric={metric}: FAILED {e!r}")
            continue
        print(f"\nmetric={metric}")
        for name, d in res.items():
            print(f"  {name:12s} -> distance={d:.6f}")
