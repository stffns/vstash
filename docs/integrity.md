# Integrity and recovery

*Added in v0.24.0 — [issue #134](https://github.com/stffns/vstash/issues/134).*

A good substrate is honest about what survived a crash. vstash now ships with idempotent re-ingest, partial-document recovery, and a built-in `vstash check` command that runs a battery of integrity invariants against the database.

## Why

If `vstash add large_directory/` is interrupted halfway through (Ctrl-C, OOM, lock contention), older versions could leave the DB with partial documents whose `chunk_count` no longer matched the `chunks` table. There was no built-in way to detect this, no way to "resume from where we left off", and no way to verify that the FTS5 index was still in sync with the chunk store.

vstash 0.24.0 fixes all three.

## Idempotent re-ingest

`vstash add` (and the SDK `Memory.add()`) now classify each candidate document into one of three states before deciding what to do:

| State | Meaning | Action without `--force` |
|---|---|---|
| `complete` | documents row exists, `chunk_count` matches `COUNT(chunks)`, and every chunk has a `vec_chunks` row | **skip** — counted as `status="skipped"` |
| `partial` | documents row exists but the chunk count or vec index does not match | **heal** — drop the partial rows and re-ingest the file fresh |
| `missing` | no documents row for this path | **ingest** from scratch |

`--force` always re-ingests regardless of state. The new `--resume` flag is an explicit alias for the default idempotent behavior; it exists so that `vstash add --resume some_dir/` reads as a deliberate "continue what got interrupted" action.

`--force` and `--resume` are mutually exclusive — passing both is a configuration error.

```bash
# First run gets interrupted at 300 of 500 files…
vstash add ~/notes/

# …re-run heals partials and processes the remainder.
vstash add --resume ~/notes/
```

## `vstash check`

```bash
vstash check                  # human-readable table, exit 0 if clean
vstash check --json           # machine-readable
vstash check --repair         # apply the safe-to-fix subset
```

The check battery:

| Check | What it verifies | Repairable |
|---|---|---|
| `chunk_count_parity` | `documents.chunk_count == COUNT(chunks)` for every doc | yes |
| `vec_index_parity` | every chunk has a `vec_chunks` row (sqlite-vec backend) | no — needs original embeddings, use `vstash reindex` |
| `snapvec_parity` | snapvec index has one entry per chunk (snapvec backend) | no |
| `fts_index_parity` | `fts_chunks` and `chunks` have matching row counts | yes — rebuilds via SQLite FTS5 `'rebuild'` |
| `no_orphan_chunks` | every chunk's `doc_id` resolves to a document | yes — orphans are deleted |
| `sqlite_integrity` | SQLite-level `PRAGMA integrity_check` returns `ok` | no — page-level corruption needs out-of-band recovery |

Exit code is `0` if every check passes after repairs (when `--repair` is given) or after the first run (without `--repair`); otherwise `1`.

### Example output

```
                vstash integrity check
┌──────────────────────┬────────┬──────────┬──────────────────────────┐
│ Check                │ Status │ Affected │ Detail                   │
├──────────────────────┼────────┼──────────┼──────────────────────────┤
│ chunk_count_parity   │ FAIL   │        2 │ /a.md (declared=99, …)   │
│ vec_index_parity     │ PASS   │        — │ every chunk has a vec_…  │
│ fts_index_parity     │ PASS   │        — │ fts_chunks has one entry │
│ no_orphan_chunks     │ PASS   │        — │ every chunk's doc_id …   │
│ sqlite_integrity     │ PASS   │        — │ SQLite PRAGMA integrity  │
└──────────────────────┴────────┴──────────┴──────────────────────────┘
2 check(s) failed. Run vstash check --repair to apply safe fixes.
```

## What `--repair` actually does

For each repairable check that's currently failing:

- **`chunk_count_parity`** — recomputes `documents.chunk_count` from `COUNT(chunks)`. No data loss.
- **`fts_index_parity`** — runs `INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')`, the canonical SQLite FTS5 rebuild incantation. Reads from the `chunks` table; no data loss.
- **`no_orphan_chunks`** — deletes chunks whose `doc_id` no longer resolves, plus the matching `vec_chunks` rows. The chunks were already unreachable through the public API; the repair just cleans up the dangling rows.

Repairs that need data the substrate no longer has — re-embedding (vec parity) or page-level SQLite recovery — are reported as not-repairable. The user-facing remedy in those cases is `vstash reindex` for vec parity, or restoring from backup for SQLite-level corruption.

## Programmatic use

```python
from vstash import Memory

mem = Memory(project="research")
checks = mem._store.integrity_check()
for c in checks:
    if not c.passed:
        print(f"{c.name}: {c.detail} ({c.affected_count} affected)")

# Apply safe repairs
repairs = mem._store.integrity_repair()
for r in repairs:
    print(f"{r.name}: {r.detail}")
```

A higher-level `Memory.check()` / `Memory.repair()` is intentionally not exposed yet — the CLI is the primary surface, and `_store` is good enough for the rare programmatic case.

## What integrity checking is *not*

- Not a backup. Always keep one if your DB matters.
- Not a journal. SQLite WAL is the only journal vstash relies on.
- Not run automatically. `vstash check` is an explicit user action — startup is silent and fast, the way operators expect.
- Not a substitute for `vstash reindex` when the embedding model changes (see [`docs/observability.md`](observability.md) for the silent-killer detection that pairs with this).
