# API stability policy

*Added in v0.25.0 — [issue #135](https://github.com/stffns/vstash/issues/135).*

vstash is meant to be a **substrate** — a stable building block that other tools and frameworks can depend on. This document is the explicit contract: which symbols are public, what changes are allowed across versions, and how deprecation works.

## What counts as public

Public symbols are everything in `vstash.*` that does **not** begin with an underscore. Concretely:

- `vstash.Memory` — the SDK class.
- `vstash.SearchResult`, `vstash.DocumentInfo`, `vstash.ChunkInfo`, `vstash.IngestResult`, `vstash.StoreStats`, `vstash.ExplainInfo` — the typed result models.
- `vstash.config.VstashConfig` and every nested config model exported from `vstash.config` (`EmbeddingsConfig`, `ChunkingConfig`, `LimitsConfig`, `ObservabilityConfig`, …).
- `vstash.validation.LimitError` and its subclasses.
- `vstash.metrics.registry` and the `Counter` / `Gauge` / `Histogram` / `Timer` primitives.
- The `vstash` CLI: every documented subcommand, flag, and exit code.
- The HTTP endpoints exposed by `vstash serve` (`/health`, `/metrics`).
- The MCP tools exposed by `vstash.mcp`.

Anything else — modules and attributes whose name begins with `_`, the `vstash.store.VstashStore` class itself (it lives behind `Memory` for a reason), internal helpers like `_PipelineTracer`, the SQLite schema as a programmatic API, the on-disk `.snpv` snapvec file format — is **private**. It can change in any release.

## Semantic versioning contract

vstash uses the standard [SemVer](https://semver.org/) `major.minor.patch` triplet.

| Version bump | Allowed changes |
|---|---|
| **Patch** (`0.24.0 → 0.24.1`) | Bug fixes, doc updates, performance improvements that don't change observable behavior. **No new flags, no removed parameters, no schema changes.** |
| **Minor** (`0.24.x → 0.25.0`) | New CLI subcommands, new optional flags, new keyword arguments to public methods (with defaults), new config fields, new metric names, additive schema migrations (`ALTER TABLE … ADD COLUMN`). Existing public APIs keep working unchanged. |
| **Major** (`0.x → 1.0`, eventually `1.x → 2.0`) | Removals, renames, type changes, schema bumps that require migration, semantics changes. |

vstash is currently `0.x`, which by SemVer convention means anything *can* change in a minor release. In practice we already follow the table above — `0.x` is treated as "stabilizing", not "wild west". Concretely: a method that exists in `0.24.0` will exist with the same signature in `0.24.x` and `0.25.0`, modulo the deprecation path below.

## Deprecation path

When a public API needs to change:

1. **Minor release N**: the new behavior ships alongside the old. The old behavior emits `DeprecationWarning` via `warnings.warn(..., DeprecationWarning, stacklevel=2)` with the version it will be removed in.
2. **Minor release N+1** (one cycle later): the old behavior is removed.

Removals never happen inside a patch release. Removals during the `0.x` series may compress to one minor cycle of deprecation; once we reach `1.0` removals will require a major bump.

## Schema versioning

The SQLite database carries a `schema_version` row in the `store_meta` table (added in v0.25.0). On open, vstash reads it and compares against `KNOWN_SCHEMA_VERSIONS`.

| Outcome | Action |
|---|---|
| Match | Open normally. |
| Missing (legacy DB) | Stamp it as the current version and continue. |
| Unknown (future DB) | Raise `SchemaVersionError` with a message naming the detected version and the set this build can read. |

Pure additive `ALTER TABLE … ADD COLUMN` migrations stay within the same `schema_version`. The version is bumped only when a change requires a migration the runtime cannot perform automatically — column drops, type changes, semantics changes. Future versions will pair a bump with a `vstash migrate` subcommand.

## Score semantics

`SearchResult.score` is the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) score with the canonical constant `k = 60`. The contract:

- **Range:** typically `[0, ~0.033]` for default RRF settings. Recency boost and post-processing can push it slightly higher.
- **Within a single result list, higher means more relevant.**
- **NOT comparable across queries.** Different candidate pools, different adaptive RRF weights, different post-processing passes — the score scale is per-query.
- For "is this match good enough?" use `vstash explain` (`miss_analysis`) or the relevance tier metadata, never the raw score.

This is also documented in the `SearchResult` docstring (`vstash/models.py`) so that IDE tooltips show it.

## Config forward compatibility

Starting in v0.25.0, `VstashConfig` accepts unknown top-level sections in `vstash.toml` with a warning instead of a hard error. This means a user on an older vstash with a newer-format config file is not blocked from running — the unknown sections are silently ignored.

Each nested section (`[embeddings]`, `[limits]`, `[observability]`, …) keeps its strict Pydantic schema, so typos inside a known section are still caught at load time.

Removed fields will be handled in the deprecation path: emit `DeprecationWarning` for one minor cycle, then remove. They are NOT silently ignored — that would mask user mistakes.

## What this policy is *not*

- **Not a guarantee of zero bugs.** Bug fixes can change observable behavior; the policy is about *intentional* changes.
- **Not a guarantee of forever-stable internal performance.** A search that takes 10ms today might take 8ms tomorrow because of a tuning change. That is not a contract violation.
- **Not type-checked.** The promise is in this document, not enforced by the type system. If you depend on something the docs don't promise, you're depending on private behavior.
- **Not retroactive.** The policy applies from v0.25.0 onward. Pre-0.25 changes are documented in the changelog but were not bound by this contract.
