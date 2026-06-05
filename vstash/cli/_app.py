"""Shared CLI core for vstash: the Typer ``app``, the root callback, and the
helpers every command module imports.

Split out of the monolithic ``cli.py`` (#284) so the command-group modules can
``from ._app import app, console, _get_store, ...`` without importing the package
``__init__`` (which would be a cycle). Holds no commands itself.
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path

import typer
from rich.console import Console

from .. import __version__
from .._store_open import open_store_for_config
from ..config import VstashConfig, load_config
from ..embed import warmup
from ..store import VstashStore


def _inference_hint(exc: ConnectionError, cfg: VstashConfig) -> str:
    """Return a user-friendly hint based on the inference error and backend."""
    msg = str(exc).lower()
    backend = cfg.inference.backend.lower()

    if "connection refused" in msg or "connect" in msg:
        if backend == "ollama":
            return "Is Ollama running? Try: ollama serve"
        return f"Could not reach {backend} server. Check your connection."
    if "rate limit" in msg or "429" in msg:
        return "Rate limited — wait a moment and try again."
    if "api key" in msg or "unauthorized" in msg or "401" in msg:
        return f"Check your API key for {backend}."
    if "timeout" in msg or "timed out" in msg:
        return "Request timed out — the server may be overloaded."
    return ""


def _version_callback(value: bool) -> None:
    if value:
        print(f"vstash {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="vstash",
    help="Local document memory with instant semantic search.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _safe_exc(exc: object) -> str:
    """Escape rich markup in an exception's text so brackets in messages
    like ``pip install vstash[ingest]`` survive ``console.print``.

    Without this, rich treats ``[ingest]`` as an opening tag and silently
    drops it from the rendered output, producing the truncated message
    ``pip install vstash`` that misled e2e users on PyPI.
    """
    from rich.markup import escape as _escape

    return _escape(str(exc))


def _load_labeled_queries_jsonl(path_str: str, *, flag: str) -> list[dict]:
    """Load and validate a JSONL file of labeled queries.

    Used by ``vstash retrain`` for both --training-queries and
    --eval-queries.  Each line must be a JSON object with a non-empty
    string ``query`` and a non-empty list of non-empty string
    ``relevant_paths``; returns the parsed records.

    Errors and exits the process with a clear message on:
        - missing file or non-file path,
        - invalid JSON (with the offending line number),
        - records that are not objects with the required keys,
        - records with non-string / empty fields,
        - empty file.
    The caller's flag name is woven into every message so the user
    knows which argument was wrong when they pass both.
    """
    path = Path(path_str).expanduser()
    if not path.is_file():
        # ``not exists`` and ``is a directory`` both end here -- distinguish
        # them so users debugging a wrong path know whether they typoed or
        # pointed at the parent dir.
        if path.exists():
            console.print(f"[red]x[/red] {flag}: path is not a file: {path}")
        else:
            console.print(f"[red]x[/red] {flag}: file not found: {path}")
        raise typer.Exit(code=1)

    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    q = json.loads(line)
                except json.JSONDecodeError as exc:
                    console.print(
                        f"[red]x[/red] {flag}: failed to parse {path} at line {line_no}: "
                        f'{exc}. Expected JSONL (one {{"query": ..., '
                        f'"relevant_paths": [...]}} per line).'
                    )
                    raise typer.Exit(code=1) from exc

                # Shape validation per line, with the line number in every
                # message so users can fix the exact record without diffing
                # the whole file.  A common mistake is passing a JSON-array-
                # per-file (``queries.json`` instead of ``queries.jsonl``),
                # which would parse as one list entry and crash opaquely
                # deep inside the miner / eval.
                if not isinstance(q, dict) or "query" not in q or "relevant_paths" not in q:
                    console.print(
                        f"[red]x[/red] {flag}: line {line_no} is not a valid query record. "
                        "Each line must be a JSON object with 'query' and "
                        "'relevant_paths' fields. If you have a single JSON array, "
                        "convert to JSONL (one object per line)."
                    )
                    raise typer.Exit(code=1)
                if not isinstance(q["query"], str) or not q["query"].strip():
                    console.print(
                        f"[red]x[/red] {flag}: line {line_no} has invalid 'query'. "
                        "The 'query' field must be a non-empty string."
                    )
                    raise typer.Exit(code=1)
                if not isinstance(q["relevant_paths"], list) or not q["relevant_paths"]:
                    console.print(
                        f"[red]x[/red] {flag}: line {line_no} has empty or non-list "
                        "'relevant_paths'. Every labeled query needs at least one "
                        "relevant document path."
                    )
                    raise typer.Exit(code=1)
                if any(not isinstance(p, str) or not p.strip() for p in q["relevant_paths"]):
                    console.print(
                        f"[red]x[/red] {flag}: line {line_no} has invalid 'relevant_paths'. "
                        "Every relevant path must be a non-empty string."
                    )
                    raise typer.Exit(code=1)
                records.append(q)
    except OSError as exc:
        console.print(f"[red]x[/red] {flag}: failed to read {path}: {exc}.")
        raise typer.Exit(code=1) from exc

    if not records:
        console.print(f"[red]x[/red] {flag}: file {path} contained no records.")
        raise typer.Exit(code=1)
    return records


def _build_miss_hint(
    *,
    cfg_auto_hint: bool,
    result_count: int,
    tier: str,
    best_distance: float,
    top_k_requested: int,
) -> dict | None:
    """Return a lightweight miss_hint dict when a search returns empty
    results or an all-low relevance tier, or ``None`` otherwise.

    Issue #157 part 3: the hint is persisted on the search_events row
    by ``record_search_event`` and consumed by ``vstash why --recent``
    so a user can drill into recent misses post-hoc without re-running
    the query from memory.

    The hint carries ONLY information that is not already a column on
    ``search_events`` (``tier``, ``best_distance``, ``result_count``
    are all stored separately). Keeping the hint dict minimal keeps
    the JSON blob small and avoids two-sources-of-truth drift when
    the columns evolve.

    The hint is intentionally cheap to compute (no extra SQL, no
    re-embedding) -- full ``miss_analysis`` is deferred to the explicit
    ``vstash why <query> --expect <path>`` call.
    """
    if not cfg_auto_hint:
        return None
    if result_count == 0:
        reason = "empty"
    elif tier == "low":
        reason = "all_low"
    else:
        return None
    return {
        "reason": reason,
        "top_k_requested": int(top_k_requested),
    }


def _record_miss_event(
    *,
    store,
    cfg,
    query: str,
    best_distance: float,
    tier: str,
    result_count: int,
    top_k_requested: int,
) -> None:
    """Build + persist a miss_hint search event in one place. Shared
    between ``vstash ask`` and ``vstash search`` to kill the
    copy-paste this PR introduced."""
    hint = _build_miss_hint(
        cfg_auto_hint=cfg.observability.auto_miss_hint,
        result_count=result_count,
        tier=tier,
        best_distance=best_distance,
        top_k_requested=top_k_requested,
    )
    store.record_search_event(
        query=query,
        best_distance=best_distance,
        relevance_tier=tier,
        result_count=result_count,
        miss_hint=hint,
    )


@app.callback()
def _app_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-P",
        help="Use a named profile (e.g., 'work', 'research').",
    ),
) -> None:
    """Local document memory with instant semantic search."""
    ctx.ensure_object(dict)
    if profile is not None:
        from ..profile import validate_name

        try:
            validate_name(profile)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    ctx.obj["profile"] = profile


def _profile_from_ctx(ctx: typer.Context) -> str | None:
    """Extract the profile name from the Typer context."""
    return (ctx.obj or {}).get("profile")


def _get_store(
    cfg: VstashConfig | None = None,
    *,
    warm: bool = False,
    profile: str | None = None,
) -> tuple[VstashConfig, VstashStore]:
    """Initialize config and create a VstashStore instance.

    Args:
        cfg: Optional pre-loaded config. If None, loads from vstash.toml.
        warm: If True, eagerly load the ONNX embedding model to
            eliminate cold-start latency on the first query.
        profile: Named profile to use. If None, uses resolution chain.

    Returns:
        Tuple of (config, store).
    """
    if cfg is None:
        cfg = load_config()
    if warm:
        warmup(cfg.embeddings.model)
    store = open_store_for_config(cfg, profile=profile)
    atexit.register(store.close)
    return cfg, store
