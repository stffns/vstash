"""
cli.py — vstash command line interface.

Commands:
  vstash add <file/url>   → ingest document
  vstash remember "text"  → ingest text directly (no file needed)
  vstash search "<query>" → semantic search (no LLM, free)
  vstash ask "<query>"    → single question (requires LLM)
  vstash chat             → interactive mode
  vstash list             → show ingested documents
  vstash stats            → memory statistics
  vstash forget <file>    → remove document
  vstash reindex           → re-embed chunks with new model
  vstash config           → show current config
  vstash profile          → manage named profiles
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import chat as chat_module
from ._store_open import open_store_for_config
from .config import VstashConfig, load_config
from .embed import embed_query, warmup
from .ingest import ingest, ingest_directory
from .services import search_with_embedding
from .store import VstashStore, relevance_tier

from . import __version__


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
        from .profile import validate_name

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


# ------------------------------------------------------------------ #
# vstash add                                                          #
# ------------------------------------------------------------------ #


@app.command()
def add(
    ctx: typer.Context,
    sources: list[str] = typer.Argument(..., help="Files, directories, or URLs to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already in memory"),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume an interrupted ingest: skip files that are already fully "
            "ingested and re-process partial ones.  Idempotent ingest is now "
            "the default; this flag exists for explicitness."
        ),
    ),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to add to"),
    project: str | None = typer.Option(
        None, "--project", "-p", help="Project tag (overrides frontmatter)"
    ),
    layer: str | None = typer.Option(
        None, "--layer", "-l", help="Layer tag (overrides frontmatter)"
    ),
    tags: str | None = typer.Option(
        None, "--tags", "-t", help="Comma-separated tags (overrides frontmatter)"
    ),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Override chunk size in tokens (default: from config)"
    ),
    chunk_overlap: int | None = typer.Option(
        None, "--chunk-overlap", help="Override chunk overlap in tokens (default: from config)"
    ),
) -> None:
    """Add documents or URLs to memory."""
    if force and resume:
        console.print(
            "[red]✗[/red] --force and --resume are mutually exclusive: "
            "--force always re-ingests, --resume only fills in gaps."
        )
        raise typer.Exit(code=2)
    cfg, store = _get_store(profile=_profile_from_ctx(ctx))
    meta = {"project": project, "layer": layer, "tags": tags}

    with store:
        for source in sources:
            path = Path(source)

            # Directory
            if path.is_dir():
                console.print(f"\n[bold]Scanning[/bold] {source}")
                results = ingest_directory(
                    source,
                    cfg,
                    store,
                    force=force,
                    collection=collection,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **meta,
                )
                ok = [r for r in results if r.status == "ok"]
                skipped = [r for r in results if r.status == "skipped"]
                msg = f"\n[green]✓[/green] {len(ok)}/{len(results)} files ingested from [bold]{source}[/bold]"
                if skipped:
                    msg += f" [dim]({len(skipped)} skipped, use --force to re-ingest)[/dim]"
                if collection != "default":
                    msg += f" [cyan]→ {collection}[/cyan]"
                console.print(msg)
                continue

            # File or URL
            source_str = str(path.resolve()) if path.exists() else source
            result = ingest(
                source_str,
                cfg,
                store,
                force=force,
                collection=collection,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **meta,
            )

            if result.status == "ok":
                parts = (
                    f"[green]✓[/green] [bold]{result.title}[/bold] — "
                    f"{result.chunks} chunks in {result.elapsed_s}s"
                )
                if collection != "default":
                    parts += f" [cyan]→ {collection}[/cyan]"
                console.print(parts)
            elif result.status == "skipped":
                console.print(f"[dim]⊊ {source} already in memory (use --force to re-ingest)[/dim]")
            elif result.status == "empty":
                console.print(f"[yellow]⚠[/yellow] No content extracted from {source}")
            elif result.status == "error":
                console.print(f"[red]✗[/red] Error: {result.error}")


# ------------------------------------------------------------------ #
# vstash ask                                                          #
# ------------------------------------------------------------------ #


@app.command()
def ask(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream the response"),
    all_profiles: bool = typer.Option(
        False, "--all-profiles", "-A", help="Search across all profiles"
    ),
) -> None:
    """Ask a question about your documents."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))

    with store:
        k = top_k or cfg.chunking.top_k

        # Embed + search via the services layer for the single-store
        # path; federated still embeds once up front because
        # federated_search wants the embedding pre-computed.
        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            if all_profiles:
                from .profile import federated_search

                q_embedding = embed_query(query, cfg.embeddings.model)
                tagged = federated_search(
                    query_embedding=q_embedding,
                    query_text=query,
                    cfg=cfg,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    expand_window=1,
                )
                chunks = [r for _, r in tagged]
            else:
                chunks = search_with_embedding(
                    cfg=cfg,
                    store=store,
                    query=query,
                    top_k=k,
                    expand_window=0,
                    collection=collection,
                    project=project,
                    layer=layer,
                )

        if not chunks:
            if not all_profiles:
                # Log the empty-result event with a miss_hint so
                # ``vstash why --recent`` can surface it later. Use
                # best_distance=1.0 (max) as the sentinel for "no
                # vector ever matched" -- federated mode has no single
                # best_distance and is skipped.
                _record_miss_event(
                    store=store,
                    cfg=cfg,
                    query=query,
                    best_distance=1.0,
                    tier="low",
                    result_count=0,
                    top_k_requested=k,
                )
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Tiered relevance signal (skip for federated — no single best_distance)
        if not all_profiles:
            tier = relevance_tier(store.last_best_distance)
            _record_miss_event(
                store=store,
                cfg=cfg,
                query=query,
                best_distance=store.last_best_distance,
                tier=tier,
                result_count=len(chunks),
                top_k_requested=k,
            )
            if tier == "low":
                console.print(
                    "[dim]⚠ Low relevance — context may not match your question well.[/dim]"
                )
            elif tier == "medium":
                console.print("[dim]? Uncertain relevance — results may be tangential.[/dim]")

        # Expand context: include adjacent chunks for richer LLM context
        # For federated mode, expansion happens inside federated_search
        # (per-store, before stores are closed).
        if not all_profiles:
            chunks = store.expand_context(chunks, window=1)

        # Show sources
        if sources:
            source_list = list({c.title for c in chunks})
            console.print(f"\n[dim]Sources: {', '.join(source_list)}[/dim]")

        # Stream response
        console.print()
        if stream:
            try:
                token_count = 0
                for token in chat_module.stream(query, chunks, cfg):
                    print(token, end="", flush=True)
                    token_count += 1
                print()  # newline after stream
            except ConnectionError as exc:
                if token_count > 0:
                    console.print(
                        f"\n[yellow]⚠ Stream interrupted after {token_count} tokens.[/yellow]"
                    )
                console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                hint = _inference_hint(exc, cfg)
                if hint:
                    console.print(f"[dim]  Hint: {hint}[/dim]")
                raise typer.Exit(1) from exc
        else:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                try:
                    response = chat_module.ask(query, chunks, cfg)
                except ConnectionError as exc:
                    console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                    hint = _inference_hint(exc, cfg)
                    if hint:
                        console.print(f"[dim]  Hint: {hint}[/dim]")
                    raise typer.Exit(1) from exc
            console.print(Markdown(response))


# ------------------------------------------------------------------ #
# vstash search                                                       #
# ------------------------------------------------------------------ #


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Your search query"),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
    all_profiles: bool = typer.Option(
        False, "--all-profiles", "-A", help="Search across all profiles"
    ),
    explain: bool = typer.Option(
        False, "--explain", "-E", help="Show why each chunk ranked where it did"
    ),
    miss: str | None = typer.Option(
        None,
        "--miss",
        help="Diagnose why this expected document path did not appear in results",
    ),
    miss_chunk: int | None = typer.Option(
        None,
        "--miss-chunk",
        help="Diagnose why this expected chunk id did not appear in results",
    ),
    exact_match: str | None = typer.Option(
        None,
        "--exact-match",
        help="Post-filter: each returned chunk's text must contain this "
        "literal substring. Bypasses FTS5 tokenization so punctuation / "
        "identifiers / code survive verbatim. Case-insensitive by default "
        "-- pair with --exact-match-case-sensitive for a strict compare.",
    ),
    exact_match_case_sensitive: bool = typer.Option(
        False,
        "--exact-match-case-sensitive/--no-exact-match-case-sensitive",
        help="Toggle case sensitivity of --exact-match.",
    ),
) -> None:
    """Semantic search without LLM (free, local)."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))
    _search_tagged: list[tuple[str, object]] | None = None

    # Miss analysis branch — short-circuits the normal search flow.
    # Exit code conventions for scripting:
    #   0 — expected document IS in top-k (no miss)
    #   1 — argument error (missing args, invalid path, etc.)
    #   2 — expected document did NOT appear (analysis completed successfully,
    #       but the miss is real — lets CI / monitoring scripts alert)
    if miss is not None or miss_chunk is not None:
        import json as _json

        def _miss_error(msg: str, exit_code: int = 1) -> typer.Exit:
            """Emit an error consistently for both JSON and pretty output modes.

            Returns the Exit exception so the caller can ``raise`` it,
            which keeps ``raise ... from None`` chaining clean.
            """
            if json_output:
                print(_json.dumps({"error": msg}))
            else:
                console.print(f"[red]✗ {msg}[/red]")
            return typer.Exit(exit_code)

        if all_profiles:
            raise _miss_error("Miss analysis is not supported with --all-profiles.")
        with store:
            k = top_k or cfg.chunking.top_k
            # Normalize the path the same way add() does
            path_arg: str | None = None
            if miss is not None:
                if miss.startswith(("http://", "https://", "text://")):
                    path_arg = miss
                else:
                    path_arg = str(Path(miss).resolve(strict=False))
            try:
                with console.status("[dim]Analyzing miss...[/dim]", spinner="dots"):
                    q_embedding = embed_query(query, cfg.embeddings.model)
                    analysis = store.miss_analysis(
                        q_embedding,
                        query,
                        expected_path=path_arg,
                        expected_chunk_id=miss_chunk,
                        top_k=k,
                        collection=collection,
                        project=project,
                        layer=layer,
                    )
            except ValueError as exc:
                raise _miss_error(str(exc)) from None

        # JSON output path (honored for both the no-miss and real-miss cases)
        if json_output:
            print(_json.dumps(analysis.model_dump(), indent=2))
            raise typer.Exit(0 if analysis.appeared_in_results else 2)

        # Pretty print (Rich-styled)
        console.print()
        console.print(f"[bold]Query:[/bold] {query}")
        expected_label = analysis.expected_path or f"chunk #{analysis.expected_chunk_id}"
        console.print(f"[bold]Expected:[/bold] {expected_label}")
        if analysis.target_resolution == "best_of_n":
            console.print(
                f"[dim]  → tracing best-matching chunk of "
                f"{analysis.total_chunks_in_doc} in the document (bias warning: "
                f"other chunks may fail differently)[/dim]"
            )
        console.print(f"[bold]top_k:[/bold] {analysis.top_k_requested}")
        console.print()
        if analysis.appeared_in_results:
            console.print(
                f"[green]✓ Expected document IS in results at rank "
                f"{(analysis.final_rank or 0) + 1}. No miss to analyze.[/green]"
            )
            raise typer.Exit(0)
        console.print(
            f"[red]✗ Expected document did NOT appear. Dropped at: "
            f"[bold]{analysis.dropped_at}[/bold][/red]\n"
        )
        console.print("[bold cyan]Pipeline trace:[/bold cyan]")
        for v in analysis.stage_verdicts:
            mark = "[green]✓[/green]" if v.passed else "[red]✗[/red]"
            console.print(f"  {mark} [bold]{v.stage}[/bold]: {v.detail}")
            if v.counterfactual:
                console.print(f"     [dim]→ {v.counterfactual}[/dim]")
        console.print()
        console.print("[bold cyan]Suggestions:[/bold cyan]")
        for s in analysis.suggestions:
            console.print(f"  • {s}")
        console.print()
        if analysis.actual_top_k:
            console.print("[bold cyan]Actual top results (for comparison):[/bold cyan]")
            for r in analysis.actual_top_k:
                console.print(f"  {r.rank + 1}. {r.title} ({r.path}) — score {r.score:.4f}")
        raise typer.Exit(2)

    with store:
        k = top_k or cfg.chunking.top_k

        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            q_embedding = embed_query(query, cfg.embeddings.model)

            if all_profiles:
                from .profile import federated_search

                tagged = federated_search(
                    query_embedding=q_embedding,
                    query_text=query,
                    cfg=cfg,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    expand_window=1,
                )
                chunks = [r for _, r in tagged]
                _search_tagged = tagged
                # Post-filter federated results the same way VstashStore.search
                # does so --exact-match works across profiles too. #106.
                if exact_match:
                    if exact_match_case_sensitive:
                        _search_tagged = [
                            (name, r) for (name, r) in _search_tagged if exact_match in r.text
                        ]
                    else:
                        _needle = exact_match.casefold()
                        _search_tagged = [
                            (name, r)
                            for (name, r) in _search_tagged
                            if _needle in r.text.casefold()
                        ]
                    chunks = [r for _, r in _search_tagged]
            else:
                chunks = store.search(
                    q_embedding,
                    query,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    explain=explain,
                    exact_match=exact_match,
                    exact_match_case_sensitive=exact_match_case_sensitive,
                )

        if not chunks:
            if not all_profiles:
                _record_miss_event(
                    store=store,
                    cfg=cfg,
                    query=query,
                    best_distance=1.0,
                    tier="low",
                    result_count=0,
                    top_k_requested=k,
                )
            if json_output:
                print("[]")
                raise typer.Exit()
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Relevance signal (skip for federated — no single best_distance)
        tier = "high"
        if not all_profiles:
            best_distance = store.last_best_distance
            tier = relevance_tier(best_distance)

            # Telemetry: record search event for discard rate analysis,
            # attaching a miss_hint when the tier is "low" (issue #157 part 3).
            _record_miss_event(
                store=store,
                cfg=cfg,
                query=query,
                best_distance=best_distance,
                tier=tier,
                result_count=len(chunks),
                top_k_requested=k,
            )

        if json_output:
            import json

            out: dict = {"chunks": [c.model_dump(exclude_none=True) for c in chunks]}
            if not all_profiles:
                out["relevance"] = tier
                out["best_distance"] = round(best_distance, 4)
            if _search_tagged:
                out["profiles"] = [name for name, _ in _search_tagged]
            print(json.dumps(out, indent=2))
            raise typer.Exit()

        if tier == "low":
            console.print("[dim]⚠ Low relevance — results may not match your query.[/dim]\n")

        # Normalize scores to [0, 1] for display
        scores = [c.score for c in chunks]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if len(scores) > 1 else 0.0
        score_range = max_score - min_score

        table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        if all_profiles and _search_tagged:
            table.add_column("Profile", style="magenta", max_width=15)
        table.add_column("Score", width=6)
        table.add_column("Source", style="green", max_width=30)
        table.add_column("Text", max_width=80)

        for i, c in enumerate(chunks, 1):
            display_score = (c.score - min_score) / score_range if score_range > 0 else 1.0
            rank_label = f"{i}?" if tier == "medium" else str(i)
            text_preview = c.text.replace("\n", " ").strip()
            if len(text_preview) > 120:
                text_preview = text_preview[:120] + "..."
            if all_profiles and _search_tagged:
                profile_name = _search_tagged[i - 1][0] if i <= len(_search_tagged) else ""
                table.add_row(
                    rank_label,
                    profile_name,
                    f"{display_score:.2f}",
                    c.title,
                    text_preview,
                )
            else:
                table.add_row(
                    rank_label,
                    f"{display_score:.2f}",
                    c.title,
                    text_preview,
                )

        console.print(table)

        # --- Explain: diagnostic breakdown per chunk ---
        if explain:
            from rich.markup import escape

            console.print()
            for i, c in enumerate(chunks, 1):
                ex = c.explain
                if ex is None:
                    continue

                console.print(
                    f"[bold cyan]#{i}[/bold cyan] [bold]{escape(c.title)}[/bold] — {escape(c.path)}"
                )

                lines = []
                # Vector
                if ex.vec_rank is not None:
                    tier_label = (
                        relevance_tier(ex.vec_distance) if ex.vec_distance is not None else "?"
                    )
                    lines.append(
                        f"  Vector:  rank {ex.vec_rank + 1}, "
                        f"distance {ex.vec_distance} ({tier_label} relevance)"
                    )
                else:
                    lines.append("  Vector:  not in vector results (FTS-only match)")

                # FTS
                if ex.fts_rank is not None:
                    terms = ", ".join(escape(t) for t in ex.fts_terms) if ex.fts_terms else "n/a"
                    lines.append(f"  FTS:     rank {ex.fts_rank + 1}, terms \\[{terms}]")
                else:
                    lines.append("  FTS:     not in keyword results (vector-only match)")

                # RRF
                weight_info = ""
                if ex.rrf_vec_weight is not None:
                    weight_info = f" w={ex.rrf_vec_weight:.1f}/{ex.rrf_fts_weight:.1f}"
                lines.append(
                    f"  RRF:     {ex.rrf_total:.4f} "
                    f"(vec: {ex.rrf_vec:.4f} + fts: {ex.rrf_fts:.4f}){weight_info}"
                )

                # MMR
                if ex.mmr_penalty > 0:
                    lines.append(f"  MMR:     -{ex.mmr_penalty:.2f} penalty (same-doc similarity)")
                else:
                    lines.append("  MMR:     no penalty (unique document)")

                for line in lines:
                    console.print(f"[dim]{line}[/dim]")
                console.print()


# ------------------------------------------------------------------ #
# vstash chat                                                         #
# ------------------------------------------------------------------ #


@app.command()
def chat(
    ctx: typer.Context,
    top_k: int = typer.Option(0, "--top-k", "-k"),
) -> None:
    """Interactive chat mode. Type 'exit' or Ctrl+C to quit."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))

    with store:
        k = top_k or cfg.chunking.top_k

        console.print(
            Panel(
                "[bold cyan]vstash[/bold cyan] · Interactive mode\n"
                "[dim]Type your question. 'exit' to quit.[/dim]",
                border_style="cyan",
            )
        )

        history: list[dict[str, str]] = []
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
        _MAX_HISTORY_TOKENS = 8192

        def _trim_history(hist: list[dict[str, str]]) -> list[dict[str, str]]:
            """Keep only the most recent turns that fit within the token budget."""
            total = 0
            trimmed: list[dict[str, str]] = []
            for msg in reversed(hist):
                msg_tokens = len(_enc.encode(msg["content"]))
                if total + msg_tokens > _MAX_HISTORY_TOKENS:
                    break
                trimmed.append(msg)
                total += msg_tokens
            return list(reversed(trimmed))

        # Telemetry: track whether user engages after non-high results
        last_event_id: int | None = None
        last_tier: str = "high"

        def _maybe_dismiss() -> None:
            """Mark the last search as dismissed if it was non-high relevance."""
            if last_event_id is not None and last_tier != "high":
                store.mark_search_dismissed(last_event_id)

        try:
            while True:
                console.print()
                try:
                    query = console.input("[bold cyan]>[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    _maybe_dismiss()
                    break

                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    _maybe_dismiss()
                    break

                # Search via the services layer: embed + validate +
                # search + expand_context in one call. expand_window=1
                # gives the LLM adjacent context, same as before.
                chunks = search_with_embedding(
                    cfg=cfg,
                    store=store,
                    query=query,
                    top_k=k,
                    expand_window=1,
                )

                if not chunks:
                    console.print("[yellow]No relevant context found.[/yellow]")
                    continue

                # Tiered relevance signal (post-expand, mirroring the
                # MCP migration pattern -- semantically identical to
                # the old pre-expand placement).
                tier = relevance_tier(store.last_best_distance)
                last_event_id = store.record_search_event(
                    query=query,
                    best_distance=store.last_best_distance,
                    relevance_tier=tier,
                    result_count=len(chunks),
                )
                last_tier = tier
                if tier == "low":
                    console.print("[dim]⚠ Low relevance — context may not match well.[/dim]")
                elif tier == "medium":
                    console.print("[dim]? Uncertain relevance — results may be tangential.[/dim]")

                source_list = list({c.title for c in chunks})
                console.print(f"[dim]Sources: {', '.join(source_list)}[/dim]\n")

                # Stream with history (trimmed to token budget)
                trimmed = _trim_history(history)
                full_response = ""
                token_count = 0
                try:
                    for token in chat_module.stream(query, chunks, cfg, history=trimmed):
                        print(token, end="", flush=True)
                        full_response += token
                        token_count += 1
                    print()
                except ConnectionError as exc:
                    if token_count > 0:
                        console.print(
                            f"\n[yellow]⚠ Stream interrupted after {token_count} tokens.[/yellow]"
                        )
                    console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                    hint = _inference_hint(exc, cfg)
                    if hint:
                        console.print(f"[dim]  Hint: {hint}[/dim]")
                    continue

                # Accumulate history for multi-turn context
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            pass

        console.print("\n[dim]Goodbye.[/dim]")


# ------------------------------------------------------------------ #
# vstash list                                                         #
# ------------------------------------------------------------------ #


@app.command(name="list")
def list_docs(
    ctx: typer.Context,
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Filter by layer"),
) -> None:
    """List all documents in memory."""
    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        docs = store.list_documents(
            collection=collection,
            project=project,
            layer=layer,
        )

        if not docs:
            msg = (
                "Memory is empty."
                if not collection
                else f'No documents in collection "{collection}".'
            )
            console.print(f"[yellow]{msg} Add documents with [bold]vstash add[/bold].[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Title", style="bold")
        table.add_column("Collection", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Chunks", justify="right")
        table.add_column("Added", style="dim")

        for doc in docs:
            added = doc.added_at[:10]  # just the date
            table.add_row(
                doc.title,
                doc.collection,
                doc.source_type,
                str(doc.chunk_count),
                added,
            )

        console.print(table)


# ------------------------------------------------------------------ #
# vstash export                                                       #
# ------------------------------------------------------------------ #


@app.command()
def export(
    ctx: typer.Context,
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    fmt: str = typer.Option("jsonl", "--format", "-f", help="Output format: jsonl or csv"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Filter by layer"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Filter by tag"),
    include_embeddings: bool = typer.Option(
        False, "--include-embeddings", help="Include embedding vectors"
    ),
) -> None:
    """Export chunks as JSONL or CSV for training data curation."""
    import csv
    import json

    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        chunks = store.export_chunks(
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            include_embeddings=include_embeddings,
        )

    if not chunks:
        console.print("[yellow]No chunks match the given filters.[/yellow]")
        raise typer.Exit(1)

    out_path = Path(output)
    fmt = fmt.lower()

    if fmt not in ("jsonl", "csv"):
        console.print(f"[red]✗ Unsupported format '{fmt}'. Use 'jsonl' or 'csv'.[/red]")
        raise typer.Exit(1)

    if fmt == "csv":
        fieldnames = ["text", "title", "path", "chunk", "collection", "project", "layer", "tags"]
        if include_embeddings:
            fieldnames.append("embedding")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for chunk in chunks:
                row = {}
                for k in fieldnames:
                    value = chunk.get(k, "")
                    if value is None:
                        value = ""
                    row[k] = value
                if include_embeddings and "embedding" in chunk:
                    row["embedding"] = json.dumps(chunk["embedding"])
                writer.writerow(row)
    else:  # jsonl
        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    console.print(f"[green]✓[/green] Exported {len(chunks)} chunks → [bold]{output}[/bold] ({fmt})")


# ------------------------------------------------------------------ #
# vstash stats                                                        #
# ------------------------------------------------------------------ #


@app.command()
def stats(
    ctx: typer.Context,
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show the full observability metrics registry (counters, gauges, histograms)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON (machine-readable, good for scrapers)",
    ),
) -> None:
    """Show memory statistics."""
    import json as _json

    from .metrics import registry

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        s = store.stats()

        if json_output:
            payload = {
                "documents": s.documents,
                "chunks": s.chunks,
                "collections": s.collections,
                "db_path": s.db_path,
                "db_size_mb": s.db_size_mb,
                "inference_backend": cfg.inference.backend,
                "inference_model": cfg.inference.model,
                "embeddings_model": cfg.embeddings.model,
            }
            if detailed:
                payload["metrics"] = registry.snapshot()
            print(_json.dumps(payload, indent=2))
            return

        console.print(
            Panel(
                f"[bold]Documents:[/bold] {s.documents}\n"
                f"[bold]Chunks:[/bold] {s.chunks}\n"
                f"[bold]Collections:[/bold] {s.collections}\n"
                f"[bold]Database:[/bold] {s.db_path}\n"
                f"[bold]Size:[/bold] {s.db_size_mb} MB\n"
                f"[bold]Backend:[/bold] {cfg.inference.backend} / {cfg.inference.model}\n"
                f"[bold]Embeddings:[/bold] {cfg.embeddings.model}",
                title="[bold cyan]vstash Memory[/bold cyan]",
                border_style="cyan",
            )
        )

        if detailed:
            snap = registry.snapshot()
            console.print()
            # The metrics registry is in-memory and per-process.  A CLI
            # invocation creates a fresh process, so uptime is always
            # ~0s and counters/histograms reflect only this single
            # invocation.  Make that explicit so the reader does not
            # mistake the values for cumulative state — the long-running
            # picture lives behind ``vstash serve``'s /metrics endpoint.
            console.print(
                f"[bold cyan]Observability metrics[/bold cyan] "
                f"[dim](this CLI process only — uptime "
                f"{snap['uptime_seconds']:.1f}s; for cumulative state "
                f"see [bold]vstash serve[/bold]'s /metrics)[/dim]"
            )
            if snap["counters"]:
                console.print("\n[bold]Counters:[/bold]")
                for name, value in sorted(snap["counters"].items()):
                    console.print(f"  {name} = {value}")
            if snap["gauges"]:
                console.print("\n[bold]Gauges:[/bold]")
                for name, value in sorted(snap["gauges"].items()):
                    console.print(f"  {name} = {value}")
            if snap["histograms"]:
                console.print("\n[bold]Histograms:[/bold]")
                for name, h in sorted(snap["histograms"].items()):
                    console.print(
                        f"  {name}: count={h['count']} "
                        f"mean={h['mean_ms']:.1f}ms sum={h['sum_ms']:.1f}ms"
                    )


# ------------------------------------------------------------------ #
# vstash forget                                                       #
# ------------------------------------------------------------------ #


@app.command()
def forget(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File path or URL to remove from memory"),
) -> None:
    """Remove a document from memory."""
    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        source = str(Path(path).resolve()) if Path(path).exists() else path
        deleted = store.delete_document(source)

        if deleted:
            console.print(f"[green]✓[/green] Removed [bold]{path}[/bold] from memory.")
        else:
            console.print(f"[yellow]Not found:[/yellow] {path}")


@app.command()
def update(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File path or URL of the document to update"),
    text: str | None = typer.Option(
        None,
        "--text",
        help=(
            "Replace the document content with this text. Triggers re-chunking "
            "and re-embedding. Pass '-' to read from stdin."
        ),
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help="New title for the document.",
    ),
    tags: str | None = typer.Option(
        None,
        "--tags",
        "-t",
        help="New comma-separated tag string. Pass an empty string to clear tags.",
    ),
) -> None:
    """Update an existing document in place.

    Three modes:

      * metadata-only: --title and/or --tags without --text -- a single
        SQL UPDATE, no embedding model invocation.
      * content: --text (with optional --title/--tags) -- re-chunks
        and re-embeds the document.
      * (no field): error.

    For multi-collection stores, the update applies to every collection
    holding `path`. Wrap with `--collection` (future) for narrower
    scopes when that lands.
    """
    if text is None and title is None and tags is None:
        console.print("[red]✗[/red] Pass at least one of --text, --title, or --tags.")
        raise typer.Exit(1)

    if text == "-":
        import sys as _sys

        text = _sys.stdin.read()

    from .memory import Memory

    # The CLI documents "the update applies to every collection
    # holding `path`", so pass ``collection=None`` explicitly to
    # override the Memory-instance default (``"default"``) and update
    # every collection's copy of ``path``.
    with Memory(profile=_profile_from_ctx(ctx)) as mem:
        result = mem.update(path, text=text, title=title, tags=tags, collection=None)

    status = result.get("status")
    mode = result.get("mode")
    if status == "not_found":
        console.print(f"[yellow]Not found:[/yellow] {path}")
        raise typer.Exit(1)
    if status == "noop":
        console.print(f"[yellow]No-op:[/yellow] the supplied text produced no chunks ({path}).")
        raise typer.Exit(1)
    fields = ", ".join(result.get("fields") or [])
    suffix = f" ({result['chunks']} chunks)" if mode == "content" else ""
    console.print(
        f"[green]✓[/green] Updated [bold]{path}[/bold] -- {mode} mode, "
        f"fields: {fields or 'none'}{suffix}."
    )


# ------------------------------------------------------------------ #
# vstash check                                                         #
# ------------------------------------------------------------------ #


@app.command()
def check(
    ctx: typer.Context,
    repair: bool = typer.Option(
        False,
        "--repair",
        help="Apply the safe-to-fix subset of repairs after running checks.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit results as JSON instead of a human-readable table.",
    ),
) -> None:
    """Run database integrity checks and optionally repair (#134).

    Verifies chunk_count parity, vec/snapvec index parity, FTS5 index
    parity, orphan chunks, and SQLite-level PRAGMA integrity_check.
    Exit code is 0 if every check passes, 1 otherwise.
    """
    import json as _json

    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))
    with store:
        results = store.integrity_check()
        repairs: list = []
        post_repair_results: list = []
        if repair:
            repairs = store.integrity_repair()
            # Re-check inside the same connection so the exit code
            # reflects the post-repair state without re-opening the DB.
            post_repair_results = store.integrity_check()

    if json_output:
        payload = {
            "checks": [r.model_dump() for r in results],
            "repairs": [r.model_dump() for r in repairs],
        }
        print(_json.dumps(payload, indent=2))
    else:
        table = Table(title="vstash integrity check", show_lines=False)
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Status")
        table.add_column("Affected", justify="right")
        table.add_column("Detail", overflow="fold")
        for r in results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            table.add_row(
                r.name,
                status,
                str(r.affected_count) if r.affected_count else "—",
                r.detail or r.description,
            )
        console.print(table)
        if repairs:
            rtable = Table(title="repairs", show_lines=False)
            rtable.add_column("Check", style="cyan", no_wrap=True)
            rtable.add_column("Status")
            rtable.add_column("Affected", justify="right")
            rtable.add_column("Detail", overflow="fold")
            for rp in repairs:
                rstatus = "[green]OK[/green]" if rp.success else "[red]FAIL[/red]"
                rtable.add_row(
                    rp.name,
                    rstatus,
                    str(rp.affected_count) if rp.affected_count else "—",
                    rp.detail,
                )
            console.print(rtable)

    failed = [r for r in results if not r.passed]
    if failed and not repair:
        console.print(
            f"[yellow]{len(failed)} check(s) failed.[/yellow] "
            f"Run [bold]vstash check --repair[/bold] to apply safe fixes."
        )
        raise typer.Exit(code=1)
    if failed and repair:
        still_failing = [c for c in post_repair_results if not c.passed]
        if still_failing:
            raise typer.Exit(code=1)


# ------------------------------------------------------------------ #
# vstash config                                                       #
# ------------------------------------------------------------------ #


@app.command(name="config")
def show_config(ctx: typer.Context) -> None:
    """Show current configuration."""
    cfg = load_config()
    from .embed import resolve_backend

    resolved = resolve_backend(cfg.embeddings.backend)
    console.print(
        Panel(
            f"[bold]Inference backend:[/bold] {cfg.inference.backend}\n"
            f"[bold]Model:[/bold] {cfg.inference.model}\n"
            f"[bold]Embedding model:[/bold] {cfg.embeddings.model}\n"
            f"[bold]Embedding backend:[/bold] {resolved} "
            f"({'Apple Silicon GPU' if resolved == 'mlx' else 'ONNX Runtime'})\n"
            f"[bold]Chunk size:[/bold] {cfg.chunking.size} tokens\n"
            f"[bold]Chunk overlap:[/bold] {cfg.chunking.overlap} tokens\n"
            f"[bold]Top-k retrieval:[/bold] {cfg.chunking.top_k}\n"
            f"[bold]Database:[/bold] {cfg.storage.db_path}\n"
            f"[bold]Cerebras key:[/bold] {'set ✓' if cfg.cerebras_api_key else 'not set ✗'}\n"
            f"[bold]OpenAI key:[/bold] {'set ✓' if cfg.openai_api_key else 'not set ✗'}",
            title="[bold cyan]vstash Config[/bold cyan]",
            border_style="cyan",
        )
    )


# ------------------------------------------------------------------ #
# vstash reindex                                                       #
# ------------------------------------------------------------------ #


@app.command()
def reindex(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="New embedding model (default: use vstash.toml setting)",
    ),
    batch_size: int = typer.Option(256, "--batch-size", help="Chunks per embedding batch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Re-embed all chunks with a different embedding model.

    Use this after changing the embedding model in vstash.toml
    (e.g., switching to a multilingual model). All existing vector
    embeddings are recomputed — text, FTS, and metadata are preserved.
    """
    from .embed import embed_texts, get_embedding_dim

    cfg = load_config()
    target_model = model or cfg.embeddings.model
    new_dim = get_embedding_dim(target_model)

    # Warm up to get accurate model name display
    warmup(target_model, cfg.embeddings.backend)

    # Open store with current dim (to read existing data)
    current_dim = get_embedding_dim(cfg.embeddings.model) if model else new_dim
    store = open_store_for_config(
        cfg,
        profile=_profile_from_ctx(ctx),
        embedding_dim=current_dim,
    )

    with store:
        total = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            console.print("[yellow]No chunks to reindex.[/yellow]")
            raise typer.Exit()

        console.print(
            f"[bold]Reindex plan:[/bold]\n"
            f"  Model:  {target_model}\n"
            f"  Dims:   {new_dim}\n"
            f"  Chunks: {total}\n"
            f"  Batch:  {batch_size}"
        )

        if not yes:
            typer.confirm("This will re-embed all chunks. Continue?", abort=True)

        from rich.progress import Progress

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Re-embedding chunks...", total=total)

            def _progress(processed: int, _total: int) -> None:
                progress.update(task, completed=processed)

            def _embed_batch(texts: list[str]) -> list[list[float]]:
                return embed_texts(texts, target_model, cfg.embeddings.backend)

            count = store.reindex(
                embed_fn=_embed_batch,
                new_dim=new_dim,
                batch_size=batch_size,
                progress_cb=_progress,
            )

        console.print(
            f"[bold green]Done![/bold green] Re-embedded {count} chunks "
            f"with {target_model} ({new_dim} dims)"
        )


# ------------------------------------------------------------------ #
# vstash watch                                                         #
# ------------------------------------------------------------------ #


@app.command()
def watch(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(..., help="Directories to watch"),
    collection: str = typer.Option(
        "default", "--collection", "-c", help="Collection for ingested files"
    ),
    ext: str | None = typer.Option(
        None, "--ext", help="Comma-separated extensions, e.g. .md,.txt,.py"
    ),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait before re-ingesting"),
) -> None:
    """Watch directories for changes and auto-ingest files."""
    from .watch import start_watch

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    extensions: frozenset[str] | None = None
    if ext:
        extensions = frozenset(
            (part if part.startswith(".") else f".{part}")
            for raw in ext.split(",")
            if (part := raw.strip().lower())
        )

    with store:
        start_watch(
            paths,
            cfg,
            store,
            collection=collection,
            extensions=extensions,
            debounce_s=debounce,
        )


@app.command()
def serve(
    ctx: typer.Context,
    port: int = typer.Option(8585, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind to. WARNING: 0.0.0.0 exposes all endpoints without authentication.",
    ),
    warm: bool = typer.Option(
        True,
        "--warm/--no-warm",
        help="Pre-load embedding model at startup (eliminates first-query cold start)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Expose the /debug/why route (miss analysis as JSON). Off by "
        "default because the endpoint echoes arbitrary query text back. "
        "Intended for local diagnostic use only. Issue #157 part 2.",
    ),
) -> None:
    """Launch the vstash web interface -- a pocket memory agent.

    Opens a browser-based chat and search interface on localhost.
    Chat with your documents, search your memory, upload files.

    The --warm flag (on by default) pre-loads the embedding model so
    the first query is fast.  The /api/embed endpoint allows CLI and
    SDK clients to use the running server's embedder instead of loading
    their own model, eliminating cold start for all clients.
    """
    # Friendly error if the serve extras (uvicorn / starlette) aren't
    # installed.  Without this catch, the user gets a raw ModuleNotFoundError
    # traceback that doesn't tell them which extra to install.
    try:
        import uvicorn

        from .web import create_app
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "starlette/uvicorn"
        console.print(
            f"[red]✗[/red] vstash serve requires the [bold]serve[/bold] extra (missing: {missing})."
        )
        console.print("  Install with: [bold]pip install 'vstash\\[serve]'[/bold]")
        raise typer.Exit(code=1) from exc

    # Prevent the daemon from delegating to itself via the embed client.
    import os

    os.environ["_VSTASH_IS_DAEMON"] = "1"

    # Also update the module-level flag in case embed.py was already imported.
    import vstash.embed as _embed_mod

    _embed_mod._is_daemon = True

    if warm:
        import threading

        from .config import load_config
        from .embed import warmup

        cfg = load_config()
        console.print("[dim]Warming up embedding model...[/dim]")

        def _warmup_safe():
            try:
                warmup(cfg.embeddings.model)
            except Exception as exc:
                console.print(f"[yellow]Warning: warmup failed: {exc}[/yellow]")

        t = threading.Thread(target=_warmup_safe, daemon=True)
        t.start()

    if host == "0.0.0.0":
        console.print(
            "[yellow]Warning: binding to 0.0.0.0 exposes all endpoints "
            "(search, chat, embed, upload) without authentication.[/yellow]"
        )

    console.print(f"[bold cyan]vstash[/bold cyan] serving at [link]http://{host}:{port}[/link]")
    if debug:
        console.print(
            "[yellow]! debug mode: /debug/why is exposed. Do not use in production.[/yellow]"
        )
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    uvicorn.run(create_app(debug=debug), host=host, port=port, log_level="warning")


@app.command()
def why(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="The search query that missed. Omit when using --recent.",
    ),
    recent: int = typer.Option(
        0,
        "--recent",
        "-r",
        help="Instead of running a new miss analysis, list the N most "
        "recent search events that carry an auto-logged miss_hint "
        "(empty / all-low results). 0 = disabled. Issue #157 part 3.",
    ),
    expect: str | None = typer.Option(
        None,
        "--expect",
        "-e",
        help="Path of the document you expected to see. Either this or "
        "--expect-chunk-id must be given.",
    ),
    expect_chunk_id: int | None = typer.Option(
        None,
        "--expect-chunk-id",
        help="Specific chunk id to track instead of resolving from a path. "
        "Useful when you already have a chunk id from an earlier search.",
    ),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Top-k window to check against (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit the raw MissAnalysis as JSON (for scripting / piping). "
        "Default output is a rich table plus a suggestions panel.",
    ),
) -> None:
    """Diagnose why an expected document did not appear in search results.

    Runs ``VstashStore.miss_analysis()`` and prints a stage-by-stage trace
    showing where the expected chunk was eliminated from the ranking, plus
    rule-based suggestions for fixing the query.

    Examples:

        vstash why "rate limits" --expect notes/api-design.md

        vstash why "renewable energy" --expect papers/2024/clean.pdf --top-k 10

        vstash why "quota" --expect-chunk-id 4213 --json | jq .suggestions
    """
    import json as _json
    import sqlite3 as _sqlite3

    def _why_error(msg: str, exit_code: int = 1) -> typer.Exit:
        """Emit an error consistently for both --json and pretty modes.
        Mirrors the pattern in ``vstash search --miss`` so script
        consumers piping ``vstash why --json`` always see JSON, even on
        the error path. ``_safe_exc`` strips Rich markup so paths like
        ``foo[bar].md`` do not get mangled by the markup parser."""
        if json_out:
            print(_json.dumps({"error": msg}))
        else:
            console.print(f"[red]x[/red] {_safe_exc(msg)}")
        return typer.Exit(exit_code)

    # --recent mode: dump the N most recent miss_hint rows and exit.
    # Issue #157 part 3: surfaces recent miss hints stored in the DB
    # for this profile (the ``search_events`` table keeps the last
    # 1000 rows, including prior runs). Users see which queries
    # recently missed without having to remember them.
    if recent < 0:
        raise _why_error(f"--recent must be >= 0 (0 = disabled), got {recent}.")
    if recent > 0:
        _, store = _get_store(warm=False, profile=_profile_from_ctx(ctx))
        with store:
            hints = store.recent_miss_hints(limit=recent)
        if json_out:
            print(_json.dumps({"recent_miss_hints": hints}, indent=2, default=str))
            raise typer.Exit(0)
        if not hints:
            console.print(
                "[dim]No recent miss_hints recorded. "
                "Run a search that returns empty / all-low results, "
                "or verify [observability] auto_miss_hint is enabled.[/dim]"
            )
            raise typer.Exit(0)
        console.print(f"[bold]{len(hints)} recent miss hint(s)[/bold]")
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("When", style="dim")
        table.add_column("Query", overflow="fold")
        table.add_column("Reason")
        table.add_column("Tier")
        table.add_column("Best dist", justify="right")
        table.add_column("Results", justify="right")
        for h in hints:
            reason = h["miss_hint"].get("reason", "-")
            table.add_row(
                h["created_at"][:19],  # truncate to seconds
                h["query"][:80],
                reason,
                h["relevance_tier"],
                f"{h['best_distance']:.4f}",
                str(h["result_count"]),
            )
        console.print(table)
        console.print()
        console.print(
            "[dim]Drill into any of these with:[/dim] "
            '[bold]vstash why "<query>" --expect <path>[/bold]'
        )
        raise typer.Exit(0)

    if query is None:
        raise _why_error("Missing argument QUERY. Pass a query or use --recent N.")

    if expect is None and expect_chunk_id is None:
        raise _why_error("Provide either --expect <path> or --expect-chunk-id <id>.")

    # ``why`` is a diagnostic tool; don't eagerly warm the embedder.
    # ``embed_query`` will load on demand if needed. Users running
    # ``vstash why`` against a misconfigured model get the actual error
    # instead of a cold-start failure that masks the diagnostic.
    cfg, store = _get_store(warm=False, profile=_profile_from_ctx(ctx))

    # Normalize --expect the same way ``vstash search --miss`` and
    # ``Memory.miss_analysis`` do: http(s)/text URIs pass through,
    # everything else resolves to an absolute path. Without this,
    # ``vstash why "q" --expect notes/doc.md`` would lookup the
    # literal relative string against a DB that stores absolute paths
    # and always report "No chunks found".
    expected_path_arg: str | None = None
    if expect is not None:
        if expect.startswith(("http://", "https://", "text://")):
            expected_path_arg = expect
        else:
            expected_path_arg = str(Path(expect).resolve(strict=False))

    effective_top_k = top_k or cfg.chunking.top_k

    with store:
        try:
            q_embedding = embed_query(query, cfg.embeddings.model)
            analysis = store.miss_analysis(
                query_embedding=q_embedding,
                query_text=query,
                expected_path=expected_path_arg,
                expected_chunk_id=expect_chunk_id,
                top_k=effective_top_k,
                collection=collection,
                project=project,
                layer=layer,
            )
        except ValueError as exc:
            # ValueError covers the documented miss_analysis failures
            # (unknown path / chunk id, no expected provided) plus
            # LimitError which subclasses ValueError.
            raise _why_error(str(exc)) from exc
        except _sqlite3.Error as exc:
            # Corrupt / locked DB surfaces here from the search() step
            # inside miss_analysis. Keep the user on the --json contract.
            raise _why_error(f"database error: {exc}") from exc
        except (RuntimeError, OSError) as exc:
            # embed_query failure modes: ONNX load failure, daemon
            # socket error, model download failure, etc.
            raise _why_error(f"embedder error: {exc}") from exc

    if json_out:
        # exclude_none strips empty-but-optional fields; indent=2 matches
        # the search --miss JSON contract. Exit code mirrors pretty-mode
        # (2 when dropped) so script consumers get reliable CI checks.
        print(_json.dumps(analysis.model_dump(exclude_none=True), indent=2, default=str))
        raise typer.Exit(0 if analysis.appeared_in_results else 2)

    # Header.
    # Ranks are 0-indexed internally; display 1-indexed to match ``vstash search``.
    if analysis.appeared_in_results:
        assert analysis.final_rank is not None
        console.print(
            f"[green]✓[/green] [bold]Expected doc IS in the top-{analysis.top_k_requested}[/bold] "
            f"(rank {analysis.final_rank + 1})."
        )
    else:
        console.print(
            f"[red]✗[/red] [bold]Expected doc NOT in top-{analysis.top_k_requested}.[/bold]"
        )
        if analysis.dropped_at:
            console.print(f"  Dropped at stage: [yellow bold]{analysis.dropped_at}[/yellow bold]")

    console.print()
    console.print(f"  [dim]Query:[/dim]          {analysis.query!r}")
    if analysis.expected_path:
        console.print(f"  [dim]Expected path:[/dim]  {analysis.expected_path}")
    if analysis.expected_chunk_id is not None:
        console.print(
            f"  [dim]Chunk id:[/dim]       {analysis.expected_chunk_id} "
            f"([dim]{analysis.target_resolution}[/dim])"
        )
    if analysis.total_chunks_in_doc > 1:
        console.print(f"  [dim]Doc has[/dim]         {analysis.total_chunks_in_doc} chunks")

    # Per-stage trace.
    console.print()
    console.print("[bold]Pipeline trace[/bold]")
    trace_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    trace_table.add_column("Stage", style="cyan")
    trace_table.add_column("Passed")
    trace_table.add_column("Rank", justify="right")
    trace_table.add_column("Score", justify="right")
    trace_table.add_column("Detail", overflow="fold")
    for v in analysis.stage_verdicts:
        passed_cell = "[green]yes[/green]" if v.passed else "[red]no[/red]"
        drop_style = "red bold" if v.stage == analysis.dropped_at else ""
        stage_cell = f"[{drop_style}]{v.stage}[/{drop_style}]" if drop_style else v.stage
        trace_table.add_row(
            stage_cell,
            passed_cell,
            "-" if v.rank is None else str(v.rank + 1),
            "-" if v.score is None else f"{v.score:.4f}",
            v.detail,
        )
    console.print(trace_table)

    # Actual top-k for contrast.
    if analysis.actual_top_k:
        console.print()
        console.print("[bold]Actual top-k[/bold]")
        top_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        top_table.add_column("Rank", justify="right")
        top_table.add_column("Path", overflow="fold")
        top_table.add_column("Title", overflow="fold")
        top_table.add_column("Score", justify="right")
        for r in analysis.actual_top_k:
            top_table.add_row(str(r.rank + 1), r.path, r.title or "-", f"{r.score:.4f}")
        console.print(top_table)

    # Suggestions panel last so it lands near the user's prompt.
    if analysis.suggestions:
        console.print()
        console.print(
            Panel.fit(
                "\n".join(f"- {s}" for s in analysis.suggestions),
                title="[bold]Suggestions[/bold]",
                border_style="yellow",
            )
        )

    if not analysis.appeared_in_results:
        raise typer.Exit(code=2)


@app.command()
def remember(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="Text to ingest (or pipe via stdin)"),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Title for the document (auto-generated if omitted)"
    ),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to add to"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project tag"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Layer tag"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Override chunk size in tokens"
    ),
    chunk_overlap: int | None = typer.Option(
        None, "--chunk-overlap", help="Override chunk overlap in tokens"
    ),
) -> None:
    """Ingest text directly — no file needed.

    Pass text as an argument, or pipe it via stdin:

        vstash remember "The API uses OAuth2 with PKCE" --title "auth-notes"
        echo "deployment steps..." | vstash remember --title "deploy"
        cat notes.md | vstash remember --title "meeting-notes" --project myproj
    """
    import sys

    # Read from argument or stdin
    if text is None:
        if sys.stdin.isatty():
            console.print("[red]✗ No text provided. Pass as argument or pipe via stdin.[/red]")
            raise typer.Exit(1)
        text = sys.stdin.read()

    if not text.strip():
        console.print("[yellow]⚠ Empty text, nothing to ingest.[/yellow]")
        raise typer.Exit(1)

    from .ingest import ingest_text

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))
    meta = {"project": project, "layer": layer, "tags": tags}

    with store:
        result = ingest_text(
            text,
            cfg,
            store,
            title=title,
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **meta,
        )

        if result.status == "ok":
            parts = (
                f"[green]✓[/green] [bold]{result.title}[/bold] — "
                f"{result.chunks} chunks in {result.elapsed_s}s"
            )
            if collection != "default":
                parts += f" [cyan]→ {collection}[/cyan]"
            console.print(parts)
        else:
            console.print(f"[yellow]⚠ {result.status}: {result.source}[/yellow]")


# ------------------------------------------------------------------ #
# vstash profile                                                      #
# ------------------------------------------------------------------ #

profile_app = typer.Typer(
    name="profile",
    help="Manage named profiles (isolated databases).",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(profile_app, name="profile")


@profile_app.command(name="list")
def profile_list() -> None:
    """List all named profiles."""
    from .profile import list_profiles as _list_profiles

    profiles = _list_profiles()
    if not profiles:
        console.print("[dim]No profiles yet. Create one with: vstash profile create <name>[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Profile", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    from .profile import PROFILES_DIR

    for name in profiles:
        db_path = PROFILES_DIR / name / "memory.db"
        size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
        table.add_row(name, f"{size_mb:.1f} MB", str(db_path))

    console.print(table)


@app.command()
def retrain(
    ctx: typer.Context,
    output: str = typer.Option(
        "~/.vstash/models/retrained",
        "--output",
        "-o",
        help="Where to save the fine-tuned model",
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick mode: 1 epoch, 1000 queries, higher LR"
    ),
    max_queries: int = typer.Option(
        5000, "--max-queries", help="Maximum pseudo-queries to generate"
    ),
    epochs: int = typer.Option(2, "--epochs", help="Training epochs"),
    lr: float = typer.Option(3e-6, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(64, "--batch-size", help="Training batch size"),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Base model to fine-tune (default: current config model)"
    ),
    min_gain: float = typer.Option(
        0.0,
        "--min-gain",
        help="Required NDCG@10 improvement over baseline (0.0 = no regression). "
        "Pass a negative value to always save.",
    ),
    no_eval: bool = typer.Option(
        False, "--no-eval", help="Skip eval gate entirely and save unconditionally"
    ),
    eval_fraction: float = typer.Option(
        0.15, "--eval-fraction", help="Fraction of corpus reserved for held-out eval"
    ),
    eval_noise_size: int = typer.Option(
        1000,
        "--eval-noise",
        help="Distractor chunks added to the eval index (higher = stricter eval)",
    ),
    synthesize: bool = typer.Option(
        False,
        "--synthesize-queries/--no-synthesize-queries",
        help="Use the configured LLM backend to generate short realistic "
        "queries for each training chunk (InPars-style). Closes the "
        "chunk-prefix vs natural-query distribution gap.",
    ),
    synth_n: int = typer.Option(
        2,
        "--synth-n",
        help="Queries synthesized per chunk when --synthesize-queries is on.",
    ),
    synth_cache: str | None = typer.Option(
        None,
        "--synth-cache",
        help="JSONL cache file for synthesized queries. Reusing the same "
        "path across runs avoids re-calling the LLM on unchanged chunks.",
    ),
    synth_model: str | None = typer.Option(
        None,
        "--synth-model",
        help="Model name override for synthesis (defaults to the configured "
        "inference backend's model).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Deterministic seed. Controls held-out split, triple sampling, "
        "torch dropout, and DataLoader shuffle, so two runs with the same "
        "inputs and seed produce identical models.",
    ),
    training_queries: str | None = typer.Option(
        None,
        "--training-queries",
        help="Path to a JSONL file of labeled training queries (shape per "
        "line: {query, relevant_paths}). When set, routes through the "
        "labeled-batched miner (v5 recipe) instead of chunk-prefix "
        "pseudo-queries. H-R8 (2026-04-21).",
    ),
    eval_queries_path: str | None = typer.Option(
        None,
        "--eval-queries",
        help="Path to a JSONL file of labeled eval queries (same shape as "
        "--training-queries). When set, the eval gate scores baseline "
        "and fine-tuned models on these real queries instead of the "
        "internal chunk-prefix split, so the gate reflects the corpus "
        "you actually care about (BEIR qrels, search logs, manual "
        "annotations). Mutually exclusive with --no-eval. Ignores "
        "--eval-fraction (the held-out set is the file you provide).",
    ),
    training_pair_source: str = typer.Option(
        "auto",
        "--training-pair-source",
        help="Resolution policy when --training-queries is not set. "
        "'auto' (default, H-R8) reuses --eval-queries as training "
        "queries when present, falling back to chunk-prefix otherwise. "
        "'labeled' errors if no labels. 'prefix' forces chunk-prefix.",
        click_type=click.Choice(["auto", "labeled", "prefix"]),
    ),
    bulk_mine: bool = typer.Option(
        False,
        "--bulk-mine/--no-bulk-mine",
        help="Route chunk-prefix mining through the GPU-batched miner "
        "(retrain_batch.generate_triples_batched). 20-50x faster than "
        "the default per-query path on FiQA-sized corpora, at the cost "
        "of holding the full corpus in GPU memory for a moment. "
        "Ignored when labeled queries are used (labeled path is always batched).",
    ),
    bulk_mine_device: str | None = typer.Option(
        None,
        "--bulk-mine-device",
        help="Device override for --bulk-mine / the labeled-batched miner "
        "('cuda' or 'cpu'). Leave unset to auto-detect.",
    ),
) -> None:
    """Fine-tune the embedding model using your own data.

    Analyzes where vector and keyword search disagree on your corpus,
    generates training pairs from those disagreements, and fine-tunes
    the embedding model to better understand your data. No human labels
    needed.

    By default the fine-tuned candidate is evaluated honestly on a
    held-out slice of the corpus (reindexed with the new model) and
    only saved if NDCG@10 meets or exceeds ``--min-gain`` over the
    baseline. Use ``--no-eval`` to skip the gate.

    Requires: pip install sentence-transformers torch

    After training, use the model with:
        vstash reindex --model <output-path>
    """
    try:
        from .retrain import retrain as run_retrain
    except ImportError as exc:
        console.print(
            "[red]x[/red] vstash retrain requires sentence-transformers, torch, "
            "and accelerate. Install with: "
            "[bold]pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))
    model_name = base_model or cfg.embeddings.model

    if quick:
        max_queries = min(max_queries, 1000)
        epochs = 1
        lr = 5e-6

    stats = store.stats()
    if stats.chunks < 10:
        console.print(
            "[yellow]! Your store has fewer than 10 chunks. "
            "Add more documents before retraining.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print("[bold cyan]vstash retrain[/bold cyan]")
    console.print(f"  Store:        {stats.documents} docs, {stats.chunks} chunks")
    console.print(f"  Base model:   {model_name}")
    console.print(f"  Max queries:  {max_queries}")
    if synthesize:
        console.print(
            f"  Query source: [cyan]LLM-synthesized[/cyan] "
            f"(n={synth_n}" + (f", cache={synth_cache}" if synth_cache else "") + ")"
        )
    else:
        console.print("  Query source: chunk prefix (legacy)")
    if no_eval:
        console.print("  Eval gate:    [yellow]disabled[/yellow]")
    else:
        console.print(
            f"  Eval gate:    min-gain={min_gain:+.4f}, "
            f"fraction={eval_fraction:.2f}, noise={eval_noise_size}"
        )
    console.print()

    loaded_training_queries: list[dict] | None = None
    if training_queries:
        loaded_training_queries = _load_labeled_queries_jsonl(
            training_queries, flag="--training-queries"
        )
        console.print(
            f"  Training queries: [cyan]{len(loaded_training_queries)} labeled[/cyan] "
            f"from {training_queries}"
        )

    loaded_eval_queries: list[dict] | None = None
    if eval_queries_path:
        if no_eval:
            console.print(
                "[red]x[/red] --eval-queries and --no-eval are mutually exclusive. Drop one."
            )
            raise typer.Exit(code=1)
        loaded_eval_queries = _load_labeled_queries_jsonl(eval_queries_path, flag="--eval-queries")
        # Warn when the eval queries reference paths absent from the
        # store.  The retrain pipeline silently scores those queries as
        # 0 (no relevant chunk to retrieve), which would invalidate
        # NDCG@10 without surfacing a single error.
        #
        # Query only the paths the user actually referenced -- a full
        # ``SELECT path FROM documents`` would pull the entire corpus
        # into memory on stores with millions of docs (MS MARCO scale,
        # production search logs).  Eval sets are typically <10k unique
        # paths so a chunked ``WHERE path IN (...)`` is O(eval_size) on
        # the indexed ``documents.path`` instead of O(corpus_size).
        eval_paths = sorted({p for q in loaded_eval_queries for p in q["relevant_paths"]})
        existing_paths: set[str] = set()
        # 500 keeps the IN-list well below SQLite's default
        # ``SQLITE_LIMIT_VARIABLE_NUMBER`` (999) with headroom.
        chunk_size = 500
        for i in range(0, len(eval_paths), chunk_size):
            chunk = eval_paths[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            existing_paths.update(
                row[0]
                for row in store._conn.execute(
                    f"SELECT path FROM documents WHERE path IN ({placeholders})", chunk
                ).fetchall()
            )
        missing = sorted(set(eval_paths) - existing_paths)
        if missing:
            preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            console.print(
                f"[yellow]! --eval-queries: {len(missing)} relevant_paths reference "
                f"docs not in this store ({preview}). Those queries will score 0; "
                "ingest the missing docs first or remove them from the eval file.[/yellow]"
            )
        console.print(
            f"  Eval queries:     [cyan]{len(loaded_eval_queries)} labeled[/cyan] "
            f"from {eval_queries_path} (gate uses these instead of internal split)"
        )

    result = run_retrain(
        store,
        base_model=model_name,
        output_path=output,
        max_queries=max_queries,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        eval_fraction=eval_fraction,
        eval_noise_size=eval_noise_size,
        min_gain=min_gain,
        skip_eval=no_eval,
        seed=seed,
        eval_queries=loaded_eval_queries,
        synthesize_queries=synthesize,
        synth_n=synth_n,
        synth_cache=synth_cache,
        synth_model=synth_model,
        training_queries=loaded_training_queries,
        training_pair_source=training_pair_source,  # type: ignore[arg-type]
        bulk_mine=bulk_mine,
        bulk_mine_device=bulk_mine_device,
        cfg=cfg,
    )

    if result.n_pairs == 0:
        console.print(
            "[yellow]! No training pairs generated. "
            "Your corpus may be too small or too homogeneous.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print(f"  Training pairs: {result.n_pairs}")

    if result.baseline is not None and result.final is not None:
        console.print()
        console.print("[bold]Eval results[/bold]")
        console.print(f"  Queries:          {result.baseline.n_queries}")
        console.print(f"  Baseline NDCG@10: {result.baseline.ndcg_at_10:.4f}")
        console.print(f"  Final NDCG@10:    {result.final.ndcg_at_10:.4f}")
        delta = result.delta_ndcg
        color = "green" if delta >= 0 else "red"
        console.print(f"  Delta NDCG@10:    [{color}]{delta:+.4f}[/{color}]")
        console.print(f"  Baseline NDCG@3:  {result.baseline.ndcg_at_3:.4f}")
        console.print(f"  Final NDCG@3:     {result.final.ndcg_at_3:.4f}")
        console.print(f"  Baseline MRR:     {result.baseline.mrr:.4f}")
        console.print(f"  Final MRR:        {result.final.mrr:.4f}")
        console.print(f"  Baseline Recall@100: {result.baseline.recall_at_100:.4f}")
        console.print(f"  Final Recall@100:    {result.final.recall_at_100:.4f}")

    if result.gated_out:
        console.print()
        if result.final is None:
            # No candidate was trained (too few pairs). Nothing to promote
            # or inspect -- the retrain simply did not produce a model.
            console.print(
                "[red]Training skipped[/red]: not enough training pairs to "
                "fine-tune. Your corpus may be too small or too homogeneous."
            )
        else:
            console.print(
                f"[red]Gated out[/red]: delta NDCG@10 did not meet min-gain "
                f"({result.min_gain:+.4f}). Candidate left at "
                f"[dim]{Path(output).expanduser()}.candidate[/dim] for inspection."
            )
        raise typer.Exit(code=2)

    if result.output_path is None:
        # Defensive: any future RetrainResult path that returns None for
        # output_path without setting gated_out=True would land here.
        console.print("[red]x[/red] retrain did not save a model. See logs for details.")
        raise typer.Exit(code=2)

    console.print()
    console.print(f"[green]Model saved to:[/green] {result.output_path}")
    console.print()
    console.print("To use the retrained model:")
    console.print(f"  [bold]vstash reindex --model {result.output_path}[/bold]")


@app.command(name="retrain-multi")
def retrain_multi_cmd(
    ctx: typer.Context,
    store_spec: list[str] = typer.Option(
        [],
        "--store",
        "-s",
        help="Extra corpus in the form NAME=PATH. Repeat for each dataset. "
        "The current profile's store is also included, aliased by its "
        "config name (or 'primary' if unnamed) unless --exclude-primary is passed.",
    ),
    exclude_primary: bool = typer.Option(
        False,
        "--exclude-primary",
        help="Do not include the current profile's store in the multi-corpus mix. "
        "Use this when --store alone covers all datasets you want to train on.",
    ),
    output: str = typer.Option(
        "~/.vstash/models/retrained-multi",
        "--output",
        "-o",
        help="Where to save the fine-tuned model",
    ),
    sampling_strategy: str = typer.Option(
        "temperature",
        "--sampling-strategy",
        help="Temperature (default) damps the largest corpus toward a more balanced triple budget.",
        click_type=click.Choice(["uniform", "proportional", "temperature"]),
    ),
    sampling_temperature: float = typer.Option(
        0.5,
        "--sampling-temperature",
        help="Exponent for the temperature strategy. 0=uniform, 1=proportional, "
        "0.5=favours smaller corpora without abandoning size signal.",
    ),
    total_triples: int = typer.Option(
        10000,
        "--total-triples",
        help="Target total pseudo-query budget across all datasets. Actual "
        "pair count may be lower when corpora are smaller than their share.",
    ),
    epochs: int = typer.Option(2, "--epochs", help="Training epochs"),
    lr: float = typer.Option(3e-6, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Training batch size. Default 32 keeps a 15 GB T4 in range when "
        "three corpora are live during eval; raise on bigger GPUs.",
    ),
    use_amp: bool = typer.Option(
        True,
        "--use-amp/--no-use-amp",
        help="Automatic mixed precision during training. Near-free and halves GPU memory on T4+.",
    ),
    max_seq_length: int | None = typer.Option(
        None,
        "--max-seq-length",
        help="Cap the encoder's max sequence length. Lower values (256, 128) "
        "further reduce memory when most chunks are short.",
    ),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Base model to fine-tune (default: current config model)"
    ),
    min_gain: float = typer.Option(
        0.0,
        "--min-gain",
        help="Required NDCG@10 improvement over baseline. Applied to the "
        "macro-average unless --per-dataset-gate is passed.",
    ),
    per_dataset_gate: bool = typer.Option(
        False,
        "--per-dataset-gate",
        help="Require every dataset individually to clear --min-gain. "
        "Stricter than the default macro-average gate.",
    ),
    no_eval: bool = typer.Option(
        False, "--no-eval", help="Skip eval gate entirely and save unconditionally"
    ),
    eval_fraction: float = typer.Option(
        0.15, "--eval-fraction", help="Fraction of each corpus reserved for held-out eval"
    ),
    eval_noise_size: int = typer.Option(
        1000,
        "--eval-noise",
        help="Distractor chunks added to each eval index (higher = stricter eval)",
    ),
    bulk_mine: bool = typer.Option(
        False,
        "--bulk-mine/--no-bulk-mine",
        help="Route triple generation through the GPU-batched miner "
        "(retrain_batch.generate_triples_batched). 20-50x faster than "
        "the default per-query path on FiQA-sized corpora, at the cost "
        "of holding the full corpus in GPU memory for a moment.",
    ),
    bulk_eval: bool = typer.Option(
        False,
        "--bulk-eval/--no-bulk-eval",
        help="Route baseline + final eval through the GPU-batched "
        "evaluator (retrain_batch.evaluate_model_batched). Biggest win "
        "when --eval-noise is large (e.g. >= 10k). Pair with --bulk-mine "
        "for end-to-end speedup on Colab T4.",
    ),
    bulk_mine_device: str | None = typer.Option(
        None,
        "--bulk-mine-device",
        help="Device override for --bulk-mine / --bulk-eval ('cuda' or "
        "'cpu'). Leave unset to auto-detect.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Deterministic seed. Derived per-dataset via SHA-256 so each "
        "corpus gets its own stable RNG, and then threaded into train_mnrl "
        "so DataLoader shuffles and torch dropout are reproducible.",
    ),
    training_pair_source: str = typer.Option(
        "auto",
        "--training-pair-source",
        help="Resolution policy for per-dataset training queries. "
        "'auto' (default, H-R1) reuses eval labels as training queries "
        "when present, falling back to chunk-prefix otherwise. "
        "'labeled' errors if any dataset lacks labels. "
        "'prefix' forces chunk-prefix even when eval labels exist.",
        click_type=click.Choice(["auto", "labeled", "prefix"]),
    ),
) -> None:
    """Fine-tune the embedding model over multiple corpora with balanced sampling.

    Wrapper around ``vstash.retrain.retrain_multi``. Give it one or more
    vstash stores (each one a separate corpus) plus the current
    profile's store; retrain-multi computes a per-dataset triple budget,
    generates (query, positive, hard_neg) triples from each corpus,
    shuffles them globally, and fine-tunes a single embedding model.

    Per-dataset NDCG@10 is reported before and after. The macro-average
    gate (or --per-dataset-gate) prevents a regressed model from being
    promoted over your current one.

    Example:
        vstash retrain-multi \\
            --store nfcorpus=/data/vstash-nfcorpus.db \\
            --store fiqa=/data/vstash-fiqa.db \\
            --sampling-strategy temperature \\
            --sampling-temperature 0.5 \\
            --total-triples 30000 \\
            --output ~/.vstash/models/multi-tuned

    Requires: pip install sentence-transformers torch
    """
    try:
        from .retrain import retrain_multi as run_retrain_multi
    except ImportError as exc:
        console.print(
            "[red]x[/red] vstash retrain-multi requires sentence-transformers, torch, "
            "and accelerate. Install with: "
            "[bold]pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    from .embed import get_embedding_dim as _get_dim

    cfg, primary_store = _get_store(profile=_profile_from_ctx(ctx))
    model_name = base_model or cfg.embeddings.model

    # Nudge when --bulk-mine-device is set without either batched path.
    # The device override would otherwise be silently ignored.
    if bulk_mine_device and not (bulk_mine or bulk_eval):
        console.print(
            "[yellow]! --bulk-mine-device was passed without --bulk-mine or "
            "--bulk-eval. The device override is ignored; pass one of the "
            "--bulk-* flags to enable the GPU-batched path.[/yellow]"
        )

    stores: dict[str, VstashStore] = {}
    opened_extra: list[VstashStore] = []
    if not exclude_primary:
        primary_alias = _profile_from_ctx(ctx) or "primary"
        stores[primary_alias] = primary_store

    try:
        dim = _get_dim(cfg.embeddings.model)
        for spec in store_spec:
            if "=" not in spec:
                console.print(
                    f"[red]x[/red] Invalid --store spec [bold]{spec}[/bold]. Expected NAME=PATH."
                )
                raise typer.Exit(code=1)
            alias, _, path = spec.partition("=")
            alias = alias.strip()
            path = path.strip()
            if not alias or not path:
                console.print(
                    f"[red]x[/red] Invalid --store spec [bold]{spec}[/bold]. "
                    "Both NAME and PATH must be non-empty."
                )
                raise typer.Exit(code=1)
            if alias in stores:
                console.print(
                    f"[red]x[/red] Duplicate store alias [bold]{alias}[/bold]. "
                    "Each --store must use a unique name."
                )
                raise typer.Exit(code=1)
            extra_store = open_store_for_config(cfg, db_path=path, embedding_dim=dim)
            opened_extra.append(extra_store)
            stores[alias] = extra_store

        if not stores:
            console.print(
                "[red]x[/red] No stores to train on. "
                "Pass one or more --store NAME=PATH, or remove --exclude-primary."
            )
            raise typer.Exit(code=1)

        # Quick sanity check: each store should have at least a few chunks,
        # otherwise we print a hint rather than failing inside retrain_multi.
        sizes = {name: s.stats().chunks for name, s in stores.items()}
        if all(n < 10 for n in sizes.values()):
            console.print(
                "[yellow]! Every provided store has fewer than 10 chunks. "
                "Add more documents before retraining.[/yellow]"
            )
            raise typer.Exit(code=1)

        console.print("[bold cyan]vstash retrain-multi[/bold cyan]")
        for name, s in stores.items():
            stats = s.stats()
            console.print(f"  {name}: {stats.documents} docs, {stats.chunks} chunks")
        console.print(f"  Base model:       {model_name}")
        console.print(
            f"  Sampling:         {sampling_strategy} (temperature={sampling_temperature})"
        )
        console.print(f"  Total triples:    {total_triples}")
        if no_eval:
            console.print("  Eval gate:        [yellow]disabled[/yellow]")
        else:
            gate_mode = "per-dataset" if per_dataset_gate else "macro-average"
            console.print(
                f"  Eval gate:        {gate_mode}, min-gain={min_gain:+.4f}, "
                f"fraction={eval_fraction:.2f}, noise={eval_noise_size}"
            )
        console.print()

        result = run_retrain_multi(
            stores,
            base_model=model_name,
            output_path=output,
            sampling=sampling_strategy,  # type: ignore[arg-type]
            temperature=sampling_temperature,
            total_triples=total_triples,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            use_amp=use_amp,
            max_seq_length=max_seq_length,
            eval_fraction=eval_fraction,
            eval_noise_size=eval_noise_size,
            min_gain=min_gain,
            per_dataset_gate=per_dataset_gate,
            skip_eval=no_eval,
            seed=seed,
            bulk_mine=bulk_mine,
            bulk_mine_device=bulk_mine_device,
            bulk_eval=bulk_eval,
            training_pair_source=training_pair_source,  # type: ignore[arg-type]
            cfg=cfg,
        )
    finally:
        for s in opened_extra:
            try:
                s.close()
            except Exception:
                pass

    console.print(f"  Total training pairs: {result.total_pairs}")
    for name, n_pairs in result.per_dataset_pairs.items():
        budget_for_name = result.per_dataset_budget.get(name, 0)
        console.print(f"    {name}: {n_pairs} pairs (budget: {budget_for_name})")

    if result.per_dataset_baseline and result.per_dataset_final:
        console.print()
        console.print("[bold]Eval results (per dataset, NDCG@10)[/bold]")
        for name in result.per_dataset_baseline:
            base = result.per_dataset_baseline[name]
            final = result.per_dataset_final.get(name)
            if final is None or base.n_queries == 0:
                continue
            delta = final.ndcg_at_10 - base.ndcg_at_10
            color = "green" if delta >= 0 else "red"
            console.print(
                f"  {name:<20} baseline={base.ndcg_at_10:.4f}  "
                f"final={final.ndcg_at_10:.4f}  "
                f"delta=[{color}]{delta:+.4f}[/{color}]  (n={base.n_queries})"
            )
        macro_color = "green" if result.macro_delta_ndcg >= 0 else "red"
        console.print(
            f"  [bold]macro-avg            baseline={result.macro_baseline_ndcg:.4f}  "
            f"final={result.macro_final_ndcg:.4f}  "
            f"delta=[{macro_color}]{result.macro_delta_ndcg:+.4f}[/{macro_color}][/bold]"
        )

        console.print()
        console.print("[bold]Head quality + candidate health (per dataset)[/bold]")
        head_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        head_table.add_column("dataset", style="magenta")
        head_table.add_column("NDCG@3 base", justify="right")
        head_table.add_column("NDCG@3 final", justify="right")
        head_table.add_column("delta", justify="right")
        head_table.add_column("Recall@100 base", justify="right")
        head_table.add_column("Recall@100 final", justify="right")
        head_table.add_column("delta", justify="right")
        for name in result.per_dataset_baseline:
            base = result.per_dataset_baseline[name]
            final = result.per_dataset_final.get(name)
            if final is None or base.n_queries == 0:
                continue
            d3 = final.ndcg_at_3 - base.ndcg_at_3
            dr = final.recall_at_100 - base.recall_at_100
            c3 = "green" if d3 >= 0 else "red"
            cr = "green" if dr >= 0 else "red"
            head_table.add_row(
                name,
                f"{base.ndcg_at_3:.4f}",
                f"{final.ndcg_at_3:.4f}",
                f"[{c3}]{d3:+.4f}[/{c3}]",
                f"{base.recall_at_100:.4f}",
                f"{final.recall_at_100:.4f}",
                f"[{cr}]{dr:+.4f}[/{cr}]",
            )
        console.print(head_table)

    if result.gated_out:
        console.print()
        if result.total_pairs < 10:
            console.print(
                "[red]Training skipped[/red]: not enough training pairs across all corpora. "
                "Try a larger --total-triples or add more documents."
            )
        else:
            gate_mode = "per-dataset" if per_dataset_gate else "macro-average"
            console.print(
                f"[red]Gated out[/red] ({gate_mode}): NDCG@10 delta did not meet "
                f"min-gain ({result.min_gain:+.4f}). Candidate left at "
                f"[dim]{Path(output).expanduser()}.candidate[/dim] for inspection."
            )
        raise typer.Exit(code=2)

    if result.output_path is None:
        console.print("[red]x[/red] retrain-multi did not save a model. See logs for details.")
        raise typer.Exit(code=2)

    console.print()
    console.print(f"[green]Model saved to:[/green] {result.output_path}")
    console.print()
    console.print("To use the retrained model:")
    console.print(f"  [bold]vstash reindex --model {result.output_path}[/bold]")


@profile_app.command(name="create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name"),
) -> None:
    """Create a new named profile."""
    from .profile import create_profile as _create

    try:
        db_path = _create(name)
        console.print(
            f"[green]✓[/green] Created profile [bold]{name}[/bold]\n"
            f"[dim]  Database: {db_path}[/dim]\n"
            f"[dim]  Use: vstash --profile {name} add ...[/dim]"
        )
    except ValueError as exc:
        console.print(f"[red]✗[/red] {_safe_exc(exc)}")
        raise typer.Exit(1) from exc


@profile_app.command(name="delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a named profile and all its data."""
    from .profile import delete_profile as _delete

    if not yes:
        typer.confirm(
            f"Delete profile '{name}' and all its data? This cannot be undone",
            abort=True,
        )

    try:
        _delete(name)
        console.print(f"[green]✓[/green] Deleted profile [bold]{name}[/bold]")
    except ValueError as exc:
        console.print(f"[red]✗[/red] {_safe_exc(exc)}")
        raise typer.Exit(1) from exc


@profile_app.command(name="active")
def profile_active(ctx: typer.Context) -> None:
    """Show which profile is currently active."""
    from .profile import active_profile_info

    explicit = _profile_from_ctx(ctx)
    name, reason = active_profile_info(explicit)

    if name:
        console.print(f"[bold cyan]{name}[/bold cyan] [dim]({reason})[/dim]")
    else:
        console.print(f"[dim]{reason}[/dim]")


# ------------------------------------------------------------------ #
# vstash journal                                                       #
# ------------------------------------------------------------------ #

snapvec_app = typer.Typer(
    name="snapvec",
    help="Manage the snapvec vector backend (IVFPQ index training).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(snapvec_app, name="snapvec")


@snapvec_app.command(name="fit")
def snapvec_fit(
    ctx: typer.Context,
    training_sample: int = typer.Option(
        50_000,
        "--training-sample",
        help="Vectors sampled for IVFPQ codebook training (FAISS rule: >=30 * nlist)",
    ),
) -> None:
    """Train and persist the IVFPQ index from the current corpus.

    Requires ``storage.vector_backend = 'snapvec-ivfpq'`` in vstash.toml.
    Reads every embedding out of vec_chunks, fits the IVF coarse centroids
    + residual PQ codebooks, indexes all rows, and saves the ``.snpi``
    file next to the database. After this completes, searches route
    through the IVFPQ backend with fp16 rerank.
    """
    _, store = _get_store(profile=_profile_from_ctx(ctx))
    try:
        stats = store.fit_ivfpq(training_sample=training_sample)
    except RuntimeError as exc:
        console.print(f"[red]x[/red] {_safe_exc(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]+[/green] IVFPQ index fit: "
        f"{stats['n_indexed']} vectors indexed, "
        f"nlist={stats['nlist']}, "
        f"training_sample={stats['training_sample']}, "
        f"build={stats['build_seconds']}s"
    )
    console.print(f"  saved to [dim]{stats['path']}[/dim]")


journal_app = typer.Typer(
    name="journal",
    help="Cross-session memory journal for agents and humans.",
    no_args_is_help=True,
)
app.add_typer(journal_app, name="journal")


@journal_app.command(name="save")
def journal_save_cmd(
    text: str | None = typer.Argument(None, help="Text to save (or pipe via stdin)"),
    title: str | None = typer.Option(None, "--title", "-t", help="Entry title (auto-generated)"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project tag"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Source (session, hook, manual)"
    ),
    from_transcript: str | None = typer.Option(
        None, "--from-transcript", help="Parse Claude Code transcript JSONL"
    ),
) -> None:
    """Save a journal entry. Accepts text as argument, stdin, or --from-transcript."""
    import sys

    from .journal import journal_save, parse_transcript

    # Resolve text from transcript, argument, or stdin
    if from_transcript:
        text = parse_transcript(from_transcript)
        if not text:
            console.print("[yellow]No content extracted from transcript.[/yellow]")
            raise typer.Exit(1)
        source = source or "transcript"
    elif text is None:
        if sys.stdin.isatty():
            console.print("[red]No text provided. Pass as argument or pipe via stdin.[/red]")
            raise typer.Exit(1)
        text = sys.stdin.read()

    result = journal_save(text, title=title, project=project, tags=tags, source=source)

    if result.get("status") == "empty":
        console.print("[yellow]No content to save.[/yellow]")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Saved: [bold]{result['title']}[/bold] "
        f"({result['chunks']} chunks, tags: {result['tags']})"
    )


@journal_app.command(name="recall")
def journal_recall_cmd(
    query: str | None = typer.Argument(None, help="Search query (omit for recent entries)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of entries"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw text (for hooks/pipes)"),
) -> None:
    """Recall relevant journal entries. Omit query for most recent."""
    from .journal import journal_recall

    entries = journal_recall(query=query, top_k=top_k, project=project)

    if not entries:
        if not raw:
            console.print("[dim]No journal entries found.[/dim]")
        raise typer.Exit()

    if raw:
        # Raw output for piping into hooks or other tools
        for entry in entries:
            print(f"## {entry.get('title', 'untitled')}")
            print(entry.get("text", ""))
            print()
        return

    for entry in entries:
        title = entry.get("title", "untitled")
        text = entry.get("text", "")
        score = entry.get("score")
        tags = entry.get("tags", "")

        header = f"[bold cyan]{title}[/bold cyan]"
        if score is not None:
            header += f" [dim](score: {score:.3f})[/dim]"
        if tags:
            header += f" [dim][{tags}][/dim]"

        console.print(Panel(text[:500] + ("..." if len(text) > 500 else ""), title=header))


@journal_app.command(name="log")
def journal_log_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
    recent: str | None = typer.Option(
        None, "--recent", "-r", help="Time window filter: 7d, 24h, 2w"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
) -> None:
    """Chronological view of journal entries (newest first)."""
    from .journal import journal_log

    entries = journal_log(limit=limit, recent=recent, project=project)

    if not entries:
        console.print("[dim]No journal entries yet.[/dim]")
        raise typer.Exit()

    table = Table(title="Journal Log")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Project", style="green")
    table.add_column("Tags", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Date", style="dim")

    for entry in entries:
        added = entry.get("added_at", "")
        if added and len(added) > 16:
            added = added[:16]
        table.add_row(
            entry.get("title", "?"),
            entry.get("project") or "-",
            entry.get("tags") or "-",
            str(entry.get("chunks", 0)),
            added,
        )

    console.print(table)


@journal_app.command(name="prune")
def journal_prune_cmd(
    age: str = typer.Argument(..., help="Max age: 30d, 2w, 24h"),
    project: str | None = typer.Option(None, "--project", "-p", help="Only prune this project"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Remove journal entries older than the specified age."""
    from .journal import journal_prune

    if dry_run:
        result = journal_prune(age, project=project, dry_run=True)
        count = result.get("would_delete", 0)
        console.print(f"[dim]Would delete {count} entries older than {age}[/dim]")
        for title in result.get("entries", []):
            console.print(f"  [dim]- {title}[/dim]")
        return

    if not yes:
        typer.confirm(f"Delete journal entries older than {age}?", abort=True)

    result = journal_prune(age, project=project)
    deleted = result.get("deleted", 0)
    console.print(f"[green]✓[/green] Pruned {deleted} journal entries.")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    """Main entry point for the vstash CLI."""
    app()


if __name__ == "__main__":
    main()
