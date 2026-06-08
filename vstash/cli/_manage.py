"""``vstash`` lifecycle / maintenance commands (#284): forget, update, compact,
check, config, reindex, watch, serve.

Imported by ``vstash/cli/__init__.py`` so the ``@app.command()`` decorators
register on the shared Typer ``app``.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from .._store_open import open_store_for_config
from ..config import load_config
from ..embed import warmup
from ._app import (
    _get_store,
    _profile_from_ctx,
    app,
    console,
)


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

    from ..memory import Memory

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


@app.command()
def compact(
    ctx: typer.Context,
    before: str | None = typer.Option(
        None,
        "--before",
        help=(
            "Prune docs older than this age (e.g. '90d', '2w', '24h') or ISO date "
            "('2026-01-15'). Omit to skip the prune phase and just VACUUM + optimize."
        ),
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict prune to this collection"
    ),
    project: str | None = typer.Option(
        None, "--project", "-p", help="Restrict prune to this project"
    ),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict prune to this layer"),
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        "-t",
        help=(
            "Restrict prune to docs tagged with this value. Repeatable for OR "
            "(``--tag archive --tag old``), or comma-joined "
            "(``--tag 'archive,old'``). Comma-anchored match."
        ),
    ),
    no_vacuum: bool = typer.Option(
        False, "--no-vacuum", help="Skip the VACUUM step (faster on large DBs)"
    ),
    no_optimize_fts: bool = typer.Option(
        False, "--no-optimize-fts", help="Skip the FTS5 optimize step"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview which docs would be pruned without modifying the store. "
        "Also skips VACUUM and optimize.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output the report as JSON"),
) -> None:
    """Housekeeping pass: prune old docs (optional), VACUUM, optimize FTS5.

    Three phases, each independently togglable:

      * **Prune**: if ``--before`` is given, deletes every document
        whose ``added_at`` is strictly older than the threshold,
        scoped by ``--collection``/``--project``/``--layer``.
      * **VACUUM**: rebuilds the SQLite file to reclaim space after
        the prune. Can take seconds-to-minutes on large DBs; pass
        ``--no-vacuum`` to defer.
      * **Optimize FTS5**: merges FTS5 segments in place. Cheap;
        ``--no-optimize-fts`` skips it.

    With ``--dry-run``, only the prune preview runs; VACUUM and
    optimize are skipped regardless of their flags because they would
    modify the file.
    """
    from ..memory import Memory

    if before is None and (collection or project or layer or tag):
        # ``Memory.compact`` skips the prune phase entirely when
        # ``before`` is ``None``, so any scope filter the caller
        # passed is silently inert. Surface this loudly so users do
        # not mistakenly believe their ``--layer scratch`` actually
        # deleted anything just because the command exited 0.
        # Route through stderr when ``--json`` is set so the stdout
        # payload remains pure JSON for downstream parsers.
        warning = (
            "--collection/--project/--layer are ignored when --before is "
            "omitted -- the prune phase is skipped entirely and this run "
            "will only VACUUM / optimize. Pass --before to actually prune "
            "by scope."
        )
        if json_output:
            import sys as _sys

            print(f"Warning: {warning}", file=_sys.stderr)
        else:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

    with Memory(profile=_profile_from_ctx(ctx)) as mem:
        report = mem.compact(
            before=before,
            vacuum=not no_vacuum,
            optimize_fts=not no_optimize_fts,
            collection=collection,
            project=project,
            layer=layer,
            tags=tag,
            dry_run=dry_run,
        )

    if json_output:
        import json as _json

        print(_json.dumps(report, indent=2))
        return

    prune_report = report.get("prune")
    if prune_report is None:
        console.print("[dim]No --before set; skipping prune phase.[/dim]")
    else:
        verb = "Would delete" if prune_report["status"] == "dry_run" else "Deleted"
        console.print(f"[green]✓[/green] {verb} [bold]{prune_report['deleted']}[/bold] docs.")
        if prune_report["paths"] and prune_report["deleted"] <= 20:
            for p in prune_report["paths"]:
                console.print(f"  [dim]- {p}[/dim]")
        elif prune_report["paths"]:
            console.print(f"  [dim](first 20 of {prune_report['deleted']})[/dim]")
            for p in prune_report["paths"][:20]:
                console.print(f"  [dim]- {p}[/dim]")
    if report.get("vacuum"):
        console.print("[green]✓[/green] VACUUM complete.")
    if report.get("optimize_fts"):
        console.print("[green]✓[/green] FTS5 optimize complete.")


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
    from ..embed import resolve_backend

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
    from ..embed import embed_texts, get_embedding_dim

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
    from ..watch import start_watch

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

        from ..web import create_app
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

        from ..config import load_config
        from ..embed import warmup

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
