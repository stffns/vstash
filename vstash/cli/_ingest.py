"""``vstash`` ingest commands (#284): add, remember.

Imported by ``vstash/cli/__init__.py`` so the ``@app.command()`` decorators
register on the shared Typer ``app``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..ingest import ingest, ingest_directory
from ._app import _get_store, _profile_from_ctx, app, console


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


# ------------------------------------------------------------------ #
# vstash list                                                         #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# vstash forget                                                       #
# ------------------------------------------------------------------ #


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

    from ..ingest import ingest_text

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
