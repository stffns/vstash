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
  vstash update <file>    → mutate metadata or replace content in place
  vstash compact          → prune old docs, VACUUM, optimize FTS5
  vstash reindex           → re-embed chunks with new model
  vstash config           → show current config
  vstash profile          → manage named profiles
"""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from ._app import (
    _get_store,
    _profile_from_ctx,
    _safe_exc,
    app,
    console,
)

# Importing the command-group modules runs their ``@app.command()`` decorators,
# which register the commands on the shared ``app`` (#284). Imported for that
# side effect only.
from . import _ingest, _inspect, _manage, _retrain, _search  # noqa: F401


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
    from ..profile import list_profiles as _list_profiles

    profiles = _list_profiles()
    if not profiles:
        console.print("[dim]No profiles yet. Create one with: vstash profile create <name>[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Profile", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    from ..profile import PROFILES_DIR

    for name in profiles:
        db_path = PROFILES_DIR / name / "memory.db"
        size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
        table.add_row(name, f"{size_mb:.1f} MB", str(db_path))

    console.print(table)


@profile_app.command(name="create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name"),
) -> None:
    """Create a new named profile."""
    from ..profile import create_profile as _create

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
    from ..profile import delete_profile as _delete

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
    from ..profile import active_profile_info

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

    from ..journal import journal_save, parse_transcript

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
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        "-t",
        help=(
            "Restrict to journal entries tagged with this value. Repeat for "
            "OR (``--tag work --tag decisions``), or pass a comma-separated "
            "string (``--tag 'work,decisions'``). Comma-anchored match."
        ),
    ),
    added_after: str | None = typer.Option(
        None,
        "--after",
        help="ISO date filter: only entries logged on or after this date.",
    ),
    added_before: str | None = typer.Option(
        None,
        "--before",
        help="ISO date filter: only entries logged strictly before this date.",
    ),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw text (for hooks/pipes)"),
) -> None:
    """Recall relevant journal entries. Omit query for most recent."""
    from ..journal import journal_recall

    entries = journal_recall(
        query=query,
        top_k=top_k,
        project=project,
        tags=tag,
        added_after=added_after,
        added_before=added_before,
    )

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
    from ..journal import journal_log

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
    from ..journal import journal_prune

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
