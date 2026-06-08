"""``vstash profile`` sub-typer (#284): list, create, delete, active.

Defines ``profile_app`` and registers it on the shared ``app`` at import time;
``vstash/cli/__init__.py`` imports this module for that side effect.
"""

from __future__ import annotations

import typer
from rich.table import Table

from ._app import _profile_from_ctx, _safe_exc, app, console

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
