"""The CLI must build on the installed typer, whatever its version.

``typer.Option(click_type=click.Choice(...))`` stopped being accepted in
typer 0.29: the decorator raised ``TypeError: Option() got an unexpected
keyword argument 'click_type'`` while ``vstash.cli`` was being imported, so
*every* ``vstash`` command died at startup, not only ``retrain``. Choices
are declared as Enums now; these tests pin that every command still builds
and that the choice options keep validating.
"""

from __future__ import annotations

import inspect

import pytest
import typer
from typer.testing import CliRunner

from vstash.cli import app

runner = CliRunner()


def _all_commands(t: typer.Typer, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every command path, read from typer's own registry (no click internals)."""
    paths = [
        (*prefix, c.name or c.callback.__name__.replace("_", "-"))  # type: ignore[union-attr]
        for c in t.registered_commands
    ]
    for g in t.registered_groups:
        assert g.name is not None and g.typer_instance is not None
        paths.append((*prefix, g.name))
        paths.extend(_all_commands(g.typer_instance, (*prefix, g.name)))
    return paths


def test_no_command_passes_click_type_unless_typer_supports_it() -> None:
    """Guard the root cause, independent of which typer the test env resolved."""
    import vstash.cli as cli_pkg

    supported = "click_type" in inspect.signature(typer.Option).parameters
    if supported:
        pytest.skip("this typer still accepts click_type")
    from pathlib import Path

    for path in Path(cli_pkg.__file__).parent.glob("*.py"):
        code = "\n".join(
            line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        assert "click_type=" not in code, f"{path.name} passes click_type to typer.Option"


@pytest.mark.parametrize("path", _all_commands(app))
def test_every_command_renders_help(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("command", "option", "choices"),
    [
        ("retrain", "--training-pair-source", {"auto", "labeled", "prefix"}),
        ("retrain-multi", "--training-pair-source", {"auto", "labeled", "prefix"}),
        ("retrain-multi", "--sampling-strategy", {"uniform", "proportional", "temperature"}),
    ],
)
def test_choice_options_still_validate(command: str, option: str, choices: set[str]) -> None:
    result = runner.invoke(app, [command, option, "no-such-choice"])
    assert result.exit_code == 2
    assert all(choice in result.output for choice in choices)
