"""Shared helpers for CLI tests after the #284 cli-package split.

The split moved commands into ``vstash/cli/_*.py`` submodules, each of which does
``from ._app import _get_store`` (etc.), creating a *per-module* binding. A test
that patches only ``vstash.cli`` misses the binding the invoked command actually
reads, so :func:`patch_cli_attr` patches the name wherever it lives.
"""

from __future__ import annotations

_CLI_SUBMODULES = ("_app", "_ingest", "_inspect", "_manage", "_search", "_retrain")


def patch_cli_attr(monkeypatch, name: str, value) -> None:
    """Patch ``name`` on every ``vstash.cli`` module (package + submodules) that binds it."""
    import vstash.cli as cli_mod

    targets = [cli_mod] + [
        getattr(cli_mod, m) for m in _CLI_SUBMODULES if hasattr(cli_mod, m)
    ]
    hit = False
    for mod in targets:
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, value)
            hit = True
    if not hit:
        raise AssertionError(f"no vstash.cli module binds {name!r}")
