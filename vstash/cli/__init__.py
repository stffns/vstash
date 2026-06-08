"""vstash command line interface (package facade).

Split out of the former monolithic ``cli.py`` (#284). The shared Typer ``app``,
the root callback, and the helpers every command imports live in ``_app``; the
commands themselves live in focused group modules:

  _ingest   add, remember
  _search   ask, search, chat
  _inspect  list, export, stats, why
  _manage   forget, update, compact, check, config, reindex, watch, serve
  _retrain  retrain, retrain-multi
  _profile  profile (list/create/delete/active)   [sub-typer]
  _snapvec  snapvec (fit)                          [sub-typer]
  _journal  journal (save/recall/log/prune)        [sub-typer]

Importing each module runs its ``@app.command()`` / ``app.add_typer`` registration
on the shared ``app``. This module re-exports ``app`` (the ``vstash`` entry point
``vstash.cli:app``) and the ``main()`` console-script wrapper.
"""

from __future__ import annotations

from ._app import app

# Re-export the store-getter and the rich-markup escaper at the package level so
# the historical ``vstash.cli._get_store`` / ``vstash.cli._safe_exc`` references
# (tests and any external callers) keep resolving after the #284 split moved the
# definitions into ``_app``.
from ._app import _get_store, _safe_exc  # noqa: F401

# Importing the command-group modules runs their ``@app.command()`` /
# ``app.add_typer`` registration on the shared ``app`` (#284). Imported for that
# side effect only.
from . import (  # noqa: F401
    _ingest,
    _inspect,
    _journal,
    _manage,
    _profile,
    _retrain,
    _search,
    _snapvec,
)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    """Main entry point for the vstash CLI."""
    app()


if __name__ == "__main__":
    main()
