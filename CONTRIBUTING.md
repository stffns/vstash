# Contributing to vstash

## First-clone setup

```bash
pip install -e ".[all,dev]"
pip install pre-commit
pre-commit install
```

The ``pre-commit install`` step wires a local git hook that runs the
same ``ruff check`` and ``ruff format --check`` commands CI runs.  This
is the fastest way to avoid the "pushed, CI failed, pushed again"
cycle -- hooks reject the commit before it lands.

Run all hooks once against the whole tree (useful after ``git pull``):

```bash
pre-commit run --all-files
```

## Testing

```bash
python -m pytest tests/ -x -q
```

Benchmark-marked BEIR regression tests are gated off by default;
run them explicitly with ``pytest -m benchmark``.  They need cached
BEIR data under ``experiments/data/`` -- run
``python -m experiments.beir_benchmark --datasets scifact`` once to
populate the cache.

## Branching

Feature branches target ``develop``.  ``main`` is updated only via
release PRs from ``develop`` and always matches the latest PyPI
published version.  See ``CLAUDE.md`` for the release flow.

## Commit style

Conventional-commit prefixes (``feat``, ``fix``, ``docs``, ``chore``,
``perf``, ``test``, ``style``).  Reference the relevant issue number
in parentheses after the prefix when applicable, for example
``fix(#289): ...``.
