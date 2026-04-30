<!--
Thanks for sending a PR. Please make sure:
- The PR targets `develop` (not `main`). Main is updated only via release PRs.
- The branch is short-lived and scoped to one feature or fix.
- Conventional-commit prefix in the title (`feat`, `fix`, `chore`, `perf`, `docs`, `test`, `style`).
- For security issues, follow SECURITY.md instead of opening a public PR.
-->

## Summary

<!-- 1-3 bullets. Why this change exists. -->

## Related issues

<!-- "Closes #123" / "Refs #45". Skip if none. -->

## Changes

<!-- Bulleted list of the user-visible or developer-visible changes. -->

## Test plan

<!-- Checklist of tests run / manual verification steps. -->
- [ ] `python -m pytest tests/ -x -q`
- [ ] `ruff check . && ruff format --check .`
- [ ]

## Risk / blast radius

<!-- One sentence: how isolated is this change? Anything that could regress
existing behaviour? Any data migration implications? -->
