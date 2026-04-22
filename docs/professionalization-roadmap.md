# vstash repo professionalization roadmap

**Audience:** single-maintainer repo with 1000+ tests, a public PyPI package, an arXiv paper, and two HF-hosted model artefacts. The goal of this roadmap is to bring repo hygiene in line with that audience without adding process that only a team can pay for.

**Status as of 2026-04-22.** Checkboxes below are the current state.

- [x] Conventional commits enforced by convention (not yet by hook)
- [x] `ruff check` + `ruff format` as the sole style gate
- [x] 1008 tests, Python 3.10/3.11/3.12 matrix in CI
- [x] BEIR regression tests marker-gated off by default
- [x] PyPI release via manual `twine upload`
- [x] Auto-generated CHANGELOG (maintained by hand)
- [ ] CI runs on all PRs (fixed in PR #270)
- [ ] Pre-commit hooks
- [ ] Coverage reporting
- [ ] Dependabot / CodeQL
- [ ] PyPI trusted publisher
- [ ] Release automation
- [ ] Issue / PR templates
- [ ] CONTRIBUTING / SECURITY / CODE_OF_CONDUCT

## Guiding principles

- **Every merge proves its own correctness.** CI runs on every PR (including develop), not only on release. No PR merges red, even if the failure looks unrelated.
- **Automation replaces vigilance, never judgement.** Tools catch drift (format, lint, dep updates, coverage regressions). People decide design.
- **Don't professionalize for professionalism's sake.** Add a process only when it prevents a class of failure that has actually hit the repo. This list is already filtered to that criterion.

## P0 -- CI health (days, not weeks)

1. **CI runs on develop + feature PRs** -- PR #270 (in flight).
2. **Pre-commit config** -- add `.pre-commit-config.yaml` with ruff + mypy. Ship `pre-commit install` in CONTRIBUTING so devs feel the drift before CI does. Prevents the v0.30 to v0.32 drift cycle from happening again.
3. **Coverage reporting** -- `pytest --cov=vstash --cov-report=xml` + Codecov badge. Does not need to gate merges; reporting alone catches coverage erosion in new PRs. `test` job already pulls pytest-cov via `[dev]`.
4. **CI status badge in README** -- replace the hand-curated "tests: 900+ passing" badge with an actual workflow-status badge. Trust-signal upgrade for external readers of the paper / PyPI page.

## P1 -- Automation and supply chain (weeks)

5. **Dependabot** -- weekly PRs for `pyproject.toml` + `.github/workflows/*.yml`. Scoped conservatively (e.g. no major Python-version bumps) to avoid drowning a solo maintainer in noise.
6. **CodeQL** -- default config (Python, security-extended). Runs on PR + weekly schedule. Because `mcp.py` and `web.py` are network-exposed, this is cheap defence against a class of CVE that a human reviewer will miss.
7. **PyPI trusted publisher** -- replace the `twine upload` flow with a GitHub Actions OIDC publisher. Kills the long-lived PyPI token; OIDC proof is in the workflow run. Required by PyPI for Trusted Publishing compliance after 2026.
8. **release-please or Changesets** -- auto-generate the CHANGELOG stanza and the version bump PR from conventional-commit titles on `develop`. Removes the "did we update `CHANGELOG.md`?" question from every PR. Integrates cleanly with the existing `develop -> release PR -> main` flow.
9. **SBOM + sigstore** -- emit `vstash-<v>-cyclonedx.json` next to the wheel and sign the release tag with sigstore. Two-liners in the publish workflow once trusted publisher is in place.

## P2 -- Contributor experience and discoverability (months or whenever)

10. **CONTRIBUTING.md** -- one page. How to set up the venv, run tests, the `feature/* -> develop -> main` rule, commit message format. This codifies what is currently tribal.
11. **SECURITY.md** -- where to report vulnerabilities. GitHub Security Advisories flow, no private email address needed.
12. **CODE_OF_CONDUCT.md** -- Contributor Covenant 2.1, unchanged boilerplate. Lets the GitHub Community Standards checker go green.
13. **Issue + PR templates** -- three issue templates (bug, feature, perf regression) and one PR template that prompts for "test plan / reproduction". The perf-regression template directly mirrors the #252/#265 pattern; surfacing it as a template should speed future audits.
14. **Docs site via mkdocs-material** -- the `docs/` directory already has 17 good markdown files and zero navigation. mkdocs-material with the `gh-pages` deploy workflow gets `docs.vstash.dev` (or a readthedocs subdomain) for free. Signal to paper readers that "local memory" is a product, not a gist.
15. **Docker image + reproducible BEIR bench** -- nightly workflow that runs `experiments/beir_benchmark.py --no-chroma` on a fresh runner. Catches retrieval regressions (scores move unexpectedly after a retrieval-pipeline edit) as soon as they hit main. Publishes the JSON artefact so the results in the paper remain verifiable over time.

## P3 -- Stretch (nice-to-have, not blocking)

16. **Typed strictness** -- the `dev` extras already pull `mypy`. No workflow step runs it. Start with `mypy vstash/` (public API only) in strict mode, ignore the rest. Most signal for the least ceremony.
17. **Structured changelog entries per PR** -- a `changelog.d/` fragment system (`towncrier`-style). Stronger than release-please alone but heavier to adopt; only worth it if we start cutting more than one release per week.
18. **Multi-OS CI matrix** -- macOS + Windows in addition to ubuntu. Not urgent: the Apple Silicon MLX path is tested locally by the maintainer, and Windows users are a minority for this kind of tool. Revisit after the first real user report from Windows.

## Decision log

- **2026-04-22** -- Roadmap drafted alongside PR #270 (CI cleanup). Trigger: discovery that CI had been silently failing on release PRs for three consecutive versions (v0.30, v0.31, v0.32) because CI only ran on main.
