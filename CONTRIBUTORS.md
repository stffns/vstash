# Contributors

## Human

**Jayson Steffens** ([@stffns](https://github.com/stffns)) is the
sole human contributor. The project lead, research direction,
benchmark design, release gatekeeping, scientific claims, and the
final word on every design decision belong to him. His professional
background is in QA engineering. All commits in this repository are
authored by him, and the legal responsibility for the code rests
with him.

## AI assistants used in development

vstash uses AI tooling in its development workflow. This section
documents what, how, and under what constraints, in the spirit of
the transparency standards documented in `docs/observability.md`
applied to the development process itself.

### Claude (Anthropic)

Used for: code generation, refactoring, test writing, documentation,
responding to automated code review, debugging, and conversational
design exploration. Long-form technical writing in this repository,
including sections of `paper/vstash-paper.md`, `CHANGELOG.md`
entries, and docs under `docs/`, was assisted by Claude.

Not used for: design decisions, algorithm selection, benchmark
methodology design, or release go/no-go judgment. Those are made by
the human contributor after reviewing proposals.

Constraint: per the project's internal no-AI-coauthor rule, Claude is
never listed as commit co-author. The commit history accurately
reflects the human contributor as author of record.

### Jules (Google)

Automated code agent used to surface performance optimization
opportunities on hot paths. Example contributions: PR #149 (cosine
similarity optimization, 5 to 11x speedup depending on Python
version), PR #167 (MMR `norm_score` hoisting, ~27% reduction in
inner loop time on the benchmark case). Jules-opened PRs are
reviewed, refined, and verified before merge. No automatic merging
of Jules output.

### Gemini Code Assist and GitHub Copilot

Automated code review on every PR. Their feedback is treated as
input, not authority. Reviewer claims are verified against the
actual codebase before being acted on. Notable instance: the review
of PR #149 incorrectly asserted that `math.hypot(*a)` was slower
than `sum(map(operator.mul, a, a))` due to argument unpacking
overhead. A direct benchmark disproved the claim, and the PR kept
the `math.hypot` approach (see PR #149 discussion).

## Why this file exists

Transparency about tool use should not be shameful, and it should
not be used as a marketing hook either. Readers can judge the work
on its merits by running the benchmarks in `experiments/`. This
file is here so anyone who asks "was AI used in building this?"
gets an honest, specific answer without having to infer it from
the commit history.
