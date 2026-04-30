# Security policy

## Supported versions

vstash is pre-1.0 (Beta). Only the latest minor release on PyPI receives
security fixes. Older versions (0.X-1 and below) are not patched; upgrade
to the current release before reporting.

| Version | Supported          |
| ------- | ------------------ |
| 0.36.x  | :white_check_mark: |
| < 0.36  | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Use one of these private channels:

1. GitHub's "Report a vulnerability" form on the
   [Security tab](https://github.com/stffns/vstash/security/advisories/new)
   (preferred -- the report is encrypted in transit and tracked by GitHub).
2. Email the maintainer at `stffens@gmail.com` with subject
   `[vstash security] <short description>`.

Please include:

- A description of the issue and the threat model you are reasoning under.
- A reproduction (commands, config, sample inputs) if you have one.
- The vstash version (`vstash --version`) and Python version.
- Whether you intend to publicly disclose, and on what timeline.

## What to expect

- Acknowledgement within 5 business days.
- An initial assessment (severity + tentative fix window) within 14 days.
- Coordinated disclosure: if a fix is needed, we agree on a public
  advisory date together. Default is 90 days from initial report or
  immediately on patch release, whichever is sooner.
- Credit in the published advisory unless you ask to remain anonymous.

## Scope

In scope:

- The `vstash` Python package on PyPI.
- The `vstash` CLI, MCP server (`vstash-mcp`), and HTTP API (`vstash serve`).
- The default SQLite + sqlite-vec + FTS5 pipeline and the optional
  snapvec / snapvec-ivfpq backends.
- Embedded or referenced threat models: ingestion of arbitrary local
  files, fetching from arbitrary URLs (with the SSRF protections in
  `vstash/ingest.py`), web upload paths, and external LLM API calls.

Out of scope:

- Vulnerabilities in third-party dependencies that have their own
  upstream security process (`fastembed`, `sqlite-vec`, `mcp`,
  `cerebras-cloud-sdk`, etc.). Report those upstream and let us know
  so we can pin a patched version.
- Issues that require an attacker to already have local execution on
  the user's machine (vstash is local-first; the security model assumes
  the user trusts their own filesystem).
- Performance-degradation reports without a clear DoS path.
