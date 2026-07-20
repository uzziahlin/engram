# Security Policy

## Supported Versions

engram is in early development (0.x). Security fixes are applied to the
latest `main` and the most recent release tag.

| Version          | Supported |
|------------------|-----------|
| latest `main`    | ✅        |
| latest release tag | ✅      |
| older tags       | ❌        |

## Reporting a Vulnerability

engram is a **local-first** tool: it runs on your machine, stores data in a
local SQLite database, and makes no outbound network calls in its default
configuration. (The opt-in `semantic` feature downloads an embedding model
from Hugging Face on first use only.)

Its primary threat model is:

- a malicious or compromised MCP client driving the server, and
- untrusted inputs (search queries, tags, file paths) flowing into SQLite/FTS5.

If you discover a security vulnerability:

1. **Do not open a public GitHub issue.**
2. Report it privately via [GitHub Private Vulnerability Reporting]
   (https://github.com/uzziahlin/engram/security/advisories/new), or email the
   maintainer listed on the GitHub profile.
3. Include: a description, steps to reproduce, the affected file/version, and
   an impact assessment.

## Response SLA

- **Acknowledgement**: within 72 hours.
- **Status update**: within 7 days.
- **Fix or mitigation**: targeted within 30 days for high-severity issues,
  best-effort otherwise.

We follow coordinated disclosure: a fix is released before any public
advisory, and credit is given to the reporter unless they prefer to remain
anonymous.

## Security Measures Already in Place

- All SQLite access uses **parameterized queries** (`params![]`); table names
  come from an enum whitelist, never from user input.
- FTS5 full-text queries are **sanitized** (each token is phrase-quoted with
  double-quote escaping) so reserved words like `UNIQUE` / `AND` cannot act as
  column filters or operators.
- File scanning (`collect_sources`) is **read-only** and **byte-bounded** to
  prevent memory exhaustion.
- `project_id` **scopes every query** for multi-project isolation.
- Deletion is **soft** (archive) by default; physical deletion requires an
  explicit `gc` step guarded by `archived_at IS NOT NULL`.
- The `semantic` embedding feature is **off by default**, keeping heavy ML
  dependencies out of the standard release binary.
