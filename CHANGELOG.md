# Changelog

All notable changes to engram are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-07

Read-path completion, lifecycle automation, and a deep-review fix batch.

### Added
- **`get_memory` MCP tool** (and `engram get` CLI): fetch one memory's complete record by id — structured fields (root_cause/fix/prevention, context/rationale/tradeoffs, steps) were previously write-only.
- **`search_memory` returns full payloads**: results now carry each memory's type-specific fields plus tags, so agents read knowledge in one call instead of a teaser summary. New `tags` (any-match) and `before` filters; CLI `search` gains `--tag`/`--before`.
- **`engram session-import`**: distills a Claude Code session transcript (JSONL) into one episodic memory — wire it to a Stop/SessionEnd hook for zero-effort memory formation (see `docs/claude-code-setup.md` §7).
- **`engram maintain`**: one-shot health pass — exact-duplicate consolidation (soft-archive), FTS index rebuild (repairs orphans/missing rows/re-applies CJK preprocessing), `query_log` retention prune (new `[storage].query_log_retention_days`, default 90), staleness report (`--repo`), and a GC preview. Never physically deletes.
- **`rebuild_fts`** repository method powering the repair above.
- MCP `ping` method (liveness probe required by newer clients).

### Changed
- **Intent routing is now soft**: every search covers all four memory types; classified intent only adjusts reranker weights. Hard source-narrowing cost recall (a Workflow query could no longer surface a decision record).
- **Unified ranking**: the context composer no longer re-sorts by fixed type priority — it preserves the reranker's order, so `results` and `context` agree within one response.
- **`ingest_commits` clusters into milestones**: one episodic memory per Conventional-Commit (type, scope) theme instead of one per commit (the noisy output git_collector's own docs advise against). Dedup now happens *before* memory generation; count clamped to 1000 (CLI parity).
- `recent_failures` / `architectural_decisions` return complete records (root_cause/fix/prevention, context/tradeoffs).
- Tool `limit` defaults now come from `[retrieval].default_limit` (was a hardcoded protocol default of 10).
- Release profile uses `panic = "unwind"` + `catch_unwind` isolation in the stdio transport, so one panicking request returns a JSON-RPC internal error instead of killing the MCP server. (0.2.0's changelog claimed this; the code now matches.)
- `serverInfo.version` from `CARGO_PKG_VERSION` (was hardcoded "0.1.0").
- collect: one shared git history walk for the git + failures dimensions (was walked twice); commit handles are reused instead of re-loaded for the newest N.
- Keyword heuristics (ingest tags/importance, fix-commit detection) match on word boundaries — "fixture" no longer counts as "fix", "docker" as "docs".
- Collector classification (workflow/docs kinds) matches on repo-relative paths — the checkout location no longer skews buckets.
- CLI `update --set` parses values as JSON when possible (numbers/bools/arrays work; everything used to fail type-check as strings) and warns on unknown keys.
- Dead config knobs removed (`storage.wal_mode`, `mcp.transport`, `[graph]`); `mcp.worker_threads` is now validated (1–64).
- GC batches deletes per memory kind in one transaction (was one commit per row) and `per_type` now reports actually-deleted counts.

### Fixed
- **FTS `tags` migration backfill skipped CJK preprocessing**, making migrated Chinese content permanently unsearchable; backfill now goes through the same preprocessing as the write path.
- **Updating a nonexistent id created orphan FTS rows** and silently returned success; updates now fail loudly and skip the index write.
- **Updates didn't refresh entity links** — stale `related_files` edges for removed files/tools; update now rebuilds the memory's graph edges.
- **`confirm_suggestion` was not atomic** (three pooled connections, two transactions): a crash between them permanently wedged the suggestion at `pending` with a PK conflict on retry. Now one transaction, idempotent under retries; empty-step suggestions are refused.
- **Reflection suggestions resurrected**: `reject` no longer lets the same proposal come back on the next `reflect --apply`, and confirmed rules are not re-proposed (block on any prior suggestion for the tag).
- **`get_ingested_commits` counted archived memories**, permanently blocking re-import of their commits after archiving.
- Legacy DBs with duplicate graph relations no longer fail startup at `CREATE UNIQUE INDEX` (pre-index dedup, mirroring the reflection-suggestions pattern).
- CLI `search` (all types) no longer swallows database errors as empty results, and merges results across types by score instead of fixed type order.
- CLI `timeline`/`queries` use saturating day arithmetic (debug builds panicked on huge `--days`); `optional_num` rejects NaN/inf.
- `engram init` warns on a broken config instead of silently using defaults.
- CJK character detection now covers Hangul.

## [0.2.0] - 2026-07-20

Open-source readiness pass — see `docs/开源就绪修复计划.md` for the full plan.

### Added
- `SECURITY.md` — vulnerability disclosure policy, response SLA, and a documented threat model for a local-first MCP server.
- Supply-chain governance: `deny.toml` (cargo-deny) gating advisories/licenses/bans, plus a CI `audit` job.
- Dependabot configuration for Cargo dependencies and GitHub Actions.
- Declared Minimum Supported Rust Version (`rust-version = "1.75"` in `Cargo.toml`) verified by a dedicated CI `msrv` job.
- `rust-toolchain.toml` pinning the development toolchain.
- `CHANGELOG.md` and `CODE_OF_CONDUCT.md`.

### Changed
- Documentation aligned with reality: MCP tool count is 22 (was 18); the relationship graph is documented as SQLite-backed (indexed adjacency via `related_files`), not an in-memory petgraph engine.
- CI build matrix aligned with the release matrix (added `macos-13` / `x86_64-apple-darwin`).
- The Release workflow now runs `cargo test` before building any artifacts.

### Fixed
- **GC no longer deletes live memories**: physical deletion now requires `archived_at IS NOT NULL`, closing a race where a memory restored between GC candidate collection and deletion was silently destroyed.
- **Reflection uniqueness**: `reflection_suggestions` enforces uniqueness on `(project_id, pattern_tag)` for pending rows, preventing duplicate suggestions under concurrent reflection.
- **Removed dead code**: the in-memory `GraphEngine` (petgraph) module and its dependency — relationship data lives in SQLite tables.
- **Hardened test coverage**: `sanitize_fts_query` now has direct injection-vector unit tests; the misleadingly-named rollback test was fixed or renamed.
- DoS hardening: bounded request body size, clamped tool-input lengths/counts, and `catch_unwind` isolation so a future panic cannot abort the server.

## [0.1.0] - 2026-06

Initial public preview.

### Added
- SQLite + FTS5 storage with four memory types: Episodic, Decision, Failure, Procedural.
- MCP server over stdio transport.
- BM25 retrieval with intent classification, intent-based source routing, and reranking (recency + importance + type-prior, configurable).
- Relationship graph (entities + relations) powering `related_files`.
- Git integration (auto-ingest commits via pure-Rust `gix`).
- Project bootstrap (`collect_sources` + `engram.bootstrap` prompt).
- CLI interface.
- Memory lifecycle: `forget` / `restore` / `update` / `forget_batch` / `consolidate` (soft-delete + dedup).
- Reflection engine (failure → procedural suggestions, gated behind pending review).
- Embedding-based semantic search (candle + Reciprocal Rank Fusion, behind the opt-in `semantic` feature).
- Retrieval feedback loop (`query_log`, `query_stats`).
