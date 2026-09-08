# Changelog

All notable changes to engram are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-08

Iteration round 2 — feedback loop, write quality, ops, and engineering-debt payoff.

### Added
- **Adoption feedback loop**: `get_memory` records which search results were actually read (24h window), `query_stats` reports adoption counts/rates, and the new `mark_relevance` MCP tool lets the agent flag a result `useful` (+0.1 importance) or `irrelevant` (−0.1) — importance now evolves with real usage (migration v5 adds `query_log.adopted`).
- **`engram stats`**: one-glance observability — memory counts per type/project, entity graph size, reflection states, retrieval feedback (query volume, zero-hit and adoption rates, top adopted/unanswered queries).
- **`engram backup / export / import`**: SQLite hot backup (safe while the server runs, retention `--keep N`), and idempotent-by-id JSON export/import.
- **HTTP MCP transport** (`[http]` config, off by default): Streamable-HTTP subset on tiny_http (no async runtime) — `POST /mcp` with mandatory Bearer token; startup refuses an empty token.
- **`engram.distill` prompt**: refines raw `session-import` memories into clean typed memories (decisions / failures with root cause / procedures) with quality bars, then archives the originals.
- **Transcript error extraction**: `session-import`/`hook` now captures `is_error` tool results as failure evidence (`has-errors` tag) — raw material for failure memories and the reflection engine.
- **Cross-project global search**: `project_id = "*"` on `search_memory` spans all projects (global knowledge); semantic + graph paths honor it too.
- **`--features jieba`**: word-level Chinese tokenization (embedded jieba dictionary). A `meta` table records which tokenizer built the FTS index; switching builds triggers exactly one automatic rebuild.
- Transport test suite: the stdio transport (`run_transport`, now reader/writer-generic) is directly tested — parse vs invalid-request distinction, jsonrpc version check, notifications, panic isolation, the 16 MiB frame cap, multi-worker dispatch.
- Semantic stub tests: a deterministic embedder exercises fuse (vector-only materialization, cosine injection, filter enforcement) and the vector cache in CI — no model download needed.

### Changed
- **Contentless FTS5** (migration v6): shadow tables store the inverted index only — the document text is no longer duplicated, halving database size; search joins by rowid.
- **Cross-type BM25 calibration**: each type's scores are rescaled to `raw/type_best × sigmoid(type_best)`, so a weak match in one table no longer outranks a strong match in another.
- **decision/procedural importance** is a real field now (migration v4; tools accept `importance`, failure feedback maps to severity) — the reranker weight is no longer dead for those types.
- **Token budget covers the `results` array** too: over budget, detail fields are stripped from the lowest-ranked results first (`detail_omitted: true`, re-fetchable via `get_memory`); previously only the `context` field was bounded.
- Semantic: the embedding model loads lazily on first use (the ~90 MB first-run download no longer blocks the MCP initialize handshake), and vectors are cached per project with write-time invalidation (was: full reload + deserialize on every query).

### Engineering debt
- `repository.rs` split into `schema.rs` (DDL/migrations/tokenization), `graph.rs` (entities/relations/neighbors), `embeddings.rs` (vector store) — 4,600 lines down to 3,200 in the core file.
- Composer dead code removed (`ContentType`/`detect_content_type`).
- CI: new `test-features` job runs the jieba + semantic test suites.

## [0.4.0] - 2026-09-08
## [0.4.0] - 2026-09-08

Deep-review fix batch (2026-09-08 全链路审查, see `docs/deep-review-2026-09.md`) plus the first feedback-loop / write-automation / graph-retrieval iterations.

### Added
- **`engram hook`**: Claude Code hook entry point — reads the Stop/SessionEnd hook payload from stdin (`transcript_path`, `session_id`, `cwd`) and distills the transcript into a memory via session-import. Empty sessions report `skipped`, not an error. Makes memory formation zero-config (see README §Integration).
- **Session-import is idempotent per session**: re-importing the same `(project, session)` refreshes the existing episodic memory in place (upsert) instead of stacking near-duplicates — a `Stop` hook firing every turn now converges to exactly one memory that tracks the session as it grows. `SessionEnd` remains the recommended one-shot wiring.
- **Graph-based retrieval**: `search_memory` now appends up to 2 memories that share an entity (file/tool) with the top hits, marked `graph_neighbor: true` at fixed low rank — the relationship graph finally participates in search. Toggle with `[retrieval].graph_expansion` (default true).
- **Knowledge-gap report** in `engram maintain`: frequent queries that consistently return zero results (the cheapest signal for what memory to write next).
- **Orphan entity GC**: `engram gc --apply` now also deletes File/Tool entities no relation references (previously unbounded growth + `related_files` noise); `maintain` previews the count.
- **`[security] allowed_roots`** config: restrict `ingest_commits`/`collect_sources` `repo_path` to configured roots.
- MCP transport: JSON-RPC `jsonrpc` version validation, and `ping`/`tools/list`/`prompts/list`/`initialize` are answered inline so a long tool call cannot starve client health probes.

### Changed
- **Ranking math rebuilt**: weights normalized to sum 1.0 (relevance dominant at 0.5), BM25 sigmoid rescaled to the real FTS5 score range, type-prior spread narrowed, and intents now raise *relevance* instead of only the query-independent priors — previously the static `failure>decision>episodic>procedural` prior effectively decided cross-type ranking regardless of the query.
- **CJK search fixed for mixed-script text**: the preprocessor now splits CJK↔latin boundaries (`用Rust写` used to index as one unmatchable token) and preprocesses tags/files columns; schema migration v2 rebuilds existing FTS indexes with the new tokenization.
- **Multi-token queries try AND first**, falling back to OR on empty results (was always-OR: "rust memory system" matched anything containing "system").
- **FTS rowid alignment** (schema migration v2): FTS rows now share the main table's rowid, making update/delete FTS maintenance O(log n) instead of a full index scan per operation (verified via EXPLAIN QUERY PLAN).
- **git ingest is lazy**: the rev walk sorts by commit time and stops after N — the entire history (100k+ commits on monorepos) is no longer decoded to fetch the newest 20.
- **CLI `search` uses the full MCP pipeline** (intent → plan → rerank → optional semantic fuse); it was a bare-BM25 fork that ranked differently from the MCP tool.
- Milestone clustering splits themes on >30-day gaps: years-apart `fix` commits no longer merge into one meaningless "milestone".
- Tool failures now return MCP `result.isError = true` (business errors) or `-32602` (bad params / unknown tool) instead of blanket `-32603`; a broken config file is fatal for the MCP server too (it used to silently fall back to the default database — splitting data across two stores).
- Semantic (feature build): cosine scores are injected into relevance (they were computed then discarded), explicit `memory_type`/`tags`/`before` filters apply to vector-only hits, `update_memory` re-embeds changed text, embeddings truncate at the model's 512-token limit (long inputs used to OOM/near-hang), and model changes with zero vectors now point at `engram reindex`.
- Collector heuristics: TODO/NOTE no longer flood the annotation budget; py/sh/sql/lua `#`/`--` decision comments are recognized; CHANGELOG "Fixed" sections match exact words; repo-root `.circleci`/`.buildkite` configs are found; conventional-commit types are whitelist-checked; fix-detection scans the subject only; file collection walks are budgeted (entries/depth/time), deterministic (sorted), and read via streaming truncation (a multi-GB file can no longer OOM the collector).
- `repo_path` from MCP clients is validated (must exist, be a directory, not the home directory or `/`; optional allowlist above).

### Fixed
- `get_*`/`update_*` enforce `project_id` in SQL (isolation no longer depends on every caller remembering to check); update can no longer move a memory across projects.
- `limit` clamped to 1..=1000 and `days` to 0..=3650 on every tool (`limit=1000000` previously flowed into SQL; extreme `days` overflowed the timestamp arithmetic).
- Empty queries and typo'd `memory_type` values are loud errors instead of silent empty results; `prompts/get` validates required arguments.
- Startup no longer runs two full-table dedup writes on every command (one-time, index-existence guarded).
- Reflection re-arm: rejecting a proposal no longer silences the tag forever — it re-proposes after `min_occurrences` NEW failures (migration v3).
- near-dup consolidation pre-filters by length ratio before building word sets (the O(n²) hot path), and stale "time-window merging" docs corrected.

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
