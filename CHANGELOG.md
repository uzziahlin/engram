# Changelog

All notable changes to engram are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
