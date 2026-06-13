# Contributing to Engram

Thank you for your interest in contributing to Engram! This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust** 1.75+ (install via [rustup](https://rustup.rs/))
- **Git** 2.0+
- A C compiler (for `rusqlite` bundled build — GCC/Clang/MSVC)

### Build & Test

```bash
# Clone
git clone https://github.com/uzziahlin/engram.git
cd engram

# Build
cargo build

# Run all tests
cargo test

# Run with logging
RUST_LOG=debug cargo run

# Lint
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

### Project Structure

```
src/
├── main.rs              # Entry point (MCP server / CLI)
├── lib.rs               # Library root
├── config.rs            # TOML configuration
├── models/              # Memory type definitions
├── storage/             # SQLite + FTS5 persistence
├── retrieval/           # BM25 search, intent classification
├── graph/               # Relationship engine (petgraph)
├── git_integration/     # Git commit event processing
├── context/             # LLM context composition
├── consolidation/       # Memory consolidation (future)
├── reflection/          # Self-improvement (future)
├── mcp/                 # MCP server implementation
└── cli/                 # Command-line interface
```

## How to Contribute

### Bug Reports

Open a [GitHub Issue](https://github.com/uzziahlin/engram/issues) with:

1. **Steps to reproduce** — exact commands or tool calls
2. **Expected behavior** vs **actual behavior**
3. **Environment** — OS, Rust version, engram version
4. **Logs** — set `RUST_LOG=debug` and include relevant output

### Feature Requests

Open an issue with the `enhancement` label. Describe:

1. **Use case** — what problem does this solve?
2. **Proposed solution** — how should it work?
3. **Alternatives considered** — what else did you think about?

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/my-feature`
3. **Make your changes** with clear, focused commits
4. **Add tests** for new functionality
5. **Ensure all checks pass**: `cargo test && cargo clippy && cargo fmt --check`
6. **Open a PR** against `main`

### PR Guidelines

- **One PR, one concern** — don't mix refactors with features
- **Write clear commit messages** — describe *what* and *why*
- **Keep PRs small** — under 400 lines is ideal
- **Update docs** if you change public APIs or behavior

## Code Style

- Follow standard Rust conventions (`cargo fmt`)
- Resolve all `cargo clippy` warnings
- Add `///` doc comments to public items
- Prefer `anyhow` for application errors, `thiserror` for library errors
- Keep functions focused — if it does two things, split it

## Architecture Notes

Engram follows a layered architecture:

```
MCP Server → MemoryProvider trait → Repository + GraphEngine
                                      ↓
                                  SQLite (FTS5)
```

- **`MemoryToolProvider`** trait in `src/mcp/server.rs` is the integration point
- **`MemoryRepository`** handles all SQLite operations
- **`GraphEngine`** manages entity relationships via petgraph
- All memories are scoped by `project_id` for multi-project isolation

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
