<div align="center">

# 🧠 Engram

**Long-term memory for AI coding agents**

A local-first memory system that gives your AI coding assistant persistent engineering knowledge — across sessions, across projects, across time.

[![CI](https://github.com/uzziahlin/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/uzziahlin/engram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

[Installation](#-installation) · [Integration](#-integration) · [Tools Reference](#-tools-reference) · [Configuration](#-configuration) · [Architecture](#-architecture)

</div>

---

## Why Engram?

AI coding assistants are powerful — but they forget everything between sessions. Engram fixes this by providing **structured, persistent memory** that any MCP-compatible agent can read and write.

- **🔍 Retrieval-first design** — memories are only useful if they're found. BM25 + intent classification + reranking ensures relevant context surfaces automatically.
- **🏠 100% local** — all data in SQLite on your machine. No cloud, no API keys, no telemetry.
- **📦 Zero config** — works out of the box. Just point your agent at the binary.
- **🔗 MCP standard** — works with any MCP-compatible tool: Claude Code, Cursor, Codex CLI, Windsurf, and more.
- **🏗️ Structured memory types** — not a bag of text. Episodic, Decision, Failure, and Procedural memories each capture different engineering knowledge.

## Memory Types

| Type | What it stores | Example |
|------|---------------|---------|
| **Episodic** | Task history, sessions | "Refactored auth module, touched login.rs and session.rs" |
| **Decision** | Architecture choices + rationale | "Chose SQLite over PostgreSQL for local-first design" |
| **Failure** | Bugs, incidents + root cause | "FTS5 crash on SQL keywords — fixed with phrase query escaping" |
| **Procedural** | Workflows, conventions | "Deploy steps: test → build → tag → push → CI" |

---

## 🚀 Installation

### Quick Install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/uzziahlin/engram/main/install.sh | sh
```

This downloads the prebuilt binary for your platform, installs it to `~/.engram/bin/engram`, initializes the database, and **auto-configures any MCP client it detects** (Claude Code, Cursor, Windsurf, Codex CLI). Just restart your editor afterward.

**Supported platforms** (auto-detected by the installer):

| OS | Architecture | Target |
|----|--------------|--------|
| macOS | Apple Silicon | `aarch64-apple-darwin` |
| macOS | Intel | `x86_64-apple-darwin` |
| Linux | x86_64 | `x86_64-unknown-linux-gnu` |
| Windows | x86_64 (WSL / Git Bash) | `x86_64-pc-windows-msvc` |

The release binary is self-contained — SQLite is bundled and the git library is pure-Rust, so there are **no system OpenSSL/libgit2 dependencies** to install.

> Environment overrides: `ENGRAM_VERSION=x.y.z` to pin a version, `ENGRAM_INSTALL_DIR=<dir>` for a custom install location.

### Build from Source (developers)

```bash
git clone https://github.com/uzziahlin/engram.git
cd engram
cargo build --release

# The binary is at target/release/engram (engram.exe on Windows)
# Optionally install system-wide:
cargo install --path .
```

Requires Rust 1.75+ and a C compiler (only for the bundled SQLite via `rusqlite`; the git layer is pure-Rust `gix`). No external databases or services needed.

### Verify

```bash
engram --help
# or run as MCP server (stdio):
engram
```

---

## 🔌 Integration

Engram speaks the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) over stdio. Any MCP-compatible tool can connect.

> 💡 The **Quick Install** script above already writes the correct MCP config for any client it detects — the manual JSON snippets below are only needed for custom setups or if a client wasn't installed at install time.

### Claude Code

Add to your project's `.mcp.json` or your global `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": []
    }
  }
}
```

Then add to your project's `CLAUDE.md` so Claude knows when to use memory:

```markdown
# Memory Guidelines
When to write:
- After completing significant features → create_episodic
- After making architecture decisions → create_decision
- After fixing bugs → create_failure (with root cause analysis)
- After establishing workflows → create_procedural

When to read:
- Before starting new tasks → search_memory
- Before modifying files → related_files
- When needing project context → architectural_decisions
```

### Cursor

Add to your Cursor MCP settings (`Settings → MCP`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": []
    }
  }
}
```

### Codex CLI (OpenAI)

Add to `~/.codex/mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": []
    }
  }
}
```

### Windsurf

Add to `.windsurf/mcp.json` in your project:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": []
    }
  }
}
```

### Any MCP Client

Engram uses stdio transport with no required arguments. Point any MCP client at the `engram` binary:

```json
{
  "command": "engram",
  "args": []
}
```

Optional environment variables:
- `RUST_LOG=debug` — enable verbose logging (to stderr, not stdout)

---

## 🛠️ Tools Reference

All tools require a `project_id` parameter for multi-project isolation.

### Read Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_memory` | Full-text search across all memory types | `query`, `limit`, `memory_type` |
| `related_files` | Find entities related to a file via relationship graph | `file`, `project_id` |
| `timeline` | Get a timeline of memory events for the past N days | `days`, `project_id` |
| `recent_failures` | List recent failure/incident memories | `limit`, `project_id` |
| `architectural_decisions` | List architecture decision records | `limit`, `project_id` |

### Write Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `create_episodic` | Record a task, debug session, or feature implementation | `summary`, `content`, `files_touched`, `importance`, `tags` |
| `create_decision` | Record an architectural or design decision | `title`, `context`, `rationale`, `tradeoffs`, `related_files` |
| `create_failure` | Record an incident with root cause analysis | `incident`, `root_cause`, `fix`, `prevention`, `severity` |
| `create_procedural` | Record a workflow or convention | `workflow_name`, `steps`, `related_tools` |
| `ingest_commits` | Auto-generate episodic memories from git history | `repo_path`, `count`, `project_id` |

---

## ⚙️ Configuration

Engram works with zero configuration. Optionally create `~/.engram/config.toml`:

```toml
[storage]
database_path = "~/.engram/memory.db"  # SQLite database location
wal_mode = true                          # Write-Ahead Logging for performance

[retrieval]
default_limit = 10                       # Default search result count
fallback_timeout_ms = 50                 # Timeout per memory source

[context]
context_window_tokens = 200000           # LLM context window size
memory_budget_percent = 15               # % of context for memories

[graph]
max_nodes = 10000                        # Max graph nodes per project
```

Data is stored in `~/.engram/memory.db` by default.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client (stdio)                      │
│              Claude Code / Cursor / Codex / ...              │
└──────────────────────────┬──────────────────────────────────┘
                           │ JSON-RPC
┌──────────────────────────▼──────────────────────────────────┐
│                      MCP Server (10 tools)                   │
├─────────────────────────────────────────────────────────────┤
│                   MemoryToolProvider trait                    │
├─────────────┬─────────────────┬─────────────────────────────┤
│  Retrieval  │   Repository    │        GraphEngine           │
│  Pipeline   │   (SQLite+FTS5) │       (Petgraph)            │
├─────────────┤                 ├─────────────────────────────┤
│ Intent →    │  Episodic       │  Entities: File, Fn,        │
│ Route →     │  Decision       │  Module, Service, Commit    │
│ BM25 →      │  Failure        │                             │
│ Rerank →    │  Procedural     │  Relations: DependsOn,      │
│ Assemble    │                 │  Calls, FixedBy, Refs       │
└─────────────┴─────────────────┴─────────────────────────────┘
```

**Key design decisions:**

- **No async runtime** — SQLite is local and synchronous; tokio would add complexity for no benefit
- **Dual-write pipeline** — main tables + FTS5 virtual tables in the same SQLite transaction
- **Project isolation** — all memories scoped by `project_id`, supporting multi-project workflows
- **BM25 via FTS5** — built into SQLite, no external search engine needed

---

## 💻 CLI Usage

Engram also works as a command-line tool:

```bash
# Search memories
engram search "authentication refactor"

# Create memories
engram create-episodic --summary "Fixed memory leak" --importance 0.7
engram create-decision --title "Use SQLite" --rationale "Local-first, zero config"
engram create-failure --incident "FTS5 crash" --severity 4
engram create-procedural --name "deploy" --steps "test,build,push"

# Ingest git history
engram ingest --project myproj --repo .

# View history
engram timeline --days 7
engram recent-failures
engram decisions
```

---

## 🗺️ Roadmap

- [x] SQLite + FTS5 storage with 4 memory types
- [x] MCP server (stdio transport)
- [x] BM25 retrieval with intent classification
- [x] Relationship graph engine
- [x] Git integration (auto-ingest commits)
- [x] CLI interface
- [ ] Memory consolidation (merge/deduplicate over time)
- [ ] Reflection engine (self-improving retrieval)
- [ ] HTTP MCP transport (for remote access)
- [ ] Embedding-based semantic search
- [ ] Multi-agent memory sharing

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

[MIT](LICENSE) © 2025 Uzziah
