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

Build locally and run the **same flow as the install script** — build → install to `~/.engram/bin` → init the database → configure detected MCP clients — via the bundled `Makefile`:

```bash
git clone https://github.com/uzziahlin/engram.git
cd engram
make            # build → install → init → configure (equivalent to install.sh)
```

Useful targets (run `make help` for the full list):

| Command | What it does |
|---|---|
| `make` / `make all` | Build, install to `~/.engram/bin`, init the DB, configure MCP clients |
| `make build` | Just compile the release binary |
| `make install` | Build + copy the binary into `INSTALL_DIR` |
| `make configure CLIENTS="claude cursor"` | (Re)configure only the listed MCP clients |
| `make uninstall` | Remove the binary and engram entries from MCP configs (`PURGE=1` also deletes `~/.engram`) |
| `make clean` | Remove cargo build artifacts |

Overridable variables: `INSTALL_DIR` (default `~/.engram/bin`), `CLIENTS` (`auto`, or a space-separated subset of `claude cursor windsurf codex`), `DATA_DIR`, `PURGE`. The Makefile shares its install/configure logic with `install.sh` through `scripts/engram-common.sh`, so both paths behave identically.

Prefer raw cargo? It still works:

```bash
cargo build --release    # binary lands in cargo's target dir (honors CARGO_TARGET_DIR)
cargo install --path .   # or install system-wide into ~/.cargo/bin

cargo build --release --features semantic   # opt-in: local embedding semantic search (larger binary; see [semantic] config)
```

Requires Rust 1.75+ and a C compiler (only for the bundled SQLite via `rusqlite`; the git layer is pure-Rust `gix`). No external databases or services needed.

### Verify

```bash
# smoke-test the binary — prints usage, no DB or --project needed
engram --help
# run as the MCP server (stdio) — this is what your editor launches with no args:
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

Instead of hand-writing the memory guidelines above, generate them per project:

```bash
engram init-guide            # writes ENGRAM.md (project_id defaults to the dir name)
                             # then asks whether to add `@ENGRAM.md` to CLAUDE.md
```

`@ENGRAM.md` uses Claude Code's import syntax, so the guide loads with your `CLAUDE.md`.

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
| `collect_sources` | Gather structured evidence from a project for bootstrap (no writes) | `repo_path`, `dimensions`, `max_commits`, `project_id` |

### Lifecycle Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `forget_memory` | Soft-delete (archive) a memory; reversible | `project_id`, `memory_type`, `id` |
| `restore_memory` | Un-archive a memory | `project_id`, `memory_type`, `id` |
| `update_memory` | Patch fields of an existing memory | `project_id`, `memory_type`, `id`, …fields |
| `forget_batch` | Archive by tags/before date (dry-run by default) | `project_id`, `memory_type`, `tags`, `before`, `apply` |
| `list_archived` | List archived (soft-deleted) memories | `project_id`, `memory_type`, `limit` |
| `consolidate_memories` | Detect and archive near-duplicate memories (dry-run by default) | `project_id`, `include_near_dup`, `apply` |

### Prompts

Engram also exposes MCP **prompts** (server-side templates any MCP client can fetch):

| Prompt | Description | Arguments |
|-------|-------------|-----------|
| `engram.bootstrap` | Seed memory for an existing project: gather evidence via `collect_sources`, then distill it into structured memories with quality bars | `project_id`, `repo_path`, `dimensions` |

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
recency_half_life_days = 30              # Recency decay half-life (days) for reranking
[context]
context_window_tokens = 200000           # LLM context window size
memory_budget_percent = 15               # % of context for memories

[graph]
max_nodes = 10000                        # Max graph nodes per project

[mcp]
worker_threads = 1                       # Concurrent request handlers (default 1 = FIFO sequential; raise only if your client pipelines independent requests)

[semantic]                               # Semantic search — only active in builds compiled with --features semantic
enabled = false                          # Turn on embedding-based retrieval
model_id = "sentence-transformers/all-MiniLM-L6-v2"   # fetched once into ~/.engram/models on first run
# model_path = "/path/to/model-dir"      # Air-gapped override: dir with config.json/tokenizer.json/model.safetensors
rrf_k = 60                               # Reciprocal Rank Fusion constant (fuses BM25 + vector ranks)
top_k = 50                               # Vector candidates fused with BM25
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
│                      MCP Server (17 tools)                   │
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
- **CJK-aware search** — Chinese/Japanese/Korean queries get character-level segmentation (spaces inserted between CJK runes), and FTS5 reserved words in a query are escaped so terms like `UNIQUE` or `AND` match as text rather than column filters. The intent classifier recognizes Chinese keywords too (e.g. 调试/修复, 架构/决策).
- **Ranking signals** — `search_memory` reranks BM25 results by recency (exponential half-life decay against the real clock), per-record `importance`, and a memory-type prior (`type_weight`). The relationship graph powers `related_files` only; it does **not** participate in search ranking.
- **Connection pool + concurrency** — the repository uses an r2d2 pool over WAL-mode SQLite (multi-reader, single-writer via `busy_timeout`); concurrent reads no longer serialize. MCP request handling dispatches to a bounded worker pool (`[mcp] worker_threads`, default 1 = FIFO sequential — safe for stdio clients that pipeline dependent requests; raise only for independent workloads).
- **Semantic search (optional)** — build with `--features semantic` to add local embedding retrieval via [candle](https://github.com/huggingface/candle) (pure Rust, no native runtime). Query and memories are embedded with a BERT model (default all-MiniLM-L6-v2), and vector top-K is fused with BM25 via Reciprocal Rank Fusion. Vectors are stored as BLOBs in SQLite and scored by brute-force cosine over the active set (no separate vector index), so semantic search suits thousands — not millions — of memories per project. The model is fetched once into `~/.engram/models/` (or supply `[semantic] model_path` for air-gapped use); inference is fully offline thereafter. **Off by default** — the standard build pulls in none of it and stays self-contained. Archived memories are excluded automatically; changing `model_id` makes existing vectors inert until memories are re-indexed.

---

## 💻 CLI Usage

Engram also works as a command-line tool:

```bash
# Search memories (query uses --query, not a positional arg)
engram search --project myproj --query "authentication refactor"

# Create memories
engram create-episodic --project myproj --summary "Fixed memory leak" --importance 0.7
engram create-decision --project myproj --title "Use SQLite" --context "local-first" --rationale "zero config"
engram create-failure --project myproj --incident "FTS5 crash" --root-cause "..." --fix "..." --prevention "..." --severity 4
engram create-procedural --project myproj --name "deploy" --steps "test,build,push"

# Ingest git history
engram ingest --project myproj --repo .

# View history
engram timeline --project myproj --days 7
engram recent-failures --project myproj
engram decisions --project myproj

# Generate an agent guide for this project (ENGRAM.md) and optionally import it
engram init-guide --project myproj

# Forget / restore / update individual memories
engram forget --project myproj --type episodic --id <id>
engram restore --project myproj --type episodic --id <id>
engram update --project myproj --type failure --id <id> --set severity=5

# Batch forget by tag (dry-run by default, add --apply to commit)
engram forget-batch --project myproj --tag bootstrap          # preview
engram forget-batch --project myproj --tag bootstrap --apply  # commit

# List archived memories and deduplicate
engram list-archived --project myproj
engram consolidate --project myproj          # report exact duplicates (dry-run)
engram consolidate --project myproj --near   # also report near-duplicates (fuzzy)
engram consolidate --project myproj --apply  # archive duplicates

# Re-index embeddings (requires the `semantic` feature)
engram reindex --project myproj              # re-embed all active memories
engram reindex --project myproj --dry-run    # preview without writing
```

---

## 🔄 Bootstrapping an Existing Project

When you adopt Engram on a project that already has history, the `engram.bootstrap` prompt + `collect_sources` tool seed memory from what's already there — git history, docs, past decisions, CI, fixes — instead of starting from zero.

**How it works:** Engram only *gathers evidence* (it stays local and LLM-free). Your agent reads that evidence and writes the actual memories, following the quality bars in the bootstrap prompt. The result is high-signal, traceable memories — not a noisy dump.

1. Have your MCP client fetch the `engram.bootstrap` prompt (the same guidance is in `docs/bootstrap.md`).
2. The prompt instructs the agent to call `collect_sources`, distill the evidence into the four memory types, and tag everything `bootstrap` for later review/cleanup.

What `collect_sources` gathers per dimension:

| Dimension | Carriers scanned | Target memory |
|-----------|------------------|---------------|
| `git` | commit themes (clustered by type/scope), migration & breaking signals | episodic |
| `decisions` | docs (README/ADR/CLAUDE.md), `// WHY:` / `//!` annotations, dependency manifests | decision |
| `failures` | `fix:` / `hotfix:` commits, CHANGELOG "Fixed" sections | failure |
| `workflow` | CI pipelines, Makefile/justfile/scripts, lint & convention config | procedural |

```bash
# CLI equivalent
engram collect --project myproj --repo .                          # all dimensions
engram collect --project myproj --repo . --dimensions git,decisions
```

Re-running is idempotent — commits already stored as episodic memories are skipped.

---

## 🗺️ Roadmap

- [x] SQLite + FTS5 storage with 4 memory types
- [x] MCP server (stdio transport)
- [x] BM25 retrieval with intent classification
- [x] Relationship graph engine
- [x] Git integration (auto-ingest commits)
- [x] Project bootstrap (collect_sources + prompts)
- [x] CLI interface
- [x] Memory consolidation (dedup + soft-delete lifecycle: forget/restore/update)
- [x] Embedding-based semantic search (candle + RRF fusion, behind the `semantic` feature)
- [ ] Reflection engine (self-improving retrieval)
- [ ] HTTP MCP transport (for remote access)
- [ ] Multi-agent memory sharing

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

[MIT](LICENSE) © 2025-2026 Uzziah
