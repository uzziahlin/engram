use crate::config::Config;
use crate::git_integration::GitIntegration;
use crate::models::*;
use crate::reflection::ReflectionEngine;
use crate::storage::MemoryRepository;
use anyhow::{Context, Result};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Agent-facing guide template, embedded so the single-file binary needs no
/// runtime assets. `{{PROJECT_ID}}` is the only placeholder.
const ENGRAM_GUIDE_TEMPLATE: &str = include_str!("templates/engram_guide.md.tmpl");

/// The line written into CLAUDE.md to pull in the guide via Claude Code's @import.
const IMPORT_LINE: &str = "@ENGRAM.md";

fn open_repo(config: &Config) -> Result<MemoryRepository> {
    let repo = MemoryRepository::new(&config.storage.database_path)?;
    // Ensure the schema is current — creates any new tables (e.g. query_log)
    // on databases that predate them. Idempotent via CREATE ... IF NOT EXISTS.
    repo.initialize_schema()?;
    Ok(repo)
}

fn load_config() -> Result<Config> {
    Config::load()
}

fn now_ts() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Print JSON output to stdout.
///
/// Serialization failure is logged and swallowed rather than panicking: CLI
/// output is best-effort, and aborting the whole command on a pretty-print
/// failure would be worse than a missing JSON blob. Kept `()`-returning so the
/// 16 call sites need no change.
fn print_json(value: &serde_json::Value) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => tracing::error!("failed to serialize JSON output: {e}"),
    }
}

/// Extract a required string argument by name (--name value).
/// Validates that the value is not another flag (starts with --).
fn require_str(args: &[String], name: &str) -> Result<String> {
    let flag = format!("--{name}");
    let pos = args
        .iter()
        .position(|a| a == &flag)
        .context(format!("Missing required argument: --{name}"))?;
    let value = args
        .get(pos + 1)
        .cloned()
        .context(format!("--{name} requires a value"))?;
    if value.starts_with("--") {
        anyhow::bail!("--{name} requires a value, but got flag '{value}'");
    }
    Ok(value)
}

/// Extract an optional string argument by name.
fn optional_str(args: &[String], name: &str) -> Option<String> {
    let flag = format!("--{name}");
    let pos = args.iter().position(|a| a == &flag)?;
    args.get(pos + 1).cloned()
}

/// Extract a numeric argument by name. Non-numeric, NaN, and infinite values
/// are rejected (None) so they can't silently become a 0/usize::MAX limit.
fn optional_num(args: &[String], name: &str) -> Option<f64> {
    optional_str(args, name)
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|n| n.is_finite())
}

/// Extract a repeated argument (--tag a --tag b).
fn repeated_args(args: &[String], name: &str) -> Vec<String> {
    let flag = format!("--{name}");
    args.iter()
        .enumerate()
        .filter(|(i, a)| a == &&flag && *i + 1 < args.len())
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// Comma-separated list argument (--files a.rs,b.rs).
fn comma_list(args: &[String], name: &str) -> Vec<String> {
    optional_str(args, name)
        .map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Prompt a y/N question on the terminal. Returns true only for y/yes.
fn prompt_yes_no(question: &str) -> Result<bool> {
    print!("{question} [y/N] ");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let ans = line.trim().to_lowercase();
    Ok(ans == "y" || ans == "yes")
}

/// Render the agent guide with the project_id substituted in.
fn render_guide(project_id: &str) -> String {
    ENGRAM_GUIDE_TEMPLATE.replace("{{PROJECT_ID}}", project_id)
}

/// Result of importing the guide reference into CLAUDE.md.
#[derive(Debug, PartialEq, Eq)]
enum ImportOutcome {
    Created,
    Appended,
    AlreadyPresent,
}

/// Append `@ENGRAM.md` to `<dir>/CLAUDE.md`, creating the file if missing.
/// Idempotent: an exact `@ENGRAM.md` line already present is left untouched.
fn import_into_claude_md(dir: &Path) -> Result<ImportOutcome> {
    let path = dir.join("CLAUDE.md");
    if !path.exists() {
        std::fs::write(&path, format!("{IMPORT_LINE}\n"))
            .with_context(|| format!("failed to create {}", path.display()))?;
        return Ok(ImportOutcome::Created);
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if content.lines().any(|l| l.trim() == IMPORT_LINE) {
        return Ok(ImportOutcome::AlreadyPresent);
    }
    let mut updated = content;
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(IMPORT_LINE);
    updated.push('\n');
    std::fs::write(&path, updated)
        .with_context(|| format!("failed to update {}", path.display()))?;
    Ok(ImportOutcome::Appended)
}

// ─── Commands ─────────────────────────────────────────────────────

pub fn init(_args: &[String]) -> Result<()> {
    // A broken config.toml must be loud, not silently replaced by defaults —
    // otherwise `init` builds the DB at the default path while the user's
    // configured path (and the MCP server) point elsewhere.
    let config = match load_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: config load failed ({e:#}); using defaults");
            Config::default()
        }
    };

    if let Some(parent) = config.storage.database_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let repo = open_repo(&config)?;
    repo.initialize_schema()?;

    println!(
        "Initialized engram database at {:?}",
        config.storage.database_path
    );
    Ok(())
}

pub fn init_guide(args: &[String]) -> Result<()> {
    // Target directory (default: current dir).
    let dir = PathBuf::from(optional_str(args, "dir").unwrap_or_else(|| ".".to_string()));
    if !dir.is_dir() {
        anyhow::bail!("directory not found: {}", dir.display());
    }

    // project_id: explicit --project, else basename of the absolute dir.
    let project_id = match optional_str(args, "project") {
        Some(p) => p,
        None => {
            let abs = std::fs::canonicalize(&dir)
                .with_context(|| format!("failed to resolve {}", dir.display()))?;
            abs.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .context("could not derive project_id from directory; pass --project")?
        }
    };

    // 1. Write ENGRAM.md (skip if present without --force).
    let force = args.iter().any(|a| a == "--force");
    let guide_path = dir.join("ENGRAM.md");
    if guide_path.exists() && !force {
        println!("ENGRAM.md already exists, skipping (use --force to overwrite)");
    } else {
        std::fs::write(&guide_path, render_guide(&project_id))
            .with_context(|| format!("failed to write {}", guide_path.display()))?;
        println!("Wrote {}", guide_path.display());
    }

    // 2. Decide whether to import into CLAUDE.md.
    let do_import = if args.iter().any(|a| a == "--import") {
        true
    } else if args.iter().any(|a| a == "--no-import") {
        false
    } else if io::stdin().is_terminal() {
        prompt_yes_no("Add '@ENGRAM.md' to CLAUDE.md?")?
    } else {
        println!("Not a terminal; skipping CLAUDE.md import. Re-run with --import, or add '@ENGRAM.md' to CLAUDE.md manually.");
        false
    };

    // 3. Import if requested.
    if do_import {
        match import_into_claude_md(&dir)? {
            ImportOutcome::Created => println!("Created CLAUDE.md with @ENGRAM.md"),
            ImportOutcome::Appended => println!("Added @ENGRAM.md to CLAUDE.md"),
            ImportOutcome::AlreadyPresent => println!("CLAUDE.md already imports @ENGRAM.md"),
        }
    }

    println!("Done. Ensure engram is registered (claude mcp add) and restart your editor for the guide to take effect.");
    Ok(())
}

/// Whether a memory passes the `--tag`/`--before` filters (ANY-tag semantics,
/// matching forget-batch and the MCP search filter).
fn matches_filters(tags: &[String], created_at: i64, ftags: &[String], before: Option<i64>) -> bool {
    (ftags.is_empty() || tags.iter().any(|t| ftags.contains(t)))
        && before.map_or(true, |b| created_at < b)
}

pub fn search(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let query = require_str(args, "query")?;
    let memory_type = optional_str(args, "type");
    let limit = optional_num(args, "limit").unwrap_or(10.0) as usize;
    let tags = repeated_args(args, "tag");
    let before = optional_num(args, "before").map(|n| n as i64);

    let config = load_config()?;
    let repo = open_repo(&config)?;

    // Collect (bm25 score, item) pairs so the all-types path can merge-sort
    // across types instead of concatenating in fixed type order and letting
    // episodic crowd out everything else at the truncate boundary.
    let mut scored: Vec<(f64, serde_json::Value)> = if let Some(ref mt) = memory_type {
        match mt.as_str() {
            "episodic" => repo
                .search_episodic(&query, &project_id, limit)?
                .into_iter()
                .filter(|m| matches_filters(&m.memory.tags, m.memory.created_at, &tags, before))
                .map(|m| {
                    (
                        m.bm25_score,
                        serde_json::json!({
                            "id": m.memory.id, "type": "episodic", "summary": m.memory.summary,
                            "content": m.memory.content, "files": m.memory.files_touched,
                            "importance": m.memory.importance, "tags": m.memory.tags,
                            "created_at": m.memory.created_at,
                        }),
                    )
                })
                .collect(),
            "decision" => repo
                .search_decisions(&query, &project_id, limit)?
                .into_iter()
                .filter(|m| matches_filters(&m.memory.tags, m.memory.created_at, &tags, before))
                .map(|m| {
                    (
                        m.bm25_score,
                        serde_json::json!({
                            "id": m.memory.id, "type": "decision", "title": m.memory.title,
                            "context": m.memory.context, "rationale": m.memory.rationale,
                            "tradeoffs": m.memory.tradeoffs, "tags": m.memory.tags,
                            "created_at": m.memory.created_at,
                        }),
                    )
                })
                .collect(),
            "failure" => repo
                .search_failures(&query, &project_id, limit)?
                .into_iter()
                .filter(|m| matches_filters(&m.memory.tags, m.memory.created_at, &tags, before))
                .map(|m| {
                    (
                        m.bm25_score,
                        serde_json::json!({
                            "id": m.memory.id, "type": "failure", "incident": m.memory.incident,
                            "root_cause": m.memory.root_cause, "fix": m.memory.fix,
                            "prevention": m.memory.prevention, "severity": m.memory.severity,
                            "tags": m.memory.tags, "created_at": m.memory.created_at,
                        }),
                    )
                })
                .collect(),
            "procedural" => repo
                .search_procedural(&query, &project_id, limit)?
                .into_iter()
                .filter(|m| matches_filters(&m.memory.tags, m.memory.created_at, &tags, before))
                .map(|m| {
                    (
                        m.bm25_score,
                        serde_json::json!({
                            "id": m.memory.id, "type": "procedural", "workflow": m.memory.workflow_name,
                            "steps": m.memory.steps, "tools": m.memory.related_tools,
                            "tags": m.memory.tags, "created_at": m.memory.created_at,
                        }),
                    )
                })
                .collect(),
            _ => anyhow::bail!(
                "Unknown memory type: {mt}. Use: episodic, decision, failure, procedural"
            ),
        }
    } else {
        // Search all types — errors propagate (`?`): a DB failure must not
        // masquerade as "no results" with exit code 0.
        let mut all: Vec<(f64, serde_json::Value)> = Vec::new();
        for m in repo.search_episodic(&query, &project_id, limit)? {
            if matches_filters(&m.memory.tags, m.memory.created_at, &tags, before) {
                all.push((
                    m.bm25_score,
                    serde_json::json!({
                        "id": m.memory.id, "type": "episodic", "summary": m.memory.summary,
                        "content": m.memory.content, "importance": m.memory.importance,
                        "tags": m.memory.tags, "created_at": m.memory.created_at,
                    }),
                ));
            }
        }
        for m in repo.search_decisions(&query, &project_id, limit)? {
            if matches_filters(&m.memory.tags, m.memory.created_at, &tags, before) {
                all.push((
                    m.bm25_score,
                    serde_json::json!({
                        "id": m.memory.id, "type": "decision", "title": m.memory.title,
                        "rationale": m.memory.rationale, "tradeoffs": m.memory.tradeoffs,
                        "tags": m.memory.tags, "created_at": m.memory.created_at,
                    }),
                ));
            }
        }
        for m in repo.search_failures(&query, &project_id, limit)? {
            if matches_filters(&m.memory.tags, m.memory.created_at, &tags, before) {
                all.push((
                    m.bm25_score,
                    serde_json::json!({
                        "id": m.memory.id, "type": "failure", "incident": m.memory.incident,
                        "severity": m.memory.severity, "fix": m.memory.fix,
                        "root_cause": m.memory.root_cause, "prevention": m.memory.prevention,
                        "tags": m.memory.tags, "created_at": m.memory.created_at,
                    }),
                ));
            }
        }
        for m in repo.search_procedural(&query, &project_id, limit)? {
            if matches_filters(&m.memory.tags, m.memory.created_at, &tags, before) {
                all.push((
                    m.bm25_score,
                    serde_json::json!({
                        "id": m.memory.id, "type": "procedural", "workflow": m.memory.workflow_name,
                        "steps": m.memory.steps, "tags": m.memory.tags,
                        "created_at": m.memory.created_at,
                    }),
                ));
            }
        }
        all
    };

    // Merge-sort by BM25 score across types, then truncate — replaces the old
    // fixed-order concat + truncate that always favored episodic.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let results: Vec<serde_json::Value> = scored
        .into_iter()
        .take(limit)
        .map(|(_, item)| item)
        .collect();

    // Best-effort retrieval feedback (mirrors MCP search_memory): log this
    // query + its hits so `engram queries` / MCP `query_stats` can surface
    // hit-rate signal. A logging failure never breaks the command.
    let result_ids: Vec<String> = results
        .iter()
        .filter_map(|v| v.get("id").and_then(|i| i.as_str()).map(String::from))
        .collect();
    if let Err(e) = repo.record_query(
        &project_id,
        &query,
        &result_ids,
        memory_type.as_deref(),
        now_ts(),
    ) {
        tracing::warn!("query log failed: {e}");
    }

    print_json(&serde_json::json!({"results": results, "total": results.len()}));
    Ok(())
}

pub fn create_episodic(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let session_id = optional_str(args, "session").unwrap_or_else(|| "cli".into());
    let summary = require_str(args, "summary")?;
    let content = optional_str(args, "content").unwrap_or_else(|| summary.clone());
    let files = comma_list(args, "files");
    let commits = comma_list(args, "commits");
    let importance = optional_num(args, "importance").unwrap_or(0.5) as f32;
    let tags = repeated_args(args, "tag");

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_ts();

    let memory = EpisodicMemory {
        id: id.clone(),
        project_id,
        session_id,
        summary,
        content,
        files_touched: files,
        related_commits: commits,
        importance,
        tags,
        created_at: now,
        updated_at: now,
    };

    repo.create_episodic(&memory)?;

    print_json(&serde_json::json!({"id": id, "status": "created", "created_at": now}));
    Ok(())
}

pub fn create_decision(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let title = require_str(args, "title")?;
    let context = require_str(args, "context")?;
    let rationale = require_str(args, "rationale")?;
    let tradeoffs = optional_str(args, "tradeoffs").unwrap_or_default();
    let files = comma_list(args, "files");
    let tags = repeated_args(args, "tag");

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_ts();

    let memory = DecisionMemory {
        id: id.clone(),
        project_id,
        title,
        context,
        rationale,
        tradeoffs,
        related_files: files,
        tags,
        created_at: now,
        updated_at: now,
    };

    repo.create_decision(&memory)?;

    print_json(&serde_json::json!({"id": id, "status": "created", "created_at": now}));
    Ok(())
}

pub fn create_failure(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let incident = require_str(args, "incident")?;
    let root_cause = require_str(args, "root-cause")?;
    let fix = require_str(args, "fix")?;
    let prevention = require_str(args, "prevention")?;
    let severity = optional_num(args, "severity").unwrap_or(3.0) as u8;
    let tags = repeated_args(args, "tag");

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_ts();

    let memory = FailureMemory {
        id: id.clone(),
        project_id,
        incident,
        root_cause,
        fix,
        prevention,
        severity,
        tags,
        created_at: now,
        updated_at: now,
    };

    repo.create_failure(&memory)?;

    print_json(
        &serde_json::json!({"id": id, "status": "created", "severity": severity, "created_at": now}),
    );
    Ok(())
}

pub fn create_procedural(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let workflow_name = require_str(args, "name")?;
    let steps_str = require_str(args, "steps")?;
    let steps: Vec<String> = steps_str.split(',').map(|s| s.trim().to_string()).collect();
    let tools = comma_list(args, "tools");
    let tags = repeated_args(args, "tag");

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_ts();

    let memory = ProceduralMemory {
        id: id.clone(),
        project_id,
        workflow_name,
        steps,
        related_tools: tools,
        tags,
        created_at: now,
        updated_at: now,
    };

    repo.create_procedural(&memory)?;

    print_json(&serde_json::json!({"id": id, "status": "created", "created_at": now}));
    Ok(())
}

pub fn ingest(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let repo_path = require_str(args, "repo")?;
    let count = (optional_num(args, "count").unwrap_or(20.0) as usize).clamp(1, 1000);
    let session_id = optional_str(args, "session").unwrap_or_else(|| "auto-ingest".into());

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let git = GitIntegration::new(Path::new(&repo_path))?;
    let events = git.get_recent_commits(count)?;

    // Dedup BEFORE generating (skip tree-diff work for known commits)…
    let ingested_hashes = repo.get_ingested_commits(&project_id)?;
    let fresh: Vec<crate::git_integration::CommitEvent> = events
        .into_iter()
        .filter(|e| !ingested_hashes.contains(&e.commit_hash))
        .collect();
    let skipped = count.saturating_sub(fresh.len());

    // …then distill into one memory per (type, scope) milestone — same
    // clustering the bootstrap collector uses, instead of one noisy memory
    // per commit.
    let memories = crate::git_integration::milestone_memories(&project_id, &session_id, &fresh);

    let mut ingested = Vec::new();
    for mem in &memories {
        repo.create_episodic(mem)?;
        ingested.push(serde_json::json!({
            "id": mem.id,
            "summary": mem.summary,
            "commits": mem.related_commits.len(),
            "files": mem.files_touched,
        }));
    }

    print_json(&serde_json::json!({
        "ingested": ingested.len(),
        "total_commits_scanned": count,
        "skipped_duplicates": skipped,
        "memories": ingested,
    }));
    Ok(())
}

pub fn collect(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let repo_path = require_str(args, "repo")?;
    let dimensions = optional_str(args, "dimensions");
    let max_commits = (optional_num(args, "max-commits").unwrap_or(200.0) as usize).min(1000);

    let config = load_config()?;
    let repo = open_repo(&config)?;
    // `collect` is a bootstrap entry point — ensure schema so it works on a
    // fresh database without requiring a prior `engram init`.
    repo.initialize_schema()?;

    let dims = crate::collectors::Dimension::parse_list(dimensions.as_deref());
    if dims.is_empty() {
        anyhow::bail!("no valid dimensions parsed; valid: git, decisions, failures, workflow");
    }

    // Mirror the MCP tool: reuse commit-hash dedup so re-running collect stays idempotent.
    let ingested = repo.get_ingested_commits(&project_id)?;
    let opts = crate::collectors::CollectOptions {
        max_commits,
        ingested_commit_hashes: ingested,
        ..Default::default()
    };

    let sources = crate::collectors::collect(&project_id, Path::new(&repo_path), &dims, &opts)?;
    print_json(&serde_json::to_value(&sources)?);
    Ok(())
}

pub fn recent_failures(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let service = optional_str(args, "service");
    let limit = optional_num(args, "limit").unwrap_or(5.0) as usize;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let query = service.as_deref().unwrap_or("");
    let results = if query.is_empty() {
        repo.list_recent_failures(&project_id, limit)?
    } else {
        repo.search_failures(query, &project_id, limit)?
            .into_iter()
            .map(|s| s.memory)
            .collect()
    };

    let failures: Vec<serde_json::Value> = results
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "incident": f.incident,
                "root_cause": f.root_cause,
                "fix": f.fix,
                "severity": f.severity,
                "created_at": f.created_at,
            })
        })
        .collect();

    print_json(&serde_json::json!({"failures": failures, "total": failures.len()}));
    Ok(())
}

pub fn decisions(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let topic = optional_str(args, "topic");
    let limit = optional_num(args, "limit").unwrap_or(5.0) as usize;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let query = topic.as_deref().unwrap_or("");
    let results = if query.is_empty() {
        repo.list_recent_decisions(&project_id, limit)?
    } else {
        repo.search_decisions(query, &project_id, limit)?
            .into_iter()
            .map(|s| s.memory)
            .collect()
    };

    let decisions: Vec<serde_json::Value> = results
        .iter()
        .map(|d| {
            serde_json::json!({
                "id": d.id,
                "title": d.title,
                "context": d.context,
                "rationale": d.rationale,
                "tradeoffs": d.tradeoffs,
                "created_at": d.created_at,
            })
        })
        .collect();

    print_json(&serde_json::json!({"decisions": decisions, "total": decisions.len()}));
    Ok(())
}

pub fn timeline(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let days = optional_num(args, "days").unwrap_or(7.0) as i64;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let since = now_ts() - days.saturating_mul(86400);
    let conn = repo.connection()?;

    let mut stmt = conn.prepare(
        "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
         FROM episodic_memories
         WHERE project_id = ?1 AND created_at >= ?2 AND archived_at IS NULL
         GROUP BY day ORDER BY day DESC",
    )?;

    let rows = stmt.query_map(rusqlite::params![project_id, since], |row| {
        let day: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok(serde_json::json!({"date": day, "episodic_count": count}))
    })?;

    let events: Vec<serde_json::Value> = rows.filter_map(|r| r.ok()).collect();

    print_json(&serde_json::json!({"events": events}));
    Ok(())
}

/// `engram queries` — retrieval feedback: aggregate past search queries by
/// frequency and average hit count. Surfaces which queries are common and
/// which return few results.
pub fn queries(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let days = optional_num(args, "days").unwrap_or(7.0) as i64;
    let limit = optional_num(args, "limit").unwrap_or(10.0) as usize;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let since = now_ts() - days.saturating_mul(86400);
    let stats = repo.query_stats(&project_id, since, limit)?;

    print_json(&serde_json::json!({
        "queries": serde_json::to_value(&stats)?
    }));
    Ok(())
}

pub fn forget(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;
    let kind = crate::storage::MemoryKind::from_type_str(&memory_type)?;

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let archived = repo.archive(kind, &id, &project_id, now_ts())?;
    print_json(&serde_json::json!({"id": id, "archived": archived}));
    Ok(())
}

pub fn restore(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;
    let kind = crate::storage::MemoryKind::from_type_str(&memory_type)?;

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let restored = repo.restore(kind, &id, &project_id)?;
    print_json(&serde_json::json!({"id": id, "restored": restored}));
    Ok(())
}

pub fn update(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;
    let kind = crate::storage::MemoryKind::from_type_str(&memory_type)?;

    // Build a patch object from generic --set key=value pairs.
    let mut patch = serde_json::Map::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--set" {
            if let Some(kv) = args.get(i + 1) {
                match kv.split_once('=') {
                    Some((k, v)) => {
                        // Parse the value as JSON when possible (numbers, bools,
                        // arrays); fall back to a plain string. Always-string
                        // values made `--set importance=0.9` fail type
                        // checking against the typed model.
                        let parsed = serde_json::from_str::<serde_json::Value>(v)
                            .unwrap_or_else(|_| serde_json::json!(v));
                        patch.insert(k.to_string(), parsed);
                    }
                    None => {
                        anyhow::bail!("--set expects key=value, got '{kv}' (missing '=')");
                    }
                }
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();

    macro_rules! guarded {
        ($get:ident, $update:ident) => {{
            let existing = repo
                .$get(&id)?
                .ok_or_else(|| anyhow::anyhow!("memory not found: {id}"))?;
            if existing.project_id != project_id {
                anyhow::bail!("memory does not belong to project {project_id}");
            }
            let mut obj = match serde_json::to_value(&existing)? {
                serde_json::Value::Object(m) => m,
                _ => anyhow::bail!("memory did not serialize to object"),
            };
            for (k, v) in &patch {
                if ["id", "project_id", "created_at", "memory_type"].contains(&k.as_str()) {
                    continue;
                }
                if obj.contains_key(k) {
                    obj.insert(k.clone(), v.clone());
                } else {
                    // A typo'd key used to vanish silently; say so.
                    eprintln!("warning: unknown field '{k}' ignored (no such field on this memory type)");
                }
            }
            obj.insert("updated_at".into(), serde_json::json!(now));
            let updated = serde_json::from_value(serde_json::Value::Object(obj))?;
            repo.$update(&updated)?;
        }};
    }
    use crate::storage::MemoryKind::*;
    match kind {
        Episodic => guarded!(get_episodic, update_episodic),
        Decision => guarded!(get_decision, update_decision),
        Failure => guarded!(get_failure, update_failure),
        Procedural => guarded!(get_procedural, update_procedural),
    }
    print_json(&serde_json::json!({"id": id, "status": "updated"}));
    Ok(())
}

pub fn forget_batch(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = optional_str(args, "type");
    let tags = repeated_args(args, "tag");
    let before = optional_num(args, "before").map(|n| n as i64);
    let apply = args.iter().any(|a| a == "--apply");

    let kinds = match memory_type {
        Some(s) => vec![crate::storage::MemoryKind::from_type_str(&s)?],
        None => crate::storage::MemoryKind::all().to_vec(),
    };

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();
    let mut matched = Vec::new();
    for kind in kinds {
        let ids = if apply {
            repo.archive_batch(kind, &project_id, &tags, before, now)?
        } else {
            repo.list_active_candidates(kind, &project_id, &tags, before)?
        };
        for id in ids {
            matched.push(serde_json::json!({"id": id, "memory_type": kind.as_str()}));
        }
    }
    print_json(&serde_json::json!({"applied": apply, "matched": matched, "count": matched.len()}));
    Ok(())
}

pub fn list_archived(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = optional_str(args, "type");
    let limit = optional_num(args, "limit").unwrap_or(20.0) as usize;
    let kinds = match memory_type {
        Some(s) => vec![crate::storage::MemoryKind::from_type_str(&s)?],
        None => crate::storage::MemoryKind::all().to_vec(),
    };
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let mut archived = Vec::new();
    for kind in kinds {
        for row in repo.list_archived(kind, &project_id, limit)? {
            archived.push(serde_json::to_value(&row)?);
        }
    }
    print_json(&serde_json::json!({"archived": archived, "count": archived.len()}));
    Ok(())
}

/// `engram gc [--older-than <dur> | --all] [--apply] [--vacuum] [--no-checkpoint]`
///
/// Physically delete archived memories and reclaim space. By default this is a
/// **dry run**: it reports what *would* be removed without touching anything.
/// Pass `--apply` to commit.
///
/// - `--older-than <dur>`: only delete memories archived longer ago than this
///   duration (`30d`, `12h`, `45m`, `8w`, `30s`). Mutually exclusive with `--all`.
/// - `--all`: delete every archived memory regardless of age.
/// - `--vacuum`: after deletion, rebuild the DB file to reclaim free pages.
///   Implies `--apply`. Requires no other process (e.g. the MCP server) to be
///   using the database.
/// - `--no-checkpoint`: skip the WAL checkpoint that otherwise follows `--apply`.
///
/// GC is global (cross-project); it is a maintenance op, not a per-project query.
pub fn gc(args: &[String]) -> Result<()> {
    let apply = args.iter().any(|a| a == "--apply");
    let vacuum = args.iter().any(|a| a == "--vacuum");
    let no_checkpoint = args.iter().any(|a| a == "--no-checkpoint");
    let all = args.iter().any(|a| a == "--all");
    let older_than = optional_str(args, "older-than");
    // --vacuum only makes sense with --apply.
    let apply = apply || vacuum;

    if all && older_than.is_some() {
        anyhow::bail!("--all and --older-than are mutually exclusive");
    }
    let older_than_seconds = if all {
        0i64
    } else if let Some(d) = older_than {
        parse_duration(&d)?
    } else {
        anyhow::bail!(
            "gc requires either --older-than <dur> (e.g. 30d, 12h, 45m, 8w, 30s) or --all"
        );
    };

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();

    let report = repo.gc_archived(older_than_seconds, apply, now)?;
    print_json(&serde_json::to_value(&report)?);

    // Reclaim space: checkpoint the WAL (best-effort), then optionally VACUUM.
    if apply && !no_checkpoint {
        if let Err(e) = repo.wal_checkpoint_truncate() {
            tracing::warn!("wal_checkpoint skipped: {e}");
        }
    }
    if vacuum {
        repo.vacuum().with_context(|| {
            "VACUUM failed — ensure no other process (e.g. the MCP server) is \
             using the database, then retry"
        })?;
    }

    Ok(())
}

/// Parse a short duration string (`30d`, `12h`, `45m`, `8w`, `30s`) into seconds.
/// Units: s=second, m=minute, h=hour, d=day, w=week.
fn parse_duration(s: &str) -> Result<i64> {
    let mut chars = s.chars();
    let unit = chars.next_back();
    let (unit, n): (char, i64) = match unit {
        Some(u @ ('s' | 'm' | 'h' | 'd' | 'w')) => {
            let num_str = chars.as_str();
            let n: i64 = num_str.parse().with_context(|| {
                format!("invalid duration '{s}': number part is not an integer")
            })?;
            (u, n)
        }
        _ => anyhow::bail!("invalid duration '{s}': expected e.g. 30d, 12h, 45m, 8w, 30s"),
    };
    let per_unit: i64 = match unit {
        's' => 1,
        'm' => 60,
        'h' => 3_600,
        'd' => 86_400,
        'w' => 604_800,
        // unreachable: unit is constrained above
        _ => unreachable!(),
    };
    n.checked_mul(per_unit)
        .with_context(|| format!("duration '{s}' overflows i64"))
}

pub fn consolidate(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = optional_str(args, "type");
    let include_near_dup = args.iter().any(|a| a == "--near");
    let apply = args.iter().any(|a| a == "--apply");
    let kinds = match memory_type {
        Some(s) => vec![crate::storage::MemoryKind::from_type_str(&s)?],
        None => crate::storage::MemoryKind::all().to_vec(),
    };
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let engine = crate::consolidation::ConsolidationEngine::new();
    let plans = engine.consolidate(
        &repo,
        &project_id,
        &kinds,
        include_near_dup,
        crate::consolidation::engine::DEFAULT_JACCARD_THRESHOLD,
        apply,
        now_ts(),
    )?;
    let total_archived: usize = plans.iter().map(|p| p.archived).sum();
    print_json(
        &serde_json::json!({"applied": apply, "plans": plans, "total_archived": total_archived}),
    );
    Ok(())
}

/// `engram reflect --project <id> [--min <n>] [--apply]` — scan a project's
/// active failures and propose preventive procedural rules for any tag
/// recurring at least `--min` times (default: `[reflection].min_occurrences`).
///
/// Dry run by default: prints proposals without writing. Pass `--apply` to
/// persist them as pending suggestions, which stay invisible to search until
/// confirmed via `confirm-suggestion`.
pub fn reflect(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let apply = args.iter().any(|a| a == "--apply");
    let config = load_config()?;
    let min = optional_num(args, "min")
        .map(|n| n as usize)
        .unwrap_or(config.reflection.min_occurrences);

    let repo = open_repo(&config)?;
    let engine = ReflectionEngine::with_min_occurrences(min);
    let plan = engine.reflect(&repo, &project_id, apply, now_ts())?;

    print_json(&serde_json::json!({
        "applied": apply,
        "min_occurrences": min,
        "proposed": plan.suggestions.len(),
        "created": plan.created,
        "suggestions": serde_json::to_value(&plan.suggestions)?,
    }));
    Ok(())
}

/// `engram suggestions --project <id>` — list pending reflection proposals
/// awaiting confirmation (each distills a recurring-failure tag into a draft
/// preventive rule).
pub fn suggestions(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let pending = repo.list_pending_suggestions(&project_id)?;
    print_json(&serde_json::json!({
        "count": pending.len(),
        "suggestions": serde_json::to_value(&pending)?,
    }));
    Ok(())
}

/// `engram confirm-suggestion --project <id> --id <suggestion-id>` — promote a
/// pending proposal into a searchable procedural memory.
pub fn confirm_suggestion(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let id = require_str(args, "id")?;
    let config = load_config()?;
    let repo = open_repo(&config)?;
    match repo.confirm_suggestion(&id, &project_id, now_ts())? {
        Some(proc_id) => print_json(&serde_json::json!({
            "id": id,
            "status": "confirmed",
            "procedural_id": proc_id,
        })),
        None => anyhow::bail!("no pending suggestion '{id}' in project '{project_id}'"),
    }
    Ok(())
}

/// `engram reject-suggestion --project <id> --id <suggestion-id>` — discard a
/// pending proposal without creating a procedural memory.
pub fn reject_suggestion(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let id = require_str(args, "id")?;
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let rejected = repo.reject_suggestion(&id, &project_id, now_ts())?;
    if !rejected {
        anyhow::bail!("no pending suggestion '{id}' in project '{project_id}'");
    }
    print_json(&serde_json::json!({ "id": id, "status": "rejected" }));
    Ok(())
}

/// `engram get --project <id> --type <t> --id <id>` — fetch one memory's full
/// record (CLI counterpart of the MCP get_memory tool).
pub fn get(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;
    let kind = crate::storage::MemoryKind::from_type_str(&memory_type)?;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    macro_rules! fetch {
        ($get:ident) => {{
            let mem = repo
                .$get(&id)?
                .ok_or_else(|| anyhow::anyhow!("memory not found: {id}"))?;
            if mem.project_id != project_id {
                anyhow::bail!("memory does not belong to project {project_id}");
            }
            serde_json::to_value(&mem)?
        }};
    }
    use crate::storage::MemoryKind::*;
    let memory = match kind {
        Episodic => fetch!(get_episodic),
        Decision => fetch!(get_decision),
        Failure => fetch!(get_failure),
        Procedural => fetch!(get_procedural),
    };
    print_json(&serde_json::json!({
        "memory_type": kind.as_str(),
        "memory": memory,
    }));
    Ok(())
}

/// `engram maintain [--project <id>] [--repo <path>] [--apply]` — one-shot
/// memory health pass. Designed to run from a hook/cron so the store doesn't
/// rot silently. Steps:
///
/// 1. consolidate: archive exact duplicates (soft-delete, reversible)
/// 2. rebuild FTS: repair orphan/missing/mis-preprocessed index rows (always)
/// 3. prune query_log past `[storage].query_log_retention_days` (apply only)
/// 4. staleness report: memories whose referenced files no longer exist
///    (read-only; requires `--repo`)
/// 5. gc preview: how many archived rows a later `engram gc --apply` would purge
///    (never auto-applied — physical deletion stays an explicit decision)
pub fn maintain(args: &[String]) -> Result<()> {
    let apply = args.iter().any(|a| a == "--apply");
    let project = optional_str(args, "project");
    let repo_path = optional_str(args, "repo");

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();
    let kinds = crate::storage::MemoryKind::all().to_vec();

    let projects = match &project {
        Some(p) => vec![p.clone()],
        None => repo.list_projects()?,
    };

    // 1. Consolidate exact duplicates per project.
    let engine = crate::consolidation::ConsolidationEngine::new();
    let mut dup_groups = 0usize;
    let mut dup_duplicates = 0usize;
    for p in &projects {
        let plans = engine.consolidate(&repo, p, &kinds, false, 0.0, apply, now)?;
        for plan in plans {
            dup_groups += plan.groups.len();
            dup_duplicates += plan
                .groups
                .iter()
                .map(|g| g.duplicate_ids.len())
                .sum::<usize>();
        }
    }

    // 2. FTS repair (idempotent — aligns the index with the main tables).
    let fts_rows = repo.rebuild_fts()?;

    // 3. query_log retention prune.
    let retention_secs =
        (config.storage.query_log_retention_days.saturating_mul(86_400)) as i64;
    let pruned = if apply {
        repo.prune_query_log(retention_secs, now)?
    } else {
        0
    };

    // 4. Staleness report (read-only).
    let stale = match &repo_path {
        Some(rp) => report_stale_memories(&repo, &projects, Path::new(rp))?,
        None => serde_json::json!({ "skipped": "pass --repo <path> to check file staleness" }),
    };

    // 5. GC preview (dry-run only; physical delete stays explicit).
    let gc_report = repo.gc_archived(0, false, now)?;

    print_json(&serde_json::json!({
        "applied": apply,
        "projects": projects,
        "consolidate": {
            "duplicate_groups": dup_groups,
            "duplicates": dup_duplicates,
            "action": if apply { "archived (restore with `engram restore`)" } else { "dry-run — pass --apply to archive" },
        },
        "fts_rebuilt_rows": fts_rows,
        "query_log_pruned": pruned,
        "stale": stale,
        "gc_preview": {
            "archived_eligible_for_gc": gc_report.deleted.len(),
            "note": "run `engram gc --older-than <dur> --apply` to physically purge",
        },
    }));
    Ok(())
}

/// Read-only staleness pass: flag active memories whose referenced files have
/// all disappeared from the working tree (deleted/renamed modules etc.).
/// Conservative — a memory with ANY surviving file is not reported.
fn report_stale_memories(
    repo: &MemoryRepository,
    projects: &[String],
    root: &Path,
) -> Result<serde_json::Value> {
    const MAX_REPORTED: usize = 100;

    let file_exists = |f: &str| {
        let p = Path::new(f);
        (p.is_absolute() && p.exists()) || root.join(p).exists()
    };

    let mut out = serde_json::Map::new();
    for p in projects {
        let mut stale = Vec::new();
        let mut total_checked = 0usize;

        let mut check = |memory_type: &str,
                         id: &str,
                         label: &str,
                         files: &[String],
                         stale: &mut Vec<serde_json::Value>| {
            if files.is_empty() {
                return; // no file references → nothing to check
            }
            total_checked += 1;
            let missing: Vec<&String> = files.iter().filter(|f| !file_exists(f)).collect();
            if missing.len() == files.len() {
                stale.push(serde_json::json!({
                    "memory_type": memory_type,
                    "id": id,
                    "label": label,
                    "files": files,
                }));
            }
        };

        for m in repo.list_active_episodic(Some(p))? {
            check("episodic", &m.id, &m.summary, &m.files_touched, &mut stale);
        }
        for m in repo.list_active_decision(Some(p))? {
            check("decision", &m.id, &m.title, &m.related_files, &mut stale);
        }

        let total_stale = stale.len();
        stale.truncate(MAX_REPORTED);
        out.insert(
            p.clone(),
            serde_json::json!({
                "memories_with_file_refs": total_checked,
                "stale": stale,
                "stale_total": total_stale,
                "truncated": total_stale > stale.len(),
            }),
        );
    }
    Ok(serde_json::Value::Object(out))
}

/// `engram session-import --project <id> --transcript <path> [--session <sid>]
/// [--dry-run]` — distill a Claude Code session transcript (JSONL) into one
/// episodic memory. Meant to be wired to a SessionEnd/Stop hook so memories
/// form without depending on the agent remembering to write them.
pub fn session_import(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let transcript = require_str(args, "transcript")?;
    let session_id =
        optional_str(args, "session").unwrap_or_else(|| session_id_from_path(&transcript));
    let dry_run = args.iter().any(|a| a == "--dry-run");

    let path = Path::new(&transcript);
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let digest = parse_transcript(&text);

    if digest.user_prompts.is_empty() && digest.files.is_empty() {
        anyhow::bail!(
            "transcript yielded no usable signal (no user prompts, no file edits); \
             not writing an empty memory"
        );
    }

    let now = now_ts();
    let first_prompt = digest
        .user_prompts
        .first()
        .map(|p| p.replace('\n', " "))
        .unwrap_or_else(|| format!("session {session_id}"));
    let summary: String = first_prompt.chars().take(160).collect();

    let mut content = String::new();
    content.push_str("User prompts:\n");
    for p in digest.user_prompts.iter().take(10) {
        let line: String = p.chars().take(400).collect();
        content.push_str(&format!("- {line}\n"));
    }
    content.push_str("\nAssistant conclusions (latest last):\n");
    for a in digest.assistant_texts.iter().rev().take(5).rev() {
        let line: String = a.chars().take(600).collect();
        content.push_str(&format!("- {line}\n"));
    }
    if !digest.files.is_empty() {
        content.push_str(&format!("\nFiles touched ({}):\n", digest.files.len()));
        content.push_str(&digest.files.iter().take(50).cloned().collect::<Vec<_>>().join(", "));
    }
    let content: String = content.chars().take(4000).collect();

    let memory = EpisodicMemory {
        id: uuid::Uuid::new_v4().to_string(),
        project_id: project_id.clone(),
        session_id: session_id.clone(),
        summary,
        content,
        files_touched: digest.files.iter().take(50).cloned().collect(),
        related_commits: vec![],
        importance: 0.5,
        tags: vec!["session-import".into()],
        created_at: now,
        updated_at: now,
    };

    if dry_run {
        print_json(&serde_json::json!({
            "dry_run": true,
            "would_create": serde_json::to_value(&memory)?,
        }));
        return Ok(());
    }

    let config = load_config()?;
    let repo = open_repo(&config)?;
    repo.create_episodic(&memory)?;
    print_json(&serde_json::json!({
        "id": memory.id,
        "status": "created",
        "session_id": session_id,
        "files": memory.files_touched.len(),
        "created_at": now,
    }));
    Ok(())
}

fn session_id_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "session".into())
}

/// What `parse_transcript` extracted from a session JSONL.
#[derive(Debug, Default)]
struct SessionDigest {
    user_prompts: Vec<String>,
    assistant_texts: Vec<String>,
    files: Vec<String>,
}

/// Parse a Claude Code transcript (JSONL, one message object per line).
/// Tolerant of schema drift: unknown line shapes are skipped, never fatal.
/// Tool-result payloads are ignored (noise); only real user prompts, visible
/// assistant text, and tool_use file targets are kept.
fn parse_transcript(text: &str) -> SessionDigest {
    let mut digest = SessionDigest::default();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue; // malformed line → skip, don't abort the whole import
        };
        // Claude Code marks injected/meta user messages; skip them.
        if v.get("isMeta").and_then(|m| m.as_bool()).unwrap_or(false) {
            continue;
        }
        let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let Some(message) = v.get("message") else {
            // "summary" lines carry session titles — useful as a prompt stand-in.
            if ty == "summary" {
                if let Some(s) = v.get("summary").and_then(|s| s.as_str()) {
                    if !s.trim().is_empty() && digest.user_prompts.is_empty() {
                        digest.user_prompts.push(format!("[session] {s}"));
                    }
                }
            }
            continue;
        };
        let content = message.get("content");

        match ty {
            "user" => match content {
                Some(serde_json::Value::String(s)) => {
                    let t = s.trim();
                    // Skip command-ish/meta wrappers Claude Code injects.
                    if !t.is_empty() && !t.starts_with('<') {
                        push_unique(&mut digest.user_prompts, t);
                    }
                }
                Some(serde_json::Value::Array(items)) => {
                    for item in items {
                        if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                let t = t.trim();
                                if !t.is_empty() && !t.starts_with('<') {
                                    push_unique(&mut digest.user_prompts, t);
                                }
                            }
                        }
                    }
                }
                _ => {}
            },
            "assistant" => {
                if let Some(serde_json::Value::Array(items)) = content {
                    for item in items {
                        if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                let t = t.trim();
                                if !t.is_empty() {
                                    push_unique(&mut digest.assistant_texts, t);
                                }
                            }
                        }
                        // Collect edited/read file targets from tool calls.
                        if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                            if let Some(input) = item.get("input") {
                                for key in ["file_path", "path", "notebook_path"] {
                                    if let Some(f) = input.get(key).and_then(|f| f.as_str()) {
                                        let f = f.trim();
                                        if !f.is_empty() {
                                            push_unique(&mut digest.files, f);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    digest
}

fn push_unique(list: &mut Vec<String>, item: &str) {
    if !list.iter().any(|s| s == item) {
        list.push(item.to_string());
    }
}

/// `engram reindex [--project <id>] [--force] [--dry-run]` — backfill embeddings
/// for active memories. Requires a binary built with `--features semantic`.
pub fn reindex(args: &[String]) -> Result<()> {
    #[cfg(not(feature = "semantic"))]
    {
        let _ = args;
        Err(anyhow::anyhow!(
            "reindex requires a binary built with --features semantic \
             (semantic support is compiled out of this build)"
        ))
    }
    #[cfg(feature = "semantic")]
    {
        let project = optional_str(args, "project");
        let force = args.iter().any(|a| a == "--force");
        let dry_run = args.iter().any(|a| a == "--dry-run");

        let config = load_config()?;
        let repo = open_repo(&config)?;
        repo.initialize_schema()?;
        let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
        let report = provider.reindex_embeddings(project.as_deref(), force, dry_run)?;
        print_json(&serde_json::to_value(&report)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_guide_substitutes_project_id() {
        let out = render_guide("my-proj");
        assert!(out.contains("project_id: \"my-proj\""));
        assert!(!out.contains("{{PROJECT_ID}}"));
    }

    #[test]
    fn import_creates_claude_md_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = import_into_claude_md(dir.path()).unwrap();
        assert_eq!(outcome, ImportOutcome::Created);
        let content = std::fs::read_to_string(dir.path().join("CLAUDE.md")).unwrap();
        assert!(content.lines().any(|l| l.trim() == "@ENGRAM.md"));
    }

    #[test]
    fn import_appends_when_line_absent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "# My rules\n").unwrap();
        let outcome = import_into_claude_md(dir.path()).unwrap();
        assert_eq!(outcome, ImportOutcome::Appended);
        let content = std::fs::read_to_string(dir.path().join("CLAUDE.md")).unwrap();
        assert!(content.contains("# My rules"));
        assert!(content.lines().any(|l| l.trim() == "@ENGRAM.md"));
    }

    #[test]
    fn import_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "@ENGRAM.md\n").unwrap();
        let outcome = import_into_claude_md(dir.path()).unwrap();
        assert_eq!(outcome, ImportOutcome::AlreadyPresent);
    }

    #[test]
    fn import_appends_without_gluing_when_no_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "rules").unwrap(); // no trailing newline
        let outcome = import_into_claude_md(dir.path()).unwrap();
        assert_eq!(outcome, ImportOutcome::Appended);
        let content = std::fs::read_to_string(dir.path().join("CLAUDE.md")).unwrap();
        assert!(content.lines().any(|l| l == "rules")); // stayed its own line
        assert!(content.lines().any(|l| l == "@ENGRAM.md")); // not glued onto "rules"
    }

    #[test]
    fn import_ignores_at_engram_substring_in_a_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "# see @ENGRAM.md docs\n").unwrap();
        let outcome = import_into_claude_md(dir.path()).unwrap();
        assert_eq!(outcome, ImportOutcome::Appended); // substring is NOT a present line
        let content = std::fs::read_to_string(dir.path().join("CLAUDE.md")).unwrap();
        assert!(content.lines().any(|l| l.trim() == "@ENGRAM.md"));
    }

    fn no_import_args(dir: &std::path::Path) -> Vec<String> {
        vec![
            "--dir".to_string(),
            dir.to_str().unwrap().to_string(),
            "--project".to_string(),
            "demo".to_string(),
            "--no-import".to_string(),
        ]
    }

    #[test]
    fn init_guide_no_import_writes_guide_only() {
        let dir = tempfile::tempdir().unwrap();
        init_guide(&no_import_args(dir.path())).unwrap();
        assert!(dir.path().join("ENGRAM.md").exists());
        assert!(!dir.path().join("CLAUDE.md").exists());
        let g = std::fs::read_to_string(dir.path().join("ENGRAM.md")).unwrap();
        assert!(g.contains("project_id: \"demo\""));
    }

    #[test]
    fn init_guide_skips_existing_without_force() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ENGRAM.md"), "OLD").unwrap();
        init_guide(&no_import_args(dir.path())).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("ENGRAM.md")).unwrap(),
            "OLD"
        );
    }

    #[test]
    fn init_guide_force_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ENGRAM.md"), "OLD").unwrap();
        let mut args = no_import_args(dir.path());
        args.push("--force".to_string());
        init_guide(&args).unwrap();
        let g = std::fs::read_to_string(dir.path().join("ENGRAM.md")).unwrap();
        assert!(g.contains("project_id: \"demo\""));
        assert_ne!(g, "OLD");
    }

    #[test]
    fn update_set_without_equals_is_an_error() {
        // `--set foo` (missing `=`) must error clearly, not silently skip.
        let args = vec![
            "update".to_string(),
            "--project".to_string(),
            "p".to_string(),
            "--type".to_string(),
            "episodic".to_string(),
            "--id".to_string(),
            "x".to_string(),
            "--set".to_string(),
            "no_equals_here".to_string(),
        ];
        let err = crate::cli::run(&args).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("--set") && msg.contains('='),
            "expected clear --set error mentioning '=', got: {msg}"
        );
    }

    #[test]
    fn parse_transcript_extracts_prompts_text_and_files() {
        let jsonl = r#"
{"type":"user","isMeta":true,"message":{"role":"user","content":"Caveat: injected"}}
{"type":"user","message":{"role":"user","content":"fix the auth middleware bug"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"..."},{"type":"text","text":"Root cause was a stale token."},{"type":"tool_use","name":"Edit","input":{"file_path":"/src/auth.rs"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Fixed and verified."},{"type":"tool_use","name":"Read","input":{"path":"/src/main.rs"}}]}}
{"type":"summary","summary":"Auth middleware fix"}
not json at all
"#;
        let d = parse_transcript(jsonl);
        assert_eq!(d.user_prompts, vec!["fix the auth middleware bug".to_string()]);
        assert_eq!(d.assistant_texts.len(), 2);
        assert!(d.assistant_texts.contains(&"Root cause was a stale token.".to_string()));
        assert!(d.files.contains(&"/src/auth.rs".to_string()));
        assert!(d.files.contains(&"/src/main.rs".to_string()));
    }

    #[test]
    fn parse_transcript_skips_command_wrappers_and_malformed() {
        let jsonl = r#"
{broken json}
{"type":"user","message":{"role":"user","content":"<command-name>/clear</command-name>"}}
"#;
        let d = parse_transcript(jsonl);
        assert!(d.user_prompts.is_empty());
        assert!(d.files.is_empty());
    }

    #[test]
    fn session_import_dry_run_writes_nothing_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join("sess.jsonl");
        std::fs::write(
            &transcript,
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"add rate limiting\"}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Done\"},{\"type\":\"tool_use\",\"name\":\"Edit\",\"input\":{\"file_path\":\"api/rate.rs\"}}]}}\n",
        )
        .unwrap();
        let args = vec![
            "--project".to_string(),
            "p".to_string(),
            "--transcript".to_string(),
            transcript.to_string_lossy().to_string(),
            "--dry-run".to_string(),
        ];
        session_import(&args).unwrap(); // must not panic; writes nothing (dry-run)
    }

    #[test]
    fn parse_duration_supports_units() {
        assert_eq!(parse_duration("30d").unwrap(), 30 * 86_400);
        assert_eq!(parse_duration("12h").unwrap(), 12 * 3_600);
        assert_eq!(parse_duration("45m").unwrap(), 45 * 60);
        assert_eq!(parse_duration("8w").unwrap(), 8 * 604_800);
        assert_eq!(parse_duration("30s").unwrap(), 30);
    }

    #[test]
    fn parse_duration_rejects_bad_input() {
        assert!(parse_duration("30").is_err()); // missing unit
        assert!(parse_duration("30x").is_err()); // unknown unit
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("d").is_err()); // no number
    }
}
