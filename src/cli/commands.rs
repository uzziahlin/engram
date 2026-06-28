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

/// Extract a numeric argument by name.
fn optional_num(args: &[String], name: &str) -> Option<f64> {
    optional_str(args, name).and_then(|s| s.parse().ok())
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
    let config = load_config();
    let config = config.unwrap_or_default();

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

pub fn search(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let query = require_str(args, "query")?;
    let memory_type = optional_str(args, "type");
    let limit = optional_num(args, "limit").unwrap_or(10.0) as usize;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let results = if let Some(ref mt) = memory_type {
        match mt.as_str() {
            "episodic" => {
                let mems = repo.search_episodic(&query, &project_id, limit)?;
                mems.iter().map(|m| serde_json::json!({
                    "id": m.memory.id, "type": "episodic", "summary": m.memory.summary,
                    "importance": m.memory.importance, "files": m.memory.files_touched, "created_at": m.memory.created_at,
                })).collect()
            }
            "decision" => {
                let mems = repo.search_decisions(&query, &project_id, limit)?;
                mems.iter()
                    .map(|m| {
                        serde_json::json!({
                            "id": m.memory.id, "type": "decision", "title": m.memory.title,
                            "rationale": m.memory.rationale, "created_at": m.memory.created_at,
                        })
                    })
                    .collect()
            }
            "failure" => {
                let mems = repo.search_failures(&query, &project_id, limit)?;
                mems.iter().map(|m| serde_json::json!({
                    "id": m.memory.id, "type": "failure", "incident": m.memory.incident,
                    "severity": m.memory.severity, "fix": m.memory.fix, "created_at": m.memory.created_at,
                })).collect()
            }
            "procedural" => {
                let mems = repo.search_procedural(&query, &project_id, limit)?;
                mems.iter().map(|m| serde_json::json!({
                    "id": m.memory.id, "type": "procedural", "workflow": m.memory.workflow_name,
                    "steps": m.memory.steps, "created_at": m.memory.created_at,
                })).collect()
            }
            _ => anyhow::bail!(
                "Unknown memory type: {mt}. Use: episodic, decision, failure, procedural"
            ),
        }
    } else {
        // Search all types
        let mut all = Vec::new();

        if let Ok(mems) = repo.search_episodic(&query, &project_id, limit) {
            for m in &mems {
                all.push(serde_json::json!({"id": m.memory.id, "type": "episodic", "summary": m.memory.summary, "importance": m.memory.importance, "created_at": m.memory.created_at}));
            }
        }
        if let Ok(mems) = repo.search_decisions(&query, &project_id, limit) {
            for m in &mems {
                all.push(serde_json::json!({"id": m.memory.id, "type": "decision", "title": m.memory.title, "created_at": m.memory.created_at}));
            }
        }
        if let Ok(mems) = repo.search_failures(&query, &project_id, limit) {
            for m in &mems {
                all.push(serde_json::json!({"id": m.memory.id, "type": "failure", "incident": m.memory.incident, "severity": m.memory.severity, "created_at": m.memory.created_at}));
            }
        }
        if let Ok(mems) = repo.search_procedural(&query, &project_id, limit) {
            for m in &mems {
                all.push(serde_json::json!({"id": m.memory.id, "type": "procedural", "workflow": m.memory.workflow_name, "created_at": m.memory.created_at}));
            }
        }
        all.truncate(limit);
        all
    };

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
    let count = optional_num(args, "count").unwrap_or(20.0) as usize;
    let session_id = optional_str(args, "session").unwrap_or_else(|| "auto-ingest".into());

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let git = GitIntegration::new(Path::new(&repo_path))?;
    let memories = git.process_recent_commits(&project_id, &session_id, count)?;

    // Deduplicate: skip commits already ingested
    let ingested_hashes = repo.get_ingested_commits(&project_id)?;
    let new_memories: Vec<_> = memories
        .into_iter()
        .filter(|m| {
            m.related_commits
                .iter()
                .all(|c| !ingested_hashes.contains(c))
        })
        .collect();

    let mut ingested = Vec::new();
    for mem in &new_memories {
        repo.create_episodic(mem)?;
        ingested.push(serde_json::json!({
            "id": mem.id,
            "summary": mem.summary,
            "files": mem.files_touched,
        }));
    }

    print_json(&serde_json::json!({
        "ingested": ingested.len(),
        "total_commits": new_memories.len(),
        "memories": ingested,
    }));
    Ok(())
}

pub fn collect(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let repo_path = require_str(args, "repo")?;
    let dimensions = optional_str(args, "dimensions");
    let max_commits = optional_num(args, "max-commits").unwrap_or(200.0) as usize;

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

    let since = now_ts() - (days * 86400);
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

    let since = now_ts() - (days * 86400);
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
                        patch.insert(k.to_string(), serde_json::json!(v));
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
        0.85,
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
