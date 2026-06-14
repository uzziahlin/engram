use crate::config::Config;
use crate::git_integration::GitIntegration;
use crate::models::*;
use crate::storage::MemoryRepository;
use anyhow::{Context, Result};
use std::path::Path;

fn open_repo(config: &Config) -> Result<MemoryRepository> {
    let repo = MemoryRepository::new(&config.storage.database_path)?;
    Ok(repo)
}

fn load_config() -> Result<Config> {
    Config::load()
}

fn now_ts() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Print JSON output to stdout.
fn print_json(value: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
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
    let conn = repo.connection();

    let mut stmt = conn.prepare(
        "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
         FROM episodic_memories
         WHERE project_id = ?1 AND created_at >= ?2
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
