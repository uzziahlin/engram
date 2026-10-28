use crate::config::Config;
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

/// Extract an optional string argument by name. A following flag is not a
/// value: `--session --dry-run` yields None instead of swallowing
/// "--dry-run" as the session id (the same guard `require_str` enforces with
/// an error).
fn optional_str(args: &[String], name: &str) -> Option<String> {
    let flag = format!("--{name}");
    let pos = args.iter().position(|a| a == &flag)?;
    args.get(pos + 1).filter(|v| !v.starts_with("--")).cloned()
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

    // 4. Offer the SessionEnd hook (automatic memory capture) — the same
    // one-command wiring `engram install-hooks` provides.
    let do_hooks = if args.iter().any(|a| a == "--hooks") {
        true
    } else if args.iter().any(|a| a == "--no-hooks") {
        false
    } else if io::stdin().is_terminal() {
        prompt_yes_no("Install the SessionEnd hook (automatic memory capture)?")?
    } else {
        println!(
            "Not a terminal; skipping hook install. Run `engram install-hooks --project {project_id}` to enable automatic memory capture."
        );
        false
    };
    if do_hooks {
        let settings = dir.join(".claude").join("settings.json");
        match install_hooks_impl(&settings, &project_id, false)? {
            "installed" => println!("Installed SessionEnd hook in {}", settings.display()),
            "already" => println!("SessionEnd hook already present in {}", settings.display()),
            _ => unreachable!(),
        }
    }

    println!("Done. Ensure engram is registered (claude mcp add) and restart your editor for the guide to take effect.");
    Ok(())
}

/// ─── Claude Code hook installation ─────────────────────────────────────
///
/// One-command wiring for automatic memory capture: instead of hand-editing
/// `.claude/settings.json`, `engram install-hooks` merges a `SessionEnd`
/// entry (invoking `engram hook`) into the existing settings — preserving
/// any other hooks, backing up the file first, and skipping when engram is
/// already wired (idempotent).
/// Marker identifying engram's own hook entries among all others. The brief
/// command is a separate marker so each event installs/uninstalls cleanly.
const HOOK_COMMAND_MARKER: &str = "engram hook";
const BRIEF_COMMAND_MARKER: &str = "engram brief";

/// Shell-quote a value embedded into a hook command string. project_id
/// defaults to a directory basename, and directory names can contain spaces
/// ("My Project" — which used to silently bind the wrong id: the hook got
/// `--project My` and treated `Project` as a stray argument) or shell
/// metacharacters (`;`, `$()`, backticks — previously executed verbatim by
/// the hook's shell; cloning a maliciously-named directory was enough).
/// Safe strings pass through unchanged; everything else is single-quoted.
fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The `engram hook` command line for a project (shell-quoting applied).
fn hook_command(project_id: &str) -> String {
    format!(
        "{HOOK_COMMAND_MARKER} --project {}",
        shell_quote(project_id)
    )
}

/// The `engram brief` command line for a project (shell-quoting applied).
fn brief_command(project_id: &str) -> String {
    format!(
        "{BRIEF_COMMAND_MARKER} --project {}",
        shell_quote(project_id)
    )
}

/// Whether a hook command string belongs to engram (either wiring).
fn is_engram_command(cmd: &str) -> bool {
    cmd.contains(HOOK_COMMAND_MARKER) || cmd.contains(BRIEF_COMMAND_MARKER)
}

/// Install one hook entry (command string) under `event` in a Claude Code
/// settings file, preserving other entries. Returns "installed", "already",
/// or "dry-run". Idempotent per event: a second engram command under the
/// same event is skipped.
fn install_hook_entry(
    settings_path: &Path,
    event: &str,
    command: &str,
    dry_run: bool,
) -> Result<&'static str> {
    let mut settings = read_settings_object(settings_path)?;

    if settings
        .get("hooks")
        .and_then(|h| h.get(event))
        .is_some_and(|_| settings_has_engram_hook_for_event(&settings, event))
    {
        return Ok("already");
    }

    let entry = serde_json::json!({
        "hooks": [{ "type": "command", "command": command }]
    });
    let obj = settings
        .as_object_mut()
        .context("settings must be an object")?;
    let hooks = obj
        .entry("hooks".to_string())
        .or_insert_with(|| serde_json::json!({}));
    let hook_events = hooks
        .as_object_mut()
        .context("`hooks` in settings.json must be an object")?;
    let list = hook_events
        .entry(event.to_string())
        .or_insert_with(|| serde_json::json!([]));
    list.as_array_mut()
        .with_context(|| format!("`hooks.{event}` in settings.json must be an array"))?
        .push(entry);

    if dry_run {
        return Ok("dry-run");
    }
    backup_then_write(settings_path, &settings)?;
    Ok("installed")
}

/// Install the SessionEnd hook (automatic memory capture via `engram hook`).
/// Returns "installed", "already", or "dry-run".
fn install_hooks_impl(
    settings_path: &Path,
    project_id: &str,
    dry_run: bool,
) -> Result<&'static str> {
    install_hook_entry(
        settings_path,
        "SessionEnd",
        &hook_command(project_id),
        dry_run,
    )
}

/// Install the SessionStart hook (context injection: `engram brief` prints
/// the session brief, which Claude Code adds to the session context).
/// Returns "installed", "already", or "dry-run".
fn install_brief_hook_impl(
    settings_path: &Path,
    project_id: &str,
    dry_run: bool,
) -> Result<&'static str> {
    install_hook_entry(
        settings_path,
        "SessionStart",
        &brief_command(project_id),
        dry_run,
    )
}

/// Whether `event` already carries any engram hook command.
fn settings_has_engram_hook_for_event(settings: &serde_json::Value, event: &str) -> bool {
    settings
        .get("hooks")
        .and_then(|h| h.as_object())
        .and_then(|events| events.get(event))
        .and_then(|groups| groups.as_array())
        .is_some_and(|gs| {
            gs.iter().any(|g| {
                g.get("hooks")
                    .and_then(|h| h.as_array())
                    .is_some_and(|entries| {
                        entries.iter().any(|e| {
                            e.get("command")
                                .and_then(|c| c.as_str())
                                .is_some_and(is_engram_command)
                        })
                    })
            })
        })
}

/// Remove every engram hook entry (any event) from a settings file.
/// Returns how many hook commands were removed.
fn uninstall_hooks_impl(settings_path: &Path) -> Result<usize> {
    if !settings_path.exists() {
        return Ok(0);
    }
    let mut settings = read_settings_object(settings_path)?;
    let mut removed = 0usize;

    if let Some(hooks) = settings.get_mut("hooks").and_then(|h| h.as_object_mut()) {
        for (_event, groups) in hooks.iter_mut() {
            let Some(groups) = groups.as_array_mut() else {
                continue;
            };
            for group in groups.iter_mut() {
                if let Some(entries) = group.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                    let before = entries.len();
                    entries.retain(|e| {
                        !e.get("command")
                            .and_then(|c| c.as_str())
                            .is_some_and(is_engram_command)
                    });
                    removed += before - entries.len();
                }
            }
            // Drop groups that no longer carry any hook command.
            groups.retain(|g| {
                g.get("hooks")
                    .and_then(|h| h.as_array())
                    .is_some_and(|a| !a.is_empty())
            });
        }
        // Drop events with no remaining groups.
        hooks.retain(|_, groups| groups.as_array().is_some_and(|a| !a.is_empty()));
        if settings
            .get("hooks")
            .and_then(|h| h.as_object())
            .is_some_and(|o| o.is_empty())
        {
            settings.as_object_mut().unwrap().remove("hooks");
        }
    }

    if removed > 0 {
        backup_then_write(settings_path, &settings)?;
    }
    Ok(removed)
}

/// Read a settings file as a JSON object; a missing file is an empty object.
/// A malformed file is a hard error — never clobber user config we can't
/// parse.
fn read_settings_object(settings_path: &Path) -> Result<serde_json::Value> {
    if !settings_path.exists() {
        return Ok(serde_json::json!({}));
    }
    let text = std::fs::read_to_string(settings_path)
        .with_context(|| format!("read {}", settings_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).with_context(|| {
        format!(
            "{} is not valid JSON — fix or remove it manually, then re-run \
             (engram will not overwrite a settings file it cannot parse)",
            settings_path.display()
        )
    })?;
    if !value.is_object() {
        anyhow::bail!(
            "{} must contain a JSON object at the top level",
            settings_path.display()
        );
    }
    Ok(value)
}

/// Back up the current file (if any), then write the new settings pretty-
/// printed with a trailing newline.
fn backup_then_write(settings_path: &Path, settings: &serde_json::Value) -> Result<()> {
    if settings_path.exists() {
        let backup = settings_path.with_extension("json.bak");
        std::fs::copy(settings_path, &backup).with_context(|| {
            format!("backup {} -> {}", settings_path.display(), backup.display())
        })?;
    }
    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(settings)?;
    text.push('\n');
    std::fs::write(settings_path, text)
        .with_context(|| format!("write {}", settings_path.display()))?;
    Ok(())
}

/// `engram install-hooks [--global] [--dir <path>] [--project <id>] [--brief]
/// [--dry-run]` — SessionEnd memory capture; `--brief` additionally wires
/// SessionStart to `engram brief` (context injection at session start).
pub fn install_hooks(args: &[String]) -> Result<()> {
    let global = args.iter().any(|a| a == "--global");
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let with_brief = args.iter().any(|a| a == "--brief");

    let (settings_path, project_id) = hooks_target(args, global)?;
    let status = install_hooks_impl(&settings_path, &project_id, dry_run)?;
    let brief_status = if with_brief {
        Some(install_brief_hook_impl(
            &settings_path,
            &project_id,
            dry_run,
        )?)
    } else {
        None
    };
    print_json(&serde_json::json!({
        "settings": settings_path.to_string_lossy(),
        "status": status,
        "command": hook_command(&project_id),
        "event": "SessionEnd",
        "brief": if with_brief {
            serde_json::json!({
                "status": brief_status,
                "command": brief_command(&project_id),
                "event": "SessionStart",
            })
        } else {
            serde_json::Value::Null
        },
    }));
    Ok(())
}

/// `engram uninstall-hooks [--global] [--dir <path>]`
pub fn uninstall_hooks(args: &[String]) -> Result<()> {
    let global = args.iter().any(|a| a == "--global");
    let (settings_path, _project_id) = hooks_target(args, global)?;
    let removed = uninstall_hooks_impl(&settings_path)?;
    print_json(&serde_json::json!({
        "settings": settings_path.to_string_lossy(),
        "removed": removed,
    }));
    Ok(())
}

/// Resolve the settings file path (and the project id for install).
fn hooks_target(args: &[String], global: bool) -> Result<(PathBuf, String)> {
    let project_id = match optional_str(args, "project") {
        Some(p) => p,
        None => {
            let dir = optional_str(args, "dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let abs = std::fs::canonicalize(&dir)
                .with_context(|| format!("failed to resolve {}", dir.display()))?;
            abs.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .context("could not derive project id from directory; pass --project")?
        }
    };
    let settings_path = if global {
        dirs::home_dir()
            .context("could not determine home directory for --global")?
            .join(".claude")
            .join("settings.json")
    } else {
        let dir = optional_str(args, "dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        dir.join(".claude").join("settings.json")
    };
    Ok((settings_path, project_id))
}

pub fn search(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let query = require_str(args, "query")?;
    let memory_type = optional_str(args, "type");
    let tags = repeated_args(args, "tag");
    let before = optional_num(args, "before").map(|n| n as i64);

    let config = load_config()?;
    let repo = open_repo(&config)?;

    // Same pipeline as the MCP search_memory tool — intent → plan → BM25 →
    // (semantic fuse) → rerank. The CLI previously ran a bare-BM25 fork that
    // ranked differently from MCP on the same query and duplicated the field
    // mapping for all four memory types (which had already drifted).
    let limit = optional_num(args, "limit")
        .map(|n| n.clamp(1.0, 1000.0) as usize)
        .unwrap_or(config.retrieval.default_limit);
    let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
    let results = provider.search_core(
        &query,
        &project_id,
        memory_type.as_deref(),
        &tags,
        before,
        limit,
    )?;

    let results_out: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let mut item = serde_json::json!({
                "id": r.id,
                "type": r.memory_type,
                "summary": r.summary,
                "score": (r.relevance_score.clamp(0.0, 1.0) * 100.0).round() / 100.0,
                "importance": (r.importance * 100.0).round() / 100.0,
                "tags": r.tags,
                "created_at": r.created_at,
            });
            if let (Some(obj), Some(detail)) = (item.as_object_mut(), r.detail.as_object()) {
                for (k, v) in detail {
                    obj.insert(k.clone(), v.clone());
                }
            }
            item
        })
        .collect();

    // Best-effort retrieval feedback (mirrors MCP search_memory): log this
    // query + its hits so `engram queries` / MCP `query_stats` can surface
    // hit-rate signal. Graph-expansion neighbors are excluded — they never
    // matched the query, and counting them as impressions poisoned
    // `engram tune`. A logging failure never breaks the command.
    let result_ids = crate::retrieval::bm25::BM25Retriever::impression_ids(&results);
    if let Err(e) = provider.record_query(&project_id, &query, &result_ids, memory_type.as_deref())
    {
        tracing::warn!("query log failed: {e}");
    }

    print_json(&serde_json::json!({
        "results": results_out,
        "total": results_out.len(),
    }));
    Ok(())
}

/// `engram brief --project <id> [--focus <query>] [--json]` — print the
/// session-start brief: key decisions, recent failures, active procedures,
/// pending reflection suggestions (and, with --focus, top search hits).
/// Default output is the plain text brief (what a SessionStart hook should
/// inject); --json returns the structured sections.
pub fn brief(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let focus = optional_str(args, "focus");
    let as_json = args.iter().any(|a| a == "--json");

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
    use crate::mcp::server::MemoryToolProvider as _;
    let value =
        provider.session_brief(crate::mcp::server::SessionBriefInput { project_id, focus })?;
    if as_json {
        print_json(&value);
    } else if let Some(text) = value["brief"].as_str() {
        println!("{text}");
    }
    Ok(())
}

/// The CLI create commands all share one shape: build the typed provider
/// input from flags, run it through `DefaultMemoryProvider` — the SAME path
/// the MCP tools use — and print the provider's JSON result. This keeps the
/// two entry points from drifting (the CLI previously skipped importance
/// validation AND semantic embedding, silently forking behavior).
macro_rules! cli_create {
    ($args:expr, $input_constructor:expr, $method:ident, $input_ty:ty) => {{
        let input: $input_ty = $input_constructor;
        let config = load_config()?;
        let repo = open_repo(&config)?;
        let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
        use crate::mcp::server::MemoryToolProvider as _;
        let value = provider.$method(input)?;
        print_json(&value);
        Ok(())
    }};
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

    cli_create!(
        args,
        crate::mcp::server::CreateEpisodicInput {
            project_id,
            session_id,
            summary,
            content,
            files_touched: files,
            related_commits: commits,
            importance,
            tags,
        },
        create_episodic,
        crate::mcp::server::CreateEpisodicInput
    )
}

pub fn create_decision(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let title = require_str(args, "title")?;
    let context = require_str(args, "context")?;
    let rationale = require_str(args, "rationale")?;
    let tradeoffs = optional_str(args, "tradeoffs").unwrap_or_default();
    let files = comma_list(args, "files");
    let tags = repeated_args(args, "tag");
    let supersedes = optional_str(args, "supersedes");
    let importance = optional_num(args, "importance")
        .map(|v| -> Result<f32> {
            if !(0.0..=1.0).contains(&v) {
                anyhow::bail!("importance must be between 0 and 1, got {v}");
            }
            Ok(v as f32)
        })
        .transpose()?;

    cli_create!(
        args,
        crate::mcp::server::CreateDecisionInput {
            project_id,
            title,
            context,
            rationale,
            tradeoffs,
            related_files: files,
            tags,
            importance,
            supersedes,
        },
        create_decision,
        crate::mcp::server::CreateDecisionInput
    )
}

pub fn create_failure(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let incident = require_str(args, "incident")?;
    let root_cause = require_str(args, "root-cause")?;
    let fix = require_str(args, "fix")?;
    let prevention = require_str(args, "prevention")?;
    let severity = {
        let v = optional_num(args, "severity").unwrap_or(3.0);
        if !(1.0..=5.0).contains(&v) {
            anyhow::bail!("severity must be between 1 and 5, got {v}");
        }
        v as u8
    };
    let tags = repeated_args(args, "tag");

    cli_create!(
        args,
        crate::mcp::server::CreateFailureInput {
            project_id,
            incident,
            root_cause,
            fix,
            prevention,
            severity,
            tags,
        },
        create_failure,
        crate::mcp::server::CreateFailureInput
    )
}

pub fn create_procedural(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let workflow_name = require_str(args, "name")?;
    let steps_str = require_str(args, "steps")?;
    let steps: Vec<String> = steps_str.split(',').map(|s| s.trim().to_string()).collect();
    let tools = comma_list(args, "tools");
    let tags = repeated_args(args, "tag");
    let importance = optional_num(args, "importance")
        .map(|v| -> Result<f32> {
            if !(0.0..=1.0).contains(&v) {
                anyhow::bail!("importance must be between 0 and 1, got {v}");
            }
            Ok(v as f32)
        })
        .transpose()?;

    cli_create!(
        args,
        crate::mcp::server::CreateProceduralInput {
            project_id,
            workflow_name,
            steps,
            related_tools: tools,
            tags,
            importance,
        },
        create_procedural,
        crate::mcp::server::CreateProceduralInput
    )
}

pub fn ingest(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let repo_path = require_str(args, "repo")?;
    let count = (optional_num(args, "count").unwrap_or(20.0) as usize).clamp(1, 1000);
    let session_id = optional_str(args, "session");

    // Route through the provider (same path as the MCP tool): repo_path
    // validation via path_guard, corrected scanned/skipped accounting, and
    // — under the semantic feature — embeddings for the created memories.
    // The previous CLI implementation talked to the repo directly and skipped
    // all three.
    cli_create!(
        args,
        crate::mcp::server::IngestCommitsInput {
            project_id,
            repo_path,
            count,
            session_id,
        },
        ingest_commits,
        crate::mcp::server::IngestCommitsInput
    )
}

pub fn collect(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let repo_path = require_str(args, "repo")?;
    let dimensions = optional_str(args, "dimensions");
    let max_commits = (optional_num(args, "max-commits").unwrap_or(200.0) as usize).clamp(1, 1000);

    // Same path as the MCP collect_sources tool (open_repo in the macro
    // ensures schema, so the bootstrap entry point still works on a fresh
    // database): commit-hash dedup for idempotent re-collection, path_guard
    // + the configured allowed_roots allowlist, and one dimension-parse
    // implementation. The previous CLI-local orchestration (~15 lines)
    // duplicated all of it.
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "repo_path": repo_path,
            "dimensions": dimensions,
            "max_commits": max_commits,
        }))?,
        collect_sources,
        crate::mcp::server::CollectSourcesInput
    )
}

pub fn recent_failures(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let service = optional_str(args, "service");
    let limit = optional_num(args, "limit").map(|n| n as usize);

    // Provider path (same renderer as the MCP recent_failures tool): the
    // output now carries `prevention` and `tags` too — prevention is the
    // core field of a failure record, and the old CLI-local renderer hid it.
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
    use crate::mcp::server::MemoryToolProvider as _;
    let mut value = provider.recent_failures(serde_json::from_value(serde_json::json!({
        "project_id": project_id,
        "service": service,
        "limit": limit,
    }))?)?;
    // Keep the CLI's `total` field (a pure derivation of the result above).
    if let Some(obj) = value.as_object_mut() {
        let total = obj
            .get("failures")
            .and_then(|f| f.as_array())
            .map_or(0, |a| a.len());
        obj.insert("total".into(), serde_json::json!(total));
    }
    print_json(&value);
    Ok(())
}

pub fn decisions(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let topic = optional_str(args, "topic");
    let limit = optional_num(args, "limit").map(|n| n as usize);
    // Current decisions by default — superseded (replaced) ones only with
    // --all, mirroring the MCP architectural_decisions tool.
    let include_superseded = args.iter().any(|a| a == "--all");

    // Provider path (same renderer as the MCP architectural_decisions tool):
    // the output now carries `related_files` and `tags` as well.
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, config);
    use crate::mcp::server::MemoryToolProvider as _;
    let mut value =
        provider.architectural_decisions(serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "topic": topic,
            "limit": limit,
            "include_superseded": include_superseded,
        }))?)?;
    if let Some(obj) = value.as_object_mut() {
        let total = obj
            .get("decisions")
            .and_then(|d| d.as_array())
            .map_or(0, |a| a.len());
        obj.insert("total".into(), serde_json::json!(total));
    }
    print_json(&value);
    Ok(())
}

pub fn timeline(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let days = optional_num(args, "days").unwrap_or(7.0) as i64;

    let config = load_config()?;
    let repo = open_repo(&config)?;

    let since = now_ts() - days.saturating_mul(86400);
    // Same repository accessor the MCP timeline tool uses — the old inline
    // SQL here was a second implementation that could drift.
    let events: Vec<serde_json::Value> = repo
        .timeline(&project_id, since)?
        .into_iter()
        .map(|r| serde_json::json!({"date": r.day, "episodic_count": r.count}))
        .collect();

    print_json(&serde_json::json!({"events": events}));
    Ok(())
}

/// `engram queries` — retrieval feedback: aggregate past search queries by
/// frequency and average hit count. Surfaces which queries are common and
/// which return few results. Output is the MCP `query_stats` rendering
/// (result_count_avg rounded to 2 decimals; days window clamped to 0..=3650
/// — the CLI's old unclamped window could land in the future).
pub fn queries(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let days = optional_num(args, "days").unwrap_or(7.0) as i64;
    let limit = optional_num(args, "limit").map(|n| n as usize);

    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "days": days,
            "limit": limit,
        }))?,
        query_stats,
        crate::mcp::server::QueryStatsInput
    )
}

pub fn forget(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;

    // Provider path (same as forget_memory): batch-3 archived-tombstone
    // semantics plus semantic vector-cache invalidation, which the old
    // direct-repository call silently skipped. Input structs whose fields
    // are private are built via their Deserialize impl — the same parse the
    // MCP dispatch performs on tool arguments.
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "id": id,
        }))?,
        forget_memory,
        crate::mcp::server::ForgetMemoryInput
    )
}

pub fn restore(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;

    // Provider path (same as restore_memory) — includes the semantic
    // vector-cache invalidation the direct-repository call skipped.
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "id": id,
        }))?,
        restore_memory,
        crate::mcp::server::RestoreMemoryInput
    )
}

/// Parse the repeated `--set key=value` pairs of `engram update` into a
/// patch object. Values are parsed as JSON when possible (numbers, bools,
/// arrays) and fall back to plain strings — always-strings made
/// `--set importance=0.9` fail type checking against the typed model.
fn parse_set_pairs(args: &[String]) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut patch = serde_json::Map::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--set" {
            if let Some(kv) = args.get(i + 1) {
                match kv.split_once('=') {
                    Some((k, v)) => {
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
    Ok(patch)
}

pub fn update(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = require_str(args, "type")?;
    let id = require_str(args, "id")?;
    let patch = parse_set_pairs(args)?;

    // Provider path (same merge-patch as the MCP update_memory tool): the
    // CLI's second inline merge-patch implementation is gone, and the
    // command inherits batch-3 semantics — updating an archived memory
    // fails with "restore it first", an edit whose new text collides with
    // another active row's content hash fails naming the conflict, and
    // (semantic builds) the updated text is re-embedded.
    //
    // The input's `patch` is serde-flattened: the --set pairs ride as plain
    // keys next to the addressing triple. Protected keys are dropped here so
    // the flag-derived addressing always wins (the old CLI skipped them in
    // the merge for the same reason).
    let mut input = serde_json::json!({
        "project_id": project_id,
        "memory_type": memory_type,
        "id": id,
    });
    if let Some(obj) = input.as_object_mut() {
        for (k, v) in patch {
            if !matches!(k.as_str(), "project_id" | "memory_type" | "id") {
                obj.insert(k, v);
            }
        }
    }
    cli_create!(
        args,
        serde_json::from_value(input)?,
        update_memory,
        crate::mcp::server::UpdateMemoryInput
    )
}

pub fn forget_batch(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = optional_str(args, "type");
    let tags = repeated_args(args, "tag");
    let before = optional_num(args, "before").map(|n| n as i64);
    let apply = args.iter().any(|a| a == "--apply");

    // Provider path (same as the MCP forget_batch tool): identical dry-run /
    // apply accounting, plus the semantic vector-cache invalidation the
    // direct-repository call skipped on --apply.
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "tags": tags,
            "before": before,
            "apply": apply,
        }))?,
        forget_batch,
        crate::mcp::server::ForgetBatchInput
    )
}

pub fn list_archived(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let memory_type = optional_str(args, "type");
    let limit = optional_num(args, "limit").map(|n| n as usize);

    // Provider path (same as the MCP list_archived tool); the provider
    // clamps the limit into 1..=1000 itself.
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "limit": limit,
        }))?,
        list_archived,
        crate::mcp::server::ListArchivedInput
    )
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
    // Orphan File/Tool entities left behind by deleted memories go with the
    // same pass (counted separately — it is not an archived-memory delete).
    let orphan_entities = repo.gc_orphan_entities(apply)?;
    print_json(&serde_json::json!({
        "applied": report.applied,
        "older_than_seconds": report.older_than_seconds,
        "per_type": report.per_type,
        "deleted": report.deleted,
        "orphan_entities_removed": orphan_entities,
    }));

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

    // Provider path (same engine invocation as the MCP consolidate_memories
    // tool, including the default Jaccard threshold and — on --apply with
    // archives — the semantic vector-cache invalidation the direct call here
    // used to skip).
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "include_near_dup": include_near_dup,
            "apply": apply,
        }))?,
        consolidate_memories,
        crate::mcp::server::ConsolidateInput
    )
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

    // Provider path: identical output shape, and the read now records an
    // adoption event like the MCP tool — CLI reads count toward query_stats
    // adoption rates and `engram tune` (best-effort inside the provider).
    cli_create!(
        args,
        serde_json::from_value(serde_json::json!({
            "project_id": project_id,
            "memory_type": memory_type,
            "id": id,
        }))?,
        get_memory,
        crate::mcp::server::GetMemoryInput
    )
}
/// `engram backup [--out <dir>] [--keep N]` — snapshot the database via the
/// SQLite online-backup API (safe while the MCP server has the DB open).
/// Defaults: `~/.engram/backups`, keep the newest 10.
pub fn backup(args: &[String]) -> Result<()> {
    let keep = (optional_num(args, "keep").unwrap_or(10.0) as usize).max(1);
    let config = load_config()?;
    let db_path = &config.storage.database_path;
    if !db_path.exists() {
        anyhow::bail!(
            "database {} does not exist — nothing to back up",
            db_path.display()
        );
    }

    let out_dir = match optional_str(args, "out") {
        Some(d) => std::path::PathBuf::from(d),
        None => db_path
            .parent()
            .context("database path has no parent dir")?
            .join("backups"),
    };
    std::fs::create_dir_all(&out_dir)?;

    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let dest = out_dir.join(format!("memory-{stamp}.db"));
    {
        let repo = open_repo(&config)?;
        use rusqlite::backup::Backup;
        let src = repo.connection()?;
        let mut dst = rusqlite::Connection::open(&dest)
            .with_context(|| format!("open backup target {}", dest.display()))?;
        let backup = Backup::new(&src, &mut dst)
            .context("init backup (is the source database busy in a transaction?)")?;
        backup.run_to_completion(64, std::time::Duration::from_millis(5), None)?;
    }

    // Retention: keep the newest `keep` memory-*.db backups.
    let mut backups: Vec<_> = std::fs::read_dir(&out_dir)?
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with("memory-") && n.ends_with(".db")
        })
        .collect();
    backups.sort_by_key(|e| e.file_name());
    let mut removed = 0usize;
    while backups.len() > keep {
        let oldest = backups.remove(0);
        std::fs::remove_file(oldest.path()).ok();
        removed += 1;
    }

    print_json(&serde_json::json!({
        "backup": dest.to_string_lossy(),
        "kept": backups.len(),
        "pruned_old": removed,
    }));
    Ok(())
}

/// `engram export [--project <id>] [--out <file>]` — dump all active memories
/// to JSON (stdout by default). Project-scoped with `--project`.
pub fn export(args: &[String]) -> Result<()> {
    let project = optional_str(args, "project");
    let out = optional_str(args, "out");
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let scope = project.as_deref();

    let payload = serde_json::json!({
        "format": "engram-export",
        "version": 1,
        "exported_at": now_ts(),
        "project": scope,
        "episodic": repo.list_active_episodic(scope)?,
        "decision": repo.list_active_decision(scope)?,
        "failure": repo.list_active_failure(scope)?,
        "procedural": repo.list_active_procedural(scope)?,
    });
    let json = serde_json::to_string_pretty(&payload)?;

    match out {
        Some(path) => {
            std::fs::write(&path, json).with_context(|| format!("write export file {path}"))?;
            let count = payload["episodic"].as_array().map_or(0, |a| a.len())
                + payload["decision"].as_array().map_or(0, |a| a.len())
                + payload["failure"].as_array().map_or(0, |a| a.len())
                + payload["procedural"].as_array().map_or(0, |a| a.len());
            print_json(&serde_json::json!({ "exported": count, "out": path }));
        }
        None => println!("{json}"),
    }
    Ok(())
}

/// `engram import --file <export.json>` — load memories from an `engram
/// export` dump. Idempotent by id: memories that already exist (same id AND
/// project) are skipped, so re-importing is safe.
pub fn import(args: &[String]) -> Result<()> {
    let file = require_str(args, "file")?;
    let config = load_config()?;
    let repo = open_repo(&config)?;

    let text =
        std::fs::read_to_string(&file).with_context(|| format!("read import file {file}"))?;
    let payload: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {file} as JSON"))?;
    if payload.get("format").and_then(|f| f.as_str()) != Some("engram-export") {
        anyhow::bail!("not an engram export (missing format: engram-export)");
    }

    let mut created = 0usize;
    let mut skipped = 0usize;

    // Embedding parity with the provider's create path: imported memories
    // were invisible to semantic search until a manual `engram reindex`.
    // No-op when the embedder is absent; compiled out without the feature.
    #[cfg(feature = "semantic")]
    let embedding = crate::mcp::embedding_service::EmbeddingService::new(&config);

    macro_rules! import_kind {
        ($key:literal, $kind:literal, $get:ident, $create:ident, $ty:ty) => {{
            if let Some(items) = payload.get($key).and_then(|v| v.as_array()) {
                for item in items {
                    let mem: $ty = serde_json::from_value(item.clone())
                        .with_context(|| format!("invalid {} entry in export", $key))?;
                    if repo.$get(&mem.id, &mem.project_id)?.is_some() {
                        skipped += 1;
                    } else {
                        #[cfg(feature = "semantic")]
                        embedding.index(
                            &repo,
                            $kind,
                            &mem.id,
                            &mem.project_id,
                            &mem.embedding_text(),
                        );
                        repo.$create(&mem)?;
                        created += 1;
                    }
                }
            }
        }};
    }
    import_kind!(
        "episodic",
        "episodic",
        get_episodic,
        create_episodic,
        crate::models::EpisodicMemory
    );
    import_kind!(
        "decision",
        "decision",
        get_decision,
        create_decision,
        crate::models::DecisionMemory
    );
    import_kind!(
        "failure",
        "failure",
        get_failure,
        create_failure,
        crate::models::FailureMemory
    );
    import_kind!(
        "procedural",
        "procedural",
        get_procedural,
        create_procedural,
        crate::models::ProceduralMemory
    );

    print_json(&serde_json::json!({
        "created": created,
        "skipped_existing": skipped,
    }));
    Ok(())
}

/// `engram stats [--days N]` — one-glance store observability: memory counts
/// per type/project, entity graph size, reflection states, and retrieval
/// feedback (query volume, zero-hit and adoption rates) over the last N days.
pub fn stats(args: &[String]) -> Result<()> {
    let days = (optional_num(args, "days").unwrap_or(30.0) as i64).clamp(1, 3650);
    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();

    let mut snapshot = serde_json::to_value(repo.stats_snapshot()?)?;

    // Retrieval feedback aggregates across all projects.
    let since = now - days * 86_400;
    let projects = repo.list_projects()?;
    let mut total_queries = 0i64;
    let mut total_zero_hit = 0i64;
    let mut total_adopted = 0i64;
    let mut top_adopted: Vec<serde_json::Value> = Vec::new();
    let mut top_zero_hit: Vec<serde_json::Value> = Vec::new();
    for p in &projects {
        for stat in repo.query_stats(p, since, 50)? {
            total_queries += stat.count;
            total_adopted += stat.adopted;
            if stat.result_count_avg < 0.5 {
                total_zero_hit += stat.count;
                if stat.count >= 2 && top_zero_hit.len() < 10 {
                    top_zero_hit.push(serde_json::json!({
                        "project": p, "query": stat.query, "searches": stat.count,
                    }));
                }
            }
            if stat.adopted > 0 && top_adopted.len() < 10 {
                top_adopted.push(serde_json::json!({
                    "project": p, "query": stat.query,
                    "searches": stat.count, "adopted": stat.adopted,
                }));
            }
        }
    }

    let db_size = std::fs::metadata(&config.storage.database_path)
        .map(|m| m.len())
        .unwrap_or(0);

    if let Some(obj) = snapshot.as_object_mut() {
        obj.insert(
            "database".to_string(),
            serde_json::json!({
                "path": config.storage.database_path,
                "size_bytes": db_size,
            }),
        );
        obj.insert(
            "retrieval_feedback".to_string(),
            serde_json::json!({
                "window_days": days,
                "queries": total_queries,
                "zero_hit_queries": total_zero_hit,
                "adopted_results": total_adopted,
                "adoption_rate": if total_queries > 0 {
                    (total_adopted as f64 / total_queries as f64 * 100.0).round() / 100.0
                } else { 0.0 },
                "top_adopted": top_adopted,
                "top_zero_hit": top_zero_hit,
            }),
        );
    }

    print_json(&snapshot);
    Ok(())
}
/// `engram eval [--json]` — retrieval-quality regression harness. Seeds the
/// golden corpus into an IN-MEMORY database (the user's store is never
/// touched) and scores the golden query set through the full search pipeline
/// using the ACTIVE config — so it also measures what custom `[retrieval]`
/// weights do to ranking quality.
pub fn eval(args: &[String]) -> Result<()> {
    let as_json = args.iter().any(|a| a == "--json");
    let config = load_config()?;
    let provider = crate::retrieval::eval::harness(config)?;
    let report = crate::retrieval::eval::run_eval(&provider)?;

    if as_json {
        print_json(&serde_json::to_value(report_summary(&report))?);
        return Ok(());
    }

    println!(
        "Golden retrieval eval — {} cases, top-{} window",
        report.case_count,
        crate::retrieval::eval::EVAL_K
    );
    println!(
        "{:<26} {:<6} {:<38} RANK   RR     NDCG",
        "CASE", "HIT", "QUERY"
    );
    for c in &report.cases {
        println!(
            "{:<26} {:<6} {:<38} {:<6} {:.2}   {:.2}",
            c.case_id,
            if c.primary_rank.is_some() {
                "yes"
            } else {
                "MISS"
            },
            c.query,
            c.primary_rank
                .map(|r| r.to_string())
                .unwrap_or_else(|| "-".into()),
            c.reciprocal_rank,
            c.ndcg,
        );
    }
    println!(
        "\nMRR {:.3} | NDCG@{} {:.3} | primary hit@3 {:.0}%",
        report.mrr,
        crate::retrieval::eval::EVAL_K,
        report.ndcg,
        report.primary_hit_at_3 * 100.0
    );
    println!(
        "Compare against the committed baseline (see retrieval::eval tests) \
         before/after ranking changes."
    );
    Ok(())
}

/// JSON shape for `engram eval --json` (kept separate so the human table and
/// the machine output can evolve independently).
fn report_summary(report: &crate::retrieval::eval::EvalReport) -> serde_json::Value {
    serde_json::json!({
        "case_count": report.case_count,
        "mrr": (report.mrr * 1000.0).round() / 1000.0,
        "ndcg_at_5": (report.ndcg * 1000.0).round() / 1000.0,
        "primary_hit_at_3": (report.primary_hit_at_3 * 1000.0).round() / 1000.0,
        "cases": report.cases.iter().map(|c| serde_json::json!({
            "case_id": c.case_id,
            "query": c.query,
            "expected": c.expected,
            "ranked": c.ranked,
            "primary_rank": c.primary_rank,
            "reciprocal_rank": (c.reciprocal_rank * 1000.0).round() / 1000.0,
            "ndcg": (c.ndcg * 1000.0).round() / 1000.0,
        })).collect::<Vec<_>>(),
    })
}

/// `engram tune --project <id> [--days N] [--apply]` — close the adoption
/// feedback loop automatically: memories that keep getting adopted move up in
/// importance, memories that keep getting served and ignored move down
/// (never memories with explicit 'useful' feedback in the window).
/// Dry-run by default (prints the evidence-backed proposals); --apply writes
/// bounded ±0.1 adjustments via the same path `mark_relevance` uses. Failure
/// memories are never auto-tuned — severity is an operational record — and
/// are reported via `failures_skipped` for manual review.
pub fn tune(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let days = (optional_num(args, "days").unwrap_or(30.0) as i64).clamp(1, 3650);
    let apply = args.iter().any(|a| a == "--apply");

    let config = load_config()?;
    let repo = open_repo(&config)?;
    let now = now_ts();

    let mut report = crate::retrieval::tuning::compute_proposals(&repo, &project_id, days, now)?;
    let applied = if apply {
        crate::retrieval::tuning::apply_proposals(&repo, &project_id, &report.proposals, now)?
    } else {
        0
    };
    report.applied = apply;

    print_json(&serde_json::json!({
        "applied": apply,
        "applied_count": applied,
        "window_days": days,
        "window_queries": report.window_queries,
        "failures_skipped": report.failures_skipped,
        "failures_note": if report.failures_skipped > 0 {
            format!(
                "{} failure memories skipped: severity is an operational record of \
                 incident impact, not auto-tuned — review them manually",
                report.failures_skipped
            )
        } else {
            String::new()
        },
        "proposal_count": report.proposals.len(),
        "proposals": report.proposals,
        "hint": if !apply && !report.proposals.is_empty() {
            "dry-run: re-run with --apply to write these adjustments"
        } else {
            ""
        },
    }));
    Ok(())
}

/// `engram maintain [--project <id>] [--repo <path>] [--apply]` — one-shot
/// memory health pass for hook/cron: consolidate duplicates, repair FTS,
/// prune query_log, staleness report (--repo), gc/orphan-entity preview, and
/// a knowledge-gap report (frequent zero-hit queries).
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

    // 2. FTS repair — gated by an actual drift check (main-table row count
    // vs FTS row count per table). The old unconditional rebuild made even a
    // dry-run maintain do a full delete+reinsert of every FTS table (a long
    // write transaction on large stores) on every run.
    let fts_drift = repo.fts_drift()?;
    let fts_rows = if fts_drift == 0 {
        // Aligned — verify the index is also internally consistent before
        // skipping (a corrupt index needs the rebuild even at equal counts).
        match repo.fts_integrity_check() {
            Ok(()) => 0,
            Err(e) => {
                tracing::warn!("FTS integrity check failed ({e}); rebuilding");
                repo.rebuild_fts()?
            }
        }
    } else if apply {
        repo.rebuild_fts()?
    } else {
        0 // dry-run: report the drift, write nothing
    };

    // 3. query_log retention prune.
    let retention_secs = (config
        .storage
        .query_log_retention_days
        .saturating_mul(86_400)) as i64;
    let pruned = if apply {
        repo.prune_query_log(retention_secs, now)?
    } else {
        0
    };

    // 3b. Reflection pass (apply only): recurring-failure tags become pending
    // preventive-rule proposals. Without this, `reflect` had no trigger —
    // suggestion pools stayed empty unless someone remembered the command.
    let mut reflected = 0usize;
    let mut suggestions_created = 0usize;
    if apply {
        let engine = crate::reflection::ReflectionEngine::with_min_occurrences(
            config.reflection.min_occurrences,
        );
        for p in &projects {
            let plan = engine.reflect(&repo, p, true, now)?;
            reflected += plan.suggestions.len();
            suggestions_created += plan.created;
        }
    }

    // 3c. Raw session-import aging: imports older than SESSION_IMPORT_STALE_DAYS
    // that were never distilled lose ranking weight (distillation into typed
    // memories — via the `engram.distill` prompt — is the promotion path).
    const SESSION_IMPORT_STALE_DAYS: i64 = 90;
    const SESSION_IMPORT_FLOOR: f32 = 0.2;
    let mut aged_imports = 0usize;
    if apply {
        for p in &projects {
            aged_imports += repo
                .downgrade_stale_session_imports(
                    p,
                    SESSION_IMPORT_STALE_DAYS * 86_400,
                    SESSION_IMPORT_FLOOR,
                    true,
                    now,
                )?
                .len();
        }
    }

    // 3d. Distillation backlog: how many raw session-import memories await
    // the `engram.distill` prompt — the maintain-side trigger for the loop
    // session_brief surfaces on the read side.
    let mut distill_backlog = 0i64;
    for p in &projects {
        distill_backlog += repo.count_session_imports(p)?;
    }

    // 4. Staleness report (read-only).
    let stale = match &repo_path {
        Some(rp) => report_stale_memories(&repo, &projects, Path::new(rp))?,
        None => serde_json::json!({ "skipped": "pass --repo <path> to check file staleness" }),
    };

    // 5. GC preview (dry-run only; physical delete stays explicit).
    let gc_report = repo.gc_archived(0, false, now)?;
    let orphan_entities = repo.gc_orphan_entities(false)?;

    // 6. Knowledge gaps (feedback loop): frequent queries that consistently
    // return zero results are the cheapest signal for what to write next —
    // surfaced here instead of rotting inside query_log.
    let gap_window = 30 * 86_400;
    let mut knowledge_gaps: Vec<serde_json::Value> = Vec::new();
    for p in &projects {
        for stat in repo.query_stats(p, now - gap_window, 20)? {
            if stat.result_count_avg < 0.5 && stat.count >= 2 {
                knowledge_gaps.push(serde_json::json!({
                    "project": p,
                    "query": stat.query,
                    "searches": stat.count,
                    "avg_hits": stat.result_count_avg,
                }));
            }
        }
    }
    knowledge_gaps.truncate(20);

    print_json(&serde_json::json!({
        "applied": apply,
        "projects": projects,
        "consolidate": {
            "duplicate_groups": dup_groups,
            "duplicates": dup_duplicates,
            "action": if apply { "archived (restore with `engram restore`)" } else { "dry-run — pass --apply to archive" },
        },
        "fts": {
            "drifted_rows": fts_drift,
            "rebuilt_rows": fts_rows,
            "note": if fts_drift == 0 { "index aligned; no rebuild needed" } else if apply { "rebuilt" } else { "dry-run: drift reported, nothing written" },
        },
        "reflection": {
            "patterns_found": reflected,
            "suggestions_created": suggestions_created,
            "note": if apply { "run `engram suggestions` to review pending proposals" } else { "dry-run: reflection only runs with --apply" },
        },
        "session_import_aging": {
            "downgraded": aged_imports,
            "older_than_days": SESSION_IMPORT_STALE_DAYS,
            "floor_importance": SESSION_IMPORT_FLOOR,
        },
        "distill_backlog": {
            "raw_session_imports": distill_backlog,
            "note": "refine them via the `engram.distill` MCP prompt (see session_brief)",
        },
        "query_log_pruned": pruned,
        "stale": stale,
        "gc_preview": {
            "archived_eligible_for_gc": gc_report.deleted.len(),
            "orphan_entities": orphan_entities,
            "note": "run `engram gc --older-than <dur> --apply` to physically purge",
        },
        "knowledge_gaps": if knowledge_gaps.is_empty() {
            serde_json::json!({ "note": "no repeated zero-hit queries in the last 30 days" })
        } else {
            serde_json::json!({
                "note": "frequent queries with no results — write a memory covering these",
                "gaps": knowledge_gaps,
            })
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
/// [--dry-run]` — distill a Claude Code or Codex CLI session transcript
/// (JSONL; format sniffed per line) into one episodic memory. Meant to be
/// wired to a SessionEnd/Stop hook so memories form without depending on the
/// agent remembering to write them.
pub fn session_import(args: &[String]) -> Result<()> {
    let project_id = require_str(args, "project")?;
    let transcript = require_str(args, "transcript")?;
    let session_id =
        optional_str(args, "session").unwrap_or_else(|| session_id_from_path(&transcript));
    let dry_run = args.iter().any(|a| a == "--dry-run");

    // Streaming bounded read: Claude Code transcripts can reach hundreds of
    // MB; the distiller only needs prompts/edits/conclusions. 64MB total =
    // 8MB head (session meta, early prompts) + tail-first remainder — the
    // final conclusions and latest edits live at the END of a transcript,
    // which a head-only cut dropped exactly when it mattered.
    const TRANSCRIPT_BUDGET_BYTES: usize = 64 * 1024 * 1024;
    const TRANSCRIPT_HEAD_BYTES: usize = 8 * 1024 * 1024;

    let path = Path::new(&transcript);
    let text = read_text_bounded(path, TRANSCRIPT_BUDGET_BYTES, TRANSCRIPT_HEAD_BYTES)
        .map_err(|e| HookSkip::TranscriptUnreadable(format!("{}: {e}", path.display())))?;
    let digest = parse_transcript(&text);

    if digest.user_prompts.is_empty() && digest.files.is_empty() && digest.errors.is_empty() {
        anyhow::bail!(HookSkip::NoSignal);
    }

    let now = now_ts();
    let first_prompt = digest
        .user_prompts
        .first()
        .map(|p| p.replace('\n', " "))
        .unwrap_or_else(|| format!("session {session_id}"));
    let summary: String = first_prompt.chars().take(160).collect();
    let content = session_import_content(&digest);

    let memory = EpisodicMemory {
        // Deterministic id per (project, session): a Stop hook and a
        // SessionEnd hook firing concurrently used to race find→create and
        // write TWO memories for one session. With a deterministic id the
        // primary key itself arbitrates — the loser's INSERT fails cleanly
        // and falls back to the refresh path (see upsert_session_memory).
        // The session lookup uses the session_id column, not this id, so
        // hasher stability across compiler versions does not matter.
        id: session_memory_id(&project_id, &session_id),
        project_id: project_id.clone(),
        session_id: session_id.clone(),
        summary,
        content,
        files_touched: digest.files.iter().take(50).cloned().collect(),
        related_commits: vec![],
        importance: 0.5,
        tags: if digest.errors.is_empty() {
            vec!["session-import".into()]
        } else {
            // has-errors lets bootstrap/reflection find sessions carrying
            // failure evidence without reading every session memory.
            vec!["session-import".into(), "has-errors".into()]
        },
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
    let action = upsert_session_memory(&repo, &memory, now)?;
    // Embedding parity with the provider's create/update paths: hook-imported
    // session memories used to be invisible to semantic search until a
    // manual `engram reindex`. Direct EmbeddingService use (not a full
    // provider) — a no-op without an embedder, compiled out entirely without
    // the feature. Re-indexing the (possibly pre-existing) id keeps the
    // vector fresh on the refresh path too.
    #[cfg(feature = "semantic")]
    {
        let embedding = crate::mcp::embedding_service::EmbeddingService::new(&config);
        embedding.index(
            &repo,
            "episodic",
            &action.id,
            &project_id,
            &memory.embedding_text(),
        );
    }
    print_json(&serde_json::json!({
        "id": action.id,
        "status": action.status, // "created" | "updated"
        "session_id": session_id,
        "files": action.files,
        "created_at": now,
    }));
    Ok(())
}

/// Build the session-import episodic `content` from a transcript digest:
/// bounded prompts, conclusions, files, and error candidates under fixed
/// template headers. The headers come from the consolidation engine's
/// session-import template constants — near-dup comparison EXCLUDES those
/// scaffold words (two templated imports from different sessions must not
/// look like duplicates on scaffold alone), and that exclusion only stays
/// correct while this builder and the exclusion list share one source.
fn session_import_content(digest: &SessionDigest) -> String {
    use crate::consolidation::engine::{
        SESSION_IMPORT_TPL_CONCLUSIONS, SESSION_IMPORT_TPL_ERRORS,
        SESSION_IMPORT_TPL_FAILURE_CANDIDATES, SESSION_IMPORT_TPL_FILES,
        SESSION_IMPORT_TPL_PROMPTS,
    };
    let mut content = String::new();
    content.push_str(&format!("{SESSION_IMPORT_TPL_PROMPTS}:\n"));
    for p in digest.user_prompts.iter().take(10) {
        let line: String = p.chars().take(400).collect();
        content.push_str(&format!("- {line}\n"));
    }
    content.push_str(&format!("\n{SESSION_IMPORT_TPL_CONCLUSIONS}:\n"));
    for a in digest.assistant_texts.iter().rev().take(5).rev() {
        let line: String = a.chars().take(600).collect();
        content.push_str(&format!("- {line}\n"));
    }
    if !digest.files.is_empty() {
        content.push_str(&format!(
            "\n{} ({}):\n",
            SESSION_IMPORT_TPL_FILES,
            digest.files.len()
        ));
        content.push_str(
            &digest
                .files
                .iter()
                .take(50)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    if !digest.errors.is_empty() {
        content.push_str(&format!(
            "\n{} ({} — {}):\n",
            SESSION_IMPORT_TPL_ERRORS,
            digest.errors.len(),
            SESSION_IMPORT_TPL_FAILURE_CANDIDATES,
        ));
        for e in digest.errors.iter().take(10) {
            content.push_str(&format!("- {e}\n"));
        }
    }
    content.chars().take(4000).collect()
}

/// Normal, non-error hook outcomes: surfaced by `engram hook` as
/// `status: "skipped"` (exit 0) instead of a failing exit that Claude Code
/// would show the user on every idle Stop. TYPED — the previous
/// string-matching on error messages silently changed hook behavior any time
/// a message was reworded.
#[derive(Debug, thiserror::Error)]
pub(crate) enum HookSkip {
    #[error(
        "transcript yielded no usable signal (no user prompts, no file edits); \
             not writing an empty memory"
    )]
    NoSignal,
    #[error("failed to read transcript {0}")]
    TranscriptUnreadable(String),
}

/// Stable per-(project, session) memory id for session imports. See the
/// comment at the construction site for why determinism matters.
fn session_memory_id(project_id: &str, session_id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    project_id.hash(&mut h);
    '\n'.hash(&mut h);
    session_id.hash(&mut h);
    format!("sess-{:016x}", h.finish())
}

/// Idempotent session-import write: if an active episodic already exists for
/// this (project, session), REFRESH it instead of creating a near-duplicate.
///
/// Hook wiring fires on every `Stop` (each agent turn), and the transcript
/// keeps growing — without this, one session produced N overlapping memories.
/// Upsert semantics make both `Stop` (memory tracks the session live) and
/// `SessionEnd` (one final import) converge to exactly one memory.
struct SessionUpsert {
    id: String,
    status: &'static str,
    files: usize,
}

fn upsert_session_memory(
    repo: &crate::storage::MemoryRepository,
    memory: &EpisodicMemory,
    now: i64,
) -> Result<SessionUpsert> {
    if let Some(existing) = repo.find_episodic_by_session(&memory.project_id, &memory.session_id)? {
        return refresh_session_memory(repo, existing, memory, now);
    }
    match repo.create_episodic(memory) {
        Ok(()) => Ok(SessionUpsert {
            id: memory.id.clone(),
            status: "created",
            files: memory.files_touched.len(),
        }),
        Err(e) if e.to_string().contains("UNIQUE constraint failed") => {
            // Lost a create race with a concurrent hook on the same session
            // (deterministic id collided on the primary key). The winner's
            // row is now findable — refresh it instead of erroring out.
            let existing = repo
                .find_episodic_by_session(&memory.project_id, &memory.session_id)?
                .ok_or(e)?;
            refresh_session_memory(repo, existing, memory, now)
        }
        Err(e) => Err(e),
    }
}

/// Refresh an existing session memory in place with the latest digest.
fn refresh_session_memory(
    repo: &crate::storage::MemoryRepository,
    mut existing: EpisodicMemory,
    memory: &EpisodicMemory,
    now: i64,
) -> Result<SessionUpsert> {
    existing.summary = memory.summary.clone();
    existing.content = memory.content.clone();
    existing.files_touched = memory.files_touched.clone();
    // Union tags: keep anything added after the first import (user
    // curation), but don't re-add a marker tag the user removed.
    let mut tags = existing.tags.clone();
    for t in &memory.tags {
        if !tags.contains(t) {
            tags.push(t.clone());
        }
    }
    existing.tags = tags;
    existing.updated_at = now;
    repo.update_episodic(&existing)?;
    Ok(SessionUpsert {
        id: existing.id,
        status: "updated",
        files: existing.files_touched.len(),
    })
}

/// Read a text file bounded by `max_bytes` total (lossy UTF-8), split as
/// head + tail: the first `head_bytes`, and — only when the file exceeds the
/// budget — the LAST `max_bytes - head_bytes` bytes, with a marker naming how
/// much was skipped. Tail-priority is deliberate: session transcripts carry
/// their meta at the top and their FINAL conclusions at the bottom, so a
/// head-first cut threw away exactly the late signal the distiller wants.
/// Streams — never slurps the whole file first. The tail may begin mid-line
/// or mid-character (lossy decoding mends the latter); `parse_transcript`
/// skips the one malformed line that can result.
fn read_text_bounded(path: &Path, max_bytes: usize, head_bytes: usize) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let head_bytes = head_bytes.min(max_bytes);

    if file_len <= max_bytes {
        // Whole file fits: one bounded streaming read.
        let mut all = Vec::with_capacity(file_len.min(max_bytes) + 1);
        (&file).take(max_bytes as u64 + 1).read_to_end(&mut all)?;
        return Ok(String::from_utf8_lossy(&all).into_owned());
    }

    let mut head = Vec::with_capacity(head_bytes);
    (&file).take(head_bytes as u64).read_to_end(&mut head)?;

    let tail_budget = max_bytes - head_bytes;
    let tail_start = file_len - tail_budget; // > head_bytes whenever file_len > max_bytes
    let mut tail = Vec::with_capacity(tail_budget);
    file.seek(SeekFrom::Start(tail_start as u64))?;
    (&file).take(tail_budget as u64).read_to_end(&mut tail)?;

    let skipped = file_len - head.len() - tail.len();
    Ok(format!(
        "{}\n…[{skipped} bytes skipped — transcript truncated, first {} and last {} \
         bytes kept (latest conclusions preserved)]…\n{}",
        String::from_utf8_lossy(&head),
        head.len(),
        tail.len(),
        String::from_utf8_lossy(&tail),
    ))
}

/// Best-effort text of a tool_result content block (string or array of text
/// parts), flattened and bounded.
fn tool_result_text(item: &serde_json::Value) -> String {
    let raw = match item.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return String::new(),
    };
    let flat = raw.replace('\n', " ");
    flat.trim().chars().take(300).collect()
}

fn session_id_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "session".into())
}

/// `engram hook [--project <id>] [--dry-run]` — Claude Code hook entry point.
///
/// Reads a hook payload from stdin (the JSON Claude Code pipes to Stop /
/// SessionEnd hooks: `{"transcript_path": …, "session_id": …, "cwd": …}`),
/// distills the transcript into an episodic memory via the session-import
/// path, and exits 0. A transcript with no usable signal is reported as
/// `skipped` (not an error) — empty sessions are normal.
///
/// Wire it in `~/.claude/settings.json` (or a project's `.claude/settings.json`):
///
/// ```json
/// {
///   "hooks": {
///     "Stop": [{ "hooks": [{ "type": "command",
///       "command": "engram hook --project myproj" }] }]
///   }
/// }
/// ```
///
/// `--project` defaults to the payload `cwd`'s directory name, matching
/// `init-guide`'s project-id convention.
pub fn hook(args: &[String]) -> Result<()> {
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let project_override = optional_str(args, "project");
    let transcript_override = optional_str(args, "transcript");

    // Hook payloads come on stdin. Missing/invalid stdin is a configuration
    // error worth surfacing loudly.
    let mut stdin = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut stdin)?;
    let payload: serde_json::Value = serde_json::from_str(stdin.trim()).with_context(|| {
        "hook: expected a JSON payload on stdin (are you running this from a Claude Code hook?)"
    })?;

    let transcript = transcript_override
        .or_else(|| {
            payload
                .get("transcript_path")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .ok_or_else(|| {
            anyhow::anyhow!("hook: no transcript path (pass --transcript or wire the hook so Claude Code provides transcript_path)")
        })?;

    let project_id = project_override
        .or_else(|| {
            payload.get("cwd").and_then(|v| v.as_str()).and_then(|cwd| {
                std::path::Path::new(cwd)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(String::from)
            })
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "hook: no project id (pass --project <id> or ensure cwd is in the hook payload)"
            )
        })?;

    let mut import_args: Vec<String> = vec![
        "--project".into(),
        project_id.clone(),
        "--transcript".into(),
        transcript.clone(),
    ];
    if let Some(sid) = payload.get("session_id").and_then(|v| v.as_str()) {
        import_args.push("--session".into());
        import_args.push(sid.to_string());
    }
    if dry_run {
        import_args.push("--dry-run".into());
    }

    // Unchanged-transcript skip: `Stop` fires on EVERY agent turn, and the
    // transcript is byte-identical between most turns — the old path re-read
    // and re-parsed up to 64 MB each time. A (mtime, size) fingerprint makes
    // the repeat work O(1); we still verify the session memory exists so a
    // manually deleted memory is re-created from the unchanged transcript.
    let unchanged = if dry_run {
        false
    } else {
        match transcript_unchanged(&transcript, &project_id, &import_args) {
            Ok(v) => v,
            // The fingerprint check maps a missing transcript to the same
            // typed skip session_import uses — handled once below.
            Err(e) if e.downcast_ref::<HookSkip>().is_some() => {
                print_json(&serde_json::json!({
                    "status": "skipped",
                    "reason": e.to_string(),
                }));
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    };
    if unchanged {
        print_json(&serde_json::json!({
            "status": "skipped",
            "reason": "transcript unchanged since last import",
        }));
        return Ok(());
    }

    // Normal empty-session outcomes are typed (HookSkip) so rewording a
    // message can never silently flip a hook from skip to error — the hook
    // must not spam Claude Code with errors every idle Stop.
    if let Err(e) = session_import(&import_args) {
        if e.downcast_ref::<HookSkip>().is_some() {
            print_json(&serde_json::json!({
                "status": "skipped",
                "reason": e.to_string(),
            }));
            return Ok(());
        }
        return Err(e);
    }
    Ok(())
}

/// Whether the transcript's (mtime, size) fingerprint matches the last
/// import AND the session memory still exists. State lives beside the
/// database in `hook-state.json` (bounded to the newest 256 transcripts).
fn transcript_unchanged(
    transcript: &str,
    project_id: &str,
    import_args: &[String],
) -> Result<bool> {
    let md = std::fs::metadata(transcript)
        .map_err(|e| HookSkip::TranscriptUnreadable(format!("{transcript}: {e}")))?;
    let fingerprint = format!(
        "{}:{}",
        md.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0),
        md.len()
    );

    let config = load_config()?;
    let state_path = config
        .storage
        .database_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("hook-state.json");
    let mut state: serde_json::Map<String, serde_json::Value> =
        std::fs::read_to_string(&state_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

    let unchanged = state.get(transcript).and_then(|v| v.as_str()) == Some(&fingerprint) && {
        // Only skip when the memory actually exists — deleting the
        // session memory must not be defeated by the fingerprint cache.
        let repo = open_repo(&config)?;
        let session_id: String = import_args
            .iter()
            .position(|a| a == "--session")
            .and_then(|i| import_args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| session_id_from_path(transcript));
        repo.find_episodic_by_session(project_id, &session_id)?
            .is_some()
    };

    if !unchanged && !dry_run_import(import_args) {
        // Record the fingerprint for next time (best-effort).
        state.insert(transcript.to_string(), serde_json::json!(fingerprint));
        while state.len() > 256 {
            // Drop the oldest-inserted entries (serde_json maps preserve
            // insertion order).
            let first = state.iter().next().map(|(k, _)| k.clone());
            match first {
                Some(k) => {
                    state.remove(&k);
                }
                None => break,
            }
        }
        let _ = std::fs::write(&state_path, serde_json::to_string(&state)?);
    }
    Ok(unchanged)
}

fn dry_run_import(import_args: &[String]) -> bool {
    import_args.iter().any(|a| a == "--dry-run")
}

/// What `parse_transcript` extracted from a session JSONL.
#[derive(Debug, Default)]
struct SessionDigest {
    user_prompts: Vec<String>,
    assistant_texts: Vec<String>,
    files: Vec<String>,
    /// Error payloads from tool results — the raw material for failure
    /// memories and the reflection engine (previously transcripts carried
    /// this signal but it was discarded).
    errors: Vec<String>,
}

/// Parse a Claude Code or Codex CLI session transcript (JSONL, one object
/// per line). Format is sniffed per line: Claude Code lines carry `message`
/// (role/content), Codex rollout lines carry `payload` (Responses-API items).
/// Tolerant of schema drift: unknown line shapes are skipped, never fatal.
/// Tool-result payloads are ignored (noise); only real user prompts, visible
/// assistant text, and file-edit targets are kept.
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
            // Codex CLI rollout shape: {"type": "...", "payload": {...}}.
            if v.get("payload").is_some() {
                parse_codex_line(&v, &mut digest);
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
                        // Tool results flagged as errors become failure
                        // evidence (bounded; dedup like every other signal).
                        if item.get("type").and_then(|t| t.as_str()) == Some("tool_result")
                            && item
                                .get("is_error")
                                .and_then(|e| e.as_bool())
                                .unwrap_or(false)
                        {
                            let text = tool_result_text(item);
                            if !text.is_empty() && digest.errors.len() < 20 {
                                push_unique(&mut digest.errors, &text);
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

/// Codex CLI rollout line: `{"timestamp": …, "type": …, "payload": {…}}` where
/// a `response_item` payload is a Responses-API item (`message`,
/// `function_call`, `function_call_output`, …). Files come from `apply_patch`
/// arguments (`*** Update File: <path>` markers); error evidence from failed
/// tool outputs. Heuristic and bounded, like the Claude path — a malformed
/// payload is skipped, never fatal.
fn parse_codex_line(v: &serde_json::Value, digest: &mut SessionDigest) {
    let Some(payload) = v.get("payload") else {
        return;
    };
    match payload.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "message" => {
            let role = payload.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let text = codex_content_text(payload.get("content"));
            let t = text.trim();
            if t.is_empty() || t.starts_with('<') {
                // Codex injects <user_instructions>/<environment_context>
                // blocks as user input_text — same wrapper filter as Claude.
                return;
            }
            match role {
                "user" => push_unique(&mut digest.user_prompts, t),
                "assistant" => push_unique(&mut digest.assistant_texts, t),
                _ => {}
            }
        }
        "function_call" => {
            // apply_patch arguments embed "*** (Update|Add|Delete) File: path"
            // markers — the reliable file-touch signal in Codex transcripts.
            let args = payload
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            for marker in ["*** Update File: ", "*** Add File: ", "*** Delete File: "] {
                let mut rest = args;
                while let Some(pos) = rest.find(marker) {
                    let after = &rest[pos + marker.len()..];
                    // Path runs to end of line. Also cut at a literal "\n"
                    // two-char sequence — arguments are a JSON string, and
                    // some emitters leave the newline escaped.
                    let line_end = after
                        .lines()
                        .next()
                        .unwrap_or("")
                        .split("\\n")
                        .next()
                        .unwrap_or("");
                    let path: String = line_end.trim().chars().take(400).collect();
                    if !path.is_empty() {
                        push_unique(&mut digest.files, &path);
                    }
                    rest = after;
                }
            }
        }
        "function_call_output" => {
            if digest.errors.len() >= 20 {
                return;
            }
            // Codex marks some outputs success=false; older/newer variants
            // only carry the text — fall back to common error markers.
            let failed = payload.get("success") == Some(&serde_json::json!(false));
            let output = payload
                .get("output")
                .and_then(|o| o.as_str())
                .unwrap_or("")
                .replace('\n', " ");
            let output = output.trim();
            let marked = ["error:", "Error:", "ERROR:", "panic:", "failed:"]
                .iter()
                .any(|m| output.starts_with(m));
            if (failed || marked) && !output.is_empty() {
                let bounded: String = output.chars().take(300).collect();
                push_unique(&mut digest.errors, &bounded);
            }
        }
        _ => {} // session_meta / event_msg / reasoning / … — no digest signal
    }
}

/// Flatten a Responses-API `content` value (string, or array of
/// `{type: input_text|output_text, text}` items) into one text blob.
fn codex_content_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|i| i.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
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
    fn install_hooks_creates_settings_with_sessionend_entry() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join(".claude/settings.json");
        let status = install_hooks_impl(&settings, "demo", false).unwrap();
        assert_eq!(status, "installed");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        let cmd = v["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert_eq!(cmd, "engram hook --project demo");
        // No backup when the file didn't exist before.
        assert!(!dir.path().join(".claude/settings.json.bak").exists());
    }

    #[test]
    fn install_hooks_preserves_existing_hooks_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join(".claude/settings.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        // A user's existing PostToolUse formatter hook must survive the merge.
        std::fs::write(
            &settings,
            r#"{"hooks":{"PostToolUse":[{"matcher":"Edit","hooks":[{"type":"command","command":"rustfmt --emit files"}]}]}}"#,
        )
        .unwrap();

        assert_eq!(
            install_hooks_impl(&settings, "demo", false).unwrap(),
            "installed"
        );
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        assert_eq!(
            v["hooks"]["PostToolUse"][0]["hooks"][0]["command"], "rustfmt --emit files",
            "existing hooks must be preserved"
        );
        assert_eq!(v["hooks"]["SessionEnd"].as_array().unwrap().len(), 1);
        // A backup of the pre-install file was made.
        assert!(dir.path().join(".claude/settings.json.bak").exists());

        // Second install is a no-op.
        assert_eq!(
            install_hooks_impl(&settings, "demo", false).unwrap(),
            "already"
        );
        let v2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        assert_eq!(v2, v, "idempotent install must not change the file");
    }

    #[test]
    fn install_hooks_refuses_malformed_settings() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join(".claude/settings.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(&settings, "not json").unwrap();
        let err = install_hooks_impl(&settings, "demo", false).unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
        assert_eq!(
            std::fs::read_to_string(&settings).unwrap(),
            "not json",
            "must not clobber a file it cannot parse"
        );
    }

    #[test]
    fn uninstall_hooks_removes_only_engram_entries() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join(".claude/settings.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(
            &settings,
            r#"{"hooks":{
                "SessionEnd":[
                    {"hooks":[{"type":"command","command":"engram hook --project demo"}]},
                    {"hooks":[{"type":"command","command":"other-tool session-end"}]}
                ],
                "PostToolUse":[{"matcher":"Edit","hooks":[{"type":"command","command":"rustfmt --emit files"}]}]
            }}"#,
        )
        .unwrap();

        let removed = uninstall_hooks_impl(&settings).unwrap();
        assert_eq!(removed, 1);
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
        // engram's entry is gone; the unrelated SessionEnd entry and the
        // PostToolUse group survive.
        let session_cmds: Vec<_> = v["hooks"]["SessionEnd"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|g| g["hooks"].as_array().unwrap().iter())
            .map(|e| e["command"].as_str().unwrap())
            .collect();
        assert_eq!(session_cmds, vec!["other-tool session-end"]);
        assert_eq!(
            v["hooks"]["PostToolUse"][0]["hooks"][0]["command"],
            "rustfmt --emit files"
        );

        // Uninstalling again is a clean no-op.
        assert_eq!(uninstall_hooks_impl(&settings).unwrap(), 0);
    }

    #[test]
    fn init_guide_hooks_flag_installs_sessionend_hook() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = no_import_args(dir.path());
        args.push("--hooks".to_string());
        init_guide(&args).unwrap();
        let settings = std::fs::read_to_string(dir.path().join(".claude/settings.json")).unwrap();
        assert!(settings.contains("engram hook --project demo"));
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
        assert_eq!(
            d.user_prompts,
            vec!["fix the auth middleware bug".to_string()]
        );
        assert_eq!(d.assistant_texts.len(), 2);
        assert!(d
            .assistant_texts
            .contains(&"Root cause was a stale token.".to_string()));
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
    fn parse_transcript_extracts_tool_errors() {
        let jsonl = concat!(
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","is_error":true,"content":"error: cannot find crate `foo`"}]}}"#,
            "\n",
            r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok fine"}]}}"#,
            "\n",
        );
        let d = parse_transcript(jsonl);
        assert_eq!(d.errors.len(), 1);
        assert!(d.errors[0].contains("cannot find crate"));
        // Non-error results must not be captured.
        assert!(!d.errors.iter().any(|e| e.contains("fine")));
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
    fn upsert_session_memory_is_idempotent_per_session() {
        // A Stop hook fires once per agent turn — re-importing the same
        // (growing) session must converge to ONE memory, refreshed in place.
        let repo = crate::storage::MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |id: &str, summary: &str, files: &[&str]| EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "sess-1".into(),
            summary: summary.into(),
            content: format!("{summary} content"),
            files_touched: files.iter().map(|f| f.to_string()).collect(),
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["session-import".into()],
            created_at: 100,
            updated_at: 100,
        };

        let first = upsert_session_memory(&repo, &mk("m1", "first turn", &["a.rs"]), 100).unwrap();
        assert_eq!(first.status, "created");

        // Same session, more content → updates m1, does not add m2.
        let second = upsert_session_memory(
            &repo,
            &mk("m2", "first turn + more", &["a.rs", "b.rs"]),
            200,
        )
        .unwrap();
        assert_eq!(second.status, "updated");
        assert_eq!(second.id, "m1", "must refresh the existing memory");
        let updated = repo.get_episodic("m1", "p").unwrap().unwrap();
        assert_eq!(updated.summary, "first turn + more");
        assert_eq!(updated.files_touched.len(), 2);
        assert_eq!(updated.updated_at, 200);

        // Exactly one memory for the session.
        {
            let conn = repo.connection().unwrap();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM episodic_memories WHERE project_id = 'p' AND session_id = 'sess-1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1);
        }

        // A DIFFERENT session still creates its own memory.
        let mut m3 = mk("m3", "other session", &[]);
        m3.session_id = "sess-2".into();
        let third = upsert_session_memory(&repo, &m3, 300).unwrap();
        assert_eq!(third.status, "created");
    }

    #[test]
    fn optional_str_does_not_swallow_a_following_flag() {
        // `--session --dry-run` must not bind "--dry-run" as the session id
        // (same guard require_str enforces with an error).
        let args = vec!["--session".to_string(), "--dry-run".to_string()];
        assert_eq!(optional_str(&args, "session"), None);
        let args = vec!["--session".to_string(), "s-1".to_string()];
        assert_eq!(optional_str(&args, "session"), Some("s-1".to_string()));
        assert_eq!(optional_str(&args, "missing"), None);
    }

    #[test]
    fn parse_set_pairs_parses_json_values_with_string_fallback() {
        let args: Vec<String> = [
            "--set",
            "importance=0.9",
            "--set",
            "summary=plain text",
            "--set",
            "tags=[\"a\",\"b\"]",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let patch = parse_set_pairs(&args).unwrap();
        assert_eq!(patch.get("importance"), Some(&serde_json::json!(0.9)));
        assert_eq!(patch.get("summary"), Some(&serde_json::json!("plain text")));
        assert_eq!(patch.get("tags"), Some(&serde_json::json!(["a", "b"])));
    }

    #[test]
    fn read_text_bounded_reads_small_files_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        std::fs::write(&path, "line1\nline2\n").unwrap();
        let s = read_text_bounded(&path, 100, 10).unwrap();
        assert_eq!(s, "line1\nline2\n"); // no truncation marker on small files
    }

    #[test]
    fn read_text_bounded_keeps_head_and_tail_on_oversize() {
        // 100-byte file, 30-byte budget, 10-byte head → head 10 + tail 20,
        // 70 bytes skipped. The TAIL must survive: a session's final
        // conclusions live at the end of the transcript.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let content = format!("{}{}{}", "H".repeat(10), "M".repeat(70), "T".repeat(20));
        std::fs::write(&path, &content).unwrap();
        let s = read_text_bounded(&path, 30, 10).unwrap();
        assert!(s.contains(&"H".repeat(10)), "head kept: {s}");
        assert!(s.contains(&"T".repeat(20)), "tail (latest lines) kept: {s}");
        assert!(!s.contains('M'), "middle skipped: {s}");
        assert!(s.contains("70 bytes skipped"), "marker names the gap: {s}");
    }

    #[test]
    fn templated_session_imports_from_different_sessions_are_not_near_dups() {
        // Regression (round-5 review): the content template's scaffold words
        // are shared by EVERY session import. Two sessions differing by one
        // real word scored raw Jaccard ≈ 0.90 — over the 0.85 near-dup
        // threshold — so `consolidate --near` would archive one of them.
        // The scaffold must be excluded so only session CONTENT is compared.
        let mk = |prompt: &str| SessionDigest {
            user_prompts: vec![prompt.into()],
            files: vec!["a.rs".into()],
            errors: vec!["boom".into()],
            ..Default::default()
        };
        let content_a = session_import_content(&mk("revert"));
        let content_b = session_import_content(&mk("retry"));

        // Document the hazard first: the RAW (unfiltered) word sets clear the
        // default threshold — exactly what the exclusion must defeat.
        let raw_a: std::collections::HashSet<&str> = content_a.split_whitespace().collect();
        let raw_b: std::collections::HashSet<&str> = content_b.split_whitespace().collect();
        let raw_j = raw_a.intersection(&raw_b).count() as f64 / raw_a.union(&raw_b).count() as f64;
        assert!(
            raw_j >= crate::consolidation::engine::DEFAULT_JACCARD_THRESHOLD,
            "test precondition: raw scaffold Jaccard {raw_j:.2} must clear the threshold"
        );

        let repo = crate::storage::MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk_mem = |summary: &str, content: &str, ts: i64| EpisodicMemory {
            id: format!("m-{ts}"),
            project_id: "p".into(),
            session_id: format!("sess-{ts}"),
            summary: summary.into(),
            content: content.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["session-import".into()],
            created_at: ts,
            updated_at: ts,
        };
        repo.create_episodic(&mk_mem("revert", &content_a, 100))
            .unwrap();
        repo.create_episodic(&mk_mem("retry", &content_b, 200))
            .unwrap();

        let engine = crate::consolidation::ConsolidationEngine::new();
        let plan = engine
            .plan_for_kind(
                &repo,
                "p",
                crate::storage::MemoryKind::Episodic,
                true,
                crate::consolidation::engine::DEFAULT_JACCARD_THRESHOLD,
            )
            .unwrap();
        assert!(
            plan.groups.is_empty(),
            "two templated imports from different sessions must not be near-dups: {:?}",
            plan.groups
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

#[test]
fn parse_transcript_handles_codex_rollout_lines() {
    // Codex CLI rollout shape: {"timestamp","type","payload"} where
    // response_item payloads are Responses-API items.
    let jsonl = concat!(
        // session_meta: recognized, no digest signal, must not break parsing.
        r#"{"timestamp":"2026-09-19T00:00:00Z","type":"session_meta","payload":{"id":"s-1","cwd":"/repo"}}"#,
        "\n",
        // Injected environment context: filtered like Claude's wrappers.
        r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>os: macOS</environment_context>"}]}}"#,
        "\n",
        // Real user prompt.
        r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"fix the flaky sqlite test"}]}}"#,
        "\n",
        // Assistant conclusion.
        r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"fixed by adding a busy_timeout"}]}}"#,
        "\n",
        // apply_patch call: file markers.
        r#"{"type":"response_item","payload":{"type":"function_call","name":"apply_patch","arguments":"{\"input\":\"*** Begin Patch\\n*** Update File: src/storage/repository.rs\\n@@…\\n*** End Patch\"}"}}"#,
        "\n",
        // Failed tool output (success:false) + marked-error output.
        r#"{"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"command exited 1","success":false}}"#,
        "\n",
        r#"{"type":"response_item","payload":{"type":"function_call_output","call_id":"c2","output":"error: cannot find crate `foo`"}}"#,
        "\n",
        // Reasoning item: skipped.
        r#"{"type":"response_item","payload":{"type":"reasoning","summary":[]}}"#,
        "\n",
    );
    let d = parse_transcript(jsonl);
    assert_eq!(
        d.user_prompts,
        vec!["fix the flaky sqlite test".to_string()],
        "env context must be filtered: {:?}",
        d.user_prompts
    );
    assert_eq!(d.assistant_texts.len(), 1);
    assert!(d.assistant_texts[0].contains("busy_timeout"));
    assert_eq!(d.files, vec!["src/storage/repository.rs".to_string()]);
    assert_eq!(d.errors.len(), 2, "{:?}", d.errors);
    assert!(d.errors.iter().any(|e| e.contains("command exited 1")));
    assert!(d.errors.iter().any(|e| e.contains("cannot find crate")));
}

#[test]
fn parse_transcript_mixed_format_lines_do_not_interfere() {
    // A single file should never contain both, but per-line sniffing must
    // still leave unknown shapes alone.
    let jsonl = concat!(
        r#"{"type":"user","message":{"role":"user","content":"claude prompt"}}"#,
        "\n",
        r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"codex prompt"}]}}"#,
        "\n",
        r#"{"type":"mystery","shape":{"no":"payload"}}"#,
        "\n",
    );
    let d = parse_transcript(jsonl);
    assert_eq!(d.user_prompts.len(), 2);
}
