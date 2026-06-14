//! Project knowledge collectors for memory bootstrap.
//!
//! Collectors gather *evidence* (raw material) from an existing project's
//! many knowledge carriers — git history, docs, CI, dependencies — and
//! return it as structured JSON. They deliberately do **not** interpret or
//! summarize: turning evidence into memories is the upper-layer agent's job,
//! guided by the `engram.bootstrap` MCP prompt.
//!
//! Design principle: **collect evidence, not conclusions.** Every item
//! carries provenance (file path / line / commit hash) so any memory the
//! agent derives from it can be traced back to its source. This keeps
//! engram free of LLM dependencies while still producing high-quality
//! bootstrap material — the understanding happens in the agent, the
//! gathering happens here.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub mod docs;
pub mod failures;
pub mod git_collector;
pub mod workflow;

pub use docs::DocsCollection;
pub use failures::FailuresCollection;
pub use git_collector::{GitCollection, GitMilestone};
pub use workflow::WorkflowCollection;

/// A dimension of project knowledge to collect.
///
/// Each dimension maps to a different carrier (git / docs / CI) and, in the
/// bootstrap prompt, to a preferred target memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Dimension {
    /// Git history → episodic candidates + migration signals.
    Git,
    /// Docs / code annotations / dependency manifests → decision candidates.
    Decisions,
    /// Fix commits / CHANGELOG → failure candidates.
    Failures,
    /// CI / scripts / conventions → procedural candidates.
    Workflow,
}

impl Dimension {
    pub const ALL: &'static [Dimension] = &[
        Dimension::Git,
        Dimension::Decisions,
        Dimension::Failures,
        Dimension::Workflow,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Dimension::Git => "git",
            Dimension::Decisions => "decisions",
            Dimension::Failures => "failures",
            Dimension::Workflow => "workflow",
        }
    }

    /// Parse a comma-separated dimension list (e.g. `"git,decisions"`).
    ///
    /// Unknown tokens are silently dropped. `None`/empty → all dimensions,
    /// which is the natural default for a first-time bootstrap.
    pub fn parse_list(s: Option<&str>) -> Vec<Dimension> {
        match s {
            None | Some("") => Dimension::ALL.to_vec(),
            Some(raw) => raw
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter_map(|t| match t.as_str() {
                    "git" | "history" | "commits" => Some(Dimension::Git),
                    "decision" | "decisions" | "docs" => Some(Dimension::Decisions),
                    "failure" | "failures" | "bugs" => Some(Dimension::Failures),
                    "workflow" | "workflows" | "ci" => Some(Dimension::Workflow),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// Tunable knobs for a collection pass. Defaults are sane for a one-shot
/// bootstrap; callers (CLI / MCP) override per-invocation.
#[derive(Debug, Clone)]
pub struct CollectOptions {
    /// How many recent commits to walk for the git dimension.
    pub max_commits: usize,
    /// Commit hashes already stored as episodic memories — skipped so
    /// re-collection stays idempotent (mirrors `ingest_commits` dedup).
    pub ingested_commit_hashes: HashSet<String>,
    /// Per-file content cap (bytes). Larger files are truncated so a giant
    /// generated file can't blow up the agent's context.
    pub max_file_bytes: usize,
    /// Cap on commits listed inside a single themed milestone — keeps each
    /// milestone digestible; overflow commits are still counted.
    pub max_commits_per_milestone: usize,
}

impl Default for CollectOptions {
    fn default() -> Self {
        Self {
            max_commits: 200,
            ingested_commit_hashes: HashSet::new(),
            max_file_bytes: 16_000,
            max_commits_per_milestone: 15,
        }
    }
}

/// The full evidence bundle returned to the agent. Only requested dimensions
/// are populated; the rest are `None` (and omitted from JSON).
#[derive(Debug, Serialize)]
pub struct CollectedSources {
    pub project_id: String,
    pub repo_path: String,
    pub dimensions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<GitCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decisions: Option<DocsCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failures: Option<FailuresCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow: Option<WorkflowCollection>,
    pub summary: CollectionSummary,
}

/// Item counts per dimension, so the agent (and the user) can see at a
/// glance how much material was gathered and whether a dimension came up
/// empty (often a sign the project lacks that carrier, not a bug).
#[derive(Debug, Default, Serialize)]
pub struct CollectionSummary {
    pub total_items: usize,
    pub git_items: usize,
    pub decision_items: usize,
    pub failure_items: usize,
    pub workflow_items: usize,
    pub skipped_ingested_commits: usize,
    pub notes: Vec<String>,
}

/// Run all requested collectors and assemble the evidence bundle.
///
/// A failure in one dimension (e.g. the path is not a git repo, so the git
/// collector cannot run) is recorded as a note rather than aborting the whole
/// pass — the other dimensions are independent and still useful.
pub fn collect(
    project_id: &str,
    repo_path: &Path,
    dimensions: &[Dimension],
    opts: &CollectOptions,
) -> Result<CollectedSources> {
    let mut summary = CollectionSummary::default();
    let mut git = None;
    let mut decisions = None;
    let mut failures = None;
    let mut workflow = None;

    for &dim in dimensions {
        match dim {
            Dimension::Git => match git_collector::collect(repo_path, opts) {
                Ok(c) => {
                    summary.skipped_ingested_commits = c.skipped_ingested;
                    let items = c.item_count();
                    summary.git_items = items;
                    git = Some(c);
                }
                Err(e) => summary.notes.push(format!("git dimension skipped: {e}")),
            },
            Dimension::Decisions => match docs::collect(repo_path, opts) {
                Ok(c) => {
                    let items = c.item_count();
                    summary.decision_items = items;
                    decisions = Some(c);
                }
                Err(e) => summary
                    .notes
                    .push(format!("decisions dimension skipped: {e}")),
            },
            Dimension::Failures => match failures::collect(repo_path, opts) {
                Ok(c) => {
                    let items = c.item_count();
                    summary.failure_items = items;
                    failures = Some(c);
                }
                Err(e) => summary
                    .notes
                    .push(format!("failures dimension skipped: {e}")),
            },
            Dimension::Workflow => match workflow::collect(repo_path, opts) {
                Ok(c) => {
                    let items = c.item_count();
                    summary.workflow_items = items;
                    workflow = Some(c);
                }
                Err(e) => summary
                    .notes
                    .push(format!("workflow dimension skipped: {e}")),
            },
        }
    }

    summary.total_items =
        summary.git_items + summary.decision_items + summary.failure_items + summary.workflow_items;

    Ok(CollectedSources {
        project_id: project_id.to_string(),
        repo_path: repo_path.to_string_lossy().into_owned(),
        dimensions: dimensions.iter().map(|d| d.as_str().to_string()).collect(),
        git,
        decisions,
        failures,
        workflow,
        summary,
    })
}

// ─── Shared helpers ─────────────────────────────────────────────────

/// Directory names never descended into when walking for file-based sources.
/// Build artifacts, VCS metadata, and dependency caches are pure noise for
/// knowledge collection.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".cache",
    ".vscode",
    ".idea",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
    "vendor",
];

/// Walk a directory tree yielding every file path not under an ignored dir.
/// Symlinks and unreadable entries are skipped rather than fatal.
pub fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_inner(root, &mut out);
    out
}

fn walk_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if IGNORED_DIRS.contains(&name) {
                    continue;
                }
            }
            walk_inner(&path, out);
        } else if ft.is_file() {
            out.push(path);
        }
    }
}

/// Read a file as UTF-8 text, truncating to `max_bytes` with a marker if it
/// exceeds the cap. Lossy decoding keeps collection robust against binary
/// files that happen to match an extension filter.
pub fn read_bounded(path: &Path, max_bytes: usize) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() <= max_bytes {
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        let mut s = String::from_utf8_lossy(&bytes[..max_bytes]).into_owned();
        s.push_str("\n…[truncated]");
        Ok(s)
    }
}

/// A captured file's path and (possibly truncated) content. Used by every
/// file-based collector so the agent always sees provenance + text together.
#[derive(Debug, Serialize)]
pub struct FileSource {
    pub path: String,
    pub content: String,
    pub truncated: bool,
    pub bytes: usize,
}

impl FileSource {
    pub fn from_path(path: &Path, root: &Path, max_bytes: usize) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let bytes = raw.len();
        let truncated = bytes > max_bytes;
        let content = if truncated {
            let mut s = String::from_utf8_lossy(&raw[..max_bytes]).into_owned();
            s.push_str("\n…[truncated]");
            s
        } else {
            String::from_utf8_lossy(&raw).into_owned()
        };
        Ok(Self {
            path: relpath(path, root),
            content,
            truncated,
            bytes,
        })
    }
}

/// Render `path` relative to `root` for compact, repo-portable provenance.
/// Falls back to the absolute path if it isn't actually under `root`.
pub fn relpath(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_parse_all_on_empty() {
        assert_eq!(Dimension::parse_list(None), Dimension::ALL);
        assert_eq!(Dimension::parse_list(Some("")), Dimension::ALL);
    }

    #[test]
    fn dimension_parse_subset_and_aliases() {
        let d = Dimension::parse_list(Some("git, ci, docs"));
        assert_eq!(d.len(), 3);
        assert!(d.contains(&Dimension::Git));
        assert!(d.contains(&Dimension::Workflow));
        assert!(d.contains(&Dimension::Decisions));
    }

    #[test]
    fn dimension_parse_drops_unknown() {
        let d = Dimension::parse_list(Some("git, bogus, failures"));
        assert_eq!(d, vec![Dimension::Git, Dimension::Failures]);
    }

    #[test]
    fn walk_files_ignores_target_and_git() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.txt"), "x").unwrap();
        std::fs::create_dir(root.join("target")).unwrap();
        std::fs::write(root.join("target/b.txt"), "y").unwrap();
        std::fs::create_dir_all(root.join(".git/objects")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "z").unwrap();

        let files = walk_files(root);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.txt".to_string()));
        assert!(!names.contains(&"b.txt".to_string()));
        assert!(!names.contains(&"HEAD".to_string()));
    }

    #[test]
    fn read_bounded_truncates_large_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        std::fs::write(&path, "a".repeat(100)).unwrap();
        let s = read_bounded(&path, 10).unwrap();
        assert!(s.contains("[truncated]"));
    }

    #[test]
    fn collect_runs_all_dimensions_on_non_git_dir() {
        // A plain temp dir (no .git) — git dimension fails gracefully, others run.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "# hi\n").unwrap();
        let result = collect(
            "proj",
            dir.path(),
            Dimension::ALL,
            &CollectOptions::default(),
        )
        .unwrap();
        assert!(result.git.is_none());
        assert!(result.decisions.is_some());
        assert!(result
            .summary
            .notes
            .iter()
            .any(|n| n.contains("git dimension skipped")));
    }
}
