//! Git-history collector.
//!
//! Improves on `ingest_commits`' flat per-commit output by clustering
//! commits into themed milestones (by Conventional-Commit type and scope),
//! flagging migration/breaking candidates, and skipping commits already
//! stored as memories. The agent turns each milestone into at most one
//! high-value episodic memory instead of N noisy ones.

use crate::git_integration::{CommitEvent, GitIntegration};
use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use super::CollectOptions;

/// Evidence gathered from git history.
#[derive(Debug, Serialize)]
pub struct GitCollection {
    /// Commits grouped into themes (by type/scope) — the primary material
    /// for episodic memories. One group → ideally one memory.
    pub milestones: Vec<GitMilestone>,
    /// Commits that look like migrations / breaking changes — strong
    /// candidates for a decision or high-importance episodic memory.
    pub migrations: Vec<CommitRef>,
    /// Individual recent commits (newest-first), for the agent to fall back
    /// on when milestones don't capture something worth recording.
    pub recent_commits: Vec<CommitRef>,
    /// Commits skipped because they were already ingested as memories.
    pub skipped_ingested: usize,
}

impl GitCollection {
    pub fn item_count(&self) -> usize {
        self.milestones.len() + self.migrations.len() + self.recent_commits.len()
    }
}

/// A themed cluster of related commits.
#[derive(Debug, Serialize)]
pub struct GitMilestone {
    /// Human-readable theme label, e.g. `"feat: auth"` or `"fix"`.
    pub theme: String,
    pub commit_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub commit_count: usize,
    /// Up to `max_commits_per_milestone` representative commits.
    pub commits: Vec<CommitRef>,
    pub files: Vec<String>,
    pub first_ts: i64,
    pub last_ts: i64,
    pub has_breaking: bool,
}

/// A provenance-bearing commit reference.
#[derive(Debug, Serialize, Clone)]
pub struct CommitRef {
    pub hash: String,
    pub message: String,
    pub files: Vec<String>,
    pub timestamp: i64,
}

impl From<&CommitEvent> for CommitRef {
    fn from(e: &CommitEvent) -> Self {
        Self {
            hash: e.commit_hash.clone(),
            message: e.message.clone(),
            files: e.files_changed.clone(),
            timestamp: e.timestamp,
        }
    }
}

/// Collect git evidence. Returns an empty collection (not an error) if the
/// path isn't a git repo — `collect()` reports that as a note.
pub fn collect(repo_path: &Path, opts: &CollectOptions) -> Result<GitCollection> {
    let git = GitIntegration::new(repo_path)?;
    let events = git.get_recent_commits(opts.max_commits)?;
    Ok(collect_from_events(&events, opts))
}

/// Collect git evidence from pre-walked commit events. The orchestrating
/// `collect()` walks the history ONCE and shares the events between the git
/// and failures dimensions — two independent walks doubled the full-history
/// traversal cost.
pub fn collect_from_events(events: &[CommitEvent], opts: &CollectOptions) -> GitCollection {
    let mut skipped = 0usize;
    let fresh: Vec<&CommitEvent> = events
        .iter()
        .filter(|e| {
            if opts.ingested_commit_hashes.contains(&e.commit_hash) {
                skipped += 1;
                false
            } else {
                true
            }
        })
        .collect();

    let migrations: Vec<CommitRef> = fresh
        .iter()
        .filter(|e| looks_like_migration(&e.message))
        .map(|e| CommitRef::from(*e))
        .collect();

    let milestones = cluster_by_theme(&fresh, opts.max_commits_per_milestone);

    let recent_commits: Vec<CommitRef> = fresh.iter().map(|e| CommitRef::from(*e)).collect();

    GitCollection {
        milestones,
        migrations,
        recent_commits,
        skipped_ingested: skipped,
    }
}

/// Group commits by Conventional-Commit `(type, scope)`. Commits that don't
/// follow the convention land in an `"other"` bucket so nothing is lost.
/// `pub(crate)`: `ingest_commits` reuses this to build one memory per theme.
pub(crate) fn cluster_by_theme(commits: &[&CommitEvent], cap: usize) -> Vec<GitMilestone> {
    let mut groups: BTreeMap<(String, Option<String>), Vec<&CommitEvent>> = BTreeMap::new();
    for c in commits {
        let (typ, scope) = parse_conventional(&c.message);
        let key = (
            typ.unwrap_or("other").to_string(),
            scope.map(|s| s.to_string()),
        );
        groups.entry(key).or_default().push(c);
    }

    // Split each theme where consecutive commits are more than
    /// MAX_THEME_GAP apart: without the window, years-apart commits merged
    /// into one "milestone" whose count and file list meant nothing.
    const MAX_THEME_GAP_SECS: i64 = 30 * 24 * 3600;

    let mut out = Vec::new();
    for ((typ, scope), mut evs) in groups {
        // Newest first (input order is newest-first from the rev walk; sort
        // explicitly so the invariant doesn't depend on the caller).
        evs.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        let mut run: Vec<&CommitEvent> = Vec::new();
        let flush = |run: &mut Vec<&CommitEvent>, out: &mut Vec<GitMilestone>| {
            if !run.is_empty() {
                out.push(build_milestone(typ.as_str(), scope.clone(), run, cap));
            }
            run.clear();
        };
        let mut prev_ts: Option<i64> = None;
        for e in evs {
            if let Some(p) = prev_ts {
                if p - e.timestamp > MAX_THEME_GAP_SECS {
                    flush(&mut run, &mut out);
                }
            }
            run.push(e);
            prev_ts = Some(e.timestamp);
        }
        flush(&mut run, &mut out);
    }
    // Most populous themes first — the agent should see the big threads early.
    out.sort_by_key(|m| std::cmp::Reverse(m.commit_count));
    out
}

fn build_milestone(
    typ: &str,
    scope: Option<String>,
    evs: &[&CommitEvent],
    cap: usize,
) -> GitMilestone {
    let mut files: HashSet<&str> = HashSet::new();
    let mut has_breaking = false;
    let mut ts_min = i64::MAX;
    let mut ts_max = 0i64;
    let mut commits = Vec::new();
    // Cap the file list: a huge theme's evidence JSON previously embedded
    // thousands of paths (overflow is still counted via commit_count).
    const MAX_FILES: usize = 50;

    for e in evs {
        for f in &e.files_changed {
            if files.len() < MAX_FILES {
                files.insert(f.as_str());
            }
        }
        if looks_like_migration(&e.message) {
            has_breaking = true;
        }
        ts_min = ts_min.min(e.timestamp);
        ts_max = ts_max.max(e.timestamp);
        if commits.len() < cap {
            commits.push(CommitRef::from(*e));
        }
    }

    GitMilestone {
        theme: format_theme(typ, scope.as_deref()),
        commit_type: typ.to_string(),
        scope,
        commit_count: evs.len(),
        commits,
        files: files.into_iter().map(String::from).collect(),
        first_ts: if ts_min == i64::MAX { 0 } else { ts_min },
        last_ts: ts_max,
        has_breaking,
    }
}

fn format_theme(typ: &str, scope: Option<&str>) -> String {
    match scope {
        Some(s) if !s.is_empty() => format!("{typ}: {s}"),
        _ => typ.to_string(),
    }
}

/// Parse the Conventional-Commit head of a message: `type(scope): desc` or
/// `type: desc`. Returns `(None, None)` if the first line isn't a recognized
/// conventional head (e.g. merge commits, prose-only subjects).
///
/// Hand-rolled rather than regex to avoid pulling in a regex dependency —
/// the grammar is tiny and this keeps engram's dep set minimal.
fn parse_conventional(msg: &str) -> (Option<&str>, Option<&str>) {
    /// Known Conventional-Commit types. Without the whitelist any
    /// lowercase-word-before-a-colon became a "type" ("http: fix url",
    /// "wip: stuff"), exploding the theme count with junk buckets.
    const KNOWN_TYPES: &[&str] = &[
        "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore",
        "revert", "hotfix", "bugfix", "wip", "release", "merge",
    ];

    let first = match msg.lines().next() {
        Some(l) => l.trim(),
        None => return (None, None),
    };
    let colon = match first.find(':') {
        Some(i) if i > 0 => i,
        _ => return (None, None),
    };
    let head = &first[..colon];
    let mut chars = head.chars();
    // Conventional types start with a lowercase letter.
    if !chars.next().is_some_and(|c| c.is_ascii_lowercase()) {
        return (None, None);
    }

    if let Some(open) = head.find('(') {
        // Unclosed paren (e.g. "fix(auth") is malformed — treat as no-scope
        // rather than leaking "fix(auth" into the theme key.
        if let Some(close) = head[open..].find(')') {
            let typ = &head[..open];
            if KNOWN_TYPES.contains(&typ) {
                let scope = &head[open + 1..open + close];
                return (Some(typ), Some(scope));
            }
            return (None, None);
        }
        return (None, None);
    }
    if KNOWN_TYPES.contains(&head) {
        return (Some(head), None);
    }
    (None, None)
}

/// Heuristic: does this commit message describe a migration or breaking
/// change worth elevating above routine churn?
fn looks_like_migration(message: &str) -> bool {
    let lower = message.to_lowercase();
    // `BREAKING CHANGE:` footer or `!:` marker from Conventional Commits.
    if lower.contains("breaking change") || lower.contains("breaking:") {
        return true;
    }
    if let Some(first) = message.lines().next() {
        if first.contains("!:") {
            return true;
        }
    }
    // "rewrite" alone was removed: "refactor: rewrite parser loop" is not a
    // breaking change — only the explicit `rewrite!` / "rewrite:" conventional
    // breaking markers count.
    lower.contains("migration") || lower.contains("migrate") || lower.contains("deprecat")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_conventional_with_scope() {
        let (t, s) = parse_conventional("feat(auth): add OAuth login");
        assert_eq!(t, Some("feat"));
        assert_eq!(s, Some("auth"));
    }

    #[test]
    fn parse_conventional_without_scope() {
        let (t, s) = parse_conventional("fix: null pointer");
        assert_eq!(t, Some("fix"));
        assert_eq!(s, None);
    }

    #[test]
    fn parse_conventional_rejects_prose() {
        assert_eq!(parse_conventional("Update the README"), (None, None));
        assert_eq!(parse_conventional("Merge branch x"), (None, None));
        assert_eq!(parse_conventional(""), (None, None));
    }

    #[test]
    fn cluster_splits_themes_across_time_gaps() {
        // Two `fix` clusters 2 years apart must become TWO milestones —
        // the old (type, scope)-only key merged years of history into one.
        let mk = |ts: i64, subject: &str| crate::git_integration::CommitEvent {
            commit_hash: format!("h{ts}"),
            message: subject.into(),
            files_changed: vec![],
            timestamp: ts,
        };
        let y = 365 * 24 * 3600;
        let events = [
            mk(0, "fix: early crash"),
            mk(3600, "fix: early leak"),
            mk(2 * y, "fix: modern panic"),
            mk(2 * y + 3600, "fix: modern race"),
        ];
        let refs: Vec<&_> = events.iter().collect();
        let milestones = cluster_by_theme(&refs, 15);
        assert_eq!(milestones.len(), 2, "gap > 30 days must split the theme");
        assert_eq!(milestones[0].commit_count, 2);
        assert_eq!(milestones[1].commit_count, 2);
        assert!(milestones[0].first_ts > milestones[1].first_ts);
    }

    #[test]
    fn parse_conventional_whitelist_rejects_junk_heads() {
        // Arbitrary word-before-colon is not a conventional type.
        assert_eq!(parse_conventional("http: fix url"), (None, None));
        assert_eq!(parse_conventional("v2: the big release"), (None, None));
        // Unclosed paren is malformed, not a scoped type.
        assert_eq!(parse_conventional("fix(auth: thing"), (None, None));
        // Real types still parse.
        let (t, s) = parse_conventional("fix(auth): token refresh");
        assert_eq!(t, Some("fix"));
        assert_eq!(s, Some("auth"));
    }

    #[test]
    fn rewrite_alone_is_not_breaking() {
        assert!(!looks_like_migration("refactor: rewrite parser loop"));
        assert!(looks_like_migration("refactor!: rewrite parser loop"));
        assert!(looks_like_migration("feat: migrate storage to SQLite"));
    }

    #[test]
    fn looks_like_migration_detects_markers() {
        assert!(looks_like_migration("feat!: drop Python 3.7"));
        assert!(looks_like_migration("BREAKING CHANGE: removes old API"));
        assert!(looks_like_migration("refactor: migrate to tokio"));
        assert!(!looks_like_migration("docs: typo"));
    }

    #[test]
    fn cluster_groups_by_type_and_scope() {
        use crate::git_integration::{add_commit, make_test_repo};
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        make_test_repo(p, "a.txt", "1", "feat(auth): login");
        add_commit(p, "a.txt", "2", "feat(auth): logout");
        add_commit(p, "b.txt", "3", "fix: crash");
        add_commit(p, "c.txt", "4", "docs: readme");

        let opts = CollectOptions::default();
        let col = collect(p, &opts).unwrap();
        // 4 themes: feat:auth, fix, docs, plus the initial `feat(auth)` root
        // commit doubles up — assert the auth theme has 2 commits.
        let auth = col
            .milestones
            .iter()
            .find(|m| m.theme == "feat: auth")
            .expect("auth theme exists");
        assert_eq!(auth.commit_count, 2);
    }

    #[test]
    fn collect_skips_ingested_commits() {
        use crate::git_integration::make_test_repo;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        make_test_repo(p, "a.txt", "1", "feat: one");

        let git = GitIntegration::new(p).unwrap();
        let first_hash = git.get_recent_commits(1).unwrap()[0].commit_hash.clone();

        let mut opts = CollectOptions::default();
        opts.ingested_commit_hashes.insert(first_hash);
        let col = collect(p, &opts).unwrap();
        assert_eq!(col.skipped_ingested, 1);
        assert!(col.recent_commits.is_empty());
    }
}
