//! Failures collector.
//!
//! Gathers evidence of past bugs and incidents: fix/hotfix/revert commits,
//! `CHANGELOG.md` "Fixed" sections, and suspiciously-named test files. The
//! commit message says *what* was fixed; the agent infers the *root cause*
//! and *prevention* (often from the changed files) to write a `failure`
//! memory — that interpretation is explicitly left to the agent.

use crate::git_integration::CommitEvent;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use super::{walk_files, CollectOptions};

const MAX_FIX_COMMITS: usize = 100;
const MAX_ERROR_TEST_FILES: usize = 30;

/// Evidence for past failures and incidents.
#[derive(Debug, Serialize)]
pub struct FailuresCollection {
    pub fix_commits: Vec<FixCommit>,
    pub changelog_fixed: Vec<ChangelogFixedEntry>,
    /// Test files whose names hint at edge/error cases previously worth
    /// pinning down — weak signal, but sometimes surfaces latent bugs.
    pub error_test_files: Vec<String>,
}

impl FailuresCollection {
    pub fn item_count(&self) -> usize {
        self.fix_commits.len() + self.changelog_fixed.len() + self.error_test_files.len()
    }
}

#[derive(Debug, Serialize)]
pub struct FixCommit {
    pub hash: String,
    pub message: String,
    pub files: Vec<String>,
    pub timestamp: i64,
}

#[derive(Debug, Serialize)]
pub struct ChangelogFixedEntry {
    pub path: String,
    pub section: String,
    pub line_start: usize,
    pub entry: String,
}

/// `events` carries the commit history pre-walked (and shared) by the
/// orchestrating `collect()`; `None` means the walk failed or wasn't needed —
/// the fix-commit section simply comes up empty and the other evidence
/// (CHANGELOG, test files) still applies.
pub fn collect(
    repo_path: &Path,
    events: Option<&[CommitEvent]>,
    opts: &CollectOptions,
) -> Result<FailuresCollection> {
    let mut fix_commits = Vec::new();

    if let Some(events) = events {
        for e in events {
            if opts.ingested_commit_hashes.contains(&e.commit_hash) {
                continue;
            }
            if is_fix_message(&e.message) && fix_commits.len() < MAX_FIX_COMMITS {
                fix_commits.push(FixCommit {
                    hash: e.commit_hash.clone(),
                    message: e.message.clone(),
                    files: e.files_changed.clone(),
                    timestamp: e.timestamp,
                });
            }
        }
    }

    let mut changelog_fixed = Vec::new();
    let mut error_test_files = Vec::new();

    for path in walk_files(repo_path) {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if name == "CHANGELOG.md" {
            changelog_fixed.extend(parse_changelog_fixed(&path, repo_path));
            continue;
        }

        if error_test_files.len() < MAX_ERROR_TEST_FILES && looks_like_error_test(name) {
            error_test_files.push(super::relpath(&path, repo_path));
        }
    }

    Ok(FailuresCollection {
        fix_commits,
        changelog_fixed,
        error_test_files,
    })
}

/// Does a commit message describe a fix? Conventional `fix:` / `hotfix:` /
/// `revert:` heads are the strongest signal (exact type match, so `fixture:`
/// is NOT a fix); otherwise fall back to whole-word keywords (so "dispatch"
/// doesn't match "patch" and "debugger" doesn't match "bug").
fn is_fix_message(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    let first = lower.lines().next().unwrap_or("").trim();
    if let Some((head, _)) = first.split_once(':') {
        let ty = head.split('(').next().unwrap_or("").trim();
        if matches!(ty, "fix" | "hotfix" | "revert" | "bugfix") {
            return true;
        }
    }
    let keywords = [
        "bug",
        "bugfix",
        "patch",
        "crash",
        "regression",
        "hotfix",
        "broken",
        "leak",
        "fix",
    ];
    // Keyword fallback inspects the SUBJECT line only: commit bodies are
    // prose ("docs: memory leak benchmark" has "leak" in the body context
    // but is not a fix) and previously caused systematic false positives.
    keywords.iter().any(|k| contains_word(first, k))
}

/// Case-folded whole-word containment: both sides of the match must be a
/// non-letter (or the string edge).
fn contains_word(haystack_lower: &str, word: &str) -> bool {
    let bytes = haystack_lower.as_bytes();
    haystack_lower.match_indices(word).any(|(i, _)| {
        let end = i + word.len();
        let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphabetic();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphabetic();
        before_ok && after_ok
    })
}

/// Parse `## Fixed` / `### Bug Fixes` style sections out of a CHANGELOG,
/// collecting each bullet as a candidate failure entry.
fn parse_changelog_fixed(path: &Path, root: &Path) -> Vec<ChangelogFixedEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let rel = super::relpath(path, root);
    let mut entries = Vec::new();
    let mut in_fixed = false;
    let mut section = String::new();

    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.starts_with('#') {
            let heading = t.trim_start_matches('#').trim().to_lowercase();
            in_fixed = is_fixed_section_heading(&heading);
            if in_fixed {
                section = t.trim_start_matches('#').trim().to_string();
            }
            continue;
        }
        if !in_fixed {
            continue;
        }
        let item = t
            .strip_prefix("- ")
            .or_else(|| t.strip_prefix("* "))
            .or_else(|| t.strip_prefix("+ "))
            .map(str::trim);
        if let Some(item) = item {
            if !item.is_empty() {
                entries.push(ChangelogFixedEntry {
                    path: rel.clone(),
                    section: section.clone(),
                    line_start: i + 1,
                    entry: item.to_string(),
                });
            }
        }
    }
    entries
}

/// Whether a CHANGELOG heading names a fixes section. Exact-word matching —
/// the old `contains("fix") || contains("bug")` also matched "Prefix
/// conventions" and "Debugging guide" and swallowed their bullets.
fn is_fixed_section_heading(heading: &str) -> bool {
    let lower = heading.to_lowercase();
    lower.split_whitespace().any(|w| {
        matches!(
            w,
            "fixed" | "fixes" | "bugfix" | "bugfixes" | "bug" | "bugs"
        )
    })
}

/// Heuristic: does a filename suggest it pins down an error/edge case?
fn looks_like_error_test(name: &str) -> bool {
    let lower = name.to_lowercase();
    (lower.starts_with("test") || lower.contains("_test") || lower.ends_with("test.rs"))
        && (lower.contains("error")
            || lower.contains("panic")
            || lower.contains("edge")
            || lower.contains("fuzz")
            || lower.contains("fail"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git_integration::{add_commit, make_test_repo};

    #[test]
    fn is_fix_detects_conventional_and_keywords() {
        assert!(is_fix_message("fix: null deref on empty input"));
        assert!(is_fix_message("hotfix: patch the leak"));
        assert!(is_fix_message("revert: bad config"));
        assert!(is_fix_message("Handle crash on startup"));
        assert!(is_fix_message("fix the login flow")); // bare verb, not conventional
        assert!(!is_fix_message("feat: add login screen"));
        assert!(!is_fix_message("feat(auth): dispatch handler")); // no "patch" word
        assert!(!is_fix_message("chore: update debugger config")); // no "bug" word
        assert!(!is_fix_message("feat: fixture for tests")); // "fixture" is not "fix:"
                                                             // Body prose must not trigger the keyword fallback — only subjects.
        assert!(!is_fix_message(
            "docs: benchmark notes\n\nmeasures memory leak rates"
        ));
    }

    #[test]
    fn changelog_section_headings_match_exact_words() {
        assert!(is_fixed_section_heading("fixed"));
        assert!(is_fixed_section_heading("bug fixes"));
        assert!(is_fixed_section_heading("Bugfixes"));
        assert!(!is_fixed_section_heading("prefix conventions"));
        assert!(!is_fixed_section_heading("debugging guide"));
    }

    #[test]
    fn parses_changelog_fixed_section() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("CHANGELOG.md"),
            "# Changelog\n\n## [1.2.0]\n### Added\n- new thing\n\n### Fixed\n- crash on empty input\n- memory leak in pool\n\n## [1.1.0]\n### Fixed\n- token expiry\n",
        )
        .unwrap();
        let entries = parse_changelog_fixed(&root.join("CHANGELOG.md"), root);
        assert!(entries.iter().any(|e| e.entry.contains("crash")));
        assert!(entries.iter().any(|e| e.entry.contains("memory leak")));
        assert!(entries.iter().any(|e| e.entry.contains("token expiry")));
        // "new thing" is in Added, not Fixed — must be excluded.
        assert!(entries.iter().all(|e| !e.entry.contains("new thing")));
    }

    #[test]
    fn collects_fix_commits_from_repo() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        make_test_repo(root, "a.txt", "1", "feat: initial");
        add_commit(root, "a.txt", "2", "fix: crash on empty");
        add_commit(root, "a.txt", "3", "docs: readme");

        let git = crate::git_integration::GitIntegration::new(root).unwrap();
        let events = git.get_recent_commits(50).unwrap();
        let col = collect(root, Some(&events), &CollectOptions::default()).unwrap();
        assert_eq!(col.fix_commits.len(), 1);
        assert!(col.fix_commits[0].message.contains("crash"));
    }
}
