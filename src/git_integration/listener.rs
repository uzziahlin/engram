use crate::models::EpisodicMemory;
use anyhow::{Context, Result};
use gix::bstr::ByteSlice;
use std::path::Path;

/// A commit event extracted from git history.
#[derive(Debug, Clone)]
pub struct CommitEvent {
    pub commit_hash: String,
    pub message: String,
    pub files_changed: Vec<String>,
    pub timestamp: i64,
}

/// Git integration for monitoring commits and auto-generating memories.
pub struct GitIntegration {
    repo: gix::Repository,
}

impl GitIntegration {
    /// Open a git repository at the given path (searching upward for `.git`).
    pub fn new(repo_path: &Path) -> Result<Self> {
        let repo = gix::discover(repo_path).context("failed to discover git repository")?;
        Ok(Self { repo })
    }

    /// Get the N most recent commits (newest first by commit time).
    ///
    /// Unlike libgit2's `Sort::TIME`, gix's rev-walk is not chronological, so we
    /// collect every reachable commit, sort by commit time descending, then
    /// truncate. `ingest` is an explicit, non-hot-path command, so walking the
    /// full history is acceptable. The loaded commit handles are KEPT after the
    /// time pass and reused for the newest N — a separate second `find_commit`
    /// pass (as this function once did) re-decoded the same objects.
    pub fn get_recent_commits(&self, count: usize) -> Result<Vec<CommitEvent>> {
        let head_id = self.repo.head_id().context("failed to resolve HEAD")?;

        // Phase 1: collect all reachable commit ids. The walk borrows the repo,
        // so we finish it before touching commits individually below.
        let mut ids = Vec::new();
        for step in self.repo.rev_walk([head_id]).all()? {
            ids.push(step?.id);
        }

        // Phase 2: load each commit once (needed for its time anyway), sort
        // newest-first, and reuse the handles for Phase 3.
        let mut commits: Vec<gix::Commit<'_>> = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok(c) = self.repo.find_commit(id) {
                commits.push(c);
            }
        }
        commits.sort_by_key(|c| std::cmp::Reverse(c.time().map(|t| t.seconds).unwrap_or(0)));

        // Phase 3: build the event list for the newest `count` commits.
        let mut events = Vec::with_capacity(commits.len().min(count));
        for commit in commits.into_iter().take(count) {
            let ts = commit.time().ok().map(|t| t.seconds).unwrap_or(0);

            // gix's message_raw() keeps the trailing newline that git appends;
            // libgit2's Commit::message() trims it. Trim to match that behavior
            // so summaries and stored content are byte-identical to before.
            let message = commit
                .message_raw()
                .ok()
                .map(|b| b.to_str().unwrap_or("").trim_end().to_string())
                .unwrap_or_default();

            let files_changed = self.commit_files_changed(&commit)?;

            events.push(CommitEvent {
                commit_hash: commit.id().to_string(),
                message,
                files_changed,
                timestamp: ts,
            });
        }

        Ok(events)
    }

    /// Files touched by a commit relative to its first parent (or the empty
    /// tree for a root commit).
    fn commit_files_changed(&self, commit: &gix::Commit<'_>) -> Result<Vec<String>> {
        let new_tree = self.repo.find_tree(commit.tree_id()?)?;

        // First parent only (mirrors the git2 implementation): parent_ids yields
        // parent *commit* ids, so resolve each to its commit, then its tree.
        let parent_tree = match commit.parent_ids().next() {
            Some(parent_id) => {
                let parent_commit = self.repo.find_commit(parent_id)?;
                Some(self.repo.find_tree(parent_commit.tree_id()?)?)
            }
            None => None,
        };

        let changes = self
            .repo
            .diff_tree_to_tree(parent_tree.as_ref(), Some(&new_tree), None)?;

        // Default diff options use `Location::Path`, so every change carries
        // its full repo-relative path. Renames (Rewrite) record both halves.
        let mut files = Vec::new();
        for change in changes {
            use gix::object::tree::diff::ChangeDetached::*;
            match change {
                Addition { location, .. }
                | Deletion { location, .. }
                | Modification { location, .. } => {
                    push_path(&mut files, &location);
                }
                Rewrite {
                    source_location,
                    location,
                    ..
                } => {
                    push_path(&mut files, &source_location);
                    push_path(&mut files, &location);
                }
            }
        }

        Ok(files)
    }

    /// Auto-generate an episodic memory from a commit event.
    ///
    /// Creates a summary from the commit message and records
    /// files touched and the commit hash.
    pub fn auto_generate_episodic(
        &self,
        event: &CommitEvent,
        project_id: &str,
        session_id: &str,
    ) -> EpisodicMemory {
        let now = chrono::Utc::now().timestamp();

        // Tag the provenance so ingest-produced memories are distinguishable
        // from bootstrap / manual ones (see docs/bootstrap.md convention).
        let mut tags = extract_tags(&event.message);
        tags.push("source:ingest".into());

        EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            session_id: session_id.to_string(),
            summary: event.message.lines().next().unwrap_or("").to_string(),
            content: event.message.clone(),
            files_touched: event.files_changed.clone(),
            related_commits: vec![event.commit_hash.clone()],
            importance: estimate_importance(&event.message),
            tags,
            created_at: now,
            updated_at: now,
        }
    }

    /// Process recent commits and generate episodic memories for each.
    pub fn process_recent_commits(
        &self,
        project_id: &str,
        session_id: &str,
        count: usize,
    ) -> Result<Vec<EpisodicMemory>> {
        let commits = self.get_recent_commits(count)?;
        let memories = commits
            .iter()
            .map(|event| self.auto_generate_episodic(event, project_id, session_id))
            .collect();
        Ok(memories)
    }
}

/// Distill (already deduped) commit events into ONE episodic memory per
/// Conventional-Commit (type, scope) milestone — the clustering
/// `collectors::git_collector` uses for bootstrap. Replaces the old per-commit
/// ingest output, which produced N low-value memories (one per commit) that
/// git_collector's own module docs call out as noise. Deterministic, no LLM:
/// the milestone theme becomes the summary, commit subjects the content.
pub fn milestone_memories(
    project_id: &str,
    session_id: &str,
    events: &[CommitEvent],
) -> Vec<EpisodicMemory> {
    // cap = usize::MAX so `commits` carries every hash — related_commits is
    // the dedup key for re-ingest, so dropping hashes would break idempotency.
    let event_refs: Vec<&CommitEvent> = events.iter().collect();
    let milestones =
        crate::collectors::git_collector::cluster_by_theme(&event_refs, usize::MAX);
    let now = chrono::Utc::now().timestamp();

    const MAX_FILES: usize = 50;
    const MAX_SUBJECTS: usize = 30;
    const MAX_SUMMARY: usize = 200;

    milestones
        .into_iter()
        .map(|m| {
            let hashes: Vec<String> = m.commits.iter().map(|c| c.hash.clone()).collect();

            let mut files: Vec<String> = Vec::new();
            for c in &m.commits {
                for f in &c.files {
                    if !files.contains(f) && files.len() < MAX_FILES {
                        files.push(f.clone());
                    }
                }
            }

            let subjects: Vec<&str> = m
                .commits
                .iter()
                .map(|c| c.message.lines().next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .take(MAX_SUBJECTS)
                .collect();

            let first_subject = subjects.first().copied().unwrap_or("");
            let mut summary = format!("{}: {} commits — {}", m.theme, m.commit_count, first_subject);
            if summary.chars().count() > MAX_SUMMARY {
                summary = summary.chars().take(MAX_SUMMARY).collect();
            }

            let content = format!(
                "Milestone {} ({} commit(s), first {} last {}){}\n\nFiles: {}",
                m.theme,
                m.commit_count,
                m.first_ts,
                m.last_ts,
                if m.has_breaking { " [BREAKING]" } else { "" },
                files.join(", "),
            ) + &subjects
                .iter()
                .map(|s| format!("\n- {s}"))
                .collect::<String>();

            let mut tags = vec![m.commit_type.clone()];
            if let Some(scope) = &m.scope {
                if !scope.is_empty() {
                    tags.push(scope.clone());
                }
            }
            tags.push("source:ingest".into());

            EpisodicMemory {
                id: uuid::Uuid::new_v4().to_string(),
                project_id: project_id.to_string(),
                session_id: session_id.to_string(),
                summary,
                content,
                files_touched: files,
                related_commits: hashes,
                importance: milestone_importance(&m.commit_type, m.has_breaking),
                tags,
                created_at: now,
                updated_at: now,
            }
        })
        .collect()
}

/// Importance by milestone type — mirrors the old per-commit keyword scale,
/// decided once per theme instead of per commit message substring.
fn milestone_importance(commit_type: &str, has_breaking: bool) -> f32 {
    if has_breaking {
        return 0.9;
    }
    match commit_type {
        "feat" | "fix" => 0.7,
        "perf" | "refactor" => 0.6,
        "docs" => 0.3,
        _ => 0.4,
    }
}

/// Push a non-empty repo-relative path into `files`, de-duplicating.
/// `location` is a `gix::bstr::BString` (owned bytes).
fn push_path(files: &mut Vec<String>, location: &gix::bstr::BString) {
    if location.is_empty() {
        return;
    }
    let path = location.to_str().unwrap_or("").to_string();
    if !path.is_empty() && !files.contains(&path) {
        files.push(path);
    }
}

/// Estimate importance of a commit based on message keywords.
/// Uses max-score strategy instead of sequential override to avoid
/// "fix: update docs" being classified as docs (0.2) instead of fix (0.7).
/// Keywords match on word boundaries so "fixture" ≠ "fix", "perfect" ≠ "perf".
fn estimate_importance(message: &str) -> f32 {
    let lower = message.to_lowercase();
    let mut max_score: f32 = 0.3;

    if has_word(&lower, "fix") || has_word(&lower, "bug") || has_word(&lower, "patch") {
        max_score = max_score.max(0.7);
    }
    if has_word(&lower, "refactor") || has_word(&lower, "rewrite") {
        max_score = max_score.max(0.6);
    }
    if lower.contains("breaking") || has_word(&lower, "migration") {
        max_score = max_score.max(0.9);
    }
    if has_word(&lower, "docs") || has_word(&lower, "comment") {
        max_score = max_score.max(0.2);
    }
    if has_word(&lower, "test") {
        max_score = max_score.max(0.4);
    }

    max_score
}

/// Extract tags from a commit message based on common prefixes.
/// Word-boundary matching: "docker" must not produce a docs tag via "doc".
fn extract_tags(message: &str) -> Vec<String> {
    let lower = message.to_lowercase();
    let mut tags = Vec::new();

    if has_word(&lower, "fix") || has_word(&lower, "bug") {
        tags.push("bugfix".into());
    }
    if has_word(&lower, "feat") || has_word(&lower, "feature") {
        tags.push("feature".into());
    }
    if has_word(&lower, "refactor") {
        tags.push("refactor".into());
    }
    if has_word(&lower, "perf") || has_word(&lower, "performance") {
        tags.push("performance".into());
    }
    if has_word(&lower, "security") || has_word(&lower, "cve") {
        tags.push("security".into());
    }
    if has_word(&lower, "doc") || has_word(&lower, "docs") {
        tags.push("docs".into());
    }
    if has_word(&lower, "test") {
        tags.push("test".into());
    }
    if has_word(&lower, "deploy") || has_word(&lower, "release") {
        tags.push("deployment".into());
    }

    tags
}

/// Case-folded whole-word containment on a lowercased haystack.
fn has_word(haystack_lower: &str, word: &str) -> bool {
    let bytes = haystack_lower.as_bytes();
    haystack_lower.match_indices(word).any(|(i, _)| {
        let end = i + word.len();
        let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphabetic();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphabetic();
        before_ok && after_ok
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git_integration::{add_commit, make_test_repo};

    #[test]
    fn test_estimate_importance() {
        assert!(estimate_importance("fix: resolve auth bug") > 0.5);
        assert!(estimate_importance("docs: update README") < 0.4);
        assert!(estimate_importance("feat: breaking migration") > 0.8);
        assert!(estimate_importance("chore: update deps") >= 0.3);
    }

    #[test]
    fn test_extract_tags() {
        let tags = extract_tags("fix: resolve auth bug and update docs");
        assert!(tags.contains(&"bugfix".to_string()));
        assert!(tags.contains(&"docs".to_string()));
    }

    #[test]
    fn test_auto_generate_episodic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path();
        make_test_repo(
            repo_path,
            "test.txt",
            "hello",
            "feat: add initial test file",
        );

        let git = GitIntegration::new(repo_path).unwrap();
        let commits = git.get_recent_commits(10).unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].files_changed, vec!["test.txt"]);

        let episodic = git.auto_generate_episodic(&commits[0], "test-project", "session-1");
        assert_eq!(episodic.project_id, "test-project");
        assert_eq!(episodic.summary, "feat: add initial test file");
        assert!(episodic.tags.contains(&"feature".to_string()));
        assert!(episodic.files_touched.contains(&"test.txt".to_string()));
    }

    #[test]
    fn test_recent_commits_diff_against_parent() {
        // A second commit exercises the parent-tree diff path (the single-commit
        // test above only hits the root-commit / empty-tree case).
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path();
        make_test_repo(repo_path, "a.txt", "first", "feat: initial commit");
        add_commit(repo_path, "a.txt", "first\nsecond", "fix: modify a.txt");
        add_commit(repo_path, "b.txt", "new", "feat: add b.txt");

        let git = GitIntegration::new(repo_path).unwrap();
        let commits = git.get_recent_commits(10).unwrap();
        assert_eq!(commits.len(), 3, "should walk the full history");

        // Newest first by commit time; all commits share timestamp 0, so the
        // topological order (later commits first) must be preserved.
        assert_eq!(commits[0].message, "feat: add b.txt");
        assert_eq!(commits[0].files_changed, vec!["b.txt"]);
        assert_eq!(commits[1].message, "fix: modify a.txt");
        assert_eq!(commits[1].files_changed, vec!["a.txt"]);
        // Root commit diff is against the empty tree → its added files.
        assert_eq!(commits[2].message, "feat: initial commit");
        assert_eq!(commits[2].files_changed, vec!["a.txt"]);

        // Truncation: only the newest commit when count = 1.
        let one = git.get_recent_commits(1).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].message, "feat: add b.txt");
    }
}
