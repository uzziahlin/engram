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
    /// full history is acceptable.
    pub fn get_recent_commits(&self, count: usize) -> Result<Vec<CommitEvent>> {
        let head_id = self.repo.head_id().context("failed to resolve HEAD")?;

        // Phase 1: collect all reachable commit ids. The walk borrows the repo,
        // so we finish it before touching commits individually below.
        let mut ids = Vec::new();
        for step in self.repo.rev_walk([head_id]).all()? {
            ids.push(step?.id);
        }

        // Phase 2: resolve each id's commit time and sort newest-first.
        let mut entries: Vec<(gix::ObjectId, i64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let time_secs = self
                .repo
                .find_commit(id)
                .ok()
                .and_then(|c| c.time().ok())
                .map(|t| t.seconds)
                .unwrap_or(0);
            entries.push((id, time_secs));
        }
        entries.sort_by_key(|&(_, ts)| std::cmp::Reverse(ts));

        // Phase 3: build the event list for the newest `count` commits.
        let mut commits = Vec::with_capacity(entries.len().min(count));
        for (id, ts) in entries.into_iter().take(count) {
            let commit = self.repo.find_commit(id)?;

            // gix's message_raw() keeps the trailing newline that git appends;
            // libgit2's Commit::message() trims it. Trim to match that behavior
            // so summaries and stored content are byte-identical to before.
            let message = commit
                .message_raw()
                .ok()
                .map(|b| b.to_str().unwrap_or("").trim_end().to_string())
                .unwrap_or_default();

            let files_changed = self.commit_files_changed(&commit)?;

            commits.push(CommitEvent {
                commit_hash: id.to_string(),
                message,
                files_changed,
                timestamp: ts,
            });
        }

        Ok(commits)
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
fn estimate_importance(message: &str) -> f32 {
    let lower = message.to_lowercase();
    let mut max_score: f32 = 0.3;

    if lower.contains("fix") || lower.contains("bug") || lower.contains("patch") {
        max_score = max_score.max(0.7);
    }
    if lower.contains("refactor") || lower.contains("rewrite") {
        max_score = max_score.max(0.6);
    }
    if lower.contains("breaking") || lower.contains("migration") {
        max_score = max_score.max(0.9);
    }
    if lower.contains("docs") || lower.contains("comment") {
        max_score = max_score.max(0.2);
    }
    if lower.contains("test") {
        max_score = max_score.max(0.4);
    }

    max_score
}

/// Extract tags from a commit message based on common prefixes.
fn extract_tags(message: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let lower = message.to_lowercase();

    if lower.contains("fix") || lower.contains("bug") {
        tags.push("bugfix".into());
    }
    if lower.contains("feat") || lower.contains("feature") {
        tags.push("feature".into());
    }
    if lower.contains("refactor") {
        tags.push("refactor".into());
    }
    if lower.contains("perf") || lower.contains("performance") {
        tags.push("performance".into());
    }
    if lower.contains("security") || lower.contains("cve") {
        tags.push("security".into());
    }
    if lower.contains("doc") {
        tags.push("docs".into());
    }
    if lower.contains("test") {
        tags.push("test".into());
    }
    if lower.contains("deploy") || lower.contains("release") {
        tags.push("deployment".into());
    }

    tags
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
