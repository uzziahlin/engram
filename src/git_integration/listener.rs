use crate::models::EpisodicMemory;
use anyhow::{Context, Result};
use git2::{Diff, Repository};
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
    repo: Repository,
}

impl GitIntegration {
    /// Open a git repository at the given path.
    pub fn new(repo_path: &Path) -> Result<Self> {
        let repo = Repository::discover(repo_path)
            .context("failed to discover git repository")?;
        Ok(Self { repo })
    }

    /// Get the N most recent commits.
    pub fn get_recent_commits(&self, count: usize) -> Result<Vec<CommitEvent>> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commits = Vec::new();
        for oid in revwalk.take(count) {
            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;

            let mut files_changed = Vec::new();
            if let Ok(diff) = self.get_commit_diff(&commit) {
                for delta in diff.deltas() {
                    if let Some(path) = delta.new_file().path() {
                        files_changed.push(path.to_string_lossy().to_string());
                    } else if let Some(path) = delta.old_file().path() {
                        files_changed.push(path.to_string_lossy().to_string());
                    }
                }
            }

            commits.push(CommitEvent {
                commit_hash: oid.to_string(),
                message: commit.message().unwrap_or("").to_string(),
                files_changed,
                timestamp: commit.time().seconds(),
            });
        }

        Ok(commits)
    }

    /// Get the diff for a specific commit.
    fn get_commit_diff(&self, commit: &git2::Commit) -> Result<Diff<'_>> {
        let tree = commit.tree()?;
        let parent_tree = commit.parent(0).ok().map(|p| p.tree()).transpose()?;

        let diff = self.repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&tree),
            None,
        )?;

        Ok(diff)
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

        EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            session_id: session_id.to_string(),
            summary: event.message.lines().next().unwrap_or("").to_string(),
            content: event.message.clone(),
            files_touched: event.files_changed.clone(),
            related_commits: vec![event.commit_hash.clone()],
            importance: estimate_importance(&event.message),
            tags: extract_tags(&event.message),
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

/// Estimate importance of a commit based on message keywords.
fn estimate_importance(message: &str) -> f32 {
    let lower = message.to_lowercase();
    let mut score = 0.3;

    if lower.contains("fix") || lower.contains("bug") || lower.contains("patch") {
        score = 0.7;
    }
    if lower.contains("refactor") || lower.contains("rewrite") {
        score = 0.6;
    }
    if lower.contains("breaking") || lower.contains("migration") {
        score = 0.9;
    }
    if lower.contains("docs") || lower.contains("comment") {
        score = 0.2;
    }
    if lower.contains("test") {
        score = 0.4;
    }

    score
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
        // Create a temp git repo for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path();

        // Init repo
        let repo = Repository::init(repo_path).unwrap();
        let sig = git2::Signature::new("Test", "test@test.com", &git2::Time::new(0, 0)).unwrap();

        // Create a file and commit
        let file_path = repo_path.join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "feat: add initial test file", &tree, &[]).unwrap();

        // Test GitIntegration
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
}
