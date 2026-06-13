pub mod listener;

pub use listener::GitIntegration;

/// Configured `git` invocation against `repo_path` with a fixed identity and
/// deterministic commit dates, for building throwaway repos in tests.
///
/// Uses the system `git` binary rather than a Rust git library so the
/// production dependency set stays minimal (no `tree-editor` feature needed).
/// `git` is present on every CI runner and dev machine.
#[cfg(test)]
fn git(repo_path: &std::path::Path, args: &[&str]) -> std::process::Output {
    std::process::Command::new("git")
        .args(["-C", repo_path.to_str().expect("utf-8 repo path")])
        .args(args)
        .env("GIT_AUTHOR_NAME", "Test")
        .env("GIT_AUTHOR_EMAIL", "test@test.com")
        .env("GIT_COMMITTER_NAME", "Test")
        .env("GIT_COMMITTER_EMAIL", "test@test.com")
        // `@<unix-secs> <tz>` is git's internal date format; the bare ISO
        // form "1970-01-01T00:00:00" is rejected as "invalid date format".
        .env("GIT_AUTHOR_DATE", "@0 +0000")
        .env("GIT_COMMITTER_DATE", "@0 +0000")
        .output()
        .expect("git command should run")
}

/// Create a fresh repository at `repo_path` with one committed file.
#[cfg(test)]
pub(crate) fn make_test_repo(repo_path: &std::path::Path, filename: &str, content: &str, message: &str) {
    assert!(
        git(repo_path, &["init", "-q"]).status.success(),
        "git init failed; is git installed?"
    );
    add_commit(repo_path, filename, content, message);
}

/// Stage `filename` (written with `content`) and commit it. For the second and
/// later commits this exercises the parent-tree diff path in `get_recent_commits`.
#[cfg(test)]
pub(crate) fn add_commit(repo_path: &std::path::Path, filename: &str, content: &str, message: &str) {
    std::fs::write(repo_path.join(filename), content).unwrap();
    assert!(
        git(repo_path, &["add", filename]).status.success(),
        "git add failed"
    );
    assert!(
        git(repo_path, &["commit", "-q", "-m", message])
            .status
            .success(),
        "git commit failed"
    );
}
