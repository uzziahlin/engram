//! Validation for agent-supplied filesystem paths (`repo_path`).
//!
//! `ingest_commits` / `collect_sources` accept a `repo_path` argument from the
//! MCP client — i.e. ultimately from the agent, which can be steered by prompt
//! injection. Left unvalidated, `collect_sources(repo_path=$HOME)` walks the
//! user's home and returns file contents in the tool response. This module
//! centralizes the guard every entry point must apply before touching the path:
//!
//! - the path must exist and be a directory (fail fast on typos);
//! - the filesystem root and the user's home directory are always rejected —
//!   no legitimate project lives there, and both turn one call into a
//!   whole-disk scan;
//! - when `[security] allowed_roots` is configured (non-empty), the path must
//!   fall under one of the configured roots — strict mode for users who want
//!   a hard boundary.
//!
//! Defense in depth: the collectors' walk budget (entry/depth/time caps) and
//! per-file bounded reads bound the damage even for paths that pass here.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Validate an agent-supplied repo path. Returns the canonicalized path.
pub fn validate_repo_path(raw: &Path, allowed_roots: &[PathBuf]) -> Result<PathBuf> {
    let canonical = raw
        .canonicalize()
        .with_context(|| format!("repo_path {:?} is not accessible", raw))?;
    if !canonical.is_dir() {
        bail!("repo_path {:?} is not a directory", canonical);
    }

    if canonical == Path::new("/") {
        bail!("repo_path / (filesystem root) is not allowed");
    }
    if let Some(home) = dirs::home_dir() {
        // Compare canonical-to-canonical: a configured root may itself be a
        // symlink or contain `~`.
        if let Ok(home_canon) = home.canonicalize() {
            if canonical == home_canon {
                bail!(
                    "repo_path {:?} is the user's home directory — point at a \
                     project directory instead",
                    canonical
                );
            }
        }
    }

    if !allowed_roots.is_empty() {
        let allowed: Vec<PathBuf> = allowed_roots
            .iter()
            .filter_map(|r| r.canonicalize().ok())
            .collect();
        let under_root = allowed.iter().any(|root| canonical.starts_with(root));
        if !under_root {
            bail!(
                "repo_path {:?} is outside the configured [security] allowed_roots {:?}",
                canonical,
                allowed_roots
            );
        }
    }

    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_path() {
        let err =
            validate_repo_path(Path::new("/nonexistent/definitely/missing"), &[]).unwrap_err();
        assert!(err.to_string().contains("not accessible"));
    }

    #[test]
    fn rejects_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "x").unwrap();
        let err = validate_repo_path(&file, &[]).unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    #[test]
    fn rejects_home_dir() {
        let home = dirs::home_dir().unwrap();
        let err = validate_repo_path(&home, &[]).unwrap_err();
        assert!(err.to_string().contains("home directory"));
    }

    #[test]
    fn rejects_root() {
        let err = validate_repo_path(Path::new("/"), &[]).unwrap_err();
        assert!(err.to_string().contains("filesystem root"));
    }

    #[test]
    fn accepts_normal_dir_and_canonicalizes() {
        let dir = tempfile::tempdir().unwrap();
        let got = validate_repo_path(dir.path(), &[]).unwrap();
        assert!(got.is_dir());
        assert!(got.is_absolute());
    }

    #[test]
    fn enforces_allowed_roots() {
        let outer = tempfile::tempdir().unwrap();
        let inside = outer.path().join("proj");
        std::fs::create_dir_all(&inside).unwrap();
        let outside = tempfile::tempdir().unwrap();

        assert!(validate_repo_path(&inside, &[outer.path().to_path_buf()]).is_ok());
        let err = validate_repo_path(outside.path(), &[outer.path().to_path_buf()]).unwrap_err();
        assert!(err.to_string().contains("allowed_roots"));
    }
}
