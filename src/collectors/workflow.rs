//! Workflow collector.
//!
//! Gathers the carriers of "how we work here": CI pipelines, task scripts
//! (Makefile/justfile/scripts), lint/format config, and contribution docs.
//! These map naturally to `procedural` memories. Files are captured whole
//! (bounded) rather than parsed — the agent reads the steps and writes the
//! procedure; we just point it at the right files.

use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use super::{walk_files, CollectOptions, FileSource};

/// Evidence for workflows, processes, and conventions.
#[derive(Debug, Serialize)]
pub struct WorkflowCollection {
    pub ci_workflows: Vec<FileSource>,
    pub scripts: Vec<FileSource>,
    pub conventions: Vec<FileSource>,
    /// Commands declared in `package.json` "scripts" — extracted so the
    /// agent sees them as discrete steps rather than buried in JSON.
    pub package_scripts: Vec<PackageScript>,
}

impl WorkflowCollection {
    pub fn item_count(&self) -> usize {
        self.ci_workflows.len()
            + self.scripts.len()
            + self.conventions.len()
            + self.package_scripts.len()
    }
}

#[derive(Debug, Serialize)]
pub struct PackageScript {
    pub manifest: String,
    pub name: String,
    pub command: String,
}

#[derive(Debug, Clone, Copy)]
enum Category {
    Ci,
    Script,
    Convention,
}

const MAX_PER_BUCKET: usize = 25;

pub fn collect(repo_path: &Path, opts: &CollectOptions) -> Result<WorkflowCollection> {
    let mut ci_workflows = Vec::new();
    let mut scripts = Vec::new();
    let mut conventions = Vec::new();
    let mut package_scripts = Vec::new();

    for path in walk_files(repo_path) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        // package.json: extract scripts, don't duplicate the whole file.
        if name == "package.json" {
            package_scripts.extend(parse_package_scripts(&path, repo_path));
            continue;
        }

        let Some(cat) = classify(&path, &name) else {
            continue;
        };

        let bucket = match cat {
            Category::Ci => &mut ci_workflows,
            Category::Script => &mut scripts,
            Category::Convention => &mut conventions,
        };
        if bucket.len() >= MAX_PER_BUCKET {
            continue;
        }
        if let Ok(src) = FileSource::from_path(&path, repo_path, opts.max_file_bytes) {
            bucket.push(src);
        }
    }

    Ok(WorkflowCollection {
        ci_workflows,
        scripts,
        conventions,
        package_scripts,
    })
}

fn classify(path: &Path, name_lower: &str) -> Option<Category> {
    let s = path.to_string_lossy().to_lowercase();

    // CI pipelines.
    if s.contains(".github/workflows/") && has_ext(path, &["yml", "yaml"]) {
        return Some(Category::Ci);
    }
    if matches!(
        name_lower,
        ".gitlab-ci.yml"
            | "azure-pipelines.yml"
            | "jenkinsfile"
            | ".travis.yml"
            | "bitbucket-pipelines.yml"
    ) {
        return Some(Category::Ci);
    }
    if s.contains("/.circleci/") || s.contains("/.buildkite/") {
        return Some(Category::Ci);
    }

    // Task scripts.
    if matches!(
        name_lower,
        "makefile"
            | "gnu makefile"
            | "justfile"
            | "taskfile.yml"
            | "taskfile.yaml"
            | "rakefile"
            | "cargo-make.toml"
    ) {
        return Some(Category::Script);
    }
    if (s.contains("/scripts/") || s.starts_with("scripts/"))
        && has_ext(path, &["sh", "bash", "zsh", "ps1", "py", "rb", "js", "ts"])
    {
        return Some(Category::Script);
    }
    if matches!(name_lower, "pyproject.toml" | "setup.py" | "tox.ini") {
        return Some(Category::Script);
    }

    // Conventions & lint/format config.
    if matches!(
        name_lower,
        "contributing.md"
            | "code_of_conduct.md"
            | ".editorconfig"
            | ".gitattributes"
            | "commit-convention.md"
            | "commit_convention.md"
    ) {
        return Some(Category::Convention);
    }
    if name_lower.starts_with(".eslintrc")
        || name_lower.starts_with(".prettierrc")
        || name_lower.starts_with(".markdownlint")
    {
        return Some(Category::Convention);
    }
    if matches!(
        name_lower,
        "rustfmt.toml"
            | "clippy.toml"
            | "rust-toolchain.toml"
            | "rust-toolchain"
            | ".rubocop.yml"
            | "biome.json"
            | "deno.json"
            | ".flake8"
            | "pylintrc"
            | ".pylintrc"
            | "go.sum"
    ) {
        // go.sum excluded below; keep the lint ones.
        if name_lower != "go.sum" {
            return Some(Category::Convention);
        }
    }

    None
}

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn parse_package_scripts(path: &Path, root: &Path) -> Vec<PackageScript> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let val: serde_json::Value =
        serde_json::from_str(&text).unwrap_or(serde_json::Value::Object(Default::default()));
    let Some(scripts) = val.get("scripts").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let manifest = super::relpath(path, root);
    scripts
        .iter()
        .map(|(name, cmd)| PackageScript {
            manifest: manifest.clone(),
            name: name.clone(),
            command: cmd.as_str().unwrap_or("").to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_github_workflow_and_makefile() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".github/workflows")).unwrap();
        std::fs::write(root.join(".github/workflows/ci.yml"), "on: [push]\n").unwrap();
        std::fs::write(root.join("Makefile"), "test:\n\tcargo test\n").unwrap();
        std::fs::write(
            root.join("package.json"),
            "{\"scripts\":{\"test\":\"jest\",\"build\":\"tsc\"}}\n",
        )
        .unwrap();
        std::fs::write(root.join("CONTRIBUTING.md"), "# Contributing\n").unwrap();

        let col = collect(root, &CollectOptions::default()).unwrap();
        assert_eq!(col.ci_workflows.len(), 1);
        assert!(col.scripts.iter().any(|s| s.path == "Makefile"));
        assert_eq!(col.package_scripts.len(), 2);
        assert!(col.package_scripts.iter().any(|p| p.name == "test"));
        assert!(col.conventions.iter().any(|c| c.path == "CONTRIBUTING.md"));
    }

    #[test]
    fn ignores_random_yml_outside_workflows() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("docker-compose.yml"), "version: '3'\n").unwrap();
        let col = collect(root, &CollectOptions::default()).unwrap();
        assert!(col.ci_workflows.is_empty());
    }
}
