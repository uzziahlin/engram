//! Architecture-decisions collector.
//!
//! Gathers the carriers where "why" knowledge lives: markdown docs (README,
//! decision records, agent guidelines), decision-flavored code annotations
//! (`// WHY:` / `//!` module docs), and dependency manifests (which encode
//! technology choices). The agent distills these into `decision` memories.

use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use super::{walk_files, CollectOptions, FileSource};

/// Evidence for architectural/design decisions.
#[derive(Debug, Serialize)]
pub struct DocsCollection {
    pub documents: Vec<DocumentExcerpt>,
    pub annotations: Vec<CodeAnnotation>,
    pub dependencies: Vec<DependencyManifest>,
}

impl DocsCollection {
    pub fn item_count(&self) -> usize {
        self.documents.len() + self.annotations.len() + self.dependencies.len()
    }
}

/// A documentation file captured for the agent to mine for rationale.
#[derive(Debug, Serialize)]
pub struct DocumentExcerpt {
    pub path: String,
    pub content: String,
    pub truncated: bool,
    pub bytes: usize,
    /// Why this doc was selected: `readme`, `decision-record`,
    /// `agent-guidelines`, `docs`, or `markdown`.
    pub kind: String,
}

/// A single decision-flavored comment line with its location.
#[derive(Debug, Serialize)]
pub struct CodeAnnotation {
    pub file: String,
    pub line: usize,
    pub kind: String,
    pub text: String,
}

/// Parsed dependencies from one manifest file.
#[derive(Debug, Serialize)]
pub struct DependencyManifest {
    pub manifest: String,
    pub manager: String,
    pub dependencies: Vec<Dependency>,
}

#[derive(Debug, Serialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
}

/// File extensions whose comments may carry decision annotations.
const SOURCE_EXTS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "go", "py", "java", "kt", "rb", "c", "cpp", "h", "hpp",
    "swift", "m", "mm", "scala", "clj", "ex", "exs", "erl", "hs", "lua", "php", "sql", "sh", "vim",
    "elm",
];

/// Comment keywords that signal a decision-flavored note worth recording.
const ANNOTATION_KEYWORDS: &[&str] = &[
    "WHY", "NOTE", "HACK", "DECISION", "REASON", "WARNING", "XXX", "TODO",
];

const MAX_DOCUMENTS: usize = 40;
const MAX_ANNOTATIONS: usize = 200;

pub fn collect(repo_path: &Path, opts: &CollectOptions) -> Result<DocsCollection> {
    let files = walk_files(repo_path);

    let mut documents = Vec::new();
    let mut annotations = Vec::new();
    let mut dependencies = Vec::new();

    for path in &files {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Dependency manifests.
        if name == "Cargo.toml" {
            if let Ok(manifest) = parse_cargo_toml(path, repo_path) {
                dependencies.push(manifest);
            }
            continue;
        }
        if name == "package.json" {
            if let Ok(manifest) = parse_package_json(path, repo_path) {
                dependencies.push(manifest);
            }
            continue;
        }
        if name == "go.mod" {
            if let Ok(manifest) = parse_go_mod(path, repo_path) {
                dependencies.push(manifest);
            }
            continue;
        }

        // Markdown documents.
        if ext == "md" || ext == "markdown" {
            if documents.len() < MAX_DOCUMENTS {
                if let Ok(src) = FileSource::from_path(path, repo_path, opts.max_file_bytes) {
                    documents.push(DocumentExcerpt {
                        kind: doc_kind(path).to_string(),
                        path: src.path,
                        content: src.content,
                        truncated: src.truncated,
                        bytes: src.bytes,
                    });
                }
            }
            continue;
        }

        // Decision annotations in source files.
        if SOURCE_EXTS.contains(&ext) && annotations.len() < MAX_ANNOTATIONS {
            scan_annotations(path, repo_path, opts.max_file_bytes, &mut annotations);
        }
    }

    Ok(DocsCollection {
        documents,
        annotations,
        dependencies,
    })
}

/// Classify a markdown file by its role so the agent can prioritize.
fn doc_kind(path: &Path) -> &'static str {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let s = path.to_string_lossy();
    match name {
        "README.md" | "README.markdown" => "readme",
        "CLAUDE.md" | "AGENTS.md" | "GEMINI.md" | "COPILOT.md" => "agent-guidelines",
        "DECISIONS.md" | "DECISION.md" => "decision-record",
        "CONTRIBUTING.md" => "contributing",
        "ARCHITECTURE.md" => "architecture",
        _ if s.contains("/ADR/") || s.contains("/adr/") || s.contains("/decisions/") => {
            "decision-record"
        }
        _ if s.contains("/docs/") => "docs",
        _ => "markdown",
    }
}

/// Scan a source file for decision-flavored comments and module docs.
fn scan_annotations(path: &Path, root: &Path, max_bytes: usize, out: &mut Vec<CodeAnnotation>) {
    // Bound the read so a generated/minified file can't exhaust memory.
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let slice = if bytes.len() > max_bytes {
        &bytes[..max_bytes]
    } else {
        &bytes[..]
    };
    let text = String::from_utf8_lossy(slice);
    let rel = super::relpath(path, root);

    for (i, line) in text.lines().enumerate() {
        if let Some((kind, text_part)) = classify_comment_line(line) {
            if text_part.is_empty() {
                continue;
            }
            out.push(CodeAnnotation {
                file: rel.clone(),
                line: i + 1,
                kind: kind.to_string(),
                text: text_part.to_string(),
            });
            if out.len() >= MAX_ANNOTATIONS {
                return;
            }
        }
    }
}

/// Return `(kind, text)` if a line is a module doc or a labeled decision note.
/// `kind` is a static keyword; `text` borrows from the input line.
fn classify_comment_line(line: &str) -> Option<(&'static str, &str)> {
    let t = line.trim_start();

    // Rust module-level doc: `//!` — usually carries architectural rationale.
    if let Some(rest) = t.strip_prefix("//!") {
        let text = rest.trim_start_matches('/').trim();
        if !text.is_empty() {
            return Some(("module-doc", text));
        }
        return None;
    }

    // Line comment `// KEYWORD ...` (language-agnostic).
    let rest = t.strip_prefix("//")?.trim_start();
    for kw in ANNOTATION_KEYWORDS {
        if let Some(after) = rest.strip_prefix(kw) {
            // Require a delimiter so prefix-collisions (e.g. "NOTES" vs "NOTE")
            // are avoided, but allow the keyword to end the line.
            if after.is_empty() || after.starts_with(':') || after.starts_with(' ') {
                let text = after.trim_start_matches(':').trim();
                if !text.is_empty() {
                    return Some((kw_lower(kw), text));
                }
            }
        }
    }
    None
}

fn kw_lower(kw: &str) -> &'static str {
    match kw {
        "WHY" => "why",
        "NOTE" => "note",
        "HACK" => "hack",
        "DECISION" => "decision",
        "REASON" => "reason",
        "WARNING" => "warning",
        "XXX" => "xxx",
        "TODO" => "todo",
        _ => "note",
    }
}

fn parse_cargo_toml(path: &Path, root: &Path) -> Result<DependencyManifest> {
    let text = std::fs::read_to_string(path)?;
    let val: toml::Value = toml::from_str(&text).unwrap_or(toml::Value::Table(Default::default()));
    let mut deps = Vec::new();
    for section in &["dependencies", "dev-dependencies"] {
        if let Some(table) = val.get(section).and_then(|v| v.as_table()) {
            for (name, spec) in table {
                deps.push(Dependency {
                    name: name.clone(),
                    version: cargo_dep_version(spec),
                });
            }
        }
    }
    Ok(DependencyManifest {
        manifest: super::relpath(path, root),
        manager: "cargo".into(),
        dependencies: deps,
    })
}

fn cargo_dep_version(spec: &toml::Value) -> String {
    match spec {
        toml::Value::String(s) => s.clone(),
        toml::Value::Table(t) => t
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

fn parse_package_json(path: &Path, root: &Path) -> Result<DependencyManifest> {
    let text = std::fs::read_to_string(path)?;
    let val: serde_json::Value =
        serde_json::from_str(&text).unwrap_or(serde_json::Value::Object(Default::default()));
    let mut deps = Vec::new();
    for section in &["dependencies", "devDependencies"] {
        if let Some(obj) = val.get(section).and_then(|v| v.as_object()) {
            for (name, ver) in obj {
                deps.push(Dependency {
                    name: name.clone(),
                    version: ver.as_str().unwrap_or("").to_string(),
                });
            }
        }
    }
    Ok(DependencyManifest {
        manifest: super::relpath(path, root),
        manager: "npm".into(),
        dependencies: deps,
    })
}

fn parse_go_mod(path: &Path, root: &Path) -> Result<DependencyManifest> {
    let text = std::fs::read_to_string(path)?;
    let mut deps = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        // Single-line: `require github.com/x/y v1.2.3`
        let r = t
            .strip_prefix("require ")
            .map(str::trim)
            .unwrap_or_else(|| {
                // Inside a `require ( ... )` block, lines are just `path version`.
                if t.starts_with("github.com/")
                    || t.starts_with("golang.org/")
                    || t.starts_with("gopkg.in/")
                {
                    t
                } else {
                    ""
                }
            });
        if r.is_empty() {
            continue;
        }
        let mut parts = r.split_whitespace();
        if let (Some(name), Some(ver)) = (parts.next(), parts.next()) {
            deps.push(Dependency {
                name: name.to_string(),
                version: ver.to_string(),
            });
        }
    }
    Ok(DependencyManifest {
        manifest: super::relpath(path, root),
        manager: "go".into(),
        dependencies: deps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_rust_module_doc() {
        let (k, t) =
            classify_comment_line("//! No async runtime — SQLite is local and sync").unwrap();
        assert_eq!(k, "module-doc");
        assert!(t.contains("No async runtime"));
    }

    #[test]
    fn classify_labeled_note() {
        let (k, t) =
            classify_comment_line("    // WHY: gix avoids libgit2 dynamic linking").unwrap();
        assert_eq!(k, "why");
        assert!(t.contains("gix"));
    }

    #[test]
    fn classify_rejects_plain_code() {
        assert!(classify_comment_line("let x = 5;").is_none());
        assert!(classify_comment_line("// ").is_none());
    }

    #[test]
    fn collects_markdown_and_cargo_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            "[dependencies]\nserde = \"1\"\ntokio = { version = \"1\", features = [\"full\"] }\n",
        )
        .unwrap();
        std::fs::write(root.join("README.md"), "# Project\n").unwrap();
        std::fs::create_dir(root.join("target")).unwrap(); // must be ignored

        let col = collect(root, &CollectOptions::default()).unwrap();
        assert_eq!(col.dependencies.len(), 1);
        assert_eq!(col.dependencies[0].manager, "cargo");
        assert_eq!(col.dependencies[0].dependencies.len(), 2);
        assert!(col.documents.iter().any(|d| d.kind == "readme"));
    }

    #[test]
    fn scans_decision_annotations() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("a.rs"),
            "//! Module-level rationale.\nfn main() {\n    // WHY: avoid allocation\n}\n",
        )
        .unwrap();
        let col = collect(root, &CollectOptions::default()).unwrap();
        assert!(col.annotations.iter().any(|a| a.kind == "module-doc"));
        assert!(col.annotations.iter().any(|a| a.kind == "why"));
    }
}
