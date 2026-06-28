use crate::models::FailureMemory;
use crate::storage::{MemoryRepository, ReflectionSuggestionRow};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Reflection engine: scans active failures for recurring patterns and proposes
/// preventive procedural rules awaiting human confirmation.
///
/// Minimal closed loop (roadmap 2.2): group active failures by tag; when a tag
/// recurs at least `min_occurrences` times, distill the group's `prevention`
/// fields into a draft procedural rule. Proposals live in the separate
/// `reflection_suggestions` table — invisible to `search_procedural` — until
/// confirmed, when [`MemoryRepository::confirm_suggestion`] promotes them into
/// `procedural_memories`.
pub struct ReflectionEngine {
    /// Minimum active failures sharing a tag before a rule is proposed.
    pub min_occurrences: usize,
}

/// One proposed preventive rule, distilled from a tag group of failures.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SuggestedRule {
    pub pattern_tag: String,
    pub source_failure_ids: Vec<String>,
    /// Source `prevention` texts (de-duplicated, order-preserving) — shown to
    /// the reviewer as the evidence behind the proposed rule.
    pub source_preventions: Vec<String>,
    pub occurrence_count: usize,
    pub suggested_workflow_name: String,
    /// De-duplicated prevention texts, promoted as the procedural rule's steps.
    pub suggested_steps: Vec<String>,
    pub suggested_tags: Vec<String>,
}

/// Outcome of a reflection pass. `suggestions` is always populated (for dry-run
/// preview); `created` counts proposals actually written when `apply = true`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReflectionPlan {
    pub suggestions: Vec<SuggestedRule>,
    pub created: usize,
}

impl Default for ReflectionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionEngine {
    /// Default threshold of 3 (matches `[reflection].min_occurrences`).
    pub fn new() -> Self {
        Self { min_occurrences: 3 }
    }

    pub fn with_min_occurrences(min_occurrences: usize) -> Self {
        Self { min_occurrences }
    }

    /// Scan a project's active failures and propose preventive rules for any
    /// tag recurring at least `min_occurrences` times.
    ///
    /// When `apply` is false this is a dry run: `suggestions` lists what *would*
    /// be proposed and nothing is written. When `apply` is true, each proposal
    /// is persisted (skipping any `(project, tag)` that already has a pending
    /// proposal, so repeated runs stay idempotent), and `created` reflects the
    /// number actually inserted.
    pub fn reflect(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
        apply: bool,
        now: i64,
    ) -> Result<ReflectionPlan> {
        let failures = repo.list_active_failures_for_reflection(project_id)?;

        // Group failures by each of their tags (a failure with N tags joins N groups).
        let mut by_tag: HashMap<&str, Vec<&FailureMemory>> = HashMap::new();
        for f in &failures {
            for tag in &f.tags {
                if !tag.trim().is_empty() {
                    by_tag.entry(tag.as_str()).or_default().push(f);
                }
            }
        }

        // Deterministic output: by descending occurrence, then by tag alphabetically.
        let mut groups: Vec<(&str, Vec<&FailureMemory>)> = by_tag.into_iter().collect();
        groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));

        let mut suggestions = Vec::new();
        let mut created = 0;
        for (tag, group) in groups {
            if group.len() < self.min_occurrences {
                continue;
            }
            let source_failure_ids: Vec<String> = group.iter().map(|f| f.id.clone()).collect();
            let source_preventions =
                dedup_preserve_order(group.iter().map(|f| f.prevention.clone()));
            let rule = SuggestedRule {
                pattern_tag: tag.to_string(),
                occurrence_count: group.len(),
                suggested_workflow_name: format!("Prevent recurring {tag} failures"),
                suggested_steps: source_preventions.clone(),
                suggested_tags: vec![
                    tag.to_string(),
                    "reflection".into(),
                    "auto-generated".into(),
                ],
                source_failure_ids,
                source_preventions,
            };

            if apply && !repo.has_pending_suggestion(project_id, &rule.pattern_tag)? {
                let row = ReflectionSuggestionRow {
                    id: uuid::Uuid::new_v4().to_string(),
                    project_id: project_id.to_string(),
                    pattern_tag: rule.pattern_tag.clone(),
                    source_failure_ids: rule.source_failure_ids.clone(),
                    source_preventions: rule.source_preventions.clone(),
                    occurrence_count: rule.occurrence_count as i64,
                    suggested_workflow_name: rule.suggested_workflow_name.clone(),
                    suggested_steps: rule.suggested_steps.clone(),
                    suggested_tags: rule.suggested_tags.clone(),
                    status: "pending".into(),
                    created_at: now,
                    resolved_at: None,
                };
                repo.insert_reflection_suggestion(&row)?;
                created += 1;
            }
            suggestions.push(rule);
        }

        Ok(ReflectionPlan {
            suggestions,
            created,
        })
    }
}

/// De-duplicate while preserving first-seen order; drop empty/whitespace items.
fn dedup_preserve_order(items: impl Iterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    items
        .filter(|s| !s.trim().is_empty())
        .filter(|s| seen.insert(s.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FailureMemory;
    use crate::storage::MemoryRepository;

    fn setup() -> MemoryRepository {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo
    }

    fn failure(repo: &MemoryRepository, id: &str, tag: &str, prevention: &str) {
        repo.create_failure(&FailureMemory {
            id: id.into(),
            project_id: "p".into(),
            incident: format!("incident {id}"),
            root_cause: "rc".into(),
            fix: "fix".into(),
            prevention: prevention.into(),
            severity: 3,
            tags: vec![tag.into()],
            created_at: 0,
            updated_at: 0,
        })
        .unwrap();
    }

    #[test]
    fn below_threshold_proposes_nothing() {
        let repo = setup();
        failure(&repo, "f1", "fts5", "sanitize");
        failure(&repo, "f2", "fts5", "sanitize");
        let plan = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", false, 100)
            .unwrap();
        assert!(plan.suggestions.is_empty());
        assert_eq!(plan.created, 0);
    }

    #[test]
    fn at_threshold_proposes_one_rule_per_recurring_tag() {
        let repo = setup();
        for i in 1..=3 {
            failure(&repo, &format!("f{i}"), "fts5", "sanitize query");
        }
        for i in 1..=3 {
            failure(&repo, &format!("a{i}"), "auth", "rotate token");
        }
        let plan = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", false, 100)
            .unwrap();
        // Two recurring tags → two proposals, ordered by occurrence (tie) then tag.
        let tags: Vec<_> = plan
            .suggestions
            .iter()
            .map(|s| s.pattern_tag.as_str())
            .collect();
        assert_eq!(tags, vec!["auth", "fts5"]);

        let fts = plan
            .suggestions
            .iter()
            .find(|s| s.pattern_tag == "fts5")
            .unwrap();
        assert_eq!(fts.occurrence_count, 3);
        assert_eq!(fts.source_failure_ids.len(), 3);
        assert_eq!(fts.suggested_steps, vec!["sanitize query".to_string()]);
        assert!(fts.suggested_tags.contains(&"reflection".to_string()));
        assert!(fts.suggested_tags.contains(&"auto-generated".to_string()));
    }

    #[test]
    fn dedups_repeated_prevention_into_distinct_steps() {
        let repo = setup();
        failure(&repo, "f1", "fts5", "sanitize input");
        failure(&repo, "f2", "fts5", "sanitize input"); // duplicate
        failure(&repo, "f3", "fts5", "wrap as phrase query");
        let plan = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", false, 100)
            .unwrap();
        let rule = &plan.suggestions[0];
        assert_eq!(
            rule.suggested_steps.len(),
            2,
            "duplicate prevention collapsed"
        );
        assert_eq!(rule.source_preventions.len(), 2);
    }

    #[test]
    fn dry_run_writes_nothing() {
        let repo = setup();
        for i in 1..=3 {
            failure(&repo, &format!("f{i}"), "fts5", "sanitize");
        }
        let plan = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", false, 100)
            .unwrap();
        assert_eq!(plan.suggestions.len(), 1);
        assert_eq!(plan.created, 0);
        assert!(!repo.has_pending_suggestion("p", "fts5").unwrap());
    }

    #[test]
    fn apply_writes_pending_and_is_idempotent() {
        let repo = setup();
        for i in 1..=3 {
            failure(&repo, &format!("f{i}"), "fts5", "sanitize");
        }
        let p1 = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", true, 100)
            .unwrap();
        assert_eq!(p1.created, 1);
        assert!(repo.has_pending_suggestion("p", "fts5").unwrap());
        assert_eq!(repo.list_pending_suggestions("p").unwrap().len(), 1);

        // Second run: a pending proposal already exists → no new proposal written.
        let p2 = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", true, 200)
            .unwrap();
        assert_eq!(p2.created, 0);
        assert_eq!(repo.list_pending_suggestions("p").unwrap().len(), 1);
    }

    #[test]
    fn ignores_failures_without_tags() {
        let repo = setup();
        for i in 1..=3 {
            repo.create_failure(&FailureMemory {
                id: format!("f{i}"),
                project_id: "p".into(),
                incident: "x".into(),
                root_cause: "y".into(),
                fix: "z".into(),
                prevention: "p".into(),
                severity: 1,
                tags: vec![],
                created_at: 0,
                updated_at: 0,
            })
            .unwrap();
        }
        let plan = ReflectionEngine::with_min_occurrences(3)
            .reflect(&repo, "p", false, 100)
            .unwrap();
        assert!(plan.suggestions.is_empty());
    }
}
