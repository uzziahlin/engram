use crate::models::MemoryIntent;
use std::collections::HashMap;

/// Rule-based intent classifier using keyword matching.
///
/// Maps keywords to intents. Supports compound intents
/// (e.g., Debugging + Incident).
pub struct IntentClassifier {
    keyword_map: HashMap<String, Vec<MemoryIntent>>,
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl IntentClassifier {
    pub fn new() -> Self {
        let mut keyword_map: HashMap<String, Vec<MemoryIntent>> = HashMap::new();

        // Debugging keywords
        for kw in &[
            "bug",
            "debug",
            "fix",
            "error",
            "crash",
            "traceback",
            "exception",
            "stack trace",
            "segfault",
            "panic",
            "assertion",
            "breakpoint",
            "调试",
            "修复",
            "错误",
            "崩溃",
            "异常",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Debugging]);
        }

        // Architecture keywords
        for kw in &[
            "architect",
            "design",
            "decision",
            "structure",
            "pattern",
            "module",
            "component",
            "service",
            "microservice",
            "why",
            "rationale",
            "tradeoff",
            "架构",
            "设计",
            "决策",
            "模块",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Architecture]);
        }

        // Workflow keywords
        for kw in &[
            "workflow",
            "process",
            "procedure",
            "pipeline",
            "ci",
            "cd",
            "build",
            "test",
            "lint",
            "step",
            "工作流",
            "流程",
            "构建",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Workflow]);
        }

        // Refactor keywords
        for kw in &[
            "refactor",
            "rewrite",
            "restructure",
            "clean up",
            "simplify",
            "consolidate",
            "merge",
            "split",
            "extract",
            "重写",
            "简化",
            "提取",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Refactor]);
        }

        // Deployment keywords
        for kw in &[
            "deploy",
            "release",
            "rollout",
            "production",
            "staging",
            "rollback",
            "migration",
            "upgrade",
            "config change",
            "发布",
            "上线",
            "回滚",
            "迁移",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Deployment]);
        }

        // Incident keywords
        for kw in &[
            "outage",
            "incident",
            "sev1",
            "sev2",
            "sev3",
            "p0",
            "p1",
            "p2",
            "downtime",
            "alert",
            "page",
            "post-mortem",
            "postmortem",
            "故障",
            "停机",
            "告警",
        ] {
            keyword_map.insert((*kw).to_lowercase(), vec![MemoryIntent::Incident]);
        }

        Self { keyword_map }
    }

    /// Classify the query into one or more intents.
    /// Returns compound intents if multiple keyword groups match.
    /// Falls back to [General] if no keywords match.
    pub fn classify(&self, query: &str) -> Vec<MemoryIntent> {
        let lower = query.to_lowercase();

        // Identifier-like queries (a single token joined by `-`/`_`, e.g. tags,
        // file names, symbols like "unique-tag-test-xyz") are precise lookups
        // meant to span all memory types. Don't let a substring keyword (e.g.
        // "test") route them away from a type the user may want — only natural-
        // language queries (whitespace-separated prose) go through intent
        // routing.
        if Self::is_identifier_like(&lower) {
            return vec![MemoryIntent::General];
        }

        let mut intents = Vec::new();

        for (keyword, intent_list) in &self.keyword_map {
            if Self::keyword_matches(&lower, keyword.as_str()) {
                for intent in intent_list {
                    if !intents.contains(intent) {
                        intents.push(intent.clone());
                    }
                }
            }
        }

        if intents.is_empty() {
            vec![MemoryIntent::General]
        } else {
            intents
        }
    }

    /// Match `keyword` against (lowercased) `text`.
    ///
    /// ASCII keywords require word boundaries on both sides, so `fix` no longer
    /// matches `fixture`/`suffix`. CJK keywords fall back to plain `contains` —
    /// CJK text has no whitespace tokenization, so a boundary check would
    /// wrongly reject valid matches like `修复` inside `修复认证模块`.
    fn keyword_matches(text: &str, keyword: &str) -> bool {
        // Non-ASCII (or mixed) keywords keep substring semantics.
        let is_ascii_token = keyword
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b' ' || b == b'-');
        if !is_ascii_token {
            return text.contains(keyword);
        }

        // ASCII token: both sides must be a non-letter or the string edge.
        let kb = keyword.as_bytes();
        let tb = text.as_bytes();
        let mut from = 0usize;
        while let Some(rel) = text[from..].find(keyword) {
            let start = from + rel;
            let end = start + kb.len();
            let left_ok = start == 0 || !tb[start - 1].is_ascii_alphabetic();
            let right_ok = end >= tb.len() || !tb[end].is_ascii_alphabetic();
            if left_ok && right_ok {
                return true;
            }
            from = end;
        }
        false
    }

    /// Whether `s` looks like an identifier (tag/file/symbol) rather than prose.
    ///
    /// True for a single whitespace-free token containing `-` or `_` joining
    /// alphanumeric segments (e.g. "unique-tag-test-xyz", "auth_utils"). Such
    /// queries are exact lookups and should hit every memory type, not be
    /// narrowed by a coincidental keyword substring.
    fn is_identifier_like(s: &str) -> bool {
        !s.chars().any(|c| c.is_whitespace())
            && (s.contains('-') || s.contains('_'))
            && s.chars().any(|c| c.is_ascii_alphanumeric())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugging_intent() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("fix the auth bug");
        assert!(intents.contains(&MemoryIntent::Debugging));
    }

    #[test]
    fn test_architecture_intent() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("why was Redis chosen for caching");
        assert!(intents.contains(&MemoryIntent::Architecture));
    }

    #[test]
    fn test_compound_intent() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("debug the production outage incident");
        assert!(intents.contains(&MemoryIntent::Debugging));
        assert!(intents.contains(&MemoryIntent::Incident));
    }

    #[test]
    fn test_fallback_to_general() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("show me the code");
        assert_eq!(intents, vec![MemoryIntent::General]);
    }

    #[test]
    fn test_chinese_keywords() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("修复认证模块的错误");
        assert!(intents.contains(&MemoryIntent::Debugging));
    }

    #[test]
    fn test_deployment_intent() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("deploy to production");
        assert!(intents.contains(&MemoryIntent::Deployment));
    }

    #[test]
    fn test_workflow_intent() {
        let classifier = IntentClassifier::new();
        let intents = classifier.classify("what is the CI pipeline process");
        assert!(intents.contains(&MemoryIntent::Workflow));
    }

    #[test]
    fn test_word_boundary_rejects_embedded_substrings() {
        let classifier = IntentClassifier::new();
        // "fix" inside "fixture"/"suffix" must NOT trigger Debugging.
        let intents = classifier.classify("update the test fixture");
        assert!(
            !intents.contains(&MemoryIntent::Debugging),
            "fixture must not match 'fix'"
        );
        let intents = classifier.classify("trim the suffix array");
        assert!(!intents.contains(&MemoryIntent::Debugging));
        // "merge" inside "emerge" must NOT trigger Refactor.
        let intents = classifier.classify("issues start to emerge here");
        assert!(!intents.contains(&MemoryIntent::Refactor));
    }

    #[test]
    fn test_word_boundary_matches_isolated_token() {
        let classifier = IntentClassifier::new();
        // Isolated or space-delimited "fix" still matches.
        assert!(classifier
            .classify("I need to fix this bug")
            .contains(&MemoryIntent::Debugging));
        assert!(classifier
            .classify("fix")
            .contains(&MemoryIntent::Debugging));
    }

    #[test]
    fn test_cjk_keyword_still_matches_mid_sentence() {
        let classifier = IntentClassifier::new();
        // CJK keywords keep substring semantics — match inside a sentence.
        assert!(classifier
            .classify("如何修复认证模块")
            .contains(&MemoryIntent::Debugging));
    }

    #[test]
    fn test_identifier_like_query_routes_to_general() {
        let classifier = IntentClassifier::new();
        // Tags / symbols / filenames are precise lookups across all types —
        // a coincidental keyword substring ("test" inside "unique-tag-test-xyz")
        // must not narrow the search to Workflow (which excludes Decision).
        for q in [
            "unique-tag-test-xyz",
            "auth_utils",
            "e2e-test",
            "fix-auth-bug",
        ] {
            let intents = classifier.classify(q);
            assert_eq!(
                intents,
                vec![MemoryIntent::General],
                "identifier-like query {q:?} should route to General, got {intents:?}"
            );
        }
    }

    #[test]
    fn test_prose_query_still_routes_by_keyword() {
        let classifier = IntentClassifier::new();
        // Natural-language queries (whitespace-separated prose) still route —
        // "test" in a sentence legitimately suggests Workflow.
        assert!(classifier
            .classify("how to test the build")
            .contains(&MemoryIntent::Workflow));
        // A bare keyword with no separator still routes (not identifier-like).
        assert!(classifier.classify("test").contains(&MemoryIntent::Workflow));
    }
}
