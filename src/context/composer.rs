use crate::retrieval::bm25::SearchResult;
use std::collections::HashSet;

/// Context budget for controlling memory usage in LLM context window.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub max_tokens: usize,
    pub reserved_for_user: usize,
    pub reserved_for_agent: usize,
    pub reserved_for_memory: usize,
}

impl ContextBudget {
    pub fn new(total_tokens: usize, memory_percent: u8) -> Self {
        let memory_tokens = total_tokens * memory_percent as usize / 100;
        let remaining = total_tokens - memory_tokens;
        Self {
            max_tokens: total_tokens,
            reserved_for_user: remaining / 2,
            reserved_for_agent: remaining / 2,
            reserved_for_memory: memory_tokens,
        }
    }
}

/// Content type for token estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    English,
    Chinese,
    Code,
}

/// Context composer for converting retrieval results into compact high-signal context.
pub struct ContextComposer;

impl Default for ContextComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextComposer {
    pub fn new() -> Self {
        Self
    }

    /// Estimate the number of tokens in a string based on content type.
    /// English: ~4 chars/token
    /// Chinese: ~2 chars/token
    /// Code: ~3 chars/token
    pub fn estimate_tokens(text: &str) -> usize {
        let mut english_chars = 0usize;
        let mut chinese_chars = 0usize;
        let mut other_chars = 0usize;

        for ch in text.chars() {
            if ch.is_ascii() {
                english_chars += 1;
            } else if is_chinese_char(ch) {
                chinese_chars += 1;
            } else {
                other_chars += 1;
            }
        }

        let tokens =
            (english_chars as f64 / 4.0 + chinese_chars as f64 / 2.0 + other_chars as f64 / 3.0)
                .ceil() as usize;

        // 20% safety margin
        let with_margin = (tokens as f64 * 1.2).ceil() as usize;
        with_margin.max(1)
    }

    /// Detect the dominant content type of a string.
    /// Empty strings default to English.
    pub fn detect_content_type(text: &str) -> ContentType {
        if text.is_empty() {
            return ContentType::English;
        }

        let mut english = 0usize;
        let mut chinese = 0usize;

        for ch in text.chars() {
            if ch.is_ascii() {
                english += 1;
            } else if is_chinese_char(ch) {
                chinese += 1;
            }
        }

        if chinese > english / 2 {
            ContentType::Chinese
        } else if english > text.chars().count() / 2 {
            ContentType::English
        } else {
            ContentType::Code
        }
    }

    /// Compose context from search results within a token budget.
    /// Priority: failures > decisions > recent episodic > procedural.
    pub fn compose_context(&self, results: &[SearchResult], budget: &ContextBudget) -> String {
        let mut context_parts = Vec::new();
        let mut tokens_used = 0;
        let token_limit = budget.reserved_for_memory;

        // Sort by priority: failure > decision > episodic > procedural
        let mut sorted = results.to_vec();
        sort_by_priority(&mut sorted);

        // Deduplicate by ID
        let mut seen_ids = HashSet::new();
        let mut deduped = Vec::new();
        for result in sorted {
            if seen_ids.insert(result.id.clone()) {
                deduped.push(result);
            }
        }

        // Build context within budget
        for result in &deduped {
            let entry = format!(
                "[{}] {} (score: {:.2}, {})",
                result.memory_type,
                result.summary,
                result.relevance_score.clamp(0.0, 1.0),
                format_timestamp(result.created_at),
            );

            let entry_tokens = Self::estimate_tokens(&entry);
            if tokens_used + entry_tokens > token_limit {
                break;
            }

            tokens_used += entry_tokens;
            context_parts.push(entry);
        }

        if context_parts.is_empty() {
            String::new()
        } else {
            format!(
                "=== Memory Context ===\n{}\n=== End Memory ===",
                context_parts.join("\n")
            )
        }
    }
}

/// Check if a character is a Chinese character (CJK Unified Ideographs range).
fn is_chinese_char(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{2F800}'..='\u{2FA1F}'
    )
}

/// Sort results by priority: failure > decision > episodic > procedural.
fn sort_by_priority(results: &mut [SearchResult]) {
    results.sort_by(|a, b| {
        let pa = memory_type_priority(&a.memory_type);
        let pb = memory_type_priority(&b.memory_type);
        pa.cmp(&pb).then_with(|| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}

fn memory_type_priority(mt: &str) -> u8 {
    match mt {
        "failure" => 0,
        "decision" => 1,
        "episodic" => 2,
        "procedural" => 3,
        _ => 4,
    }
}

/// Format a Unix timestamp as a human-readable date.
fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_english() {
        let tokens = ContextComposer::estimate_tokens("Hello world test");
        // 16 chars / 4 chars per token = 4, with 20% margin = 5
        assert!(
            (tokens as f64 - 5.0).abs() < 2.0,
            "expected ~5, got {tokens}"
        );
    }

    #[test]
    fn test_estimate_tokens_chinese() {
        let tokens = ContextComposer::estimate_tokens("你好世界测试");
        // 6 chars / 2 chars per token = 3, with 20% margin = 4
        assert!(
            (tokens as f64 - 4.0).abs() < 2.0,
            "expected ~4, got {tokens}"
        );
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        let tokens = ContextComposer::estimate_tokens("Hello 你好 world 世界");
        // 12 ascii + 4 chinese + 3 spaces = 19 total
        // 12/4 + 4/2 + 3/3 = 3+2+1 = 6, with 20% margin = 8
        assert!(
            tokens > 0 && tokens < 15,
            "expected reasonable estimate, got {tokens}"
        );
    }

    #[test]
    fn test_detect_content_type() {
        assert_eq!(
            ContextComposer::detect_content_type("Hello world"),
            ContentType::English
        );
        assert_eq!(
            ContextComposer::detect_content_type("你好世界测试代码"),
            ContentType::Chinese
        );
    }

    #[test]
    fn test_compose_context_within_budget() {
        let composer = ContextComposer::new();
        let budget = ContextBudget::new(10000, 15);

        let results: Vec<SearchResult> = (0..20)
            .map(|i| SearchResult {
                id: format!("id-{i}"),
                memory_type: if i < 5 {
                    "failure".into()
                } else if i < 10 {
                    "decision".into()
                } else {
                    "episodic".into()
                },
                summary: format!("Test result number {i}"),
                relevance_score: 0.5 + (i as f32 * 0.02),
                importance: 0.5,
                created_at: 1716940800 + i as i64 * 1000,
            })
            .collect();

        let context = composer.compose_context(&results, &budget);
        assert!(!context.is_empty());
        assert!(context.starts_with("=== Memory Context ==="));
        assert!(context.ends_with("=== End Memory ==="));
    }

    #[test]
    fn test_compose_context_priority_ordering() {
        let composer = ContextComposer::new();
        let budget = ContextBudget::new(100000, 50);

        let results = vec![
            SearchResult {
                id: "1".into(),
                memory_type: "procedural".into(),
                summary: "deployment workflow".into(),
                relevance_score: 0.9,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "2".into(),
                memory_type: "failure".into(),
                summary: "auth outage".into(),
                relevance_score: 0.5,
                importance: 0.5,
                created_at: 2000,
            },
            SearchResult {
                id: "3".into(),
                memory_type: "decision".into(),
                summary: "use redis".into(),
                relevance_score: 0.7,
                importance: 0.5,
                created_at: 1500,
            },
        ];

        let context = composer.compose_context(&results, &budget);
        // Failure should come first
        let failure_pos = context.find("failure").unwrap();
        let decision_pos = context.find("decision").unwrap();
        let procedural_pos = context.find("procedural").unwrap();
        assert!(failure_pos < decision_pos);
        assert!(decision_pos < procedural_pos);
    }

    #[test]
    fn test_deduplication() {
        let composer = ContextComposer::new();
        let budget = ContextBudget::new(100000, 50);

        let results = vec![
            SearchResult {
                id: "dup".into(),
                memory_type: "failure".into(),
                summary: "first copy".into(),
                relevance_score: 0.9,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "dup".into(),
                memory_type: "failure".into(),
                summary: "duplicate".into(),
                relevance_score: 0.8,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "unique".into(),
                memory_type: "decision".into(),
                summary: "unique entry".into(),
                relevance_score: 0.7,
                importance: 0.5,
                created_at: 2000,
            },
        ];

        let context = composer.compose_context(&results, &budget);
        // Should only contain 2 entries (deduped)
        let count = context.matches("[failure]").count() + context.matches("[decision]").count();
        assert_eq!(count, 2);
    }
}
