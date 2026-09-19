//! Retrieval-quality evaluation harness: a golden corpus + golden query set
//! scored against the REAL search pipeline (intent → BM25 → rerank).
//!
//! The ranking math has been rebuilt several times (v0.3 unified ordering,
//! v0.4 ranking rebuild, v0.5 cross-type BM25 calibration) and each change
//! was judged by hand. This module makes retrieval quality measurable:
//!
//! - `seed_corpus` writes a fixed, realistic engineering-knowledge corpus
//!   (deterministic ids, timestamps relative to `now`) into a repository.
//! - `golden_cases()` returns queries — English and Chinese, across intents —
//!   each with the memory ids a *good* ranking should surface, best first.
//! - `run_eval` pushes every query through `DefaultMemoryProvider::search_core`
//!   and scores the outcome with MRR / NDCG@k / primary-hit@3.
//!
//! `engram eval` runs the same harness from the CLI; the unit tests at the
//! bottom pin the current quality as a regression baseline — an intentional
//! ranking change updates the thresholds (or the corpus) consciously.

use crate::mcp::server::DefaultMemoryProvider;
use crate::models::{DecisionMemory, EpisodicMemory, FailureMemory, ProceduralMemory};
use crate::storage::MemoryRepository;

use anyhow::Result;

/// Project id the golden corpus lives in. Isolated on purpose: the harness
/// never touches a real database (the CLI seeds an in-memory repo).
pub const EVAL_PROJECT: &str = "engram-eval";

/// One golden query: `expected` lists memory ids a good ranking surfaces,
/// best match first (grades for NDCG descend with position).
pub struct GoldenCase {
    pub id: &'static str,
    pub query: &'static str,
    pub expected: &'static [&'static str],
}

/// The golden query set. Kept deliberately small and sharp: each case targets
/// one retrieval behavior (cross-type ranking, intent keywords, CJK, filters
/// are exercised in unit tests elsewhere). When a ranking change moves a case
/// here, decide whether the case or the change is wrong — that judgment is
/// the whole point of this harness.
pub fn golden_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase {
            id: "failure-fts5-keywords",
            query: "FTS5 crash on SQL keywords",
            expected: &["f-fts5-keywords"],
        },
        GoldenCase {
            id: "failure-cjk-backfill",
            query: "Chinese search returns nothing",
            expected: &["f-cjk-backfill"],
        },
        GoldenCase {
            id: "failure-wal-growth",
            query: "database file keeps growing WAL",
            expected: &["f-wal-growth"],
        },
        GoldenCase {
            id: "decision-storage",
            query: "why SQLite instead of Postgres",
            expected: &["d-sqlite-storage"],
        },
        GoldenCase {
            id: "decision-no-async",
            query: "why no tokio async runtime",
            expected: &["d-no-async"],
        },
        GoldenCase {
            id: "decision-embeddings",
            query: "embedding model choice local inference",
            expected: &["d-candle-embeddings"],
        },
        GoldenCase {
            id: "episodic-auth-refactor",
            query: "authentication refactor login session",
            expected: &["e-auth-refactor"],
        },
        GoldenCase {
            id: "episodic-http-transport",
            query: "HTTP transport bearer token",
            expected: &["e-http-transport"],
        },
        GoldenCase {
            id: "procedural-release",
            query: "release 发布流程",
            expected: &["p-release"],
        },
        GoldenCase {
            id: "procedural-review",
            query: "code review 代码评审 checklist",
            expected: &["p-review"],
        },
        GoldenCase {
            id: "procedural-incident",
            query: "incident response steps",
            expected: &["p-incident"],
        },
        GoldenCase {
            id: "cross-type-migration",
            // Both a failure (the incident) and the decision that resolved the
            // strategy are relevant; the failure is the stronger match.
            query: "migration broke search index",
            expected: &["f-cjk-backfill", "d-sqlite-storage"],
        },
    ]
}

/// Seed the golden corpus into `repo`. Ids are fixed (`f-*`/`d-*`/`p-*`/`e-*`)
/// so golden cases can name them; timestamps are `now - age_days*86400` so the
/// reranker's recency decay sees a stable relative ordering regardless of when
/// the harness runs.
pub fn seed_corpus(repo: &MemoryRepository, now: i64) -> Result<()> {
    let ts = |age_days: i64| now - age_days * 86400;

    let failures = [
        FailureMemory {
            id: "f-fts5-keywords".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "FTS5 search crashed when the query contained SQL keywords like UNIQUE"
                .into(),
            root_cause: "Raw user input flowed into the MATCH expression; UNIQUE and AND were \
                         parsed as column filters, not text terms"
                .into(),
            fix: "Escape the query into FTS5 phrase syntax before matching".into(),
            prevention: "Always route search terms through the phrase-quoting sanitizer".into(),
            severity: 4,
            tags: vec!["search".into(), "fts5".into()],
            created_at: ts(9),
            updated_at: ts(9),
        },
        FailureMemory {
            id: "f-cjk-backfill".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "Chinese search returned nothing after the schema migration".into(),
            root_cause: "The FTS tags backfill skipped CJK preprocessing, so migrated Chinese \
                         text was indexed as one unmatchable token"
                .into(),
            fix: "Rebuild the FTS index through the same preprocessing as the write path".into(),
            prevention: "Migrations that rewrite indexed text must reuse the write-path \
                         preprocessing, never a raw SQL copy"
                .into(),
            severity: 3,
            tags: vec!["search".into(), "migration".into(), "cjk".into()],
            created_at: ts(21),
            updated_at: ts(21),
        },
        FailureMemory {
            id: "f-wal-growth".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "The database file and its WAL kept growing without bound over months \
                       of use"
                .into(),
            root_cause: "Archived rows kept their FTS and embedding payload; WAL was never \
                         checkpointed"
                .into(),
            fix: "Periodic gc + wal_checkpoint(TRUNCATE) via the maintain command".into(),
            prevention: "Schedule a weekly maintenance pass".into(),
            severity: 2,
            tags: vec!["storage".into(), "ops".into()],
            created_at: ts(35),
            updated_at: ts(35),
        },
    ];

    let decisions = [
        DecisionMemory {
            id: "d-sqlite-storage".into(),
            project_id: EVAL_PROJECT.into(),
            title: "Use SQLite with FTS5 as the memory store".into(),
            context: "Need local-first storage with full-text search and zero configuration".into(),
            rationale: "SQLite bundles BM25 via FTS5, needs no server or API keys, and keeps \
                        every memory in one file on the user's machine; Postgres would trade \
                        the local-first guarantee for scale we do not have"
                .into(),
            tradeoffs: "Single-writer concurrency; brute-force vector scan instead of a \
                        dedicated index"
                .into(),
            related_files: vec![],
            tags: vec!["storage".into(), "architecture".into()],
            importance: 0.8,
            superseded_by: None,
            created_at: ts(40),
            updated_at: ts(40),
        },
        DecisionMemory {
            id: "d-no-async".into(),
            project_id: EVAL_PROJECT.into(),
            title: "No async runtime: synchronous SQLite everywhere".into(),
            context: "Whether to build the MCP server on tokio".into(),
            rationale: "SQLite is local and synchronous; an async runtime would add complexity \
                        and dependency weight for no throughput benefit"
                .into(),
            tradeoffs: "Concurrency comes from a bounded worker pool over an r2d2 connection \
                        pool instead of async tasks"
                .into(),
            related_files: vec![],
            tags: vec!["architecture".into()],
            importance: 0.7,
            superseded_by: None,
            created_at: ts(38),
            updated_at: ts(38),
        },
        DecisionMemory {
            id: "d-candle-embeddings".into(),
            project_id: EVAL_PROJECT.into(),
            title: "Local embeddings via pure-Rust candle, feature-gated".into(),
            context: "Semantic search needs an embedding model without a native runtime or \
                      API key"
                .into(),
            rationale: "candle runs BERT inference fully offline in-process; gating it behind a \
                        feature keeps the default binary self-contained"
                .into(),
            tradeoffs: "Brute-force cosine scoring suits thousands of memories, not millions"
                .into(),
            related_files: vec![],
            tags: vec!["retrieval".into(), "semantic".into()],
            importance: 0.6,
            superseded_by: None,
            created_at: ts(18),
            updated_at: ts(18),
        },
    ];

    let procedures = [
        ProceduralMemory {
            id: "p-release".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "Release workflow".into(),
            steps: vec![
                "Run the full test suite".into(),
                "Build the release binary for all targets".into(),
                "Update CHANGELOG and bump the version".into(),
                "Tag the commit and push".into(),
                "Let CI publish the release artifacts".into(),
            ],
            related_tools: vec!["cargo".into(), "git".into()],
            tags: vec!["release".into(), "workflow".into()],
            importance: 0.7,
            created_at: ts(12),
            updated_at: ts(12),
        },
        ProceduralMemory {
            id: "p-review".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "Code review checklist".into(),
            steps: vec![
                "Check tests cover the changed paths".into(),
                "Verify clippy passes with -D warnings".into(),
                "Read the diff for unintended behavior changes".into(),
                "Confirm documentation and CHANGELOG are updated".into(),
            ],
            related_tools: vec![],
            tags: vec!["review".into(), "quality".into()],
            importance: 0.5,
            created_at: ts(16),
            updated_at: ts(16),
        },
        ProceduralMemory {
            id: "p-incident".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "Incident response".into(),
            steps: vec![
                "Reproduce the failure locally".into(),
                "Identify the root cause before fixing".into(),
                "Record a failure memory with root cause and prevention".into(),
                "Verify the fix and close the loop".into(),
            ],
            related_tools: vec![],
            tags: vec!["incident".into(), "workflow".into()],
            importance: 0.6,
            created_at: ts(7),
            updated_at: ts(7),
        },
    ];

    let episodic = [
        EpisodicMemory {
            id: "e-auth-refactor".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-auth".into(),
            summary: "Refactored the auth module to token-based sessions".into(),
            content: "Replaced cookie login state in login.rs and session.rs with signed \
                      bearer-token sessions; all login flows now go through one helper"
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.6,
            tags: vec!["auth".into(), "refactor".into()],
            created_at: ts(3),
            updated_at: ts(3),
        },
        EpisodicMemory {
            id: "e-http-transport".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-http".into(),
            summary: "Added the Streamable HTTP MCP transport with bearer auth".into(),
            content: "Optional HTTP listener behind a config flag; startup refuses an empty \
                      auth token; stdio stays the default transport"
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["transport".into(), "http".into()],
            created_at: ts(6),
            updated_at: ts(6),
        },
    ];

    for m in &failures {
        repo.create_failure(m)?;
    }
    for m in &decisions {
        repo.create_decision(m)?;
    }
    for m in &procedures {
        repo.create_procedural(m)?;
    }
    for m in &episodic {
        repo.create_episodic(m)?;
    }
    Ok(())
}

/// Per-case scoring outcome.
#[derive(Debug, Clone)]
pub struct CaseReport {
    pub case_id: String,
    pub query: String,
    pub expected: Vec<String>,
    /// Ranked result ids, in pipeline order.
    pub ranked: Vec<String>,
    /// 1-based rank of the primary (first) expected id; None = miss.
    pub primary_rank: Option<usize>,
    /// 1/rank of the first expected id found (0.0 on total miss).
    pub reciprocal_rank: f64,
    /// NDCG over the top `EVAL_K` results (graded by expected position).
    pub ndcg: f64,
}

/// Aggregate scores over the whole golden set.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub cases: Vec<CaseReport>,
    pub case_count: usize,
    /// Mean reciprocal rank.
    pub mrr: f64,
    /// Mean NDCG@{EVAL_K}.
    pub ndcg: f64,
    /// Fraction of cases whose primary expected id ranked in the top 3.
    pub primary_hit_at_3: f64,
}

/// Cutoff for NDCG and the returned ranking window.
pub const EVAL_K: usize = 5;

/// Run every golden case through the provider's real search pipeline and
/// score the outcome. `seed_corpus` must have been called on the provider's
/// repository first (see `harness()`).
pub fn run_eval(provider: &DefaultMemoryProvider) -> Result<EvalReport> {
    let mut cases = Vec::new();
    for case in golden_cases() {
        let results = provider.search_core(case.query, EVAL_PROJECT, None, &[], None, EVAL_K)?;
        let ranked: Vec<String> = results.iter().map(|r| r.id.clone()).collect();

        let primary_rank = ranked
            .iter()
            .position(|id| id == case.expected[0])
            .map(|i| i + 1);
        let first_hit_rank = case
            .expected
            .iter()
            .filter_map(|e| ranked.iter().position(|id| id == *e))
            .min()
            .map(|i| i + 1);
        let reciprocal_rank = first_hit_rank.map(|r| 1.0 / r as f64).unwrap_or(0.0);

        // Graded relevance by expected position: first expected = len, then
        // descending — a strict ordering signal for NDCG.
        let grade = |idx: usize| (case.expected.len() - idx) as f64;
        let mut dcg = 0.0;
        for (rank_idx, id) in ranked.iter().take(EVAL_K).enumerate() {
            if let Some(exp_idx) = case.expected.iter().position(|e| e == id) {
                dcg += grade(exp_idx) / (rank_idx as f64 + 2.0).log2();
            }
        }
        let ideal_len = case.expected.len().min(EVAL_K);
        let idcg: f64 = (0..ideal_len)
            .map(|i| grade(i) / (i as f64 + 2.0).log2())
            .sum();
        let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };

        cases.push(CaseReport {
            case_id: case.id.to_string(),
            query: case.query.to_string(),
            expected: case.expected.iter().map(|s| s.to_string()).collect(),
            ranked,
            primary_rank,
            reciprocal_rank,
            ndcg,
        });
    }

    let case_count = cases.len();
    let mrr = cases.iter().map(|c| c.reciprocal_rank).sum::<f64>() / case_count as f64;
    let ndcg = cases.iter().map(|c| c.ndcg).sum::<f64>() / case_count as f64;
    let primary_hit_at_3 = cases
        .iter()
        .filter(|c| c.primary_rank.is_some_and(|r| r <= 3))
        .count() as f64
        / case_count as f64;

    Ok(EvalReport {
        cases,
        case_count,
        mrr,
        ndcg,
        primary_hit_at_3,
    })
}

/// Build a provider over a freshly seeded in-memory repository. The harness
/// never opens the user's database.
pub fn harness(config: crate::config::Config) -> Result<DefaultMemoryProvider> {
    let repo = MemoryRepository::new_in_memory()?;
    repo.initialize_schema()?;
    seed_corpus(&repo, chrono::Utc::now().timestamp())?;
    Ok(DefaultMemoryProvider::new(repo, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> EvalReport {
        let provider = harness(crate::config::Config::default()).unwrap();
        run_eval(&provider).unwrap()
    }

    #[test]
    fn corpus_seeds_all_types() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_corpus(&repo, 1_700_000_000).unwrap();
        // 3 failures + 3 decisions + 3 procedures + 2 episodic = 11 rows.
        assert_eq!(
            repo.list_recent_failures(EVAL_PROJECT, 50).unwrap().len(),
            3
        );
        assert_eq!(
            repo.list_recent_decisions(EVAL_PROJECT, 50).unwrap().len(),
            3
        );
        assert_eq!(
            repo.list_projects().unwrap(),
            vec![EVAL_PROJECT.to_string()]
        );
    }

    #[test]
    fn every_case_finds_its_primary_in_top_3() {
        let r = report();
        let misses: Vec<&CaseReport> = r
            .cases
            .iter()
            .filter(|c| c.primary_rank.is_none_or(|rank| rank > 3))
            .collect();
        assert!(
            misses.is_empty(),
            "golden cases whose primary result fell out of the top 3: {misses:#?}"
        );
    }

    #[test]
    fn aggregate_quality_baseline() {
        // Regression baseline for the CURRENT pipeline (measured 2026-09:
        // MRR 0.764 / NDCG@5 0.825 / hit@3 100% on the committed corpus).
        // A ranking change that moves these numbers must either be a measured
        // improvement (run `engram eval` before and after) or come with an
        // updated corpus/threshold — never a silent drift.
        //
        // Known weakness this baseline documents (do NOT "fix" by loosening
        // the corpus): on multi-term queries where the AND match fails, a
        // single-rare-term hit in a short-field table (e.g. failures) can
        // carry a higher raw bm25() than a multi-term hit in another table,
        // so static importance/type priors outrank the lexically stronger
        // match (`decision-no-async`, `failure-wal-growth` rank 3). The
        // principled fix is a query-term coverage signal in cross-type
        // calibration — tracked for the next ranking iteration.
        let r = report();
        assert!(r.mrr >= 0.75, "MRR regressed below 0.75: {}", r.mrr);
        assert!(r.ndcg >= 0.80, "NDCG@5 regressed below 0.80: {}", r.ndcg);
        assert!(
            r.primary_hit_at_3 >= 0.99,
            "primary hit@3 regressed: {}",
            r.primary_hit_at_3
        );
    }

    #[test]
    fn metrics_math_on_known_ranking() {
        // Hand-checked: expected [A, B], ranking [B, X, A].
        // RR = 1/1 (B found at rank 1). NDCG@5: grades A=2, B=1.
        // DCG = 1/log2(2) + 2/log2(4) = 1 + 1 = 2. IDCG = 2/log2(2) + 1/log2(3)
        // ≈ 2 + 0.6309 = 2.6309 → NDCG ≈ 0.7604.
        let ranked = ["B".to_string(), "X".to_string(), "A".to_string()];
        let expected = ["A", "B"];
        let first_hit = expected
            .iter()
            .filter_map(|e| ranked.iter().position(|id| id == *e))
            .min()
            .map(|i| i + 1)
            .unwrap();
        assert_eq!(first_hit, 1);
        let grade = |idx: usize| (expected.len() - idx) as f64;
        let mut dcg = 0.0;
        for (i, id) in ranked.iter().take(EVAL_K).enumerate() {
            if let Some(e) = expected.iter().position(|x| *x == id) {
                dcg += grade(e) / (i as f64 + 2.0).log2();
            }
        }
        let idcg: f64 = (0..expected.len().min(EVAL_K))
            .map(|i| grade(i) / (i as f64 + 2.0).log2())
            .sum();
        assert!(
            (dcg - 2.0).abs() < 1e-9,
            "DCG should be exactly 2.0, got {dcg}"
        );
        assert!((dcg / idcg - 0.7604).abs() < 1e-3);
    }
}
