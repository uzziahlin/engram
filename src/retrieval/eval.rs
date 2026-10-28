//! Retrieval-quality evaluation harness: a golden corpus + golden query set
//! scored against the REAL search pipeline (intent → BM25 (+filters) →
//! graph expansion → rerank).
//!
//! The ranking math has been rebuilt several times (v0.3 unified ordering,
//! v0.4 ranking rebuild, v0.5 cross-type BM25 calibration) and each change
//! was judged by hand. This module makes retrieval quality measurable:
//!
//! - `seed_corpus` writes a fixed, realistic engineering-knowledge corpus
//!   (deterministic ids, timestamps relative to `now`) into a repository —
//!   including graph-linked rows (shared file entities), Chinese-content
//!   rows, and a second project for global-search cases.
//! - `golden_cases()` returns queries — English and Chinese, across
//!   intents, with optional explicit filters and project scopes — each
//!   with the memory ids a *good* ranking should surface, best first.
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

/// Second project a few corpus memories live in, so the global-search
/// (`project_id = "*"`) path is exercised against cross-project hits.
pub const EVAL_ALT_PROJECT: &str = "engram-eval-alt";

/// One golden query: `expected` lists memory ids a good ranking surfaces,
/// best match first (grades for NDCG descend with position). The optional
/// filter fields route the case through the corresponding search paths
/// (explicit filters, global scope) — measured, not aspirational.
#[derive(Default)]
pub struct GoldenCase {
    pub id: &'static str,
    pub query: &'static str,
    pub expected: &'static [&'static str],
    /// Explicit `memory_type` filter (None = search all kinds).
    pub memory_type: Option<&'static str>,
    /// Explicit `tags` filter (empty = no tag filter).
    pub tags: &'static [&'static str],
    /// Explicit `before` cutoff, as days before the corpus seeding `now`
    /// (the corpus timestamps are `now - age_days*86400`, so a cutoff
    /// strictly between two ages is deterministic).
    pub before_days: Option<i64>,
    /// Project scope override; None = EVAL_PROJECT, Some("*") = global.
    pub project: Option<&'static str>,
}

/// Plain (unfiltered, EVAL_PROJECT-scoped) golden case — the shape every
/// original case uses.
fn case(id: &'static str, query: &'static str, expected: &'static [&'static str]) -> GoldenCase {
    GoldenCase {
        id,
        query,
        expected,
        memory_type: None,
        tags: &[],
        before_days: None,
        project: None,
    }
}

/// The golden query set. Kept deliberately sharp: each case targets one
/// retrieval behavior — cross-type ranking, intent keywords, CJK, explicit
/// filters, graph expansion, global scope. When a ranking change moves a
/// case here, decide whether the case or the change is wrong — that
/// judgment is the whole point of this harness.
pub fn golden_cases() -> Vec<GoldenCase> {
    vec![
        case(
            "failure-fts5-keywords",
            "FTS5 crash on SQL keywords",
            &["f-fts5-keywords"],
        ),
        case(
            "failure-cjk-backfill",
            "Chinese search returns nothing",
            &["f-cjk-backfill"],
        ),
        case(
            "failure-wal-growth",
            "database file keeps growing WAL",
            &["f-wal-growth"],
        ),
        case(
            "decision-storage",
            "why SQLite instead of Postgres",
            &["d-sqlite-storage"],
        ),
        case(
            "decision-no-async",
            "why no tokio async runtime",
            &["d-no-async"],
        ),
        case(
            "decision-embeddings",
            "embedding model choice local inference",
            &["d-candle-embeddings"],
        ),
        case(
            "episodic-auth-refactor",
            "authentication refactor login session",
            &["e-auth-refactor"],
        ),
        case(
            "episodic-http-transport",
            "HTTP transport bearer token",
            &["e-http-transport"],
        ),
        case("procedural-release", "release 发布流程", &["p-release"]),
        case(
            "procedural-review",
            "code review 代码评审 checklist",
            &["p-review"],
        ),
        case(
            "procedural-incident",
            "incident response steps",
            &["p-incident"],
        ),
        GoldenCase {
            // Both a failure (the incident) and the decision that resolved the
            // strategy are relevant; the failure is the stronger match.
            id: "cross-type-migration",
            query: "migration broke search index",
            expected: &["f-cjk-backfill", "d-sqlite-storage"],
            ..GoldenCase::default()
        },
        // ── Paraphrase cases: queries that do NOT reuse the memory's exact
        // wording. The original set's queries were near-verbatim copies of
        // their targets — it measured lexical echo, not retrieval. These
        // must still rank via partial/synonym overlap.
        case(
            "paraphrase-ship-release",
            "how do we publish a new version",
            &["p-release"],
        ),
        case(
            "paraphrase-db-bloat",
            "database growing maintenance cleanup",
            &["f-wal-growth"],
        ),
        // ── Distractor cases: a near-miss memory shares part of the query
        // vocabulary; the better-matching target must still win.
        GoldenCase {
            // f-auth-expiry shares "login/session/token" with e-auth-refactor;
            // the query's expiry wording must disambiguate.
            id: "distractor-login-expiry",
            query: "login sessions expired immediately after rotation",
            expected: &["f-auth-expiry"],
            ..GoldenCase::default()
        },
        GoldenCase {
            // d-transport-choice discusses transport choice (incl. HTTP); the
            // bearer-token wording belongs to the episodic.
            id: "distractor-bearer-auth",
            query: "HTTP transport bearer token auth",
            expected: &["e-http-transport"],
            ..GoldenCase::default()
        },
        GoldenCase {
            // Exercises the query-term coverage signal: "release" alone hits
            // p-release strongly, but p-review matches 3 of the 4 terms
            // (checklist/review/clippy) and must outrank the single-term hit.
            id: "coverage-release-vs-review",
            query: "release checklist review clippy",
            expected: &["p-review", "p-release"],
            ..GoldenCase::default()
        },
        // ── Graph-expansion case: the corpus now contains memories linked
        // by shared File entities (see e-graph-fuse / e-graph-cache /
        // d-graph-rrf-choice). The query textually hits e-graph-fuse only;
        // the entity-sharing memories must surface as graph_neighbor
        // second-tier results (asserted precisely in the test module —
        // golden scoring only pins the primary hit).
        case(
            "graph-shared-file-neighbor",
            "reciprocal rank fusion candidate lists",
            &["e-graph-fuse"],
        ),
        // ── Filter cases: the explicit memory_type/tags/before paths were
        // completely outside the eval until now. Expected sets are EXACT
        // (asserted in filter_cases_return_exactly_the_expected_set).
        GoldenCase {
            id: "filter-memory-type-failure",
            query: "Chinese search returned nothing",
            expected: &["f-cjk-backfill"],
            memory_type: Some("failure"),
            ..GoldenCase::default()
        },
        GoldenCase {
            // "workflow" matches p-release/p-incident/p-bugfix textually;
            // the tags filter must reduce that to the release-tagged one.
            id: "filter-tags-release",
            query: "workflow",
            expected: &["p-release"],
            tags: &["release"],
            ..GoldenCase::default()
        },
        GoldenCase {
            // p-review (age 16d) predates the cutoff; p-release (age 12d)
            // does not and must be excluded by `before`.
            id: "filter-before-excludes-recent",
            query: "release checklist review clippy",
            expected: &["p-review"],
            before_days: Some(14),
            ..GoldenCase::default()
        },
        // ── CJK cases: Chinese queries against Chinese-content memories
        // (the corpus's Chinese rows used to be zero). cjk-paraphrase is a
        // pure-Chinese rewording — no verbatim sentence overlap.
        case("cjk-lock-direct", "数据库文件被锁死", &["f-cjk-lock"]),
        case(
            "cjk-paraphrase-backup",
            "每天晚上自动备份数据",
            &["p-cjk-backup"],
        ),
        // ── Global case: project_id="*" must surface hits from BOTH corpus
        // projects (measured order) with each hit carrying its owning
        // project (asserted in global_search_hits_carry_owner_project).
        GoldenCase {
            id: "global-vector-cache",
            query: "vector cache reload",
            expected: &["e-graph-cache", "e-alt-vector-cache"],
            project: Some("*"),
            ..GoldenCase::default()
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
            id: "f-ci-flaky".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "CI pipeline intermittently red on parallel test runs".into(),
            root_cause: "Integration tests shared one SQLite file; concurrent runners                          locked each other"
                .into(),
            fix: "Give each test a throwaway database in its own tempdir".into(),
            prevention: "Never share mutable fixtures across parallel tests".into(),
            severity: 3,
            tags: vec!["ci".into(), "testing".into()],
            created_at: ts(11),
            updated_at: ts(11),
        },
        FailureMemory {
            id: "f-dep-audit".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "Dependency audit flagged an outdated HTTP client with a known CVE"
                .into(),
            root_cause: "The lockfile had not been regenerated after the advisory dropped"
                .into(),
            fix: "Bump the crate, regenerate the lockfile, add an audit step to CI".into(),
            prevention: "cargo audit in CI catches stale vulnerable deps automatically".into(),
            severity: 2,
            tags: vec!["security".into(), "deps".into()],
            created_at: ts(44),
            updated_at: ts(44),
        },
        FailureMemory {
            id: "f-auth-expiry".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "Login sessions expired immediately after the token rotation deploy"
                .into(),
            root_cause: "The rotation wrote new expiry timestamps in milliseconds while the \
                         validator read seconds"
                .into(),
            fix: "Normalize all expiry math to seconds at the session-store boundary".into(),
            prevention: "Keep one time unit per domain concept; assert it in tests".into(),
            severity: 3,
            tags: vec!["auth".into(), "login".into()],
            created_at: ts(28),
            updated_at: ts(28),
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
        FailureMemory {
            // Chinese-content failure: until this row the CJK query path had
            // zero Chinese-content memories to hit in the eval corpus.
            // Worded to avoid 发/布/流/程 — those chars appear in the
            // procedural-release golden query (发布流程), and per-character
            // CJK tokens carry huge IDF in a mostly-English corpus, easily
            // outranking the intended English hit.
            id: "f-cjk-lock".into(),
            project_id: EVAL_PROJECT.into(),
            incident: "多进程写入时数据库文件被锁死".into(),
            root_cause: "多个进程同时打开同一个数据库文件，写锁互相阻塞".into(),
            fix: "所有写入收敛到同一个进程内的连接池".into(),
            prevention: "不要让多个进程直接写同一个文件".into(),
            severity: 3,
            tags: vec!["storage".into(), "cjk".into()],
            created_at: ts(18),
            updated_at: ts(18),
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
            id: "d-config-format".into(),
            project_id: EVAL_PROJECT.into(),
            title: "TOML as the user configuration format".into(),
            context: "Choosing the config file format for ~/.engram/config.toml".into(),
            rationale: "TOML maps cleanly onto typed structs with serde, fails loudly on                         typos, and is the Rust-ecosystem default users expect"
                .into(),
            tradeoffs: "Less expressive than YAML for deeply nested data; that ceiling is \
                        far above what a config file here needs"
                .into(),
            related_files: vec![],
            tags: vec!["config".into()],
            importance: 0.5,
            superseded_by: None,
            created_at: ts(33),
            updated_at: ts(33),
        },
        DecisionMemory {
            id: "d-backup-retention".into(),
            project_id: EVAL_PROJECT.into(),
            title: "Online SQLite backups with N-generation retention".into(),
            context: "How to protect the single-file memory database".into(),
            rationale: "The online backup API copies pages without locking out readers; \
                        keeping the last five generations bounds disk usage while covering \
                        a bad-write window of weeks"
                .into(),
            tradeoffs: "Backups are whole-file copies; incremental page-level deltas would \
                        complicate restore for little space win at this scale"
                .into(),
            related_files: vec![],
            tags: vec!["ops".into(), "backup".into()],
            importance: 0.6,
            superseded_by: None,
            created_at: ts(50),
            updated_at: ts(50),
        },
        DecisionMemory {
            id: "d-transport-choice".into(),
            project_id: EVAL_PROJECT.into(),
            title: "stdio as the default MCP transport".into(),
            context: "Which transport the server should primarily speak".into(),
            rationale: "Every MCP client launches stdio servers natively; HTTP adds an auth \
                        surface and a bind address to secure for a single-user local tool"
                .into(),
            tradeoffs: "Remote/multi-client setups need the optional HTTP listener".into(),
            related_files: vec![],
            tags: vec!["transport".into(), "architecture".into()],
            importance: 0.6,
            superseded_by: None,
            created_at: ts(26),
            updated_at: ts(26),
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
        DecisionMemory {
            // Graph-linked decision: shares the embedding_service.rs File
            // entity with e-graph-fuse/e-graph-cache, so graph expansion can
            // surface it (or them) as second-tier neighbors.
            id: "d-graph-rrf-choice".into(),
            project_id: EVAL_PROJECT.into(),
            title: "Combine lexical and vector hits by reciprocal rank".into(),
            context: "How to merge FTS candidates with vector-neighborhood candidates when \
                      their score scales are incomparable"
                .into(),
            rationale: "Rank-based merging is scale-blind: a cosine noise floor and a \
                        calibrated text score each contribute a rank instead of one \
                        overwriting the other"
                .into(),
            tradeoffs: "Ranks discard magnitude; two candidates one rank apart are \
                        treated as close regardless of raw score gaps"
                .into(),
            related_files: vec![
                "src/retrieval/fusion.rs".into(),
                "src/mcp/embedding_service.rs".into(),
            ],
            tags: vec!["retrieval".into(), "semantic".into()],
            importance: 0.6,
            superseded_by: None,
            created_at: ts(22),
            updated_at: ts(22),
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
        ProceduralMemory {
            id: "p-bugfix".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "Bugfix workflow".into(),
            steps: vec![
                "Reproduce the bug with a minimal failing case".into(),
                "Write the regression test first".into(),
                "Fix the cause, not the symptom".into(),
                "Record a failure memory with root cause and prevention".into(),
            ],
            related_tools: vec![],
            tags: vec!["bugfix".into(), "workflow".into()],
            importance: 0.6,
            created_at: ts(20),
            updated_at: ts(20),
        },
        ProceduralMemory {
            id: "p-onboarding".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "New contributor onboarding".into(),
            steps: vec![
                "Clone and build with cargo build".into(),
                "Read the architecture section of the README".into(),
                "Run the test suite once before changing anything".into(),
                "Pick a starter issue labeled good-first-issue".into(),
            ],
            related_tools: vec![],
            tags: vec!["onboarding".into()],
            importance: 0.4,
            created_at: ts(25),
            updated_at: ts(25),
        },
        ProceduralMemory {
            id: "p-nightly".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "Nightly maintenance pass".into(),
            steps: vec![
                "Back up the database".into(),
                "Prune the query log past retention".into(),
                "Consolidate duplicate memories".into(),
                "Checkpoint the WAL".into(),
            ],
            related_tools: vec![],
            tags: vec!["maintenance".into(), "ops".into()],
            importance: 0.5,
            created_at: ts(8),
            updated_at: ts(8),
        },
        ProceduralMemory {
            // Chinese-content procedural; the cjk-paraphrase-backup case
            // queries it with DIFFERENT wording (每晚 vs 每天晚上), so the
            // case measures partial CJK overlap, not verbatim echo.
            // "任务" instead of "流程": 流/程 sit in the procedural-release
            // golden query (发布流程) and per-char CJK tokens carry huge IDF.
            id: "p-cjk-backup".into(),
            project_id: EVAL_PROJECT.into(),
            workflow_name: "每晚维护任务".into(),
            steps: vec![
                "备份整个数据库文件".into(),
                "清理过期的查询日志".into(),
                "合并重复的记忆条目".into(),
            ],
            related_tools: vec![],
            tags: vec!["maintenance".into(), "cjk".into()],
            importance: 0.5,
            created_at: ts(10),
            updated_at: ts(10),
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
            id: "e-ci-arm-runners".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-ci".into(),
            summary: "Added native ARM runners to the CI build matrix".into(),
            content: "Cross-compiling aarch64 Linux was slower than emulated runners; \
                      native arm64 jobs cut the workflow from 25 minutes to 6"
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["ci".into(), "build".into()],
            created_at: ts(5),
            updated_at: ts(5),
        },
        EpisodicMemory {
            id: "e-query-log".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-qlog".into(),
            summary: "Wired the query log and adoption tracking into search".into(),
            content: "Every search now records its result ids; reading a memory later \
                      marks it adopted so rankings can learn from real usage"
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.6,
            tags: vec!["retrieval".into(), "feedback".into()],
            created_at: ts(2),
            updated_at: ts(2),
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
        EpisodicMemory {
            // Graph-linked: e-graph-fuse and e-graph-cache share the
            // embedding_service.rs File entity (as does d-graph-rrf-choice),
            // giving the eval corpus real graph edges — until these the
            // graph-expansion path was dead code in the harness.
            id: "e-graph-fuse".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-fuse".into(),
            summary: "Fused BM25 and vector candidates by reciprocal rank".into(),
            content: "replaced the max-injection score with reciprocal rank fusion across \
                      the two ranked candidate lists"
                .into(),
            files_touched: vec!["src/mcp/embedding_service.rs".into()],
            related_commits: vec![],
            importance: 0.6,
            tags: vec!["retrieval".into(), "semantic".into()],
            created_at: ts(4),
            updated_at: ts(4),
        },
        EpisodicMemory {
            id: "e-graph-cache".into(),
            project_id: EVAL_PROJECT.into(),
            session_id: "s-fuse".into(),
            summary: "Vector cache reloads on PRAGMA data_version change".into(),
            content: "cached embedding vectors are reloaded when another process commits, \
                      so a stale set never serves"
                .into(),
            files_touched: vec![
                "src/mcp/embedding_service.rs".into(),
                "src/storage/repository.rs".into(),
            ],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["retrieval".into(), "semantic".into()],
            created_at: ts(9),
            updated_at: ts(9),
        },
        EpisodicMemory {
            // Second-project twin for the global-search case: same topic
            // vocabulary as e-graph-cache, different project.
            id: "e-alt-vector-cache".into(),
            project_id: EVAL_ALT_PROJECT.into(),
            session_id: "s-alt".into(),
            summary: "Cross-project vector cache warmup".into(),
            content: "the wildcard project search reloads vector caches per project; this \
                      memory lives in the alt project only"
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["retrieval".into()],
            created_at: ts(13),
            updated_at: ts(13),
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
/// repository first (see `harness()`). Cases carrying explicit filters or a
/// project override route through the corresponding search paths, so the
/// golden set exercises them end to end.
pub fn run_eval(provider: &DefaultMemoryProvider) -> Result<EvalReport> {
    let mut cases = Vec::new();
    for case in golden_cases() {
        // `before_days` is relative to the corpus seeding `now` (see
        // seed_corpus); harness seeding and this `now` are milliseconds
        // apart, and cutoffs sit between whole-day ages, so the comparison
        // is deterministic.
        let now = chrono::Utc::now().timestamp();
        let before = case.before_days.map(|d| now - d * 86_400);
        let tags: Vec<String> = case.tags.iter().map(|t| t.to_string()).collect();
        let results = provider.search_core(
            case.query,
            case.project.unwrap_or(EVAL_PROJECT),
            case.memory_type,
            &tags,
            before,
            EVAL_K,
        )?;
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
    use crate::retrieval::bm25::BM25Retriever;

    fn report() -> EvalReport {
        let provider = harness(crate::config::Config::default()).unwrap();
        run_eval(&provider).unwrap()
    }

    #[test]
    fn corpus_seeds_all_types() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_corpus(&repo, 1_700_000_000).unwrap();
        // 7 failures + 7 decisions + 7 procedures + 7 episodic = 28 rows
        // (batch 4 added the graph-linked trio, the Chinese pair, and the
        // alt-project episodic). Corpus size matters beyond variety: SQLite's
        // bm25 IDF has no +1 smoothing — any term present in >=50% of a
        // table's rows scores ~0 — so a 2-3 row table zeroes most of its own
        // vocabulary.
        assert_eq!(
            repo.list_recent_failures(EVAL_PROJECT, 50).unwrap().len(),
            7
        );
        assert_eq!(
            repo.list_recent_decisions(EVAL_PROJECT, 50).unwrap().len(),
            7
        );
        assert_eq!(
            repo.list_projects().unwrap(),
            vec![EVAL_PROJECT.to_string(), EVAL_ALT_PROJECT.to_string()]
        );
        // Graph edges exist: the shared embedding_service.rs File entity
        // links e-graph-fuse, e-graph-cache, and d-graph-rrf-choice.
        let neighbors = repo
            .neighbor_memories(EVAL_PROJECT, &["e-graph-fuse".to_string()], 5)
            .unwrap();
        let neighbor_ids: std::collections::HashSet<&str> =
            neighbors.iter().map(|(_, id)| id.as_str()).collect();
        assert!(
            neighbor_ids.contains("e-graph-cache") && neighbor_ids.contains("d-graph-rrf-choice"),
            "corpus must carry real graph edges, got {neighbor_ids:?}"
        );
    }

    #[test]
    fn print_metrics_baseline() {
        // Not an assertion — prints current aggregate metrics so baseline
        // re-pinning copies observed numbers. Run with -- --nocapture.
        let r = report();
        println!(
            "METRICS cases={} mrr={:.4} ndcg={:.4} hit@3={:.4}",
            r.case_count, r.mrr, r.ndcg, r.primary_hit_at_3
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
        // Regression baseline for the CURRENT pipeline. Measured 2026-10
        // after batch 4 extended the corpus 22 → 28 rows and the case set
        // 17 → 24 (graph / explicit-filter / CJK-content / global cases):
        // MRR 1.000 / NDCG@5 0.9979 / hit@3 100%. The one imperfect case
        // is cross-type-migration (NDCG 0.950): under the whole-corpus IDF
        // the f-fts5-keywords distractor slips between the two expected ids
        // (RR stays 1.0) — the per-type tables suppressed it only because
        // their IDF corpora were isolated, which is exactly the artifact
        // the unified table removes. The pre-batch-4 17-case numbers
        // (1.000 / 0.997 / 100%) still hold case-for-case: no existing
        // expectation moved. A ranking change that moves these numbers
        // must either be a measured improvement (run `engram eval` before
        // and after) or come with an updated corpus/threshold — never a
        // silent drift.
        //
        // History, so the numbers stay honest: the 2026-09 baseline (0.764 /
        // 0.825) was measured with a cross-type calibration bug that zeroed
        // the BM25 contribution entirely (normalize_bm25 negated an
        // already-positive type_best → sigmoid(0) = 0) — those scores came
        // from the reranker's static priors alone. It also predated the
        // query-term coverage signal and the stopword filter. The 0.6.0
        // baseline (1.000 / 1.000) was achieved on the per-type calibration
        // layer whose sigmoid scale was tuned against this same small
        // corpus — it saturated as corpora grew, which is what the unified
        // table + per-query monotone normalization replace.
        //
        // Corpus-size caveat: SQLite bm25 IDF has no +1 smoothing — a term
        // in >=50% of the WHOLE corpus scores ~0 (per-type isolation made
        // that threshold per-kind, zeroing common words in small kinds) — so
        // keep the corpus diverse when adding cases; scale behavior is
        // pinned separately by the scale_corpus_* tests below.
        let r = report();
        assert!(r.mrr >= 0.99, "MRR regressed below 0.99: {}", r.mrr);
        assert!(r.ndcg >= 0.95, "NDCG@5 regressed below 0.95: {}", r.ndcg);
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

    // ─── Batch 4: graph / filter / negative / CJK / global dimensions ──
    // These pin the search-pipeline dimensions the golden aggregate does
    // not isolate. Every expectation below is measured, not aspirational.

    /// Query textually hits e-graph-fuse; the memories sharing its File
    /// entity must surface as second-tier results flagged graph_neighbor,
    /// ranked below every true hit.
    #[test]
    fn graph_shared_file_entity_surfaces_graph_neighbor() {
        let provider = harness(crate::config::Config::default()).unwrap();
        let results = provider
            .search_core(
                "reciprocal rank fusion candidate lists",
                EVAL_PROJECT,
                None,
                &[],
                None,
                EVAL_K,
            )
            .unwrap();
        let hit_pos = results.iter().position(|r| r.id == "e-graph-fuse");
        assert!(
            hit_pos.is_some_and(|p| p == 0),
            "the textual hit must rank first, got {:?}",
            results.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
        );
        let neighbor = results
            .iter()
            .find(|r| r.id == "e-graph-cache")
            .expect("memory sharing the top hit's File entity must surface");
        assert_eq!(
            neighbor
                .detail
                .get("graph_neighbor")
                .and_then(|v| v.as_bool()),
            Some(true),
            "second-tier graph result must be flagged graph_neighbor"
        );
        let neighbor_pos = results
            .iter()
            .position(|r| r.id == "e-graph-cache")
            .unwrap();
        assert!(
            neighbor_pos > hit_pos.unwrap(),
            "graph neighbor must rank below the true hit"
        );
    }

    /// The filter golden cases must return EXACTLY their expected sets —
    /// filters previously sat completely outside the eval, and the graph /
    /// vector penetration bugs of earlier rounds were exactly "one extra
    /// row leaked through a filter".
    #[test]
    fn filter_cases_return_exactly_the_expected_set() {
        let provider = harness(crate::config::Config::default()).unwrap();
        for case in golden_cases() {
            if !case.id.starts_with("filter-") {
                continue;
            }
            let now = chrono::Utc::now().timestamp();
            let before = case.before_days.map(|d| now - d * 86_400);
            let tags: Vec<String> = case.tags.iter().map(|t| t.to_string()).collect();
            let results = provider
                .search_core(
                    case.query,
                    case.project.unwrap_or(EVAL_PROJECT),
                    case.memory_type,
                    &tags,
                    before,
                    EVAL_K,
                )
                .unwrap();
            let mut got: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            got.sort_unstable();
            let mut want: Vec<&str> = case.expected.to_vec();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "case {} must return exactly its expected set",
                case.id
            );
        }
    }

    /// A query hitting nothing must return NOTHING — not static-prior
    /// filler. (The 2026-09 era scored 0.764 MRR on reranker priors alone;
    /// this pins that an empty candidate list stays empty end to end.)
    #[test]
    fn zero_hit_query_returns_empty_not_static_fillers() {
        let provider = harness(crate::config::Config::default()).unwrap();
        let results = provider
            .search_core(
                "zzzqxv blorptastic nonexistentterm",
                EVAL_PROJECT,
                None,
                &[],
                None,
                EVAL_K,
            )
            .unwrap();
        assert!(
            results.is_empty(),
            "zero-hit query must return no results, got {:?}",
            results.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
        );
    }

    /// Recall floor: a query whose only match is a weak partial overlap
    /// (AND fails, OR fallback finds one shared term) must still return
    /// that match — recall beats silence for mid-tail queries.
    #[test]
    fn weak_partial_match_still_returned() {
        let provider = harness(crate::config::Config::default()).unwrap();
        let results = provider
            .search_core(
                "xylophone onboarding",
                EVAL_PROJECT,
                None,
                &[],
                None,
                EVAL_K,
            )
            .unwrap();
        assert!(
            results.iter().any(|r| r.id == "p-onboarding"),
            "the only (weak) match must still be returned, got {:?}",
            results.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
        );
    }

    /// Global search must mix both corpus projects and carry each hit's
    /// owning project (the batch-4 global case in the golden set covers
    /// ranking; this pins attribution).
    #[test]
    fn global_search_hits_carry_owner_project() {
        let provider = harness(crate::config::Config::default()).unwrap();
        let results = provider
            .search_core("vector cache reload", "*", None, &[], None, EVAL_K)
            .unwrap();
        let owner_of = |id: &str| -> Option<&str> {
            results
                .iter()
                .find(|r| r.id == id)
                .map(|r| r.project_id.as_str())
        };
        assert_eq!(owner_of("e-graph-cache"), Some(EVAL_PROJECT));
        assert_eq!(owner_of("e-alt-vector-cache"), Some(EVAL_ALT_PROJECT));
        // Every hit must name a corpus project it truly belongs to — no
        // wildcard leakage of an unowned id.
        for r in &results {
            let known = r.project_id == EVAL_PROJECT || r.project_id == EVAL_ALT_PROJECT;
            assert!(
                known,
                "hit {} carries unknown project {}",
                r.id, r.project_id
            );
        }
    }

    // ─── Scale / saturation suite ────────────────────────────────────
    // The golden corpus is 22 rows; these tests pin behavior at ≥1000 rows,
    // where the per-type-table era had two failure modes: (a) the sigmoid
    // calibration (tuned on 22 rows) saturated and lost discrimination, and
    // (b) a common word in ≥50% of a SMALL kind's rows scored ~0 for that
    // kind because IDF was computed per kind. The unified table computes IDF
    // over the whole corpus; relevance is a per-query monotone normalization
    // with no fixed-scale constant.

    /// Seed a deterministic synthetic corpus: 4 kinds × 3 projects, ≥1000
    /// rows, every row combining a rotating topic word (shared across many
    /// rows) with row-unique discriminative text. `SCALE_COMMON_WORD` lands
    /// in ~1/3 of all rows; `zephyr` in exactly one (the discriminative
    /// needle). Returns (rows written, needle id, needle project).
    /// The needle id seeded by [`seed_scale_corpus`] (the corpus's only
    /// "zephyr" row, living in project scale-a).
    const SCALE_NEEDLE: &str = "zephyr-migration-decision";

    fn seed_scale_corpus(repo: &MemoryRepository) -> usize {
        const TOPICS: [&str; 8] = [
            "storage",
            "transport",
            "auth",
            "build",
            "release",
            "search",
            "config",
            "logging",
        ];
        const PROJECTS: [&str; 3] = ["scale-a", "scale-b", "scale-c"];
        // ~33% of rows carry this word — common, but far under the 50% IDF
        // clamp even before per-kind isolation stopped mattering.
        const SCALE_COMMON_WORD: &str = "boilerplate";
        let needle = "zephyr-migration-decision";
        let mut rows = 0usize;
        let mut needle_written = false;

        for i in 0..280 {
            let topic = TOPICS[i % TOPICS.len()];
            let project = PROJECTS[i % 7 % 3]; // ~57%/29%/14% split
            let common = if i % 3 == 0 {
                SCALE_COMMON_WORD
            } else {
                "plumbing"
            };
            let disc = format!("marker{i:04}");
            // Length variation: rows with different body lengths score
            // differently for the same term (bm25 length normalization) —
            // without it the formulaic corpus produces exact score ties and
            // no discrimination to assert on.
            let filler = format!("context padding sentence {i} ",).repeat(i % 9);

            let ep = EpisodicMemory {
                id: format!("sc-ep-{i:04}"),
                project_id: project.into(),
                session_id: format!("s{i}"),
                summary: format!("{topic} session {disc}"),
                content: format!("worked on {topic} internals with {disc} and {common} {filler}"),
                files_touched: vec![format!("src/{topic}_{i}.rs")],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![topic.into()],
                created_at: 1_000_000 + i as i64,
                updated_at: 1_000_000 + i as i64,
            };
            let _ = repo.create_episodic(&ep);
            rows += 1;

            let dec = DecisionMemory {
                id: format!("sc-de-{i:04}"),
                project_id: project.into(),
                title: format!("{topic} architecture {disc}"),
                context: format!("{topic} subsystem context {common} {filler}"),
                rationale: format!("{disc} rationale for {topic} design {filler}"),
                tradeoffs: format!("{topic} tradeoffs {common}"),
                related_files: vec![],
                tags: vec![topic.into()],
                importance: 0.5,
                superseded_by: None,
                created_at: 1_000_000 + i as i64,
                updated_at: 1_000_000 + i as i64,
            };
            let _ = repo.create_decision(&if dec.id == "sc-de-0000" {
                // The needle: the ONE row mentioning zephyr. i==0 → project
                // scale-a, topic "storage".
                DecisionMemory {
                    id: needle.into(),
                    title: "zephyr storage migration".into(),
                    context: format!("migrating storage to zephyr {common}"),
                    rationale: format!("zephyr handles {topic} load {disc}"),
                    tradeoffs: "zephyr is new".into(),
                    ..dec
                }
            } else {
                dec
            });
            rows += 1;
            if i == 0 {
                needle_written = true;
            }

            let fai = FailureMemory {
                id: format!("sc-fa-{i:04}"),
                project_id: project.into(),
                incident: format!("{topic} outage {disc}"),
                root_cause: format!("{topic} race in {disc} path {common} {filler}"),
                fix: format!("serialize {disc} access"),
                prevention: "add regression test".into(),
                severity: 3,
                tags: vec![topic.into()],
                created_at: 1_000_000 + i as i64,
                updated_at: 1_000_000 + i as i64,
            };
            let _ = repo.create_failure(&fai);
            rows += 1;

            let pro = ProceduralMemory {
                id: format!("sc-pr-{i:04}"),
                project_id: project.into(),
                workflow_name: format!("{topic} workflow {disc}"),
                steps: vec![
                    format!("run {topic} checks for {disc}"),
                    format!("verify {common}"),
                    format!("record {filler}"),
                ],
                related_tools: vec!["cargo".into()],
                tags: vec![topic.into()],
                importance: 0.5,
                created_at: 1_000_000 + i as i64,
                updated_at: 1_000_000 + i as i64,
            };
            let _ = repo.create_procedural(&pro);
            rows += 1;
        }
        assert!(needle_written);
        rows
    }

    fn scale_repo() -> MemoryRepository {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_scale_corpus(&repo);
        repo
    }

    #[test]
    fn scale_corpus_discriminative_query_hits_needle_and_keeps_spread() {
        let repo = scale_repo();
        let rows: i64 = {
            let conn = repo.connection().unwrap();
            conn.query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))
                .unwrap()
        };
        let needle = SCALE_NEEDLE;
        assert!(rows >= 1000, "corpus must be >=1000 rows, got {rows}");

        // (a) Discriminative multi-token query: the needle (only row with
        // "zephyr") must be top-1, ranked above ~350 "storage" rows.
        let r = BM25Retriever::search_all(&repo, "zephyr storage", "scale-a", 10).unwrap();
        assert!(!r.is_empty());
        assert_eq!(r[0].id, needle, "needle must rank first: {r:?}");
        assert!(r[0].relevance_score > 0.0);

        // Single common-topic query over the whole corpus: the top of the
        // result set must still DISCRIMINATE (the old sigmoid calibration
        // saturated at scale and tied strong with weak matches). Measured
        // 2026-10 on the 1120-row corpus: max/min ≈ 1.13 across the top
        // 100; the 1.1 threshold catches collapse (ratio → 1.0), not noise.
        let r = BM25Retriever::search_all(&repo, "storage", "*", 100).unwrap();
        assert_eq!(r.len(), 100);
        let max = r[0].relevance_score;
        let min = r
            .iter()
            .map(|x| x.relevance_score)
            .fold(f32::INFINITY, f32::min);
        assert!(
            min > 0.0,
            "top-100 scores must stay positive (got min {min})"
        );
        assert!(
            max / min >= 1.1,
            "top-100 relevance spread collapsed (max {max} / min {min} < 1.1) — saturation regression"
        );
    }

    #[test]
    fn scale_corpus_common_word_still_scores() {
        // (b) A word present in ~1/3 of ALL rows must still produce ranked,
        // non-zero-scored results. Under the per-type tables this was the
        // small-kind zeroing bug: the word sat in >=50% of a 2-3 row kind's
        // corpus and scored ~0 there.
        let repo = scale_repo();
        let r = BM25Retriever::search_all(&repo, "boilerplate", "*", 10).unwrap();
        assert_eq!(r.len(), 10, "a 33%-corpus word has hundreds of matches");
        assert!(
            r.iter().all(|x| x.relevance_score > 0.0),
            "scores must stay positive: {:?}",
            r.iter()
                .map(|x| (x.id.clone(), x.relevance_score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn scale_corpus_project_scoping_returns_only_that_project() {
        // (c) project hex filter: scoped search must not leak other projects'
        // rows even when they match the same terms.
        let repo = scale_repo();
        let r = BM25Retriever::search_all(&repo, "storage", "scale-b", 10).unwrap();
        assert!(!r.is_empty());
        assert!(
            r.iter().all(|x| x.project_id == "scale-b"),
            "scoped search leaked foreign rows: {:?}",
            r.iter()
                .map(|x| (x.id.clone(), x.project_id.clone()))
                .collect::<Vec<_>>()
        );
    }
}
