use crate::models::*;
use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Wrapper for a memory with its FTS5 BM25 relevance score.
#[derive(Debug, Clone)]
pub struct ScoredMemory<T> {
    pub memory: T,
    pub bm25_score: f64,
}

/// A lightweight row describing an archived memory (for `list_archived`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArchivedRow {
    pub id: String,
    pub memory_type: String,
    pub label: String,
    pub archived_at: i64,
}

/// Outcome of a garbage-collection pass over archived memories (for `gc_archived`).
///
/// `applied=false` (dry-run) lists what *would* be removed; `applied=true`
/// lists what was actually deleted.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GcReport {
    pub applied: bool,
    pub older_than_seconds: i64,
    /// `(memory_type, count)` pairs in `MemoryKind::all()` order.
    pub per_type: Vec<(String, usize)>,
    /// Deleted (or would-be-deleted) memory ids with their type.
    pub deleted: Vec<GcDeletedRow>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GcDeletedRow {
    pub memory_type: String,
    pub id: String,
}

/// One edge of a file entity's neighborhood, for `related_files_for` / the MCP
/// `related_files` tool. `direction` is relative to the queried file entity
/// (`outgoing` = it is the `from_entity`, `incoming` = it is the `to_entity`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RelatedFileEdge {
    pub other_name: String,
    pub relation_type: String,
    pub direction: String,
}

/// Aggregated feedback for one distinct query string (for `query_stats`).
/// `count` = how often the query ran; `result_count_avg` = average hits per
/// run (a low value flags a query the store fails to satisfy).
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryStatRow {
    pub query: String,
    pub count: i64,
    pub result_count_avg: f64,
    pub last_at: i64,
}

/// One day bucket of episodic memory counts (for `timeline`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TimelineRow {
    pub day: String,
    pub count: i64,
}

/// One reflection proposal awaiting human confirmation (for `list_pending_suggestions`).
/// `status` is `pending` while unconfirmed; `confirm_suggestion` promotes the
/// draft into `procedural_memories` and sets `status = "confirmed"` + `resolved_at`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReflectionSuggestionRow {
    pub id: String,
    pub project_id: String,
    pub pattern_tag: String,
    pub source_failure_ids: Vec<String>,
    pub source_preventions: Vec<String>,
    pub occurrence_count: i64,
    pub suggested_workflow_name: String,
    pub suggested_steps: Vec<String>,
    pub suggested_tags: Vec<String>,
    pub status: String,
    pub created_at: i64,
    pub resolved_at: Option<i64>,
}

/// The four memory types, mapped to their physical tables.
/// Used by lifecycle ops (archive/restore/list/consolidate) so a single
/// generic implementation serves all four tables. Table names come only from
/// this whitelist — never from user input — so interpolating them into SQL is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Episodic,
    Decision,
    Failure,
    Procedural,
}

impl MemoryKind {
    pub fn all() -> [MemoryKind; 4] {
        [
            MemoryKind::Episodic,
            MemoryKind::Decision,
            MemoryKind::Failure,
            MemoryKind::Procedural,
        ]
    }

    pub fn table(self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic_memories",
            MemoryKind::Decision => "decision_memories",
            MemoryKind::Failure => "failure_memories",
            MemoryKind::Procedural => "procedural_memories",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic",
            MemoryKind::Decision => "decision",
            MemoryKind::Failure => "failure",
            MemoryKind::Procedural => "procedural",
        }
    }

    /// Parse the `memory_type` string from MCP/CLI input.
    /// NOT named `from_str` on purpose: avoids clippy `should_implement_trait`.
    pub fn from_type_str(s: &str) -> Result<Self> {
        match s {
            "episodic" => Ok(MemoryKind::Episodic),
            "decision" => Ok(MemoryKind::Decision),
            "failure" => Ok(MemoryKind::Failure),
            "procedural" => Ok(MemoryKind::Procedural),
            other => anyhow::bail!(
                "invalid memory_type: {other} (use episodic|decision|failure|procedural)"
            ),
        }
    }

    /// Column shown as a human label in `list_archived`.
    pub fn display_col(self) -> &'static str {
        match self {
            MemoryKind::Episodic => "summary",
            MemoryKind::Decision => "title",
            MemoryKind::Failure => "incident",
            MemoryKind::Procedural => "workflow_name",
        }
    }

    /// SQL expression concatenating the text columns used for dedup hashing.
    pub fn dedup_text_expr(self) -> &'static str {
        match self {
            MemoryKind::Episodic => "summary || char(10) || content",
            MemoryKind::Decision => {
                "title || char(10) || context || char(10) || rationale || char(10) || tradeoffs"
            }
            MemoryKind::Failure => {
                "incident || char(10) || root_cause || char(10) || fix || char(10) || prevention"
            }
            MemoryKind::Procedural => "workflow_name || char(10) || steps",
        }
    }
}

/// Highest schema version produced by [`MemoryRepository::initialize_schema`].
///
/// Existing databases whose `PRAGMA user_version` is below this are migrated
/// forward; new databases are created at this version directly (no registered
/// migration runs on a fresh DB — the CREATE statements already encode it).
/// Bump this whenever a new migration is appended to
/// [`MemoryRepository::migrations`].
const CURRENT_SCHEMA_VERSION: i32 = 3;

/// DDL for the four FTS5 shadow tables. Shared by `initialize_schema` (fresh
/// DBs) and the v2 rebuild migration so the two cannot drift.
const FTS_TABLES_DDL: &str = "
    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memories_fts USING fts5(
        memory_id UNINDEXED,
        summary,
        content,
        files_touched,
        tags,
        tokenize='porter unicode61'
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS decision_memories_fts USING fts5(
        memory_id UNINDEXED,
        title,
        context,
        rationale,
        tradeoffs,
        tags,
        tokenize='porter unicode61'
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS failure_memories_fts USING fts5(
        memory_id UNINDEXED,
        incident,
        root_cause,
        fix,
        prevention,
        tags,
        tokenize='porter unicode61'
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS procedural_memories_fts USING fts5(
        memory_id UNINDEXED,
        workflow_name,
        steps,
        related_tools,
        tags,
        tokenize='porter unicode61'
    );
";

/// A single registered migration: receives the repository and either succeeds
/// or returns an error. Factored into a type alias to keep `migrations()`'s
/// signature readable (clippy::type_complexity).
type MigrationFn = fn(&MemoryRepository) -> Result<()>;

/// Repository for all memory CRUD operations with FTS5 dual-write.
///
pub struct MemoryRepository {
    pool: Pool<SqliteConnectionManager>,
    /// Filesystem path of the database (`None` for in-memory). Used to open an
    /// isolated connection for operations that cannot share the pool
    /// (e.g. `VACUUM`, which requires no other active connection).
    db_path: Option<PathBuf>,
}

impl MemoryRepository {
    /// Open (or create) the database at the given path.
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).context("failed to create database directory")?;
        }
        let manager = SqliteConnectionManager::file(db_path).with_init(Self::pragmas);
        let pool = Pool::builder()
            .build(manager)
            .context("failed to build connection pool")?;
        Ok(Self {
            pool,
            db_path: Some(db_path.to_path_buf()),
        })
    }

    /// Open an in-memory database (for testing).
    ///
    /// `max_size(1)` keeps the single in-memory database alive for the pool's
    /// lifetime — each `SqliteConnectionManager::memory()` connection would
    /// otherwise be a fresh, empty database.
    pub fn new_in_memory() -> Result<Self> {
        let manager = SqliteConnectionManager::memory().with_init(Self::pragmas);
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .context("failed to build in-memory pool")?;
        Ok(Self {
            pool,
            db_path: None,
        })
    }

    /// PRAGMAs applied to every pooled connection at init (via `with_init`).
    fn pragmas(c: &mut rusqlite::Connection) -> rusqlite::Result<()> {
        c.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;",
        )
    }

    /// Borrow a pooled connection.
    fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("failed to get pooled connection")
    }

    /// Initialize the schema: PRAGMAs, tables, FTS5 virtual tables, indexes.
    pub fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction()?;

        // --- Main memory tables ---
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                files_touched TEXT NOT NULL,
                related_commits TEXT NOT NULL,
                importance REAL DEFAULT 0,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS decision_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                context TEXT NOT NULL,
                rationale TEXT NOT NULL,
                tradeoffs TEXT NOT NULL,
                related_files TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS failure_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                incident TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                fix TEXT NOT NULL,
                prevention TEXT NOT NULL,
                severity INTEGER NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS procedural_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                workflow_name TEXT NOT NULL,
                steps TEXT NOT NULL,
                related_tools TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER
            );",
        )?;

        // --- FTS5 virtual tables ---
        tx.execute_batch(FTS_TABLES_DDL)?;

        // --- Entity + Graph tables ---
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, entity_type, name)
            );

            CREATE TABLE IF NOT EXISTS graph_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (from_entity) REFERENCES entities(id),
                FOREIGN KEY (to_entity) REFERENCES entities(id)
            );",
        )?;

        // --- Indexes ---
        // Historical de-dup (ONE-TIME): legacy DBs can carry duplicate graph
        // relations / pending suggestions that would make the unique indexes
        // below fail. These probes are skipped once the indexes exist — the
        // old unconditional DELETE/UPDATE pair ran two full-table writes on
        // EVERY `initialize_schema` call (i.e. every CLI command).
        if !Self::index_exists(&tx, "idx_graph_unique_relation")? {
            tx.execute(
                "DELETE FROM graph_relations
                 WHERE rowid NOT IN (
                     SELECT MIN(rowid) FROM graph_relations
                     GROUP BY from_entity, to_entity, relation_type
                 )",
                [],
            )?;
        }
        tx.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project_id);
            CREATE INDEX IF NOT EXISTS idx_graph_project ON graph_relations(project_id);
            CREATE INDEX IF NOT EXISTS idx_graph_from ON graph_relations(from_entity);
            CREATE INDEX IF NOT EXISTS idx_graph_to ON graph_relations(to_entity);
            CREATE INDEX IF NOT EXISTS idx_episodic_project_time ON episodic_memories(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_decision_project_time ON decision_memories(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_procedural_project_time ON procedural_memories(project_id, created_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_unique_relation ON graph_relations(from_entity, to_entity, relation_type);
            CREATE INDEX IF NOT EXISTS idx_failure_project_time ON failure_memories(project_id, created_at DESC);",
        )?;

        // Semantic retrieval vector store (populated only when `semantic` is enabled).
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id   TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                project_id  TEXT NOT NULL,
                vector      BLOB NOT NULL,
                model_id    TEXT NOT NULL,
                dim         INTEGER NOT NULL,
                created_at  INTEGER NOT NULL,
                PRIMARY KEY (memory_id, memory_type)
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_project_model
                ON memory_embeddings(project_id, model_id);",
        )?;

        // Retrieval feedback log: one row per search_memory call. Powers
        // `query_stats` / `engram queries` for hit-rate analysis. Independent of
        // the four memory kinds — deliberately outside MemoryKind / the CRUD macro.
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS query_log (
                id           TEXT PRIMARY KEY,
                project_id   TEXT NOT NULL,
                query        TEXT NOT NULL,
                memory_type  TEXT,
                result_ids   TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                created_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_log_project_time
                ON query_log(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_query_log_query
                ON query_log(project_id, query);",
        )?;

        // Reflection proposals: preventive rules distilled from recurring
        // failures, awaiting human confirmation. Deliberately a separate table
        // (NOT procedural_memories) so search_procedural never sees pending
        // proposals — they only enter main retrieval after `confirm_suggestion`
        // promotes one into procedural_memories. Independent of MemoryKind /
        // the CRUD macro, like query_log, so CREATE IF NOT EXISTS suffices
        // (no user_version bump needed on existing DBs).
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS reflection_suggestions (
                id                      TEXT PRIMARY KEY,
                project_id              TEXT NOT NULL,
                pattern_tag             TEXT NOT NULL,
                source_failure_ids      TEXT NOT NULL,
                source_preventions      TEXT NOT NULL,
                occurrence_count        INTEGER NOT NULL,
                suggested_workflow_name TEXT NOT NULL,
                suggested_steps         TEXT NOT NULL,
                suggested_tags          TEXT NOT NULL,
                status                  TEXT NOT NULL DEFAULT 'pending',
                created_at              INTEGER NOT NULL,
                resolved_at             INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_reflection_pending
                ON reflection_suggestions(project_id, status, created_at DESC);",
        )?;

        // Same one-time guard as the graph dedup above — see its comment.
        // Legacy DBs can carry duplicate pending rows per (project_id,
        // pattern_tag), which would make the partial unique index below fail.
        if !Self::index_exists(&tx, "idx_reflection_pending_unique")? {
            tx.execute(
                "UPDATE reflection_suggestions
                 SET status = 'rejected',
                     resolved_at = CAST(strftime('%s','now') AS INTEGER)
                 WHERE status = 'pending'
                   AND rowid NOT IN (
                       SELECT MIN(rowid) FROM reflection_suggestions
                       WHERE status = 'pending'
                       GROUP BY project_id, pattern_tag
                   )",
                [],
            )?;
        }
        tx.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_reflection_pending_unique
                ON reflection_suggestions(project_id, pattern_tag)
                WHERE status = 'pending';",
        )?;

        tx.commit()?;
        drop(conn);

        // Versioned migrations: stamp/advance `PRAGMA user_version` and run any
        // pending registered migrations. See `run_migrations`.
        self.run_migrations()?;

        Ok(())
    }

    /// Preprocess text for FTS5 so CJK content is searchable.
    ///
    /// The unicode61 tokenizer treats consecutive CJK characters AND any
    /// adjacent alphanumeric run as a single token, so `修复auth模块` or
    /// `用Rust写` index as one opaque token that no sub-query can match.
    /// Inserting spaces at every script boundary (CJK↔CJK and CJK↔alnum)
    /// makes each CJK character and each latin word its own token.
    /// Used for both indexing (FTS5 INSERT) and querying (FTS MATCH).
    fn preprocess_cjk(text: &str) -> String {
        let has_cjk = text.chars().any(is_cjk_character);
        if !has_cjk {
            return text.to_string();
        }

        // Estimate capacity: original length + space for CJK separators (worst case ~50% extra)
        let mut result = String::with_capacity(text.len() + text.len() / 2);
        let mut prev_cjk = false;
        let mut prev_alnum = false;

        for ch in text.chars() {
            let is_cjk = is_cjk_character(ch);
            // Non-CJK alphanumeric (latin/greek/digits…): a token character on
            // the other side of a CJK boundary.
            let is_alnum = !is_cjk && ch.is_alphanumeric();
            let needs_split = match (prev_cjk, prev_alnum, is_cjk, is_alnum) {
                // CJK after any token character, or a token character after
                // CJK — the two scripts must not fuse into one token.
                (true, _, true, _) | (true, _, _, true) | (_, true, true, _) => true,
                _ => false,
            };
            if needs_split {
                result.push(' ');
            }
            result.push(ch);
            prev_cjk = is_cjk;
            prev_alnum = is_alnum;
        }

        result
    }

    /// Sanitize a user query for safe use in FTS5 MATCH expressions.
    /// Splits the query into tokens, wraps each in double quotes (to escape
    /// FTS5 operators like AND/OR/NOT), and joins them with `AND` or `OR` so
    /// that each token is matched independently instead of as an exact phrase.
    /// Must be called AFTER preprocess_cjk for CJK support.
    ///
    /// `require_all = true` produces an AND query (all tokens must match —
    /// precision); `false` produces OR (any token matches — recall). The
    /// search paths try AND first and fall back to OR on an empty result.
    fn sanitize_fts_query(query: &str, require_all: bool) -> String {
        let processed = Self::preprocess_cjk(query);
        let tokens: Vec<&str> = processed.split_whitespace().collect();
        if tokens.is_empty() {
            return "\"\"".to_string();
        }
        let joiner = if require_all { " AND " } else { " OR " };
        tokens
            .into_iter()
            .map(|t| {
                let escaped = t.replace('"', "\"\"");
                format!("\"{}\"", escaped)
            })
            .collect::<Vec<_>>()
            .join(joiner)
    }

    /// Serialize a list column (tags/files/steps…) for FTS indexing with the
    /// same CJK preprocessing as text columns. Without it the JSON blob
    /// `["认证","auth"]` indexes CJK tags as one opaque token that no
    /// split-character query can ever match — an index/query asymmetry that
    /// silently made Chinese tags and file names unsearchable.
    fn fts_json<T: serde::Serialize>(v: &T) -> Result<String> {
        Ok(Self::preprocess_cjk(&serde_json::to_string(v)?))
    }

    /// Migrate FTS5 tables to include tags column.
    /// Drops and recreates all FTS5 virtual tables, then reindexes from main tables.
    /// Safe to call idempotently on fresh or existing databases.
    pub fn migrate_fts5_add_tags(&self) -> Result<()> {
        // Check if migration is needed by probing episodic_memories_fts for tags column
        let conn = self.conn()?;
        let needs_migration = conn
            .prepare("SELECT tags FROM episodic_memories_fts LIMIT 0")
            .is_err();
        if !needs_migration {
            return Ok(());
        }

        tracing::info!("Migrating FTS5 tables to include tags column...");
        let tx = conn.unchecked_transaction()?;

        tx.execute_batch(
            "
            DROP TABLE IF EXISTS episodic_memories_fts;
            DROP TABLE IF EXISTS decision_memories_fts;
            DROP TABLE IF EXISTS failure_memories_fts;
            DROP TABLE IF EXISTS procedural_memories_fts;
        ",
        )?;

        tx.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memories_fts USING fts5(
                memory_id UNINDEXED, summary, content, files_touched, tags,
                tokenize='porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS decision_memories_fts USING fts5(
                memory_id UNINDEXED, title, context, rationale, tradeoffs, tags,
                tokenize='porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS failure_memories_fts USING fts5(
                memory_id UNINDEXED, incident, root_cause, fix, prevention, tags,
                tokenize='porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS procedural_memories_fts USING fts5(
                memory_id UNINDEXED, workflow_name, steps, related_tools, tags,
                tokenize='porter unicode61'
            );",
        )?;

        // Backfill through Rust so text columns go through `preprocess_cjk`.
        // A plain INSERT..SELECT copies raw text, and the unicode61 tokenizer
        // treats consecutive CJK as one token — while the query side always
        // preprocesses — which would make migrated CJK content permanently
        // unsearchable.
        Self::backfill_fts_from_main(&tx)?;

        tx.commit()?;
        tracing::info!("FTS5 migration complete — tags column added.");
        Ok(())
    }

    /// Rebuild all four FTS indexes from their main tables in one transaction:
    /// removes orphan FTS rows, restores missing ones, and re-applies CJK
    /// preprocessing (which older migrations/paths may have skipped). Safe to
    /// run any time — it only ever makes the index match the main tables.
    /// Returns the number of FTS rows written.
    pub fn rebuild_fts(&self) -> Result<usize> {
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        for table in [
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            tx.execute(&format!("DELETE FROM {table}"), [])?;
        }
        let written = Self::backfill_fts_from_main(&tx)?;
        tx.commit()?;
        Ok(written)
    }

    /// Insert one FTS row per main-table row, with the same column
    /// preprocessing as the live write path (CJK split on text columns, raw
    /// JSON for list columns). Runs inside the caller's transaction.
    fn backfill_fts_from_main(tx: &rusqlite::Transaction<'_>) -> Result<usize> {
        let mut written = 0usize;

        {
            // Streams row-by-row instead of materializing the whole table:
            // the old collect() peaked at the full text of every memory.
            let mut stmt = tx.prepare(
                "SELECT rowid, id, summary, content, files_touched, tags FROM episodic_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    row.get::<_, String>(1)?,
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
                    Self::preprocess_cjk(&row.get::<_, String>(5)?),
                    row.get::<_, i64>(0)?,
                ])?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, id, title, context, rationale, tradeoffs, tags FROM decision_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6,?7)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    row.get::<_, String>(1)?,
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
                    Self::preprocess_cjk(&row.get::<_, String>(5)?),
                    Self::preprocess_cjk(&row.get::<_, String>(6)?),
                    row.get::<_, i64>(0)?,
                ])?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, id, incident, root_cause, fix, prevention, tags FROM failure_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6,?7)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    row.get::<_, String>(1)?,
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
                    Self::preprocess_cjk(&row.get::<_, String>(5)?),
                    Self::preprocess_cjk(&row.get::<_, String>(6)?),
                    row.get::<_, i64>(0)?,
                ])?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, id, workflow_name, steps, related_tools, tags FROM procedural_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    row.get::<_, String>(1)?,
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
                    Self::preprocess_cjk(&row.get::<_, String>(5)?),
                    row.get::<_, i64>(0)?,
                ])?;
                written += 1;
            }
        }

        Ok(written)
    }

    /// Add the nullable `archived_at` column to all four memory tables if missing.
    /// Idempotent: probes each table and only ALTERs when the column is absent.
    /// Fresh databases already get the column via `initialize_schema`'s CREATE TABLE.
    pub fn migrate_add_archived_at(&self) -> Result<()> {
        let conn = self.conn()?;
        for table in [
            "episodic_memories",
            "decision_memories",
            "failure_memories",
            "procedural_memories",
        ] {
            let has_col = conn
                .prepare(&format!("SELECT archived_at FROM {table} LIMIT 0"))
                .is_ok();
            if !has_col {
                tracing::info!("Migrating {table}: adding archived_at column");
                conn.execute_batch(&format!(
                    "ALTER TABLE {table} ADD COLUMN archived_at INTEGER;"
                ))?;
            }
        }
        Ok(())
    }

    // ─── Schema versioning ───────────────────────────────────────

    /// Registered migrations, ordered by their target schema version.
    ///
    /// Each entry is `(target_version, migrate_fn)` — `target_version` is the
    /// version reached *after* the migration runs. A migration only runs when
    /// `predecessor < user_version <= target` would advance it, i.e. when the
    /// current `user_version` is below its target.
    ///
    /// Migrations MUST be idempotent (re-running on an already-migrated DB is a
    /// no-op) and MUST NOT rely on an outer transaction.
    ///
    /// NOTE: the legacy `archived_at` column and the FTS5 `tags` column are NOT
    /// in this registry — they are baked into `initialize_schema`'s CREATE
    /// statements. New databases get them directly. Legacy databases (detected
    /// at `user_version == 0`) have these two historical migrations applied
    /// idempotently by [`Self::run_migrations`] before being stamped to
    /// `CURRENT_SCHEMA_VERSION`; the FTS rebuild only fires when the `tags`
    /// column is actually missing.
    fn migrations() -> &'static [(i32, MigrationFn)] {
        static REG: [(i32, MigrationFn); 2] = [
            (2, MemoryRepository::migrate_v2_fts_rowid),
            (3, MemoryRepository::migrate_v3_reflection_rearm),
        ];
        &REG
    }

    /// v3 — reflection re-arm bookkeeping: `rejected_occurrence_count`
    /// records how much evidence had accumulated when a proposal was
    /// rejected, so a tag can be re-proposed once *new* evidence arrives
    /// (previously a single rejection silenced the tag forever).
    fn migrate_v3_reflection_rearm(&self) -> Result<()> {
        let conn = self.conn()?;
        let has_col = conn
            .prepare("SELECT rejected_occurrence_count FROM reflection_suggestions LIMIT 0")
            .is_ok();
        if !has_col {
            conn.execute_batch(
                "ALTER TABLE reflection_suggestions \
                 ADD COLUMN rejected_occurrence_count INTEGER;",
            )?;
        }
        Ok(())
    }

    /// v2 — rebuild all FTS5 shadow tables. Two reasons, one rebuild:
    ///
    /// 1. **Rowid alignment**: pre-v2 FTS rows carried auto-assigned rowids;
    ///    the write path now keeps the FTS rowid in lockstep with the main
    ///    table's rowid so deletes/updates address FTS rows in O(log n). The
    ///    old `WHERE memory_id = ?` delete on the UNINDEXED column was a full
    ///    FTS scan per update/delete/GC row (verified via EXPLAIN QUERY PLAN).
    /// 2. **Tokenizer preprocessing refresh**: CJK↔latin boundaries are now
    ///    split and tags/files columns are preprocessed, so mixed-script
    ///    content indexed before v2 is searchable.
    ///
    /// Drop + recreate + streaming backfill in one transaction. Idempotent by
    /// construction (rebuilding from the main tables).
    fn migrate_v2_fts_rowid(&self) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        for table in [
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            tx.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
        }
        tx.execute_batch(FTS_TABLES_DDL)?;
        let written = Self::backfill_fts_from_main(&tx)?;
        tx.commit()?;
        tracing::info!("migration v2: FTS rebuilt rowid-aligned ({written} rows)");
        Ok(())
    }

    /// Advance the database schema to [`CURRENT_SCHEMA_VERSION`].
    ///
    /// Safe mapping of the legacy (un-versioned) world:
    /// - `user_version == CURRENT` → nothing to do.
    /// - `user_version == 0` with memory tables already present → a legacy
    ///   database created/migrated by the older probe-based code (which never
    ///   set `user_version`). Run the two historical idempotent migrations
    ///   (`archived_at`, FTS `tags`) to ensure completeness, then stamp to
    ///   CURRENT. Both probe before altering, so an already-complete DB is
    ///   untouched — the FTS rebuild only fires when the `tags` column is
    ///   actually missing.
    /// - `user_version == 0` with no memory tables → fresh DB just created by
    ///   `initialize_schema`; stamped to CURRENT.
    /// - `0 < user_version < CURRENT` → run registered migrations forward,
    ///   bumping the version after each successful migration.
    pub fn run_migrations(&self) -> Result<()> {
        let current = self.user_version()?;
        if current == CURRENT_SCHEMA_VERSION {
            return Ok(());
        }

        if current == 0 {
            if self.has_memory_tables()? {
                // Legacy DB created/migrated by the older probe-based code
                // (which never set user_version). Its schema may be incomplete:
                // the FTS5 `tags` column was historically added only by an
                // explicit startup call that not every entry point made. Both
                // historical migrations probe before altering, so an
                // already-complete DB is untouched.
                tracing::info!(
                    "schema migration: legacy DB at user_version=0; applying \
                     idempotent historical migrations then stamping to \
                     v{CURRENT_SCHEMA_VERSION}"
                );
                self.migrate_add_archived_at()?;
                self.migrate_fts5_add_tags()?;
                // Stamp to v1 (the world those probes produced), then fall
                // through to the registered-migration loop below so legacy
                // DBs also receive later rebuilds (v2 rowid alignment).
                self.set_user_version(1)?;
            } else {
                // Fresh DB: initialize_schema already created the current
                // schema (FTS tables empty; rows will be written by the
                // rowid-aligned code path). No rebuild needed.
                tracing::info!("schema migration: fresh DB; stamping to v{CURRENT_SCHEMA_VERSION}");
                self.set_user_version(CURRENT_SCHEMA_VERSION)?;
                return Ok(());
            }
        }

        for &(target, migrate_fn) in Self::migrations() {
            if target > current && target <= CURRENT_SCHEMA_VERSION {
                tracing::info!("schema migration: applying migration to v{target}");
                migrate_fn(self)?;
                self.set_user_version(target)?;
            }
        }
        Ok(())
    }

    /// Read the current `PRAGMA user_version`.
    pub fn user_version(&self) -> Result<i32> {
        let conn = self.conn()?;
        let v: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        Ok(v)
    }

    /// Stamp `PRAGMA user_version` to `v`. Uses `execute_batch` — the pragma
    /// helper wraps the call in a savepoint, which is unnecessary here.
    pub fn set_user_version(&self, v: i32) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(&format!("PRAGMA user_version = {v};"))?;
        Ok(())
    }

    /// Whether a named index exists (checked against sqlite_master).
    fn index_exists(conn: &rusqlite::Connection, name: &str) -> Result<bool> {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?1)",
            params![name],
            |r| r.get(0),
        )?;
        Ok(exists)
    }

    /// Whether the legacy memory tables exist (heuristic for "was this DB
    /// created/migrated by the older probe-based code at user_version=0").
    fn has_memory_tables(&self) -> Result<bool> {
        let conn = self.conn()?;
        let exists = conn
            .prepare("SELECT 1 FROM episodic_memories LIMIT 0")
            .is_ok();
        Ok(exists)
    }

    // ─── Entity Linking Helper ────────────────────────────────────

    /// Ensure entities and relations exist for a memory's linked items (files/tools).
    /// Upserts a Memory entity for the record itself, then creates File/Tool entities
    /// and links them via the specified relation type. All within the caller's transaction.
    fn ensure_linked_entities(
        tx: &rusqlite::Transaction,
        project_id: &str,
        memory_id: &str,
        entity_type: &str,
        names: &[String],
        relation_type: &str,
        now: i64,
    ) -> Result<()> {
        // Upsert a Memory entity for the record itself
        tx.execute(
            "INSERT INTO entities (id, project_id, entity_type, name, metadata, created_at, updated_at)
             VALUES (?1, ?2, 'Memory', ?3, '{}', ?4, ?4)
             ON CONFLICT(project_id, entity_type, name) DO UPDATE SET updated_at = excluded.updated_at",
            params![memory_id, project_id, memory_id, now],
        )?;

        for name in names {
            // Upsert entity and retrieve its id via RETURNING — eliminates the
            // extra SELECT that previously caused N+1 queries.
            let eid = uuid::Uuid::new_v4().to_string();
            let actual_id: String = tx.query_row(
                "INSERT INTO entities (id, project_id, entity_type, name, metadata, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, '{}', ?5, ?5)
                 ON CONFLICT(project_id, entity_type, name) DO UPDATE SET updated_at = excluded.updated_at
                 RETURNING id",
                params![eid, project_id, entity_type, name, now],
                |row| row.get(0),
            )?;
            // Insert relation with dedup via unique index
            tx.execute(
                "INSERT INTO graph_relations (project_id, from_entity, to_entity, relation_type, weight, created_at)
                 VALUES (?1, ?2, ?3, ?4, 1.0, ?5)
                 ON CONFLICT(from_entity, to_entity, relation_type) DO NOTHING",
                params![project_id, memory_id, actual_id, relation_type, now],
            )?;
        }
        Ok(())
    }

    // ─── Macro-generated Memory CRUD + FTS5 Search ─────────────────

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,
        link_ts = link_ts,
        fts_rowid = fts_rowid,

        struct_type = EpisodicMemory,
        table = "episodic_memories",
        fts_table = "episodic_memories_fts",

        create_fn = create_episodic,
        get_fn = get_episodic,
        update_fn = update_episodic,
        delete_fn = delete_episodic,
        search_fn = search_episodic,
        list_active_fn = list_active_episodic,

        select_cols = "id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at",
        search_cols = "m.id, m.project_id, m.session_id, m.summary, m.content, m.files_touched, m.related_commits, m.importance, m.tags, m.created_at, m.updated_at",

        insert_sql = "INSERT INTO episodic_memories (id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",

        fts_insert_sql = "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags, rowid) VALUES (?1,?2,?3,?4,?5,?6)",

        update_sql = "UPDATE episodic_memories SET session_id=?3, summary=?4, content=?5, files_touched=?6, related_commits=?7, importance=?8, tags=?9, updated_at=?10 WHERE id=?1 AND project_id=?2",

        score_col_idx = 11,

        insert_params = {
            params![
                mem.id, mem.project_id, mem.session_id,
                mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.related_commits)?,
                mem.importance,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ]
        },

        fts_params = {
            params![
                mem.id,
                Self::preprocess_cjk(&mem.summary),
                Self::preprocess_cjk(&mem.content),
                Self::fts_json(&mem.files_touched)?,
                Self::fts_json(&mem.tags)?,
                fts_rowid,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.session_id,
                mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.related_commits)?,
                mem.importance,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ]
        },

        row_mapper = {
            EpisodicMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                session_id: row.get(2)?,
                summary: row.get(3)?,
                content: row.get(4)?,
                files_touched: row_get_json!(row, 5, Vec<String>),
                related_commits: row_get_json!(row, 6, Vec<String>),
                importance: row.get(7)?,
                tags: row_get_json!(row, 8, Vec<String>),
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            }
        },

        entity_link = {
            if !mem.files_touched.is_empty() {
                Self::ensure_linked_entities(
                    &tx, &mem.project_id, &mem.id, "File", &mem.files_touched,
                    "Touches", link_ts,
                )?;
            }
        },
    }

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,
        link_ts = link_ts,
        fts_rowid = fts_rowid,

        struct_type = DecisionMemory,
        table = "decision_memories",
        fts_table = "decision_memories_fts",

        create_fn = create_decision,
        get_fn = get_decision,
        update_fn = update_decision,
        delete_fn = delete_decision,
        search_fn = search_decisions,
        list_active_fn = list_active_decision,

        select_cols = "id, project_id, title, context, rationale, tradeoffs, related_files, tags, created_at, updated_at",
        search_cols = "m.id, m.project_id, m.title, m.context, m.rationale, m.tradeoffs, m.related_files, m.tags, m.created_at, m.updated_at",

        insert_sql = "INSERT INTO decision_memories (id, project_id, title, context, rationale, tradeoffs, related_files, tags, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",

        fts_insert_sql = "INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs, tags, rowid) VALUES (?1,?2,?3,?4,?5,?6,?7)",

        update_sql = "UPDATE decision_memories SET title=?3, context=?4, rationale=?5, tradeoffs=?6, related_files=?7, tags=?8, updated_at=?9 WHERE id=?1 AND project_id=?2",

        score_col_idx = 10,

        insert_params = {
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ]
        },

        fts_params = {
            params![
                mem.id,
                Self::preprocess_cjk(&mem.title),
                Self::preprocess_cjk(&mem.context),
                Self::preprocess_cjk(&mem.rationale),
                Self::preprocess_cjk(&mem.tradeoffs),
                Self::fts_json(&mem.tags)?,
                fts_rowid,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ]
        },

        row_mapper = {
            DecisionMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                title: row.get(2)?,
                context: row.get(3)?,
                rationale: row.get(4)?,
                tradeoffs: row.get(5)?,
                related_files: row_get_json!(row, 6, Vec<String>),
                tags: row_get_json!(row, 7, Vec<String>),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            }
        },

        entity_link = {
            if !mem.related_files.is_empty() {
                Self::ensure_linked_entities(
                    &tx, &mem.project_id, &mem.id, "File", &mem.related_files,
                    "References", link_ts,
                )?;
            }
        },
    }

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,
        link_ts = link_ts,
        fts_rowid = fts_rowid,

        struct_type = FailureMemory,
        table = "failure_memories",
        fts_table = "failure_memories_fts",

        create_fn = create_failure,
        get_fn = get_failure,
        update_fn = update_failure,
        delete_fn = delete_failure,
        search_fn = search_failures,
        list_active_fn = list_active_failure,

        select_cols = "id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at",
        search_cols = "m.id, m.project_id, m.incident, m.root_cause, m.fix, m.prevention, m.severity, m.tags, m.created_at, m.updated_at",

        insert_sql = "INSERT INTO failure_memories (id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",

        fts_insert_sql = "INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention, tags, rowid) VALUES (?1,?2,?3,?4,?5,?6,?7)",

        update_sql = "UPDATE failure_memories SET incident=?3, root_cause=?4, fix=?5, prevention=?6, severity=?7, tags=?8, updated_at=?9 WHERE id=?1 AND project_id=?2",

        score_col_idx = 10,

        insert_params = {
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ]
        },

        fts_params = {
            params![
                mem.id,
                Self::preprocess_cjk(&mem.incident),
                Self::preprocess_cjk(&mem.root_cause),
                Self::preprocess_cjk(&mem.fix),
                Self::preprocess_cjk(&mem.prevention),
                Self::fts_json(&mem.tags)?,
                fts_rowid,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ]
        },

        row_mapper = {
            FailureMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                incident: row.get(2)?,
                root_cause: row.get(3)?,
                fix: row.get(4)?,
                prevention: row.get(5)?,
                severity: row.get(6)?,
                tags: row_get_json!(row, 7, Vec<String>),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            }
        },

        entity_link = {},
    }

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,
        link_ts = link_ts,
        fts_rowid = fts_rowid,

        struct_type = ProceduralMemory,
        table = "procedural_memories",
        fts_table = "procedural_memories_fts",

        create_fn = create_procedural,
        get_fn = get_procedural,
        update_fn = update_procedural,
        delete_fn = delete_procedural,
        search_fn = search_procedural,
        list_active_fn = list_active_procedural,

        select_cols = "id, project_id, workflow_name, steps, related_tools, tags, created_at, updated_at",
        search_cols = "m.id, m.project_id, m.workflow_name, m.steps, m.related_tools, m.tags, m.created_at, m.updated_at",

        insert_sql = "INSERT INTO procedural_memories (id, project_id, workflow_name, steps, related_tools, tags, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",

        fts_insert_sql = "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools, tags, rowid) VALUES (?1,?2,?3,?4,?5,?6)",

        update_sql = "UPDATE procedural_memories SET workflow_name=?3, steps=?4, related_tools=?5, tags=?6, updated_at=?7 WHERE id=?1 AND project_id=?2",

        score_col_idx = 8,

        insert_params = {
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ]
        },

        fts_params = {
            params![
                mem.id,
                Self::preprocess_cjk(&mem.workflow_name),
                Self::fts_json(&mem.steps)?,
                Self::fts_json(&mem.related_tools)?,
                Self::fts_json(&mem.tags)?,
                fts_rowid,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ]
        },

        row_mapper = {
            ProceduralMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                workflow_name: row.get(2)?,
                steps: row_get_json!(row, 3, Vec<String>),
                related_tools: row_get_json!(row, 4, Vec<String>),
                tags: row_get_json!(row, 5, Vec<String>),
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            }
        },

        entity_link = {
            if !mem.related_tools.is_empty() {
                Self::ensure_linked_entities(
                    &tx, &mem.project_id, &mem.id, "Tool", &mem.related_tools,
                    "Uses", link_ts,
                )?;
            }
        },
    }

    // ─── Lifecycle: archive / restore (soft delete) ────────────────

    /// Soft-delete (archive) a memory by id within a project.
    /// Returns true if a currently-active row was archived.
    pub fn archive(&self, kind: MemoryKind, id: &str, project_id: &str, now: i64) -> Result<bool> {
        let sql = format!(
            "UPDATE {} SET archived_at = ?3 WHERE id = ?1 AND project_id = ?2 AND archived_at IS NULL",
            kind.table()
        );
        let conn = self.conn()?;
        let affected = conn.execute(&sql, params![id, project_id, now])?;
        Ok(affected > 0)
    }

    /// Un-archive a previously soft-deleted memory.
    /// Returns true if a currently-archived row was restored.
    pub fn restore(&self, kind: MemoryKind, id: &str, project_id: &str) -> Result<bool> {
        let sql = format!(
            "UPDATE {} SET archived_at = NULL WHERE id = ?1 AND project_id = ?2 AND archived_at IS NOT NULL",
            kind.table()
        );
        let conn = self.conn()?;
        let affected = conn.execute(&sql, params![id, project_id])?;
        Ok(affected > 0)
    }

    /// Active memory ids matching the batch filters (read-only; used for dry-run).
    ///
    /// - `tags`: if non-empty, the memory must carry at least one of these tags.
    /// - `before`: if Some, only memories with `created_at < before`.
    ///
    /// Filtering is done in Rust (tags are JSON-encoded); fine for manual/dry-run scale.
    pub fn list_active_candidates(
        &self,
        kind: MemoryKind,
        project_id: &str,
        tags: &[String],
        before: Option<i64>,
    ) -> Result<Vec<String>> {
        let sql = format!(
            "SELECT id, tags, created_at FROM {} WHERE project_id = ?1 AND archived_at IS NULL",
            kind.table()
        );
        let conn = self.conn()?;
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        let mut matched = Vec::new();
        for row in rows {
            let (id, tags_json, created_at) = row?;
            if let Some(b) = before {
                if created_at >= b {
                    continue;
                }
            }
            if !tags.is_empty() {
                let mem_tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                if !mem_tags.iter().any(|t| tags.contains(t)) {
                    continue;
                }
            }
            matched.push(id);
        }
        Ok(matched)
    }

    /// Archive all active memories of a kind matching the filters.
    ///
    /// - `tags`: if non-empty, the memory must carry at least one of these tags.
    /// - `before`: if Some, only memories with `created_at < before`.
    ///
    /// Returns the ids that were archived. All updates run in one transaction.
    pub fn archive_batch(
        &self,
        kind: MemoryKind,
        project_id: &str,
        tags: &[String],
        before: Option<i64>,
        now: i64,
    ) -> Result<Vec<String>> {
        let matched = self.list_active_candidates(kind, project_id, tags, before)?;
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction()?;
        let update_sql = format!(
            "UPDATE {} SET archived_at = ?2 WHERE id = ?1 AND archived_at IS NULL",
            kind.table()
        );
        for id in &matched {
            tx.execute(&update_sql, params![id, now])?;
        }
        tx.commit()?;
        Ok(matched)
    }

    /// List archived memories of a kind, newest-archived first.
    pub fn list_archived(
        &self,
        kind: MemoryKind,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<ArchivedRow>> {
        let sql = format!(
            "SELECT id, {} AS label, archived_at FROM {} \
             WHERE project_id = ?1 AND archived_at IS NOT NULL \
             ORDER BY archived_at DESC LIMIT ?2",
            kind.display_col(),
            kind.table()
        );
        let conn = self.conn()?;
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![project_id, limit], |row| {
            Ok(ArchivedRow {
                id: row.get(0)?,
                memory_type: kind.as_str().to_string(),
                label: row.get(1)?,
                archived_at: row.get(2)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ─── Retrieval feedback (query_log) ─────────────────────────────
    /// Record one `search_memory` invocation. Best-effort by design — callers
    /// swallow the error so a logging failure never breaks search.
    pub fn record_query(
        &self,
        project_id: &str,
        query: &str,
        result_ids: &[String],
        memory_type: Option<&str>,
        now: i64,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO query_log \
             (id, project_id, query, memory_type, result_ids, result_count, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                uuid::Uuid::new_v4().to_string(),
                project_id,
                query,
                memory_type,
                serde_json::to_string(result_ids)?,
                result_ids.len() as i64,
                now,
            ],
        )?;
        Ok(())
    }

    /// Aggregate `query_log` by query string over a time window: how often each
    /// query ran and its average hit count, most-frequent first. A low
    /// `result_count_avg` flags queries the store fails to satisfy.
    pub fn query_stats(
        &self,
        project_id: &str,
        since: i64,
        limit: usize,
    ) -> Result<Vec<QueryStatRow>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT query, COUNT(*) AS cnt, AVG(result_count) AS avg_hits, \
             MAX(created_at) AS last_at FROM query_log \
             WHERE project_id = ?1 AND created_at >= ?2 \
             GROUP BY query ORDER BY cnt DESC, last_at DESC LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![project_id, since, limit], |row| {
            Ok(QueryStatRow {
                query: row.get(0)?,
                count: row.get(1)?,
                result_count_avg: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                last_at: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Group non-archived episodic memories by UTC day over a time window,
    /// most-recent day first. `since` is an absolute unix timestamp computed
    /// by the caller.
    pub fn timeline(&self, project_id: &str, since: i64) -> Result<Vec<TimelineRow>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
             FROM episodic_memories
             WHERE project_id = ?1 AND created_at >= ?2 AND archived_at IS NULL
             GROUP BY day ORDER BY day DESC",
        )?;
        let rows = stmt.query_map(params![project_id, since], |row| {
            Ok(TimelineRow {
                day: row.get(0)?,
                count: row.get(1)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
    /// Check which commit hashes have already been ingested as episodic memories.
    /// Returns a HashSet of already-ingested commit hashes for O(1) lookup.
    /// Archived memories are excluded, so archiving old session memories no
    /// longer permanently blocks re-importing their commits.
    pub fn get_ingested_commits(&self, project_id: &str) -> Result<HashSet<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT related_commits FROM episodic_memories \
             WHERE project_id = ?1 AND archived_at IS NULL",
        )?;
        let rows = stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;
        let mut hashes = HashSet::new();
        for row in rows {
            let json_str = row?;
            let commits: Vec<String> = serde_json::from_str(&json_str).unwrap_or_default();
            hashes.extend(commits);
        }
        Ok(hashes)
    }

    /// List recent failure memories for a project, ordered by creation time descending.
    /// Uses plain SELECT without FTS5 MATCH — safe for listing without a filter.
    pub fn list_recent_failures(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<FailureMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at
             FROM failure_memories
             WHERE project_id = ?1 AND archived_at IS NULL
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project_id, limit], |row| {
            Ok(FailureMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                incident: row.get(2)?,
                root_cause: row.get(3)?,
                fix: row.get(4)?,
                prevention: row.get(5)?,
                severity: row.get(6)?,
                tags: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// List recent decision memories for a project, ordered by creation time descending.
    /// Uses plain SELECT without FTS5 MATCH — safe for listing without a filter.
    pub fn list_recent_decisions(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<DecisionMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, title, context, rationale, tradeoffs, related_files, tags, created_at, updated_at
             FROM decision_memories
             WHERE project_id = ?1 AND archived_at IS NULL
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project_id, limit], |row| {
            Ok(DecisionMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                title: row.get(2)?,
                context: row.get(3)?,
                rationale: row.get(4)?,
                tradeoffs: row.get(5)?,
                related_files: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                tags: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ─── Reflection (recurring-failure → preventive rule) ─────────

    /// All active (non-archived) failure memories for a project. The reflection
    /// engine is the sole full-scan consumer; [`list_recent_failures`] is a
    /// bounded "recent N" list, so this gets its own unbounded accessor.
    pub fn list_active_failures_for_reflection(
        &self,
        project_id: &str,
    ) -> Result<Vec<FailureMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at
             FROM failure_memories
             WHERE project_id = ?1 AND archived_at IS NULL",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            Ok(FailureMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                incident: row.get(2)?,
                root_cause: row.get(3)?,
                fix: row.get(4)?,
                prevention: row.get(5)?,
                severity: row.get(6)?,
                tags: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Insert a reflection proposal (status='pending'). The caller de-duplicates
    /// via [`has_suggestion_for_tag`] before calling.
    pub fn insert_reflection_suggestion(&self, row: &ReflectionSuggestionRow) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO reflection_suggestions \
             (id, project_id, pattern_tag, source_failure_ids, source_preventions, \
              occurrence_count, suggested_workflow_name, suggested_steps, suggested_tags, \
              status, created_at, resolved_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                row.id,
                row.project_id,
                row.pattern_tag,
                serde_json::to_string(&row.source_failure_ids)?,
                serde_json::to_string(&row.source_preventions)?,
                row.occurrence_count,
                row.suggested_workflow_name,
                serde_json::to_string(&row.suggested_steps)?,
                serde_json::to_string(&row.suggested_tags)?,
                row.status,
                row.created_at,
                row.resolved_at,
            ],
        )?;
        Ok(())
    }

    /// Whether a pending proposal already exists for `(project_id, pattern_tag)`.
    /// Keeps `reflect` idempotent across repeated runs.
    pub fn has_pending_suggestion(&self, project_id: &str, pattern_tag: &str) -> Result<bool> {
        let conn = self.conn()?;
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM reflection_suggestions \
             WHERE project_id = ?1 AND pattern_tag = ?2 AND status = 'pending')",
            params![project_id, pattern_tag],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    /// Whether ANY proposal (pending, confirmed, or rejected) exists for
    /// `(project_id, pattern_tag)`. The reflection engine uses this so a tag
    /// the user already decided on — either way — is never re-proposed:
    /// rejecting must not let the same suggestion resurrect on the next
    /// `reflect --apply`, and confirming must not re-propose the promoted rule.
    pub fn has_suggestion_for_tag(
        &self,
        project_id: &str,
        pattern_tag: &str,
        current_occurrences: i64,
        min_occurrences: i64,
    ) -> Result<bool> {
        let conn = self.conn()?;
        // pending/confirmed proposals always block a duplicate. A REJECTED
        // proposal blocks only until enough NEW evidence accumulates: the
        // reject stamped the occurrence count it saw; once failures have
        // grown by another `min_occurrences`, the pattern is re-armable.
        let blocked: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM reflection_suggestions \
             WHERE project_id = ?1 AND pattern_tag = ?2 \
               AND (status != 'rejected' \
                    OR COALESCE(rejected_occurrence_count, occurrence_count) + ?3 > ?4))",
            params![
                project_id,
                pattern_tag,
                min_occurrences,
                current_occurrences
            ],
            |row| row.get(0),
        )?;
        Ok(blocked)
    }

    /// All distinct project ids that have at least one memory (any kind).
    /// Ordered for deterministic maintenance output.
    pub fn list_projects(&self) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT project_id FROM (
                SELECT project_id FROM episodic_memories
                UNION SELECT project_id FROM decision_memories
                UNION SELECT project_id FROM failure_memories
                UNION SELECT project_id FROM procedural_memories
            ) ORDER BY project_id",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Delete `query_log` rows older than `older_than_seconds`. `query_log`
    /// grows by one row per search and has no natural bound; maintenance
    /// prunes it so the DB doesn't bloat indefinitely. Returns rows removed.
    pub fn prune_query_log(&self, older_than_seconds: i64, now: i64) -> Result<usize> {
        let threshold = now.checked_sub(older_than_seconds).unwrap_or(0);
        let conn = self.conn()?;
        let removed = conn.execute(
            "DELETE FROM query_log WHERE created_at < ?1",
            params![threshold],
        )?;
        Ok(removed)
    }

    /// All pending proposals for a project, newest first.
    pub fn list_pending_suggestions(
        &self,
        project_id: &str,
    ) -> Result<Vec<ReflectionSuggestionRow>> {
        Self::query_suggestions(
            &self.conn()?,
            "WHERE project_id = ?1 AND status = 'pending' ORDER BY created_at DESC",
            params![project_id],
        )
    }

    /// Confirm a proposal: promote its draft into `procedural_memories` and mark
    /// the proposal `confirmed` + `resolved_at` — both inside ONE transaction on
    /// ONE connection, so a crash can no longer leave "procedural created but
    /// suggestion still pending". The promoted id is `"{suggestion_id}-proc"`;
    /// if that row already exists (a previous attempt crashed after INSERT),
    /// the insert is skipped and the suggestion is simply marked confirmed —
    /// making confirm idempotent under retries. Returns `None` if no matching
    /// pending proposal was found (already resolved / wrong project / unknown id).
    pub fn confirm_suggestion(
        &self,
        id: &str,
        project_id: &str,
        now: i64,
    ) -> Result<Option<String>> {
        let conn = self.conn()?;
        // Read the pending draft first (None if already resolved / wrong project).
        let draft = match Self::query_suggestion(
            &conn,
            "WHERE id = ?1 AND project_id = ?2",
            params![id, project_id],
        )? {
            Some(d) if d.status == "pending" => d,
            _ => return Ok(None),
        };
        if draft.suggested_steps.is_empty() {
            anyhow::bail!(
                "suggestion '{}' has no steps; reject it instead of confirming an empty rule",
                draft.id
            );
        }

        let proc_id = format!("{}-proc", draft.id);
        let tx = conn.unchecked_transaction()?;

        // Idempotent promote: skip the INSERT if a crashed earlier attempt
        // already created the procedural memory.
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM procedural_memories WHERE id = ?1)",
            params![proc_id],
            |row| row.get(0),
        )?;
        if !exists {
            tx.execute(
                "INSERT INTO procedural_memories \
                 (id, project_id, workflow_name, steps, related_tools, tags, created_at, updated_at) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?7)",
                params![
                    proc_id,
                    draft.project_id,
                    draft.suggested_workflow_name,
                    serde_json::to_string(&draft.suggested_steps)?,
                    serde_json::to_string(&Vec::<String>::new())?,
                    serde_json::to_string(&draft.suggested_tags)?,
                    now,
                ],
            )?;
            // Mirror the standard create path's FTS dual-write: same
            // preprocessing AND rowid alignment, so this hand-rolled path
            // stays consistent with the macro-generated one.
            let fts_rowid: i64 = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools, tags, rowid) \
                 VALUES (?1,?2,?3,?4,?5,?6)",
                params![
                    proc_id,
                    Self::preprocess_cjk(&draft.suggested_workflow_name),
                    Self::fts_json(&draft.suggested_steps)?,
                    Self::fts_json(&Vec::<String>::new())?,
                    Self::fts_json(&draft.suggested_tags)?,
                    fts_rowid,
                ],
            )?;
        }

        let confirmed = tx.execute(
            "UPDATE reflection_suggestions SET status = 'confirmed', resolved_at = ?1 \
             WHERE id = ?2 AND project_id = ?3 AND status = 'pending'",
            params![now, id, project_id],
        )?;
        tx.commit()?;

        if confirmed == 0 {
            // Lost a race with a concurrent confirm/reject between the read and
            // the UPDATE. Report "not pending" rather than falsely claiming success.
            return Ok(None);
        }
        Ok(Some(proc_id))
    }

    /// Reject a pending proposal: mark `rejected` + `resolved_at`. No procedural
    /// memory is created. Returns false if no matching pending proposal exists.
    pub fn reject_suggestion(&self, id: &str, project_id: &str, now: i64) -> Result<bool> {
        let conn = self.conn()?;
        let affected = conn.execute(
            "UPDATE reflection_suggestions SET status = 'rejected', resolved_at = ?1, \
             rejected_occurrence_count = occurrence_count \
             WHERE id = ?2 AND project_id = ?3 AND status = 'pending'",
            params![now, id, project_id],
        )?;
        Ok(affected > 0)
    }

    /// Shared row mapper for a single suggestion SELECT.
    fn query_suggestion(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        where_clause: &str,
        params: impl rusqlite::Params,
    ) -> Result<Option<ReflectionSuggestionRow>> {
        let sql = format!(
            "SELECT id, project_id, pattern_tag, source_failure_ids, source_preventions, \
                    occurrence_count, suggested_workflow_name, suggested_steps, suggested_tags, \
                    status, created_at, resolved_at
             FROM reflection_suggestions {where_clause}"
        );
        let row = conn
            .query_row(&sql, params, Self::map_suggestion_row)
            .optional()?;
        Ok(row)
    }

    /// Shared row mapper for a multi-row suggestion SELECT.
    fn query_suggestions(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        where_clause: &str,
        params: impl rusqlite::Params,
    ) -> Result<Vec<ReflectionSuggestionRow>> {
        let sql = format!(
            "SELECT id, project_id, pattern_tag, source_failure_ids, source_preventions, \
                    occurrence_count, suggested_workflow_name, suggested_steps, suggested_tags, \
                    status, created_at, resolved_at
             FROM reflection_suggestions {where_clause}"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params, Self::map_suggestion_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Map a rusqlite Row to a [`ReflectionSuggestionRow`]. JSON columns
    /// degrade to defaults on parse failure (corrupted cell), matching the
    /// `row_get_json!` convention used elsewhere.
    fn map_suggestion_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ReflectionSuggestionRow> {
        Ok(ReflectionSuggestionRow {
            id: row.get(0)?,
            project_id: row.get(1)?,
            pattern_tag: row.get(2)?,
            source_failure_ids: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
            source_preventions: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            occurrence_count: row.get(5)?,
            suggested_workflow_name: row.get(6)?,
            suggested_steps: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
            suggested_tags: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
            status: row.get(9)?,
            created_at: row.get(10)?,
            resolved_at: row.get(11)?,
        })
    }

    // ─── Garbage Collection ───────────────────────────────────────

    /// Physically delete archived memories older than `older_than_seconds`.
    ///
    /// Candidates are rows with `archived_at IS NOT NULL AND
    /// archived_at < now - older_than_seconds`. When `older_than_seconds <= 0`,
    /// *all* archived rows match (the CLI `--all` mode). `apply=false` is a
    /// dry run: nothing is deleted, but the report lists what *would* go.
    /// Deleting also purges the FTS index, graph relations/entities, and any
    /// stored embedding (see the `delete_*` macros in `storage/macros.rs`).
    ///
    /// Checkpoint/vacuum are deliberately separate (`wal_checkpoint_truncate`,
    /// `vacuum`) so callers control when free space is actually reclaimed.
    pub fn gc_archived(&self, older_than_seconds: i64, apply: bool, now: i64) -> Result<GcReport> {
        // `<= 0` means "all archived": use a threshold nothing is below.
        // checked_sub avoids the debug-panic / release-wrap on absurd durations.
        let threshold = if older_than_seconds > 0 {
            now.checked_sub(older_than_seconds).unwrap_or(i64::MIN)
        } else {
            i64::MAX
        };

        let mut per_type = Vec::new();
        let mut deleted = Vec::new();

        for kind in MemoryKind::all() {
            // Collect candidates first so we don't hold a read cursor while
            // the purge transaction runs.
            let candidates: Vec<(String, String, i64)> = {
                let conn = self.conn()?;
                let sql = format!(
                    "SELECT id, project_id, rowid FROM {} \
                     WHERE archived_at IS NOT NULL AND archived_at < ?1",
                    kind.table()
                );
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(params![threshold], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })?;
                rows.collect::<Result<Vec<_>, _>>()?
            };

            let kind_str = kind.as_str().to_string();
            let mut removed_rows: Vec<GcDeletedRow> = Vec::new();

            if apply && !candidates.is_empty() {
                // One transaction per kind (not per row): a batch GC commits
                // once instead of N times. The `archived_at IS NOT NULL` guard
                // stays in the DELETE itself, so a memory restored by another
                // process between collection and purge is NOT destroyed —
                // `engram gc` is a separate CLI process sharing the DB with
                // the live MCP server, and this closes a silent-data-loss
                // race that does not need worker_threads > 1 to trigger.
                let fts_table = kind_fts_table(kind);
                let conn = self.conn()?;
                let tx = conn.unchecked_transaction()?;
                for (id, project_id, fts_rowid) in &candidates {
                    let affected = tx.execute(
                        &format!(
                            "DELETE FROM {} WHERE id = ?1 AND project_id = ?2 \
                             AND archived_at IS NOT NULL",
                            kind.table()
                        ),
                        params![id, project_id],
                    )?;
                    if affected > 0 {
                        // FTS rows are addressed by the main-table rowid —
                        // matching the UNINDEXED memory_id column scanned the
                        // whole FTS table for every GC'd row.
                        tx.execute(
                            &format!("DELETE FROM {} WHERE rowid = ?1", fts_table),
                            params![fts_rowid],
                        )?;
                        tx.execute(
                            "DELETE FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
                            params![id],
                        )?;
                        tx.execute("DELETE FROM entities WHERE id = ?1", params![id])?;
                        tx.execute(
                            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                            params![id],
                        )?;
                        removed_rows.push(GcDeletedRow {
                            memory_type: kind_str.clone(),
                            id: id.clone(),
                        });
                    }
                }
                tx.commit()?;
            } else if !apply {
                // Dry run: report every candidate as "would delete".
                removed_rows = candidates
                    .into_iter()
                    .map(|(id, _, _)| GcDeletedRow {
                        memory_type: kind_str.clone(),
                        id,
                    })
                    .collect();
            }

            // per_type counts what was (or would be) actually removed —
            // identical to `deleted` — rather than the candidate count, which
            // can overcount when the restore guard skips rows.
            per_type.push((kind_str, removed_rows.len()));
            deleted.extend(removed_rows);
        }

        Ok(GcReport {
            applied: apply,
            older_than_seconds,
            per_type,
            deleted,
        })
    }

    /// Memory ids that share a graph entity (file/tool) with any of the
    /// `seed` memories — the relationship graph's contribution to retrieval.
    /// Returns `(memory_type, id)` pairs, at most `limit`, excluding seeds.
    /// Only ACTIVE memories are returned (each candidate is probed against
    /// its main table with an archived_at IS NULL check).
    pub fn neighbor_memories(
        &self,
        project_id: &str,
        seeds: &[String],
        limit: usize,
    ) -> Result<Vec<(String, String)>> {
        if seeds.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;
        let placeholders = seeds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT DISTINCT r2.from_entity AS mid \
             FROM graph_relations r1 \
             JOIN graph_relations r2 \
               ON r1.to_entity = r2.to_entity AND r2.from_entity <> r1.from_entity \
             WHERE r1.project_id = ?1 AND r2.project_id = ?1 \
               AND r1.from_entity IN ({placeholders}) \
             LIMIT {limit2}",
            limit2 = limit * 4, // over-fetch; type resolution drops non-memories
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(project_id.to_string())];
        for s in seeds {
            params_vec.push(Box::new(s.clone()));
        }
        let params_ref: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|b| b.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let ids = stmt
            .query_map(params_ref.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Resolve each id to its (active) memory type.
        let mut out = Vec::new();
        for id in ids {
            if out.len() >= limit {
                break;
            }
            let ty: Option<String> = conn.query_row(
                "SELECT CASE \
                    WHEN EXISTS (SELECT 1 FROM episodic_memories m WHERE m.id = ?1 AND m.archived_at IS NULL) THEN 'episodic' \
                    WHEN EXISTS (SELECT 1 FROM decision_memories m WHERE m.id = ?1 AND m.archived_at IS NULL) THEN 'decision' \
                    WHEN EXISTS (SELECT 1 FROM failure_memories m WHERE m.id = ?1 AND m.archived_at IS NULL) THEN 'failure' \
                    WHEN EXISTS (SELECT 1 FROM procedural_memories m WHERE m.id = ?1 AND m.archived_at IS NULL) THEN 'procedural' \
                 END",
                params![id],
                |r| r.get(0),
            ).unwrap_or(None);
            if let Some(ty) = ty {
                out.push((ty, id));
            }
        }
        Ok(out)
    }

    /// Most recent ACTIVE episodic memory for a (project, session) pair —
    /// used by session-import to make repeated hook firings idempotent
    /// (upsert semantics instead of stacking near-duplicate memories).
    pub fn find_episodic_by_session(
        &self,
        project_id: &str,
        session_id: &str,
    ) -> Result<Option<EpisodicMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, session_id, summary, content, files_touched, \
             related_commits, importance, tags, created_at, updated_at \
             FROM episodic_memories \
             WHERE project_id = ?1 AND session_id = ?2 AND archived_at IS NULL \
             ORDER BY created_at DESC LIMIT 1",
        )?;
        let row = stmt
            .query_row(params![project_id, session_id], |row| {
                Ok(EpisodicMemory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    session_id: row.get(2)?,
                    summary: row.get(3)?,
                    content: row.get(4)?,
                    files_touched: row_get_json!(row, 5, Vec<String>),
                    related_commits: row_get_json!(row, 6, Vec<String>),
                    importance: row.get(7)?,
                    tags: row_get_json!(row, 8, Vec<String>),
                    created_at: row.get(9)?,
                    updated_at: row.get(10)?,
                })
            })
            .optional()?;
        Ok(row)
    }

    /// Physically delete File/Tool entities that no relation references
    /// anymore. Memories' own entities (type `Memory`) and entities still
    /// attached to any edge are preserved. Returns the number removed
    /// (or that would be removed, when `apply=false`).
    ///
    /// Before this, deleted/archived memories left their File/Tool entities
    /// and edges behind forever — the entity table grew unboundedly and
    /// `related_files` results filled with orphan noise.
    pub fn gc_orphan_entities(&self, apply: bool) -> Result<usize> {
        let conn = self.conn()?;
        let orphan_sql = "SELECT COUNT(*) FROM entities e \
             WHERE e.entity_type IN ('File', 'Tool') \
               AND NOT EXISTS (SELECT 1 FROM graph_relations r \
                                WHERE r.from_entity = e.id OR r.to_entity = e.id)";
        if !apply {
            let n: i64 = conn.query_row(orphan_sql, [], |r| r.get(0))?;
            return Ok(n as usize);
        }
        let tx = conn.unchecked_transaction()?;
        let deleted = tx.execute(
            "DELETE FROM entities WHERE id IN ( \
                SELECT e.id FROM entities e \
                 WHERE e.entity_type IN ('File', 'Tool') \
                   AND NOT EXISTS (SELECT 1 FROM graph_relations r \
                                    WHERE r.from_entity = e.id OR r.to_entity = e.id) \
             )",
            [],
        )?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Truncate the WAL back into the main database file. Best-effort: any
    /// failure is logged and swallowed (SQLite self-checkpoints on close).
    ///
    /// Uses `execute_batch` because the pragma helper wraps the call in a
    /// savepoint and `wal_checkpoint` cannot run inside a transaction. The
    /// pooled connection is in autocommit mode outside an explicit
    /// transaction, so this runs cleanly; if a caller left a transaction open
    /// the checkpoint simply no-ops with a warning.
    pub fn wal_checkpoint_truncate(&self) -> Result<()> {
        let conn = self.conn()?;
        if let Err(e) = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);") {
            tracing::warn!("wal_checkpoint(TRUNCATE) failed (non-fatal): {e}");
        }
        Ok(())
    }

    /// Rebuild the database file, reclaiming free pages. Opens an isolated
    /// (non-pooled) connection because `VACUUM` requires no other active
    /// connection in WAL mode. Errors out for in-memory databases; callers
    /// should run this while the MCP server is stopped.
    pub fn vacuum(&self) -> Result<()> {
        let db_path = self
            .db_path
            .as_ref()
            .context("VACUUM requires a file-backed database (in-memory DBs cannot be vacuumed)")?;
        let conn = rusqlite::Connection::open(db_path).with_context(|| {
            format!(
                "failed to open isolated connection for VACUUM at {}",
                db_path.display()
            )
        })?;
        conn.execute_batch("VACUUM;")
            .context("VACUUM failed (is another process using the database?)")?;
        Ok(())
    }

    // ─── Entity / Graph CRUD ───────────────────────────────────────

    /// Map a row to an Entity. Centralizes the repeated row mapping logic.
    fn row_to_entity(row: &rusqlite::Row<'_>) -> Result<Entity> {
        let et: String = row.get(2)?;
        Ok(Entity {
            id: row.get(0)?,
            project_id: row.get(1)?,
            entity_type: et
                .parse()
                .map_err(|e: String| anyhow::anyhow!("invalid entity_type: {e}"))?,
            name: row.get(3)?,
            metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    }

    /// Map a row to a GraphRelation. Centralizes the repeated row mapping logic.
    fn row_to_relation(row: &rusqlite::Row<'_>) -> Result<GraphRelation> {
        let rt: String = row.get(4)?;
        Ok(GraphRelation {
            id: row.get(0)?,
            project_id: row.get(1)?,
            from_entity: row.get(2)?,
            to_entity: row.get(3)?,
            relation_type: rt
                .parse()
                .map_err(|e: String| anyhow::anyhow!("invalid relation_type: {e}"))?,
            weight: row.get(5)?,
            created_at: row.get(6)?,
        })
    }

    pub fn create_entity(&self, entity: &Entity) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO entities (id, project_id, entity_type, name, metadata, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(project_id, entity_type, name) DO UPDATE SET
                metadata=excluded.metadata, updated_at=excluded.updated_at",
            params![
                entity.id,
                entity.project_id,
                entity.entity_type.as_str(),
                entity.name,
                serde_json::to_string(&entity.metadata)?,
                entity.created_at,
                entity.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, entity_type, name, metadata, created_at, updated_at FROM entities WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(Self::row_to_entity(row)?)),
            None => Ok(None),
        }
    }

    pub fn remove_entity(&self, id: &str) -> Result<bool> {
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction()?;
        // Delete related relations first to satisfy foreign key constraints
        tx.execute(
            "DELETE FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
            params![id],
        )?;
        let affected = tx.execute("DELETE FROM entities WHERE id = ?1", params![id])?;
        tx.commit()?;
        Ok(affected > 0)
    }

    pub fn create_relation(&self, rel: &GraphRelation) -> Result<()> {
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO graph_relations (project_id, from_entity, to_entity, relation_type, weight, created_at)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![
                rel.project_id,
                rel.from_entity,
                rel.to_entity,
                rel.relation_type.as_str(),
                rel.weight,
                rel.created_at,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_relations_for_entity(&self, entity_id: &str) -> Result<Vec<GraphRelation>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, from_entity, to_entity, relation_type, weight, created_at
             FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
        )?;
        let mut rows = stmt.query(params![entity_id])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(Self::row_to_relation(row)?);
        }
        Ok(results)
    }

    /// Resolve a file entity by name and return its graph neighborhood
    /// directly from the indexed `graph_relations` table — *without* loading
    /// the whole project graph into memory. Powers the MCP `related_files`
    /// tool, replacing the earlier full-graph `load_from_repo` approach.
    ///
    /// Returns `(entity_id, edges)`; `entity_id` is `None` when no `File`
    /// entity with that name exists in the project.
    pub fn related_files_for(
        &self,
        file_name: &str,
        project_id: &str,
    ) -> Result<(Option<String>, Vec<RelatedFileEdge>)> {
        let conn = self.conn()?;

        // Resolve the File entity id, scoped to entity_type='File' (entities may
        // share a name across types under the UNIQUE(project_id,entity_type,name)
        // constraint; the tool's semantics are file-scoped).
        let entity_id: Option<String> = conn
            .query_row(
                "SELECT id FROM entities \
                 WHERE name = ?1 AND project_id = ?2 AND entity_type = 'File' LIMIT 1",
                params![file_name, project_id],
                |row| row.get(0),
            )
            .optional()?;

        let edges = match entity_id {
            None => Vec::new(),
            Some(ref eid) => {
                // One JOIN pulls every neighbor: the other endpoint's name, the
                // relation type, and the direction relative to this file.
                let mut stmt = conn.prepare(
                    "SELECT o.name, r.relation_type,
                            CASE WHEN r.from_entity = e.id THEN 'outgoing'
                                 ELSE 'incoming' END AS direction
                     FROM entities e
                     JOIN graph_relations r
                       ON r.from_entity = e.id OR r.to_entity = e.id
                     JOIN entities o
                       ON o.id = CASE WHEN r.from_entity = e.id
                                      THEN r.to_entity ELSE r.from_entity END
                     WHERE e.id = ?1",
                )?;
                let rows = stmt.query_map(params![eid.as_str()], |row| {
                    Ok(RelatedFileEdge {
                        other_name: row.get(0)?,
                        relation_type: row.get(1)?,
                        direction: row.get(2)?,
                    })
                })?;
                rows.collect::<Result<Vec<_>, _>>()?
            }
        };

        Ok((entity_id, edges))
    }

    pub fn remove_relation(&self, id: i64) -> Result<bool> {
        let conn = self.conn()?;
        let affected = conn.execute("DELETE FROM graph_relations WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    /// Load all entities for a given project (plus cross-project with project_id IS NULL).
    pub fn load_entities_for_project(&self, project_id: &str) -> Result<Vec<Entity>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, entity_type, name, metadata, created_at, updated_at
             FROM entities WHERE project_id = ?1 OR project_id IS NULL",
        )?;
        let mut rows = stmt.query(params![project_id])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(Self::row_to_entity(row)?);
        }
        Ok(results)
    }

    /// Load all relations for a given project (plus cross-project with project_id IS NULL).
    pub fn load_relations_for_project(&self, project_id: &str) -> Result<Vec<GraphRelation>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, from_entity, to_entity, relation_type, weight, created_at
             FROM graph_relations WHERE project_id = ?1 OR project_id IS NULL",
        )?;
        let mut rows = stmt.query(params![project_id])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            results.push(Self::row_to_relation(row)?);
        }
        Ok(results)
    }

    /// Run FTS5 integrity check on all virtual tables. Returns Ok(()) if all pass.
    pub fn fts_integrity_check(&self) -> Result<()> {
        let conn = self.conn()?;
        for table in &[
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            let query = format!("INSERT INTO {table}({table}) VALUES('integrity-check')");
            conn.execute_batch(&query)?;
        }
        Ok(())
    }

    /// Borrow a pooled connection for advanced queries (CLI/MCP/consolidation).
    ///
    /// The returned guard derefs to `&rusqlite::Connection`, so `.prepare()` /
    /// `.query_row()` work unchanged at call sites after `?`.
    pub fn connection(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.conn()
    }

    // ─── Semantic embeddings (vector store) ──────────────────────

    /// Serialize an f32 vector to a little-endian byte BLOB.
    fn vec_to_blob(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    /// Deserialize a little-endian byte BLOB back to an f32 vector. A blob
    /// whose length is not a multiple of 4 (corruption / model change) is
    /// logged and its trailing bytes dropped — silently keeping them would
    /// yield a wrong-dimension vector whose cosine is always 0.
    fn blob_to_vec(b: &[u8]) -> Vec<f32> {
        if b.len() % 4 != 0 {
            tracing::warn!(
                "embedding blob has {} bytes (not f32-aligned); truncating",
                b.len()
            );
        }
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Insert or replace the embedding for a memory (keyed by memory_id+type).
    pub fn upsert_embedding(
        &self,
        memory_id: &str,
        memory_type: &str,
        project_id: &str,
        vector: &[f32],
        model_id: &str,
        dim: usize,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, memory_type, project_id, vector, model_id, dim, created_at)
             VALUES (?1,?2,?3,?4,?5,?6, strftime('%s','now'))
             ON CONFLICT(memory_id, memory_type) DO UPDATE SET
               vector=excluded.vector, model_id=excluded.model_id, dim=excluded.dim, project_id=excluded.project_id",
            params![memory_id, memory_type, project_id, Self::vec_to_blob(vector), model_id, dim as i64],
        )?;
        Ok(())
    }

    /// Delete the embedding(s) for a memory id (all types).
    pub fn delete_embedding(&self, memory_id: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![memory_id],
        )?;
        Ok(())
    }

    /// Load (memory_id, vector) for ACTIVE (archived_at IS NULL) memories of
    /// `project_id` matching `model_id`. Joins each memory table so archived
    /// rows and stale-model vectors are excluded.
    pub fn load_active_embeddings(
        &self,
        project_id: &str,
        model_id: &str,
    ) -> Result<Vec<(String, String, Vec<f32>)>> {
        let conn = self.conn()?;
        let mut out = Vec::new();
        for (ty, table) in [
            ("episodic", "episodic_memories"),
            ("decision", "decision_memories"),
            ("failure", "failure_memories"),
            ("procedural", "procedural_memories"),
        ] {
            let sql = format!(
                "SELECT e.memory_id, e.vector FROM memory_embeddings e
                 JOIN {table} m ON e.memory_id = m.id
                 WHERE e.project_id = ?1 AND e.model_id = ?2 AND e.memory_type = ?3
                   AND m.archived_at IS NULL"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![project_id, model_id, ty], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
            for r in rows {
                let (id, blob) = r?;
                out.push((id, ty.to_string(), Self::blob_to_vec(&blob)));
            }
        }
        Ok(out)
    }

    /// Memory ids that already have an embedding for `model_id` (optionally
    /// scoped to `project`). Lightweight (ids only) — used by reindex to skip
    /// already-embedded memories. Not joined to memory tables, so it may include
    /// ids of archived memories; reindex intersects this with the active set
    /// from `list_active_*`, so archived rows are never re-embedded.
    pub fn embedded_ids(&self, project: Option<&str>, model_id: &str) -> Result<HashSet<String>> {
        let conn = self.conn()?;
        let mut set = HashSet::new();
        match project {
            Some(p) => {
                let mut stmt = conn.prepare(
                    "SELECT memory_id FROM memory_embeddings WHERE model_id = ?1 AND project_id = ?2",
                )?;
                let rows = stmt.query_map(params![model_id, p], |r| r.get::<_, String>(0))?;
                for r in rows {
                    set.insert(r?);
                }
            }
            None => {
                let mut stmt =
                    conn.prepare("SELECT memory_id FROM memory_embeddings WHERE model_id = ?1")?;
                let rows = stmt.query_map(params![model_id], |r| r.get::<_, String>(0))?;
                for r in rows {
                    set.insert(r?);
                }
            }
        }
        Ok(set)
    }
}

/// Check if a character is a CJK (Chinese/Japanese/Korean) ideograph.
fn is_cjk_character(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'     // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}'   // CJK Unified Ideographs Extension A
        | '\u{F900}'..='\u{FAFF}'   // CJK Compatibility Ideographs
        | '\u{2F800}'..='\u{2FA1F}' // CJK Compatibility Ideographs Supplement
        | '\u{3000}'..='\u{303F}'   // CJK Symbols and Punctuation
        | '\u{3040}'..='\u{309F}'   // Hiragana
        | '\u{30A0}'..='\u{30FF}'   // Katakana
        | '\u{AC00}'..='\u{D7AF}'   // Hangul Syllables
    )
}

/// The FTS shadow table for a memory kind (enum-derived, never user input).
fn kind_fts_table(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Episodic => "episodic_memories_fts",
        MemoryKind::Decision => "decision_memories_fts",
        MemoryKind::Failure => "failure_memories_fts",
        MemoryKind::Procedural => "procedural_memories_fts",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_repo() -> MemoryRepository {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo
    }

    fn now() -> i64 {
        chrono::Utc::now().timestamp()
    }

    // ─── Schema Versioning Tests ──────────────────────────────────

    #[test]
    fn new_db_is_stamped_to_current_version() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn legacy_db_at_v0_with_tables_is_stamped_without_destructive_migrations() {
        // Simulate a legacy DB: schema built, then user_version forced back to 0
        // (as if created by the older probe-based code that never set the pragma).
        let repo = setup_repo();
        repo.set_user_version(0).unwrap();
        assert_eq!(repo.user_version().unwrap(), 0);

        // Seed a row + its FTS index entry; if run_migrations re-ran the legacy
        // FTS rebuild it would DROP+recreate the FTS tables and lose searchability.
        let mem = EpisodicMemory {
            id: "legacy-1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "legacy summary".into(),
            content: "legacy content".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.0,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();

        // v0 DB with tables present → stamped to CURRENT, no migration runs.
        repo.run_migrations().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);

        // FTS index intact → the seeded row is still searchable.
        let hits = repo.search_episodic("legacy", "p", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "legacy-1");
    }

    #[test]
    fn run_migrations_is_idempotent() {
        let repo = setup_repo();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
        // Re-running on an already-current DB (and re-initializing) is a no-op.
        repo.run_migrations().unwrap();
        repo.initialize_schema().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
    }

    // ─── Episodic Memory Tests ─────────────────────────────────────

    #[test]
    fn test_episodic_crud() {
        let repo = setup_repo();
        let mem = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            session_id: "session-1".into(),
            summary: "Fixed OAuth refresh loop".into(),
            content: "The OAuth refresh token was looping due to stale cache".into(),
            files_touched: vec!["auth.ts".into(), "token.rs".into()],
            related_commits: vec!["abc123".into()],
            importance: 0.8,
            tags: vec!["auth".into(), "oauth".into()],
            created_at: now(),
            updated_at: now(),
        };

        // Create
        repo.create_episodic(&mem).unwrap();

        // Read
        let retrieved = repo.get_episodic(&mem.id, "test-project").unwrap().unwrap();
        assert_eq!(retrieved.summary, "Fixed OAuth refresh loop");
        assert_eq!(retrieved.files_touched, vec!["auth.ts", "token.rs"]);
        assert_eq!(retrieved.importance, 0.8);

        // Update
        let mut updated = retrieved.clone();
        updated.summary = "Fixed OAuth refresh loop v2".into();
        updated.importance = 0.9;
        repo.update_episodic(&updated).unwrap();

        let after_update = repo.get_episodic(&mem.id, "test-project").unwrap().unwrap();
        assert_eq!(after_update.summary, "Fixed OAuth refresh loop v2");
        assert_eq!(after_update.importance, 0.9);

        // Delete
        assert!(repo.delete_episodic(&mem.id, "test-project").unwrap());
        assert!(!repo.delete_episodic(&mem.id, "wrong-project").unwrap());
        assert!(repo
            .get_episodic(&mem.id, "test-project")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_decision_crud() {
        let repo = setup_repo();
        let mem = DecisionMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            title: "Use Redis for session caching".into(),
            context: "Auth service needs sub-ms latency".into(),
            rationale: "Redis provides sub-millisecond reads".into(),
            tradeoffs: "Added infrastructure complexity".into(),
            related_files: vec!["auth.ts".into()],
            tags: vec!["architecture".into()],
            created_at: now(),
            updated_at: now(),
        };

        repo.create_decision(&mem).unwrap();
        let retrieved = repo.get_decision(&mem.id, "test-project").unwrap().unwrap();
        assert_eq!(retrieved.title, "Use Redis for session caching");

        let mut updated = retrieved;
        updated.title = "Use Redis for all caching".into();
        repo.update_decision(&updated).unwrap();
        assert_eq!(
            repo.get_decision(&mem.id, "test-project")
                .unwrap()
                .unwrap()
                .title,
            "Use Redis for all caching"
        );

        assert!(repo.delete_decision(&mem.id, "test-project").unwrap());
        assert!(repo
            .get_decision(&mem.id, "test-project")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_failure_crud() {
        let repo = setup_repo();
        let mem = FailureMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            incident: "Auth token expiry mismatch".into(),
            root_cause: "Clock skew between services".into(),
            fix: "Added clock tolerance window".into(),
            prevention: "Monitor clock sync across services".into(),
            severity: 3,
            tags: vec!["auth".into()],
            created_at: now(),
            updated_at: now(),
        };

        repo.create_failure(&mem).unwrap();
        let retrieved = repo.get_failure(&mem.id, "test-project").unwrap().unwrap();
        assert_eq!(retrieved.incident, "Auth token expiry mismatch");
        assert_eq!(retrieved.severity, 3);

        let mut updated = retrieved;
        updated.severity = 5;
        repo.update_failure(&updated).unwrap();
        assert_eq!(
            repo.get_failure(&mem.id, "test-project")
                .unwrap()
                .unwrap()
                .severity,
            5
        );

        assert!(repo.delete_failure(&mem.id, "test-project").unwrap());
        assert!(repo.get_failure(&mem.id, "test-project").unwrap().is_none());
    }

    #[test]
    fn test_procedural_crud() {
        let repo = setup_repo();
        let mem = ProceduralMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            workflow_name: "deployment".into(),
            steps: vec![
                "run tests".into(),
                "build docker".into(),
                "push to registry".into(),
            ],
            related_tools: vec!["docker".into(), "kubernetes".into()],
            tags: vec!["deploy".into()],
            created_at: now(),
            updated_at: now(),
        };

        repo.create_procedural(&mem).unwrap();
        let retrieved = repo
            .get_procedural(&mem.id, "test-project")
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.workflow_name, "deployment");
        assert_eq!(retrieved.steps.len(), 3);

        let mut updated = retrieved;
        updated.steps.push("verify deployment".into());
        repo.update_procedural(&updated).unwrap();
        assert_eq!(
            repo.get_procedural(&mem.id, "test-project")
                .unwrap()
                .unwrap()
                .steps
                .len(),
            4
        );

        assert!(repo.delete_procedural(&mem.id, "test-project").unwrap());
        assert!(repo
            .get_procedural(&mem.id, "test-project")
            .unwrap()
            .is_none());
    }

    // ─── FTS5 Consistency Tests ────────────────────────────────────

    #[test]
    fn test_fts5_search_episodic() {
        let repo = setup_repo();
        let mem = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            session_id: "session-1".into(),
            summary: "Fixed OAuth refresh loop".into(),
            content: "The refresh token was looping due to stale cache in Redis".into(),
            files_touched: vec!["auth.ts".into()],
            related_commits: vec![],
            importance: 0.8,
            tags: vec!["auth".into()],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();

        // Search should find it
        let results = repo
            .search_episodic("OAuth refresh", "test-project", 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, mem.id);

        // After delete, search should not find it
        repo.delete_episodic(&mem.id, "test-project").unwrap();
        let results = repo
            .search_episodic("OAuth refresh", "test-project", 10)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts5_search_after_update() {
        let repo = setup_repo();
        let mem = DecisionMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            title: "Use Postgres for storage".into(),
            context: "Need relational data".into(),
            rationale: "ACID compliance required".into(),
            tradeoffs: "Higher latency than NoSQL".into(),
            related_files: vec![],
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_decision(&mem).unwrap();

        // Search should find it
        let results = repo
            .search_decisions("Postgres", "test-project", 10)
            .unwrap();
        assert_eq!(results.len(), 1);

        // Update title
        let mut updated = mem.clone();
        updated.title = "Use MySQL for storage".into();
        repo.update_decision(&updated).unwrap();

        // Old term should not match
        let old_results = repo
            .search_decisions("Postgres", "test-project", 10)
            .unwrap();
        assert!(old_results.is_empty());

        // New term should match
        let new_results = repo.search_decisions("MySQL", "test-project", 10).unwrap();
        assert_eq!(new_results.len(), 1);
    }

    #[test]
    fn test_fts5_project_isolation() {
        let repo = setup_repo();

        // Create same content in different projects
        for pid in &["project-a", "project-b"] {
            let mem = FailureMemory {
                id: uuid::Uuid::new_v4().to_string(),
                project_id: (*pid).into(),
                incident: "Database connection timeout".into(),
                root_cause: "Connection pool exhausted".into(),
                fix: "Increased pool size".into(),
                prevention: "Monitor pool usage".into(),
                severity: 3,
                tags: vec![],
                created_at: now(),
                updated_at: now(),
            };
            repo.create_failure(&mem).unwrap();
        }

        // Search in project-a should only return 1 result
        let results_a = repo.search_failures("Database", "project-a", 10).unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0].memory.project_id, "project-a");

        let results_b = repo.search_failures("Database", "project-b", 10).unwrap();
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].memory.project_id, "project-b");
    }

    #[test]
    fn test_fts5_integrity_check() {
        let repo = setup_repo();
        // Should pass on empty db
        repo.fts_integrity_check().unwrap();

        // Create some data
        let mem = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test".into(),
            session_id: "s1".into(),
            summary: "test".into(),
            content: "test content".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();
        repo.fts_integrity_check().unwrap();
    }

    // ─── Entity / Graph Tests ──────────────────────────────────────

    #[test]
    fn related_files_for_returns_file_neighborhood() {
        let repo = setup_repo();
        // create_episodic links a Memory entity to a File entity per
        // files_touched via a "Touches" relation (ensure_linked_entities).
        let mem = EpisodicMemory {
            id: "e1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "summary".into(),
            content: "content".into(),
            files_touched: vec!["auth.ts".into()],
            related_commits: vec![],
            importance: 0.0,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();

        let (entity_id, edges) = repo.related_files_for("auth.ts", "p").unwrap();
        assert!(entity_id.is_some(), "File entity for auth.ts should exist");
        let touches: Vec<_> = edges
            .iter()
            .filter(|e| e.relation_type == "Touches")
            .collect();
        assert_eq!(touches.len(), 1, "expected one Touches edge");
        // The other endpoint is the memory entity, named by its memory id.
        assert_eq!(touches[0].other_name, "e1");
        assert_eq!(touches[0].direction, "incoming");
    }

    #[test]
    fn related_files_for_unknown_file_is_empty() {
        let repo = setup_repo();
        let (entity_id, edges) = repo.related_files_for("nope.ts", "p").unwrap();
        assert!(entity_id.is_none());
        assert!(edges.is_empty());
    }

    #[test]
    fn test_entity_and_graph_crud() {
        let repo = setup_repo();

        let entity1 = Entity {
            id: "ent-1".into(),
            project_id: "test-project".into(),
            entity_type: EntityType::File,
            name: "auth.ts".into(),
            metadata: serde_json::json!({"lines": 200}),
            created_at: now(),
            updated_at: now(),
        };
        let entity2 = Entity {
            id: "ent-2".into(),
            project_id: "test-project".into(),
            entity_type: EntityType::File,
            name: "redis.ts".into(),
            metadata: serde_json::json!({}),
            created_at: now(),
            updated_at: now(),
        };

        repo.create_entity(&entity1).unwrap();
        repo.create_entity(&entity2).unwrap();

        let retrieved = repo.get_entity("ent-1").unwrap().unwrap();
        assert_eq!(retrieved.name, "auth.ts");
        assert_eq!(retrieved.entity_type, EntityType::File);

        let rel = GraphRelation {
            id: 0, // auto-increment
            project_id: Some("test-project".into()),
            from_entity: "ent-1".into(),
            to_entity: "ent-2".into(),
            relation_type: RelationType::DependsOn,
            weight: 1.0,
            created_at: now(),
        };
        repo.create_relation(&rel).unwrap();

        let relations = repo.get_relations_for_entity("ent-1").unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, RelationType::DependsOn);

        assert!(repo.remove_entity("ent-1").unwrap());
        assert!(repo.get_entity("ent-1").unwrap().is_none());
    }

    #[test]
    fn test_load_entities_for_project() {
        let repo = setup_repo();

        // Create entities in different projects
        for (id, pid) in &[("e1", "proj-a"), ("e2", "proj-a"), ("e3", "proj-b")] {
            let entity = Entity {
                id: (*id).into(),
                project_id: (*pid).into(),
                entity_type: EntityType::Service,
                name: format!("service-{id}"),
                metadata: serde_json::json!({}),
                created_at: now(),
                updated_at: now(),
            };
            repo.create_entity(&entity).unwrap();
        }

        // Create a cross-project relation (project_id IS NULL)
        // Relations can be cross-project, entities cannot
        let rel = GraphRelation {
            id: 0,
            project_id: None, // cross-project
            from_entity: "e1".into(),
            to_entity: "e3".into(),
            relation_type: RelationType::RelatedTo,
            weight: 1.0,
            created_at: now(),
        };
        repo.create_relation(&rel).unwrap();

        // Loading proj-a should get e1, e2 (no cross-project entities)
        let entities = repo.load_entities_for_project("proj-a").unwrap();
        let ids: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e1"));
        assert!(ids.contains(&"e2"));
        assert!(!ids.contains(&"e3")); // different project

        // Cross-project relations should load for any project
        let relations = repo.load_relations_for_project("proj-a").unwrap();
        assert_eq!(relations.len(), 1); // the cross-project relation

        let relations_b = repo.load_relations_for_project("proj-b").unwrap();
        assert_eq!(relations_b.len(), 1); // same cross-project relation
    }

    // ─── Archived At Column Tests ──────────────────────────────────

    #[test]
    fn test_archived_at_column_exists_after_init() {
        let repo = setup_repo();
        // 四张主表都应有 archived_at 列；SELECT 不报错即通过。
        for t in [
            "episodic_memories",
            "decision_memories",
            "failure_memories",
            "procedural_memories",
        ] {
            let sql = format!("SELECT archived_at FROM {t} LIMIT 0");
            assert!(
                repo.connection().unwrap().prepare(&sql).is_ok(),
                "missing archived_at on {t}"
            );
        }
    }

    #[test]
    fn test_migrate_add_archived_at_is_idempotent() {
        let repo = setup_repo();
        // 已含列时再次迁移应为 no-op，不报错。
        repo.migrate_add_archived_at().unwrap();
        repo.migrate_add_archived_at().unwrap();
    }

    // ─── MemoryKind / archive / restore Tests ─────────────────────

    #[test]
    fn test_archive_and_restore_episodic() {
        let repo = setup_repo();
        let mem = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p1".into(),
            session_id: "s".into(),
            summary: "to archive".into(),
            content: "body".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();

        // 归档命中。
        assert!(repo
            .archive(MemoryKind::Episodic, &mem.id, "p1", now())
            .unwrap());
        // 重复归档不命中。
        assert!(!repo
            .archive(MemoryKind::Episodic, &mem.id, "p1", now())
            .unwrap());
        // 跨 project 不能恢复。
        assert!(!repo
            .restore(MemoryKind::Episodic, &mem.id, "other")
            .unwrap());
        // 正确恢复命中。
        assert!(repo.restore(MemoryKind::Episodic, &mem.id, "p1").unwrap());
        // 已活跃再恢复不命中。
        assert!(!repo.restore(MemoryKind::Episodic, &mem.id, "p1").unwrap());
    }

    #[test]
    fn test_memory_kind_from_type_str() {
        assert_eq!(
            MemoryKind::from_type_str("failure").unwrap(),
            MemoryKind::Failure
        );
        assert!(MemoryKind::from_type_str("bogus").is_err());
    }

    // ─── Archived Exclusion Tests ──────────────────────────────────

    #[test]
    fn test_archived_excluded_from_search_and_lists() {
        let repo = setup_repo();
        let n = now();

        // 一条 episodic + 一条 failure + 一条 decision，便于覆盖 search/list 两条路径。
        let ep = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "needle alpha".into(),
            content: "body".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: n,
            updated_at: n,
        };
        repo.create_episodic(&ep).unwrap();

        let fa = FailureMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            incident: "needle beta".into(),
            root_cause: "rc".into(),
            fix: "fx".into(),
            prevention: "pv".into(),
            severity: 3,
            tags: vec![],
            created_at: n,
            updated_at: n,
        };
        repo.create_failure(&fa).unwrap();

        // 归档前：search 命中、list 命中。
        assert_eq!(repo.search_episodic("needle", "p", 10).unwrap().len(), 1);
        assert_eq!(repo.list_recent_failures("p", 10).unwrap().len(), 1);

        // 归档两条。
        assert!(repo.archive(MemoryKind::Episodic, &ep.id, "p", n).unwrap());
        assert!(repo.archive(MemoryKind::Failure, &fa.id, "p", n).unwrap());

        // 归档后：search 不命中、list 不命中。
        assert_eq!(repo.search_episodic("needle", "p", 10).unwrap().len(), 0);
        assert_eq!(repo.list_recent_failures("p", 10).unwrap().len(), 0);

        // get 仍可取到（按 id 显式取，不过滤）。
        assert!(repo.get_episodic(&ep.id, "p").unwrap().is_some());
    }

    // ─── Task 4: archive_batch + list_archived Tests ───────────────

    #[test]
    fn test_archive_batch_by_tag_and_before() {
        let repo = setup_repo();
        let mk = |summary: &str, tags: Vec<String>, ts: i64| EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags,
            created_at: ts,
            updated_at: ts,
        };
        let a = mk("a", vec!["bootstrap".into()], 100);
        let b = mk("b", vec!["keep".into()], 100);
        let c = mk("c", vec!["bootstrap".into()], 5000);
        for m in [&a, &b, &c] {
            repo.create_episodic(m).unwrap();
        }

        // 按标签归档：只归档带 bootstrap 的 a、c。
        let ids = repo
            .archive_batch(
                MemoryKind::Episodic,
                "p",
                &["bootstrap".to_string()],
                None,
                now(),
            )
            .unwrap();
        assert_eq!(ids.len(), 2);
        assert!(repo.get_episodic(&b.id, "p").unwrap().is_some());
        assert_eq!(repo.search_episodic("a", "p", 10).unwrap().len(), 0);

        // before 过滤：恢复后按 created_at < 1000 归档，a(100) 和 b(100) 命中，c(5000) 不在。
        repo.restore(MemoryKind::Episodic, &a.id, "p").unwrap();
        repo.restore(MemoryKind::Episodic, &c.id, "p").unwrap();
        let ids2 = repo
            .archive_batch(MemoryKind::Episodic, "p", &[], Some(1000), now())
            .unwrap();
        assert_eq!(ids2.len(), 2); // a(100) 和 b(100)；c(5000) 不在
    }

    #[test]
    fn test_list_archived() {
        let repo = setup_repo();
        let n = now();
        let mem = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "archived one".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: n,
            updated_at: n,
        };
        repo.create_episodic(&mem).unwrap();
        assert!(repo
            .list_archived(MemoryKind::Episodic, "p", 10)
            .unwrap()
            .is_empty());
        repo.archive(MemoryKind::Episodic, &mem.id, "p", n).unwrap();
        let rows = repo.list_archived(MemoryKind::Episodic, "p", 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].label, "archived one");
        assert_eq!(rows[0].memory_type, "episodic");
    }

    #[test]
    fn query_log_records_and_aggregates() {
        let repo = setup_repo();
        // Two runs of "auth" (3 then 1 hits), one run of "cache" (0 hits).
        repo.record_query(
            "p",
            "auth",
            &["a".into(), "b".into(), "c".into()],
            None,
            100,
        )
        .unwrap();
        repo.record_query("p", "auth", &["a".into()], None, 200)
            .unwrap();
        repo.record_query("p", "cache", &[], None, 150).unwrap();

        let stats = repo.query_stats("p", 0, 10).unwrap();
        assert_eq!(stats.len(), 2);
        // "auth" is the most frequent query → first.
        assert_eq!(stats[0].query, "auth");
        assert_eq!(stats[0].count, 2);
        assert!(
            (stats[0].result_count_avg - 2.0).abs() < 1e-6,
            "avg hits = (3+1)/2 = 2.0"
        );
        assert_eq!(stats[0].last_at, 200);
        assert_eq!(stats[1].query, "cache");
        assert_eq!(stats[1].count, 1);

        // Time window filters out entries before the cutoff.
        let recent = repo.query_stats("p", 180, 10).unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].query, "auth");

        // Project isolation.
        assert!(repo.query_stats("other", 0, 10).unwrap().is_empty());
    }

    // ─── Reflection Suggestion Tests ───────────────────────────────

    fn make_suggestion(id: &str, tag: &str) -> ReflectionSuggestionRow {
        ReflectionSuggestionRow {
            id: id.into(),
            project_id: "p".into(),
            pattern_tag: tag.into(),
            source_failure_ids: vec!["f1".into(), "f2".into(), "f3".into()],
            source_preventions: vec![format!("prevent {tag}")],
            occurrence_count: 3,
            suggested_workflow_name: format!("Prevent recurring {tag} failures"),
            suggested_steps: vec![format!("prevent {tag}")],
            suggested_tags: vec![tag.into(), "reflection".into(), "auto-generated".into()],
            status: "pending".into(),
            created_at: now(),
            resolved_at: None,
        }
    }

    #[test]
    fn reflection_pending_is_isolated_from_search() {
        let repo = setup_repo();
        repo.insert_reflection_suggestion(&make_suggestion("s1", "fts5"))
            .unwrap();

        // Core acceptance: a pending proposal is NOT in procedural_memories, so
        // search_procedural cannot see it (the whole point of the separate table).
        assert!(repo.search_procedural("fts5", "p", 10).unwrap().is_empty());

        // But the reflection accessors do see it.
        assert!(repo.has_pending_suggestion("p", "fts5").unwrap());
        assert!(!repo.has_pending_suggestion("p", "auth").unwrap());
        let pending = repo.list_pending_suggestions("p").unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].pattern_tag, "fts5");
        assert_eq!(pending[0].source_failure_ids.len(), 3);

        // Project isolation.
        assert!(repo.list_pending_suggestions("other").unwrap().is_empty());
        assert!(!repo.has_pending_suggestion("other", "fts5").unwrap());
    }

    #[test]
    fn reflection_confirm_promotes_into_searchable_procedural() {
        let repo = setup_repo();
        let now = now();
        repo.insert_reflection_suggestion(&make_suggestion("s1", "fts5"))
            .unwrap();

        // Confirm → draft promoted into procedural_memories.
        let proc_id = repo.confirm_suggestion("s1", "p", now).unwrap();
        assert_eq!(proc_id.as_deref(), Some("s1-proc"));

        // No longer pending; now searchable through the normal procedural path.
        assert!(repo.list_pending_suggestions("p").unwrap().is_empty());
        let hits = repo.search_procedural("fts5", "p", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id, "s1-proc");
        assert_eq!(
            hits[0].memory.workflow_name,
            "Prevent recurring fts5 failures"
        );

        // Confirming again is a no-op (proposal already resolved).
        assert!(repo.confirm_suggestion("s1", "p", now).unwrap().is_none());
        // Still exactly one procedural (idempotent — no duplicate promotion).
        assert_eq!(repo.search_procedural("fts5", "p", 10).unwrap().len(), 1);

        // Unknown id / wrong project → None, nothing created.
        assert!(repo.confirm_suggestion("nope", "p", now).unwrap().is_none());
        assert!(repo
            .confirm_suggestion("s1", "other", now)
            .unwrap()
            .is_none());
    }

    #[test]
    fn reflection_reject_drops_from_pending_without_creating_procedural() {
        let repo = setup_repo();
        let now = now();
        repo.insert_reflection_suggestion(&make_suggestion("s2", "auth"))
            .unwrap();

        assert!(repo.reject_suggestion("s2", "p", now).unwrap());
        assert!(repo.list_pending_suggestions("p").unwrap().is_empty());
        // Rejected → no procedural memory created.
        assert!(repo.search_procedural("auth", "p", 10).unwrap().is_empty());

        // Rejecting again is a no-op.
        assert!(!repo.reject_suggestion("s2", "p", now).unwrap());
    }

    #[test]
    fn reflection_table_created_idempotently_on_existing_db() {
        // An already-initialized DB re-running initialize_schema must not error
        // on the reflection_suggestions table (CREATE IF NOT EXISTS).
        let repo = setup_repo();
        repo.initialize_schema().unwrap();
        repo.insert_reflection_suggestion(&make_suggestion("s1", "fts5"))
            .unwrap();
        repo.initialize_schema().unwrap();
        assert_eq!(repo.list_pending_suggestions("p").unwrap().len(), 1);
    }

    #[test]
    fn reflection_unique_constraint_prevents_duplicate_pending() {
        let repo = setup_repo();

        // First pending proposal for (p, tag1) succeeds.
        repo.insert_reflection_suggestion(&make_suggestion("s1", "tag1"))
            .unwrap();

        // A second pending proposal for the same (p, tag1) is rejected by the
        // partial unique index idx_reflection_pending_unique.
        assert!(
            repo.insert_reflection_suggestion(&make_suggestion("s2", "tag1"))
                .is_err(),
            "duplicate pending insert must be rejected by the unique index"
        );

        // A different pattern_tag is unconstrained.
        repo.insert_reflection_suggestion(&make_suggestion("s3", "tag2"))
            .unwrap();

        // Once the first proposal leaves `pending` (confirmed), the (p, tag1)
        // slot frees up and a new pending proposal is allowed again.
        repo.confirm_suggestion("s1", "p", now()).unwrap();
        repo.insert_reflection_suggestion(&make_suggestion("s4", "tag1"))
            .unwrap();
    }

    #[test]
    fn gc_does_not_delete_restored_memory() {
        // Race regression: gc collects archived candidates, then physically
        // deletes them. If a memory is restored (archived_at = NULL) between
        // collection and deletion, gc must NOT destroy it. Simulate the race:
        // create -> archive -> (gc would collect here) -> restore -> gc(apply).
        // The guarded delete (archived_at IS NOT NULL) in the GC purge must
        // leave the now-active memory intact.
        let repo = setup_repo();
        let now = now();

        let mem = EpisodicMemory {
            id: "gc-restore-race-1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "GCRESTOREMARKERX race".into(),
            content: "content".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now,
            updated_at: now,
        };
        repo.create_episodic(&mem).unwrap();

        // Archive, then immediately restore — emulating "gc already collected
        // mem.id as a candidate" followed by a restore from another process.
        repo.archive(MemoryKind::Episodic, &mem.id, &mem.project_id, now)
            .unwrap();
        repo.restore(MemoryKind::Episodic, &mem.id, &mem.project_id)
            .unwrap();

        // Run GC over all archived rows, applying deletions.
        let _report = repo.gc_archived(0, true, now).unwrap();

        // The memory must survive and remain active.
        // The memory must survive — GC's guarded delete must not physically
        // remove it.
        let still = repo.get_episodic(&mem.id, "p").unwrap();
        assert!(still.is_some(), "restored memory must survive GC");

        // Its FTS index row must still be present (cascade did not purge it).
        let hits = repo
            .search_episodic("GCRESTOREMARKERX", &mem.project_id, 10)
            .unwrap();
        assert!(
            !hits.is_empty(),
            "FTS index must still contain the restored memory"
        );
    }

    #[test]
    fn fts_rowid_stays_aligned_with_main_table() {
        // The FTS rowid must equal the main-table rowid: deletes/updates
        // address FTS rows by rowid (O(log n)); a misaligned index would
        // silently strand or destroy rows.
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("r1", "rowid alignment marker"))
            .unwrap();
        {
            let conn = repo.connection().unwrap();
            let (main_rowid, fts_rowid): (i64, i64) = conn
                .query_row(
                    "SELECT m.rowid, f.rowid FROM episodic_memories m \
                     JOIN episodic_memories_fts f ON f.memory_id = m.id \
                     WHERE m.id = 'r1'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(main_rowid, fts_rowid);
        }

        // Update replaces the FTS row in place: still exactly one row,
        // still aligned, and the new text is searchable.
        let mut mem = episodic_fixture("r1", "updated marker REIDXPROBE");
        mem.summary = "updated marker REIDXPROBE".into();
        repo.update_episodic(&mem).unwrap();
        {
            let conn = repo.connection().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM episodic_memories_fts WHERE memory_id = 'r1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(count, 1, "update must not duplicate FTS rows");
        }
        assert!(!repo
            .search_episodic("REIDXPROBE", "p", 10)
            .unwrap()
            .is_empty());

        // Delete removes the FTS row entirely (by rowid).
        repo.delete_episodic("r1", "p").unwrap();
        {
            let conn = repo.connection().unwrap();
            let leftover: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM episodic_memories_fts WHERE memory_id = 'r1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(leftover, 0, "delete must purge the FTS row");
        }
    }

    #[test]
    fn migrate_v2_rebuilds_legacy_fts_rowid_aligned() {
        // A v1-style DB (FTS rows with auto rowids, legacy preprocessing)
        // must be rebuilt aligned + re-preprocessed by the v2 migration.
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("m1", "用Rust写mixed text"))
            .unwrap();
        {
            let conn = repo.connection().unwrap();
            // Reset to v1 and install a legacy-style index (auto rowid, no
            // CJK-latin split) — as an old binary would have written it.
            conn.execute_batch("PRAGMA user_version = 1;").unwrap();
            conn.execute("DELETE FROM episodic_memories_fts", [])
                .unwrap();
            conn.execute(
                "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags) \
                 VALUES ('m1', '用Rust写mixed text', '用Rust写mixed text', '[]', '[]')",
                [],
            )
            .unwrap();
        }
        // Legacy query style (pre-boundary-split) can't match…
        assert!(repo.search_episodic("rust", "p", 10).unwrap().is_empty());

        repo.run_migrations().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
        // …but after the v2 rebuild the latin term matches the mixed text.
        assert!(
            !repo.search_episodic("rust", "p", 10).unwrap().is_empty(),
            "v2 rebuild must apply boundary-aware preprocessing"
        );
        let conn = repo.connection().unwrap();
        let aligned: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodic_memories m \
                 JOIN episodic_memories_fts f ON f.memory_id = m.id AND f.rowid = m.rowid",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(aligned, 1, "FTS rowid must equal main-table rowid");
    }
    #[test]
    fn sanitize_fts_query_escapes_reserved_words_and_operators() {
        // sanitize_fts_query wraps each whitespace-separated token in a
        // double-quoted FTS5 phrase (doubling embedded quotes) and joins with
        // AND or OR. This must prevent FTS5 reserved words/operators in user
        // input from acting as column filters or boolean operators — the
        // historical UNIQUE-crash regression this guards against.
        use MemoryRepository as R;

        // Reserved word becomes a literal phrase, not an operator/column filter.
        assert_eq!(R::sanitize_fts_query("UNIQUE", true), "\"UNIQUE\"");

        // Multiple tokens joined with AND or OR, each quoted.
        assert_eq!(
            R::sanitize_fts_query("foo bar", false),
            "\"foo\" OR \"bar\""
        );
        assert_eq!(
            R::sanitize_fts_query("foo bar", true),
            "\"foo\" AND \"bar\""
        );

        // Boolean operators inside the query are wrapped as literal phrases,
        // NOT parsed as FTS5 AND/OR/NOT.
        assert_eq!(
            R::sanitize_fts_query("a AND b", false),
            "\"a\" OR \"AND\" OR \"b\""
        );
        assert_eq!(
            R::sanitize_fts_query("x OR y", false),
            "\"x\" OR \"OR\" OR \"y\""
        );
        assert_eq!(
            R::sanitize_fts_query("p NOT q", false),
            "\"p\" OR \"NOT\" OR \"q\""
        );

        // Embedded double-quote is escaped by doubling (FTS5 phrase escape).
        assert_eq!(R::sanitize_fts_query("a\"b", false), "\"a\"\"b\"");

        // FTS5 special chars (*, :, ^) inside quotes → literal.
        assert_eq!(R::sanitize_fts_query("foo*", false), "\"foo*\"");
        assert_eq!(R::sanitize_fts_query("a:b", false), "\"a:b\"");

        // Empty / whitespace-only query yields a valid empty phrase (no crash,
        // no malformed MATCH).
        assert_eq!(R::sanitize_fts_query("", false), "\"\"");
        assert_eq!(R::sanitize_fts_query("   ", true), "\"\"");
    }

    #[test]
    fn preprocess_cjk_splits_script_boundaries() {
        use MemoryRepository as R;
        // Pure CJK: every character becomes its own token (pre-existing).
        assert_eq!(R::preprocess_cjk("修复认证"), "修 复 认 证");
        // CJK ↔ latin boundaries must also split: without this, `用Rust写`
        // indexes as ONE opaque token that no sub-query can match.
        assert_eq!(R::preprocess_cjk("用Rust写"), "用 Rust 写");
        assert_eq!(R::preprocess_cjk("修复auth模块"), "修 复 auth 模 块");
        assert_eq!(R::preprocess_cjk("SQLite做存储"), "SQLite 做 存 储");
        // CJK ↔ digits split too.
        assert_eq!(R::preprocess_cjk("修复了3个bug"), "修 复 了 3 个 bug");
        // CJK punctuation (。etc.) is itself in the CJK ranges, so it splits
        // like an ideograph — harmless for search (punctuation carries no
        // query signal) and keeps the rule uniform.
        assert_eq!(R::preprocess_cjk("修复。完成"), "修 复 。 完 成");
        // No CJK: untouched.
        assert_eq!(R::preprocess_cjk("plain ascii"), "plain ascii");
    }

    #[test]
    fn search_finds_mixed_cjk_latin_text() {
        // End-to-end: a memory written in mixed Chinese/English must be
        // findable by either script's query terms.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mem = crate::models::EpisodicMemory {
            id: "mix1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "用Rust重写认证模块".into(),
            content: "SQLite存储 token 刷新逻辑".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["认证".into()],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mem).unwrap();

        let by_latin = repo.search_episodic("rust", "p", 10).unwrap();
        assert!(
            !by_latin.is_empty(),
            "latin term must match mixed-script text"
        );
        let by_cjk = repo.search_episodic("认证", "p", 10).unwrap();
        assert!(!by_cjk.is_empty(), "CJK term must match mixed-script text");
    }

    #[test]
    fn search_finds_cjk_tags() {
        // Tags are indexed through the same CJK preprocessing as text —
        // previously the raw JSON blob made Chinese tags unsearchable.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mem = crate::models::EpisodicMemory {
            id: "tag1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "english only summary".into(),
            content: "english only content".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["部署策略".into()],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mem).unwrap();

        let r = repo.search_episodic("部署", "p", 10).unwrap();
        assert!(!r.is_empty(), "CJK tag must be searchable");
    }

    #[test]
    fn search_and_requires_all_tokens_with_or_fallback() {
        // "rust memory" with AND hits only memories containing BOTH terms;
        // the OR fallback keeps recall when AND matches nothing.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |id: &str, summary: &str| crate::models::EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: summary.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mk("e1", "rust memory system"))
            .unwrap();
        repo.create_episodic(&mk("e2", "python memory system"))
            .unwrap();
        repo.create_episodic(&mk("e3", "kotlin coroutine flow"))
            .unwrap();

        // AND precision: "rust memory" must match ONLY the memory containing
        // both terms — not the python/kotlin ones.
        let both = repo.search_episodic("rust memory", "p", 10).unwrap();
        let ids: Vec<&str> = both.iter().map(|s| s.memory.id.as_str()).collect();
        assert_eq!(ids, vec!["e1"], "AND must require every token, got {ids:?}");

        // OR recall: no memory has BOTH "zig" and "memory" → the AND pass is
        // empty → the OR fallback returns the "memory"-only hits.
        let fallback = repo.search_episodic("zig memory", "p", 10).unwrap();
        assert!(
            !fallback.is_empty(),
            "OR fallback must preserve recall for partial matches"
        );
    }

    // ─── Transaction Rollback Test ─────────────────────────────────

    #[test]
    fn create_then_read_roundtrip() {
        let repo = setup_repo();

        // Create a valid memory first
        let mem = EpisodicMemory {
            id: "rollback-test-id".into(),
            project_id: "test".into(),
            session_id: "s1".into(),
            summary: "original summary".into(),
            content: "original content".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mem).unwrap();

        // Verify it exists
        let retrieved = repo
            .get_episodic("rollback-test-id", "test")
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.summary, "original summary");
    }
    #[test]
    fn pool_allows_concurrent_reads() {
        use std::sync::Arc;
        let tmp = tempfile::TempDir::new().unwrap();
        let repo = Arc::new(MemoryRepository::new(&tmp.path().join("pool.db")).unwrap());
        repo.initialize_schema().unwrap();

        // 8 threads concurrently borrow pooled connections to read — verify no panic/deadlock.
        let mut handles = vec![];
        for _ in 0..8 {
            let repo = Arc::clone(&repo);
            handles.push(std::thread::spawn(move || {
                let conn = repo.connection().unwrap();
                let _: i64 = conn.query_row("SELECT 1", [], |r| r.get(0)).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    fn seed_episodic(repo: &MemoryRepository, id: &str) {
        repo.create_episodic(&EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "x".into(),
            content: "y".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        })
        .unwrap();
    }

    #[test]
    fn embedding_upsert_load_delete_roundtrip() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_episodic(&repo, "m1");

        repo.upsert_embedding("m1", "episodic", "p", &[0.1, 0.2, 0.3], "minilm", 3)
            .unwrap();
        let loaded = repo.load_active_embeddings("p", "minilm").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "m1");
        assert_eq!(loaded[0].1, "episodic");
        assert_eq!(loaded[0].2.len(), 3);
        assert!((loaded[0].2[1] - 0.2).abs() < 1e-6);

        // upsert same id replaces, not duplicates
        repo.upsert_embedding("m1", "episodic", "p", &[0.4, 0.5, 0.6], "minilm", 3)
            .unwrap();
        assert_eq!(repo.load_active_embeddings("p", "minilm").unwrap().len(), 1);

        repo.delete_embedding("m1").unwrap();
        assert!(repo
            .load_active_embeddings("p", "minilm")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn delete_memory_also_removes_its_embedding() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_episodic(&repo, "m1");
        repo.upsert_embedding("m1", "episodic", "p", &[0.1, 0.2, 0.3], "minilm", 3)
            .unwrap();
        assert_eq!(repo.load_active_embeddings("p", "minilm").unwrap().len(), 1);

        // Hard-deleting the memory must purge its embedding too (regression:
        // the delete macro used to leave memory_embeddings rows orphaned).
        assert!(repo.delete_episodic("m1", "p").unwrap());
        assert!(repo
            .load_active_embeddings("p", "minilm")
            .unwrap()
            .is_empty());
    }

    // ─── Garbage Collection Tests ─────────────────────────────────

    #[test]
    fn gc_dry_run_keeps_archived_rows() {
        let repo = setup_repo();
        seed_episodic(&repo, "g1");
        // Archived long ago (archived_at = 1000).
        repo.archive(MemoryKind::Episodic, "g1", "p", 1000).unwrap();

        let report = repo.gc_archived(1, false, now()).unwrap();
        assert!(!report.applied);
        assert_eq!(report.deleted.len(), 1);
        // Dry run: row still present.
        assert!(repo.get_episodic("g1", "p").unwrap().is_some());
    }

    #[test]
    fn gc_apply_deletes_expired_archived_and_embedding() {
        let repo = setup_repo();
        seed_episodic(&repo, "g1");
        repo.upsert_embedding("g1", "episodic", "p", &[0.1], "minilm", 1)
            .unwrap();
        repo.archive(MemoryKind::Episodic, "g1", "p", 1000).unwrap();

        let report = repo.gc_archived(1, true, now()).unwrap();
        assert!(report.applied);
        assert_eq!(report.deleted.len(), 1);
        assert_eq!(report.per_type[0].0, "episodic");
        assert_eq!(report.per_type[0].1, 1);
        // Physically gone, and its embedding purged too.
        assert!(repo.get_episodic("g1", "p").unwrap().is_none());
        assert!(repo
            .load_active_embeddings("p", "minilm")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn gc_skips_recently_archived() {
        let repo = setup_repo();
        seed_episodic(&repo, "g1");
        let recent = now();
        repo.archive(MemoryKind::Episodic, "g1", "p", recent)
            .unwrap();

        // older_than=3600s: the just-archived row is not yet expired.
        let report = repo.gc_archived(3600, true, now()).unwrap();
        assert_eq!(report.deleted.len(), 0);
        assert!(repo.get_episodic("g1", "p").unwrap().is_some());
    }

    #[test]
    fn gc_all_mode_deletes_every_archived() {
        let repo = setup_repo();
        seed_episodic(&repo, "g1");
        // Recently archived, but --all (older_than=0) ignores age.
        repo.archive(MemoryKind::Episodic, "g1", "p", now())
            .unwrap();

        let report = repo.gc_archived(0, true, now()).unwrap();
        assert_eq!(report.deleted.len(), 1);
        assert!(repo.get_episodic("g1", "p").unwrap().is_none());
    }

    #[test]
    fn wal_checkpoint_runs_on_file_db() {
        let tmp = tempfile::TempDir::new().unwrap();
        let repo = MemoryRepository::new(&tmp.path().join("gc.db")).unwrap();
        repo.initialize_schema().unwrap();
        seed_episodic(&repo, "g1");
        // Should not error (best-effort, but expected to succeed on a quiet DB).
        repo.wal_checkpoint_truncate().unwrap();
    }

    #[test]
    fn vacuum_reclaims_file_db() {
        let tmp = tempfile::TempDir::new().unwrap();
        let repo = MemoryRepository::new(&tmp.path().join("gc.db")).unwrap();
        repo.initialize_schema().unwrap();
        seed_episodic(&repo, "g1");
        // VACUUM via an isolated connection on a quiet DB succeeds.
        repo.vacuum().unwrap();
    }

    #[test]
    fn vacuum_errors_on_in_memory_db() {
        let repo = setup_repo();
        assert!(repo.vacuum().is_err());
    }

    #[test]
    fn archived_memory_embedding_is_excluded() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        seed_episodic(&repo, "m1");
        repo.upsert_embedding("m1", "episodic", "p", &[0.1, 0.2, 0.3], "minilm", 3)
            .unwrap();
        assert_eq!(repo.load_active_embeddings("p", "minilm").unwrap().len(), 1);

        repo.archive(MemoryKind::Episodic, "m1", "p", 999).unwrap();
        assert!(
            repo.load_active_embeddings("p", "minilm")
                .unwrap()
                .is_empty(),
            "archived memory's embedding must be excluded from active set"
        );

        // different model_id is not matched
        repo.upsert_embedding("m1", "episodic", "p", &[0.1, 0.2, 0.3], "minilm", 3)
            .unwrap();
        assert!(repo
            .load_active_embeddings("p", "other-model")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn list_active_episodic_excludes_archived_and_filters_project() {
        let repo = setup_repo();
        let mk = |id: &str, proj: &str| EpisodicMemory {
            id: id.into(),
            project_id: proj.into(),
            session_id: "s".into(),
            summary: "sum".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now(),
            updated_at: now(),
        };
        repo.create_episodic(&mk("a", "p1")).unwrap();
        repo.create_episodic(&mk("b", "p1")).unwrap();
        repo.create_episodic(&mk("c", "p2")).unwrap();
        repo.archive(MemoryKind::Episodic, "b", "p1", now())
            .unwrap();

        let all = repo.list_active_episodic(None).unwrap();
        assert_eq!(all.len(), 2, "active across projects (a,c); b archived");
        let p1 = repo.list_active_episodic(Some("p1")).unwrap();
        assert_eq!(p1.len(), 1, "only active p1 = a");
        assert_eq!(p1[0].id, "a");
    }

    #[test]
    fn embedded_ids_returns_ids_for_model_optionally_by_project() {
        let repo = setup_repo();
        repo.upsert_embedding("m1", "episodic", "p1", &[0.1, 0.2, 0.3], "minilm", 3)
            .unwrap();
        repo.upsert_embedding("m2", "decision", "p2", &[0.4, 0.5, 0.6], "minilm", 3)
            .unwrap();
        repo.upsert_embedding("m3", "episodic", "p1", &[0.7, 0.8, 0.9], "other", 3)
            .unwrap();

        let all = repo.embedded_ids(None, "minilm").unwrap();
        assert_eq!(all.len(), 2, "m1,m2 for minilm; m3 is other model");
        assert!(all.contains("m1") && all.contains("m2"));

        let p1 = repo.embedded_ids(Some("p1"), "minilm").unwrap();
        assert_eq!(p1.len(), 1, "only m1 in p1 for minilm");
        assert!(p1.contains("m1"));
    }

    fn episodic_fixture(id: &str, summary: &str) -> EpisodicMemory {
        EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: format!("{summary} body"),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 100,
            updated_at: 100,
        }
    }

    #[test]
    fn update_nonexistent_id_errors_and_leaves_no_orphan_fts() {
        let repo = setup_repo();
        let mut mem = episodic_fixture("ghost", "phantom summary");
        mem.id = "does-not-exist".into();

        let err = repo.update_episodic(&mem);
        assert!(err.is_err(), "updating a missing id must fail");

        // No orphan FTS row may survive the failed update.
        let conn = repo.connection().unwrap();
        let orphan: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodic_memories_fts WHERE memory_id = 'does-not-exist'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(orphan, 0, "failed update must not create orphan FTS rows");
    }

    #[test]
    fn rebuild_fts_repairs_orphans_and_missing_rows() {
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("e1", "alpha engine"))
            .unwrap();
        repo.create_episodic(&episodic_fixture("e2", "beta wheel"))
            .unwrap();

        {
            let conn = repo.connection().unwrap();
            // Corrupt the index both ways: an orphan row + a missing row.
            conn.execute(
                "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags)
                 VALUES ('ghost', 'orphan', 'orphan', '[]', '[]')",
                [],
            )
            .unwrap();
            conn.execute(
                "DELETE FROM episodic_memories_fts WHERE memory_id = 'e2'",
                [],
            )
            .unwrap();
        }

        let written = repo.rebuild_fts().unwrap();
        assert_eq!(written, 2, "one FTS row per main-table row");

        let orphan: i64 = {
            let conn = repo.connection().unwrap();
            conn.query_row(
                "SELECT COUNT(*) FROM episodic_memories_fts WHERE memory_id = 'ghost'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(orphan, 0, "orphan FTS rows must be purged");
        assert!(
            !repo.search_episodic("beta", "p", 10).unwrap().is_empty(),
            "restored row must be searchable again"
        );
    }

    #[test]
    fn rebuild_fts_reapplies_cjk_preprocessing() {
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("c1", "修复认证模块"))
            .unwrap();
        // Simulate a legacy raw-text index (no CJK split): unreadable by the
        // preprocessing-aware query side.
        {
            let conn = repo.connection().unwrap();
            conn.execute(
                "DELETE FROM episodic_memories_fts WHERE memory_id = 'c1'",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags)
                 VALUES ('c1', '修复认证模块', '修复认证模块 body', '[]', '[]')",
                [],
            )
            .unwrap();
        }
        assert!(
            repo.search_episodic("认证", "p", 10).unwrap().is_empty(),
            "raw CJK index should not match per-character queries"
        );

        repo.rebuild_fts().unwrap();
        assert!(
            !repo.search_episodic("认证", "p", 10).unwrap().is_empty(),
            "rebuild must re-apply CJK preprocessing"
        );
    }

    #[test]
    fn get_ingested_commits_excludes_archived_memories() {
        let repo = setup_repo();
        let mut mem = episodic_fixture("e1", "with commit");
        mem.related_commits = vec!["abc123".into()];
        repo.create_episodic(&mem).unwrap();

        assert!(repo.get_ingested_commits("p").unwrap().contains("abc123"));

        repo.archive(MemoryKind::Episodic, "e1", "p", 200).unwrap();
        assert!(
            !repo.get_ingested_commits("p").unwrap().contains("abc123"),
            "archiving a memory must release its commits for re-import"
        );
    }

    #[test]
    fn update_refreshes_entity_links_for_changed_files() {
        let repo = setup_repo();
        let mut mem = episodic_fixture("e1", "touches files");
        mem.files_touched = vec!["old.rs".into()];
        repo.create_episodic(&mem).unwrap();

        // Change the file list: old edge must go, new edge must exist.
        mem.files_touched = vec!["new.rs".into()];
        mem.updated_at = 200;
        repo.update_episodic(&mem).unwrap();

        let (_, edges) = repo.related_files_for("old.rs", "p").unwrap();
        assert!(edges.is_empty(), "stale edge to old.rs must be dropped");
        let (_, edges) = repo.related_files_for("new.rs", "p").unwrap();
        assert!(!edges.is_empty(), "edge to new.rs must be linked");
    }

    #[test]
    fn prune_query_log_removes_only_old_rows() {
        let repo = setup_repo();
        repo.record_query("p", "old", &[], None, 100).unwrap();
        repo.record_query("p", "new", &[], None, 10_000).unwrap();

        let removed = repo.prune_query_log(500, 10_000).unwrap();
        assert_eq!(removed, 1);
        let stats = repo.query_stats("p", 0, 10).unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].query, "new");
    }

    #[test]
    fn list_projects_unions_all_kinds() {
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("e1", "x")).unwrap();
        repo.create_decision(&DecisionMemory {
            id: "d1".into(),
            project_id: "other".into(),
            title: "t".into(),
            context: "c".into(),
            rationale: "r".into(),
            tradeoffs: "t".into(),
            related_files: vec![],
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        })
        .unwrap();

        assert_eq!(
            repo.list_projects().unwrap(),
            vec!["other".to_string(), "p".to_string()]
        );
    }
}
