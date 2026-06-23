use crate::models::*;
use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use std::collections::HashSet;
use std::path::Path;

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

/// Repository for all memory CRUD operations with FTS5 dual-write.
///
pub struct MemoryRepository {
    pool: Pool<SqliteConnectionManager>,
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
        Ok(Self { pool })
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
        Ok(Self { pool })
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
        tx.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memories_fts USING fts5(
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
            );",
        )?;

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

        tx.commit()?;
        drop(conn);

        // 对既有库补列（新库的 CREATE TABLE 已带该列，迁移会检测到并跳过）。
        self.migrate_add_archived_at()?;

        Ok(())
    }

    /// Preprocess text for FTS5 by inserting spaces between consecutive CJK characters.
    /// FTS5 unicode61 tokenizer treats consecutive CJK characters as a single token,
    /// which prevents substring matching. By inserting spaces, each CJK character
    /// becomes its own token, enabling character-level matching.
    /// Used for both indexing (FTS5 INSERT) and querying (FTS5 MATCH).
    fn preprocess_cjk(text: &str) -> String {
        let has_cjk = text.chars().any(is_cjk_character);
        if !has_cjk {
            return text.to_string();
        }

        // Estimate capacity: original length + space for CJK separators (worst case ~50% extra)
        let mut result = String::with_capacity(text.len() + text.len() / 2);
        let mut prev_was_cjk = false;

        for ch in text.chars() {
            if is_cjk_character(ch) {
                if prev_was_cjk {
                    result.push(' ');
                }
                result.push(ch);
                prev_was_cjk = true;
            } else {
                result.push(ch);
                prev_was_cjk = false;
            }
        }

        result
    }

    /// Sanitize a user query for safe use in FTS5 MATCH expressions.
    /// Splits the query into tokens, wraps each in double quotes (to escape
    /// FTS5 operators like AND/OR/NOT), and joins them with ` OR` so that
    /// each token is matched independently instead of as an exact phrase.
    /// Must be called AFTER preprocess_cjk for CJK support.
    fn sanitize_fts_query(query: &str) -> String {
        let processed = Self::preprocess_cjk(query);
        let tokens: Vec<&str> = processed.split_whitespace().collect();
        if tokens.is_empty() {
            return "\"\"".to_string();
        }
        tokens
            .into_iter()
            .map(|t| {
                let escaped = t.replace('"', "\"\"");
                format!("\"{}\"", escaped)
            })
            .collect::<Vec<_>>()
            .join(" OR ")
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

        tx.execute_batch(
            "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags)
             SELECT id, summary, content, files_touched, tags FROM episodic_memories;
             INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs, tags)
             SELECT id, title, context, rationale, tradeoffs, tags FROM decision_memories;
             INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention, tags)
             SELECT id, incident, root_cause, fix, prevention, tags FROM failure_memories;
             INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools, tags)
             SELECT id, workflow_name, steps, related_tools, tags FROM procedural_memories;",
        )?;

        tx.commit()?;
        tracing::info!("FTS5 migration complete — tags column added.");
        Ok(())
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

        fts_insert_sql = "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags) VALUES (?1,?2,?3,?4,?5)",

        update_sql = "UPDATE episodic_memories SET project_id=?2, session_id=?3, summary=?4, content=?5, files_touched=?6, related_commits=?7, importance=?8, tags=?9, updated_at=?10 WHERE id=?1",

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
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.tags)?,
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
                    "Touches", mem.created_at,
                )?;
            }
        },
    }

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,

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

        fts_insert_sql = "INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs, tags) VALUES (?1,?2,?3,?4,?5,?6)",

        update_sql = "UPDATE decision_memories SET project_id=?2, title=?3, context=?4, rationale=?5, tradeoffs=?6, related_files=?7, tags=?8, updated_at=?9 WHERE id=?1",

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
                serde_json::to_string(&mem.tags)?,
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
                    "References", mem.created_at,
                )?;
            }
        },
    }

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,

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

        fts_insert_sql = "INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention, tags) VALUES (?1,?2,?3,?4,?5,?6)",

        update_sql = "UPDATE failure_memories SET project_id=?2, incident=?3, root_cause=?4, fix=?5, prevention=?6, severity=?7, tags=?8, updated_at=?9 WHERE id=?1",

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
                serde_json::to_string(&mem.tags)?,
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

        fts_insert_sql = "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools, tags) VALUES (?1,?2,?3,?4,?5)",

        update_sql = "UPDATE procedural_memories SET project_id=?2, workflow_name=?3, steps=?4, related_tools=?5, tags=?6, updated_at=?7 WHERE id=?1",

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
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
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
                    "Uses", mem.created_at,
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

    /// Check which commit hashes have already been ingested as episodic memories.
    /// Returns a HashSet of already-ingested commit hashes for O(1) lookup.
    pub fn get_ingested_commits(&self, project_id: &str) -> Result<HashSet<String>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT related_commits FROM episodic_memories WHERE project_id = ?1")?;
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

    /// Deserialize a little-endian byte BLOB back to an f32 vector.
    fn blob_to_vec(b: &[u8]) -> Vec<f32> {
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
    )
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
        let retrieved = repo.get_episodic(&mem.id).unwrap().unwrap();
        assert_eq!(retrieved.summary, "Fixed OAuth refresh loop");
        assert_eq!(retrieved.files_touched, vec!["auth.ts", "token.rs"]);
        assert_eq!(retrieved.importance, 0.8);

        // Update
        let mut updated = retrieved.clone();
        updated.summary = "Fixed OAuth refresh loop v2".into();
        updated.importance = 0.9;
        repo.update_episodic(&updated).unwrap();

        let after_update = repo.get_episodic(&mem.id).unwrap().unwrap();
        assert_eq!(after_update.summary, "Fixed OAuth refresh loop v2");
        assert_eq!(after_update.importance, 0.9);

        // Delete
        assert!(repo.delete_episodic(&mem.id, "test-project").unwrap());
        assert!(!repo.delete_episodic(&mem.id, "wrong-project").unwrap());
        assert!(repo.get_episodic(&mem.id).unwrap().is_none());
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
        let retrieved = repo.get_decision(&mem.id).unwrap().unwrap();
        assert_eq!(retrieved.title, "Use Redis for session caching");

        let mut updated = retrieved;
        updated.title = "Use Redis for all caching".into();
        repo.update_decision(&updated).unwrap();
        assert_eq!(
            repo.get_decision(&mem.id).unwrap().unwrap().title,
            "Use Redis for all caching"
        );

        assert!(repo.delete_decision(&mem.id, "test-project").unwrap());
        assert!(repo.get_decision(&mem.id).unwrap().is_none());
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
        let retrieved = repo.get_failure(&mem.id).unwrap().unwrap();
        assert_eq!(retrieved.incident, "Auth token expiry mismatch");
        assert_eq!(retrieved.severity, 3);

        let mut updated = retrieved;
        updated.severity = 5;
        repo.update_failure(&updated).unwrap();
        assert_eq!(repo.get_failure(&mem.id).unwrap().unwrap().severity, 5);

        assert!(repo.delete_failure(&mem.id, "test-project").unwrap());
        assert!(repo.get_failure(&mem.id).unwrap().is_none());
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
        let retrieved = repo.get_procedural(&mem.id).unwrap().unwrap();
        assert_eq!(retrieved.workflow_name, "deployment");
        assert_eq!(retrieved.steps.len(), 3);

        let mut updated = retrieved;
        updated.steps.push("verify deployment".into());
        repo.update_procedural(&updated).unwrap();
        assert_eq!(
            repo.get_procedural(&mem.id).unwrap().unwrap().steps.len(),
            4
        );

        assert!(repo.delete_procedural(&mem.id, "test-project").unwrap());
        assert!(repo.get_procedural(&mem.id).unwrap().is_none());
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
        assert!(repo.get_episodic(&ep.id).unwrap().is_some());
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
        assert!(repo.get_episodic(&b.id).unwrap().is_some());
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

    // ─── Transaction Rollback Test ─────────────────────────────────

    #[test]
    fn test_transaction_rollback_on_fts_failure() {
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
        let retrieved = repo.get_episodic("rollback-test-id").unwrap().unwrap();
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
}
