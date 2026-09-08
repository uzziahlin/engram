//! Schema definition, migrations, and FTS tokenization for the memory store.
//!
//! Split from `repository.rs` (which had grown past 4,500 lines): this file
//! owns everything that shapes the database — DDL, the versioned migration
//! registry, FTS preprocessing/tokenization, and index-integrity checks.

use super::repository::is_cjk_character;
use super::repository::MemoryRepository;
use anyhow::Result;
use rusqlite::{params, OptionalExtension};

/// Highest schema version produced by [`MemoryRepository::initialize_schema`].
///
/// Existing databases whose `PRAGMA user_version` is below this are migrated
/// forward; new databases are created at this version directly (no registered
/// migration runs on a fresh DB — the CREATE statements already encode it).
/// Bump this whenever a new migration is appended to
/// [`MemoryRepository::migrations`].
pub(crate) const CURRENT_SCHEMA_VERSION: i32 = 6;

/// DDL for the four FTS5 shadow tables. Shared by `initialize_schema` (fresh
/// DBs) and the v2 rebuild migration so the two cannot drift.
pub(crate) const FTS_TABLES_DDL: &str = "
    -- Contentless FTS5 with delete support (SQLite >= 3.43): the inverted
    -- index is stored, the document text is NOT — the main table already
    -- holds it. Halves the database vs the old full-copy shadow tables.
    -- The trade: FTS rows return no column values, so search joins by
    -- rowid (kept in lockstep with the main table) instead of memory_id.
    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memories_fts USING fts5(
        summary, content, files_touched, tags,
        tokenize='porter unicode61',
        content='', contentless_delete=1
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS decision_memories_fts USING fts5(
        title, context, rationale, tradeoffs, tags,
        tokenize='porter unicode61',
        content='', contentless_delete=1
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS failure_memories_fts USING fts5(
        incident, root_cause, fix, prevention, tags,
        tokenize='porter unicode61',
        content='', contentless_delete=1
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS procedural_memories_fts USING fts5(
        workflow_name, steps, related_tools, tags,
        tokenize='porter unicode61',
        content='', contentless_delete=1
    );
";

/// A single registered migration: receives the repository and either succeeds
/// or returns an error. Factored into a type alias to keep `migrations()`'s
/// signature readable (clippy::type_complexity).
pub(crate) type MigrationFn = fn(&MemoryRepository) -> Result<()>;
impl MemoryRepository {
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
                importance REAL NOT NULL DEFAULT 0.5,
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
                adopted      INTEGER NOT NULL DEFAULT 0,
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

        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;

        tx.commit()?;
        drop(conn);

        // Versioned migrations: stamp/advance `PRAGMA user_version` and run any
        // pending registered migrations. See `run_migrations`.
        self.run_migrations()?;
        // Tokenizer consistency: the FTS index must have been built by the
        // same tokenizer this binary uses for queries (char-split vs jieba),
        // or matches silently degrade. Rebuild on mismatch.
        self.ensure_fts_tokenizer()?;

        Ok(())
    }

    /// Identifier of the FTS tokenization this binary produces. Recorded in
    /// the `meta` table so switching between char-split and jieba builds
    /// triggers exactly one index rebuild.
    fn fts_tokenizer_id() -> &'static str {
        if cfg!(feature = "jieba") {
            "jieba"
        } else {
            "char"
        }
    }

    /// Rebuild the FTS indexes when they were built by a different tokenizer
    /// than this binary uses. Idempotent: a matching record is a no-op.
    pub fn ensure_fts_tokenizer(&self) -> Result<()> {
        let want = Self::fts_tokenizer_id();
        let have = self.meta_get("fts_tokenizer")?;
        if have.as_deref() == Some(want) {
            return Ok(());
        }
        let rows = self.rebuild_fts()?;
        self.meta_set("fts_tokenizer", want)?;
        tracing::info!(
            "FTS rebuilt for tokenizer '{want}' ({rows} rows) — index was built by {:?}",
            have.as_deref().unwrap_or("<none>")
        );
        Ok(())
    }

    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn()?;
        let v = conn
            .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(v)
    }

    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
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
    pub(crate) fn preprocess_cjk(text: &str) -> String {
        let has_cjk = text.chars().any(is_cjk_character);
        if !has_cjk {
            return text.to_string();
        }
        #[cfg(feature = "jieba")]
        {
            Self::segment_jieba(text)
        }
        #[cfg(not(feature = "jieba"))]
        {
            Self::segment_per_char(text)
        }
    }

    /// Word-level CJK segmentation via the embedded jieba dictionary. The
    /// index and query sides both go through here, so tokens always agree
    /// (`认证` is one token, matching exactly instead of "both chars
    /// somewhere in the doc"). The dictionary load is one-time and cached.
    #[cfg(feature = "jieba")]
    fn segment_jieba(text: &str) -> String {
        static JIEBA: std::sync::OnceLock<jieba_rs::Jieba> = std::sync::OnceLock::new();
        let jieba = JIEBA.get_or_init(jieba_rs::Jieba::new);
        let words = jieba.cut(text, true);
        let mut out = String::with_capacity(text.len() + words.len());
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(w);
        }
        out
    }

    /// Default (no-dependency) CJK handling: per-character tokens plus
    /// script-boundary splitting.
    #[cfg(not(feature = "jieba"))]
    fn segment_per_char(text: &str) -> String {
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
    pub(crate) fn sanitize_fts_query(query: &str, require_all: bool) -> String {
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
    pub(crate) fn fts_json<T: serde::Serialize>(v: &T) -> Result<String> {
        Ok(Self::preprocess_cjk(&serde_json::to_string(v)?))
    }

    // ─── FTS integrity ────────────────────────────────────────────
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
}

impl MemoryRepository {
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
                "SELECT rowid, summary, content, files_touched, tags FROM episodic_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO episodic_memories_fts (summary, content, files_touched, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    Self::preprocess_cjk(&row.get::<_, String>(1)?),
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
                    row.get::<_, i64>(0)?,
                ])?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, title, context, rationale, tradeoffs, tags FROM decision_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO decision_memories_fts (title, context, rationale, tradeoffs, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    Self::preprocess_cjk(&row.get::<_, String>(1)?),
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
                "SELECT rowid, incident, root_cause, fix, prevention, tags FROM failure_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO failure_memories_fts (incident, root_cause, fix, prevention, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5,?6)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    Self::preprocess_cjk(&row.get::<_, String>(1)?),
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
                "SELECT rowid, workflow_name, steps, related_tools, tags FROM procedural_memories",
            )?;
            let mut rows = stmt.query([])?;
            let mut ins = tx.prepare(
                "INSERT INTO procedural_memories_fts (workflow_name, steps, related_tools, tags, rowid)
                 VALUES (?1,?2,?3,?4,?5)",
            )?;
            while let Some(row) = rows.next()? {
                ins.execute(params![
                    Self::preprocess_cjk(&row.get::<_, String>(1)?),
                    Self::preprocess_cjk(&row.get::<_, String>(2)?),
                    Self::preprocess_cjk(&row.get::<_, String>(3)?),
                    Self::preprocess_cjk(&row.get::<_, String>(4)?),
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
        static REG: [(i32, MigrationFn); 5] = [
            (2, MemoryRepository::migrate_v2_fts_rowid),
            (3, MemoryRepository::migrate_v3_reflection_rearm),
            (4, MemoryRepository::migrate_v4_decision_importance),
            (5, MemoryRepository::migrate_v5_query_log_adoption),
            (6, MemoryRepository::migrate_v6_fts_contentless),
        ];
        &REG
    }

    /// v6 — rebuild the FTS shadow tables as contentless tables (no text
    /// copy; join by rowid). See FTS_TABLES_DDL for the rationale.
    fn migrate_v6_fts_contentless(&self) -> Result<()> {
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
        tracing::info!("migration v6: FTS rebuilt contentless ({written} rows)");
        Ok(())
    }

    /// v5 — `query_log.adopted`: how many times a memory returned by a
    /// search was later fetched via `get_memory` (the adoption signal that
    /// feeds `query_stats` and the stats command).
    fn migrate_v5_query_log_adoption(&self) -> Result<()> {
        let conn = self.conn()?;
        let has_col = conn
            .prepare("SELECT adopted FROM query_log LIMIT 0")
            .is_ok();
        if !has_col {
            conn.execute_batch(
                "ALTER TABLE query_log ADD COLUMN adopted INTEGER NOT NULL DEFAULT 0;",
            )?;
        }
        Ok(())
    }

    /// v4 — `importance` columns for decision/procedural memories. Both
    /// types previously hard-coded 0.5 in the reranker, making the
    /// importance weight dead for them; the field is real now (tools accept
    /// it, `mark_relevance` feedback adjusts it).
    fn migrate_v4_decision_importance(&self) -> Result<()> {
        let conn = self.conn()?;
        for table in ["decision_memories", "procedural_memories"] {
            let has_col = conn
                .prepare(&format!("SELECT importance FROM {table} LIMIT 0"))
                .is_ok();
            if !has_col {
                conn.execute_batch(&format!(
                    "ALTER TABLE {table} ADD COLUMN importance REAL NOT NULL DEFAULT 0.5;"
                ))?;
            }
        }
        Ok(())
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
}
