//! Schema definition, migrations, and FTS tokenization for the memory store.
//!
//! Split from `repository.rs` (which had grown past 4,500 lines): this file
//! owns everything that shapes the database — DDL, the versioned migration
//! registry, FTS preprocessing/tokenization, and index-integrity checks.

use super::repository::{encode_fts_rowid, is_cjk_character, MemoryKind, MemoryRepository};
use anyhow::Result;
use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};

/// Highest schema version produced by [`MemoryRepository::initialize_schema`].
///
/// Existing databases whose `PRAGMA user_version` is below this are migrated
/// forward; new databases are created at this version directly (the CREATE
/// statements below encode the complete current schema, and a fresh DB —
/// detected BEFORE any table exists — is stamped without running migrations).
/// Bump this whenever a new migration is appended to
/// [`MemoryRepository::migrations`], and mirror any new column into the
/// CREATE statements so fresh DBs really do start complete.
pub(crate) const CURRENT_SCHEMA_VERSION: i32 = 10;

/// DDL for the unified FTS5 index over all four memory kinds. Shared by
/// `initialize_schema` (fresh DBs), the v9 migration, and `rebuild_fts` so
/// the three cannot drift.
///
/// ONE table for all kinds (v9, replacing the four per-type tables) so bm25
/// IDF is computed over the whole corpus: per-type tables gave each kind an
/// independent IDF scale, making raw scores incomparable across types (and
/// zeroing any term present in >=50% of a *small type's* rows). Column
/// mapping per kind (same title/body/tags abstraction as `Searchable`):
///
///   episodic    title=summary      body=content + files_touched
///   decision    title=title        body=context + rationale + tradeoffs
///   failure     title=incident     body=root_cause + fix + prevention
///   procedural  title=workflow_name body=steps + related_tools
///
/// `project` holds the project id's SHA-256 hex ([0-9a-f] only — one token
/// under any unicode61-class tokenizer, and index/query sides stem it
/// identically) and `kind` the MemoryKind name. Both are MATCH-only filter
/// columns with bm25 weight 0.0: they resolve project/kind filtering inside
/// the index (exact, no post-filtering of the ranked stream) without
/// contributing to the score or to bm25's column length normalization.
///
/// Contentless with delete support (SQLite >= 3.43) as before: the inverted
/// index is stored, the document text is NOT — the main tables already hold
/// it. The trade: FTS rows return no column values, so materialization walks
/// back to the main tables through the rowid encoding
/// `main_rowid * 4 + kind_index` (see [`encode_fts_rowid`]).
pub(crate) const FTS_TABLES_DDL: &str = "
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        title, body, tags, project, kind,
        tokenize='porter unicode61',
        content='', contentless_delete=1
    );
";

/// DDL for the PRE-v9 per-type FTS5 shadow tables. Frozen history: only the
/// v2/v6 migrations (which rebuild the index as it looked in their era, for
/// DBs upgrading through the chain) reference it. Everything current lives in
/// [`FTS_TABLES_DDL`].
pub(crate) const LEGACY_FTS_TABLES_DDL: &str = "
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

/// Insert for the unified FTS table: (title, body, tags, project_hex, kind,
/// rowid). One statement shared by every write path (CRUD macro, hand-rolled
/// creates, backfill/rebuild) so the column order cannot drift.
pub(crate) const UNIFIED_FTS_INSERT: &str =
    "INSERT INTO memories_fts (title, body, tags, project, kind, rowid) \
     VALUES (?1,?2,?3,?4,?5,?6)";

/// Per-column bm25 weights for `memories_fts`, in column order
/// (title, body, tags, project, kind). Filter columns weigh 0.0.
pub(crate) const UNIFIED_BM25_WEIGHTS: &str = "2.5, 1.0, 1.5, 0.0, 0.0";

/// Partial unique indexes enforcing write-time dedup (v10): ONE ACTIVE row
/// per (project_id, content_hash) per table. Rows with `content_hash IS NULL`
/// never participate — that is the mechanism behind every boundary case:
/// - write_dedup disabled → creates write NULL → duplicates are created freely;
/// - the v10 backfill keeps later byte-identical duplicates NULL → the index
///   is creatable on stores that already hold duplicates;
/// - a restore that would collide releases the restored row's hash to NULL
///   (restore wins over dedup bookkeeping).
///
/// Shared by `initialize_schema` (fresh DBs) and the v10 migration so the two
/// cannot drift — same discipline as FTS_TABLES_DDL.
pub(crate) const CONTENT_HASH_INDEX_DDL: &str = "
    CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_content_hash
        ON episodic_memories(project_id, content_hash)
        WHERE archived_at IS NULL AND content_hash IS NOT NULL;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_content_hash
        ON decision_memories(project_id, content_hash)
        WHERE archived_at IS NULL AND content_hash IS NOT NULL;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_failure_content_hash
        ON failure_memories(project_id, content_hash)
        WHERE archived_at IS NULL AND content_hash IS NOT NULL;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_procedural_content_hash
        ON procedural_memories(project_id, content_hash)
        WHERE archived_at IS NULL AND content_hash IS NOT NULL;
";

/// DDL for the explicit-feedback table (`mark_relevance`). Shared by
/// `initialize_schema` (fresh DBs) and the v8 migration so the two cannot
/// drift — same discipline as FTS_TABLES_DDL.
pub(crate) const MEMORY_FEEDBACK_DDL: &str = "
    -- One row per recorded mark_relevance judgment ('useful'/'irrelevant').
    -- Deliberately separate from `adoptions`: a passive get_memory read is
    -- behavioral evidence, explicit feedback is a judgment. The feedback
    -- cooldown and the tuner's demote guard key off THIS table; the adoption
    -- rate keys off `adoptions`.
    CREATE TABLE IF NOT EXISTS memory_feedback (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id  TEXT NOT NULL,
        memory_id   TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        feedback    TEXT NOT NULL,
        created_at  INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_feedback_memory
        ON memory_feedback(project_id, memory_id);
    CREATE INDEX IF NOT EXISTS idx_memory_feedback_project_time
        ON memory_feedback(project_id, created_at DESC);
";

/// A single registered migration: receives the repository and either succeeds
/// or returns an error. Factored into a type alias to keep `migrations()`'s
/// signature readable (clippy::type_complexity).
pub(crate) type MigrationFn = fn(&MemoryRepository) -> Result<()>;
impl MemoryRepository {
    pub fn initialize_schema(&self) -> Result<()> {
        // Fresh-DB detection MUST happen before any CREATE: after the batch
        // below, `has_memory_tables` is trivially true and every new database
        // would be misclassified as a legacy DB (running — and re-running —
        // the whole migration chain, including two full FTS rebuilds, on an
        // empty store). A truly fresh DB is stamped directly and skips
        // migrations; the CREATE statements below encode the complete
        // CURRENT_SCHEMA_VERSION schema.
        let fresh = self.user_version()? == 0 && !self.has_memory_tables()?;
        let conn = self.conn()?;
        // BEGIN IMMEDIATE, not the default deferred BEGIN: two processes
        // cold-starting the same fresh DB (CLI + MCP server) both upgrade a
        // deferred transaction to a write lock on the first CREATE; that
        // upgrade path can return SQLITE_BUSY without honoring busy_timeout.
        // IMMEDIATE takes the write lock up front, where busy_timeout applies.
        let tx = Transaction::new_unchecked(&conn, TransactionBehavior::Immediate)?;

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
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER,
                content_hash TEXT
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
                importance REAL NOT NULL DEFAULT 0.5,
                superseded_by TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                archived_at INTEGER,
                content_hash TEXT
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
                archived_at INTEGER,
                content_hash TEXT
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
                archived_at INTEGER,
                content_hash TEXT
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

        // Write-time dedup indexes (v10): one ACTIVE row per
        // (project, content_hash). NULL-hash rows are exempt — see
        // CONTENT_HASH_INDEX_DDL for the boundary semantics. On an EXISTING
        // pre-v10 DB the column does not exist yet — the v10 migration (run
        // by `run_migrations` immediately after this) adds the column and
        // then these indexes; probing keeps this batch from failing first.
        if tx
            .prepare("SELECT content_hash FROM episodic_memories LIMIT 0")
            .is_ok()
        {
            tx.execute_batch(CONTENT_HASH_INDEX_DDL)?;
        }

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

        // Per-event adoption log: one row each time a memory returned by a
        // search is actually read (get_memory) or flagged useful
        // (mark_relevance). query_log.adopted only counts per-query totals —
        // it cannot say WHICH memory was adopted. This table can, which is
        // what `engram tune` needs to learn per-memory importance from real
        // usage. Standalone table like reflection_suggestions: CREATE IF NOT
        // EXISTS suffices on existing DBs (no user_version bump).
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS adoptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_adoptions_project_time
                ON adoptions(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_adoptions_memory
                ON adoptions(project_id, memory_id);",
        )?;

        // Explicit-feedback log — the judgment half of the feedback signal,
        // split from the behavioral (adoption) half. See
        // MEMORY_FEEDBACK_DDL for the rationale; v8 on existing DBs.
        tx.execute_batch(MEMORY_FEEDBACK_DDL)?;

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
                resolved_at             INTEGER,
                rejected_occurrence_count INTEGER
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

        if fresh {
            // The CREATE statements above already produced the complete
            // CURRENT_SCHEMA_VERSION schema; stamp and skip migrations.
            self.set_user_version(CURRENT_SCHEMA_VERSION)?;
        } else {
            // Versioned migrations: stamp/advance `PRAGMA user_version` and
            // run any pending registered migrations. See `run_migrations`.
            self.run_migrations()?;
        }
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

    /// Tokens with no retrieval signal. Dropped from FTS queries AND the
    /// coverage denominator: without this, paraphrase queries like "how do
    /// we publish a new version" scored documents on "we"/"do" — rare-ish
    /// in a small corpus, so a document containing them could outscore the
    /// document matching the actual content words. Kept minimal and
    /// deliberately conservative (no negations).
    pub(crate) fn is_stopword(tok: &str) -> bool {
        matches!(
            tok,
            "a" | "an"
                | "the"
                | "do"
                | "does"
                | "how"
                | "what"
                | "why"
                | "when"
                | "we"
                | "i"
                | "you"
                | "it"
                | "is"
                | "are"
                | "of"
                | "for"
                | "to"
                | "in"
                | "on"
                | "and"
                | "or"
                | "with"
                | "my"
                | "our"
        )
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
    /// Stopwords are dropped from both forms.
    pub(crate) fn sanitize_fts_query(query: &str, require_all: bool) -> String {
        let processed = Self::preprocess_cjk(query);
        let tokens: Vec<&str> = processed
            .split_whitespace()
            .filter(|t| !Self::is_stopword(t))
            .collect();
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

    // ─── FTS integrity ────────────────────────────────────────────
    /// SHA-256 hex of a project id — the value stored in `memories_fts.project`
    /// and matched as `project:"<hex>"`. Hashed, not raw, because project ids
    /// are arbitrary user strings: raw ids could contain CJK (per-char
    /// splitting makes the phrase unmatchable) or FTS metacharacters, while a
    /// hex digest is one stable [0-9a-f] token under any tokenizer config.
    pub(crate) fn project_fts_hex(project_id: &str) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(project_id.as_bytes());
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(digest.len() * 2);
        for b in digest {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }

    /// Row-count drift between the main tables and the unified FTS index
    /// (sum of absolute per-kind differences: active main rows vs FTS rows of
    /// that kind). Every write path maintains a 1:1 mapping between ACTIVE
    /// main rows and FTS rows — archive deletes the FTS row, restore
    /// reinserts it — so 0 means aligned; non-zero means a repair
    /// (`rebuild_fts`) is warranted. Cheap COUNT(*)s — used to GATE the
    /// rebuild in `engram maintain` instead of rebuilding unconditionally.
    pub fn fts_drift(&self) -> Result<u64> {
        let conn = self.conn()?;
        let mut drift = 0u64;
        for kind in MemoryKind::all() {
            // Per-kind FTS row count via the kind filter column (contentless
            // tables expose no column values to COUNT over directly).
            let sql = format!(
                "SELECT ABS((SELECT COUNT(*) FROM {} WHERE archived_at IS NULL) - \
                        (SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?1))",
                kind.table()
            );
            let d: i64 =
                conn.query_row(&sql, params![format!("kind:\"{}\"", kind.as_str())], |r| {
                    r.get(0)
                })?;
            drift += d.unsigned_abs();
        }
        Ok(drift)
    }

    pub fn fts_integrity_check(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")?;
        Ok(())
    }
}

impl MemoryRepository {
    pub fn migrate_fts5_add_tags(&self) -> Result<()> {
        // Check if migration is needed by probing episodic_memories_fts for tags column
        let mut conn = self.conn()?;
        let needs_migration = conn
            .prepare("SELECT tags FROM episodic_memories_fts LIMIT 0")
            .is_err();
        if !needs_migration {
            return Ok(());
        }

        tracing::info!("Migrating FTS5 tables to include tags column...");
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

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
        Self::backfill_legacy_fts_from_main(&tx)?;

        tx.commit()?;
        tracing::info!("FTS5 migration complete — tags column added.");
        Ok(())
    }

    /// Rebuild the unified FTS index from the main tables in one transaction:
    /// removes orphan FTS rows, restores missing ones, drops rows of archived
    /// memories, and re-applies CJK preprocessing (which older migrations/
    /// paths may have skipped). Safe to run any time — it only ever makes the
    /// index match the ACTIVE main-table rows. Returns FTS rows written.
    pub fn rebuild_fts(&self) -> Result<usize> {
        let mut conn = self.conn()?;
        // IMMEDIATE: this transaction's first statement is a write (DELETE);
        // an up-front write lock plays far better with busy_timeout under
        // concurrent maintain/servers than a deferred BEGIN upgrading.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM memories_fts", [])?;
        let written = Self::backfill_unified_fts_from_main(&tx)?;
        tx.commit()?;
        Ok(written)
    }

    /// Insert one unified-FTS row per ACTIVE main-table row across all four
    /// kinds, with the same column preprocessing as the live write path (CJK
    /// split on text columns). Runs inside the caller's transaction; streams
    /// row-by-row instead of materializing whole tables.
    fn backfill_unified_fts_from_main(tx: &rusqlite::Transaction<'_>) -> Result<usize> {
        use super::repository::{
            unified_doc_decision, unified_doc_episodic, unified_doc_failure, unified_doc_procedural,
        };
        let mut written = 0usize;
        let insert = |fts_rowid: i64,
                      project_id: &str,
                      kind: MemoryKind,
                      doc: (String, String, String)|
         -> Result<()> {
            tx.execute(
                UNIFIED_FTS_INSERT,
                params![
                    doc.0,
                    doc.1,
                    doc.2,
                    Self::project_fts_hex(project_id),
                    kind.as_str(),
                    fts_rowid
                ],
            )?;
            Ok(())
        };

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, summary, content, files_touched, tags \
                 FROM episodic_memories WHERE archived_at IS NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let doc = unified_doc_episodic(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                );
                insert(
                    encode_fts_rowid(MemoryKind::Episodic, rid),
                    &project,
                    MemoryKind::Episodic,
                    doc,
                )?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, title, context, rationale, tradeoffs, tags \
                 FROM decision_memories WHERE archived_at IS NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let doc = unified_doc_decision(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                    &r.get::<_, String>(6)?,
                );
                insert(
                    encode_fts_rowid(MemoryKind::Decision, rid),
                    &project,
                    MemoryKind::Decision,
                    doc,
                )?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, incident, root_cause, fix, prevention, tags \
                 FROM failure_memories WHERE archived_at IS NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let doc = unified_doc_failure(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                    &r.get::<_, String>(6)?,
                );
                insert(
                    encode_fts_rowid(MemoryKind::Failure, rid),
                    &project,
                    MemoryKind::Failure,
                    doc,
                )?;
                written += 1;
            }
        }

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, workflow_name, steps, related_tools, tags \
                 FROM procedural_memories WHERE archived_at IS NULL",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let doc = unified_doc_procedural(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                );
                insert(
                    encode_fts_rowid(MemoryKind::Procedural, rid),
                    &project,
                    MemoryKind::Procedural,
                    doc,
                )?;
                written += 1;
            }
        }

        Ok(written)
    }

    /// Insert one FTS row per main-table row into the PRE-v9 per-type shadow
    /// tables, with the same column preprocessing as the era's write path.
    /// Legacy: only the v2/v6 migrations (rebuilding the index as it looked
    /// pre-v9 for DBs upgrading through the chain) call this; the unified
    /// table's backfill is [`Self::backfill_unified_fts_from_main`].
    fn backfill_legacy_fts_from_main(tx: &rusqlite::Transaction<'_>) -> Result<usize> {
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
        static REG: [(i32, MigrationFn); 9] = [
            (2, MemoryRepository::migrate_v2_fts_rowid),
            (3, MemoryRepository::migrate_v3_reflection_rearm),
            (4, MemoryRepository::migrate_v4_decision_importance),
            (5, MemoryRepository::migrate_v5_query_log_adoption),
            (6, MemoryRepository::migrate_v6_fts_contentless),
            (7, MemoryRepository::migrate_v7_decision_supersede),
            (8, MemoryRepository::migrate_v8_memory_feedback),
            (9, MemoryRepository::migrate_v9_unified_fts),
            (10, MemoryRepository::migrate_v10_content_hash),
        ];
        &REG
    }

    /// v10 — `content_hash` columns + partial unique indexes: write-time dedup
    /// moves from a two-step check-then-act (`find_exact_duplicate` full scan →
    /// create, two racing callers could both pass, O(N) on the session-hook
    /// hot path) into the PARTIAL UNIQUE INDEX — the INSERT statement itself
    /// is the guard. See [`CONTENT_HASH_INDEX_DDL`] and the `create_*_dedup`
    /// writers for the boundary semantics (NULL hash = not tracked).
    ///
    /// The backfill hashes every ACTIVE row's dedup text in Rust (same
    /// builders as the live write path). When the store already holds
    /// byte-identical ACTIVE duplicates (write_dedup was off, or direct
    /// storage writes), only the EARLIEST row per (project, hash) keeps its
    /// hash — later duplicates stay NULL so creating the unique index cannot
    /// fail and the rows keep working untracked, exactly like write_dedup
    /// off. Archived rows stay NULL. Streaming row-by-row (v9 pattern) inside
    /// ONE IMMEDIATE transaction. Idempotent: probes before altering,
    /// recomputing a hash yields the same value, CREATE INDEX IF NOT EXISTS.
    ///
    /// Accepted multi-process boundary: no cross-process mutex around the
    /// migration — every statement is idempotent, so two processes racing
    /// through v10 both succeed (the write lock serializes them).
    fn migrate_v10_content_hash(&self) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for kind in MemoryKind::all() {
            let table = kind.table();
            let has_col = tx
                .prepare(&format!("SELECT content_hash FROM {table} LIMIT 0"))
                .is_ok();
            if !has_col {
                tx.execute(
                    &format!("ALTER TABLE {table} ADD COLUMN content_hash TEXT"),
                    [],
                )?;
            }
        }
        let backfilled = Self::backfill_content_hash(&tx)?;
        tx.execute_batch(CONTENT_HASH_INDEX_DDL)?;
        tx.commit()?;
        tracing::info!("migration v10: content_hash backfilled ({backfilled} rows)");
        Ok(())
    }

    /// Stream every kind's ACTIVE rows (oldest first) and write their
    /// content_hash — unless an earlier row of the same project already
    /// carries the same hash (pre-existing duplicate: stays NULL, see
    /// `migrate_v10_content_hash`). The dedup text is assembled in Rust with
    /// the same per-kind builders the live write path uses. Runs inside the
    /// caller's transaction; column lists match the builders' expected order.
    fn backfill_content_hash(tx: &rusqlite::Transaction<'_>) -> Result<usize> {
        use super::repository::{
            content_hash, dedup_text_decision, dedup_text_episodic, dedup_text_failure,
            dedup_text_procedural,
        };
        use std::collections::HashSet;
        let mut written = 0usize;
        let mut seen: HashSet<(String, String)> = HashSet::new();

        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, summary, content FROM episodic_memories \
                 WHERE archived_at IS NULL ORDER BY created_at ASC, rowid ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let hash = content_hash(&dedup_text_episodic(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                ));
                if seen.insert((project, hash.clone())) {
                    tx.execute(
                        "UPDATE episodic_memories SET content_hash = ?1 WHERE rowid = ?2",
                        params![hash, rid],
                    )?;
                    written += 1;
                }
            }
        }
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, title, context, rationale, tradeoffs \
                 FROM decision_memories \
                 WHERE archived_at IS NULL ORDER BY created_at ASC, rowid ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let hash = content_hash(&dedup_text_decision(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                ));
                if seen.insert((project, hash.clone())) {
                    tx.execute(
                        "UPDATE decision_memories SET content_hash = ?1 WHERE rowid = ?2",
                        params![hash, rid],
                    )?;
                    written += 1;
                }
            }
        }
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, incident, root_cause, fix, prevention \
                 FROM failure_memories \
                 WHERE archived_at IS NULL ORDER BY created_at ASC, rowid ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let hash = content_hash(&dedup_text_failure(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                    &r.get::<_, String>(4)?,
                    &r.get::<_, String>(5)?,
                ));
                if seen.insert((project, hash.clone())) {
                    tx.execute(
                        "UPDATE failure_memories SET content_hash = ?1 WHERE rowid = ?2",
                        params![hash, rid],
                    )?;
                    written += 1;
                }
            }
        }
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, project_id, workflow_name, steps FROM procedural_memories \
                 WHERE archived_at IS NULL ORDER BY created_at ASC, rowid ASC",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let (rid, project): (i64, String) = (r.get(0)?, r.get(1)?);
                let hash = content_hash(&dedup_text_procedural(
                    &r.get::<_, String>(2)?,
                    &r.get::<_, String>(3)?,
                ));
                if seen.insert((project, hash.clone())) {
                    tx.execute(
                        "UPDATE procedural_memories SET content_hash = ?1 WHERE rowid = ?2",
                        params![hash, rid],
                    )?;
                    written += 1;
                }
            }
        }
        Ok(written)
    }

    /// v9 — replace the four per-type FTS tables with ONE unified index over
    /// all memory kinds (see FTS_TABLES_DDL). Per-type tables computed bm25
    /// IDF over four independent corpora: raw scores were incomparable across
    /// types (retrieval needed a whole calibration layer) and any term
    /// present in >=50% of a SMALL type's rows scored ~0. One table gives
    /// every query one IDF scale over the whole corpus.
    ///
    /// The backfill goes through Rust (not INSERT..SELECT) for the same CJK
    /// reason as v2, and also to compute the project-hex/kind filter columns
    /// and the `rowid * 4 + kind_index` encoding SQL cannot express. Only
    /// ACTIVE rows are indexed — with the unified table, archive deletes the
    /// FTS row (see `MemoryRepository::archive`), so the index invariant
    /// changes from "one row per main row" to "one row per ACTIVE row".
    /// Idempotent by construction (rebuilds from the main tables).
    fn migrate_v9_unified_fts(&self) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for table in [
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
            // Drop-first also covers a partially-migrated re-run.
            "memories_fts",
        ] {
            tx.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
        }
        tx.execute_batch(FTS_TABLES_DDL)?;
        let written = Self::backfill_unified_fts_from_main(&tx)?;
        tx.commit()?;
        tracing::info!("migration v9: unified FTS index built ({written} rows)");
        Ok(())
    }

    /// v8 — `memory_feedback`: explicit `mark_relevance` feedback split from
    /// the passive `adoptions` read signal, so reading a memory (an adoption
    /// event) no longer trips the feedback cooldown. Idempotent by
    /// construction (CREATE IF NOT EXISTS).
    fn migrate_v8_memory_feedback(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(MEMORY_FEEDBACK_DDL)?;
        Ok(())
    }

    /// v7 — `decision_memories.superseded_by`: the decision supersede chain.
    /// A decision replaced by a newer one keeps its row (searchable history)
    /// but is downranked and flagged in retrieval.
    fn migrate_v7_decision_supersede(&self) -> Result<()> {
        let conn = self.conn()?;
        let has_col = conn
            .prepare("SELECT superseded_by FROM decision_memories LIMIT 0")
            .is_ok();
        if !has_col {
            conn.execute_batch("ALTER TABLE decision_memories ADD COLUMN superseded_by TEXT;")?;
        }
        Ok(())
    }

    /// v6 — rebuild the FTS shadow tables as contentless tables (no text
    /// copy; join by rowid). See LEGACY_FTS_TABLES_DDL for the rationale.
    /// Recreates the v6-ERA per-type tables on purpose: this migration
    /// reproduces the historical schema for DBs upgrading through the chain;
    /// v9 later replaces these tables with the unified index.
    fn migrate_v6_fts_contentless(&self) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for table in [
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            tx.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
        }
        tx.execute_batch(LEGACY_FTS_TABLES_DDL)?;
        let written = Self::backfill_legacy_fts_from_main(&tx)?;
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
    /// construction (rebuilding from the main tables). Recreates the v2-ERA
    /// per-type tables (see LEGACY_FTS_TABLES_DDL) — DBs upgrading the whole
    /// chain are converted to the unified index by the v9 migration.
    fn migrate_v2_fts_rowid(&self) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for table in [
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            tx.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
        }
        tx.execute_batch(LEGACY_FTS_TABLES_DDL)?;
        let written = Self::backfill_legacy_fts_from_main(&tx)?;
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
