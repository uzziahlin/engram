use crate::models::SanitizeMemory;
use crate::models::*;
use crate::storage::schema::UNIFIED_BM25_WEIGHTS;
use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension, TransactionBehavior};
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

/// An active memory touching a file, for `related_memories_for_file` / the
/// MCP `related_files` tool. Unlike `RelatedFileEdge`, this carries the
/// memory's TYPE and a readable label — agents can consume it directly
/// (fetch more via `get_memory(memory_type, id)`) instead of a bare UUID.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RelatedMemory {
    pub memory_type: String,
    pub id: String,
    pub label: String,
    pub relation: String,
}

/// Aggregated feedback for one distinct query string (for `query_stats`).
/// `count` = how often the query ran; `result_count_avg` = average hits per
/// run (a low value flags a query the store fails to satisfy).
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryStatRow {
    pub query: String,
    pub count: i64,
    pub result_count_avg: f64,
    /// Times a result of this query was later fetched (get_memory) — the
    /// adoption signal for ranking-quality feedback.
    pub adopted: i64,
    pub last_at: i64,
}

/// Per-project active memory count (`engram stats`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProjectCount {
    pub project_id: String,
    pub active: i64,
}

/// Reflection proposal state counts (`engram stats`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReflectionCounts {
    pub pending: i64,
    pub confirmed: i64,
    pub rejected: i64,
}

/// Aggregate store snapshot (`engram stats`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct StatsSnapshot {
    pub schema_version: i32,
    pub fts_tokenizer: Option<String>,
    /// (kind, active, archived) per memory type.
    pub memories: Vec<(String, i64, i64)>,
    pub total_active: i64,
    pub projects: Vec<ProjectCount>,
    pub entities_total: i64,
    pub orphan_entities: i64,
    pub reflections: ReflectionCounts,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Slot of this kind in the unified-FTS rowid encoding. MUST stay in
    /// `MemoryKind::all()` order and in lockstep with
    /// [`MemoryRepository::encode_fts_rowid`] / `decode_fts_rowid`.
    pub fn index(self) -> i64 {
        match self {
            MemoryKind::Episodic => 0,
            MemoryKind::Decision => 1,
            MemoryKind::Failure => 2,
            MemoryKind::Procedural => 3,
        }
    }

    /// Inverse of [`Self::index`].
    pub fn from_index(i: i64) -> Option<Self> {
        match i {
            0 => Some(MemoryKind::Episodic),
            1 => Some(MemoryKind::Decision),
            2 => Some(MemoryKind::Failure),
            3 => Some(MemoryKind::Procedural),
            _ => None,
        }
    }
}

/// Kind slots packed into the low bits of a unified-FTS rowid.
pub(crate) const FTS_KIND_SLOTS: i64 = 4;

/// Deterministic, collision-free pack of `(kind, main-table rowid)` into the
/// single i64 rowid a contentless FTS5 table can be addressed by. The FTS
/// table stores no column values, so this encoding is the ONLY way search
/// materialization walks back to the owning main-table row.
pub(crate) fn encode_fts_rowid(kind: MemoryKind, main_rowid: i64) -> i64 {
    main_rowid * FTS_KIND_SLOTS + kind.index()
}

/// Inverse of [`encode_fts_rowid`]; `None` when the rowid is not in the
/// encoding's range (corrupt / foreign row).
pub(crate) fn decode_fts_rowid(fts_rowid: i64) -> Option<(MemoryKind, i64)> {
    if fts_rowid < FTS_KIND_SLOTS {
        return None;
    }
    let kind = MemoryKind::from_index(fts_rowid % FTS_KIND_SLOTS)?;
    Some((kind, fts_rowid / FTS_KIND_SLOTS))
}

/// (title, body, tags) projection of an episodic memory for the unified FTS
/// table. `files_json`/`tags_json` are the stored JSON strings — brackets and
/// commas are tokenizer separators, so the JSON form yields the same tokens
/// as a plain join and lets the live path and the backfill share ONE formula
/// per kind. Keep in sync with the macro invocation sites.
pub(crate) fn unified_doc_episodic(
    summary: &str,
    content: &str,
    files_json: &str,
    tags_json: &str,
) -> (String, String, String) {
    (
        MemoryRepository::preprocess_cjk(summary),
        MemoryRepository::preprocess_cjk(&format!("{content}\n{files_json}")),
        MemoryRepository::preprocess_cjk(tags_json),
    )
}

/// (title, body, tags) projection of a decision memory; see
/// [`unified_doc_episodic`].
pub(crate) fn unified_doc_decision(
    title: &str,
    context: &str,
    rationale: &str,
    tradeoffs: &str,
    tags_json: &str,
) -> (String, String, String) {
    (
        MemoryRepository::preprocess_cjk(title),
        MemoryRepository::preprocess_cjk(&format!("{context}\n{rationale}\n{tradeoffs}")),
        MemoryRepository::preprocess_cjk(tags_json),
    )
}

/// (title, body, tags) projection of a failure memory; see
/// [`unified_doc_episodic`].
pub(crate) fn unified_doc_failure(
    incident: &str,
    root_cause: &str,
    fix: &str,
    prevention: &str,
    tags_json: &str,
) -> (String, String, String) {
    (
        MemoryRepository::preprocess_cjk(incident),
        MemoryRepository::preprocess_cjk(&format!("{root_cause}\n{fix}\n{prevention}")),
        MemoryRepository::preprocess_cjk(tags_json),
    )
}

/// (title, body, tags) projection of a procedural memory; see
/// [`unified_doc_episodic`].
pub(crate) fn unified_doc_procedural(
    workflow_name: &str,
    steps_json: &str,
    tools_json: &str,
    tags_json: &str,
) -> (String, String, String) {
    (
        MemoryRepository::preprocess_cjk(workflow_name),
        MemoryRepository::preprocess_cjk(&format!("{steps_json}\n{tools_json}")),
        MemoryRepository::preprocess_cjk(tags_json),
    )
}

/// SHA-256 hex of a memory's dedup text — the value stored in the main
/// tables' `content_hash` column and enforced unique per project among ACTIVE
/// rows (partial unique indexes `idx_<table>_content_hash`, migration v10).
/// sha256 over the full text: the unique index treats hash equality as text
/// equality, so a collision-prone short digest would reject legitimate writes.
pub(crate) fn content_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Whether this error is SQLite rejecting a row through a UNIQUE constraint.
/// This is the enforcement mechanism of the content-hash dedup (v10): the
/// INSERT/UPDATE statement itself is the guard, replacing the old two-step
/// `find_exact_duplicate` → write that two racing callers could both pass.
pub(crate) fn is_unique_violation(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(ffi, _)
            if ffi.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
    )
}

/// Rust-side dedup-text concatenation per kind — the SAME byte sequence the
/// old SQL `dedup_text_expr` produced (`col || char(10) || col`; columns are
/// NOT NULL so SQL-NULL propagation never applied). One formula now lives
/// here, shared by the hash write path (`create_*_dedup`, updates), the
/// consolidation scan, and the v10 backfill. Keep the column ORDER in
/// lockstep with the table DDL and these builders.
pub(crate) fn dedup_text_episodic(summary: &str, content: &str) -> String {
    format!("{summary}\n{content}")
}

/// See [`dedup_text_episodic`].
pub(crate) fn dedup_text_decision(
    title: &str,
    context: &str,
    rationale: &str,
    tradeoffs: &str,
) -> String {
    format!("{title}\n{context}\n{rationale}\n{tradeoffs}")
}

/// See [`dedup_text_episodic`].
pub(crate) fn dedup_text_failure(
    incident: &str,
    root_cause: &str,
    fix: &str,
    prevention: &str,
) -> String {
    format!("{incident}\n{root_cause}\n{fix}\n{prevention}")
}

/// See [`dedup_text_episodic`]. `steps` is the STORED JSON string of the step
/// list (exactly what the `steps` column holds — the SQL form concatenated
/// the raw column, not a re-serialized array).
pub(crate) fn dedup_text_procedural(workflow_name: &str, steps_json: &str) -> String {
    format!("{workflow_name}\n{steps_json}")
}

/// Type-erased active memory for batch id materialization (vector-only
/// hits, graph neighbors) — see
/// [`MemoryRepository::get_active_any_batch`].
pub enum ActiveMemory {
    Episodic(EpisodicMemory),
    Decision(DecisionMemory),
    Failure(FailureMemory),
    Procedural(ProceduralMemory),
}

/// One BM25 hit from the unified index: encoded rowid + raw score.
struct UnifiedFtsHit {
    rowid: i64,
    /// Raw `bm25()` output (negative, more negative = better). Kept in
    /// FTS5's own sign convention so `ScoredMemory` consumers see the same
    /// scale the per-type tables used to produce.
    raw: f64,
}

/// Typed result buckets of a unified search ([`MemoryRepository::search_unified`]).
/// Order inside each bucket is BM25 rank order; `bm25_score` is the RAW
/// unified score — comparable across buckets, which is the whole point of
/// the unified table.
#[derive(Default)]
pub struct UnifiedHits {
    pub episodic: Vec<ScoredMemory<EpisodicMemory>>,
    pub decisions: Vec<ScoredMemory<DecisionMemory>>,
    pub failures: Vec<ScoredMemory<FailureMemory>>,
    pub procedures: Vec<ScoredMemory<ProceduralMemory>>,
}

/// Shared body of the typed materializers: fill `by_rowid` from chunked
/// batch queries, then emit `ScoredMemory`s in rank order. Module-level
/// macro_rules must textually precede their first use.
macro_rules! materialize_body {
    ($conn:expr, $group:expr, $project_id:expr, $kind:expr, $rid_idx:literal, $mapper:expr) => {{
        let mut by_rowid: std::collections::HashMap<i64, _> = std::collections::HashMap::new();
        let rowids: Vec<i64> = $group.iter().map(|(r, _)| *r).collect();
        for chunk in rowids.chunks(500) {
            let (sql, boxed) = Self::unified_materialize_sql($kind, chunk, $project_id);
            let params_ref: Vec<&dyn rusqlite::ToSql> = boxed.iter().map(|b| b.as_ref()).collect();
            let mut stmt = $conn.prepare(&sql)?;
            let mapper = $mapper;
            let rows = stmt.query_map(params_ref.as_slice(), |row| {
                Ok((row.get::<_, i64>($rid_idx)?, mapper(row)?))
            })?;
            for r in rows {
                let (rid, mem) = r?;
                by_rowid.insert(rid, mem);
            }
        }
        let mut out = Vec::new();
        for (rid, raw) in $group {
            if let Some(mem) = by_rowid.get(rid) {
                out.push(ScoredMemory {
                    memory: mem.clone(),
                    bm25_score: *raw,
                });
            }
        }
        Ok(out)
    }};
}

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
    pub(crate) fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("failed to get pooled connection")
    }

    // ─── Macro-generated Memory CRUD + FTS5 Search ─────────────────

    impl_memory_crud! {
        mem = mem,
        tx = tx,
        row = row,
        link_ts = link_ts,
        fts_rowid = fts_rowid,
        hash = hash,

        struct_type = EpisodicMemory,
        table = "episodic_memories",

        kind_variant = Episodic,
        kind_index = 0,
        kind_str = "episodic",
        unified_hits_field = episodic,

        create_fn = create_episodic,
        create_dedup_fn = create_episodic_dedup,
        get_fn = get_episodic,
        get_active_fn = get_active_episodic,
        update_fn = update_episodic,
        delete_fn = delete_episodic,
        search_fn = search_episodic,
        list_active_fn = list_active_episodic,

        select_cols = "id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at",

        insert_sql = "INSERT INTO episodic_memories (id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at, content_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",

        unified_doc = {
            unified_doc_episodic(
                &mem.summary,
                &mem.content,
                &serde_json::to_string(&mem.files_touched)?,
                &serde_json::to_string(&mem.tags)?,
            )
        },

        dedup_text = {
            dedup_text_episodic(&mem.summary, &mem.content)
        },

        update_sql = "UPDATE episodic_memories SET session_id=?3, summary=?4, content=?5, files_touched=?6, related_commits=?7, importance=?8, tags=?9, updated_at=?10, content_hash=CASE WHEN content_hash IS NULL THEN NULL ELSE ?11 END WHERE id=?1 AND project_id=?2 AND archived_at IS NULL",

        insert_params = {
            params![
                mem.id, mem.project_id, mem.session_id,
                mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.related_commits)?,
                mem.importance,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
                hash,
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
                hash,
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
        hash = hash,

        struct_type = DecisionMemory,
        table = "decision_memories",

        kind_variant = Decision,
        kind_index = 1,
        kind_str = "decision",
        unified_hits_field = decisions,

        create_fn = create_decision,
        create_dedup_fn = create_decision_dedup,
        get_fn = get_decision,
        get_active_fn = get_active_decision,
        update_fn = update_decision,
        delete_fn = delete_decision,
        search_fn = search_decisions,
        list_active_fn = list_active_decision,

        select_cols = "id, project_id, title, context, rationale, tradeoffs, related_files, tags, importance, superseded_by, created_at, updated_at",

        insert_sql = "INSERT INTO decision_memories (id, project_id, title, context, rationale, tradeoffs, related_files, tags, importance, superseded_by, created_at, updated_at, content_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",

        unified_doc = {
            unified_doc_decision(
                &mem.title,
                &mem.context,
                &mem.rationale,
                &mem.tradeoffs,
                &serde_json::to_string(&mem.tags)?,
            )
        },

        dedup_text = {
            dedup_text_decision(&mem.title, &mem.context, &mem.rationale, &mem.tradeoffs)
        },

        update_sql = "UPDATE decision_memories SET title=?3, context=?4, rationale=?5, tradeoffs=?6, related_files=?7, tags=?8, importance=?9, updated_at=?10, superseded_by=?11, content_hash=CASE WHEN content_hash IS NULL THEN NULL ELSE ?12 END WHERE id=?1 AND project_id=?2 AND archived_at IS NULL",

        insert_params = {
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.importance,
                mem.superseded_by,
                mem.created_at, mem.updated_at,
                hash,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.importance,
                mem.updated_at,
                mem.superseded_by,
                hash,
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
                importance: row.get(8)?,
                superseded_by: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
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
        hash = hash,

        struct_type = FailureMemory,
        table = "failure_memories",

        kind_variant = Failure,
        kind_index = 2,
        kind_str = "failure",
        unified_hits_field = failures,

        create_fn = create_failure,
        create_dedup_fn = create_failure_dedup,
        get_fn = get_failure,
        get_active_fn = get_active_failure,
        update_fn = update_failure,
        delete_fn = delete_failure,
        search_fn = search_failures,
        list_active_fn = list_active_failure,

        select_cols = "id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at",

        insert_sql = "INSERT INTO failure_memories (id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at, content_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",

        unified_doc = {
            unified_doc_failure(
                &mem.incident,
                &mem.root_cause,
                &mem.fix,
                &mem.prevention,
                &serde_json::to_string(&mem.tags)?,
            )
        },

        dedup_text = {
            dedup_text_failure(&mem.incident, &mem.root_cause, &mem.fix, &mem.prevention)
        },

        update_sql = "UPDATE failure_memories SET incident=?3, root_cause=?4, fix=?5, prevention=?6, severity=?7, tags=?8, updated_at=?9, content_hash=CASE WHEN content_hash IS NULL THEN NULL ELSE ?10 END WHERE id=?1 AND project_id=?2 AND archived_at IS NULL",

        insert_params = {
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
                hash,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
                hash,
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
        hash = hash,

        struct_type = ProceduralMemory,
        table = "procedural_memories",

        kind_variant = Procedural,
        kind_index = 3,
        kind_str = "procedural",
        unified_hits_field = procedures,

        create_fn = create_procedural,
        create_dedup_fn = create_procedural_dedup,
        get_fn = get_procedural,
        get_active_fn = get_active_procedural,
        update_fn = update_procedural,
        delete_fn = delete_procedural,
        search_fn = search_procedural,
        list_active_fn = list_active_procedural,

        select_cols = "id, project_id, workflow_name, steps, related_tools, tags, importance, created_at, updated_at",

        insert_sql = "INSERT INTO procedural_memories (id, project_id, workflow_name, steps, related_tools, tags, importance, created_at, updated_at, content_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",

        unified_doc = {
            unified_doc_procedural(
                &mem.workflow_name,
                &serde_json::to_string(&mem.steps)?,
                &serde_json::to_string(&mem.related_tools)?,
                &serde_json::to_string(&mem.tags)?,
            )
        },

        dedup_text = {
            dedup_text_procedural(
                &mem.workflow_name,
                &serde_json::to_string(&mem.steps)?,
            )
        },

        update_sql = "UPDATE procedural_memories SET workflow_name=?3, steps=?4, related_tools=?5, tags=?6, importance=?7, updated_at=?8, content_hash=CASE WHEN content_hash IS NULL THEN NULL ELSE ?9 END WHERE id=?1 AND project_id=?2 AND archived_at IS NULL",

        insert_params = {
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.importance,
                mem.created_at, mem.updated_at,
                hash,
            ]
        },

        update_params = {
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.importance,
                mem.updated_at,
                hash,
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
                importance: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
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

    // ─── Unified FTS search ────────────────────────────────────────

    /// Search every memory kind through the ONE unified FTS table.
    ///
    /// All filtering happens inside the MATCH expression — business terms
    /// (phrase-quoted by `sanitize_fts_query`, wrapped in parens so the
    /// appended filters cannot bind into an OR arm), plus optional
    /// `project:"<sha256-hex>"` / `kind:"<kind>"` column filters with bm25
    /// weight 0. The AND-form runs first; on an empty result the OR-form
    /// runs — ONE fallback decision over the whole corpus (the per-type
    /// tables re-decided per kind, so one kind could match under AND
    /// semantics while another fell back to OR).
    ///
    /// Hits are decoded (`kind = rowid % 4`) and materialized per kind in
    /// batches. The main-table query re-checks `archived_at IS NULL` and the
    /// project id — the index should never contain archived rows (archive
    /// deletes them) and the hex filter should never leak a row, but the
    /// main tables remain the authority.
    /// Materialize `(memory_type, id)` pairs in ONE batched query per kind:
    /// `get_active_*` per id was a round-trip per candidate. Same guards as
    /// those getters — `project_id = '*'` matches any project, archived rows
    /// never materialize. Unknown types are dropped; input order is kept.
    pub fn get_active_any_batch(
        &self,
        ids: &[(String, String)],
        project_id: &str,
    ) -> Result<Vec<ActiveMemory>> {
        use std::collections::HashMap;
        let mut by_kind: HashMap<MemoryKind, Vec<&str>> = HashMap::new();
        for (memory_type, id) in ids {
            if let Ok(kind) = MemoryKind::from_type_str(memory_type) {
                by_kind.entry(kind).or_default().push(id);
            }
        }
        let mut by_id: HashMap<(MemoryKind, String), ActiveMemory> = HashMap::new();
        let conn = self.conn()?;
        for (kind, kind_ids) in by_kind {
            for chunk in kind_ids.chunks(500) {
                let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!(
                    "SELECT {} FROM {} WHERE id IN ({placeholders}) \
                     AND archived_at IS NULL AND (?{} = '*' OR project_id = ?{})",
                    Self::unified_select_cols(kind),
                    kind.table(),
                    chunk.len() + 1,
                    chunk.len() + 1
                );
                let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = chunk
                    .iter()
                    .map(|i| Box::new(i.to_string()) as Box<dyn rusqlite::ToSql>)
                    .collect();
                params_vec.push(Box::new(project_id.to_string()));
                let params_ref: Vec<&dyn rusqlite::ToSql> =
                    params_vec.iter().map(|b| b.as_ref()).collect();
                let mut stmt = conn.prepare(&sql)?;
                match kind {
                    MemoryKind::Episodic => {
                        for m in stmt.query_map(params_ref.as_slice(), Self::map_episodic_row)? {
                            let m = m?;
                            by_id.insert((kind, m.id.clone()), ActiveMemory::Episodic(m));
                        }
                    }
                    MemoryKind::Decision => {
                        for m in stmt.query_map(params_ref.as_slice(), Self::map_decision_row)? {
                            let m = m?;
                            by_id.insert((kind, m.id.clone()), ActiveMemory::Decision(m));
                        }
                    }
                    MemoryKind::Failure => {
                        for m in stmt.query_map(params_ref.as_slice(), Self::map_failure_row)? {
                            let m = m?;
                            by_id.insert((kind, m.id.clone()), ActiveMemory::Failure(m));
                        }
                    }
                    MemoryKind::Procedural => {
                        for m in stmt.query_map(params_ref.as_slice(), Self::map_procedural_row)? {
                            let m = m?;
                            by_id.insert((kind, m.id.clone()), ActiveMemory::Procedural(m));
                        }
                    }
                }
            }
        }
        let mut out = Vec::new();
        for (memory_type, id) in ids {
            let key = MemoryKind::from_type_str(memory_type)
                .ok()
                .map(|k| (k, id.clone()));
            if let Some((kind, id)) = key {
                if let Some(m) = by_id.remove(&(kind, id)) {
                    out.push(m);
                }
            }
        }
        Ok(out)
    }

    pub fn search_unified(
        &self,
        query: &str,
        project_id: &str,
        kind_filter: Option<MemoryKind>,
        limit: usize,
    ) -> Result<UnifiedHits> {
        if limit == 0 {
            return Ok(UnifiedHits::default());
        }
        let and_query = Self::sanitize_fts_query(query, true);
        let or_query = Self::sanitize_fts_query(query, false);
        let project_hex = if project_id == "*" {
            None
        } else {
            Some(Self::project_fts_hex(project_id))
        };

        // Business part in parens: `("a" OR "b") AND project:"…"` — without
        // them the trailing filters would bind only into the last OR arm.
        // A business query that sanitized to `""` (no content tokens) can
        // match nothing, filters or not.
        let compose = |business: &str| -> Option<String> {
            if business == "\"\"" {
                return None;
            }
            let mut m = format!("({business})");
            if let Some(hex) = &project_hex {
                m.push_str(&format!(" AND project:\"{hex}\""));
            }
            if let Some(kind) = kind_filter {
                m.push_str(&format!(" AND kind:\"{}\"", kind.as_str()));
            }
            Some(m)
        };

        let fetch = |match_expr: &str| -> Result<Vec<UnifiedFtsHit>> {
            let sql = format!(
                "SELECT rowid, bm25(memories_fts, {UNIFIED_BM25_WEIGHTS}) as score \
                 FROM memories_fts WHERE memories_fts MATCH ?1 \
                 ORDER BY score LIMIT ?2"
            );
            let conn = self.conn()?;
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![match_expr, limit as i64], |r| {
                Ok(UnifiedFtsHit {
                    rowid: r.get(0)?,
                    raw: r.get(1)?,
                })
            })?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok(out)
        };

        let (and_expr, or_expr) = (compose(&and_query), compose(&or_query));
        let mut hits = match &and_expr {
            Some(expr) => fetch(expr)?,
            None => Vec::new(),
        };
        // OR fallback: one decision for the whole corpus (see method doc).
        if hits.is_empty() {
            if let Some(expr) = or_expr {
                if Some(&expr) != and_expr.as_ref() {
                    hits = fetch(&expr)?;
                }
            }
        }

        // Decode → per-kind (main_rowid, raw) groups, preserving rank order.
        let mut per_kind: [Vec<(i64, f64)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for hit in hits {
            if let Some((kind, main_rowid)) = decode_fts_rowid(hit.rowid) {
                per_kind[kind.index() as usize].push((main_rowid, hit.raw));
            }
        }

        let conn = self.conn()?;
        let mut out = UnifiedHits::default();
        for (kind, group) in MemoryKind::all().into_iter().zip(per_kind) {
            if group.is_empty() {
                continue;
            }
            match kind {
                MemoryKind::Episodic => {
                    out.episodic = Self::materialize_episodic(&conn, &group, project_id)?;
                }
                MemoryKind::Decision => {
                    out.decisions = Self::materialize_decisions(&conn, &group, project_id)?;
                }
                MemoryKind::Failure => {
                    out.failures = Self::materialize_failures(&conn, &group, project_id)?;
                }
                MemoryKind::Procedural => {
                    out.procedures = Self::materialize_procedures(&conn, &group, project_id)?;
                }
            }
        }
        Ok(out)
    }

    /// SQL + bound params for one batch-materialization chunk. `_rid` rides
    /// as the LAST column so the map_*_row mappers keep reading columns 0..n.
    fn unified_materialize_sql(
        kind: MemoryKind,
        rowids: &[i64],
        project_id: &str,
    ) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
        let placeholders = rowids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT {}, rowid AS _rid FROM {} WHERE rowid IN ({placeholders}) \
             AND archived_at IS NULL AND (?{} = '*' OR project_id = ?{})",
            Self::unified_select_cols(kind),
            kind.table(),
            rowids.len() + 1,
            rowids.len() + 1
        );
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = rowids
            .iter()
            .map(|r| Box::new(*r) as Box<dyn rusqlite::ToSql>)
            .collect();
        params.push(Box::new(project_id.to_string()));
        (sql, params)
    }

    fn materialize_episodic(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        group: &[(i64, f64)],
        project_id: &str,
    ) -> Result<Vec<ScoredMemory<EpisodicMemory>>> {
        materialize_body!(
            conn,
            group,
            project_id,
            MemoryKind::Episodic,
            11,
            Self::map_episodic_row
        )
    }

    fn materialize_decisions(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        group: &[(i64, f64)],
        project_id: &str,
    ) -> Result<Vec<ScoredMemory<DecisionMemory>>> {
        materialize_body!(
            conn,
            group,
            project_id,
            MemoryKind::Decision,
            12,
            Self::map_decision_row
        )
    }

    fn materialize_failures(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        group: &[(i64, f64)],
        project_id: &str,
    ) -> Result<Vec<ScoredMemory<FailureMemory>>> {
        materialize_body!(
            conn,
            group,
            project_id,
            MemoryKind::Failure,
            10,
            Self::map_failure_row
        )
    }

    fn materialize_procedures(
        conn: &r2d2::PooledConnection<SqliteConnectionManager>,
        group: &[(i64, f64)],
        project_id: &str,
    ) -> Result<Vec<ScoredMemory<ProceduralMemory>>> {
        materialize_body!(
            conn,
            group,
            project_id,
            MemoryKind::Procedural,
            9,
            Self::map_procedural_row
        )
    }

    // ─── Lifecycle: archive / restore (soft delete) ────────────────

    /// Column list for batch materialization per kind (order matches the
    /// map_*_row mappers).
    fn unified_select_cols(kind: MemoryKind) -> &'static str {
        match kind {
            MemoryKind::Episodic => {
                "id, project_id, session_id, summary, content, files_touched, \
                 related_commits, importance, tags, created_at, updated_at"
            }
            MemoryKind::Decision => {
                "id, project_id, title, context, rationale, tradeoffs, \
                 related_files, tags, importance, superseded_by, created_at, updated_at"
            }
            MemoryKind::Failure => {
                "id, project_id, incident, root_cause, fix, prevention, \
                 severity, tags, created_at, updated_at"
            }
            MemoryKind::Procedural => {
                "id, project_id, workflow_name, steps, related_tools, tags, \
                 importance, created_at, updated_at"
            }
        }
    }

    fn map_episodic_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<EpisodicMemory> {
        Ok(EpisodicMemory {
            id: row.get(0)?,
            project_id: row.get(1)?,
            session_id: row.get(2)?,
            summary: row.get(3)?,
            content: row.get(4)?,
            files_touched: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
            related_commits: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
            importance: row.get(7)?,
            tags: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
        })
    }

    fn map_decision_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DecisionMemory> {
        Ok(DecisionMemory {
            id: row.get(0)?,
            project_id: row.get(1)?,
            title: row.get(2)?,
            context: row.get(3)?,
            rationale: row.get(4)?,
            tradeoffs: row.get(5)?,
            related_files: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
            tags: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
            importance: row.get(8)?,
            superseded_by: row.get(9)?,
            created_at: row.get(10)?,
            updated_at: row.get(11)?,
        })
    }

    fn map_failure_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<FailureMemory> {
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
    }

    fn map_procedural_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProceduralMemory> {
        Ok(ProceduralMemory {
            id: row.get(0)?,
            project_id: row.get(1)?,
            workflow_name: row.get(2)?,
            steps: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
            related_tools: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            tags: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
            importance: row.get(6)?,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
        })
    }

    // ─── Lifecycle: archive / restore (soft delete) ────────────────

    /// Soft-delete (archive) a memory by id within a project.
    /// Returns true if a currently-active row was archived.
    ///
    /// The unified FTS row is deleted in the same transaction: the index
    /// invariant is "one FTS row per ACTIVE main row" (archive must not
    /// leave FTS rows behind — `fts_drift` and `rebuild_fts` both key on it,
    /// and a stale row would resurface the memory in search until the next
    /// rebuild).
    pub fn archive(&self, kind: MemoryKind, id: &str, project_id: &str, now: i64) -> Result<bool> {
        let sql = format!(
            "UPDATE {} SET archived_at = ?3 WHERE id = ?1 AND project_id = ?2 AND archived_at IS NULL",
            kind.table()
        );
        let mut conn = self.conn()?;
        // Write tx takes the write lock up front (BEGIN IMMEDIATE): busy_timeout
        // applies to contention instead of the deferred-upgrade SQLITE_BUSY path.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let affected = tx.execute(&sql, params![id, project_id, now])?;
        if affected > 0 {
            Self::delete_fts_row(&tx, kind, id, project_id)?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    /// Un-archive a previously soft-deleted memory.
    /// Returns true if a currently-archived row was restored. The unified
    /// FTS row is re-inserted in the same transaction (archive deleted it).
    pub fn restore(&self, kind: MemoryKind, id: &str, project_id: &str) -> Result<bool> {
        let sql = format!(
            "UPDATE {} SET archived_at = NULL WHERE id = ?1 AND project_id = ?2 AND archived_at IS NOT NULL",
            kind.table()
        );
        let mut conn = self.conn()?;
        // Write tx takes the write lock up front (BEGIN IMMEDIATE): busy_timeout
        // applies to contention instead of the deferred-upgrade SQLITE_BUSY path.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Restore-vs-dedup boundary (v10): if the restored row's content_hash
        // equals the hash of an ACTIVE row, clearing `archived_at` trips the
        // partial unique index. Restore WINS over dedup bookkeeping: THIS
        // row's hash is set to NULL (NULL rows never participate in the index)
        // and the restore proceeds — the memory comes back fully usable, only
        // its future dedup tracking is waived. Logged for the audit trail.
        let affected = match tx.execute(&sql, params![id, project_id]) {
            Ok(n) => n,
            Err(e) if is_unique_violation(&e) => {
                // The failed statement rolled back only itself (ON CONFLICT
                // ABORT); release this row's hash and retry in-transaction.
                tracing::warn!(
                    kind = kind.as_str(),
                    id = %id,
                    "restore: content_hash conflicts with an active duplicate; \
                     releasing this row's hash (dedup tracking waived)"
                );
                tx.execute(
                    &format!(
                        "UPDATE {} SET content_hash = NULL WHERE id = ?1 AND project_id = ?2",
                        kind.table()
                    ),
                    params![id, project_id],
                )?;
                tx.execute(&sql, params![id, project_id])?
            }
            Err(e) => return Err(e.into()),
        };
        if affected > 0 {
            Self::reinsert_fts_row(&tx, kind, id, project_id)?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    /// Delete the unified FTS row of a (kind, id) pair, resolving the main
    /// rowid first. No-op when no main row exists. Runs inside the caller's
    /// transaction.
    fn delete_fts_row(
        tx: &rusqlite::Transaction<'_>,
        kind: MemoryKind,
        id: &str,
        project_id: &str,
    ) -> Result<()> {
        let rowid: Option<i64> = tx
            .query_row(
                &format!(
                    "SELECT rowid FROM {} WHERE id = ?1 AND project_id = ?2",
                    kind.table()
                ),
                params![id, project_id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(rid) = rowid {
            tx.execute(
                "DELETE FROM memories_fts WHERE rowid = ?1",
                params![encode_fts_rowid(kind, rid)],
            )?;
        }
        Ok(())
    }

    /// Re-insert the unified FTS row for an existing (kind, id) pair by
    /// reading the main row's FTS columns — the mirror of `delete_fts_row`.
    /// Runs inside the caller's transaction; no-op when the main row is gone.
    fn reinsert_fts_row(
        tx: &rusqlite::Transaction<'_>,
        kind: MemoryKind,
        id: &str,
        project_id: &str,
    ) -> Result<()> {
        let insert = |rid: i64, project: &str, doc: (String, String, String)| -> Result<()> {
            tx.execute(
                crate::storage::schema::UNIFIED_FTS_INSERT,
                params![
                    doc.0,
                    doc.1,
                    doc.2,
                    Self::project_fts_hex(project),
                    kind.as_str(),
                    rid
                ],
            )?;
            Ok(())
        };
        match kind {
            MemoryKind::Episodic => {
                let row = tx
                    .query_row(
                        "SELECT rowid, project_id, summary, content, files_touched, tags \
                         FROM episodic_memories WHERE id = ?1 AND project_id = ?2",
                        params![id, project_id],
                        |r| {
                            Ok((
                                r.get::<_, i64>(0)?,
                                r.get::<_, String>(1)?,
                                unified_doc_episodic(
                                    &r.get::<_, String>(2)?,
                                    &r.get::<_, String>(3)?,
                                    &r.get::<_, String>(4)?,
                                    &r.get::<_, String>(5)?,
                                ),
                            ))
                        },
                    )
                    .optional()?;
                if let Some((rid, project, doc)) = row {
                    insert(encode_fts_rowid(kind, rid), &project, doc)?;
                }
            }
            MemoryKind::Decision => {
                let row = tx
                    .query_row(
                        "SELECT rowid, project_id, title, context, rationale, tradeoffs, tags \
                         FROM decision_memories WHERE id = ?1 AND project_id = ?2",
                        params![id, project_id],
                        |r| {
                            Ok((
                                r.get::<_, i64>(0)?,
                                r.get::<_, String>(1)?,
                                unified_doc_decision(
                                    &r.get::<_, String>(2)?,
                                    &r.get::<_, String>(3)?,
                                    &r.get::<_, String>(4)?,
                                    &r.get::<_, String>(5)?,
                                    &r.get::<_, String>(6)?,
                                ),
                            ))
                        },
                    )
                    .optional()?;
                if let Some((rid, project, doc)) = row {
                    insert(encode_fts_rowid(kind, rid), &project, doc)?;
                }
            }
            MemoryKind::Failure => {
                let row = tx
                    .query_row(
                        "SELECT rowid, project_id, incident, root_cause, fix, prevention, tags \
                         FROM failure_memories WHERE id = ?1 AND project_id = ?2",
                        params![id, project_id],
                        |r| {
                            Ok((
                                r.get::<_, i64>(0)?,
                                r.get::<_, String>(1)?,
                                unified_doc_failure(
                                    &r.get::<_, String>(2)?,
                                    &r.get::<_, String>(3)?,
                                    &r.get::<_, String>(4)?,
                                    &r.get::<_, String>(5)?,
                                    &r.get::<_, String>(6)?,
                                ),
                            ))
                        },
                    )
                    .optional()?;
                if let Some((rid, project, doc)) = row {
                    insert(encode_fts_rowid(kind, rid), &project, doc)?;
                }
            }
            MemoryKind::Procedural => {
                let row = tx
                    .query_row(
                        "SELECT rowid, project_id, workflow_name, steps, related_tools, tags \
                         FROM procedural_memories WHERE id = ?1 AND project_id = ?2",
                        params![id, project_id],
                        |r| {
                            Ok((
                                r.get::<_, i64>(0)?,
                                r.get::<_, String>(1)?,
                                unified_doc_procedural(
                                    &r.get::<_, String>(2)?,
                                    &r.get::<_, String>(3)?,
                                    &r.get::<_, String>(4)?,
                                    &r.get::<_, String>(5)?,
                                ),
                            ))
                        },
                    )
                    .optional()?;
                if let Some((rid, project, doc)) = row {
                    insert(encode_fts_rowid(kind, rid), &project, doc)?;
                }
            }
        }
        Ok(())
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
    /// Returns the ids that were archived. All updates run in one transaction
    /// (including the unified-FTS row deletions — same invariant as `archive`).
    pub fn archive_batch(
        &self,
        kind: MemoryKind,
        project_id: &str,
        tags: &[String],
        before: Option<i64>,
        now: i64,
    ) -> Result<Vec<String>> {
        let matched = self.list_active_candidates(kind, project_id, tags, before)?;
        let mut conn = self.conn()?;
        // Write tx takes the write lock up front (BEGIN IMMEDIATE): busy_timeout
        // applies to contention instead of the deferred-upgrade SQLITE_BUSY path.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for id in &matched {
            tx.execute(
                &format!(
                    "UPDATE {} SET archived_at = ?2 WHERE id = ?1 AND archived_at IS NULL",
                    kind.table()
                ),
                params![id, now],
            )?;
            Self::delete_fts_row(&tx, kind, id, project_id)?;
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
             SUM(adopted) AS adopted, MAX(created_at) AS last_at FROM query_log \
             WHERE project_id = ?1 AND created_at >= ?2 \
             GROUP BY query ORDER BY cnt DESC, last_at DESC LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![project_id, since, limit], |row| {
            Ok(QueryStatRow {
                query: row.get(0)?,
                count: row.get(1)?,
                result_count_avg: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                adopted: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                last_at: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// One aggregate snapshot for `engram stats`: counts per kind/project,
    /// entity/graph totals, reflection proposal states.
    pub fn stats_snapshot(&self) -> Result<StatsSnapshot> {
        // Values that borrow their own pooled connection are collected FIRST:
        // computing them after `let conn` below held two connections at once,
        // and ≥10 concurrent stats calls could exhaust the pool.
        let schema_version = self.user_version()?;
        let fts_tokenizer = self.meta_get("fts_tokenizer")?;
        let orphan_entities = self.gc_orphan_entities(false)? as i64;
        let conn = self.conn()?;
        let mut memories = Vec::new();
        let mut total_active = 0i64;
        for kind in MemoryKind::all() {
            let (active, archived): (i64, i64) = conn.query_row(
                &format!(
                    "SELECT \
                        COALESCE(SUM(CASE WHEN archived_at IS NULL THEN 1 ELSE 0 END), 0), \
                        COALESCE(SUM(CASE WHEN archived_at IS NOT NULL THEN 1 ELSE 0 END), 0) \
                     FROM {}",
                    kind.table()
                ),
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            total_active += active;
            memories.push((kind.as_str().to_string(), active, archived));
        }

        let mut stmt = conn.prepare(
            "SELECT project_id, COUNT(*) FROM (
                SELECT project_id, archived_at FROM episodic_memories
                UNION ALL SELECT project_id, archived_at FROM decision_memories
                UNION ALL SELECT project_id, archived_at FROM failure_memories
                UNION ALL SELECT project_id, archived_at FROM procedural_memories
             ) WHERE archived_at IS NULL GROUP BY project_id ORDER BY COUNT(*) DESC",
        )?;
        let projects = stmt
            .query_map([], |r| {
                Ok(ProjectCount {
                    project_id: r.get(0)?,
                    active: r.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let entities_total: i64 =
            conn.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))?;
        let (pending, confirmed, rejected): (i64, i64, i64) = conn.query_row(
            "SELECT \
                COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END), 0), \
                COALESCE(SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END), 0) \
             FROM reflection_suggestions",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;

        Ok(StatsSnapshot {
            schema_version,
            fts_tokenizer,
            memories,
            total_active,
            projects,
            entities_total,
            orphan_entities,
            reflections: ReflectionCounts {
                pending,
                confirmed,
                rejected,
            },
        })
    }

    /// Record that `memory_id` (returned by a recent search) was actually
    /// fetched — the adoption signal. Bumps `adopted` on the most recent
    /// query_log row (within `window_seconds`) whose result set contains the
    /// id. Quoted-substring match against the JSON array is exact: memory ids
    /// are fixed-length UUIDs, so `"id"` cannot substring-match another id.
    ///
    /// Deduplicated per memory within `window_seconds`: before the dedup, one
    /// logical adoption ("search → read → mark useful") wrote TWO `adoptions`
    /// rows (get_memory and mark_relevance each inserted one), and a client
    /// loop calling get_memory could inflate `engram tune`'s adoption rate
    /// above 1.0. Repeated signals inside the window are ignored. Both writes
    /// run in one transaction so query_log and adoptions cannot disagree.
    pub fn record_adoption(
        &self,
        project_id: &str,
        memory_id: &str,
        memory_type: &str,
        now: i64,
        window_seconds: i64,
    ) -> Result<usize> {
        let mut conn = self.conn()?;
        // Write tx takes the write lock up front (BEGIN IMMEDIATE): busy_timeout
        // applies to contention instead of the deferred-upgrade SQLITE_BUSY path.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let needle = format!("\"{memory_id}\"");
        let since = now.saturating_sub(window_seconds);
        // Cooldown: an adoption row for this memory inside the window means
        // the signal was already recorded — skip both writes.
        let already: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM adoptions \
             WHERE project_id = ?1 AND memory_id = ?2 AND created_at >= ?3)",
            params![project_id, memory_id, since],
            |row| row.get(0),
        )?;
        let mut updated = 0;
        if !already {
            updated = tx.execute(
                "UPDATE query_log SET adopted = adopted + 1 \
                 WHERE id = ( \
                     SELECT id FROM query_log \
                      WHERE project_id = ?1 AND created_at >= ?2 \
                        AND result_ids LIKE '%' || ?3 || '%' \
                      ORDER BY created_at DESC LIMIT 1 \
                 )",
                params![project_id, since, needle],
            )?;
            tx.execute(
                "INSERT INTO adoptions (project_id, memory_id, memory_type, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![project_id, memory_id, memory_type, now],
            )?;
        }
        tx.commit()?;
        Ok(updated)
    }

    /// Per-memory adoption evidence over a window, most evidence first: the
    /// raw event count plus the number of DISTINCT UTC days the memory was
    /// adopted on. `engram tune` rates adoption per calendar day (both sides
    /// of the rate are deduplicated the same way), so one day with many
    /// reads still counts as one adoption day. Grouped by memory_id only —
    /// ids are UUIDs, unique across kinds.
    pub fn adoption_day_counts(
        &self,
        project_id: &str,
        since: i64,
    ) -> Result<Vec<(String, i64, i64)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT memory_id, COUNT(*) AS cnt, \
                    COUNT(DISTINCT created_at / 86400) AS days FROM adoptions \
             WHERE project_id = ?1 AND created_at >= ?2 \
             GROUP BY memory_id \
             ORDER BY cnt DESC, memory_id",
        )?;
        let rows = stmt.query_map(params![project_id, since], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// `(created_at, result_ids)` pairs from query_log over a window — the
    /// impression side of the tuning signal (how often each memory was
    /// SERVED to the agent, regardless of whether it was read). The
    /// timestamp lets the caller bucket impressions per UTC calendar day,
    /// mirroring the day-deduplicated adoption numerator.
    pub fn recent_result_sets(&self, project_id: &str, since: i64) -> Result<Vec<(i64, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT created_at, result_ids FROM query_log \
             WHERE project_id = ?1 AND created_at >= ?2",
        )?;
        let rows = stmt.query_map(params![project_id, since], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Record one explicit `mark_relevance` judgment ('useful'/'irrelevant').
    /// Separate from `record_adoption` on purpose: a passive read is
    /// behavioral evidence, this is a judgment — conflating the two made the
    /// recommended search → get_memory → mark_relevance flow trip the
    /// adoption cooldown, so explicit feedback never took effect.
    pub fn record_feedback(
        &self,
        project_id: &str,
        memory_id: &str,
        memory_type: &str,
        feedback: &str,
        now: i64,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO memory_feedback \
             (project_id, memory_id, memory_type, feedback, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![project_id, memory_id, memory_type, feedback, now],
        )?;
        Ok(())
    }

    /// Whether ANY explicit feedback for this memory was recorded within
    /// `window_seconds` — deliberately direction-agnostic: 'useful' and
    /// 'irrelevant' share one cooldown, so a client loop can neither pump
    /// importance up nor grind it down.
    pub fn has_recent_feedback(
        &self,
        project_id: &str,
        memory_id: &str,
        now: i64,
        window_seconds: i64,
    ) -> Result<bool> {
        let conn = self.conn()?;
        let since = now.saturating_sub(window_seconds);
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM memory_feedback \
             WHERE project_id = ?1 AND memory_id = ?2 AND created_at >= ?3)",
            params![project_id, memory_id, since],
            |row| row.get(0),
        )?;
        Ok(exists)
    }

    /// Whether a 'useful' judgment exists for this memory since `since`. The
    /// tuner's demote guard: an explicitly endorsed memory must not be
    /// auto-demoted for being rarely read.
    pub fn has_useful_feedback_since(
        &self,
        project_id: &str,
        memory_id: &str,
        since: i64,
    ) -> Result<bool> {
        let conn = self.conn()?;
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM memory_feedback \
             WHERE project_id = ?1 AND memory_id = ?2 \
               AND feedback = 'useful' AND created_at >= ?3)",
            params![project_id, memory_id, since],
            |row| row.get(0),
        )?;
        Ok(exists)
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
            "SELECT id, project_id, title, context, rationale, tradeoffs, related_files, tags, importance, superseded_by, created_at, updated_at
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
                importance: row.get(8)?,
                superseded_by: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Mark decision `old_id` as superseded by `new_id`: the old decision
    /// stays searchable as history but is flagged and downranked in retrieval.
    /// Fails loudly when the old id does not exist in this project. An already
    /// superseded decision is re-pointed at the newest one (chain flattening —
    /// "current" is always one hop from any replaced decision).
    ///
    /// Single targeted UPDATE (no full-row rewrite): the previous version did
    /// get_decision → mutate → update_decision, which rebuilds the FTS row and
    /// graph edges AND loses a concurrent update that landed in between.
    /// `superseded_by` is not an FTS-indexed column, so no FTS rebuild needed.
    pub fn supersede_decision(
        &self,
        project_id: &str,
        old_id: &str,
        new_id: &str,
        now: i64,
    ) -> Result<()> {
        let conn = self.conn()?;
        let affected = conn.execute(
            "UPDATE decision_memories SET superseded_by = ?3, updated_at = ?4 \
             WHERE id = ?1 AND project_id = ?2",
            params![old_id, project_id, new_id, now],
        )?;
        if affected == 0 {
            anyhow::bail!("cannot supersede: no decision {old_id} in project {project_id}");
        }
        Ok(())
    }

    /// Create a decision AND point its predecessor at it in ONE transaction.
    ///
    /// The two-step version (create, then supersede) was not atomic: a failed
    /// supersede left the new decision committed but the chain broken, and a
    /// client retry hit the write-dedup guard which returned early WITHOUT
    /// ever performing the supersede. Here the old id is validated inside the
    /// transaction before any write, and both writes commit or roll back
    /// together. Mirrors the macro-generated create (insert + FTS dual-write +
    /// entity link) — keep the two in sync.
    ///
    /// `dedup = true` (write_dedup config) writes the row's content_hash and
    /// lets the partial unique index enforce dedup: on a collision the
    /// existing ACTIVE decision's id is returned (`Ok(Some(id))`) and nothing
    /// is written — the caller applies the missing supersede to that existing
    /// row (retry-after-failure semantics) and answers "duplicate". With
    /// `dedup = false` the row carries no hash and this always returns
    /// `Ok(None)`.
    pub fn create_decision_with_supersede(
        &self,
        mem: &DecisionMemory,
        old_id: &str,
        dedup: bool,
    ) -> Result<Option<String>> {
        let mem = mem.clone().sanitized();
        let mut conn = self.conn()?;
        // Write tx takes the write lock up front (BEGIN IMMEDIATE): busy_timeout
        // applies to contention instead of the deferred-upgrade SQLITE_BUSY path.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if old_id == mem.id {
            anyhow::bail!("cannot supersede: decision {old_id} cannot supersede itself");
        }
        let old_exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM decision_memories WHERE id = ?1 AND project_id = ?2)",
            params![old_id, mem.project_id],
            |row| row.get(0),
        )?;
        if !old_exists {
            anyhow::bail!(
                "cannot supersede: no decision {old_id} in project {}",
                mem.project_id
            );
        }
        // content_hash: the dedup-enforcing write (see create_*_dedup) or NULL
        // when write_dedup is off — NULL rows never participate in the index.
        let hash = if dedup {
            Some(content_hash(&dedup_text_decision(
                &mem.title,
                &mem.context,
                &mem.rationale,
                &mem.tradeoffs,
            )))
        } else {
            None
        };
        let insert = tx.execute(
            "INSERT INTO decision_memories (id, project_id, title, context, rationale, tradeoffs, related_files, tags, importance, superseded_by, created_at, updated_at, content_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.importance,
                mem.superseded_by,
                mem.created_at, mem.updated_at,
                hash,
            ],
        );
        if let Err(e) = insert {
            if dedup && is_unique_violation(&e) {
                // The index rejected OUR row: an ACTIVE decision already
                // carries this content_hash. Return it for the duplicate
                // response; the (empty) transaction is discarded on drop.
                let existing: Option<String> = tx
                    .query_row(
                        "SELECT id FROM decision_memories \
                         WHERE project_id = ?1 AND content_hash = ?2 AND archived_at IS NULL \
                         LIMIT 1",
                        params![&mem.project_id, &hash],
                        |r| r.get(0),
                    )
                    .optional()?;
                if let Some(id) = existing {
                    return Ok(Some(id));
                }
            }
            return Err(e.into());
        }
        // Unified-FTS dual-write, identical to the macro-generated create.
        let fts_rowid: i64 = encode_fts_rowid(MemoryKind::Decision, tx.last_insert_rowid());
        let (u_title, u_body, u_tags) = unified_doc_decision(
            &mem.title,
            &mem.context,
            &mem.rationale,
            &mem.tradeoffs,
            &serde_json::to_string(&mem.tags)?,
        );
        tx.execute(
            crate::storage::schema::UNIFIED_FTS_INSERT,
            params![
                u_title,
                u_body,
                u_tags,
                Self::project_fts_hex(&mem.project_id),
                MemoryKind::Decision.as_str(),
                fts_rowid,
            ],
        )?;
        if !mem.related_files.is_empty() {
            Self::ensure_linked_entities(
                &tx,
                &mem.project_id,
                &mem.id,
                "File",
                &mem.related_files,
                "References",
                mem.created_at,
            )?;
        }
        let affected = tx.execute(
            "UPDATE decision_memories SET superseded_by = ?3, updated_at = ?4 \
             WHERE id = ?1 AND project_id = ?2",
            params![old_id, mem.project_id, mem.id, mem.updated_at],
        )?;
        if affected == 0 {
            // Impossible after the existence probe above unless the row was
            // deleted mid-transaction; surface it rather than commit a broken
            // chain.
            anyhow::bail!("cannot supersede: decision {old_id} vanished mid-transaction");
        }
        tx.commit()?;
        Ok(None)
    }

    /// List a project's procedural memories, highest importance first, then
    /// most recent. Powers the session brief: procedures a team actually
    /// follows are the ones importance feedback (mark_relevance/tune) raised.
    pub fn list_top_procedural(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<ProceduralMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, project_id, workflow_name, steps, related_tools, tags, importance, created_at, updated_at
             FROM procedural_memories
             WHERE project_id = ?1 AND archived_at IS NULL
             ORDER BY importance DESC, created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![project_id, limit], |row| {
            Ok(ProceduralMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                workflow_name: row.get(2)?,
                steps: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                related_tools: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                tags: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                importance: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
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
        let mut conn = self.conn()?;
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
        // IMMEDIATE: the promote INSERT + status flip hold the write lock from
        // the start (same discipline as the other write transactions).
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

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
                 (id, project_id, workflow_name, steps, related_tools, tags, importance, created_at, updated_at) \
                 VALUES (?1,?2,?3,?4,?5,?6,0.5,?7,?7)",
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
            // Mirror the standard create path's unified-FTS dual-write: same
            // doc projection AND encoded rowid, so this hand-rolled path
            // stays consistent with the macro-generated one.
            let fts_rowid: i64 = encode_fts_rowid(MemoryKind::Procedural, tx.last_insert_rowid());
            let (u_title, u_body, u_tags) = unified_doc_procedural(
                &draft.suggested_workflow_name,
                &serde_json::to_string(&draft.suggested_steps)?,
                &serde_json::to_string(&Vec::<String>::new())?,
                &serde_json::to_string(&draft.suggested_tags)?,
            );
            tx.execute(
                crate::storage::schema::UNIFIED_FTS_INSERT,
                params![
                    u_title,
                    u_body,
                    u_tags,
                    Self::project_fts_hex(&draft.project_id),
                    MemoryKind::Procedural.as_str(),
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
                let mut conn = self.conn()?;
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
                        // Unified-FTS row addressed by the encoded rowid.
                        // Normally already gone (archive deleted it); kept
                        // as a defensive purge for indexes from older
                        // binaries or interrupted migrations.
                        tx.execute(
                            "DELETE FROM memories_fts WHERE rowid = ?1",
                            params![encode_fts_rowid(kind, *fts_rowid)],
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

    /// Active episodic memories tagged `session-import` — the raw hook dumps
    /// awaiting distillation (prompt `engram.distill`). Powers the session
    /// brief's distillation hint and `engram maintain`'s aging report.
    /// The LIKE match on the fixed tag inside the JSON array is exact enough
    /// (the tag has no delimiters that could substring-match another tag).
    pub fn count_session_imports(&self, project_id: &str) -> Result<i64> {
        let conn = self.conn()?;
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM episodic_memories \
             WHERE project_id = ?1 AND archived_at IS NULL \
               AND tags LIKE '%\"session-import\"%'",
            params![project_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// Lifecycle for raw session imports: session-import memories whose
    /// `updated_at` is older than `older_than_seconds` get their importance
    /// lowered to `floor` (default path: 0.5 → 0.2) — raw prompt dumps age
    /// out of prime ranking unless distilled into typed memories, which
    /// become the durable knowledge. `apply=false` only reports candidates.
    /// Returns `(id, current_importance)` pairs.
    pub fn downgrade_stale_session_imports(
        &self,
        project_id: &str,
        older_than_seconds: i64,
        floor: f32,
        apply: bool,
        now: i64,
    ) -> Result<Vec<(String, f32)>> {
        let cutoff = now.checked_sub(older_than_seconds).unwrap_or(0);
        let mut conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, importance FROM episodic_memories \
             WHERE project_id = ?1 AND archived_at IS NULL \
               AND tags LIKE '%\"session-import\"%' \
               AND updated_at < ?2 AND importance > ?3",
        )?;
        let rows = stmt.query_map(params![project_id, cutoff, floor], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?;
        let mut candidates = Vec::new();
        for r in rows {
            candidates.push(r?);
        }
        drop(stmt);
        if apply && !candidates.is_empty() {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (id, _) in &candidates {
                tx.execute(
                    "UPDATE episodic_memories SET importance = ?3, updated_at = ?4 \
                     WHERE id = ?1 AND project_id = ?2",
                    params![id, project_id, floor, now],
                )?;
            }
            tx.commit()?;
        }
        Ok(candidates)
    }

    /// Most recent ACTIVE episodic memory for a (project, session) pair —
    /// used by session-import to make repeated hook firings idempotent.
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

    /// Truncate the WAL back into the main database file. Safe to run while
    /// other connections are idle; a busy DB simply leaves the WAL in place.
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

    pub fn connection(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.conn()
    }
}

/// Check if a character is a CJK (Chinese/Japanese/Korean) ideograph.
pub(crate) fn is_cjk_character(ch: char) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::schema::CURRENT_SCHEMA_VERSION;

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
            superseded_by: None,
            created_at: now(),
            updated_at: now(),
            importance: 0.5,
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
            importance: 0.5,
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
            superseded_by: None,
            created_at: now(),
            updated_at: now(),
            importance: 0.5,
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
        // The unified FTS rowid must decode back to the main-table row:
        // deletes/updates address FTS rows by rowid (O(log n)); a misaligned
        // encoding would silently strand or destroy rows.
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("r1", "rowid alignment marker"))
            .unwrap();
        {
            let conn = repo.connection().unwrap();
            let (main_rowid, fts_rowid): (i64, i64) = conn
                .query_row(
                    "SELECT m.rowid, f.rowid FROM episodic_memories m \
                     JOIN memories_fts f ON f.rowid = m.rowid * 4 + 0 \
                     WHERE m.id = 'r1'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(main_rowid * 4, fts_rowid);
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
                    "SELECT COUNT(*) FROM memories_fts f \
                     JOIN episodic_memories m ON f.rowid = m.rowid * 4 + 0 WHERE m.id = 'r1'",
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

        // Delete removes the FTS row entirely (by encoded rowid).
        repo.delete_episodic("r1", "p").unwrap();
        {
            let conn = repo.connection().unwrap();
            let leftover: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories_fts f \
                     JOIN episodic_memories m ON f.rowid = m.rowid * 4 + 0 WHERE m.id = 'r1'",
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
            // Install a genuine PRE-v2 shadow table (full-copy, memory_id
            // column, raw preprocessing) as an old binary would have left it.
            conn.execute_batch(
                "DROP TABLE IF EXISTS memories_fts; \
                 DROP TABLE IF EXISTS episodic_memories_fts; \
                 CREATE VIRTUAL TABLE episodic_memories_fts USING fts5( \
                     memory_id UNINDEXED, summary, content, files_touched, tags, \
                     tokenize='porter unicode61');",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched, tags) \
                 VALUES ('m1', '用Rust写mixed text', '用Rust写mixed text', '[]', '[]')",
                [],
            )
            .unwrap();
        }
        // With the v1-era index installed the unified table does not exist
        // yet — search cannot even run until the migration chain builds it.
        assert!(
            repo.search_episodic("rust", "p", 10).is_err(),
            "unified FTS table must not exist before the chain reaches v9"
        );

        repo.run_migrations().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
        // After the full chain (v2 preprocessing rebuilds + v9 unified
        // conversion) the latin term matches the mixed-script text.
        assert!(
            !repo.search_episodic("rust", "p", 10).unwrap().is_empty(),
            "migration chain must apply boundary-aware preprocessing"
        );
        let conn = repo.connection().unwrap();
        let aligned: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodic_memories m \
                 JOIN memories_fts f ON f.rowid = m.rowid * 4 + 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(aligned, 1, "FTS rowid must decode to main-table rowid");
        // The legacy per-type tables are gone.
        let legacy: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' \
                 AND name IN ('episodic_memories_fts','decision_memories_fts',\
                              'failure_memories_fts','procedural_memories_fts')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(legacy, 0, "v9 must drop the per-type FTS tables");
    }

    #[test]
    fn archive_deletes_and_restore_reinserts_the_fts_row() {
        // Unified-table invariant: one FTS row per ACTIVE main row. Archive
        // removing the row (instead of relying on a search-time filter) is
        // what keeps `fts_drift` meaningful and stale docs out of the index.
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("ar1", "archive fts marker"))
            .unwrap();
        assert_eq!(repo.fts_drift().unwrap(), 0);

        repo.archive(MemoryKind::Episodic, "ar1", "p", 200).unwrap();
        assert_eq!(
            repo.fts_drift().unwrap(),
            0,
            "archived row must leave the index"
        );
        assert!(
            repo.search_episodic("marker", "p", 10).unwrap().is_empty(),
            "archived memory must not be searchable"
        );

        repo.restore(MemoryKind::Episodic, "ar1", "p").unwrap();
        assert_eq!(
            repo.fts_drift().unwrap(),
            0,
            "restored row must re-enter the index"
        );
        assert!(
            !repo.search_episodic("marker", "p", 10).unwrap().is_empty(),
            "restored memory must be searchable again"
        );
    }

    #[test]
    fn migrate_v9_converts_v8_db_to_unified_fts() {
        // Simulate a v8 DB: main rows + per-type FTS tables, no unified table.
        let repo = setup_repo();
        repo.create_episodic(&episodic_fixture("v9e", "v9 conversion marker"))
            .unwrap();
        {
            let conn = repo.connection().unwrap();
            conn.execute_batch(
                "DROP TABLE memories_fts; \
                 CREATE VIRTUAL TABLE episodic_memories_fts USING fts5( \
                     summary, content, files_touched, tags, \
                     tokenize='porter unicode61', content='', contentless_delete=1); \
                 PRAGMA user_version = 8;",
            )
            .unwrap();
            // A stale legacy FTS row for an ARCHIVED memory — v9's backfill
            // must not carry it into the unified index.
            conn.execute_batch(
                "INSERT INTO episodic_memories (id, project_id, session_id, summary, content, \
                 files_touched, related_commits, importance, tags, created_at, updated_at, archived_at) \
                 VALUES ('v9arch', 'p', 's', 'stale archived marker', 'x', '[]', '[]', 0.5, '[]', 1, 1, 5); \
                 INSERT INTO episodic_memories_fts (summary, content, files_touched, tags, rowid) \
                 VALUES ('stale archived marker', 'x', '[]', '[]', \
                 (SELECT rowid FROM episodic_memories WHERE id = 'v9arch'));",
            )
            .unwrap();
        }

        repo.run_migrations().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);
        assert!(
            !repo
                .search_episodic("conversion", "p", 10)
                .unwrap()
                .is_empty(),
            "v9 backfill must index active rows"
        );
        assert!(
            repo.search_episodic("stale", "p", 10).unwrap().is_empty(),
            "v9 backfill must skip archived rows even if the legacy index had them"
        );
        assert_eq!(repo.fts_drift().unwrap(), 0);
        let conn = repo.connection().unwrap();
        let legacy: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'episodic_memories_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(legacy, 0, "v9 must drop the legacy FTS tables");
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
        // NOT parsed as FTS5 AND/OR/NOT. ("a" is a stopword and dropped —
        // stopwords never reach the MATCH expression.)
        assert_eq!(R::sanitize_fts_query("a AND b", false), "\"AND\" OR \"b\"");
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
    #[cfg(not(feature = "jieba"))]
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
    #[cfg(feature = "jieba")]
    fn jieba_indexes_and_queries_by_word() {
        // With the jieba feature, `认证` is ONE token — the query must match
        // the word exactly (char-split builds match "both chars anywhere").
        use MemoryRepository as R;
        assert_eq!(
            R::preprocess_cjk("用Rust重写认证模块"),
            "用 Rust 重写 认证 模块"
        );
        assert_eq!(R::preprocess_cjk("部署策略"), "部署 策略");

        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mem = crate::models::EpisodicMemory {
            id: "j1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "重写认证模块".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec!["部署策略".into()],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mem).unwrap();
        assert!(!repo.search_episodic("认证", "p", 10).unwrap().is_empty());
        assert!(!repo.search_episodic("重写", "p", 10).unwrap().is_empty());
        // Word token from a tag is searchable.
        assert!(!repo.search_episodic("部署", "p", 10).unwrap().is_empty());
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
                "SELECT COUNT(*) FROM memories_fts f \
                 WHERE NOT EXISTS (SELECT 1 FROM episodic_memories m \
                                   WHERE f.rowid = m.rowid * 4 + 0)",
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
            // Corrupt the index both ways: an orphan row (an encoded rowid
            // with no main-table row) + a missing row.
            conn.execute(
                "INSERT INTO memories_fts (title, body, tags, project, kind, rowid)
                 VALUES ('orphan', 'orphan', '[]', '00', 'episodic', 99999 * 4)",
                [],
            )
            .unwrap();
            conn.execute(
                "DELETE FROM memories_fts WHERE rowid = \
                 (SELECT rowid * 4 + 0 FROM episodic_memories WHERE id = 'e2')",
                [],
            )
            .unwrap();
        }

        let written = repo.rebuild_fts().unwrap();
        assert_eq!(written, 2, "one FTS row per active main-table row");
        assert_eq!(
            repo.fts_drift().unwrap(),
            0,
            "index must align after rebuild"
        );

        let orphan: i64 = {
            let conn = repo.connection().unwrap();
            conn.query_row(
                "SELECT COUNT(*) FROM memories_fts f \
                 WHERE NOT EXISTS (SELECT 1 FROM episodic_memories m \
                                   WHERE f.rowid = m.rowid * 4 + 0)",
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
            let (rid, project): (i64, String) = conn
                .query_row(
                    "SELECT rowid, project_id FROM episodic_memories WHERE id = 'c1'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            let hex = MemoryRepository::project_fts_hex(&project);
            conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?1",
                params![rid * 4],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO memories_fts (title, body, tags, project, kind, rowid) \
                 VALUES ('修复认证模块', '修复认证模块 body', '[]', ?1, 'episodic', ?2)",
                params![hex, rid * 4],
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

    // ─── content_hash dedup (v10) Tests ────────────────────────────

    fn dedup_fixture(id: &str, summary: &str) -> EpisodicMemory {
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

    fn stored_hash(repo: &MemoryRepository, table: &str, id: &str) -> Option<String> {
        repo.connection()
            .unwrap()
            .query_row(
                &format!("SELECT content_hash FROM {table} WHERE id = ?1"),
                params![id],
                |r| r.get::<_, Option<String>>(0),
            )
            .unwrap()
    }

    #[test]
    fn dedup_create_returns_existing_id_and_writes_hash() {
        let repo = setup_repo();
        assert_eq!(
            repo.create_episodic_dedup(&dedup_fixture("a1", "dup needle"))
                .unwrap(),
            None,
            "first create wins"
        );
        let existing = repo
            .create_episodic_dedup(&dedup_fixture("a2", "dup needle"))
            .unwrap();
        assert_eq!(
            existing.as_deref(),
            Some("a1"),
            "collision returns the existing id"
        );
        // Nothing was written for the loser.
        assert!(repo.get_episodic("a2", "p").unwrap().is_none());
        // The winner is hash-tracked.
        assert!(stored_hash(&repo, "episodic_memories", "a1").is_some());
        // Search still finds exactly one row (FTS dual-write intact).
        assert_eq!(repo.search_episodic("needle", "p", 10).unwrap().len(), 1);
    }

    #[test]
    fn plain_create_writes_null_hash_and_allows_duplicates() {
        // write_dedup=false semantics: no hash → the partial unique index
        // ignores the rows → duplicates are created freely.
        let repo = setup_repo();
        repo.create_episodic(&dedup_fixture("a1", "same text"))
            .unwrap();
        repo.create_episodic(&dedup_fixture("a2", "same text"))
            .unwrap();
        assert_eq!(repo.search_episodic("same", "p", 10).unwrap().len(), 2);
        assert_eq!(stored_hash(&repo, "episodic_memories", "a1"), None);
        assert_eq!(stored_hash(&repo, "episodic_memories", "a2"), None);
    }

    #[test]
    fn dedup_create_is_project_scoped() {
        let repo = setup_repo();
        let mut a = dedup_fixture("a1", "cross project");
        a.project_id = "p1".into();
        let mut b = dedup_fixture("a2", "cross project");
        b.project_id = "p2".into();
        repo.create_episodic_dedup(&a).unwrap();
        repo.create_episodic_dedup(&b).unwrap();
        assert!(repo.get_episodic("a1", "p1").unwrap().is_some());
        assert!(repo.get_episodic("a2", "p2").unwrap().is_some());
    }

    #[test]
    fn dedup_create_procedural_hashes_stored_steps_json() {
        let repo = setup_repo();
        let mk = |id: &str| ProceduralMemory {
            id: id.into(),
            project_id: "p".into(),
            workflow_name: "release".into(),
            steps: vec!["test".into(), "deploy".into()],
            related_tools: vec![],
            tags: vec![],
            importance: 0.5,
            created_at: 1,
            updated_at: 1,
        };
        assert_eq!(repo.create_procedural_dedup(&mk("p1")).unwrap(), None);
        assert_eq!(
            repo.create_procedural_dedup(&mk("p2")).unwrap().as_deref(),
            Some("p1"),
            "identical name+steps is a duplicate"
        );
        assert!(stored_hash(&repo, "procedural_memories", "p1").is_some());
    }

    #[test]
    fn dedup_create_races_have_a_single_winner() {
        // The regression this whole mechanism fixes: N concurrent identical
        // creates used to ALL pass the find_exact_duplicate check and all
        // write. Now the unique index picks exactly one winner; the rest get
        // the winner's id back as a duplicate.
        use std::sync::Arc;
        let tmp = tempfile::TempDir::new().unwrap();
        let repo = Arc::new(MemoryRepository::new(&tmp.path().join("race.db")).unwrap());
        repo.initialize_schema().unwrap();

        let mut handles = vec![];
        for i in 0..8 {
            let repo = Arc::clone(&repo);
            handles.push(std::thread::spawn(move || {
                let mut m = dedup_fixture("loser", "race condition probe");
                m.id = format!("racer-{i}");
                m.created_at = 100 + i;
                m.updated_at = m.created_at;
                repo.create_episodic_dedup(&m).unwrap()
            }));
        }
        let mut winners = 0;
        let mut loser_ids = Vec::new();
        for h in handles {
            match h.join().unwrap() {
                None => winners += 1,
                Some(existing) => loser_ids.push(existing),
            }
        }
        assert_eq!(winners, 1, "exactly one create must win the race");
        assert_eq!(loser_ids.len(), 7);
        let winner = loser_ids[0].clone();
        assert!(loser_ids.iter().all(|id| *id == winner));
        // Exactly one row survived.
        assert_eq!(
            repo.search_episodic("condition probe", "p", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn restore_releases_conflicting_hash_and_still_restores() {
        // Boundary (2a): restore must WIN over dedup bookkeeping. A is
        // archived while holding a hash; B (same text) was created after —
        // restoring A would collide with B, so A's hash is released (NULL)
        // and the restore succeeds.
        let repo = setup_repo();
        repo.create_episodic_dedup(&dedup_fixture("a1", "restore clash"))
            .unwrap();
        repo.archive(MemoryKind::Episodic, "a1", "p", 200).unwrap();
        // B can be created while A is archived (partial index = active rows only).
        repo.create_episodic_dedup(&dedup_fixture("a2", "restore clash"))
            .unwrap();

        assert!(repo.restore(MemoryKind::Episodic, "a1", "p").unwrap());
        // Both rows are active; A lost its hash, B kept its own hash.
        assert!(repo.get_active_episodic("a1", "p").unwrap().is_some());
        assert!(repo.get_active_episodic("a2", "p").unwrap().is_some());
        assert_eq!(stored_hash(&repo, "episodic_memories", "a1"), None);
        assert!(stored_hash(&repo, "episodic_memories", "a2").is_some());
        // FTS row re-inserted for the restored memory.
        assert_eq!(repo.search_episodic("clash", "p", 10).unwrap().len(), 2);
    }

    #[test]
    fn restore_without_conflict_keeps_hash() {
        let repo = setup_repo();
        repo.create_episodic_dedup(&dedup_fixture("a1", "plain restore"))
            .unwrap();
        repo.archive(MemoryKind::Episodic, "a1", "p", 200).unwrap();
        assert!(repo.restore(MemoryKind::Episodic, "a1", "p").unwrap());
        assert!(stored_hash(&repo, "episodic_memories", "a1").is_some());
    }

    #[test]
    fn update_collision_errors_naming_the_conflict() {
        // Boundary (2b): updates are explicit edits — colliding with another
        // tracked ACTIVE row must fail loudly naming it, never merge silently.
        let repo = setup_repo();
        repo.create_episodic_dedup(&dedup_fixture("a1", "original alpha"))
            .unwrap();
        repo.create_episodic_dedup(&dedup_fixture("a2", "target bravo"))
            .unwrap();

        let mut edited = dedup_fixture("a1", "target bravo");
        edited.updated_at = 300;
        let err = repo.update_episodic(&edited).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("a2") && msg.contains("identical"),
            "error must name the conflicting memory, got: {msg}"
        );
        // a1's text was NOT changed.
        assert_eq!(
            repo.get_episodic("a1", "p").unwrap().unwrap().summary,
            "original alpha"
        );
    }

    #[test]
    fn update_refreshes_tracked_hash_but_preserves_null() {
        // Tracked rows keep a fresh hash after edits; untracked (write_dedup
        // off) rows never GAIN one — otherwise editing one of two legacy
        // duplicates would arm a collision for the other's next update.
        let repo = setup_repo();
        repo.create_episodic(&dedup_fixture("u1", "untracked one"))
            .unwrap();
        repo.create_episodic(&dedup_fixture("u2", "untracked one"))
            .unwrap();
        let mut e1 = dedup_fixture("u1", "untracked one");
        e1.updated_at = 300;
        repo.update_episodic(&e1).unwrap();
        assert_eq!(stored_hash(&repo, "episodic_memories", "u1"), None);

        repo.create_episodic_dedup(&dedup_fixture("t1", "tracked one"))
            .unwrap();
        let mut e2 = dedup_fixture("t1", "tracked rewritten");
        e2.updated_at = 300;
        repo.update_episodic(&e2).unwrap();
        assert_eq!(
            stored_hash(&repo, "episodic_memories", "t1").as_deref(),
            Some(
                content_hash(&dedup_text_episodic(
                    "tracked rewritten",
                    "tracked rewritten body"
                ))
                .as_str()
            ),
        );
    }

    #[test]
    fn update_archived_memory_is_rejected_with_actionable_error() {
        // Tombstone semantics: a forgotten memory cannot be modified — the
        // error must say so (not a bogus "no row").
        let repo = setup_repo();
        repo.create_episodic(&dedup_fixture("a1", "archived text"))
            .unwrap();
        repo.archive(MemoryKind::Episodic, "a1", "p", 200).unwrap();
        let mut e = dedup_fixture("a1", "archived text");
        e.updated_at = 300;
        let err = repo.update_episodic(&e).unwrap_err();
        assert!(
            err.to_string().contains("archived"),
            "error must say the memory is archived, got: {err}"
        );
        // The row is untouched.
        assert!(repo.get_episodic("a1", "p").unwrap().is_some());
    }

    #[test]
    fn delete_archived_memory_returns_false_and_keeps_row() {
        // Tombstone semantics: hard delete must not silently remove a
        // forgotten memory (gc_archived is the only legitimate reaper).
        let repo = setup_repo();
        repo.create_episodic(&dedup_fixture("a1", "protected"))
            .unwrap();
        repo.archive(MemoryKind::Episodic, "a1", "p", 200).unwrap();
        assert!(!repo.delete_episodic("a1", "p").unwrap());
        assert!(repo.get_episodic("a1", "p").unwrap().is_some());
        // Active rows delete as before.
        repo.restore(MemoryKind::Episodic, "a1", "p").unwrap();
        assert!(repo.delete_episodic("a1", "p").unwrap());
    }

    #[test]
    fn migrate_v10_backfills_active_rows_and_skips_archived_and_duplicates() {
        // Boundary (2c): the backfill hashes ACTIVE rows; archived rows stay
        // NULL; pre-existing duplicates keep only the EARLIEST row tracked so
        // creating the unique index cannot fail.
        let repo = setup_repo();
        repo.create_episodic(&dedup_fixture("m1", "backfill alpha"))
            .unwrap();
        repo.create_episodic(&dedup_fixture("m2", "backfill alpha"))
            .unwrap(); // duplicate
        repo.create_episodic(&dedup_fixture("m3", "backfill bravo"))
            .unwrap();
        repo.archive(MemoryKind::Episodic, "m3", "p", 500).unwrap();

        // Rewind to v9 and let the chain re-run v10.
        repo.set_user_version(9).unwrap();
        repo.run_migrations().unwrap();
        assert_eq!(repo.user_version().unwrap(), CURRENT_SCHEMA_VERSION);

        // Earliest active duplicate is tracked; the later one stays NULL.
        assert!(stored_hash(&repo, "episodic_memories", "m1").is_some());
        assert_eq!(stored_hash(&repo, "episodic_memories", "m2"), None);
        // Archived row stays NULL.
        assert_eq!(stored_hash(&repo, "episodic_memories", "m3"), None);

        // The dedup index now enforces from the backfilled state.
        assert_eq!(
            repo.create_episodic_dedup(&dedup_fixture("m4", "backfill alpha"))
                .unwrap()
                .as_deref(),
            Some("m1")
        );
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
            superseded_by: None,
            created_at: 1,
            updated_at: 1,
            importance: 0.5,
        })
        .unwrap();

        assert_eq!(
            repo.list_projects().unwrap(),
            vec!["other".to_string(), "p".to_string()]
        );
    }
}
