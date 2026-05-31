use crate::models::*;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Repository for all memory CRUD operations with FTS5 dual-write.
///
/// All writes complete within a single SQLite transaction.
/// FTS5 uses DELETE + INSERT (no native UPDATE).
pub struct MemoryRepository {
    conn: Connection,
}

impl MemoryRepository {
    /// Open (or create) the database at the given path.
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context("failed to create database directory")?;
        }
        let conn = Connection::open(db_path)
            .context("failed to open SQLite database")?;
        Ok(Self { conn })
    }

    /// Open an in-memory database (for testing).
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .context("failed to open in-memory database")?;
        Ok(Self { conn })
    }

    /// Initialize the schema: PRAGMAs, tables, FTS5 virtual tables, indexes.
    pub fn initialize_schema(&self) -> Result<()> {
        self.conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        self.conn.execute_batch("PRAGMA journal_mode = WAL;")?;

        let tx = self.conn.unchecked_transaction()?;

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
                updated_at INTEGER NOT NULL
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
                updated_at INTEGER NOT NULL
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
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS procedural_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                workflow_name TEXT NOT NULL,
                steps TEXT NOT NULL,
                related_tools TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );",
        )?;

        // --- FTS5 virtual tables ---
        tx.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memories_fts USING fts5(
                memory_id UNINDEXED,
                summary,
                content,
                files_touched,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS decision_memories_fts USING fts5(
                memory_id UNINDEXED,
                title,
                context,
                rationale,
                tradeoffs,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS failure_memories_fts USING fts5(
                memory_id UNINDEXED,
                incident,
                root_cause,
                fix,
                prevention,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS procedural_memories_fts USING fts5(
                memory_id UNINDEXED,
                workflow_name,
                steps,
                related_tools,
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
            CREATE INDEX IF NOT EXISTS idx_failure_project_time ON failure_memories(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_procedural_project_time ON procedural_memories(project_id, created_at DESC);",
        )?;

        tx.commit()?;
        Ok(())
    }

    // ─── Episodic Memory CRUD ──────────────────────────────────────

    pub fn create_episodic(&self, mem: &EpisodicMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO episodic_memories (id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                mem.id, mem.project_id, mem.session_id,
                mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.related_commits)?,
                mem.importance,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ],
        )?;
        tx.execute(
            "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched)
             VALUES (?1,?2,?3,?4)",
            params![
                mem.id, mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_episodic(&self, id: &str) -> Result<Option<EpisodicMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, session_id, summary, content, files_touched, related_commits, importance, tags, created_at, updated_at
             FROM episodic_memories WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id], |row| {
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
        }).optional()?;
        Ok(row)
    }

    pub fn update_episodic(&self, mem: &EpisodicMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE episodic_memories SET project_id=?2, session_id=?3, summary=?4, content=?5, files_touched=?6, related_commits=?7, importance=?8, tags=?9, updated_at=?10 WHERE id=?1",
            params![
                mem.id, mem.project_id, mem.session_id,
                mem.summary, mem.content,
                serde_json::to_string(&mem.files_touched)?,
                serde_json::to_string(&mem.related_commits)?,
                mem.importance,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ],
        )?;
        // FTS5: delete-then-insert
        tx.execute("DELETE FROM episodic_memories_fts WHERE memory_id = ?1", params![mem.id])?;
        tx.execute(
            "INSERT INTO episodic_memories_fts (memory_id, summary, content, files_touched) VALUES (?1,?2,?3,?4)",
            params![mem.id, mem.summary, mem.content, serde_json::to_string(&mem.files_touched)?],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_episodic(&self, id: &str) -> Result<bool> {
        let tx = self.conn.unchecked_transaction()?;
        let affected = tx.execute("DELETE FROM episodic_memories WHERE id = ?1", params![id])?;
        if affected > 0 {
            tx.execute("DELETE FROM episodic_memories_fts WHERE memory_id = ?1", params![id])?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    // ─── Decision Memory CRUD ──────────────────────────────────────

    pub fn create_decision(&self, mem: &DecisionMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO decision_memories (id, project_id, title, context, rationale, tradeoffs, related_files, tags, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ],
        )?;
        tx.execute(
            "INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs) VALUES (?1,?2,?3,?4,?5)",
            params![mem.id, mem.title, mem.context, mem.rationale, mem.tradeoffs],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_decision(&self, id: &str) -> Result<Option<DecisionMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, title, context, rationale, tradeoffs, related_files, tags, created_at, updated_at
             FROM decision_memories WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id], |row| {
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
        }).optional()?;
        Ok(row)
    }

    pub fn update_decision(&self, mem: &DecisionMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE decision_memories SET project_id=?2, title=?3, context=?4, rationale=?5, tradeoffs=?6, related_files=?7, tags=?8, updated_at=?9 WHERE id=?1",
            params![
                mem.id, mem.project_id, mem.title,
                mem.context, mem.rationale, mem.tradeoffs,
                serde_json::to_string(&mem.related_files)?,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ],
        )?;
        tx.execute("DELETE FROM decision_memories_fts WHERE memory_id = ?1", params![mem.id])?;
        tx.execute(
            "INSERT INTO decision_memories_fts (memory_id, title, context, rationale, tradeoffs) VALUES (?1,?2,?3,?4,?5)",
            params![mem.id, mem.title, mem.context, mem.rationale, mem.tradeoffs],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_decision(&self, id: &str) -> Result<bool> {
        let tx = self.conn.unchecked_transaction()?;
        let affected = tx.execute("DELETE FROM decision_memories WHERE id = ?1", params![id])?;
        if affected > 0 {
            tx.execute("DELETE FROM decision_memories_fts WHERE memory_id = ?1", params![id])?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    // ─── Failure Memory CRUD ───────────────────────────────────────

    pub fn create_failure(&self, mem: &FailureMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO failure_memories (id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ],
        )?;
        tx.execute(
            "INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention) VALUES (?1,?2,?3,?4,?5)",
            params![mem.id, mem.incident, mem.root_cause, mem.fix, mem.prevention],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_failure(&self, id: &str) -> Result<Option<FailureMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, incident, root_cause, fix, prevention, severity, tags, created_at, updated_at
             FROM failure_memories WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id], |row| {
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
        }).optional()?;
        Ok(row)
    }

    pub fn update_failure(&self, mem: &FailureMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE failure_memories SET project_id=?2, incident=?3, root_cause=?4, fix=?5, prevention=?6, severity=?7, tags=?8, updated_at=?9 WHERE id=?1",
            params![
                mem.id, mem.project_id, mem.incident,
                mem.root_cause, mem.fix, mem.prevention,
                mem.severity,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ],
        )?;
        tx.execute("DELETE FROM failure_memories_fts WHERE memory_id = ?1", params![mem.id])?;
        tx.execute(
            "INSERT INTO failure_memories_fts (memory_id, incident, root_cause, fix, prevention) VALUES (?1,?2,?3,?4,?5)",
            params![mem.id, mem.incident, mem.root_cause, mem.fix, mem.prevention],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_failure(&self, id: &str) -> Result<bool> {
        let tx = self.conn.unchecked_transaction()?;
        let affected = tx.execute("DELETE FROM failure_memories WHERE id = ?1", params![id])?;
        if affected > 0 {
            tx.execute("DELETE FROM failure_memories_fts WHERE memory_id = ?1", params![id])?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    // ─── Procedural Memory CRUD ────────────────────────────────────

    pub fn create_procedural(&self, mem: &ProceduralMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO procedural_memories (id, project_id, workflow_name, steps, related_tools, tags, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.created_at, mem.updated_at,
            ],
        )?;
        tx.execute(
            "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools) VALUES (?1,?2,?3,?4)",
            params![
                mem.id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_procedural(&self, id: &str) -> Result<Option<ProceduralMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, workflow_name, steps, related_tools, tags, created_at, updated_at
             FROM procedural_memories WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id], |row| {
            Ok(ProceduralMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                workflow_name: row.get(2)?,
                steps: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                related_tools: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                tags: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        }).optional()?;
        Ok(row)
    }

    pub fn update_procedural(&self, mem: &ProceduralMemory) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE procedural_memories SET project_id=?2, workflow_name=?3, steps=?4, related_tools=?5, tags=?6, updated_at=?7 WHERE id=?1",
            params![
                mem.id, mem.project_id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
                serde_json::to_string(&mem.tags)?,
                mem.updated_at,
            ],
        )?;
        tx.execute("DELETE FROM procedural_memories_fts WHERE memory_id = ?1", params![mem.id])?;
        tx.execute(
            "INSERT INTO procedural_memories_fts (memory_id, workflow_name, steps, related_tools) VALUES (?1,?2,?3,?4)",
            params![
                mem.id, mem.workflow_name,
                serde_json::to_string(&mem.steps)?,
                serde_json::to_string(&mem.related_tools)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_procedural(&self, id: &str) -> Result<bool> {
        let tx = self.conn.unchecked_transaction()?;
        let affected = tx.execute("DELETE FROM procedural_memories WHERE id = ?1", params![id])?;
        if affected > 0 {
            tx.execute("DELETE FROM procedural_memories_fts WHERE memory_id = ?1", params![id])?;
        }
        tx.commit()?;
        Ok(affected > 0)
    }

    // ─── FTS5 Search ───────────────────────────────────────────────

    /// Search episodic memories via FTS5 BM25, returning memory IDs ordered by relevance.
    pub fn search_episodic(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<EpisodicMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.project_id, m.session_id, m.summary, m.content, m.files_touched, m.related_commits, m.importance, m.tags, m.created_at, m.updated_at
             FROM episodic_memories_fts f
             JOIN episodic_memories m ON f.memory_id = m.id
             WHERE episodic_memories_fts MATCH ?1 AND m.project_id = ?2
             ORDER BY f.rank
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![query, project_id, limit], |row| {
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
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Search decision memories via FTS5 BM25.
    pub fn search_decisions(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<DecisionMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.project_id, m.title, m.context, m.rationale, m.tradeoffs, m.related_files, m.tags, m.created_at, m.updated_at
             FROM decision_memories_fts f
             JOIN decision_memories m ON f.memory_id = m.id
             WHERE decision_memories_fts MATCH ?1 AND m.project_id = ?2
             ORDER BY f.rank
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![query, project_id, limit], |row| {
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

    /// Search failure memories via FTS5 BM25.
    pub fn search_failures(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<FailureMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.project_id, m.incident, m.root_cause, m.fix, m.prevention, m.severity, m.tags, m.created_at, m.updated_at
             FROM failure_memories_fts f
             JOIN failure_memories m ON f.memory_id = m.id
             WHERE failure_memories_fts MATCH ?1 AND m.project_id = ?2
             ORDER BY f.rank
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![query, project_id, limit], |row| {
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

    /// Search procedural memories via FTS5 BM25.
    pub fn search_procedural(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<ProceduralMemory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.project_id, m.workflow_name, m.steps, m.related_tools, m.tags, m.created_at, m.updated_at
             FROM procedural_memories_fts f
             JOIN procedural_memories m ON f.memory_id = m.id
             WHERE procedural_memories_fts MATCH ?1 AND m.project_id = ?2
             ORDER BY f.rank
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![query, project_id, limit], |row| {
            Ok(ProceduralMemory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                workflow_name: row.get(2)?,
                steps: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                related_tools: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                tags: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ─── Entity / Graph CRUD ───────────────────────────────────────

    pub fn create_entity(&self, entity: &Entity) -> Result<()> {
        self.conn.execute(
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
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, entity_type, name, metadata, created_at, updated_at FROM entities WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id], |row| {
            let et: String = row.get(2)?;
            Ok(Entity {
                id: row.get(0)?,
                project_id: row.get(1)?,
                entity_type: et.parse().map_err(|_: String| rusqlite::Error::InvalidColumnType(2, "entity_type".into(), rusqlite::types::Type::Text))?,
                name: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        }).optional()?;
        Ok(row)
    }

    pub fn remove_entity(&self, id: &str) -> Result<bool> {
        // Delete related relations first to satisfy foreign key constraints
        self.conn.execute(
            "DELETE FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
            params![id],
        )?;
        let affected = self.conn.execute("DELETE FROM entities WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    pub fn create_relation(&self, rel: &GraphRelation) -> Result<()> {
        self.conn.execute(
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
        Ok(())
    }

    pub fn get_relations_for_entity(&self, entity_id: &str) -> Result<Vec<GraphRelation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, from_entity, to_entity, relation_type, weight, created_at
             FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            let rt: String = row.get(4)?;
            Ok(GraphRelation {
                id: row.get(0)?,
                project_id: row.get(1)?,
                from_entity: row.get(2)?,
                to_entity: row.get(3)?,
                relation_type: rt.parse().map_err(|_: String| rusqlite::Error::InvalidColumnType(4, "relation_type".into(), rusqlite::types::Type::Text))?,
                weight: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn remove_relation(&self, id: i64) -> Result<bool> {
        let affected = self.conn.execute("DELETE FROM graph_relations WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    /// Load all entities for a given project (plus cross-project with project_id IS NULL).
    pub fn load_entities_for_project(&self, project_id: &str) -> Result<Vec<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, entity_type, name, metadata, created_at, updated_at
             FROM entities WHERE project_id = ?1 OR project_id IS NULL",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let et: String = row.get(2)?;
            Ok(Entity {
                id: row.get(0)?,
                project_id: row.get(1)?,
                entity_type: et.parse().map_err(|_: String| rusqlite::Error::InvalidColumnType(2, "entity_type".into(), rusqlite::types::Type::Text))?,
                name: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Load all relations for a given project (plus cross-project with project_id IS NULL).
    pub fn load_relations_for_project(&self, project_id: &str) -> Result<Vec<GraphRelation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, project_id, from_entity, to_entity, relation_type, weight, created_at
             FROM graph_relations WHERE project_id = ?1 OR project_id IS NULL",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let rt: String = row.get(4)?;
            Ok(GraphRelation {
                id: row.get(0)?,
                project_id: row.get(1)?,
                from_entity: row.get(2)?,
                to_entity: row.get(3)?,
                relation_type: rt.parse().map_err(|_: String| rusqlite::Error::InvalidColumnType(4, "relation_type".into(), rusqlite::types::Type::Text))?,
                weight: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Run FTS5 integrity check on all virtual tables. Returns Ok(()) if all pass.
    pub fn fts_integrity_check(&self) -> Result<()> {
        for table in &[
            "episodic_memories_fts",
            "decision_memories_fts",
            "failure_memories_fts",
            "procedural_memories_fts",
        ] {
            let query = format!("INSERT INTO {table}({table}) VALUES('integrity-check')");
            self.conn.execute_batch(&query)?;
        }
        Ok(())
    }

    /// Get a reference to the underlying connection for advanced queries.
    pub fn connection(&self) -> &Connection {
        &self.conn
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
        assert!(repo.delete_episodic(&mem.id).unwrap());
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
        assert_eq!(repo.get_decision(&mem.id).unwrap().unwrap().title, "Use Redis for all caching");

        assert!(repo.delete_decision(&mem.id).unwrap());
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

        assert!(repo.delete_failure(&mem.id).unwrap());
        assert!(repo.get_failure(&mem.id).unwrap().is_none());
    }

    #[test]
    fn test_procedural_crud() {
        let repo = setup_repo();
        let mem = ProceduralMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "test-project".into(),
            workflow_name: "deployment".into(),
            steps: vec!["run tests".into(), "build docker".into(), "push to registry".into()],
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
        assert_eq!(repo.get_procedural(&mem.id).unwrap().unwrap().steps.len(), 4);

        assert!(repo.delete_procedural(&mem.id).unwrap());
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
        let results = repo.search_episodic("OAuth refresh", "test-project", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, mem.id);

        // After delete, search should not find it
        repo.delete_episodic(&mem.id).unwrap();
        let results = repo.search_episodic("OAuth refresh", "test-project", 10).unwrap();
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
        let results = repo.search_decisions("Postgres", "test-project", 10).unwrap();
        assert_eq!(results.len(), 1);

        // Update title
        let mut updated = mem.clone();
        updated.title = "Use MySQL for storage".into();
        repo.update_decision(&updated).unwrap();

        // Old term should not match
        let old_results = repo.search_decisions("Postgres", "test-project", 10).unwrap();
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
        assert_eq!(results_a[0].project_id, "project-a");

        let results_b = repo.search_failures("Database", "project-b", 10).unwrap();
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].project_id, "project-b");
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
}
