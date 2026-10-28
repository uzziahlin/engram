//! Entity graph: files/tools touched by memories, their relations, and the
//! neighbor queries that let the graph participate in retrieval.

use super::repository::{MemoryKind, MemoryRepository, RelatedFileEdge, RelatedMemory};
use crate::models::{Entity, GraphRelation};
use anyhow::Result;
use rusqlite::{params, OptionalExtension, TransactionBehavior};

impl MemoryRepository {
    // ─── Entity Linking Helper ────────────────────────────────────

    /// Ensure entities and relations exist for a memory's linked items (files/tools).
    /// Upserts a Memory entity for the record itself, then creates File/Tool entities
    /// and links them via the specified relation type. All within the caller's transaction.
    pub(crate) fn ensure_linked_entities(
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

    // ─── Graph-participating retrieval ────────────────────────────
    /// Memory ids that share a graph entity (file/tool) with any of the
    /// `seed` memories — the relationship graph's contribution to retrieval.
    /// Returns `(memory_type, id)` pairs, at most `limit`, excluding seeds.
    /// Only ACTIVE memories are returned: candidate ids are probed against
    /// the main tables in ONE batched query per kind (the old path ran a
    /// 4-EXISTS probe per id — four correlated subqueries per candidate).
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
             WHERE (?1 = '*' OR r1.project_id = ?1) \
               AND (?1 = '*' OR r2.project_id = ?1) \
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

        // Resolve ids to (active) memories: one IN-batch probe per kind.
        // Memory ids are UUIDs unique across kinds, so an id can match at
        // most one kind; `seen` keeps a kind-ambiguous id from being
        // reported twice anyway.
        let mut active: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut kinds: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        for kind in MemoryKind::all() {
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id FROM {} WHERE id IN ({ph}) AND archived_at IS NULL",
                kind.table()
            );
            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
            for id in stmt.query_map(params.as_slice(), |r| r.get::<_, String>(0))? {
                let id = id?;
                if active.insert(id.clone()) {
                    kinds.insert(id, kind.as_str());
                }
            }
        }
        let mut out = Vec::new();
        for id in ids {
            if out.len() >= limit {
                break;
            }
            if let Some(kind) = kinds.get(&id) {
                out.push((kind.to_string(), id));
            }
        }
        Ok(out)
    }

    // ─── Orphan entity GC ─────────────────────────────────────────
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
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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

    // ─── Entity / Graph CRUD ──────────────────────────────────────

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
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
        let mut conn = self.conn()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
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

    /// The ACTIVE memories that touch a file, with their type and a
    /// human-readable label — the consumable form of `related_files_for`.
    ///
    /// The old tool output listed neighbor entity NAMES, which for Memory
    /// entities are memory UUIDs: agents could not act on them (get_memory
    /// needs a type, and a bare uuid tells the agent nothing). This joins
    /// each Memory neighbor back to its main-table row for type + label.
    /// Non-memory neighbors (File/Tool entities) are not included here.
    pub fn related_memories_for_file(
        &self,
        file_name: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<RelatedMemory>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT mtype, mid, label, rel FROM ( \
                SELECT o.name AS mid, r.relation_type AS rel, \
                    CASE \
                        WHEN EXISTS (SELECT 1 FROM episodic_memories x WHERE x.id = o.name AND x.archived_at IS NULL) THEN 'episodic' \
                        WHEN EXISTS (SELECT 1 FROM decision_memories x WHERE x.id = o.name AND x.archived_at IS NULL) THEN 'decision' \
                        WHEN EXISTS (SELECT 1 FROM failure_memories x WHERE x.id = o.name AND x.archived_at IS NULL) THEN 'failure' \
                        WHEN EXISTS (SELECT 1 FROM procedural_memories x WHERE x.id = o.name AND x.archived_at IS NULL) THEN 'procedural' \
                    END AS mtype, \
                    COALESCE( \
                        (SELECT summary FROM episodic_memories WHERE id = o.name), \
                        (SELECT title FROM decision_memories WHERE id = o.name), \
                        (SELECT incident FROM failure_memories WHERE id = o.name), \
                        (SELECT workflow_name FROM procedural_memories WHERE id = o.name) \
                    ) AS label \
                FROM entities e \
                JOIN graph_relations r \
                  ON r.from_entity = e.id OR r.to_entity = e.id \
                JOIN entities o \
                  ON o.id = CASE WHEN r.from_entity = e.id \
                                 THEN r.to_entity ELSE r.from_entity END \
                WHERE e.name = ?1 AND e.project_id = ?2 AND e.entity_type = 'File' \
                  AND o.entity_type = 'Memory' \
             ) WHERE mtype IS NOT NULL \
             ORDER BY mid \
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![file_name, project_id, limit], |row| {
            Ok(RelatedMemory {
                memory_type: row.get(0)?,
                id: row.get(1)?,
                label: row.get(2)?,
                relation: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
}
