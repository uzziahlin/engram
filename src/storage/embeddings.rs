//! Vector store for semantic retrieval (populated when the `semantic`
//! feature writes embeddings). BLOB serialization + per-model loading.

use super::repository::MemoryRepository;
use anyhow::Result;
use rusqlite::params;
use std::collections::HashSet;

impl MemoryRepository {
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
                 WHERE (?1 = '*' OR e.project_id = ?1) AND e.model_id = ?2 AND e.memory_type = ?3
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
