/// Read a JSON-serialized column from a rusqlite Row, deserializing with serde.
/// Returns the default value on parse failure and logs a warning.
///
/// Usage: `row_get_json!(row, 5, Vec<String>)`
macro_rules! row_get_json {
    ($row:expr, $idx:expr, $ty:ty) => {
        match serde_json::from_str::<$ty>(&$row.get::<_, String>($idx)?) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("JSON parse failed at column {}: {e}", $idx);
                <$ty>::default()
            }
        }
    };
}

/// Generate full CRUD + FTS5 search methods for a memory type.
///
/// Produces 5 public methods on the implementing type:
/// - `$create_fn(&self, mem: &Struct) -> Result<()>`
/// - `$get_fn(&self, id: &str) -> Result<Option<Struct>>`
/// - `$update_fn(&self, mem: &Struct) -> Result<()>` (errors if no row matched)
/// - `$delete_fn(&self, id: &str) -> Result<bool>`
/// - `$search_fn(&self, query, project_id, limit) -> Result<Vec<ScoredMemory<Struct>>>`
///
/// `mem`, `tx`, `row`, and `link_ts` are passed as ident parameters to preserve
/// macro hygiene — the invocation-site tokens reference these same names.
/// `link_ts` is the timestamp handed to the entity-linking block: the memory's
/// `created_at` on create, `updated_at` on update.
macro_rules! impl_memory_crud {
    (
        // Parameter names — from invocation scope for hygiene
        mem = $mem:ident,
        tx = $tx:ident,
        row = $row:ident,
        link_ts = $link_ts:ident,
        fts_rowid = $fts_rowid:ident,

        struct_type = $Struct:ident,
        table = $table:literal,
        fts_table = $fts_table:literal,

        create_fn = $create_fn:ident,
        get_fn = $get_fn:ident,
        update_fn = $update_fn:ident,
        delete_fn = $delete_fn:ident,
        search_fn = $search_fn:ident,
        list_active_fn = $list_active_fn:ident,

        select_cols = $select_cols:literal,
        search_cols = $search_cols:literal,
        insert_sql = $insert_sql:literal,
        fts_insert_sql = $fts_insert_sql:literal,
        update_sql = $update_sql:literal,
        score_col_idx = $score_col_idx:literal,

        insert_params = { $($insert_params:tt)* },
        fts_params = { $($fts_params:tt)* },
        update_params = { $($update_params:tt)* },

        row_mapper = { $($row_mapper:tt)* },
        entity_link = { $($entity_link:tt)* },
    ) => {
        // ─── CREATE ──────────────────────────────────────────────
        pub fn $create_fn(&self, $mem: &$Struct) -> Result<()> {
            let conn = self.conn()?;
            let $tx = conn.unchecked_transaction()?;
            #[allow(unused_variables)]
            let $link_ts = $mem.created_at;
            $tx.execute($insert_sql, $($insert_params)*)?;
            // Keep the FTS rowid in lockstep with the main-table rowid so
            // deletes/updates address FTS rows by rowid (O(log n)). Matching
            // the UNINDEXED memory_id column instead was a full FTS scan on
            // EVERY update/delete (confirmed via EXPLAIN QUERY PLAN).
            let $fts_rowid: i64 = $tx.last_insert_rowid();
            $tx.execute($fts_insert_sql, $($fts_params)*)?;
            $($entity_link)*
            $tx.commit()?;
            Ok(())
        }

        // ─── GET ─────────────────────────────────────────────────
        /// Project-scoped at the SQL level: isolation no longer depends on
        /// every caller remembering to compare `project_id` afterwards (the
        /// 2026-09 review found unguarded call paths, e.g. vector-candidate
        /// materialization in `fetch_by_ids`).
        pub fn $get_fn(&self, id: &str, project_id: &str) -> Result<Option<$Struct>> {
            let conn = self.conn()?;
            let mut stmt = conn.prepare(concat!(
                "SELECT ", $select_cols, " FROM ", $table,
                " WHERE id = ?1 AND project_id = ?2"
            ))?;
            let $row = stmt.query_row(params![id, project_id], |$row| {
                Ok($($row_mapper)*)
            }).optional()?;
            Ok($row)
        }

        // ─── UPDATE ──────────────────────────────────────────────
        pub fn $update_fn(&self, $mem: &$Struct) -> Result<()> {
            let conn = self.conn()?;
            let $tx = conn.unchecked_transaction()?;
            #[allow(unused_variables)]
            let $link_ts = $mem.updated_at;
            let affected = $tx.execute($update_sql, $($update_params)*)?;
            if affected == 0 {
                // No row matched. Failing here (before any FTS write) prevents
                // the orphan-index bug: previously the delete-then-insert below
                // ran unconditionally, leaving FTS rows for nonexistent ids.
                // The uncommitted transaction rolls back on drop.
                anyhow::bail!(
                    "update failed: no row with id {} in {}",
                    $mem.id,
                    $table
                );
            }
            // FTS5: delete-then-insert, addressed by the stable main-table
            // rowid (rowids are unchanged by UPDATE).
            let $fts_rowid: i64 = $tx.query_row(
                concat!("SELECT rowid FROM ", $table, " WHERE id = ?1 AND project_id = ?2"),
                params![$mem.id, $mem.project_id],
                |r| r.get(0),
            )?;
            $tx.execute(
                concat!("DELETE FROM ", $fts_table, " WHERE rowid = ?1"),
                params![$fts_rowid],
            )?;
            $tx.execute($fts_insert_sql, $($fts_params)*)?;
            // Refresh graph edges: files/tools may have changed since create,
            // so stale relations are dropped before re-linking (ON CONFLICT
            // DO NOTHING alone can't remove edges for removed names).
            $tx.execute(
                "DELETE FROM graph_relations WHERE from_entity = ?1",
                params![$mem.id],
            )?;
            $($entity_link)*
            $tx.commit()?;
            Ok(())
        }

        // ─── DELETE ──────────────────────────────────────────────
        pub fn $delete_fn(&self, id: &str, project_id: &str) -> Result<bool> {
            let conn = self.conn()?;
            let $tx = conn.unchecked_transaction()?;
            // Resolve the FTS row by rowid before the main row goes away.
            let fts_rowid: Option<i64> = $tx
                .query_row(
                    concat!("SELECT rowid FROM ", $table, " WHERE id = ?1 AND project_id = ?2"),
                    params![id, project_id],
                    |r| r.get(0),
                )
                .optional()?;
            let affected = $tx.execute(
                concat!("DELETE FROM ", $table, " WHERE id = ?1 AND project_id = ?2"),
                params![id, project_id],
            )?;
            if affected > 0 {
                if let Some(rid) = fts_rowid {
                    $tx.execute(
                        concat!("DELETE FROM ", $fts_table, " WHERE rowid = ?1"),
                        params![rid],
                    )?;
                }
                // Clean up graph relations and entity for this memory
                $tx.execute(
                    "DELETE FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
                    params![id],
                )?;
                $tx.execute(
                    "DELETE FROM entities WHERE id = ?1",
                    params![id],
                )?;
                // Clean up any stored embedding for this memory
                $tx.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                    params![id],
                )?;
            }
            $tx.commit()?;
            Ok(affected > 0)
        }

        // ─── SEARCH (FTS5 BM25) ─────────────────────────────────
        pub fn $search_fn(
            &self, query: &str, project_id: &str, limit: usize,
        ) -> Result<Vec<ScoredMemory<$Struct>>> {
            // Multi-token queries try AND first (every token must match) for
            // precision, and fall back to OR (any token matches) when AND
            // returns nothing — the old always-OR behavior made "rust memory
            // system" match any memory containing just "system".
            let and_query = Self::sanitize_fts_query(query, true);
            let or_query = Self::sanitize_fts_query(query, false);
            let conn = self.conn()?;
            let mut stmt = conn.prepare(concat!(
                "SELECT ", $search_cols, ", bm25(", $fts_table, ") as score",
                " FROM ", $fts_table, " f",
                " JOIN ", $table, " m ON f.memory_id = m.id",
                " WHERE ", $fts_table, " MATCH ?1 AND m.project_id = ?2 AND m.archived_at IS NULL",
                " ORDER BY f.rank LIMIT ?3"
            ))?;
            let mut fetch = |fts_query: &str| -> Result<Vec<ScoredMemory<$Struct>>> {
                let rows = stmt.query_map(params![fts_query, project_id, limit], |$row| {
                    Ok(ScoredMemory {
                        memory: $($row_mapper)*,
                        bm25_score: $row.get::<_, f64>($score_col_idx)?,
                    })
                })?;
                let mut out = Vec::new();
                for row in rows {
                    out.push(row?);
                }
                Ok(out)
            };
            let results = fetch(&and_query)?;
            if !results.is_empty() || and_query == or_query {
                return Ok(results);
            }
            fetch(&or_query)
        }

        // ─── LIST ACTIVE (plain SELECT, no FTS) — for reindex/backfill ──
        pub fn $list_active_fn(&self, project: Option<&str>) -> Result<Vec<$Struct>> {
            let conn = self.conn()?;
            let mut results = Vec::new();
            match project {
                Some(p) => {
                    let mut stmt = conn.prepare(concat!(
                        "SELECT ", $select_cols, " FROM ", $table,
                        " WHERE archived_at IS NULL AND project_id = ?1 ORDER BY created_at"
                    ))?;
                    let rows = stmt.query_map(params![p], |$row| Ok($($row_mapper)*))?;
                    for r in rows {
                        results.push(r?);
                    }
                }
                None => {
                    let mut stmt = conn.prepare(concat!(
                        "SELECT ", $select_cols, " FROM ", $table,
                        " WHERE archived_at IS NULL ORDER BY created_at"
                    ))?;
                    let rows = stmt.query_map([], |$row| Ok($($row_mapper)*))?;
                    for r in rows {
                        results.push(r?);
                    }
                }
            }
            Ok(results)
        }
    };
}
