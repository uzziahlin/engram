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
/// - `$update_fn(&self, mem: &Struct) -> Result<()>`
/// - `$delete_fn(&self, id: &str) -> Result<bool>`
/// - `$search_fn(&self, query, project_id, limit) -> Result<Vec<ScoredMemory<Struct>>>`
///
/// `mem`, `tx`, and `row` are passed as ident parameters to preserve
/// macro hygiene — the invocation-site tokens reference these same names.
macro_rules! impl_memory_crud {
    (
        // Parameter names — from invocation scope for hygiene
        mem = $mem:ident,
        tx = $tx:ident,
        row = $row:ident,

        struct_type = $Struct:ident,
        table = $table:literal,
        fts_table = $fts_table:literal,

        create_fn = $create_fn:ident,
        get_fn = $get_fn:ident,
        update_fn = $update_fn:ident,
        delete_fn = $delete_fn:ident,
        search_fn = $search_fn:ident,

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
            let $tx = self.conn.unchecked_transaction()?;
            $tx.execute($insert_sql, $($insert_params)*)?;
            $tx.execute($fts_insert_sql, $($fts_params)*)?;
            $($entity_link)*
            $tx.commit()?;
            Ok(())
        }

        // ─── GET ─────────────────────────────────────────────────
        pub fn $get_fn(&self, id: &str) -> Result<Option<$Struct>> {
            let mut stmt = self.conn.prepare(concat!(
                "SELECT ", $select_cols, " FROM ", $table, " WHERE id = ?1"
            ))?;
            let $row = stmt.query_row(params![id], |$row| {
                Ok($($row_mapper)*)
            }).optional()?;
            Ok($row)
        }

        // ─── UPDATE ──────────────────────────────────────────────
        pub fn $update_fn(&self, $mem: &$Struct) -> Result<()> {
            let $tx = self.conn.unchecked_transaction()?;
            $tx.execute($update_sql, $($update_params)*)?;
            // FTS5: delete-then-insert
            $tx.execute(
                concat!("DELETE FROM ", $fts_table, " WHERE memory_id = ?1"),
                params![$mem.id],
            )?;
            $tx.execute($fts_insert_sql, $($fts_params)*)?;
            $tx.commit()?;
            Ok(())
        }

        // ─── DELETE ──────────────────────────────────────────────
        pub fn $delete_fn(&self, id: &str, project_id: &str) -> Result<bool> {
            let $tx = self.conn.unchecked_transaction()?;
            let affected = $tx.execute(
                concat!("DELETE FROM ", $table, " WHERE id = ?1 AND project_id = ?2"),
                params![id, project_id],
            )?;
            if affected > 0 {
                $tx.execute(
                    concat!("DELETE FROM ", $fts_table, " WHERE memory_id = ?1"),
                    params![id],
                )?;
                // Clean up graph relations and entity for this memory
                $tx.execute(
                    "DELETE FROM graph_relations WHERE from_entity = ?1 OR to_entity = ?1",
                    params![id],
                )?;
                $tx.execute(
                    "DELETE FROM entities WHERE id = ?1",
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
            let fts_query = Self::sanitize_fts_query(query);
            let mut stmt = self.conn.prepare(concat!(
                "SELECT ", $search_cols, ", bm25(", $fts_table, ") as score",
                " FROM ", $fts_table, " f",
                " JOIN ", $table, " m ON f.memory_id = m.id",
                " WHERE ", $fts_table, " MATCH ?1 AND m.project_id = ?2",
                " ORDER BY f.rank LIMIT ?3"
            ))?;
            let rows = stmt.query_map(params![fts_query, project_id, limit], |$row| {
                Ok(ScoredMemory {
                    memory: $($row_mapper)*,
                    bm25_score: $row.get::<_, f64>($score_col_idx)?,
                })
            })?;
            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok(results)
        }
    };
}
