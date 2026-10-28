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
/// Produces 6 public methods on the implementing type:
/// - `$create_fn(&self, mem: &Struct) -> Result<()>`
/// - `$create_dedup_fn(&self, mem: &Struct) -> Result<Option<String>>`
/// - `$get_fn(&self, id: &str, project_id: &str) -> Result<Option<Struct>>`
/// - `$update_fn(&self, mem: &Struct) -> Result<()>` (errors if no ACTIVE row matched)
/// - `$delete_fn(&self, id: &str, project_id: &str) -> Result<bool>`
/// - `$search_fn(&self, query, project_id, limit) -> Result<Vec<ScoredMemory<Struct>>>`
///
/// FTS handling targets the UNIFIED `memories_fts` table shared by all four
/// kinds: inserts carry `rowid = main_rowid * 4 + kind_index` plus the
/// project-hex/kind filter columns, and the search wrapper delegates to
/// `search_unified` (one MATCH over the whole corpus, kind-filtered).
///
/// `mem`, `tx`, `row`, `link_ts`, and `hash` are passed as ident parameters to
/// preserve macro hygiene — the invocation-site tokens reference these same
/// names. `link_ts` is the timestamp handed to the entity-linking block: the
/// memory's `created_at` on create, `updated_at` on update. `hash` is the
/// row's `content_hash` participation value: `None` on plain create
/// (write_dedup off / direct storage callers), `Some(sha256)` on the dedup
/// create and on updates.
macro_rules! impl_memory_crud {
    (
        // Parameter names — from invocation scope for hygiene
        mem = $mem:ident,
        tx = $tx:ident,
        row = $row:ident,
        link_ts = $link_ts:ident,
        fts_rowid = $fts_rowid:ident,
        hash = $hash:ident,

        struct_type = $Struct:ident,
        table = $table:literal,

        kind_variant = $kind_variant:ident,
        kind_index = $kind_index:literal,
        kind_str = $kind_str:literal,
        unified_hits_field = $unified_field:ident,

        create_fn = $create_fn:ident,
        create_dedup_fn = $create_dedup_fn:ident,
        get_fn = $get_fn:ident,
        get_active_fn = $get_active_fn:ident,
        update_fn = $update_fn:ident,
        delete_fn = $delete_fn:ident,
        search_fn = $search_fn:ident,
        list_active_fn = $list_active_fn:ident,

        select_cols = $select_cols:literal,

        insert_sql = $insert_sql:literal,

        unified_doc = { $($unified_doc:tt)* },

        dedup_text = { $($dedup_text:tt)* },

        update_sql = $update_sql:literal,

        insert_params = { $($insert_params:tt)* },
        update_params = { $($update_params:tt)* },

        row_mapper = { $($row_mapper:tt)* },
        entity_link = { $($entity_link:tt)* },
    ) => {
        // ─── CREATE ──────────────────────────────────────────────
        pub fn $create_fn(&self, $mem: &$Struct) -> Result<()> {
            // Sanitize at the storage boundary: importance/severity are
            // clamped here no matter which entry point (MCP/CLI/import)
            // produced the struct, instead of trusting per-caller validation.
            let $mem = $mem.clone().sanitized();
            let mut conn = self.conn()?;
            // IMMEDIATE, not the default deferred BEGIN: every write tx takes
            // the write lock up front, where busy_timeout applies (deferred
            // transactions upgrading on their first write can return SQLITE_BUSY
            // without honoring it — see the rationale on initialize_schema).
            // Concurrency contract: handler calls are serialized by the
            // transport's single-worker Dispatcher; IMMEDIATE is the second
            // line of defense (multi-worker / multi-process configs).
            let $tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            #[allow(unused_variables)]
            let $link_ts = $mem.created_at;
            // write_dedup=false semantics (also: direct storage callers such
            // as the CLI session-import upsert and the consolidation tests):
            // the row carries NO content_hash, so the partial unique index
            // ignores it and duplicates are created freely. Only
            // $create_dedup_fn participates in dedup.
            #[allow(unused_variables)]
            let $hash: Option<String> = None;
            $tx.execute($insert_sql, $($insert_params)*)?;
            // Unified-FTS rowid encodes (kind, main rowid) in one i64:
            // deletes/updates address the FTS row by rowid (O(log n)) and
            // materialization can decode its way back to the main table —
            // a contentless table exposes no column values to join on.
            let $fts_rowid: i64 =
                encode_fts_rowid(MemoryKind::$kind_variant, $tx.last_insert_rowid());
            let (u_title, u_body, u_tags) = $($unified_doc)*;
            $tx.execute(
                crate::storage::schema::UNIFIED_FTS_INSERT,
                params![
                    u_title,
                    u_body,
                    u_tags,
                    Self::project_fts_hex(&$mem.project_id),
                    $kind_str,
                    $fts_rowid,
                ],
            )?;
            $($entity_link)*
            $tx.commit()?;
            Ok(())
        }

        // ─── CREATE (dedup-guarded) ──────────────────────────────
        /// Write-time exact-dedup create (write_dedup=true path): the row's
        /// content_hash is written and the PARTIAL UNIQUE INDEX is the guard —
        /// the INSERT statement itself enforces dedup atomically, replacing
        /// the old two-step `find_exact_duplicate` → create that two racing
        /// callers could both pass (and that scanned every active row of the
        /// kind on the session-hook hot path).
        ///
        /// On a unique-index collision the existing ACTIVE row's id is
        /// returned (`Ok(Some(id))`) and NOTHING is written; the caller turns
        /// that into the "duplicate" response — it is not an error. sha256
        /// over the full dedup text makes a false collision negligible.
        pub fn $create_dedup_fn(&self, $mem: &$Struct) -> Result<Option<String>> {
            let $mem = $mem.clone().sanitized();
            let $hash: Option<String> = Some(crate::storage::repository::content_hash(
                &($($dedup_text)*),
            ));
            let mut conn = self.conn()?;
            let $tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            #[allow(unused_variables)]
            let $link_ts = $mem.created_at;
            let inserted = $tx.execute($insert_sql, $($insert_params)*);
            if let Err(e) = inserted {
                if crate::storage::repository::is_unique_violation(&e) {
                    // The index rejected OUR row: an ACTIVE row of this kind
                    // already carries this content_hash (project-scoped — the
                    // same text in another project is unaffected). Report it
                    // for the duplicate response. The failed statement rolled
                    // back only itself; dropping `$tx` discards the empty
                    // transaction.
                    let existing: Option<String> = $tx
                        .query_row(
                            concat!(
                                "SELECT id FROM ", $table,
                                " WHERE project_id = ?1 AND content_hash = ?2",
                                " AND archived_at IS NULL LIMIT 1"
                            ),
                            params![&$mem.project_id, &$hash],
                            |r| r.get(0),
                        )
                        .optional()?;
                    if let Some(existing) = existing {
                        return Ok(Some(existing));
                    }
                    // A UNIQUE violation with no matching hash row means some
                    // other constraint fired (e.g. the id PRIMARY KEY):
                    // surface the original error rather than a bogus duplicate.
                }
                return Err(e.into());
            }
            let $fts_rowid: i64 =
                encode_fts_rowid(MemoryKind::$kind_variant, $tx.last_insert_rowid());
            let (u_title, u_body, u_tags) = $($unified_doc)*;
            $tx.execute(
                crate::storage::schema::UNIFIED_FTS_INSERT,
                params![
                    u_title,
                    u_body,
                    u_tags,
                    Self::project_fts_hex(&$mem.project_id),
                    $kind_str,
                    $fts_rowid,
                ],
            )?;
            $($entity_link)*
            $tx.commit()?;
            Ok(None)
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

        // ─── GET ACTIVE (wildcard project, archived excluded) ──────
        /// Like the getter above, but (a) `project_id = '*'` matches any
        /// project — global search materialization — and (b) archived rows
        /// are excluded, so a soft-deleted memory can never resurface via
        /// vector or graph candidates (its stale embedding/edges may still
        /// exist even after `forget_memory`).
        pub fn $get_active_fn(&self, id: &str, project_id: &str) -> Result<Option<$Struct>> {
            let conn = self.conn()?;
            let mut stmt = conn.prepare(concat!(
                "SELECT ", $select_cols, " FROM ", $table,
                " WHERE id = ?1 AND (?2 = '*' OR project_id = ?2) \
                  AND archived_at IS NULL"
            ))?;
            let $row = stmt
                .query_row(params![id, project_id], |$row| {
                    Ok($($row_mapper)*)
                })
                .optional()?;
            Ok($row)
        }

        // ─── UPDATE ──────────────────────────────────────────────
        pub fn $update_fn(&self, $mem: &$Struct) -> Result<()> {
            // Same storage-boundary sanitization as create (see above).
            let $mem = $mem.clone().sanitized();
            let mut conn = self.conn()?;
            let $tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            #[allow(unused_variables)]
            let $link_ts = $mem.updated_at;
            // Recompute the hash for the edited text. Rows created WITHOUT a
            // hash (write_dedup off) stay untracked: the SET clause's CASE
            // propagates the old NULL — otherwise marking relevance on one of
            // two pre-existing duplicates would suddenly write a hash and the
            // second duplicate's update would collide. Tracked rows keep a
            // fresh hash after every edit.
            #[allow(unused_variables)]
            let $hash: Option<String> = Some(crate::storage::repository::content_hash(
                &($($dedup_text)*),
            ));
            let affected = match $tx.execute($update_sql, $($update_params)*) {
                Ok(n) => n,
                Err(e) => {
                    if crate::storage::repository::is_unique_violation(&e) {
                        // Update-collision (v10): the edited text hashes to a
                        // hash another ACTIVE row of this project already
                        // carries. Updates are explicit edits — they must NOT
                        // silently merge duplicates (unlike create, which
                        // reports "duplicate"); name the conflicting memory.
                        let other: Option<String> = $tx
                            .query_row(
                                concat!(
                                    "SELECT id FROM ", $table,
                                    " WHERE project_id = ?1 AND content_hash = ?2",
                                    " AND archived_at IS NULL AND id != ?3 LIMIT 1"
                                ),
                                params![&$mem.project_id, &$hash, &$mem.id],
                                |r| r.get(0),
                            )
                            .optional()?;
                        if let Some(other) = other {
                            anyhow::bail!(
                                "update rejected: text is identical to active memory {other} \
                                 in project {}; edits must not silently merge duplicates",
                                $mem.project_id
                            );
                        }
                    }
                    return Err(e.into());
                }
            };
            if affected == 0 {
                // Tombstone semantics: the WHERE now requires
                // `archived_at IS NULL`, so an archived row no longer matches.
                // Distinguish that from a missing id — "archived; restore
                // first" is actionable, a bogus "no row" is not.
                let archived: Option<Option<i64>> = $tx
                    .query_row(
                        concat!(
                            "SELECT archived_at FROM ", $table,
                            " WHERE id = ?1 AND project_id = ?2"
                        ),
                        params![&$mem.id, &$mem.project_id],
                        |r| r.get::<_, Option<i64>>(0),
                    )
                    .optional()?;
                if matches!(archived, Some(Some(_))) {
                    anyhow::bail!(
                        "cannot update {}: memory is archived; restore it first (restore_memory)",
                        $mem.id
                    );
                }
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
            // Unified FTS: delete-then-insert, addressed by the encoded
            // rowid (main rowids are unchanged by UPDATE).
            let main_rowid: i64 = $tx.query_row(
                concat!("SELECT rowid FROM ", $table, " WHERE id = ?1 AND project_id = ?2"),
                params![$mem.id, $mem.project_id],
                |r| r.get(0),
            )?;
            let $fts_rowid: i64 =
                encode_fts_rowid(MemoryKind::$kind_variant, main_rowid);
            $tx.execute(
                "DELETE FROM memories_fts WHERE rowid = ?1",
                params![$fts_rowid],
            )?;
            let (u_title, u_body, u_tags) = $($unified_doc)*;
            $tx.execute(
                crate::storage::schema::UNIFIED_FTS_INSERT,
                params![
                    u_title,
                    u_body,
                    u_tags,
                    Self::project_fts_hex(&$mem.project_id),
                    $kind_str,
                    $fts_rowid,
                ],
            )?;
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
        /// Tombstone semantics: only ACTIVE rows can be hard-deleted here —
        /// an archived (forgotten) memory is protected from silent removal.
        /// Returns false for archived ids (callers treat that as not-found;
        /// `gc_archived` is the only path that physically removes archived
        /// rows, and it owns its guarded SQL).
        pub fn $delete_fn(&self, id: &str, project_id: &str) -> Result<bool> {
            let mut conn = self.conn()?;
            let $tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            // Resolve the main rowid before the main row goes away; the FTS
            // row is addressed by its encoded rowid.
            let main_rowid: Option<i64> = $tx
                .query_row(
                    concat!(
                        "SELECT rowid FROM ", $table,
                        " WHERE id = ?1 AND project_id = ?2 AND archived_at IS NULL"
                    ),
                    params![id, project_id],
                    |r| r.get(0),
                )
                .optional()?;
            let affected = $tx.execute(
                concat!(
                    "DELETE FROM ", $table,
                    " WHERE id = ?1 AND project_id = ?2 AND archived_at IS NULL"
                ),
                params![id, project_id],
            )?;
            if affected > 0 {
                if let Some(rid) = main_rowid {
                    $tx.execute(
                        "DELETE FROM memories_fts WHERE rowid = ?1",
                        params![encode_fts_rowid(MemoryKind::$kind_variant, rid)],
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
        /// Kind-filtered view over the unified index. The AND→OR fallback is
        /// decided ONCE over the whole corpus inside `search_unified` — the
        /// per-type fallback used to re-decide per table, so a query could
        /// run as AND against one kind and OR against another.
        pub fn $search_fn(
            &self, query: &str, project_id: &str, limit: usize,
        ) -> Result<Vec<ScoredMemory<$Struct>>> {
            Ok(self
                .search_unified(query, project_id, Some(MemoryKind::$kind_variant), limit)?
                .$unified_field)
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
