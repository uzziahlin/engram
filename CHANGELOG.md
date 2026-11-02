# Changelog

All notable changes to engram are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-11-02

Iteration rounds 4+5 (two deep reviews): the feedback loop's causal chain
repaired, retrieval moved onto a single unified FTS index, concurrency
contracts pushed down from the transports into the data layer, the
`update_memory` tool contract fixed, the eval harness scaled up, and the CLI
converged onto the same provider path as MCP.

### Fixed - CLI/MCP behavior unification (batch 5)
Round-5 review: 13 CLI commands still bypassed `DefaultMemoryProvider` and
talked to the repository directly — six of them with real behavior forks
(semantic vector-cache/index inconsistency, adoption not accounted, output
field drift, a second merge-patch implementation).

- **The remaining 13 CLI commands route through the provider** (`forget`, `restore`, `forget-batch`, `list-archived`, `consolidate`, `update`, `get`, `recent-failures`, `decisions`, `queries`, `collect`, plus `session-import`/`import` embedding): forget/restore/forget-batch/consolidate now invalidate the semantic vector cache (CLI-side archives used to leave stale vectors until a manual reindex); `update` deletes its second inline merge-patch and inherits batch-3 semantics for free — updating an archived memory fails with "restore it first", an edit colliding with another active row's content hash errors naming the conflict, and semantic builds re-embed the updated text (`--set k=v` keeps its JSON-parse-then-string value semantics); `get` records an adoption event, so CLI reads now count toward `query_stats` adoption rates and `engram tune`; `collect` shares the MCP orchestration (path_guard + allowed_roots + commit dedup) instead of a ~15-line duplicate; `session-import` and `import` index embeddings for what they write (compiled out without the semantic feature — zero cost otherwise).
- **CLI read commands gain the MCP fields** (same renderers now): `recent-failures` includes `prevention` and `tags` (prevention is the core of a failure record — the old CLI renderer hid it), `decisions` includes `related_files` and `tags`, `forget`/`restore` include a machine-readable `status`, `update` echoes `applied`/`updated_at`. `engram queries` output now matches MCP `query_stats` exactly (days window clamped to 0..=3650, `result_count_avg` rounded; the CLI-only per-query `adopted` field is gone — adoption aggregates remain in `engram stats`). `list-archived`/`forget-batch` limits now default to `[retrieval].default_limit` like MCP instead of a CLI-local 20.
- **`optional_str` no longer swallows a following flag as a value** (`--session --dry-run` used to bind "--dry-run" as the session id; the same guard `require_str` already enforced with an error).
- **Near-dup consolidation excludes session-import template scaffold**: the content-template phrases ("User prompts:", "Assistant conclusions (latest last):", "Files touched (…):", bullet dashes, section counts) are now shared constants consumed by BOTH the session-import content builder and the consolidation word-set computation, which drops them for episodic memories. Two templated imports from different sessions sharing ~15 scaffold words with one real word of difference scored raw Jaccard ≈ 0.90 — over the 0.85 threshold — so `consolidate --near` archived one of them; a regression test pins the raw-hazard precondition and the fixed outcome.
- **The 64 MB transcript read is head+tail instead of head-only** (8 MB head for session meta + tail-priority remainder): a session's final conclusions and latest edits live at the END of the transcript, which the head-first cut dropped exactly when it mattered; the skip marker names how many bytes were cut.

### Changed - dead code removed (batch 5)
- `git_integration/listener.rs` renamed to `git.rs` (nothing in it listens — it is polling-free history access); the unused `auto_generate_episodic` / `process_recent_commits` / `estimate_importance` / `extract_tags` (superseded by milestone clustering) and their tests are deleted (~170 lines). `git_collector::collect` (unreferenced outside its own tests) is deleted — the orchestrating `collectors::collect` and the tests use `collect_from_events`. `ConsolidationEngine::jaccard_similarity` is gone as public API; the production comparison (`set_similar`) moved to module scope and the tests exercise it directly.

### Fixed - update_memory tool contract
- **`update_memory` silently no-op'd under strict MCP clients, and reported success anyway**: the inputSchema declared only project_id/memory_type/id, with the patch fields riding on `additionalProperties: true` — strict clients strip keys the schema doesn't declare, so the server received an empty patch and still answered `status: "updated"`, changing nothing but updated_at. Agents read the memory back unchanged, retried with more parameters, and looped (user-visible as parameter amplification; DB evidence: `updated_at == created_at` store-wide — not one update had ever landed; the server-side merge itself was correct, verified over raw JSON-RPC). The schema now declares every patchable field flat (per-type applicability in each field's description — no anyOf/$ref, which most MCP clients can't digest) with `additionalProperties: false`; a patch with nothing to apply is an isError distinguishing three causes (empty patch / only protected keys / fields of another memory_type, listing the applicable fields); a partial match applies the valid fields and reports the rest; and the response echoes `applied` (field → new value, no verification get_memory needed) plus `ignored` (field → reason, omitted when empty). `updated_at` joined the protected keys (it is server-stamped — a patch value for it never took effect and used to be silently overwritten). An anti-drift test pins the schema's declared field set to the model structs' serialized fields, so a new model field that misses the schema fails CI instead of reintroducing the silent-strip bug.

Batch 4 — eval coverage for the remaining dimensions + scale-blind semantic
fusion (2026-10): the eval harness exercised none of graph expansion,
explicit filters, zero-hit behavior, Chinese-content memories, or global
search, and semantic fusion let cosine noise override BM25.

### Changed - semantic fusion (batch 4, `--features semantic` only)
- **Two-arm RRF replaces max-injection in `EmbeddingService::fuse`**: a candidate's relevance becomes `1/(60+bm25_rank) + 1/(60+cosine_rank)` (`fusion::rrf_two_arm`), normalized to [0,1] by the theoretical best `2/k` so a single-signal candidate caps at 0.5. An arm the candidate did not rank in is ABSENT (contributes nothing), not last place — semantics is a vector-only candidate's only signal, symmetrically for BM25-only candidates. Rationale: MiniLM's cosine noise floor (0.25-0.45 for unrelated pairs) sits exactly where calibrated weak BM25 matches land, so the old `max(normalized_bm25, cosine)` let pure noise override real lexical evidence — with semantic enabled, BM25 was effectively off for the whole vector neighborhood. Ranks are scale-blind by construction; the reranker consumes the fused score as its relevance input with unchanged weights, and the default (non-semantic) build is untouched. Unit tests inject synthetic ranks/scores (no model download) and pin the shootout: a 0.42-cosine noise doc that beat a 0.35 BM25 match under max() loses to it under RRF. The internal merged-order tie-break now feeds rrf_fuse the vector ids in COSINE order (they used to come out of a HashMap — random per-process tie-breaks could leak into the caller's stable sort). README gains a "how to manually evaluate semantic retrieval quality" section (reindex → paraphrase queries → compare with semantic on/off), since the real-model path cannot run in CI.

### Added - eval dimensions (batch 4)
- **The golden set grew from 17 to 24 cases and the corpus from 22 to 28 rows**, adding: graph-linked memories (a shared `src/mcp/embedding_service.rs` File entity across two episodic memories and a decision — until now the graph-expansion path was dead code in the harness because every corpus row had empty files/tools) with a case asserting the shared-entity memory surfaces as a flagged `graph_neighbor` below the true hit; three explicit-filter cases (`memory_type=failure`, `tags=[release]`, `before` cutoff excluding the newer match) asserted to return EXACTLY their expected sets; two Chinese-content cases (a direct Chinese query and a pure-Chinese paraphrase with different wording — the corpus previously had zero Chinese rows for CJK queries to hit); and a global `project_id="*"` case with a second project in the corpus, asserting hits carry their owning project. Negative tests pin that a zero-hit query returns empty (not static-prior filler) and that a weak partial match is still returned (recall floor). All expectations are measured, not aspirational.
- **A 30-day longitudinal feedback-loop simulation** (tuning.rs): memories served at rotating ranks → some adopted → a few explicitly endorsed → `tune --apply` every 5 days → serving continues. Pins the batch-1 fixes as a closed loop: adopted memories' importance is monotone non-decreasing and stops at the 0.9 ceiling; never-read mid-tail memories sink exactly to the 0.2 floor and no further; a 'useful'-endorsed memory is never demoted; failure severity never moves (failures_skipped); and a same-day re-search burst does not dilute the adoption rate (the tune right after the burst still promotes, rate computed over distinct days while impressions keep the raw count).

Batch 3 — concurrency contracts moved from the transport layer into the data
layer (2026-10): correctness no longer depends on "the transport happens to
serialize handler calls" (worker_threads > 1 and multiple processes are real
configurations).

### Fixed - concurrency contracts (batch 3)
- **Every CRUD/write transaction now opens with `BEGIN IMMEDIATE`** (macros-generated create/update/delete, `archive`/`restore`/`archive_batch`, `record_adoption`, `create_decision_with_supersede`, `confirm_suggestion`, `gc_archived`, `downgrade_stale_session_imports`, the graph write paths, and the legacy FTS-tags migration). The write lock is taken up front, where `busy_timeout` applies — a deferred transaction upgrading on its first write can return SQLITE_BUSY without honoring it. No call path holds two transactions on two connections at once (entity linking runs inside the caller's transaction).
- **Write-time dedup is now a database constraint, not a check-then-act race** (migration v10): the four main tables gain a `content_hash` column (sha256 over the same dedup text the old SQL expression concatenated — now one set of Rust builders shared by the write path, consolidation, and the backfill) and a PARTIAL UNIQUE INDEX per table on `(project_id, content_hash) WHERE archived_at IS NULL AND content_hash IS NOT NULL`. With `write_dedup` on, the INSERT itself is the guard: `create_*_dedup` maps a unique-index collision to the existing row's id and the tool answers "duplicate" — two concurrent identical creates used to BOTH pass `find_exact_duplicate` (an O(N) full scan on the session-hook hot path) and double-write; now exactly one wins. With `write_dedup` off, rows carry a NULL hash and never participate in the index — duplicates are created freely, by design. Boundary semantics: `restore` of an archived row whose hash collides with an active row WINS over dedup bookkeeping (the restored row's hash is released to NULL, logged); an `update` whose new text hashes onto another tracked ACTIVE row fails with an explicit error naming the conflicting memory (updates are explicit edits and must not silently merge); the v10 backfill hashes ACTIVE rows only (archived stay NULL) and keeps only the EARLIEST of pre-existing duplicates tracked, so creating the index cannot fail on stores that already hold duplicates.
- **The vector cache invalidates across processes**: cached vector sets record the `PRAGMA data_version` observed at load, and every vector search re-reads it (one cheap pragma). A bump (some OTHER connection/process committed — e.g. `engram reindex` in a CLI while the MCP server runs) forces a reload; previously the MCP process held stale vectors indefinitely until an in-process write happened to fire. The `("*", model)` wildcard entry goes through the same path; same-process write invalidation is unchanged (SQLite does not bump data_version for same-connection commits).
- **stdio and HTTP share ONE frame-validation core** (`validate_frame` / `rejection_response` / `handle_isolated` in transport.rs): -32700 vs -32600 classification, batch rejection, the jsonrpc-version check, and panic isolation can no longer drift — the HTTP batch rejection used to surface as an opaque serde type error where stdio printed a clear message, and HTTP's panic responses omitted the panic payload. An equivalence test feeds the same malformed-frame matrix through both transports and asserts identical error codes and messages.
- **`id: null` frames are requests, not notifications** (JSON-RPC 2.0 allows the null id): `Option<Value>` deserialization made it indistinguishable from a MISSING id, so such requests were silently dropped. The id KEY's presence is now authoritative; `id: null` gets a correlated response (`"id": null`).
- **Archived memories are tombstones**: `update_*` and `delete_*` no longer match rows with `archived_at IS NULL` missing — a forgotten memory cannot be silently modified or hard-deleted (to change it, restore it first). Updating an archived memory returns "cannot update …: memory is archived; restore it first" instead of a bogus "no row"; hard delete returns false and leaves the row; `forget` on an already-archived row stays idempotent; `gc_archived` remains the only path that physically removes archived rows.
- **First-use embedding init no longer blocks all concurrent searches on the model download**: `embedder()` was holding its mutex for the whole ~90MB fetch. Double-checked now — the lock is held only to check/clone; the download runs outside both locks with a one-shot gate (latecomers wait for the winner to succeed or fail instead of re-downloading in parallel); the 600s failure backoff is unchanged.

Batch 2 — unified FTS index (2026-10): one BM25 table for all four memory
kinds, whole-corpus IDF, and the retrieval-layer rewrite that follows from it.

### Changed - unified FTS (batch 2)
- **Migration v9 replaces the four per-type FTS tables with ONE unified `memories_fts` table** (contentless, `porter unicode61`, columns `title, body, tags, project, kind`). Per-type tables computed bm25 IDF over four isolated corpora, so raw scores were incomparable across types (retrieval needed a whole cross-type calibration layer) and any term present in >=50% of a SMALL kind's rows scored ~0 (SQLite bm25 IDF has no +1 smoothing — a 3-row kind zeroed half its own vocabulary). One table gives every query a single IDF scale over the whole corpus. Column mapping per kind: title = summary/title/incident/workflow_name; body = content+files / context+rationale+tradeoffs / root_cause+fix+prevention / steps+related_tools; tags unchanged. bm25 column weights are (title 2.5, body 1.0, tags 1.5, project 0, kind 0).
- **Project and kind filtering moved into the MATCH expression** (`AND project:"<sha256-hex>"`, `AND kind:"<type>"` — weight-0 columns, exact index-side filtering instead of post-filtering the ranked stream). The project column stores the project id's SHA-256 hex: a fixed-length `[0-9a-f]` token that survives the tokenizer under any configuration (raw ids could contain CJK or FTS metacharacters); the main-table materialization query still re-checks project/archived defensively. Unified FTS rowids encode `(kind, main rowid)` as `main_rowid * 4 + kind_index` — a contentless table stores no column values, so materialization decodes its way back to the main tables. A query containing a kind name as a business word (e.g. "failure") now also matches that kind's filter column with zero score contribution — recall-widening, not a filter bypass.
- **AND→OR fallback is now decided ONCE over the whole corpus** (inside `search_unified`): the per-type tables re-decided per kind, so one kind could match under AND semantics while another had already fallen back to OR. Behavior change for multi-token queries that AND-match in some kinds but not others.
- **relevance is a per-query monotone normalization**: `raw_bm25 / query_best` (query_best = the best raw score of THIS query's result set; all-zero → all 0). Ordering by relevance is always identical to ordering by the raw score, values are bounded [0,1], and no fixed-scale constant is involved — the old `SIGMOID_SCALE = 2.0` was calibrated against the 22-row eval corpus and saturated as corpora grew. Absolute relevance stays incomparable ACROSS queries (inherent to bm25); ranking and within-result relative scores are what the reranker and agents consume. The per-type `raw / type_best * sigmoid(type_best)` calibration layer and `normalize_bm25` are gone. The query-term coverage signal (0.35 floor) is retained.
- **Index invariant is now "one FTS row per ACTIVE main row"**: `archive`/`archive_batch` delete the FTS row in the same transaction, `restore` re-inserts it, and the v9 backfill + `rebuild_fts` index only `archived_at IS NULL` rows. Previously archived memories kept stale FTS rows that search filtered out at join time — harmless but invisible to `fts_drift`; now drift is an exact active-row gate. `engram maintain`'s FTS repair, `fts_drift`, and the integrity check all target the unified table; the four legacy per-type tables are dropped by v9.
- **Search materialization is batched**: unified hits are decoded per kind and fetched with one `rowid IN (…)` query per kind (500-row chunks), replacing four per-kind searches that each re-ran MATCH; `fetch_by_ids` (vector-only hits, graph neighbors) is likewise one batched query per kind instead of a `get_active_*` round-trip per id. `neighbor_memories`' per-id 4-EXISTS type probe became one IN-batch probe per kind.
- **Eval baseline re-pinned for the unified index** (22-row corpus, 17 cases): MRR 1.000 / NDCG@5 0.997 / hit@3 100% (was 1.000 / 1.000 / 100% under the per-type calibration layer whose sigmoid scale was tuned on this corpus). The only moved case is `cross-type-migration`: the `f-fts5-keywords` distractor now ranks between the two expected ids — the per-type IDF isolation used to suppress it; the golden expectation is unchanged because the relevance judgment is unchanged. New scale/saturation suite pins behavior at 1120 synthetic rows: a unique "zephyr" needle ranks top-1 over ~350 same-topic rows, top-100 relevance spread stays ≥1.1x (measured 1.13; saturation collapse would tie it at 1.0), a word present in ~33% of ALL rows still returns fully-scored results (the per-type small-kind zeroing bug), and project-scoped queries return only that project's rows.

Deep review round 4 (2026-10): four P1 correctness fixes, a latent ranking
bug that had silenced BM25 in cross-type scoring, and the "knowledge
formation" loop (distill / reflection / related_files) actually wired up.

### Fixed - correctness (P1)
- **HTTP transport ran the handler 4-way concurrent on its accept threads**, breaking the serial-handler contract the write paths rely on: two simultaneous identical `create_*` POSTs both passed the write-dedup guard and double-wrote, and `[mcp] worker_threads` was silently ignored (the startup log even reported it). Handler calls now go through the same bounded worker dispatcher as stdio (default 1 = serial); accept threads keep only connection reading/validation, and fast-path methods (`ping`/`tools/list`/…) still answer inline so health probes stay responsive during long tool calls.
- **Global search (`project_id = "*"`)**: the graph-expansion path always returned empty (`r2.project_id = '*'` matched nothing); vector-only semantic candidates from other projects were dropped at materialization (strict project getters); and the `("*", model)` vector cache survived every invalidation for the life of the process - serving stale (even archived) vectors. All three paths now honor the wildcard, materialize through archived-filtering getters (`get_active_*`), and the wildcard cache entry is dropped on any project write.
- **Global search results are actionable now**: every `search_memory` result item carries its owning `project_id` (a `"*"` hit previously could not be fed to `get_memory`/`mark_relevance`/`forget_memory`, which all require the exact project — cross-project knowledge was visible but dead-ended; the field also survives the over-budget detail strip). And graph-expansion neighbors respect the caller's explicit `memory_type`/`tags`/`before` filters with the same semantics as the vector path — a `memory_type=failure` search could surface episodic neighbors sharing a file, and a `before` cutoff could surface newer ones.
- **`create_decision` + `supersedes` is now one transaction** (`create_decision_with_supersede`): a failed supersede used to leave the new decision committed with the chain broken, and the retry then hit the write-dedup guard which returned early WITHOUT performing the supersede. Bad `supersedes` ids now fail before any write; the dedup "duplicate" path performs the missing supersede on the existing record.
- **Feedback-signal integrity**: adoption events are deduplicated per memory per 24h window (one "search -> read -> mark useful" cycle used to write TWO events, and a `get_memory` loop could push `engram tune`'s adoption rate above 1.0); `mark_relevance` has a 24h cooldown per memory (a client loop could previously swing importance to the bounds); graph-expansion neighbors are excluded from query_log impressions (they never matched the query - `tune` systematically demoted them); the tuner caps adoption rate at 1.0.
- **The feedback loop actually closes now** (migration v8: `memory_feedback` table): `mark_relevance`'s 24h cooldown is keyed on explicit feedback rows instead of `adoptions` — the documented search → get_memory → mark_relevance flow always tripped the adoption-based cooldown, so explicit feedback NEVER took effect; 'irrelevant' now shares the same symmetric per-memory cooldown (it previously wrote no cooldown signal at all and could be looped to the importance floor). `engram tune` computes the adoption rate over distinct UTC days on BOTH sides (the numerator was day-deduplicated while the denominator counted every search, so a same-day search→read→re-search cycle could only depress the rate — the most-served memories were the hardest to promote), refuses to demote memories with explicit 'useful' feedback in the window, and no longer auto-adjusts failure severity at all (severity is an operational record of incident impact; skipped failures are reported as `failures_skipped` for manual review).
- **Cross-type BM25 calibration had been contributing exactly zero** since it shipped: `normalize_bm25` negated an already-positivized `type_best` -> sigmoid(0) = 0, so all ranking came from the reranker's static priors. (The 0.6.0 eval baseline of MRR 0.764 was achieved on statics alone.)

### Fixed - design/consistency
- **HTTP transport is actually wired up now** (`[http] enabled = true` starts it; previously `run_http` had no caller and the README/CHANGELOG claims were false). Hardened along the way: constant-time Bearer-token comparison, `WWW-Authenticate` on 401, exact `/mcp` path matching, and `mcp-session-id` sessions (`initialize` mints one, unknown ids -> 404, `DELETE /mcp` terminates).
- **Fresh-DB migration path made real**: the CREATE statements now encode the complete v7 schema (`decision_memories.importance`, `reflection_suggestions.rejected_occurrence_count` were missing), and a fresh database is detected BEFORE any table exists and stamped directly - previously every new DB was misclassified as "legacy" and re-ran the whole migration chain, including two full FTS rebuilds. DDL and FTS-rebuild transactions use `BEGIN IMMEDIATE` (cold-start races no longer hit the deferred-upgrade SQLITE_BUSY path).
- **importance/severity validation moved into the storage layer** (CRUD-macro sanitize on every create/update): the CLI accepted `--importance 99`, and `update_memory` merge-patches merged raw values. NaN maps to the neutral 0.5.
- **One write path**: CLI `create-*` and `ingest` now route through the same provider the MCP tools use (the CLI path previously skipped importance validation AND semantic embedding - CLI-written memories were invisible to semantic search until a manual reindex). CLI `ingest` also validates `repo_path` through path_guard; CLI `collect` passes the configured `allowed_roots` (it silently dropped them). CLI `timeline` uses the repository accessor instead of a second inline SQL implementation.
- **`ingest_commits` accounting**: `skipped_duplicates`/`total_commits_scanned` count against what was actually available (a shallow repo with 5 commits and count=20 reported "15 skipped").
- **`engram maintain` no longer rebuilds FTS unconditionally** - a row-count drift check gates the rebuild (dry-run now writes nothing; previously even dry-run did a full FTS delete+reinsert).
- **stdio transport**: the 16 MiB frame cap is enforced WHILE reading (a multi-GB single line was fully allocated before the check ran); the request queue is bounded (64 frames, then back-pressure) instead of unbounded; batch (top-level array) requests fail with an explicit -32600; CRLF frames accepted; `initialize` negotiates the protocol version (echoes a supported client version) instead of hardcoding.
- **`related_files` returns memories agents can consume**: `{memory_type, id, summary, relation}` per touching memory (the old output listed Memory-entity names, which are bare UUIDs `get_memory` cannot be called with). Unknown files report `found: false` explicitly.
- **Session-import hook races**: deterministic per-(project, session) memory ids make the primary key arbitrate concurrent Stop/SessionEnd firings (the loser refreshes instead of double-creating). Unchanged transcripts are skipped via a (mtime, size) fingerprint - Stop fired on every turn re-read and re-parsed up to 64 MB each time. Skip outcomes are typed errors, not string matching. `install-hooks` shell-quotes the project id (a directory named `My Project` silently bound `--project My`; metacharacters were executed).
- **Storage housekeeping**: `supersede_decision` is a single targeted UPDATE (no full-row rewrite, no FTS/graph rebuild); `record_adoption` writes run in one transaction; `stats_snapshot` no longer holds two pool connections.

### Changed - retrieval quality
- **Query-term coverage signal** in cross-type calibration: a doc's calibrated BM25 score scales with how many of the query's terms it actually contains (0.35 floor), fixing the documented weakness where a single-rare-term hit outranked a multi-term match. `search_by_type` now delegates to `search_by_types` (one normalization path). Minimal stopword list dropped from FTS queries and the coverage denominator. FTS5 per-column bm25 weights (title-ish columns 2.5-3.0x, tags 1.5x, body 1.0x). Deployment intent no longer raises recency (migrate-style queries usually want historical decisions; a 30-day half-life was suppressing exactly those). `plan.sources` dead code and the no-op `[semantic] rrf_k` knob removed (fusion is documented max-injection; configs carrying `rrf_k` still parse).
- **Eval corpus doubled to 22 memories / 17 golden cases**, adding paraphrase queries (no verbatim overlap with targets), distractor memories, and a coverage-sensitive case. New baseline: MRR 1.000 / NDCG@5 1.000 / hit@3 100% (old: 0.764 / 0.825 - measured with BM25 dead). Corpus-size caveat documented: SQLite bm25 IDF has no +1 smoothing, so a term in >=50% of a table's rows scores ~0 - keep type corpora at 6+ diverse rows.
- **Consolidation**: near-duplicate comparison materializes word sets once per record (was two HashSet builds per pair over an inherent O(N^2)); procedural memories compare by step SEQUENCE (LCS) - order-blind Jaccard scored `{test,build,deploy}` vs `{deploy,build,test}` at 1.0 and archived the newer one.

### Added - knowledge-formation loop
- **Distillation has triggers now**: `session_brief` reports the raw-session-import backlog and points at the `engram.distill` prompt; `engram maintain` reports the same backlog, runs the reflection pass on `--apply` (recurring failures became pending suggestions only if someone remembered the `reflect` command), and ages raw session-imports (>90 days, importance -> 0.2 - distilling into typed memories is the promotion path). The generated ENGRAM.md guide covers when to refine.
- **`[reflection] agent_confirm`** (default true): set false to enforce the documented "awaiting human confirmation" model - the MCP `confirm_suggestion` tool then refuses and points at `engram confirm-suggestion`.
- **Semantic robustness**: a failed embedder init (network hiccup during the one-time model fetch) retries every 10 minutes instead of disabling semantic search for the process lifetime; vector scoring borrows cached vectors instead of cloning every blob per query.

### Fixed - install/publish trust
- `install.sh` checksum verification is fail-closed (a missing `.sha256` asset previously installed an unverified binary silently) and `ENGRAM_VERSION=v0.6.0` no longer builds a `/download/vv0.6.0/` URL. `engram-common.sh` refuses to rewrite MCP client configs it cannot parse (it used to overwrite them with `{}`), release/CI build `aarch64-unknown-linux-gnu` on a native arm runner (install.sh accepted the target but no artifact existed - ARM Linux installs 404'd), and release tags run clippy/fmt like CI does.

## [0.6.0] - 2026-09-19

Iteration round 3 — the self-improving loop closes: retrieval quality is
measured (`eval`), adoption evidence is consumed (`tune`), memory is in
context without searching (`brief`), decisions age out (`supersedes`),
writes stop polluting (dedup), and capture spans clients (Codex).

### Added
- **`engram install-hooks` / `uninstall-hooks`**: one-command wiring for automatic memory capture — merges a `SessionEnd` entry invoking `engram hook` into Claude Code settings (project-level by default, `--global` for all projects). Preserves existing hooks, backs the file up to `settings.json.bak`, refuses to touch settings it cannot parse, and is idempotent. `init-guide` offers the same wiring interactively (`--hooks`/`--no-hooks` flags for scripts). `--brief` additionally wires a `SessionStart` entry running `engram brief` (context injection, see below).
- **`session_brief` MCP tool / `engram brief`**: one-call context assembly for session start — key decisions, recent failures, active procedures, and pending reflection suggestions as a compact text brief plus structured sections, within the memory token budget. Optional `focus` also returns top search hits for the task at hand. Decisions superseded by a newer one are excluded (current constraints only). Wire it to Claude Code's `SessionStart` hook via `install-hooks --brief` and memory is in context without the agent searching first.
- **`engram eval`**: retrieval-quality regression harness — seeds a golden engineering corpus into an in-memory DB (never touches the real store) and scores a golden query set (EN + 中文, all four memory types) through the REAL pipeline with the ACTIVE config. Reports per-case rank, MRR, NDCG@5, and primary-hit@3 (`--json` for machine output). Baselines are pinned by unit tests, so ranking changes must move the numbers consciously. The harness already surfaced a known weakness, documented in its tests: on multi-term OR-fallback queries a single-rare-term hit can outrank a lexically stronger match (query-term coverage is the candidate fix for the next ranking iteration).
- **`engram tune`**: closes the adoption feedback loop automatically — memories served ≥3 times with ≥50% adoption rate get +0.1 importance (severity for failures); served ≥5 times and never adopted get −0.1. Bounded (0.2..0.9), dry-run by default, `--apply` writes via the same path `mark_relevance` uses. Backed by a new per-event `adoptions` table (migration-free standalone table): `query_log.adopted` only counted per-query totals and could not say WHICH memory was adopted.
- **Decision supersede chain** (migration v7: `decision_memories.superseded_by`): `create_decision` accepts `supersedes` (CLI: `--supersedes`); the replaced decision stays searchable as history but is flagged (`superseded: true`, `superseded_by: <id>`) and score-halved + re-sorted so it cannot outrank its replacement. `architectural_decisions` excludes superseded records by default (`include_superseded: true` for history); re-superseding re-points the chain head. `update_memory` can also patch `superseded_by` directly.
- **Write-time exact-duplicate guard** (`[retrieval] write_dedup`, default on): `create_*` tools and CLI creates return the existing memory (`status: "duplicate"`) instead of storing a byte-identical copy — agents retry writes, and identical rows were pollution that only `consolidate` would later clean. Byte-exact only (same dedup text consolidation hashes); near-duplicates remain `consolidate --near` territory. Repo-level writers (`session-import` upsert, ingest dedup-by-commit) keep their own idempotency.
- **Codex CLI transcript support**: `session-import`/`engram hook` now parse both Claude Code and Codex CLI session JSONL (format sniffed per line) — user prompts/assistant conclusions from `message` items, file targets from `apply_patch` `*** Update/Add/Delete File:` markers, error evidence from failed `function_call_output`s. Automatic memory capture is no longer Claude-Code-only.

### Changed
- `session-import`/`hook` skip Codex-injected `<user_instructions>`/`<environment_context>` blocks, mirroring the Claude wrapper filter.
- `install-hooks` is idempotent per event (a SessionEnd entry no longer blocks installing a SessionStart entry); `uninstall-hooks` removes both wirings.
- CLI `USAGE` help now lists all dispatched commands (it previously omitted `get`/`maintain`/`stats`/`backup`/`export`/`import`).
- `record_adoption` records the memory type alongside the event (feeds `engram tune`).

## [0.5.0] - 2026-09-08

Iteration round 2 — feedback loop, write quality, ops, and engineering-debt payoff.

### Added
- **Adoption feedback loop**: `get_memory` records which search results were actually read (24h window), `query_stats` reports adoption counts/rates, and the new `mark_relevance` MCP tool lets the agent flag a result `useful` (+0.1 importance) or `irrelevant` (−0.1) — importance now evolves with real usage (migration v5 adds `query_log.adopted`).
- **`engram stats`**: one-glance observability — memory counts per type/project, entity graph size, reflection states, retrieval feedback (query volume, zero-hit and adoption rates, top adopted/unanswered queries).
- **`engram backup / export / import`**: SQLite hot backup (safe while the server runs, retention `--keep N`), and idempotent-by-id JSON export/import.
- **HTTP MCP transport** (`[http]` config, off by default): Streamable-HTTP subset on tiny_http (no async runtime) — `POST /mcp` with mandatory Bearer token; startup refuses an empty token.
- **`engram.distill` prompt**: refines raw `session-import` memories into clean typed memories (decisions / failures with root cause / procedures) with quality bars, then archives the originals.
- **Transcript error extraction**: `session-import`/`hook` now captures `is_error` tool results as failure evidence (`has-errors` tag) — raw material for failure memories and the reflection engine.
- **Cross-project global search**: `project_id = "*"` on `search_memory` spans all projects (global knowledge); semantic + graph paths honor it too.
- **`--features jieba`**: word-level Chinese tokenization (embedded jieba dictionary). A `meta` table records which tokenizer built the FTS index; switching builds triggers exactly one automatic rebuild.
- Transport test suite: the stdio transport (`run_transport`, now reader/writer-generic) is directly tested — parse vs invalid-request distinction, jsonrpc version check, notifications, panic isolation, the 16 MiB frame cap, multi-worker dispatch.
- Semantic stub tests: a deterministic embedder exercises fuse (vector-only materialization, cosine injection, filter enforcement) and the vector cache in CI — no model download needed.

### Changed
- **Contentless FTS5** (migration v6): shadow tables store the inverted index only — the document text is no longer duplicated, halving database size; search joins by rowid.
- **Cross-type BM25 calibration**: each type's scores are rescaled to `raw/type_best × sigmoid(type_best)`, so a weak match in one table no longer outranks a strong match in another.
- **decision/procedural importance** is a real field now (migration v4; tools accept `importance`, failure feedback maps to severity) — the reranker weight is no longer dead for those types.
- **Token budget covers the `results` array** too: over budget, detail fields are stripped from the lowest-ranked results first (`detail_omitted: true`, re-fetchable via `get_memory`); previously only the `context` field was bounded.
- Semantic: the embedding model loads lazily on first use (the ~90 MB first-run download no longer blocks the MCP initialize handshake), and vectors are cached per project with write-time invalidation (was: full reload + deserialize on every query).

### Engineering debt
- **MSRV bumped 1.75 → 1.85**: the dependency tree (candle/tokenizers/jieba → rand 0.9/0.10) now pulls edition-2024 manifests that cargo 1.75 cannot even parse under `--locked`; pinning dozens of transitive crates was judged unmaintainable. 1.85 is the first edition-2024-capable stable.
- `repository.rs` split into `schema.rs` (DDL/migrations/tokenization), `graph.rs` (entities/relations/neighbors), `embeddings.rs` (vector store) — 4,600 lines down to 3,200 in the core file.
- Composer dead code removed (`ContentType`/`detect_content_type`).
- CI: new `test-features` job runs the jieba + semantic test suites.

## [0.4.0] - 2026-09-08
## [0.4.0] - 2026-09-08

Deep-review fix batch (2026-09-08 全链路审查, see `docs/deep-review-2026-09.md`) plus the first feedback-loop / write-automation / graph-retrieval iterations.

### Added
- **`engram hook`**: Claude Code hook entry point — reads the Stop/SessionEnd hook payload from stdin (`transcript_path`, `session_id`, `cwd`) and distills the transcript into a memory via session-import. Empty sessions report `skipped`, not an error. Makes memory formation zero-config (see README §Integration).
- **Session-import is idempotent per session**: re-importing the same `(project, session)` refreshes the existing episodic memory in place (upsert) instead of stacking near-duplicates — a `Stop` hook firing every turn now converges to exactly one memory that tracks the session as it grows. `SessionEnd` remains the recommended one-shot wiring.
- **Graph-based retrieval**: `search_memory` now appends up to 2 memories that share an entity (file/tool) with the top hits, marked `graph_neighbor: true` at fixed low rank — the relationship graph finally participates in search. Toggle with `[retrieval].graph_expansion` (default true).
- **Knowledge-gap report** in `engram maintain`: frequent queries that consistently return zero results (the cheapest signal for what memory to write next).
- **Orphan entity GC**: `engram gc --apply` now also deletes File/Tool entities no relation references (previously unbounded growth + `related_files` noise); `maintain` previews the count.
- **`[security] allowed_roots`** config: restrict `ingest_commits`/`collect_sources` `repo_path` to configured roots.
- MCP transport: JSON-RPC `jsonrpc` version validation, and `ping`/`tools/list`/`prompts/list`/`initialize` are answered inline so a long tool call cannot starve client health probes.

### Changed
- **Ranking math rebuilt**: weights normalized to sum 1.0 (relevance dominant at 0.5), BM25 sigmoid rescaled to the real FTS5 score range, type-prior spread narrowed, and intents now raise *relevance* instead of only the query-independent priors — previously the static `failure>decision>episodic>procedural` prior effectively decided cross-type ranking regardless of the query.
- **CJK search fixed for mixed-script text**: the preprocessor now splits CJK↔latin boundaries (`用Rust写` used to index as one unmatchable token) and preprocesses tags/files columns; schema migration v2 rebuilds existing FTS indexes with the new tokenization.
- **Multi-token queries try AND first**, falling back to OR on empty results (was always-OR: "rust memory system" matched anything containing "system").
- **FTS rowid alignment** (schema migration v2): FTS rows now share the main table's rowid, making update/delete FTS maintenance O(log n) instead of a full index scan per operation (verified via EXPLAIN QUERY PLAN).
- **git ingest is lazy**: the rev walk sorts by commit time and stops after N — the entire history (100k+ commits on monorepos) is no longer decoded to fetch the newest 20.
- **CLI `search` uses the full MCP pipeline** (intent → plan → rerank → optional semantic fuse); it was a bare-BM25 fork that ranked differently from the MCP tool.
- Milestone clustering splits themes on >30-day gaps: years-apart `fix` commits no longer merge into one meaningless "milestone".
- Tool failures now return MCP `result.isError = true` (business errors) or `-32602` (bad params / unknown tool) instead of blanket `-32603`; a broken config file is fatal for the MCP server too (it used to silently fall back to the default database — splitting data across two stores).
- Semantic (feature build): cosine scores are injected into relevance (they were computed then discarded), explicit `memory_type`/`tags`/`before` filters apply to vector-only hits, `update_memory` re-embeds changed text, embeddings truncate at the model's 512-token limit (long inputs used to OOM/near-hang), and model changes with zero vectors now point at `engram reindex`.
- Collector heuristics: TODO/NOTE no longer flood the annotation budget; py/sh/sql/lua `#`/`--` decision comments are recognized; CHANGELOG "Fixed" sections match exact words; repo-root `.circleci`/`.buildkite` configs are found; conventional-commit types are whitelist-checked; fix-detection scans the subject only; file collection walks are budgeted (entries/depth/time), deterministic (sorted), and read via streaming truncation (a multi-GB file can no longer OOM the collector).
- `repo_path` from MCP clients is validated (must exist, be a directory, not the home directory or `/`; optional allowlist above).

### Fixed
- `get_*`/`update_*` enforce `project_id` in SQL (isolation no longer depends on every caller remembering to check); update can no longer move a memory across projects.
- `limit` clamped to 1..=1000 and `days` to 0..=3650 on every tool (`limit=1000000` previously flowed into SQL; extreme `days` overflowed the timestamp arithmetic).
- Empty queries and typo'd `memory_type` values are loud errors instead of silent empty results; `prompts/get` validates required arguments.
- Startup no longer runs two full-table dedup writes on every command (one-time, index-existence guarded).
- Reflection re-arm: rejecting a proposal no longer silences the tag forever — it re-proposes after `min_occurrences` NEW failures (migration v3).
- near-dup consolidation pre-filters by length ratio before building word sets (the O(n²) hot path), and stale "time-window merging" docs corrected.

## [0.3.0] - 2026-09-07

Read-path completion, lifecycle automation, and a deep-review fix batch.

### Added
- **`get_memory` MCP tool** (and `engram get` CLI): fetch one memory's complete record by id — structured fields (root_cause/fix/prevention, context/rationale/tradeoffs, steps) were previously write-only.
- **`search_memory` returns full payloads**: results now carry each memory's type-specific fields plus tags, so agents read knowledge in one call instead of a teaser summary. New `tags` (any-match) and `before` filters; CLI `search` gains `--tag`/`--before`.
- **`engram session-import`**: distills a Claude Code session transcript (JSONL) into one episodic memory — wire it to a Stop/SessionEnd hook for zero-effort memory formation (see `docs/claude-code-setup.md` §7).
- **`engram maintain`**: one-shot health pass — exact-duplicate consolidation (soft-archive), FTS index rebuild (repairs orphans/missing rows/re-applies CJK preprocessing), `query_log` retention prune (new `[storage].query_log_retention_days`, default 90), staleness report (`--repo`), and a GC preview. Never physically deletes.
- **`rebuild_fts`** repository method powering the repair above.
- MCP `ping` method (liveness probe required by newer clients).

### Changed
- **Intent routing is now soft**: every search covers all four memory types; classified intent only adjusts reranker weights. Hard source-narrowing cost recall (a Workflow query could no longer surface a decision record).
- **Unified ranking**: the context composer no longer re-sorts by fixed type priority — it preserves the reranker's order, so `results` and `context` agree within one response.
- **`ingest_commits` clusters into milestones**: one episodic memory per Conventional-Commit (type, scope) theme instead of one per commit (the noisy output git_collector's own docs advise against). Dedup now happens *before* memory generation; count clamped to 1000 (CLI parity).
- `recent_failures` / `architectural_decisions` return complete records (root_cause/fix/prevention, context/tradeoffs).
- Tool `limit` defaults now come from `[retrieval].default_limit` (was a hardcoded protocol default of 10).
- Release profile uses `panic = "unwind"` + `catch_unwind` isolation in the stdio transport, so one panicking request returns a JSON-RPC internal error instead of killing the MCP server. (0.2.0's changelog claimed this; the code now matches.)
- `serverInfo.version` from `CARGO_PKG_VERSION` (was hardcoded "0.1.0").
- collect: one shared git history walk for the git + failures dimensions (was walked twice); commit handles are reused instead of re-loaded for the newest N.
- Keyword heuristics (ingest tags/importance, fix-commit detection) match on word boundaries — "fixture" no longer counts as "fix", "docker" as "docs".
- Collector classification (workflow/docs kinds) matches on repo-relative paths — the checkout location no longer skews buckets.
- CLI `update --set` parses values as JSON when possible (numbers/bools/arrays work; everything used to fail type-check as strings) and warns on unknown keys.
- Dead config knobs removed (`storage.wal_mode`, `mcp.transport`, `[graph]`); `mcp.worker_threads` is now validated (1–64).
- GC batches deletes per memory kind in one transaction (was one commit per row) and `per_type` now reports actually-deleted counts.

### Fixed
- **FTS `tags` migration backfill skipped CJK preprocessing**, making migrated Chinese content permanently unsearchable; backfill now goes through the same preprocessing as the write path.
- **Updating a nonexistent id created orphan FTS rows** and silently returned success; updates now fail loudly and skip the index write.
- **Updates didn't refresh entity links** — stale `related_files` edges for removed files/tools; update now rebuilds the memory's graph edges.
- **`confirm_suggestion` was not atomic** (three pooled connections, two transactions): a crash between them permanently wedged the suggestion at `pending` with a PK conflict on retry. Now one transaction, idempotent under retries; empty-step suggestions are refused.
- **Reflection suggestions resurrected**: `reject` no longer lets the same proposal come back on the next `reflect --apply`, and confirmed rules are not re-proposed (block on any prior suggestion for the tag).
- **`get_ingested_commits` counted archived memories**, permanently blocking re-import of their commits after archiving.
- Legacy DBs with duplicate graph relations no longer fail startup at `CREATE UNIQUE INDEX` (pre-index dedup, mirroring the reflection-suggestions pattern).
- CLI `search` (all types) no longer swallows database errors as empty results, and merges results across types by score instead of fixed type order.
- CLI `timeline`/`queries` use saturating day arithmetic (debug builds panicked on huge `--days`); `optional_num` rejects NaN/inf.
- `engram init` warns on a broken config instead of silently using defaults.
- CJK character detection now covers Hangul.

## [0.2.0] - 2026-07-20

Open-source readiness pass — see `docs/开源就绪修复计划.md` for the full plan.

### Added
- `SECURITY.md` — vulnerability disclosure policy, response SLA, and a documented threat model for a local-first MCP server.
- Supply-chain governance: `deny.toml` (cargo-deny) gating advisories/licenses/bans, plus a CI `audit` job.
- Dependabot configuration for Cargo dependencies and GitHub Actions.
- Declared Minimum Supported Rust Version (`rust-version = "1.75"` in `Cargo.toml`) verified by a dedicated CI `msrv` job.
- `rust-toolchain.toml` pinning the development toolchain.
- `CHANGELOG.md` and `CODE_OF_CONDUCT.md`.

### Changed
- Documentation aligned with reality: MCP tool count is 22 (was 18); the relationship graph is documented as SQLite-backed (indexed adjacency via `related_files`), not an in-memory petgraph engine.
- CI build matrix aligned with the release matrix (added `macos-13` / `x86_64-apple-darwin`).
- The Release workflow now runs `cargo test` before building any artifacts.

### Fixed
- **GC no longer deletes live memories**: physical deletion now requires `archived_at IS NOT NULL`, closing a race where a memory restored between GC candidate collection and deletion was silently destroyed.
- **Reflection uniqueness**: `reflection_suggestions` enforces uniqueness on `(project_id, pattern_tag)` for pending rows, preventing duplicate suggestions under concurrent reflection.
- **Removed dead code**: the in-memory `GraphEngine` (petgraph) module and its dependency — relationship data lives in SQLite tables.
- **Hardened test coverage**: `sanitize_fts_query` now has direct injection-vector unit tests; the misleadingly-named rollback test was fixed or renamed.
- DoS hardening: bounded request body size, clamped tool-input lengths/counts, and `catch_unwind` isolation so a future panic cannot abort the server.

## [0.1.0] - 2026-06

Initial public preview.

### Added
- SQLite + FTS5 storage with four memory types: Episodic, Decision, Failure, Procedural.
- MCP server over stdio transport.
- BM25 retrieval with intent classification, intent-based source routing, and reranking (recency + importance + type-prior, configurable).
- Relationship graph (entities + relations) powering `related_files`.
- Git integration (auto-ingest commits via pure-Rust `gix`).
- Project bootstrap (`collect_sources` + `engram.bootstrap` prompt).
- CLI interface.
- Memory lifecycle: `forget` / `restore` / `update` / `forget_batch` / `consolidate` (soft-delete + dedup).
- Reflection engine (failure → procedural suggestions, gated behind pending review).
- Embedding-based semantic search (candle + Reciprocal Rank Fusion, behind the opt-in `semantic` feature).
- Retrieval feedback loop (`query_log`, `query_stats`).
