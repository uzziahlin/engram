use crate::config::Config;
#[cfg(feature = "semantic")]
use crate::retrieval::bm25::SearchResult;
#[cfg(feature = "semantic")]
use crate::storage::repository::MemoryRepository;
#[cfg(feature = "semantic")]
use anyhow::Result;
#[cfg(feature = "semantic")]
use serde::Serialize;
#[cfg(feature = "semantic")]
use std::collections::HashSet;

/// Result of a `reindex_embeddings` run; serialized as the CLI's JSON output.
#[cfg(feature = "semantic")]
#[derive(Debug, Default, Serialize)]
pub struct ReindexReport {
    pub total: usize,
    pub embedded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub dry_run: bool,
}

pub struct EmbeddingService {
    #[cfg(feature = "semantic")]
    embedder: std::sync::Mutex<EmbedderSlot>,
    /// One-time initialization gate. The ~90MB model download happens OUTSIDE
    /// `embedder`'s mutex (double-checked: lock `init_lock`, re-check the
    /// slot, download, then take `embedder` to publish) so concurrent searches
    /// don't queue on the model fetch — they queue HERE instead, waiting for
    /// the winner to succeed or fail (never re-downloading in parallel).
    /// Lock order is always `init_lock` → `embedder`, never inverted.
    #[cfg(feature = "semantic")]
    init_lock: std::sync::Mutex<()>,
    #[cfg(feature = "semantic")]
    lazy_config: Box<crate::config::SemanticConfig>,
    /// Per-(project, model) vector cache: brute-force cosine used to
    /// re-deserialize every vector from SQLite on EVERY query (~15 MB per
    /// 10k memories). Entries are evicted wholesale on any write — coarse,
    /// but obviously correct (materialization drops ids that are gone or
    /// archived; see `get_active_*`).
    #[cfg(feature = "semantic")]
    vector_cache: std::sync::Mutex<VectorCache>,
}

/// Lazy embedder slot with bounded retry: a failed init (network hiccup
/// during the one-time model fetch, corrupted download) must not disable
/// semantic search for the whole process lifetime — retry at most once per
/// [`RETRY_INTERVAL_SECS`] instead.
#[cfg(feature = "semantic")]
struct EmbedderSlot {
    embedder: Option<std::sync::Arc<dyn crate::retrieval::embedding::EmbeddingProvider>>,
    next_retry_at: i64,
}

#[cfg(feature = "semantic")]
const RETRY_INTERVAL_SECS: i64 = 600;

/// One cached vector with its memory identity: (memory_id, memory_type, vec).
#[cfg(feature = "semantic")]
pub type VectorEntry = (String, String, Vec<f32>);

/// One cached vector set: the project's active vectors plus the
/// `PRAGMA data_version` observed when they were loaded.
#[cfg(feature = "semantic")]
struct CachedVectors {
    /// `PRAGMA data_version` of the connection that loaded `vectors`. SQLite
    /// bumps this on a connection when ANOTHER connection commits — and leaves
    /// it unchanged for same-connection commits. It is therefore a free
    /// cross-process invalidation signal (a CLI process writing the DB is seen
    /// by the MCP server's next search), complementing — not replacing — the
    /// in-process `invalidate_vectors` (same-process writes do NOT bump it).
    data_version: i64,
    vectors: std::sync::Arc<Vec<VectorEntry>>,
}

/// (project_id, model_id) → the project's cached active vectors.
/// The `("*", model)` wildcard entry goes through the exact same path, so the
/// data_version check covers global search too.
#[cfg(feature = "semantic")]
type VectorCache = std::collections::HashMap<(String, String), CachedVectors>;

impl EmbeddingService {
    pub fn new(config: &Config) -> Self {
        #[cfg(feature = "semantic")]
        {
            Self {
                embedder: std::sync::Mutex::new(EmbedderSlot {
                    embedder: None,
                    next_retry_at: 0,
                }),
                init_lock: std::sync::Mutex::new(()),
                lazy_config: Box::new(config.semantic.clone()),
                vector_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }
        #[cfg(not(feature = "semantic"))]
        {
            let _ = config;
            Self {}
        }
    }

    /// Test seam: build the service around a stub embedder (no model files,
    /// deterministic vectors) so the semantic path is testable in CI.
    #[cfg(feature = "semantic")]
    pub fn with_embedder(
        embedder: Box<dyn crate::retrieval::embedding::EmbeddingProvider>,
    ) -> Self {
        Self {
            embedder: std::sync::Mutex::new(EmbedderSlot {
                embedder: Some(std::sync::Arc::from(embedder)),
                next_retry_at: i64::MAX,
            }),
            init_lock: std::sync::Mutex::new(()),
            lazy_config: Box::new(crate::config::SemanticConfig::default()),
            vector_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Lazily resolve the embedder: model download + load happens on FIRST
    /// USE, not at server startup — it used to block the MCP initialize
    /// handshake for the whole ~90 MB first-run fetch.
    ///
    /// Double-checked: the fast path holds `embedder` only to clone the Arc;
    /// the slow path downloads OUTSIDE both locks. Concurrent first searches
    /// serialize on `init_lock` — the first caller fetches the model while
    /// the others WAIT, then observe either the finished embedder or the
    /// failure's retry backoff (never a second download). A failure keeps
    /// the existing [`RETRY_INTERVAL_SECS`] backoff.
    #[cfg(feature = "semantic")]
    fn embedder(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::retrieval::embedding::EmbeddingProvider>> {
        {
            let slot = self.embedder.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(embedder) = slot.embedder.clone() {
                return Some(embedder);
            }
            if chrono::Utc::now().timestamp() < slot.next_retry_at {
                return None; // a recent init failed: back off instead of retrying
            }
        }
        // Slow path: exactly one thread downloads; latecomers block here
        // until it finishes (succeeds or fails), then re-check below.
        let _init_gate = self.init_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut slot = self.embedder.lock().unwrap_or_else(|e| e.into_inner());
        // Re-check after the wait: the winner may have succeeded…
        if let Some(embedder) = slot.embedder.clone() {
            return Some(embedder);
        }
        // …or failed and armed the backoff while we were queued.
        if chrono::Utc::now().timestamp() < slot.next_retry_at {
            return None;
        }
        let initialized = Self::init_embedder(&self.lazy_config);
        if initialized.is_none() {
            // Failed: back off instead of re-attempting (and re-fetching) on
            // every search call.
            slot.next_retry_at = chrono::Utc::now().timestamp() + RETRY_INTERVAL_SECS;
        }
        slot.embedder = initialized;
        slot.embedder.clone()
    }

    /// Drop cached vectors for a project (any write that can change the
    /// active vector set). `None` = drop everything (reindex, model change).
    ///
    /// A specific project's write ALSO drops the `("*", model)` global-search
    /// entry: the wildcard key never equals a real project id, so before this
    /// fix the global cache survived every invalidation and served stale
    /// (even archived) vectors for the life of the process.
    #[cfg(feature = "semantic")]
    pub fn invalidate_vectors(&self, project: Option<&str>) {
        let mut cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
        match project {
            Some(p) => {
                cache.retain(|(proj, _), _| proj != p && proj != "*");
            }
            None => cache.clear(),
        }
    }

    /// Load the active vectors for (project, model), through the cache.
    ///
    /// Cross-process invalidation: `PRAGMA data_version` is read on a pooled
    /// connection before every lookup (one cheap pragma, no query). A value
    /// different from the cached entry's means SOME other connection/process
    /// committed since the load — reload instead of serving stale vectors
    /// (e.g. an `engram reindex` in a CLI process that the long-lived MCP
    /// server would otherwise never observe). Same-process writes are handled
    /// by `invalidate_vectors`, as before.
    #[cfg(feature = "semantic")]
    fn cached_vectors(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
        model_id: &str,
    ) -> Result<std::sync::Arc<Vec<VectorEntry>>> {
        let key = (project_id.to_string(), model_id.to_string());
        let data_version: i64 = {
            let conn = repo.connection()?;
            conn.query_row("PRAGMA data_version", [], |r| r.get(0))?
        };
        {
            let cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&key) {
                if hit.data_version == data_version {
                    return Ok(std::sync::Arc::clone(&hit.vectors));
                }
                // Stale: another connection committed since the load — reload.
            }
        }
        let loaded: Vec<VectorEntry> = repo.load_active_embeddings(project_id, model_id)?;
        let arc = std::sync::Arc::new(loaded);
        let mut cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(
            key,
            CachedVectors {
                data_version,
                vectors: std::sync::Arc::clone(&arc),
            },
        );
        Ok(arc)
    }

    /// Build the embedding model from config (semantic enabled + model available).
    /// Returns None (logging a warning) on any failure so the server still runs
    /// in pure-BM25 mode.
    #[cfg(feature = "semantic")]
    fn init_embedder(
        config: &crate::config::SemanticConfig,
    ) -> Option<std::sync::Arc<dyn crate::retrieval::embedding::EmbeddingProvider>> {
        use crate::retrieval::embedding::{ensure_model, CandleBertEmbedder};
        if !config.enabled {
            return None;
        }
        let dir = config.model_path.clone().unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_default()
                .join(".engram/models")
                .join(config.model_id.rsplit('/').next().unwrap_or("model"))
        });
        // Only auto-fetch when no explicit path was given (air-gapped users
        // provide model_path and we never touch the network).
        if config.model_path.is_none() {
            if let Err(e) = ensure_model(&config.model_id, &dir) {
                tracing::warn!(
                    "semantic enabled but model fetch failed ({e}); retrying later, BM25-only meanwhile"
                );
                return None;
            }
        }
        match CandleBertEmbedder::from_local(&dir, &config.model_id) {
            Ok(e) => {
                tracing::info!("semantic search enabled: model {}", config.model_id);
                Some(std::sync::Arc::new(e))
            }
            Err(e) => {
                tracing::warn!(
                    "semantic model load failed ({e}); retrying later, BM25-only meanwhile"
                );
                None
            }
        }
    }

    /// Embed `text` and store the vector for `id`. No-op when the embedder is
    /// absent. Failures are logged, never fatal (embedding is an enhancement).
    #[cfg(feature = "semantic")]
    pub fn index(
        &self,
        repo: &MemoryRepository,
        memory_type: &str,
        id: &str,
        project_id: &str,
        text: &str,
    ) {
        if let Some(e) = self.embedder() {
            match e.embed(&[text]) {
                Ok(v) if !v.is_empty() => {
                    if let Err(err) = repo.upsert_embedding(
                        id,
                        memory_type,
                        project_id,
                        &v[0],
                        e.model_id(),
                        e.dim(),
                    ) {
                        tracing::warn!("upsert_embedding failed for {id}: {err}");
                    }
                    self.invalidate_vectors(Some(project_id));
                }
                Ok(_) => {}
                Err(err) => tracing::warn!("embed failed for {id}: {err}"),
            }
        }
    }

    /// Embed one memory during reindex, updating `report`. Skips when already
    /// embedded (unless `force`); counts only when `dry_run`. Per-memory errors
    /// are logged and counted as `failed`, never aborting the batch.
    #[cfg(feature = "semantic")]
    #[allow(clippy::too_many_arguments)]
    fn reindex_one(
        &self,
        repo: &MemoryRepository,
        report: &mut ReindexReport,
        already: &HashSet<String>,
        memory_type: &str,
        id: &str,
        project_id: &str,
        text: String,
        force: bool,
        dry_run: bool,
    ) {
        report.total += 1;
        if !force && already.contains(id) {
            report.skipped += 1;
            return;
        }
        if dry_run {
            report.embedded += 1; // would embed
            return;
        }
        let embedder = self
            .embedder()
            .expect("embedder presence is checked by reindex_embeddings");
        match embedder.embed(&[text.as_str()]) {
            Ok(v) if !v.is_empty() => {
                match repo.upsert_embedding(
                    id,
                    memory_type,
                    project_id,
                    &v[0],
                    embedder.model_id(),
                    embedder.dim(),
                ) {
                    Ok(()) => report.embedded += 1,
                    Err(e) => {
                        tracing::warn!("reindex upsert failed for {id}: {e}");
                        report.failed += 1;
                    }
                }
            }
            Ok(_) => {
                tracing::warn!("reindex embed returned empty for {id}");
                report.failed += 1;
            }
            Err(e) => {
                tracing::warn!("reindex embed failed for {id}: {e}");
                report.failed += 1;
            }
        }
    }

    /// Backfill embeddings for active memories. `project=None` = all projects.
    /// Default skips memories already embedded for the current model; `force`
    /// re-embeds all. `dry_run` counts without writing. Errors when the embedder
    /// is absent (semantic disabled or model load failed).
    #[cfg(feature = "semantic")]
    pub fn reindex(
        &self,
        repo: &MemoryRepository,
        project: Option<&str>,
        force: bool,
        dry_run: bool,
    ) -> Result<ReindexReport> {
        let embedder = self.embedder().ok_or_else(|| {
            anyhow::anyhow!(
                "semantic search is not active: enable [semantic] in ~/.engram/config.toml \
                 and ensure the model is available"
            )
        })?;
        let model_id = embedder.model_id().to_string();
        let already = if force {
            HashSet::new()
        } else {
            repo.embedded_ids(project, &model_id)?
        };

        let mut report = ReindexReport {
            dry_run,
            ..Default::default()
        };
        for m in repo.list_active_episodic(project)? {
            self.reindex_one(
                repo,
                &mut report,
                &already,
                "episodic",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in repo.list_active_decision(project)? {
            self.reindex_one(
                repo,
                &mut report,
                &already,
                "decision",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in repo.list_active_failure(project)? {
            self.reindex_one(
                repo,
                &mut report,
                &already,
                "failure",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in repo.list_active_procedural(project)? {
            self.reindex_one(
                repo,
                &mut report,
                &already,
                "procedural",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        if !dry_run {
            self.invalidate_vectors(None);
        }
        tracing::info!(
            "reindex: total={} embedded={} skipped={} failed={} dry_run={}",
            report.total,
            report.embedded,
            report.skipped,
            report.failed,
            report.dry_run
        );
        Ok(report)
    }

    /// Semantic fusion: blend vector top-K with the BM25 candidates.
    /// Vector-only hits are materialized (through the archived-filtering,
    /// wildcard-aware `fetch_by_ids`) so they can surface even when BM25
    /// missed them entirely.
    ///
    /// Fusion semantics — two-arm RRF: a candidate's relevance becomes
    /// `fusion::rrf_two_arm(bm25_rank, cosine_rank)` — the reciprocal-rank
    /// fusion of the BM25 ranking (the caller's input order) and the vector
    /// ranking, each arm contributing only when the candidate actually
    /// ranked in it (absent arm = missing evidence, not last place —
    /// semantics is a vector-only candidate's ONLY signal). Normalized to
    /// [0,1] with a single-signal ceiling of 0.5 (see `rrf_two_arm`).
    /// Rationale: the previous max-injection (`max(normalized_bm25,
    /// cosine)`) was scale-blind in the worst way — MiniLM's cosine noise
    /// floor (0.25-0.45 for unrelated pairs) sits exactly where calibrated
    /// weak BM25 matches land, so enabling semantic search effectively
    /// switched BM25 off for the whole vector neighborhood. Ranks cannot
    /// override each other's scale; they only add. The reranker consumes
    /// this as its relevance input with unchanged weights.
    ///
    /// Explicit filters (`memory_type`/`tags`/`before`) apply to vector-only
    /// candidates too — the vector path must not bypass them.
    ///
    /// Returns `bm25` unchanged when the embedder is absent or embedding fails.
    #[cfg(feature = "semantic")]
    #[allow(clippy::too_many_arguments)]
    pub fn fuse(
        &self,
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        bm25: Vec<SearchResult>,
        top_k: usize,
        memory_type: Option<&str>,
        tags: &[String],
        before: Option<i64>,
    ) -> Result<Vec<SearchResult>> {
        use std::collections::HashMap;

        let Some(embedder) = self.embedder() else {
            return Ok(bm25);
        };
        let query_vec = match embedder.embed(&[query]) {
            Ok(mut v) if !v.is_empty() => v.remove(0),
            Ok(_) => return Ok(bm25),
            Err(err) => {
                // Degradation must be observable, not a silent BM25-only path.
                tracing::warn!("semantic query embed failed, BM25-only fallback: {err}");
                return Ok(bm25);
            }
        };

        let loaded = self.cached_vectors(repo, project_id, embedder.model_id())?;
        if loaded.is_empty() {
            // Likely a model_id change stranded all vectors: tell the user
            // how to recover instead of silently degrading to BM25.
            tracing::warn!(
                "semantic active but project {project_id:?} has no embeddings for model {}; \
                 run `engram reindex` to (re)build them",
                embedder.model_id()
            );
            return Ok(bm25);
        }
        let type_of: HashMap<String, String> = loaded
            .iter()
            .map(|(id, ty, _)| (id.clone(), ty.clone()))
            .collect();
        // Borrow the vectors instead of cloning every one (the old path
        // duplicated the whole project's blobs per query). `cosine_ranked`
        // keeps top_k_cosine's descending order — the vector ARM ranking.
        let cosine_ranked = crate::retrieval::vector::top_k_cosine(
            &query_vec,
            loaded.iter().map(|(id, _, v)| (id.as_str(), v.as_slice())),
            top_k,
        );

        // Materialize vector-only candidates that survive the caller's
        // explicit filters, exactly like the BM25 side did.
        let bm25_ids: Vec<String> = bm25.iter().map(|r| r.id.clone()).collect();
        let mut by_id: HashMap<String, SearchResult> =
            bm25.into_iter().map(|r| (r.id.clone(), r)).collect();
        let missing: Vec<(String, String)> = cosine_ranked
            .iter()
            .map(|(id, _)| id)
            .filter(|id| !by_id.contains_key(*id))
            .filter_map(|id| type_of.get(id).map(|ty| (ty.clone(), id.clone())))
            .collect();
        for sr in crate::retrieval::bm25::BM25Retriever::fetch_by_ids(repo, &missing, project_id)? {
            let type_ok = memory_type.is_none_or(|mt| sr.memory_type == mt);
            let tags_ok = tags.is_empty() || sr.tags.iter().any(|t| tags.contains(t));
            let before_ok = before.is_none_or(|b| sr.created_at < b);
            if type_ok && tags_ok && before_ok {
                by_id.entry(sr.id.clone()).or_insert(sr);
            }
        }

        // RRF relevance: the BM25 arm ranks by the caller's input order
        // (BM25-sorted — filters retain order), the vector arm by cosine
        // order. Candidates absent from an arm keep the other arm alone.
        let bm25_rank_of: HashMap<&str, usize> = bm25_ids
            .iter()
            .enumerate()
            .map(|(rank, id)| (id.as_str(), rank))
            .collect();
        let cosine_rank_of: HashMap<&str, usize> = cosine_ranked
            .iter()
            .map(|(id, _)| id.as_str())
            .enumerate()
            .map(|(rank, id)| (id, rank))
            .collect();
        for (id, r) in by_id.iter_mut() {
            r.relevance_score = crate::retrieval::fusion::rrf_two_arm(
                bm25_rank_of.get(id.as_str()).copied(),
                cosine_rank_of.get(id.as_str()).copied(),
            );
        }

        // Deterministic merged order by the same RRF (tie-break only — the
        // caller re-sorts). vec_ids in COSINE order, not HashMap order: the
        // arm's ranks must be real ranks, and HashMap iteration order would
        // leak random per-process tie-breaks into the caller's stable sort.
        let vec_ids: Vec<String> = cosine_ranked.into_iter().map(|(id, _)| id).collect();
        let fused = crate::retrieval::fusion::rrf_fuse(
            &[bm25_ids, vec_ids],
            crate::retrieval::fusion::RRF_K,
        );
        Ok(fused
            .into_iter()
            .filter_map(|id| by_id.remove(&id))
            .collect())
    }

    #[cfg(feature = "semantic")]
    pub fn is_active(&self) -> bool {
        self.embedder().is_some()
    }
}

#[cfg(all(test, feature = "semantic"))]
mod tests {
    use super::*;
    use crate::models::EpisodicMemory;
    use crate::storage::MemoryRepository;

    /// Deterministic stub: 4-dim vector from a cheap character hash. Similar
    /// texts produce similar vectors (same prefix), different texts don't.
    struct StubEmbedder;
    impl crate::retrieval::embedding::EmbeddingProvider for StubEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 4];
                    for (i, b) in t.bytes().take(16).enumerate() {
                        v[i % 4] += b as f32;
                    }
                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        v.iter().map(|x| x / norm).collect()
                    } else {
                        v
                    }
                })
                .collect())
        }
        fn dim(&self) -> usize {
            4
        }
        fn model_id(&self) -> &str {
            "stub-model"
        }
    }

    fn mem(id: &str, summary: &str) -> EpisodicMemory {
        EpisodicMemory {
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
        }
    }

    #[test]
    fn fuse_materializes_vector_hits_with_rrf_relevance() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let target = mem("v1", "rust language features");
        repo.create_episodic(&target).unwrap();
        let other = mem("v2", "completely unrelated words");
        repo.create_episodic(&other).unwrap();

        let svc = EmbeddingService::with_embedder(Box::new(StubEmbedder));
        svc.index(&repo, "episodic", "v1", "p", "rust language features");
        svc.index(&repo, "episodic", "v2", "p", "completely unrelated words");

        // BM25 side finds only the textually matching v2.
        let bm25 = crate::retrieval::bm25::BM25Retriever::search_by_type(
            &repo,
            "unrelated",
            "p",
            "episodic",
            10,
        )
        .unwrap();
        assert_eq!(bm25.len(), 1);
        assert_eq!(bm25[0].id, "v2");

        // Query embedding equals v1's embedding (same text) → v1 must be
        // materialized as a vector-only hit. Under two-arm RRF a single-arm
        // candidate at rank 0 scores exactly 0.5 (the single-signal ceiling:
        // half the possible evidence), NOT the old max()'s cosine ~1.0.
        let fused = svc
            .fuse(
                &repo,
                "rust language features",
                "p",
                bm25,
                5,
                None,
                &[],
                None,
            )
            .unwrap();
        assert_eq!(fused.len(), 2, "vector-only hit must be materialized");
        let v1 = fused.iter().find(|r| r.id == "v1").unwrap();
        assert!(
            (v1.relevance_score - 0.5).abs() < 1e-4,
            "single-arm rank-0 candidate must score the 0.5 ceiling, got {}",
            v1.relevance_score
        );
        // v2 is BM25 rank 0 AND present in the vector list (rank 1 of 2):
        // dual evidence must outrank the vector-only perfect match.
        let v2 = fused.iter().find(|r| r.id == "v2").unwrap();
        let expected_v2 = crate::retrieval::fusion::rrf_two_arm(Some(0), Some(1));
        assert!(
            (v2.relevance_score - expected_v2).abs() < 1e-4,
            "dual-evidence candidate must score rrf(bm25=0, cosine=1) = {expected_v2}, got {}",
            v2.relevance_score
        );
        assert!(v2.relevance_score > v1.relevance_score);
    }

    #[test]
    fn fuse_balances_arms_instead_of_letting_cosine_override_bm25() {
        // End-to-end (stub embedder, no model download) version of the
        // fusion.rs shootout: a candidate with BOTH a BM25 rank and a
        // (weaker) cosine rank must outrank a candidate whose only evidence
        // is the top cosine slot — the exact ordering max-injection got
        // wrong when the cosine leader was a noise-floor neighbor.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        // "textual": the true lexical match (BM25 rank 0). "neighbor":
        // query-identical text → cosine 1.0, but no BM25 hit for the query.
        repo.create_episodic(&mem("textual", "textual evidence match"))
            .unwrap();
        repo.create_episodic(&mem("neighbor", "query twin words here"))
            .unwrap();

        let svc = EmbeddingService::with_embedder(Box::new(StubEmbedder));
        svc.index(&repo, "episodic", "textual", "p", "textual evidence match");
        svc.index(&repo, "episodic", "neighbor", "p", "query twin words here");

        let bm25 = crate::retrieval::bm25::BM25Retriever::search_by_type(
            &repo, "textual", "p", "episodic", 10,
        )
        .unwrap();
        assert_eq!(bm25[0].id, "textual");

        // Query embeds to the neighbor's exact vector (cosine 1.0 → vector
        // rank 0) while "textual" keeps only its BM25 arm plus a weak cosine.
        let fused = svc
            .fuse(
                &repo,
                "query twin words here",
                "p",
                bm25,
                5,
                None,
                &[],
                None,
            )
            .unwrap();
        let textual = fused.iter().find(|r| r.id == "textual").unwrap();
        let neighbor = fused.iter().find(|r| r.id == "neighbor").unwrap();
        // Measured arm ranks: textual = (bm25 0, cosine 1); neighbor =
        // (bm25 absent, cosine 0). Dual evidence must win.
        assert!(
            textual.relevance_score > neighbor.relevance_score,
            "RRF must rank the dual-evidence match ({}) above the cosine-only twin ({})",
            textual.relevance_score,
            neighbor.relevance_score
        );
    }

    #[test]
    fn fuse_respects_explicit_memory_type_filter() {
        // A vector hit of the WRONG type must not bypass an explicit
        // memory_type filter (the 2026-09 review's filter-penetration bug).
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let target = mem("v1", "rust language features");
        repo.create_episodic(&target).unwrap();

        let svc = EmbeddingService::with_embedder(Box::new(StubEmbedder));
        svc.index(&repo, "episodic", "v1", "p", "rust language features");

        let fused = svc
            .fuse(
                &repo,
                "rust language features",
                "p",
                vec![],
                5,
                Some("failure"), // user asked for failures only
                &[],
                None,
            )
            .unwrap();
        assert!(
            fused.is_empty(),
            "episodic vector hit must not leak through memory_type=failure"
        );
    }

    #[test]
    fn vector_cache_invalidates_on_index() {
        // Writing a new embedding must evict the project's cached vectors —
        // otherwise a stale cache would serve the pre-write set forever.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mem("v1", "alpha")).unwrap();
        repo.create_episodic(&mem("v2", "beta")).unwrap();

        let svc = EmbeddingService::with_embedder(Box::new(StubEmbedder));
        svc.index(&repo, "episodic", "v1", "p", "alpha");
        // Prime the cache with an empty vector set.
        let _ = svc
            .fuse(&repo, "alpha", "p", vec![], 5, None, &[], None)
            .unwrap();
        svc.index(&repo, "episodic", "v2", "p", "beta");
        let fused = svc
            .fuse(&repo, "beta", "p", vec![], 5, None, &[], None)
            .unwrap();
        assert!(
            fused.iter().any(|r| r.id == "v2"),
            "cache must reflect the newly indexed vector"
        );
    }

    #[test]
    fn vector_cache_reloads_when_another_process_writes() {
        // Cross-process invalidation: a DIFFERENT repository handle (standing
        // in for the CLI process) commits an embedding after this service
        // cached its vector set. No in-process invalidation fires, but
        // `PRAGMA data_version` moves — the next search must reload instead of
        // serving the stale set forever.
        let tmp = tempfile::TempDir::new().unwrap();
        let db = tmp.path().join("vectors.db");
        let repo = MemoryRepository::new(&db).unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mem("v1", "alpha words")).unwrap();
        repo.create_episodic(&mem("v2", "beta words")).unwrap();

        let svc = EmbeddingService::with_embedder(Box::new(StubEmbedder));
        svc.index(&repo, "episodic", "v1", "p", "alpha words");
        // Prime the cache: only v1 is embedded at this point.
        let first = svc
            .fuse(&repo, "beta", "p", vec![], 5, None, &[], None)
            .unwrap();
        assert!(
            !first.iter().any(|r| r.id == "v2"),
            "precondition: v2 has no vector yet"
        );

        // "Another process" writes v2's embedding through its own connection.
        let repo2 = MemoryRepository::new(&db).unwrap();
        repo2
            .upsert_embedding(
                "v2",
                "episodic",
                "p",
                &[0.1, 0.2, 0.3, 0.4],
                "stub-model",
                4,
            )
            .unwrap();

        let fused = svc
            .fuse(&repo, "beta", "p", vec![], 5, None, &[], None)
            .unwrap();
        assert!(
            fused.iter().any(|r| r.id == "v2"),
            "data_version bump from the other process must trigger a reload"
        );
    }
}
