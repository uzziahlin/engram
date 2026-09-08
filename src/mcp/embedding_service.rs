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
    embedder: std::sync::OnceLock<Option<Box<dyn crate::retrieval::embedding::EmbeddingProvider>>>,
    #[cfg(feature = "semantic")]
    lazy_config: Box<crate::config::SemanticConfig>,
    /// Per-(project, model) vector cache: brute-force cosine used to
    /// re-deserialize every vector from SQLite on EVERY query (~15 MB per
    /// 10k memories). Entries are evicted wholesale on any write — coarse,
    /// but obviously correct (staleness self-heals: materialization of a
    /// deleted id returns None and the hit is dropped).
    #[cfg(feature = "semantic")]
    vector_cache: std::sync::Mutex<VectorCache>,
}

/// One cached vector with its memory identity: (memory_id, memory_type, vec).
#[cfg(feature = "semantic")]
pub type VectorEntry = (String, String, Vec<f32>);

/// (project_id, model_id) → the project's cached active vectors.
#[cfg(feature = "semantic")]
type VectorCache = std::collections::HashMap<(String, String), std::sync::Arc<Vec<VectorEntry>>>;

impl EmbeddingService {
    pub fn new(config: &Config) -> Self {
        #[cfg(feature = "semantic")]
        {
            Self {
                embedder: std::sync::OnceLock::new(),
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
            embedder: std::sync::OnceLock::from(Some(embedder)),
            lazy_config: Box::new(crate::config::SemanticConfig::default()),
            vector_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Lazily resolve the embedder: model download + load happens on FIRST
    /// USE, not at server startup — it used to block the MCP initialize
    /// handshake for the whole ~90 MB first-run fetch.
    #[cfg(feature = "semantic")]
    fn embedder(&self) -> Option<&dyn crate::retrieval::embedding::EmbeddingProvider> {
        self.embedder
            .get_or_init(|| Self::init_embedder(&self.lazy_config))
            .as_deref()
    }

    /// Drop cached vectors for a project (any write that can change the
    /// active vector set). `None` = drop everything (reindex, model change).
    #[cfg(feature = "semantic")]
    pub fn invalidate_vectors(&self, project: Option<&str>) {
        let mut cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
        match project {
            Some(p) => {
                cache.retain(|(proj, _), _| proj != p);
            }
            None => cache.clear(),
        }
    }

    /// Load the active vectors for (project, model), through the cache.
    #[cfg(feature = "semantic")]
    fn cached_vectors(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
        model_id: &str,
    ) -> Result<std::sync::Arc<Vec<VectorEntry>>> {
        let key = (project_id.to_string(), model_id.to_string());
        {
            let cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&key) {
                return Ok(std::sync::Arc::clone(hit));
            }
        }
        let loaded: Vec<VectorEntry> = repo.load_active_embeddings(project_id, model_id)?;
        let arc = std::sync::Arc::new(loaded);
        let mut cache = self.vector_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(key, std::sync::Arc::clone(&arc));
        Ok(arc)
    }

    /// Build the embedding model from config (semantic enabled + model available).
    /// Returns None (logging a warning) on any failure so the server still runs
    /// in pure-BM25 mode.
    #[cfg(feature = "semantic")]
    fn init_embedder(
        config: &crate::config::SemanticConfig,
    ) -> Option<Box<dyn crate::retrieval::embedding::EmbeddingProvider>> {
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
                    "semantic enabled but model fetch failed ({e}); disabling semantic search"
                );
                return None;
            }
        }
        match CandleBertEmbedder::from_local(&dir, &config.model_id) {
            Ok(e) => {
                tracing::info!("semantic search enabled: model {}", config.model_id);
                Some(Box::new(e))
            }
            Err(e) => {
                tracing::warn!("semantic model load failed ({e}); disabling semantic search");
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

    /// Semantic fusion: blend vector top-K with the BM25 candidates via RRF.
    /// Vector-only hits are materialized so they can surface even when BM25
    /// missed them entirely.
    ///
    /// Two invariants fixed in the 2026-09 review:
    /// - the caller's explicit filters (`memory_type`/`tags`/`before`) apply
    ///   to vector-only candidates too — the vector path must not bypass them;
    /// - the cosine score is injected into `relevance_score` (max with the
    ///   normalized BM25 score) instead of being discarded, so a strong
    ///   semantic match actually influences the final ranking.
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
        rrf_k: f32,
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
        let cands: Vec<(String, Vec<f32>)> = loaded
            .iter()
            .map(|(id, _, v)| (id.clone(), v.clone()))
            .collect();
        let cosine_of: HashMap<String, f32> =
            crate::retrieval::vector::top_k_cosine(&query_vec, &cands, top_k)
                .into_iter()
                .collect();

        // Materialize vector-only candidates that survive the caller's
        // explicit filters, exactly like the BM25 side did.
        let bm25_ids: Vec<String> = bm25.iter().map(|r| r.id.clone()).collect();
        let mut by_id: HashMap<String, SearchResult> =
            bm25.into_iter().map(|r| (r.id.clone(), r)).collect();
        let missing: Vec<(String, String)> = cosine_of
            .keys()
            .filter(|id| !by_id.contains_key(*id))
            .filter_map(|id| type_of.get(id).map(|ty| (ty.clone(), id.clone())))
            .collect();
        for sr in crate::retrieval::bm25::BM25Retriever::fetch_by_ids(repo, &missing, project_id)? {
            let type_ok = memory_type.map_or(true, |mt| sr.memory_type == mt);
            let tags_ok = tags.is_empty() || sr.tags.iter().any(|t| tags.contains(t));
            let before_ok = before.map_or(true, |b| sr.created_at < b);
            if type_ok && tags_ok && before_ok {
                by_id.entry(sr.id.clone()).or_insert(sr);
            }
        }

        // Inject cosine into relevance: take the max with the normalized BM25
        // score so either signal at full strength carries the result.
        for (id, cosine) in &cosine_of {
            if let Some(r) = by_id.get_mut(id) {
                r.relevance_score = r.relevance_score.max(cosine.clamp(0.0, 1.0));
            }
        }

        let vec_ids: Vec<String> = cosine_of.into_keys().collect();
        let fused = crate::retrieval::fusion::rrf_fuse(&[bm25_ids, vec_ids], rrf_k);
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
    fn fuse_materializes_vector_hits_and_injects_cosine() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        // "rust language features" is the vector target; the query shares its
        // bytes but BM25 misses it via a tag-only FTS mismatch is hard to
        // force — instead assert on the union + score-injection directly.
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
        // materialized as a vector-only hit with cosine ~1.0 injected.
        let fused = svc
            .fuse(
                &repo,
                "rust language features",
                "p",
                bm25,
                5,
                60.0,
                None,
                &[],
                None,
            )
            .unwrap();
        assert_eq!(fused.len(), 2, "vector-only hit must be materialized");
        let v1 = fused.iter().find(|r| r.id == "v1").unwrap();
        assert!(
            v1.relevance_score > 0.99,
            "cosine of an identical text must inject ~1.0 relevance, got {}",
            v1.relevance_score
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
                60.0,
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
            .fuse(&repo, "alpha", "p", vec![], 5, 60.0, None, &[], None)
            .unwrap();
        svc.index(&repo, "episodic", "v2", "p", "beta");
        let fused = svc
            .fuse(&repo, "beta", "p", vec![], 5, 60.0, None, &[], None)
            .unwrap();
        assert!(
            fused.iter().any(|r| r.id == "v2"),
            "cache must reflect the newly indexed vector"
        );
    }
}
