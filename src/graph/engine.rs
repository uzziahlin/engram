use crate::models::*;
use crate::storage::MemoryRepository;
use anyhow::{Context, Result};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet};

/// Graph engine using Petgraph for in-memory entity relationships.
///
/// Loaded from SQLite on demand. Supports incremental multi-project loading
/// and cross-project edges (project_id IS NULL).
pub struct GraphEngine {
    graph: DiGraph<Entity, GraphRelation>,
    entity_index: HashMap<String, NodeIndex>,
    loaded_projects: HashSet<String>,
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphEngine {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            entity_index: HashMap::new(),
            loaded_projects: HashSet::new(),
        }
    }

    /// Load entities and relations for a project from SQLite into memory.
    /// Incremental: skips already-loaded projects, appends new data without
    /// clearing existing graph state.
    pub fn load_from_repo(&mut self, repo: &MemoryRepository, project_id: &str) -> Result<()> {
        if self.loaded_projects.contains(project_id) {
            return Ok(());
        }

        let entities = repo
            .load_entities_for_project(project_id)
            .context("failed to load entities")?;
        let relations = repo
            .load_relations_for_project(project_id)
            .context("failed to load relations")?;

        // Add entities as nodes
        for entity in &entities {
            let idx = self.graph.add_node(entity.clone());
            self.entity_index.insert(entity.id.clone(), idx);
        }

        // Add relations as edges
        for rel in &relations {
            if let (Some(&from_idx), Some(&to_idx)) = (
                self.entity_index.get(&rel.from_entity),
                self.entity_index.get(&rel.to_entity),
            ) {
                self.graph.add_edge(from_idx, to_idx, rel.clone());
            }
        }

        self.loaded_projects.insert(project_id.to_string());
        Ok(())
    }

    /// Add an entity to the graph (call after SQLite write).
    pub fn add_entity(&mut self, entity: Entity) {
        let idx = self.graph.add_node(entity.clone());
        self.entity_index.insert(entity.id.clone(), idx);
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entity_index
            .get(id)
            .map(|&idx| &self.graph[idx])
    }

    /// Remove an entity and its edges from the graph.
    pub fn remove_entity(&mut self, id: &str) -> bool {
        if let Some(idx) = self.entity_index.remove(id) {
            self.graph.remove_node(idx);
            true
        } else {
            false
        }
    }

    /// Add a relation (call after SQLite write).
    /// Returns false if either entity doesn't exist.
    pub fn add_relation(&mut self, rel: GraphRelation) -> bool {
        if let (Some(&from_idx), Some(&to_idx)) = (
            self.entity_index.get(&rel.from_entity),
            self.entity_index.get(&rel.to_entity),
        ) {
            self.graph.add_edge(from_idx, to_idx, rel);
            true
        } else {
            false
        }
    }

    /// Get all relations for an entity (both outgoing and incoming).
    pub fn get_relations(&self, entity_id: &str) -> Vec<&GraphRelation> {
        if let Some(&idx) = self.entity_index.get(entity_id) {
            let mut relations = Vec::new();
            // Outgoing edges
            for edge_idx in self.graph.edges(idx) {
                relations.push(edge_idx.weight());
            }
            // Incoming edges
            for edge_idx in self.graph.edges_directed(idx, petgraph::Direction::Incoming) {
                relations.push(edge_idx.weight());
            }
            relations
        } else {
            Vec::new()
        }
    }

    /// Traverse from an entity, following all edges up to `max_depth` hops.
    /// Uses BFS with a visited set (handles cycles gracefully).
    pub fn traverse_from(&self, entity_id: &str, max_depth: usize) -> Vec<Entity> {
        let mut result = Vec::new();
        let start_idx = match self.entity_index.get(entity_id) {
            Some(&idx) => idx,
            None => return result,
        };

        let mut visited = std::collections::HashSet::new();
        visited.insert(start_idx);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_idx, 0usize));

        while let Some((node_idx, depth)) = queue.pop_front() {
            if depth > 0 {
                if let Some(entity) = self.graph.node_weight(node_idx) {
                    result.push(entity.clone());
                }
            }

            if depth < max_depth {
                for neighbor in self.graph.neighbors(node_idx) {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
                // Also traverse incoming edges
                for neighbor in self
                    .graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        result
    }

    /// Check if the graph contains cycles.
    pub fn is_cyclic(&self) -> bool {
        petgraph::algo::is_cyclic_directed(&self.graph)
    }

    /// Get the number of entities (nodes) in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of relations (edges) in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, name: &str) -> Entity {
        Entity {
            id: id.into(),
            project_id: "test".into(),
            entity_type: EntityType::File,
            name: name.into(),
            metadata: serde_json::json!({}),
            created_at: 1000,
            updated_at: 1000,
        }
    }

    fn make_relation(from: &str, to: &str, rt: RelationType) -> GraphRelation {
        GraphRelation {
            id: 0,
            project_id: Some("test".into()),
            from_entity: from.into(),
            to_entity: to.into(),
            relation_type: rt,
            weight: 1.0,
            created_at: 1000,
        }
    }

    #[test]
    fn test_add_and_get_entity() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "auth.ts"));
        engine.add_entity(make_entity("e2", "redis.ts"));

        assert_eq!(engine.node_count(), 2);
        assert_eq!(engine.get_entity("e1").unwrap().name, "auth.ts");
        assert!(engine.get_entity("nonexistent").is_none());
    }

    #[test]
    fn test_remove_entity() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "auth.ts"));
        assert!(engine.remove_entity("e1"));
        assert_eq!(engine.node_count(), 0);
        assert!(!engine.remove_entity("nonexistent"));
    }

    #[test]
    fn test_add_relation() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "auth.ts"));
        engine.add_entity(make_entity("e2", "redis.ts"));

        let added = engine.add_relation(make_relation("e1", "e2", RelationType::DependsOn));
        assert!(added);
        assert_eq!(engine.edge_count(), 1);

        // Relation with missing entity should fail
        let added = engine.add_relation(make_relation("e1", "missing", RelationType::Calls));
        assert!(!added);
    }

    #[test]
    fn test_get_relations() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "auth.ts"));
        engine.add_entity(make_entity("e2", "redis.ts"));
        engine.add_entity(make_entity("e3", "token.rs"));

        engine.add_relation(make_relation("e1", "e2", RelationType::DependsOn));
        engine.add_relation(make_relation("e3", "e1", RelationType::Calls));

        let relations = engine.get_relations("e1");
        assert_eq!(relations.len(), 2);
    }

    #[test]
    fn test_traverse_from() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "auth.ts"));
        engine.add_entity(make_entity("e2", "redis.ts"));
        engine.add_entity(make_entity("e3", "token.rs"));
        engine.add_entity(make_entity("e4", "config.yaml"));

        // e1 -> e2 -> e3
        engine.add_relation(make_relation("e1", "e2", RelationType::DependsOn));
        engine.add_relation(make_relation("e2", "e3", RelationType::Calls));

        let neighbors = engine.traverse_from("e1", 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].name, "redis.ts");

        let depth2 = engine.traverse_from("e1", 2);
        assert_eq!(depth2.len(), 2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "a"));
        engine.add_entity(make_entity("e2", "b"));

        engine.add_relation(make_relation("e1", "e2", RelationType::DependsOn));
        assert!(!engine.is_cyclic());

        engine.add_relation(make_relation("e2", "e1", RelationType::DependsOn));
        assert!(engine.is_cyclic());
    }

    #[test]
    fn test_load_from_repo_is_incremental_and_idempotent() {
        use crate::storage::MemoryRepository;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("graph.db");
        let repo = MemoryRepository::new(&db_path).unwrap();
        repo.initialize_schema().unwrap();

        // Seed two projects with one entity each.
        let mut a = make_entity("ea", "a.ts");
        a.project_id = "proj-a".into();
        let mut b = make_entity("eb", "b.ts");
        b.project_id = "proj-b".into();
        repo.create_entity(&a).unwrap();
        repo.create_entity(&b).unwrap();

        let mut engine = GraphEngine::new();

        // First load — pulls proj-a only.
        engine.load_from_repo(&repo, "proj-a").unwrap();
        assert_eq!(engine.node_count(), 1);
        assert!(engine.get_entity("ea").is_some());
        assert!(engine.get_entity("eb").is_none());

        // Second load of the SAME project must be a no-op (idempotent).
        engine.load_from_repo(&repo, "proj-a").unwrap();
        assert_eq!(engine.node_count(), 1);

        // Third load of a different project appends, does NOT clear.
        engine.load_from_repo(&repo, "proj-b").unwrap();
        assert_eq!(engine.node_count(), 2);
        assert!(engine.get_entity("ea").is_some());
        assert!(engine.get_entity("eb").is_some());
    }

    #[test]
    fn test_traverse_handles_cycles() {
        let mut engine = GraphEngine::new();
        engine.add_entity(make_entity("e1", "a"));
        engine.add_entity(make_entity("e2", "b"));
        engine.add_entity(make_entity("e3", "c"));

        // Create cycle: e1 -> e2 -> e3 -> e1
        engine.add_relation(make_relation("e1", "e2", RelationType::DependsOn));
        engine.add_relation(make_relation("e2", "e3", RelationType::DependsOn));
        engine.add_relation(make_relation("e3", "e1", RelationType::DependsOn));

        // Should not infinite loop
        let results = engine.traverse_from("e1", 10);
        assert_eq!(results.len(), 2); // e2 and e3 (visited set prevents revisiting e1)
    }
}
