use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Entity types in the engineering topology graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    File,
    Function,
    Module,
    Service,
    Commit,
    PullRequest,
    Bug,
    Incident,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::File => "File",
            EntityType::Function => "Function",
            EntityType::Module => "Module",
            EntityType::Service => "Service",
            EntityType::Commit => "Commit",
            EntityType::PullRequest => "PullRequest",
            EntityType::Bug => "Bug",
            EntityType::Incident => "Incident",
        }
    }
}

impl FromStr for EntityType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "File" => Ok(EntityType::File),
            "Function" => Ok(EntityType::Function),
            "Module" => Ok(EntityType::Module),
            "Service" => Ok(EntityType::Service),
            "Commit" => Ok(EntityType::Commit),
            "PullRequest" => Ok(EntityType::PullRequest),
            "Bug" => Ok(EntityType::Bug),
            "Incident" => Ok(EntityType::Incident),
            other => Err(format!("unknown entity type: {other}")),
        }
    }
}

/// Relation types between entities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RelationType {
    DependsOn,
    Calls,
    FixedBy,
    IntroducedBy,
    RelatedTo,
    References,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::DependsOn => "DependsOn",
            RelationType::Calls => "Calls",
            RelationType::FixedBy => "FixedBy",
            RelationType::IntroducedBy => "IntroducedBy",
            RelationType::RelatedTo => "RelatedTo",
            RelationType::References => "References",
        }
    }
}

impl FromStr for RelationType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "DependsOn" => Ok(RelationType::DependsOn),
            "Calls" => Ok(RelationType::Calls),
            "FixedBy" => Ok(RelationType::FixedBy),
            "IntroducedBy" => Ok(RelationType::IntroducedBy),
            "RelatedTo" => Ok(RelationType::RelatedTo),
            "References" => Ok(RelationType::References),
            other => Err(format!("unknown relation type: {other}")),
        }
    }
}

/// An entity in the relationship graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub project_id: String,
    pub entity_type: EntityType,
    pub name: String,
    pub metadata: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A relation between two entities in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    pub id: i64,
    pub project_id: Option<String>,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: RelationType,
    pub weight: f64,
    pub created_at: i64,
}
