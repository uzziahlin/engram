use serde::{Deserialize, Serialize};

/// Intent types for memory retrieval routing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryIntent {
    Debugging,
    Architecture,
    Workflow,
    Refactor,
    Deployment,
    Incident,
    General,
}
