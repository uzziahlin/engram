pub mod decision;
pub mod entity;
pub mod episodic;
pub mod failure;
pub mod intent;
pub mod procedural;

pub use decision::*;
pub use entity::*;
pub use episodic::*;
pub use failure::*;
pub use intent::*;
pub use procedural::*;

/// Clamp an importance to the valid 0..=1 range. Non-finite values (NaN/inf,
/// which can arrive via deserialized JSON or unvalidated CLI flags) map to the
/// neutral 0.5 instead of poisoning ranking arithmetic downstream.
pub fn sanitize_importance(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Write-path sanitization applied by the CRUD macro on EVERY create/update,
/// regardless of entry point (MCP tools, CLI flags, session-import, import).
/// Previously validation lived per-caller and drifted: the CLI accepted
/// `--importance 99`, and `update_memory` merge-patches merged raw values.
pub trait SanitizeMemory: Clone + Sized {
    fn sanitized(self) -> Self;
}

impl SanitizeMemory for EpisodicMemory {
    fn sanitized(mut self) -> Self {
        self.importance = sanitize_importance(self.importance);
        self
    }
}

impl SanitizeMemory for DecisionMemory {
    fn sanitized(mut self) -> Self {
        self.importance = sanitize_importance(self.importance);
        self
    }
}

impl SanitizeMemory for ProceduralMemory {
    fn sanitized(mut self) -> Self {
        self.importance = sanitize_importance(self.importance);
        self
    }
}

impl SanitizeMemory for FailureMemory {
    fn sanitized(mut self) -> Self {
        // severity is the failure-type ranking signal; clamp to the 1..=5
        // scale mark_relevance feedback also uses.
        self.severity = self.severity.clamp(1, 5);
        self
    }
}
