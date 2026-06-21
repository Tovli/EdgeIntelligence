//! `el-runtime` — the Core: inference session lifecycle, the port traits that
//! the collaborator contexts plug into, and the decode-loop orchestrator
//! (ADR-001). Air-gap is structural (ADR-004): this crate has no network
//! dependency, and the only outbound seam is the opt-in [`ports::HybridRelay`].
//!
//! The decode step composes collaborators in a fixed, invariant order
//! (`docs/ddd/domain-events.md`): **grammar mask → safety adjust → sample →
//! commit**, so safety steering only ever operates over already-legal tokens.

#![forbid(unsafe_code)]

mod defaults;
mod ports;
mod session;

pub use defaults::{AllowAllMasker, IdentityCompressor, NullEngine};
pub use ports::{GrammarMasker, HybridRelay, InferenceEngine, Ports, PromptCompressor};
pub use session::InferenceSession;

// Re-export the safety ports and the concrete on-device steerers/guards so
// adapters wire one type system (they depend on `el-runtime`, not `el-safety`).
pub use el_safety::{
    contrastive_adjustment, AnchorGuard, Checkpoint, CheckpointManager, ChunkGuard,
    ContrastiveSteerer, ExpertLogits, LightweightFilter, LogitAdjustment, NoSafety, RollbackPolicy,
    SafetyModeSelector, SafetyScore, SafetySteerer,
};
