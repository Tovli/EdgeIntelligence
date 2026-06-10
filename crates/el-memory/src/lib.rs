//! `el-memory` — Static memory planning, a single contiguous arena, and
//! descriptor-only KV-cache compaction (ADR-003).
//!
//! The discipline (reproduced from ExecuTorch's technique, in pure Rust): assign
//! every tensor a fixed offset *before* inference so the decode loop performs no
//! heap allocation, and resolve KV pruning by shuffling descriptors rather than
//! copying data.

#![forbid(unsafe_code)]

mod arena;
mod kv;
mod planner;

pub use arena::Arena;
pub use kv::{KvRegion, KvSlot};
pub use planner::{
    BufferLifetime, MemoryPlan, MemoryTier, Placement, StaticMemoryPlanner, TensorId, TensorOffset,
    TensorSpec,
};
