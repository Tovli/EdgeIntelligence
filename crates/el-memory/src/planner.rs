//! Ahead-of-time static memory planner.
//!
//! Packs tensor lifetimes so that tensors whose lifetimes do **not** overlap
//! reuse the same offset (interval colouring). Each tier (`Sram`/`Dram`) is a
//! separate offset space.

use el_core::{EdgeError, Result};

pub type TensorId = u32;
pub type TensorOffset = u64;

/// Fast (SRAM, e.g. KV cache) vs constant (DRAM, weights) placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    Sram,
    Dram,
}

/// The `[first_use, last_use]` step range a buffer is live for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferLifetime {
    pub first_use: u32,
    pub last_use: u32,
}

impl BufferLifetime {
    pub fn new(first_use: u32, last_use: u32) -> Self {
        debug_assert!(first_use <= last_use);
        Self {
            first_use,
            last_use,
        }
    }

    /// Closed-interval overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.first_use <= other.last_use && other.first_use <= self.last_use
    }
}

/// A tensor the planner must place.
#[derive(Debug, Clone, Copy)]
pub struct TensorSpec {
    pub id: TensorId,
    pub size: u64,
    pub tier: MemoryTier,
    pub lifetime: BufferLifetime,
}

/// Final offset assignment for one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub id: TensorId,
    pub offset: TensorOffset,
    pub size: u64,
    pub tier: MemoryTier,
}

/// The complete static allocation. Immutable during inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    placements: Vec<Placement>,
    sram_bytes: u64,
    dram_bytes: u64,
}

impl MemoryPlan {
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }
    pub fn placement(&self, id: TensorId) -> Option<&Placement> {
        self.placements.iter().find(|p| p.id == id)
    }
    pub fn sram_bytes(&self) -> u64 {
        self.sram_bytes
    }
    pub fn dram_bytes(&self) -> u64 {
        self.dram_bytes
    }
    pub fn total_bytes(&self) -> u64 {
        self.sram_bytes + self.dram_bytes
    }
}

/// Computes a [`MemoryPlan`] by interval-colouring lifetimes per tier.
pub struct StaticMemoryPlanner;

impl StaticMemoryPlanner {
    /// Plan placement for `tensors` within `budget_bytes` (ADR-003). Returns
    /// [`EdgeError::MemoryBudgetExceeded`] if the packed plan exceeds the budget
    /// — the signal the runtime uses to disable optional features or spill.
    pub fn plan(tensors: &[TensorSpec], budget_bytes: u64) -> Result<MemoryPlan> {
        let mut placements = Vec::with_capacity(tensors.len());
        let sram_bytes = Self::plan_tier(tensors, MemoryTier::Sram, &mut placements);
        let dram_bytes = Self::plan_tier(tensors, MemoryTier::Dram, &mut placements);

        let total = sram_bytes + dram_bytes;
        if total > budget_bytes {
            return Err(EdgeError::MemoryBudgetExceeded {
                requested: total,
                budget: budget_bytes,
            });
        }
        Ok(MemoryPlan {
            placements,
            sram_bytes,
            dram_bytes,
        })
    }

    /// Colour one tier; append placements; return the tier's packed size.
    fn plan_tier(tensors: &[TensorSpec], tier: MemoryTier, out: &mut Vec<Placement>) -> u64 {
        // Tensors in this tier, ordered by lifetime start (greedy colouring).
        let mut items: Vec<&TensorSpec> = tensors.iter().filter(|t| t.tier == tier).collect();
        items.sort_by_key(|t| (t.lifetime.first_use, t.lifetime.last_use));

        // Each bucket = a reusable slot. `last_use` is the end of the lifetime
        // currently occupying it; `size` is the max size assigned so far.
        struct Bucket {
            last_use: u32,
            size: u64,
        }
        let mut buckets: Vec<Bucket> = Vec::new();
        // bucket index chosen per tensor (parallel to `items`).
        let mut bucket_of: Vec<usize> = Vec::with_capacity(items.len());

        for t in &items {
            // Reuse the first bucket whose occupant has already ended.
            let mut chosen = None;
            for (i, b) in buckets.iter_mut().enumerate() {
                if b.last_use < t.lifetime.first_use {
                    b.last_use = t.lifetime.last_use;
                    if t.size > b.size {
                        b.size = t.size;
                    }
                    chosen = Some(i);
                    break;
                }
            }
            let idx = match chosen {
                Some(i) => i,
                None => {
                    buckets.push(Bucket {
                        last_use: t.lifetime.last_use,
                        size: t.size,
                    });
                    buckets.len() - 1
                }
            };
            bucket_of.push(idx);
        }

        // Lay buckets out sequentially within this tier (offsets start at 0).
        let mut bucket_offset = Vec::with_capacity(buckets.len());
        let mut running = 0u64;
        for b in &buckets {
            bucket_offset.push(running);
            running += b.size;
        }

        for (t, &b) in items.iter().zip(bucket_of.iter()) {
            out.push(Placement {
                id: t.id,
                offset: bucket_offset[b],
                size: t.size,
                tier,
            });
        }
        running // total bytes used by this tier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: u32, size: u64, tier: MemoryTier, f: u32, l: u32) -> TensorSpec {
        TensorSpec {
            id,
            size,
            tier,
            lifetime: BufferLifetime::new(f, l),
        }
    }

    #[test]
    fn non_overlapping_lifetimes_reuse_one_offset() {
        // Two same-tier tensors with disjoint lifetimes → share an offset, so
        // the tier needs max(size), not the sum.
        let t = [
            spec(1, 100, MemoryTier::Dram, 0, 2),
            spec(2, 100, MemoryTier::Dram, 3, 5),
        ];
        let plan = StaticMemoryPlanner::plan(&t, 10_000).unwrap();
        assert_eq!(
            plan.dram_bytes(),
            100,
            "disjoint lifetimes must reuse space"
        );
        assert_eq!(
            plan.placement(1).unwrap().offset,
            plan.placement(2).unwrap().offset
        );
    }

    #[test]
    fn overlapping_lifetimes_get_distinct_offsets() {
        let t = [
            spec(1, 100, MemoryTier::Dram, 0, 4),
            spec(2, 100, MemoryTier::Dram, 2, 6),
        ];
        let plan = StaticMemoryPlanner::plan(&t, 10_000).unwrap();
        assert_eq!(plan.dram_bytes(), 200);
        assert_ne!(
            plan.placement(1).unwrap().offset,
            plan.placement(2).unwrap().offset
        );
    }

    #[test]
    fn tiers_are_separate_offset_spaces() {
        let t = [
            spec(1, 64, MemoryTier::Sram, 0, 9),
            spec(2, 256, MemoryTier::Dram, 0, 9),
        ];
        let plan = StaticMemoryPlanner::plan(&t, 10_000).unwrap();
        assert_eq!(plan.sram_bytes(), 64);
        assert_eq!(plan.dram_bytes(), 256);
        assert_eq!(plan.total_bytes(), 320);
    }

    #[test]
    fn exceeding_budget_is_an_error() {
        let t = [spec(1, 2000, MemoryTier::Dram, 0, 1)];
        let err = StaticMemoryPlanner::plan(&t, 1000).unwrap_err();
        assert_eq!(
            err,
            EdgeError::MemoryBudgetExceeded {
                requested: 2000,
                budget: 1000
            }
        );
    }
}
