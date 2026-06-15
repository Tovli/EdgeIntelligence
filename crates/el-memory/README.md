# el-memory — static memory planning

Ahead-of-time memory planning, a single contiguous arena allocated once, and
descriptor-only KV-cache compaction (ADR-003). The discipline — reproduced from
ExecuTorch's technique, in pure Rust — is to assign every tensor a fixed offset
*before* inference so the decode loop performs **no heap allocation**, and to
resolve KV pruning by shuffling descriptors rather than copying data.

Depends only on `el-core`. No `unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

- **`StaticMemoryPlanner::plan(tensors, budget_bytes)`** — interval-colours
  tensor lifetimes per memory tier so that tensors whose lifetimes do **not**
  overlap reuse the same offset. Returns a `MemoryPlan`, or
  `EdgeError::MemoryBudgetExceeded` when the packed plan does not fit the budget
  (the signal the runtime uses to disable optional features or spill).
- **`MemoryPlan`** — the immutable allocation: `placements()`, `placement(id)`,
  `sram_bytes()`, `dram_bytes()`, `total_bytes()`.
- **`TensorSpec`** / **`BufferLifetime`** / **`MemoryTier`** (`Sram` | `Dram`) /
  **`Placement`** / **`TensorId`** / **`TensorOffset`** — the planner's inputs
  and outputs.
- **`Arena`** — a contiguous byte buffer allocated once at session init; the
  decode loop borrows planned sub-slices via `region()` / `region_mut()` and
  never allocates again.
- **`KvRegion`** / **`KvSlot`** — the KV cache addressed through descriptors.
  `compact()` removes pruned descriptors and re-indexes survivors **without
  moving payload bytes** — survivors keep their original `offset`.

## Usage

```rust
use el_memory::{Arena, BufferLifetime, KvRegion, MemoryTier, StaticMemoryPlanner, TensorSpec};

// Two same-tier tensors with disjoint lifetimes share one offset, so the tier
// needs max(size), not the sum.
let tensors = [
    TensorSpec { id: 1, size: 100, tier: MemoryTier::Dram, lifetime: BufferLifetime::new(0, 2) },
    TensorSpec { id: 2, size: 100, tier: MemoryTier::Dram, lifetime: BufferLifetime::new(3, 5) },
];
let plan = StaticMemoryPlanner::plan(&tensors, 1024 * 1024)?;
assert_eq!(plan.dram_bytes(), 100); // reused, not 200

// Allocate the arena once; the decode loop borrows planned regions.
let mut arena = Arena::new(plan.total_bytes() as usize);
if let Some(region) = arena.region_mut(0, 100) { /* fill weights */ }

// KV compaction shuffles descriptors only — no data copy.
let mut kv = KvRegion::new();
kv.push(0);   // token 0 @ offset 0
kv.push(64);  // token 1 @ offset 64
kv.mark_pruned(0);
let reclaimed = kv.compact(); // 1; the survivor keeps offset 64
# Ok::<(), el_core::EdgeError>(())
```

## Place in the workspace

Used by `el-runtime` (the session owns a `KvRegion`). The planner's
`MemoryBudgetExceeded` error and the `MemoryPlanCreated`/`KvCacheCompacted`
domain events tie this crate to the Telemetry & Privacy context.

## Status

Implemented and tested. The `Arena` models the allocate-once + fixed-offset
contract in safe Rust; true OS page-alignment and DMA wiring are a platform
concern handled when a real engine path is attached.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-003](../../docs/adr/ADR-003-static-memory-planning-with-zero-allocation-arena.md);
see the [Memory Management](../../docs/ddd/bounded-contexts/06-memory-management.md) context.
