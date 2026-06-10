# ADR-003: Static memory planning with a zero-allocation arena (Runtime↔Memory Shared Kernel)

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: memory, performance, shared-kernel, core

## Context

Autoregressive decoding on mobile is memory-bound, and OS heap fragmentation or
runtime `malloc` calls cause latency jitter and UI stalls. The PRD (§"Memory
Allocation Strategy", §"KV-Cache and Memory") requires a contiguous, heap-less
buffer with tensor offsets assigned ahead of time (reproducing ExecuTorch's
memory-planning *technique* in Rust — see [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)),
tiered SRAM/DRAM placement, `memmap2`-mapped constant weights, and KV
compaction by descriptor shuffle rather than data copy.

This decision creates a tight coupling: the
[Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md) decode loop
and the [Memory Management](../ddd/bounded-contexts/06-memory-management.md)
planner must agree exactly on arena layout and tensor descriptors. That shared
model is the riskiest cross-context contract in the system.

## Decision

Compute a **static `MemoryPlan` before prefill** that assigns every tensor a
fixed `TensorOffset` and `MemoryTier` in a single, page-aligned contiguous
**`Arena`** (a Rust `Box<[u8]>` / bump allocator) allocated once at session init.
**No heap allocation occurs in the Rust decode-loop hot path.** Constant weights
are `memmap2`-mapped read-only (zero-copy). KV-cache
pruning is resolved by **descriptor-only compaction** (pointer arithmetic over
reserved headroom), never a realloc or data copy.

Model the Runtime↔Memory relationship as a **Shared Kernel**: the arena layout
and tensor-descriptor types are co-owned and changed jointly, not via a
one-directional service contract.

## Consequences

### Positive
- Eliminates runtime allocation jitter; the GPU/Metal graph always sees a
  contiguous region. Rust's ownership model makes the buffer lifecycle explicit
  and leak-resistant.
- Enforces the PRD memory budgets (~300–500 MB peak for a 4-bit 0.5B model; cap
  ~1 GB) as a checkable invariant before inference starts.
- `MemoryBudgetExceeded` becomes the single trigger for graceful degradation
  (see [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md) and the degradation flow in [domain-events](../ddd/domain-events.md)).

### Negative
- A Shared Kernel is the highest-coupling DDD relationship: changes ripple across
  two contexts and demand joint review, slowing independent evolution.
- Static planning constrains dynamic shapes; variable-length features (e.g.
  growing KV) must pre-reserve worst-case headroom.

### Neutral
- The planner is a pure-Rust `StaticMemoryPlanner` that feeds pre-allocated
  storage to the engine through `RuntimeAcl`, decoupling the domain from the
  engine chosen in [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)
  (Candle).

## Links
- PRD: `docs/prd.md` §"Memory Allocation Strategy", §"KV-Cache and Memory"
- DDD: [Memory Management](../ddd/bounded-contexts/06-memory-management.md), [Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md), [context-map](../ddd/context-map.md)
- Related: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md), [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
