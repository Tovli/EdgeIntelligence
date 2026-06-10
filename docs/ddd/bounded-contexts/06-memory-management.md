# Bounded Context 6 — Memory Management  (Supporting)

> Owns the ahead-of-time static memory plan, the single contiguous arena, tiered
> placement (SRAM/DRAM), zero-copy weight loading, and KV-cache compaction.
> Source: PRD §"Memory Allocation Strategy", §"KV-Cache and Memory",
> §ExecuTorch static-planning technique (reproduced in pure Rust — `docs/adr/ADR-003`).

## Purpose

Make "zero heap allocation during inference" an enforceable invariant by
assigning every tensor a fixed offset ahead of time, so the Rust decode-loop hot
path never fragments memory or allocates.

## Strategic role

**Supporting**, and a **Shared Kernel** with the Core (Inference Runtime): the
arena layout and tensor-descriptor model are co-owned. The discipline (static
planning à la ExecuTorch) is well-understood, but it is foundational to the
Core's guarantees.

## Ubiquitous language (context-local)

`Memory Plan`, `Arena`, `Memory Tier`, `Zero-Copy Load`, `KV Compaction`,
`Tensor Offset`, `Buffer Lifetime`.

## Aggregates

### `MemoryPlan` (Aggregate Root)
The complete static allocation for one session.

- **Identity:** `MemoryPlanId` (bound to a `SessionId` + `Model`).
- **Holds:** the `Arena` descriptor, the map of tensor → `TensorOffset` +
  `MemoryTier`, and reserved fragmentation headroom for compaction.
- **Invariants:**
  - The plan is computed **before** prefill and is immutable during inference.
  - Tensor lifetimes are packed so non-overlapping lifetimes reuse the same
    offset; overlapping lifetimes never alias.
  - Total planned size ≤ the configured budget (PRD: cap ~1 GB for the LLM;
    peak ~300–500 MB for a 4-bit 0.5B model). If exceeded, the plan signals the
    runtime to disable optional features (compression/speculation) or spill to
    flash rather than crash.
  - Constant weights are `memmap2`-mapped read-only (Zero-Copy Load), not copied
    into the arena.

### `Arena` (Entity)
- **Holds:** the single page-aligned contiguous buffer (for DMA) — a Rust
  `Box<[u8]>` / bump allocator — split across `MemoryTier`s.
- **Invariant:** allocated once at session init (no `malloc` in the Rust hot
  path); never grown during decode; page-aligned.

### `KvRegion` (Entity)
- **Holds:** the contiguous KV-cache slice + tensor descriptors.
- **Invariant:** **KV Compaction** is a descriptor/pointer shuffle only — the
  underlying data is not copied, and the GPU/Metal/WebGPU graph always sees a
  contiguous region.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `TensorOffset` | byte offset into the arena | |
| `MemoryTier` | `Sram` \| `Dram` | hot (KV) vs constant (weights) |
| `BufferLifetime` | [firstUse, lastUse] step range | drives reuse packing |
| `MemoryBudget` | bytes cap | from `SessionConfig` / device profile |
| `MemoryHighWaterMark` | peak bytes used | reported to Telemetry |

## Domain Services

- **`StaticMemoryPlanner`** — a pure-Rust planner that computes lifetimes and
  packs offsets into the arena (reproducing ExecuTorch-style static memory
  planning in Rust).
- **`Compactor`** — performs descriptor-only KV compaction after pruning.
- **`WeightMapper`** — maps constant weight pages via `memmap2` (Zero-Copy Load),
  enabling cross-process sharing and lazy loading.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `TierHints` | Hardware & Delegate (7) | inbound (C/S) — what can pin to SRAM |
| `Arena` / `MemoryPlan` | exposed to Inference Runtime (1) | outbound (SK) |
| host buffer / external memory ptr | host app | inbound (app-controlled memory) |

## Anti-Corruption Layer

There is no third-party planner to wrap — the `StaticMemoryPlanner` is
pure Rust. The ACL surface is narrow: `memmap2` mapping details and Candle's
storage-handle expectations sit behind `WeightMapper`, exposing only
`MemoryPlan`/`TensorOffset`/arena `Tensor` to the domain.

## Domain Events (published)

`MemoryPlanCreated`, `ArenaAllocated`, `KvCacheCompacted`, `MemoryBudgetExceeded`
(triggers feature downgrade or flash spill), `HighWaterMarkSampled`. See
[domain-events.md](../domain-events.md).

## Relationships

**Shared Kernel** with Inference Runtime (1) — they jointly own the arena/tensor
contract. Customer of Hardware & Delegate (7) for tier hints. Emits the
`MemoryBudgetExceeded` signal that drives the PRD's graceful-degradation policy.
