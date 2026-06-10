# Bounded Context 7 — Hardware & Delegate  (Generic)

> Detects device capabilities, classifies the device profile, and partitions the
> compute graph across available accelerators with CPU fallback.
> Source: PRD §"Device Profiles", §"Delegate Mapping (Heterogeneous
> Acceleration)", §Risks → "Hardware Variability".

## Purpose

Turn whatever silicon is present into an ordered, safe execution plan, so every
other context can request acceleration without knowing the device.

## Strategic role

**Generic.** Capability detection and delegate routing are commodity concerns;
correctness and breadth matter more than novelty. Default-to-CPU keeps
compatibility broad.

## Ubiquitous language (context-local)

`Device Profile`, `Delegate`, `Delegate Plan`, `Capability Detection`, `Graph
Partition`, `TOPS`, `Bandwidth`.

## Aggregates

### `DeviceProfile` (Aggregate Root)
The resolved capabilities of the running device.

- **Identity:** `DeviceProfileId` (per device/session).
- **Holds:** `ProfileKind` (`MidRange`/`HighEnd`), RAM, memory `Bandwidth`,
  NPU `Tops`, and the set of available `Delegate`s.
- **Invariants:**
  - The **Candle CPU** path (Rust NEON/SIMD) is **always** available as a
    fallback (PRD: "default to the always-available Candle CPU path").
  - `ProfileKind` is derived from probed facts, not trusted from the app.
  - Hot-plug: if a faster delegate becomes available mid-session, a new plan may
    be proposed (the architecture allows running CPU until a delegate is ready).

### `DelegatePlan` (Aggregate Root)
The ordered partition of the graph across delegates.

- **Identity:** `DelegatePlanId` (bound to a `Model` + `DeviceProfile`).
- **Holds:** an ordered list of `GraphPartition`→`Delegate` assignments.
- **Invariants:**
  - Priority order is honored (per `docs/adr/ADR-002`): (1) NPU via OS FFI
    (`Nnapi`/`Qnn`/`CoreMl`) where present **and** beneficial, (2) GPU
    (`Metal` on Apple / `WebGpu` via `wgpu`), (3) Candle **CPU** (NEON) — the
    universal fallback. This is CPU/GPU-centric, not NPU-first.
  - Every partition resolves to an available delegate or the Candle CPU fallback
    — no partition is left unassigned.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `ProfileKind` | `MidRange` \| `HighEnd` | per PRD device profiles |
| `Delegate` | `Cpu` (Candle NEON) \| `Metal` \| `WebGpu` \| optional `Nnapi`/`Qnn` \| `CoreMl` | `Cpu` always present; others where available (NPU via FFI) |
| `Tops` | NPU throughput class (e.g. 2–5, 20–60) | from detection |
| `Bandwidth` | GB/s (e.g. ~30 mid, 60–90 high) | from detection |
| `GraphPartition` | a contiguous subgraph | unit of delegation |
| `TierHint` | `Sram`/`Dram` placement advice | sent to Memory (6) |

## Domain Services

- **`CapabilityDetector`** — probes NPUs/GPUs/instruction sets (ARMv9 i8mm,
  SVE2, SME) and builds the `DeviceProfile`.
- **`GraphPartitioner`** — assigns subgraphs to delegates per the priority order,
  with CPU fallback.
- **`DelegateRouter`** — binds partitions to the concrete backend (Candle
  CPU/Metal, `wgpu` WebGPU, or optional NNAPI/QNN/CoreML via FFI).

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `PlatformAccelerators` | Candle CPU/Metal, `wgpu`; NNAPI/QNN/CoreML via thin FFI | inbound |
| `DelegatePlanProvider` | exposed to Inference Runtime (1) | outbound (C/S, U) |
| `DeviceProfile` reader | Safety (5), Speculative (3) | outbound (gates their modes) |
| `TierHints` | exposed to Memory (6) | outbound (C/S) |

## Anti-Corruption Layer

`DelegateAcl` wraps Candle's backends (CPU/Metal), `wgpu` (WebGPU), and the
thin FFI shims (`ndk` → NNAPI/QNN, `objc2` → CoreML) used only where a dedicated
accelerator is present. It exposes a uniform `probe()` / `dispatch(partition)` so
no backend or OS-SDK type reaches the domain.

## Domain Events (published)

`DeviceProfiled`, `DelegateSelected`, `DelegateFellBack` (NPU/GPU unavailable →
CPU), `DelegateHotPlugged`. See [domain-events.md](../domain-events.md).

## Relationships

**Upstream** (Customer/Supplier) of Inference Runtime (1) and Memory (6); a
read-only capability source for Safety (5) and Speculative Decoding (3), which
gate their expensive modes on `ProfileKind`/`Tops`.
