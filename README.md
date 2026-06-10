# Edge-Native LLM SDK (Rust)

A private, offline-capable, on-device LLM SDK for ~0.5B models. Pure **Rust**, no
C/C++ (ADR-008), targeting native ARM and `wasm32`. See [`docs/prd.md`](docs/prd.md),
the DDD model in [`docs/ddd/`](docs/ddd/README.md), and the decision records in
[`docs/adr/`](docs/adr/README.md).

## Workspace layout

Member crates are **pure Rust (std-only)** — they build and test offline on the
host. Adapters that need external crates or cross-targets are **excluded** until
network + targets are available.

| Crate | Realizes | Bounded context | Status |
|-------|----------|-----------------|--------|
| `crates/el-core` | ADR-007/008 | (shared) ubiquitous language, content-free events | ✅ implemented + tested |
| `crates/el-memory` | ADR-003 | Memory Management | ✅ implemented + tested |
| `crates/el-telemetry` | ADR-007 | Telemetry & Privacy | ✅ implemented + tested |
| `crates/el-provenance` | ADR-006 | Model Provenance (gate logic) | ✅ implemented + tested |
| `crates/el-safety` | ADR-005 | Safety (Lightweight real) | ◑ partial |
| `crates/el-runtime` | ADR-001/004 | Inference Runtime + air-gap | ✅ implemented + tested |
| `crates/adapters/el-provenance-ed25519` | ADR-006 | real ED25519 (`ed25519-dalek`) | ▢ excluded — complete, build needs network |
| `crates/adapters/el-engine-candle` | ADR-002 | Candle inference engine | ▢ excluded — skeleton |
| `crates/adapters/el-grammar-llguidance` | ADR-004 | llguidance grammar masking | ▢ excluded — skeleton |
| `crates/adapters/el-ffi` | ADR-001 | UniFFI / wasm-bindgen host bindings | ▢ excluded — skeleton |

## Build & test (host)

```sh
cargo build --workspace   # zero external deps → builds offline
cargo test  --workspace   # 23 tests across the 6 member crates
```

## What's implemented vs. follow-up

**Implemented & host-verified (this increment):**
- Rust workspace, two-target config (`.cargo/config.toml` ARM target-features).
- Domain vocabulary + **content-free events enforced at compile time** (events
  derive `Copy`, so no `String`/heap field can ride on an event — ADR-007).
- Static memory planner with lifetime-based offset reuse, allocate-once arena,
  descriptor-only KV compaction (ADR-003).
- Provenance **hard load gate**: no `Verified` → no `LoadPermit` → can't build a
  session (ADR-006); a real `ed25519-dalek` verifier is written in the excluded
  adapter.
- Tiered safety with `SecDecoding`→`Lightweight` downgrade on mid-range + a real
  blacklist filter (ADR-005).
- Session state machine + decode orchestrator enforcing the invariant order
  **grammar-mask → safety-adjust → sample → commit**, plus air-gap (no network
  dependency; opt-in `HybridRelay` blocked unless `hybrid_mode`) (ADR-001/004).

**Follow-up (tracked tasks):** real Candle prefill/decode + GGUF/model assets
(ADR-002); llguidance JSON-schema masking (ADR-004); SecDecoding/CSD model-backed
safety (ADR-005); UniFFI/wasm-bindgen binding generation + `wasm32`/mobile
cross-compilation (ADR-001). Bringing the excluded adapters into the workspace
requires crates.io access (and, for Candle, a quantized model + real inference
work).
