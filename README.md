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
| `crates/el-grammar` | context 4 (grammar) | DFA token masking | ✅ in workspace — real, 3 tests |
| `crates/adapters/el-provenance-ed25519` | ADR-006 | real ED25519 (`ed25519-dalek`) | ✅ in workspace — real crypto, 3 tests |
| `crates/adapters/el-engine-candle` | ADR-002 | Candle inference engine | ✅ in workspace — real Candle CPU forward (toy model), 2 tests |
| `crates/adapters/el-grammar-llguidance` | context 4 (grammar) | llguidance (CFG/JSON-schema) | ▢ excluded — scale-up (needs tokenizer env) |
| `crates/adapters/el-ffi` | ADR-001 | UniFFI / wasm-bindgen host bindings | ▢ excluded — skeleton |

## Build & test (host)

```sh
cargo build --workspace   # 6 core crates are dep-free; ed25519 + candle adapters pull (cached) trees
cargo test  --workspace   # 31 tests across the 9 member crates
```

## What's implemented vs. follow-up

**Implemented & host-verified (this increment):**
- Rust workspace, two-target config (`.cargo/config.toml` ARM target-features).
- Domain vocabulary + **content-free events enforced at compile time** (events
  derive `Copy`, so no `String`/heap field can ride on an event — ADR-007).
- Static memory planner with lifetime-based offset reuse, allocate-once arena,
  descriptor-only KV compaction (ADR-003).
- Provenance **hard load gate**: no `Verified` → no `LoadPermit` → can't build a
  session (ADR-006); a real `ed25519-dalek` verifier is now a tested workspace
  member (`el-provenance-ed25519`) — genuine signatures verify, tampered/forged/
  unknown-key inputs are hard-stopped.
- **Candle engine (ADR-002):** `CandleEngine` runs a real Candle CPU forward
  (embedding × projection → quantised milli-logits at the ACL) and drives the
  `el-runtime` decode loop end-to-end. Built on a toy in-code model; production
  GGUF/safetensors loading + transformer is the documented follow-up.
- **Grammar-constrained decoding (Grammar Constraint context):** `el-grammar`
  compiles a regular grammar to a token-level DFA and masks each decode step; a
  test shows it forcing a specific output sequence despite uniform engine
  logits. Full CFG/JSON-schema via llguidance is the scale-up.
- Tiered safety with `SecDecoding`→`Lightweight` downgrade on mid-range + a real
  blacklist filter (ADR-005).
- Session state machine + decode orchestrator enforcing the invariant order
  **grammar-mask → safety-adjust → sample → commit**, plus air-gap (no network
  dependency; opt-in `HybridRelay` blocked unless `hybrid_mode`) (ADR-001/004).

**Follow-up (tracked tasks):** production GGUF/safetensors loading + real
transformer + KV wiring for Candle (ADR-002 — the engine seam itself is now
proven); llguidance JSON-schema masking (ADR-004); SecDecoding/CSD model-backed
safety with runtime backtracking (ADR-005); UniFFI/wasm-bindgen binding
generation + `wasm32`/mobile cross-compilation (ADR-001). crates.io is confirmed reachable (Increment 2
promoted the ed25519 adapter into the workspace); Candle additionally needs a
quantized model + real inference work, and `wasm32`/mobile targets must be
installed for the FFI bindings.
