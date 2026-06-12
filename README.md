![Edge Intelligence banner](docs/assets/edge-intelligence-banner.png)

# Edge Intelligence

![Rust](https://img.shields.io/badge/Rust-2021-f46623?style=for-the-badge&logo=rust&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20prototype-2563eb?style=for-the-badge)
![Privacy](https://img.shields.io/badge/default-air--gapped-10b981?style=for-the-badge)
![Targets](https://img.shields.io/badge/targets-native%20ARM%20%2B%20WASM-7c3aed?style=for-the-badge)
![License](https://img.shields.io/badge/license-Apache--2.0-f59e0b?style=for-the-badge)

**A Rust SDK for private, edge-native LLM applications.**

Edge Intelligence is an offline-first runtime for small language models on phones,
embedded devices, and portable WASM hosts. It is built around one constraint:
the useful parts of an LLM app should keep working when the network disappears
and sensitive user data should not need to leave the device.

The SDK targets approximately 0.5B parameter models, with a pure Rust core,
static memory planning, signed model loading, grammar-constrained decoding,
privacy-preserving telemetry, and host bindings for mobile and web runtimes.
Local inference is the default path. Frontier and OpenAI-compatible providers
exist only behind an explicit opt-in backend.

## Why Edge Intelligence?

Most mobile LLM stacks start in the cloud and add local features later. This
project starts at the edge:

| Principle | What it means in the SDK |
|-----------|--------------------------|
| **Local first** | The runtime, memory planner, safety checks, and telemetry core have no network dependency. |
| **Rust all the way down** | Project-owned SDK code avoids C/C++ and keeps unsafe code out of the core crates. |
| **Predictable memory** | The decode loop uses pre-planned arenas and descriptor-based KV-cache management. |
| **Structured output** | Grammar masks run before sampling so tool calls and JSON outputs stay valid. |
| **Provable provenance** | Model signatures are verified before a session can be constructed. |
| **Opt-in egress** | Cloud/frontier providers are a separate adapter and must be wired deliberately by the host app. |

## What It Does

```text
Host app
  |
  v
LlmProvider trait
  |------------------------------|
  v                              v
Local Candle runtime         Opt-in cloud adapter
  |
  v
load gate -> memory plan -> prefill -> decode loop
                                      |
                                      v
                         grammar mask -> safety adjust -> sample -> commit KV
                                      |
                                      v
                         content-free events and metrics
```

The current workspace proves the main seams of the SDK:

- **Runtime orchestration:** `el-runtime` owns the session state machine and
  enforces the decode order: grammar mask, safety adjustment, sampling, commit.
- **Static memory:** `el-memory` plans tensor lifetimes into a reusable arena and
  models descriptor-only KV-cache compaction.
- **Model provenance:** `el-provenance` and `el-provenance-ed25519` implement the
  hard load gate: no verified signature, no load permit, no session.
- **Grammar constraints:** `el-grammar` compiles regular grammars into a
  token-level DFA masker; the `el-grammar-llguidance` adapter provides real
  JSON-schema masking over llguidance with a HuggingFace tokenizer bridge.
- **Safety:** `el-safety` provides the tiered policy model and lightweight
  blacklist steering path, with SecDecoding-style model-backed safety tracked as
  follow-up work.
- **Inference engine seam:** `el-engine-candle` runs a real Candle CPU forward
  on a toy in-code model and drives the runtime loop end to end.
- **Provider seam:** `el-core::LlmProvider` gives local and frontier backends one
  host-facing API; `el-cloud` implements the opt-in OpenAI-compatible path.

## Quick Start

Prerequisite: Rust 1.96 or newer, matching the workspace `rust-version`.

```sh
cargo build --workspace
cargo test --workspace
```

Build just the dependency-light local core:

```sh
cargo test -p el-core -p el-memory -p el-telemetry -p el-provenance -p el-safety -p el-runtime -p el-grammar
```

Cross-compile the pure Rust core for WASM:

```sh
rustup target add wasm32-wasip1 wasm32-unknown-unknown

cargo build --target wasm32-wasip1 -p el-core -p el-memory -p el-telemetry -p el-provenance -p el-safety -p el-runtime -p el-grammar
```

## Workspace Map

| Crate | Role | Current state |
|-------|------|---------------|
| [`crates/el-core`](crates/el-core) | Shared types, IDs, errors, events, provider trait | Implemented and tested |
| [`crates/el-memory`](crates/el-memory) | Static arena planning and KV-cache descriptors | Implemented and tested |
| [`crates/el-telemetry`](crates/el-telemetry) | Content-free event handling and privacy metrics | Implemented and tested |
| [`crates/el-provenance`](crates/el-provenance) | Verified model load permits | Implemented and tested |
| [`crates/el-safety`](crates/el-safety) | Tiered decoder-time safety policy | Partial, lightweight path implemented |
| [`crates/el-runtime`](crates/el-runtime) | Session lifecycle and decode-loop orchestration | Implemented and tested |
| [`crates/el-grammar`](crates/el-grammar) | DFA grammar masking | Implemented and tested |
| [`crates/adapters/el-provenance-ed25519`](crates/adapters/el-provenance-ed25519) | Real ED25519 signature verification | Implemented and tested |
| [`crates/adapters/el-engine-candle`](crates/adapters/el-engine-candle) | Candle inference adapter | Host CPU proof implemented |
| [`crates/adapters/el-cloud`](crates/adapters/el-cloud) | Opt-in OpenAI-compatible provider backend | Implemented as an explicit egress adapter |
| [`crates/adapters/el-grammar-llguidance`](crates/adapters/el-grammar-llguidance) | llguidance JSON-schema token masking | Implemented and tested; workspace-excluded (crates.io deps) |
| [`crates/adapters/el-ffi`](crates/adapters/el-ffi) | Flutter/UniFFI/wasm-bindgen binding surfaces | Implemented and tested (native + wasm32 compile); workspace-excluded (cross toolchains) |

## Architecture Decisions

The project is intentionally decision-heavy because mobile LLM runtimes are easy
to overfit to one device, model, or provider. The core choices are recorded as
ADRs:

| ADR | Decision |
|-----|----------|
| [ADR-001](docs/adr/ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) | Native ARM plus WASM as first-class runtime targets |
| [ADR-002](docs/adr/ADR-002-candle-as-rust-native-inference-engine.md) | Candle as the Rust-native inference engine |
| [ADR-003](docs/adr/ADR-003-static-memory-planning-with-zero-allocation-arena.md) | Static memory planning with a zero-allocation arena |
| [ADR-004](docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) | Air-gapped by default |
| [ADR-005](docs/adr/ADR-005-on-device-only-tiered-decoder-time-safety.md) | On-device decoder-time safety |
| [ADR-006](docs/adr/ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) | Mandatory ED25519 model-signature verification |
| [ADR-007](docs/adr/ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md) | Content-free domain events for privacy-preserving telemetry |
| [ADR-008](docs/adr/ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) | Rust instead of C/C++ |
| [ADR-009](docs/adr/ADR-009-flutter-rust-bridge-for-dart-bindings.md) | Flutter Rust Bridge for Dart bindings |
| [ADR-010](docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md) | Unified local/cloud `LlmProvider` trait |

See the full index in [`docs/adr/README.md`](docs/adr/README.md).

## Domain Model

The DDD model lives in [`docs/ddd`](docs/ddd/README.md). It breaks the SDK into
nine bounded contexts:

1. Inference Runtime
2. Prompt Compression
3. Speculative Decoding
4. Grammar Constraint
5. Safety
6. Memory Management
7. Hardware and Delegate
8. Model Provenance and Security
9. Telemetry and Privacy

The key invariant across those contexts: **air-gap is the default runtime shape,
not a feature flag sprinkled through the code.** Any outbound behavior must be
modeled as an explicit port or adapter.

## What Is Next

The prototype has proven the architectural seams. The next engineering work is
to replace toy proofs with production-grade runtime pieces:

- Production GGUF/safetensors loading and transformer execution in
  `el-engine-candle`.
- Lark/CFG grammars in the llguidance adapter (JSON-schema masking is done).
- SecDecoding/CSD-style model-backed safety with runtime backtracking.
- Binding codegen and packaging — FRB Dart codegen, uniffi-bindgen-react-native,
  wasm-pack npm publishing (the Rust binding surfaces exist in `el-ffi`).
- Mobile toolchain validation for Android and iOS `aarch64` targets.
- On-device benchmarks for time-to-first-token, decode throughput, memory
  high-water marks, and thermal behavior.

## Documentation

- Product and technical rationale: [`docs/prd.md`](docs/prd.md)
- Domain model: [`docs/ddd/README.md`](docs/ddd/README.md)
- Architecture decisions: [`docs/adr/README.md`](docs/adr/README.md)

Edge Intelligence is still early, but the direction is deliberate: a small,
auditable, Rust-native SDK that lets app developers choose local inference first
and add remote intelligence only when the product explicitly calls for it.
