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

> **New here?** Jump to [Architecture and entry points](#architecture-and-entry-points)
> to see where the SDK starts and how the crates fit together, then
> [Quick start](#quick-start) to build it.

## Contents

- [Why Edge Intelligence?](#why-edge-intelligence)
- [Architecture and entry points](#architecture-and-entry-points)
- [Quick start](#quick-start)
- [Local chat test client](#local-chat-test-client)
- [Workspace map](#workspace-map)
- [Architecture decisions](#architecture-decisions)
- [Domain model](#domain-model)
- [Roadmap](#roadmap)
- [Documentation](#documentation)

## Why Edge Intelligence?

Most mobile LLM stacks start in the cloud and add local features later. This
project starts at the edge.

<details>
<summary><b>The six principles that shape every crate</b></summary>

| Principle | What it means in the SDK |
|-----------|--------------------------|
| **Local first** | The runtime, memory planner, safety checks, and telemetry core have no network dependency. |
| **Rust all the way down** | Project-owned SDK code avoids C/C++ and keeps unsafe code out of the core crates. |
| **Predictable memory** | The decode loop uses pre-planned arenas and descriptor-based KV-cache management. |
| **Structured output** | Grammar masks run before sampling so tool calls and JSON outputs stay valid. |
| **Provable provenance** | Model signatures are verified before a session can be constructed. |
| **Opt-in egress** | Cloud/frontier providers are a separate adapter and must be wired deliberately by the host app. |

</details>

## Architecture and entry points

Edge Intelligence is a **hexagonal (ports-and-adapters) workspace, not a single
monolithic crate**. There is no `edge-intelligence` umbrella crate; coherence
comes from three layers, each with a clear entry point:

| Layer | Crate · symbol | Where it fits |
|-------|----------------|---------------|
| **Device SDK facade** | [`el-ffi`](crates/adapters/el-ffi) · `EdgeLlm` | The composition root shipped to devices. Wires the local engine and opt-in cloud behind one flat API and projects it to React Native (UniFFI/JSI), Flutter (FRB), and Web (wasm-bindgen). **Start here to build an app.** |
| **Rust API seam** | [`el-core`](crates/el-core) · `LlmProvider` | The single trait every backend implements and every Rust consumer calls (ADR-010). **Start here to embed the SDK in Rust.** |
| **Orchestrator** | [`el-runtime`](crates/el-runtime) · `InferenceSession` | Composes provenance, memory, safety, and grammar into the decode loop — the engine the providers drive. |

```text
  Edge device app  ·  Kotlin · Swift · Dart · TypeScript
          │
          ▼
  el-ffi · EdgeLlm                      ← device SDK entry point (composition root)
          │
          ▼
  el-core · LlmProvider (trait)         ← unified Rust API seam (ADR-010)
          │
     ┌────┴───────────────┐
     ▼                     ▼
  el-engine-candle      el-cloud        ← backends: local (default) · opt-in frontier
     │
     ▼
  el-runtime · InferenceSession         ← orchestrator
     │   composes
     ▼
  el-memory · el-provenance · el-safety · el-grammar
```

<details>
<summary><b>Which entry point should I use?</b></summary>

- **Building a mobile or web app** → use [`el-ffi`](crates/adapters/el-ffi)'s
  `EdgeLlm`. Construct `EdgeLlm::local(model_uri)` (air-gapped) or
  `EdgeLlm::cloud(model, api_key)` (opt-in), then call `ask(...)` /
  `ask_stream(...)`. The crate compiles to a native library and a wasm package
  and ships generated TypeScript/Dart bindings.
- **Embedding the SDK in Rust** → construct a concrete provider and talk to it
  through [`el_core::LlmProvider`](crates/el-core): `el_engine_candle::QwenChatProvider`
  for on-device chat, or `el_cloud::CloudProvider` for a frontier backend.
  [`apps/el-chat`](apps/el-chat) is a worked example.
- **Extending the SDK** (new engine, grammar, safety, or compression) →
  implement the matching port trait from [`el-runtime`](crates/el-runtime)
  (`InferenceEngine`, `GrammarMasker`, `SafetySteerer`, `PromptCompressor`),
  or implement `LlmProvider` directly for a whole new backend.

</details>

<details>
<summary><b>The per-token decode pipeline and SDK seams</b></summary>

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

The workspace proves the main seams of the SDK:

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
- **Inference engine seam:** `el-engine-candle` runs a real Candle CPU forward —
  a single-projection seam proof plus a real Qwen2 transformer — and drives the
  runtime loop end to end.
- **Provider seam:** `el-core::LlmProvider` gives local and frontier backends one
  host-facing API; `el-cloud` implements the opt-in OpenAI-compatible path.

</details>

## Quick start

Prerequisite: Rust 1.96 or newer, matching the workspace `rust-version`.

```sh
cargo build --workspace
cargo test --workspace
```

<details>
<summary><b>Build just the dependency-light core, or cross-compile to WASM</b></summary>

Build and test only the pure-Rust local core (no Candle, no network):

```sh
cargo test -p el-core -p el-memory -p el-telemetry -p el-provenance -p el-safety -p el-runtime -p el-grammar
```

Cross-compile that core for WASM:

```sh
rustup target add wasm32-wasip1 wasm32-unknown-unknown

cargo build --target wasm32-wasip1 -p el-core -p el-memory -p el-telemetry -p el-provenance -p el-safety -p el-runtime -p el-grammar
```

Cross-compile the device bindings (`el-ffi`) for Android / iOS / Web via the
[`Makefile`](Makefile): `make build-android`, `make build-ios`, `make build-wasm`,
or `make bindings` for all three codegen surfaces.

</details>

## Local chat test client

[`apps/el-chat`](apps/el-chat) is an interactive REPL that holds a real
multi-turn conversation with a small LLM running **entirely on-device**. It
exists to exercise the SDK end-to-end, so its only direct dependencies are SDK
crates (`el-core`, `el-engine-candle`) — it contains no inference, model, or
tokenizer code of its own.

<details>
<summary><b>Fetch a model and run it</b></summary>

Every reply flows through the ADR-010 `LlmProvider` seam:

```text
el-chat  →  el_core::LlmProvider  →  el_engine_candle::QwenChatProvider
                                       (real Qwen2 forward via candle-transformers)
                                  →  el_runtime::InferenceSession
                                       (provenance gate → prefill → decode loop)
```

Decoding is the runtime's deterministic greedy argmax, so replies are
reproducible. The model is supplied as a local file — there is no runtime
network egress (ADR-004 air-gap by default). Fetch a small instruct model once:

```sh
mkdir -p models
curl -sSL -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
curl -sSL -o models/qwen2.5-0.5b-instruct.tokenizer.json \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
```

Then chat (the defaults point at the files above):

```sh
cargo run -p el-chat                                  # interactive REPL
cargo run -p el-chat -- --prompt "Hello!" --once      # one-shot
cargo run -p el-chat -- --system "Be terse." --max-tokens 128
```

REPL commands: `/reset`, `/system <text>`, `/help`, `/exit`. Other flags:
`--model`, `--tokenizer`, `--system`, `--max-tokens`. The `models/` directory is
git-ignored. See [`apps/el-chat/README.md`](apps/el-chat/README.md) for the full
user guide.

</details>

## Workspace map

Twelve crates plus the chat app. Each crate has its own README (linked below)
covering its public API, a usage example, and the ADRs it realizes.

<details>
<summary><b>Show the full crate table</b></summary>

| Crate | Role | Current state |
|-------|------|---------------|
| [`crates/el-core`](crates/el-core) | Shared types, IDs, errors, events, `LlmProvider` trait | Implemented and tested |
| [`crates/el-memory`](crates/el-memory) | Static arena planning and KV-cache descriptors | Implemented and tested |
| [`crates/el-telemetry`](crates/el-telemetry) | Content-free event handling and privacy metrics | Implemented and tested |
| [`crates/el-provenance`](crates/el-provenance) | Verified model load permits | Implemented and tested |
| [`crates/el-safety`](crates/el-safety) | Tiered decoder-time safety policy | Partial, lightweight path implemented |
| [`crates/el-runtime`](crates/el-runtime) | Session lifecycle and decode-loop orchestration | Implemented and tested |
| [`crates/el-grammar`](crates/el-grammar) | DFA grammar masking | Implemented and tested |
| [`crates/adapters/el-provenance-ed25519`](crates/adapters/el-provenance-ed25519) | Real ED25519 signature verification | Implemented and tested |
| [`crates/adapters/el-engine-candle`](crates/adapters/el-engine-candle) | Candle inference adapter: engine-seam proof plus a real Qwen2 transformer engine and chat provider | Implemented; real on-device chat |
| [`crates/adapters/el-cloud`](crates/adapters/el-cloud) | Opt-in OpenAI-compatible provider backend | Implemented; egress opt-in at construction |
| [`crates/adapters/el-ffi`](crates/adapters/el-ffi) | **Device SDK facade (`EdgeLlm`):** Flutter / UniFFI / wasm-bindgen binding surfaces | Implemented and tested; host build is a workspace member, cross-target builds via `make` |
| [`crates/adapters/el-grammar-llguidance`](crates/adapters/el-grammar-llguidance) | llguidance JSON-schema token masking | Implemented and tested; workspace-excluded (crates.io deps) |
| [`apps/el-chat`](apps/el-chat) | Interactive chat test client; SDK-only deps, drives the runtime end-to-end | Implemented; runs real on-device chat |

Of the adapters, only `el-grammar-llguidance` is excluded from the default
workspace build (it pulls crates.io-only grammar dependencies); `el-cloud` and
`el-ffi` are regular members whose host targets build and test with
`cargo test --workspace`.

</details>

## Architecture decisions

The project is intentionally decision-heavy because mobile LLM runtimes are easy
to overfit to one device, model, or provider. The core choices are recorded as
ADRs.

<details>
<summary><b>The ten architecture decision records</b></summary>

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

</details>

## Domain model

The DDD model lives in [`docs/ddd`](docs/ddd/README.md). The key invariant
across its contexts: **air-gap is the default runtime shape, not a feature flag
sprinkled through the code** — any outbound behavior must be modeled as an
explicit port or adapter.

<details>
<summary><b>The nine bounded contexts</b></summary>

1. Inference Runtime
2. Prompt Compression
3. Speculative Decoding
4. Grammar Constraint
5. Safety
6. Memory Management
7. Hardware and Delegate
8. Model Provenance and Security
9. Telemetry and Privacy

</details>

## Roadmap

The prototype has proven the architectural seams. The next engineering work is
to replace toy proofs with production-grade runtime pieces.

<details>
<summary><b>What's next</b></summary>

- Production GGUF/safetensors loading and transformer execution in
  `el-engine-candle`.
- Lark/CFG grammars in the llguidance adapter (JSON-schema masking is done).
- SecDecoding/CSD-style model-backed safety with runtime backtracking.
- Binding codegen and packaging — FRB Dart codegen, uniffi-bindgen-react-native,
  wasm-pack npm publishing (the Rust binding surfaces exist in `el-ffi`).
- Mobile toolchain validation for Android and iOS `aarch64` targets.
- On-device benchmarks for time-to-first-token, decode throughput, memory
  high-water marks, and thermal behavior.

</details>

## Documentation

- Product and technical rationale: [`docs/prd.md`](docs/prd.md)
- Domain model: [`docs/ddd/README.md`](docs/ddd/README.md)
- Architecture decisions: [`docs/adr/README.md`](docs/adr/README.md)
- Per-crate guides: see the [Workspace map](#workspace-map) — every crate links to its own README.

Edge Intelligence is still early, but the direction is deliberate: a small,
auditable, Rust-native SDK that lets app developers choose local inference first
and add remote intelligence only when the product explicitly calls for it.
