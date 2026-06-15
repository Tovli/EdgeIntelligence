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

## Local Chat Test Client

[`apps/el-chat`](apps/el-chat) is an interactive REPL that holds a real
multi-turn conversation with a small LLM running **entirely on-device**. Its
purpose is to exercise the SDK end-to-end, so its only direct dependencies are
SDK crates (`el-core`, `el-engine-candle`) — it contains no inference, model, or
tokenizer code of its own. Every reply flows through the ADR-010
`LlmProvider` seam:

```
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
`--model`, `--tokenizer`, `--system`, `--max-tokens`. The `models/`
directory is git-ignored. Decoding is deterministic (the SDK runtime decodes
greedily), so the same prompt yields the same reply.

See [`apps/el-chat/README.md`](apps/el-chat/README.md) for the full user guide.

## Benchmarks

The SDK ships two reproducible benchmark harnesses. Both run inference through the
public `LlmProvider` seam, so they characterize the **SDK's own behavior** and are
**model-agnostic** — point them at whichever signed model your product loads. The
model is pluggable and not part of this repo, so no per-model results are published
here; run the harnesses against your own model to produce them.

**1. Runtime performance / overhead** —
[`docs/benchmarks/2026-06-14-qwen-chat-bottleneck.md`](docs/benchmarks/2026-06-14-qwen-chat-bottleneck.md)

A phase-level latency breakdown of the local decode path, behind opt-in,
zero-cost-when-unset instrumentation (`EL_BENCH=1`). The SDK-side conclusion: the
runtime's own per-token work — decode loop, grammar mask, full-vocab argmax, KV
commit, and content-free event emission — is **under ~1.2% of decode time**.
Latency is dominated by model compute plus two orchestration costs the SDK can
remove, independent of model choice:

- **Batch the prefill** — feed the prompt as one `(1, prompt_len)` forward instead
  of one forward per prompt token.
- **Load weights once; reset only the KV cache** — and reuse KV across turns, so a
  growing conversation is not re-prefilled from scratch each turn.
- **Stream tokens for real** — emit from inside the decode loop so time-to-first-
  token is `load + prefill + 1 token`, not full-generation time.

**2. Clinical-quality & safety evaluation** — [`apps/el-bench`](apps/el-bench) ·
[`benchmarks/README.md`](benchmarks/README.md)

`el-bench` is an SDK-only test client (a sibling to `el-chat`) that replays
published mental-health benchmarks — CounselBench, MindEval, and the VERA-MH
suicide-risk safety suite — through the runtime and records transcripts for
scoring against each benchmark's rubric. Datasets and transcripts are fetched or
produced locally and are git-ignored (third-party data); only the harness and the
methodology are committed. Decoding is deterministic, so a given model + task set
yields identical transcripts — it is designed to run as a **CI safety gate**, so a
change to the model, the system prompt, or the ADR-005 safety tier can be
regression-tested against a fixed rubric.

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
| [`crates/adapters/el-engine-candle`](crates/adapters/el-engine-candle) | Candle inference adapter: engine-seam proof plus a real Qwen2 transformer engine and chat provider | Implemented; real on-device chat |
| [`crates/adapters/el-cloud`](crates/adapters/el-cloud) | Opt-in OpenAI-compatible provider backend | Implemented as an explicit egress adapter |
| [`crates/adapters/el-grammar-llguidance`](crates/adapters/el-grammar-llguidance) | llguidance JSON-schema token masking | Implemented and tested; workspace-excluded (crates.io deps) |
| [`crates/adapters/el-ffi`](crates/adapters/el-ffi) | Flutter/UniFFI/wasm-bindgen binding surfaces | Implemented and tested (native + wasm32 compile); workspace-excluded (cross toolchains) |
| [`apps/el-chat`](apps/el-chat) | Interactive chat test client; SDK-only deps, drives the runtime end-to-end | Implemented; runs real on-device chat |
| [`apps/el-bench`](apps/el-bench) | Benchmark harness; SDK-only deps, replays quality/safety task sets through the runtime | Implemented; model-agnostic, reproducible |

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
- On-device benchmarks for memory high-water marks and thermal behavior, and
  wiring the `el-bench` VERA-MH safety suite into CI as a release gate (latency
  and quality/safety harnesses already exist — see [Benchmarks](#benchmarks)).

## Documentation

- Product and technical rationale: [`docs/prd.md`](docs/prd.md)
- Domain model: [`docs/ddd/README.md`](docs/ddd/README.md)
- Architecture decisions: [`docs/adr/README.md`](docs/adr/README.md)

Edge Intelligence is still early, but the direction is deliberate: a small,
auditable, Rust-native SDK that lets app developers choose local inference first
and add remote intelligence only when the product explicitly calls for it.
