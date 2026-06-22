# Architecture Decision Records

Significant, hard-to-reverse architectural decisions for the Edge-Native LLM SDK,
derived from [`docs/prd.md`](../prd.md) and the DDD model in
[`docs/ddd/`](../ddd/README.md). The SDK is implemented in **Rust** (no C/C++ —
see ADR-008). Each ADR is registered in AgentDB (semantic tier) and searchable in
the `adr-patterns` namespace.

| ADR | Title | Status | Tags |
|-----|-------|--------|------|
| [001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) | WebAssembly (Wasmtime) as the portable target, alongside native ARM | accepted | runtime, wasm, rust, core |
| [002](./ADR-002-candle-as-rust-native-inference-engine.md) | Candle as the Rust-native inference engine | accepted | runtime, candle, rust, core |
| [003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) | Static memory planning with a zero-allocation arena (Runtime↔Memory Shared Kernel) | accepted | memory, shared-kernel, core |
| [004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) | Air-gapped by default with opt-in local-network HybridMode | accepted | privacy, networking, invariant |
| [005](./ADR-005-on-device-only-tiered-decoder-time-safety.md) | On-device-only, tiered decoder-time safety (no cloud moderation) | accepted | safety, compliance, supporting |
| [006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) | Mandatory ED25519 model-signature verification as a hard load gate | accepted | security, provenance, generic |
| [007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md) | Content-free domain events for privacy-by-construction telemetry | accepted | privacy, telemetry, generic |
| [008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) | Implement the SDK in Rust instead of C/C++ | accepted | language, rust, foundational |
| [009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) | flutter_rust_bridge v2 for Dart/Flutter bindings | accepted | ffi, flutter, dart, mobile |
| [010](./ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md) | Unified LlmProvider trait with opt-in frontier LLM cloud egress | accepted | llm, cloud, networking, trait |
| [011](./ADR-011-multi-registry-release-ci-crates-io-npm-pub-dev.md) | Multi-Registry Release CI (crates.io, npm, pub.dev) | accepted | ci, release, crates.io, npm, pub.dev, packaging |
| [012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) | Layered decode-time safety control loop with checkpointed rollback | accepted | safety, security, on-device, runtime, supporting |
| [013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md) | Model-backed steering layers for the hybrid safety control loop | proposed | safety, security, on-device, runtime, follow-up |
| [018](./ADR-018-persistent-model-instances-and-stateful-sessions.md) | Persistent model instances and stateful inference sessions | proposed | runtime, performance, on-device, follow-up, P0 |
| [019](./ADR-019-in-loop-incremental-decoding-and-token-streaming.md) | In-loop incremental decoding and real token streaming | proposed | runtime, performance, bindings, follow-up, P0 |
| [020](./ADR-020-batched-single-pass-prefill.md) | Batched (single-pass) prefill | proposed | runtime, performance, on-device, follow-up, P0 |
| [021](./ADR-021-memory-mapped-verified-gguf-loading.md) | Memory-mapped verified GGUF loading | proposed | runtime, performance, provenance, on-device, follow-up, P0 |
| [022](./ADR-022-two-tier-quantized-kv-cache-with-attention-aware-eviction.md) | Two-tier quantized KV cache with attention-aware eviction | proposed | memory, runtime, performance, on-device, follow-up, P0 |
| [023](./ADR-023-baseline-performance-instrumentation.md) | Baseline performance instrumentation | proposed | telemetry, performance, testing, follow-up, P0 |

## Decision relationships

```mermaid
flowchart LR
    A008[008 Rust, no C/C++] --> A001[001 WASM/Wasmtime + native ARM]
    A008 --> A002[002 Candle engine]
    A008 --> A006[006 ed25519-dalek load gate]
    A001 --> A002
    A001 --> A009[009 flutter_rust_bridge]
    A002 --> A003[003 Static memory / Shared Kernel]
    A002 --> A010[010 LlmProvider + el-cloud]
    A004[004 Air-gap + HybridMode] --> A005[005 On-device tiered safety]
    A004 --> A006
    A004 --> A007[007 Content-free telemetry]
    A004 -. partially amended by .-> A010
    A003 -. budget triggers degradation .-> A005
    A005 --> A012[012 Decode-time safety loop + rollback]
    A012 --> A013[013 Model-backed steering layers]
    A003 -. budget triggers degradation .-> A012
    A006 -. signs safety weights .-> A013

    %% P0 performance milestone (docs/research/improvements-plan.md, EPIC 1-4)
    A010 --> A018[018 Persistent model + stateful sessions]
    A018 --> A019[019 In-loop streaming]
    A012 -. emit only guard-verified tokens .-> A019
    A018 --> A020[020 Batched prefill]
    A018 --> A021[021 Mmap'd verified load]
    A006 -. verify before map .-> A021
    A018 --> A022[022 Two-tier quantized KV + eviction]
    A003 -. KV budget / degradation .-> A022
    A007 --> A023[023 Baseline instrumentation]
    A023 -. measures .-> A018
    A023 -. measures .-> A019
    A023 -. measures .-> A020
    A023 -. measures .-> A021
    A023 -. measures .-> A022
```

> **ADR-008 is foundational** (the language decision) and drives the revisions to
> ADR-001 and ADR-002. It carries a higher number only because it was recorded
> after the initial C++-assuming set; ordering is chronological, not by
> importance.

## Conventions

- **Numbering**: sequential `ADR-NNN`, never reused.
- **Status lifecycle**: `proposed → accepted → (deprecated | superseded by ADR-MMM)`.
- **New ADRs**: run `/ruflo-adr:adr-create`. Rebuild the AgentDB graph with
  `/ruflo-adr:adr-index`. Check code against decisions with `/ruflo-adr:adr-review`.
