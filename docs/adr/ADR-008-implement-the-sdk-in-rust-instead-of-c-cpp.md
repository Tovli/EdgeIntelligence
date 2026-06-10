# ADR-008: Implement the SDK in Rust instead of C/C++

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: language, rust, memory-safety, foundational, supersedes

## Context

The original PRD and the first cut of these ADRs assumed a **C++** core compiled
to WASM, leaning on C/C++ engines (ExecuTorch, llama.cpp), a C++ WASM runtime
(WasmEdge), a C++/CUDA grammar engine (XGrammar-2), and hand-written JNI/Swift/
Objective-C bindings. The team has decided to **avoid C and C++ entirely** — in
both our own code and our chosen libraries — and implement the SDK in **Rust**.

Drivers: memory safety without a GC (critical for an inference hot loop that
manages a large arena and `mmap`-ed weights), a single codebase that compiles to
**both native ARM and `wasm32`**, a strong FFI story for the few unavoidable OS
boundaries, and a now-mature Rust ML/edge ecosystem (Candle, Wasmtime,
llguidance, `ed25519-dalek`, `memmap2`, `wgpu`, UniFFI).

This is a foundational decision and is the reason
[ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) and
[ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) are revised.

## Decision

**Implement the entire SDK in Rust (edition 2021), with no C or C++ in our
codebase or dependency set.** Concretely, the C/C++ assumptions are replaced:

| Was (C/C++) | Now (Rust) |
|-------------|------------|
| C++ core compiled to WASM | Rust core → native `aarch64` **and** `wasm32` |
| WasmEdge runtime | **Wasmtime** (Rust) — see [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) |
| ExecuTorch / llama.cpp | **Candle** (pure Rust) — see [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) |
| XGrammar-2 (C++/CUDA) | **llguidance** (Rust) |
| WASI `malloc` / C++ arena | Rust bump/arena allocator + `memmap2` |
| ED25519 via platform/bundled C | **`ed25519-dalek`** (RustCrypto) — see [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) |
| C++ wrapper + JNI + Swift/Obj-C | **UniFFI** (Kotlin/Swift) + `wasm-bindgen`/`wit-bindgen` (web) |

Unavoidable OS-level boundaries (CoreML/ANE, NNAPI/QNN, the platform keystore)
are reached through **thin Rust FFI crates** (`objc2`, `ndk`) only where present;
calling an OS C/Obj-C API across FFI is not "writing C/C++" and does not
introduce a C/C++ build dependency.

## Consequences

### Positive
- Memory safety and data-race freedom across the arena, KV-cache, and threading
  model — directly reinforcing the zero-allocation decode-loop invariant
  ([ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)).
- One language and one codebase for native ARM and WASM; smaller cognitive and
  build surface than a C++/WASM split.
- Idiomatic, auto-generated host bindings (UniFFI) instead of hand-written JNI.

### Negative
- The mobile **NPU** ecosystem is C/Obj-C-centric; pure-Rust acceleration is
  CPU(NEON)+GPU(Metal/WebGPU) by default, with NPU only as optional FFI. This is
  a real throughput trade-off versus the original NPU-first design (reflected in
  revised targets in the PRD and [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)).
- Some Rust accelerator/back-end bindings are younger and less battle-tested than
  their C++ counterparts.

### Neutral
- Research systems cited in the PRD remain valid as **technique lineage**; only
  their *implementations* change to Rust-native equivalents.

## Links
- PRD: `docs/prd.md` → "Implementation Stack (Rust)" note
- DDD: [README](../ddd/README.md), [ubiquitous-language](../ddd/ubiquitous-language.md), [context-map](../ddd/context-map.md)
- Supersedes the C++ assumption in: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)
- Related: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md)
