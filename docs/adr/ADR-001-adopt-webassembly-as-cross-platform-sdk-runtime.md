# ADR-001: WebAssembly (Wasmtime) as the portable target, alongside native ARM

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: runtime, portability, wasm, rust, core

## Context

The SDK must run the same inference engine across Android and iOS without
maintaining divergent codebases, while staying close to native performance on
ARM. The PRD (§"Proposed Edge-Native Pipeline", §"Runtime & Target Selection")
requires a single portable core that host apps embed via thin platform bindings.

Per [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) the core
is **Rust**, which changes the runtime story: Rust cross-compiles directly to
native ARM *and* to `wasm32`, so WASM is no longer the only portability mechanism
— it becomes the sandboxed/portable option next to a faster native path. It also
rules out the originally-preferred WasmEdge runtime, which is C++.

This decision shapes the
[Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md) context's
public boundary and how every other context is packaged.

## Decision

Compile the Rust core to **two targets from one codebase**:

1. **Native ARM** (`aarch64-linux-android`, `aarch64-apple-ios`) for maximum
   performance — the default on-device path.
2. **`wasm32`** (`wasm32-wasip1` / `wasm32-unknown-unknown`) for the portable,
   sandboxed, OTA-friendly path, executed by **Wasmtime** (a Rust WASM runtime;
   **WasmEdge is rejected because it is C++**). WebGPU compute in WASM is reached
   via `wgpu`.

The Published Language `init / load_prompt / generate / commit / reset` is
identical across targets. Host bindings are generated from **one Rust API in
`el-ffi`** across three surfaces — no hand-written JNI, Objective-C, or Dart FFI:

| Consumer | Tool | Output |
|----------|------|--------|
| React Native (Android + iOS) | **`uniffi-bindgen-react-native`** | TypeScript + JSI C++ + Turbo Module |
| Flutter (Android + iOS + desktop) | **`flutter_rust_bridge` v2** | Dart package + Rust glue (see [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md)) |
| Web / npm | **`wasm-bindgen`** | TypeScript ESM package |

`uniffi-bindgen-react-native` (Mozilla/Filament, 2024) generates TypeScript and
JSI C++ directly from the same UniFFI definitions — bypassing the older
UniFFI → Kotlin/Swift → hand-wired RN native module path. Data crosses the
boundary via shared linear memory (WASM) or shared `Arc<[u8]>` buffers (native).

## Consequences

### Positive
- One Rust codebase serves native ARM, mobile, and web; native gives full speed,
  `wasm32` gives sandboxing and an OTA-updatable logic path.
- Wasmtime keeps the entire stack in Rust (no C++ runtime dependency).
- `uniffi-bindgen-react-native` generates TypeScript + JSI C++ directly —
  no hand-written JNI, Obj-C, or intermediate Kotlin/Swift bridge layer.
- `flutter_rust_bridge` v2 gives Flutter a `Stream<String>` token callback
  and opaque handles without any hand-written `dart:ffi`.

### Negative
- Two active targets (native + wasm32) widen the test matrix and require
  feature-gating accelerator code (Metal/WebGPU/FFI availability differs).
- The `wasm32` path carries a residual (~10%) overhead vs native and has
  narrower accelerator access (WebGPU yes; OS NPU FFI typically native-only).

### Neutral
- OS accelerators (Metal, WebGPU, NNAPI/CoreML) are brokered per target — see
  [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) and the
  [Hardware & Delegate](../ddd/bounded-contexts/07-hardware-delegate.md) context.

## Links
- PRD: `docs/prd.md` §"Proposed Edge-Native Pipeline" → "Runtime & Target Selection"
- DDD: [Inference Runtime context](../ddd/bounded-contexts/01-inference-runtime.md)
- Driven by: [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
- Extended by: [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) (Flutter bindings)
- Related: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
