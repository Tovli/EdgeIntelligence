# ADR-009: flutter_rust_bridge for Dart/Flutter bindings

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: ffi, flutter, dart, bindings, mobile

## Context

Flutter is a confirmed deployment target for the SDK alongside React Native.
[ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) already
specifies **UniFFI** (Kotlin/Swift) and **`wasm-bindgen`** (TypeScript) as the
two binding layers generated from `el-ffi`. UniFFI does not generate Dart
bindings, so Flutter needs a separate mechanism.

Options considered:

| Option | Dart bindings | Rust glue | Verdict |
|--------|--------------|-----------|---------|
| `flutter_rust_bridge` v2 | Auto-generated | Auto-generated | **Chosen** |
| Raw `dart:ffi` + `cbindgen` | Hand-written | Hand-written | Too much FFI boilerplate |
| UniFFI Dart backend (community) | Experimental, unmaintained | UniFFI proc-macro | Not production-ready |

`flutter_rust_bridge` (FRB) v2 is the de-facto standard for Flutter+Rust. It
reads the Rust API directly and emits both the Rust FFI glue (`frb_generated.rs`)
and the Dart package (`lib/src/frb_generated.dart`), handling async, Result
types, opaque handles, and streams with no hand-written code.

## Decision

Add **`flutter_rust_bridge` v2** as a binding layer in `el-ffi`, alongside the
existing UniFFI and `wasm-bindgen` surfaces:

| Surface | Mechanism | Consumer |
|---------|-----------|----------|
| React Native (Android) | UniFFI → Kotlin | RN native module |
| React Native (iOS) | UniFFI → Swift | RN native module |
| Flutter (Android + iOS) | `flutter_rust_bridge` → Dart | Flutter plugin |
| Web / npm | `wasm-bindgen` → TypeScript | ESM/CJS npm package |

All four surfaces are generated from **one Rust API** in `el-ffi`. The
compile targets for Flutter (`aarch64-linux-android`,
`x86_64-linux-android` for emulators, `aarch64-apple-ios`,
`x86_64-apple-ios` for simulators) overlap with the UniFFI targets — no
new toolchains beyond what ADR-001 already requires.

`flutter_rust_bridge` is added as a `[build-dependencies]` entry in
`el-ffi/Cargo.toml`; the `flutter_rust_bridge_codegen` binary runs as
part of the Flutter plugin build step (not the core Cargo workspace
build), keeping the offline-build guarantee of the 7 core crates intact.

## Consequences

### Positive
- Flutter support without hand-written `dart:ffi` glue or C header maintenance.
- Async Dart API (`Future<String>`) maps naturally onto Rust's
  `async`/channel model without extra bridging.
- FRB v2 supports opaque Rust types as Dart handles — `EdgeLlm` can be
  passed around in Dart without exposing internals.
- Same native compile targets as UniFFI — no toolchain expansion.

### Negative
- A second code-generator (`flutter_rust_bridge_codegen`) in the dev
  toolchain alongside `uniffi-bindgen`.
- FRB generates sizeable Dart boilerplate (`frb_generated.dart`) that must
  be committed or regenerated in CI.
- `el-ffi` must keep its public API compatible with FRB's supported type
  set (primitives, `String`, `Vec<u8>`, opaque handles, `Result` — no
  raw pointers across the boundary).

### Neutral
- The Flutter plugin is a separate pub.dev package wrapping the compiled
  `.so`/`.dylib`; the Rust workspace itself is unchanged.

## Links
- Supersedes: nothing; extends ADR-001's binding strategy
- Related: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md),
  [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
- Crate: `crates/adapters/el-ffi`
