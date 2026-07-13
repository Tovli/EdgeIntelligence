# ADR-024: Dart-first, platform-agnostic pub.dev SDK with Flutter mobile runtimes

- **Status**: accepted
- **Date**: 2026-06-25
- **Deciders**:
- **Tags**: dart, pub.dev, bindings, packaging, platform-agnostic

## Context

[ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) framed the
pub.dev package as a Flutter binding layer. That framing made sense when
Flutter was treated as the primary Dart consumer, but it is too narrow for the
SDK architecture.

Edge Intelligence is a platform-agnostic SDK with a Rust core and generated host
bindings. Flutter is the installation host for the pub.dev plugin, while the SDK
surface remains plain Dart for mobile, desktop, test, and embedded application
code. The package must expose one framework-neutral Dart API while packaging the
native runtime in the form each supported host requires.

The current package review also exposed a packaging smell: generated
platform-specific files were public top-level Dart libraries, so pub.dev/pana
evaluated the IO and web implementation files as independent libraries and
reported false platform incompatibilities. The deeper issue is the same: the
public package boundary should be one stable, framework-neutral Dart facade, not
generated implementation files or Flutter-specific layout.

## Decision

Supersede ADR-009's Flutter-specific API decision. The `edge_intelligence`
pub.dev artifact is a Dart-first hybrid FFI package: its public API is plain
Dart, and its Android/iOS runtime installation uses Flutter FFI plugin metadata.

The Dart package must follow these rules:

1. The public entrypoint is `lib/edge_intelligence.dart`.
2. Generated binding files live under `lib/src/` and are implementation detail.
3. Public examples live under `example/` and demonstrate both plain Dart and a
   runnable Flutter Android/iOS application.
4. `pubspec.yaml` declares both Dart and Flutter SDK environments because pub.dev
   requires a Flutter lower bound for plugin packages. The package must not add
   Flutter API dependencies; Flutter FFI plugin metadata exists solely to bundle
   native artifacts.
5. Android shared libraries and the iOS XCFramework are implementation details
   under the standard Flutter plugin directories. They do not introduce a
   Flutter-specific Dart API or platform-channel contract.
6. The release pipeline should use Dart naming (`codegen-dart`, `out/dart`,
   `packaging/dart`) for new work. Transitional Flutter-named aliases may remain
   only to avoid breaking existing automation while the implementation migrates.

`flutter_rust_bridge` remains an internal code generation and FFI mechanism. The
published facade must remain usable without importing Flutter. Mobile hosts use
the same `initEdgeIntelligence`, `EdgeLlm.local`, `ask`, and `askStream` API as
desktop Dart hosts.

The package is platform-agnostic at the SDK boundary: the same Dart API selects
the available runtime for the host. Releases declare and ship native runtimes
for Flutter Android and iOS plus Dart Linux, macOS, and Windows. Android bundles
ABI-specific shared libraries; iOS bundles a dynamic XCFramework for device and
simulator builds. Browser/WASM remains a separate npm surface until the Dart
package has a working FRB web loader. A release must ship every runtime artifact
for a declared platform or fail clearly during initialization.

## Consequences

### Positive

- Dart and Flutter consumers use one stable, framework-neutral API.
- Flutter Android/iOS applications receive the native runtime automatically.
- Flutter mobile and desktop applications use the same plain-Dart API surface;
  non-UI Dart entrypoints can use it when dependency resolution is performed by
  a Flutter SDK.
- Generated IO/web implementation files stop defining the public platform
  compatibility signal.
- The package can regain pub.dev platform and documentation points without
  overstating Flutter support.
- Release and documentation language align with the SDK's platform-agnostic
  architecture.

### Negative

- Mobile packaging adds Android/iOS project metadata and increases the pub.dev
  archive size.
- Every supported Android ABI and iOS device/simulator slice must be built and
  tested in release CI.
- Existing workflow names and generated paths that contain `flutter` must be
  migrated or kept as compatibility aliases for a transition period.
- Flutter applications that need framework-specific installation helpers may
  require a separate adapter package later.

### Neutral

- The Rust core, native targets, WASM target, and `el-ffi` ownership remain
  unchanged.
- React Native and npm/web packaging are unaffected and remain separate
  distribution surfaces for mobile and browser hosts.
- Flutter remains a host and packaging mechanism, not a separate SDK API.
- The Flutter SDK constraint means a standalone Dart SDK by itself cannot
  resolve this plugin package.

## Links

- Supersedes: [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md)
- Amends: [ADR-011](./ADR-011-multi-registry-release-ci-crates-io-npm-pub-dev.md)
  for the pub.dev package identity and naming
- Partially amends: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md)
  by replacing the Flutter binding surface with a Dart SDK surface
- Related: [ADR-019](./ADR-019-in-loop-incremental-decoding-and-token-streaming.md)
  for Dart stream semantics
- Package: `edge_intelligence` on pub.dev
