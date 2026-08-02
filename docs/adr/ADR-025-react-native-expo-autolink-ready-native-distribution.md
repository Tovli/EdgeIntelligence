# ADR-025: Expo-autolink-ready React Native native distribution

- **Status**: accepted
- **Date**: 2026-07-13
- **Deciders**: Tovli
- **Tags**: react-native, expo, npm, packaging, jsi, bindings

## Context

The published `edge-intelligence-sdk` npm package contains generated React
Native TypeScript bindings and prebuilt Android/iOS native artifacts, but it is
not a complete React Native native module. At discovery, an Expo SDK 56 / React
Native 0.85 development build exposed that the generated bindings require
`@ubjs/core` even though the package does not declare it, and
`globalThis.NativeElFfi` is never registered. The verified host for this
decision is Expo SDK 57 / React Native 0.86 with React 19.2.
The package also lacks React Native autolinking metadata, Android Gradle module
files, an iOS podspec, and supported-host setup guidance.

This leaves consumers with an npm package that resolves the TypeScript facade
but cannot construct a native `EdgeLlm` session without undocumented manual
linking. It contradicts ADR-001's decision to project the `el-ffi` Rust facade
to React Native through generated TypeScript, JSI C++, and Turbo Module output.
Native artifact delivery and native module registration must be part of the
published React Native product boundary, not application-specific work.

## Decision

Publish `edge-intelligence-sdk` as an Expo-compatible React Native native
module, with the generated UniFFI React Native bridge as its implementation.

1. Declare every JavaScript runtime dependency required by the generated
   bindings in `package.json`. `@ubjs/core` must be a direct dependency with a
   version range validated against the generated output. React Native remains a
   peer dependency with an explicit supported version range. Expo is not a
   peer dependency because React Native autolinking, rather than an Expo API,
   discovers this native module.
2. Ship standard React Native autolinking metadata and native module projects:
   an Android Gradle library that packages ABI-specific `libel_ffi.so` files,
   and an iOS podspec that links the supported `libel_ffi` artifact. These
   projects own the JSI/Turbo Module installation that exposes `NativeElFfi`.
3. Use React Native and Expo native autolinking for managed applications.
   Because no app configuration mutation is required, do not ship or document a
   no-op Expo config plugin. The package must work in an Expo development or
   production native build after installation and rebuild; Expo Go is not a
   supported runtime because it cannot load arbitrary native modules.
4. Treat the generated native entrypoint and Turbo Module installer as internal
   bridge details. Before invoking an SDK constructor, the TypeScript facade
   must load that entrypoint and convert Turbo Module or Rust-install failures
   into an actionable error explaining that a native build/rebuild and
   supported setup are required.
5. Document the supported Expo and React Native versions, installation and
   rebuild steps, model-file placement, platform artifact coverage, and a
   minimal verified Expo example. Release CI must build that example and verify
   autolinking on Android and iOS. Android CI must additionally launch the
   release app and verify a basic local-session construction through visible UI;
   iOS remains a deterministic build-and-link check to avoid putting simulator
   launch reliability on the registry-publish critical path.

The browser/WASM surface remains an ESM initialization flow and does not load
the React Native native module. The package may use conditional exports or
separate entrypoints to keep those runtime contracts isolated.

## Consequences

### Positive

- Expo and bare React Native consumers get a defined installation contract and
  automatic native linking rather than undocumented application patches.
- The package owns its generated JavaScript dependency graph and reports a
  clear native-setup error when the bridge is unavailable.
- Android and iOS artifact ownership, JSI registration, and compatibility
  testing become release requirements instead of consumer responsibilities.
- ADR-001's React Native projection is delivered as a usable host integration.

### Negative

- The npm package gains Android and CocoaPods maintenance and a
  broader React Native/Expo compatibility test matrix.
- Consumers must use a custom Expo development build or production build and
  rebuild after installing or upgrading the native package; Expo Go remains
  unsupported.
- The release pipeline must produce and validate native artifacts that match
  each declared ABI and iOS architecture.

### Neutral

- `el-ffi` remains the Rust composition root and UniFFI remains the source of
  generated binding contracts; this decision does not introduce a separate
  application-level JNI, Objective-C, or Swift API.
- Dart/pub.dev packaging remains governed by ADR-024, and browser/WASM remains
  a separate runtime path in the npm distribution.

## Links

- Issue: [#13](https://github.com/Tovli/EdgeIntelligence/issues/13)
- Extends: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md)
  for the React Native native-module delivery contract
- Related: [ADR-011](./ADR-011-multi-registry-release-ci-crates-io-npm-pub-dev.md)
  for npm release ownership
- Related: [ADR-024](./ADR-024-dart-only-platform-agnostic-pub-dev-sdk.md)
  for the separate Dart/pub.dev packaging contract
