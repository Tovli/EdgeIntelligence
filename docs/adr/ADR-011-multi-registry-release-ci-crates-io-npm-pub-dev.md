# ADR-011: Multi-Registry Release CI (crates.io, npm, pub.dev)

- **Status**: proposed
- **Date**: 2026-06-12
- **Deciders**:
- **Tags**: ci, release, crates.io, npm, pub.dev, packaging

## Context

The SDK produces three independently consumable artefacts:

| Artefact | Registry | Consumer |
|----------|----------|----------|
| `el-*` Rust crates | crates.io | Rust projects embedding the SDK directly |
| `@edge-intelligence/sdk` npm package | npmjs.com | React Native / web TypeScript consumers (wasm-bindgen output, ADR-001) |
| `edge_intelligence` Dart package | pub.dev | Flutter consumers (FRB v2 output, ADR-009) |

Each registry has its own publish toolchain, credential model, and version
contract. Without automation, publishing is a manual, error-prone multi-step
process that must be repeated in exact order (core crates before adapter crates
due to path-dependency resolution on crates.io).

The `Makefile` added alongside ADR-011 already drives the binding codegen
(`make codegen-rn`, `make codegen-flutter`, `make build-wasm`). A release CI
workflow sits one step above: it gates on semver tags, runs the codegen, and
then publishes to all three registries from a single push.

Key constraints:
- **Publish order on crates.io**: el-core → el-memory / el-telemetry / el-provenance / el-safety → el-runtime → el-grammar → el-provenance-ed25519 → el-engine-candle → el-cloud → el-ffi. Each crate must be published before its dependants.
- **npm publish** requires the wasm-pack output in `out/web/` to be present and a valid `package.json` with the correct `name` / `version`.
- **pub.dev publish** requires the FRB-generated Dart package in `out/flutter/` to have a `pubspec.yaml` with matching version and a valid `dart pub publish --dry-run` pass.
- **Credentials**: crates.io API token, npm access token, pub.dev refresh token — all injected as GitHub Actions secrets, never committed.
- **Versioning**: the single source of truth for the version number is the git tag (`v0.2.0`). Cargo workspace version, npm `package.json` version, and Dart `pubspec.yaml` version must all be stamped from the tag before publish.

## Decision

Add a **`release.yml`** GitHub Actions workflow triggered by `push` to tags
matching `v[0-9]+.[0-9]+.[0-9]+` (semver). The workflow has four sequential
stages:

### Stage 1 — Verify
Run `cargo test --locked --workspace` and `cargo fmt --check` on Ubuntu. Gate
everything on this; no publish happens if tests fail.

### Stage 2 — Stamp versions
Extract the semver from the git tag and patch:
- Each `[package] version` in the workspace `Cargo.toml` members (via `cargo
  set-version` from `cargo-edit`, or `sed` on the TOML).
- `version` in `out/web/package.json` (created by `wasm-pack`).
- `version` in `out/flutter/pubspec.yaml` (created by `flutter_rust_bridge_codegen`).

### Stage 3 — Build artefacts
Run `make codegen-rn` (Android + RN), `make build-ios`, `make build-wasm`,
`make codegen-flutter` on their respective runners (reuse the
`bindings.yml` matrix). Each job uploads its output as a workflow artefact.

### Stage 4 — Publish (serial, on separate runners that download Stage 3 artefacts)

| Job | Tool | Secret |
|-----|------|--------|
| `publish-crates` | `cargo publish -p el-core`, then each dependant in order, with `--no-verify` only when the crate was already verified in Stage 1 | `CARGO_REGISTRY_TOKEN` |
| `publish-npm` | `npm publish out/web/ --access public` | `NPM_TOKEN` |
| `publish-pub` | `dart pub publish --force` in `out/flutter/` | `PUB_CREDENTIALS` (JSON refresh token) |

`publish-crates` inserts a 10-second sleep between crates to allow crates.io's
index propagation before the next dependent is submitted.

The `bindings.yml` CI (ADR-011, triggered on every PR/push to affected paths)
provides the pre-release confidence gate. `release.yml` only runs on explicit
version tags.

## Consequences

### Positive
- One `git tag v0.x.y && git push --tags` publishes the SDK to all three
  registries atomically; no manual steps.
- Version numbers are always consistent across Cargo / npm / pub because they
  are all stamped from the same tag in the same workflow run.
- crates.io publish order is encoded in the workflow and not dependent on
  human memory.
- Credentials never leave GitHub Actions secrets.

### Negative
- `cargo publish` has no official dry-run that validates the full dependency
  chain against the live registry index; a broken publish-order causes the
  workflow to fail mid-run and requires a patch tag.
- The Dart `pub publish --force` flag bypasses the interactive confirmation;
  a mistake in `pubspec.yaml` cannot be undone (pub.dev packages are
  immutable once published).
- Version stamping via `sed`/`cargo set-version` means the working tree is
  dirty at publish time; the Cargo.lock must be re-committed or `--allow-dirty`
  must be passed (preferred: commit the version bump before tagging).

### Neutral
- The Flutter and npm packages are thin wrappers around the compiled native
  artefact; API surface is defined by `el-ffi` (ADR-001, ADR-009), not
  duplicated here.
- Yanking a broken release requires separate registry-specific commands
  (`cargo yank`, `npm deprecate`, pub.dev retract) — no single-command
  undo exists.

## Links
- Extends: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) (wasm-bindgen / npm surface)
- Extends: [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) (FRB / pub.dev surface)
- Extends: [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) (Rust workspace / crates.io surface)
- Related: `Makefile`, `.github/workflows/bindings.yml`
- Implements: `crates/adapters/el-ffi`, all `crates/el-*` members
