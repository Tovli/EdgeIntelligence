# ADR-011: Multi-Registry Release CI (crates.io, npm, pub.dev)

- **Status**: accepted - partially amended by [ADR-024](./ADR-024-dart-only-platform-agnostic-pub-dev-sdk.md)
- **Date**: 2026-06-12
- **Deciders**:
- **Tags**: ci, release, crates.io, npm, pub.dev, packaging

## Context

The SDK produces three independently consumable artefacts:

| Artefact | Registry | Consumer |
|----------|----------|----------|
| `el-*` Rust crates | crates.io | Rust projects embedding the SDK directly |
| `edge-intelligence-sdk` npm package | npmjs.com | React Native / web TypeScript consumers (wasm-bindgen output, ADR-001) |
| `edge_intelligence` Dart package | pub.dev | Framework-neutral Dart consumers (ADR-024) |

Each registry has its own publish toolchain, credential model, and version
contract. Without automation, publishing is a manual, error-prone multi-step
process that must be repeated in exact order (core crates before adapter crates
due to path-dependency resolution on crates.io).

The `Makefile` added alongside ADR-011 already drives the binding codegen
(`make codegen-rn`, `make codegen-dart`, `make build-wasm`; legacy
`codegen-flutter` aliases may exist during migration). A release CI workflow
sits one step above: it gates on semver tags, runs the codegen, and then
publishes to all three registries from a single push.

Key constraints:
- **Publish order on crates.io**: el-core → el-memory / el-telemetry / el-provenance / el-safety → el-runtime → el-grammar → el-provenance-ed25519 → el-engine-candle → el-cloud → el-ffi. Each crate must be published before its dependants.
- **npm publish** requires the wasm-pack output in `out/web/` to be present and a valid `package.json` with the correct `name` / `version`.
- **pub.dev publish** requires the generated Dart package to have a
  `pubspec.yaml` with matching version and a valid `dart pub publish --dry-run`
  pass, plus packaged native desktop artifacts for each declared Dart platform
  (Linux, macOS, Windows). New work should use Dart naming (`out/dart/`,
  `codegen-dart`) per ADR-024; transitional `out/flutter/` paths may remain
  only as compatibility aliases during migration.
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
- `version` in the generated Dart package `pubspec.yaml` (`out/dart/` for new
  work; transitional `out/flutter/` aliases may exist during migration).

### Stage 3 — Build artefacts
Run `make codegen-rn` (Android + RN), `make build-ios`, `make build-wasm`,
`make codegen-dart`, and desktop native builds for Linux, macOS, and Windows
on their respective runners. Each job uploads its output as a workflow
artefact. The pub.dev assembly consumes only the generated Dart package and the
desktop native artifacts declared by ADR-024.

### Stage 4 — Publish (serial, on separate runners that download Stage 3 artefacts)

| Job | Tool | Secret |
|-----|------|--------|
| `publish-crates` | `cargo publish -p el-core`, then each dependant in order, with `--no-verify` only when the crate was already verified in Stage 1 | `CARGO_REGISTRY_TOKEN` |
| `publish-npm` | `npm publish out/web/ --access public` | `NPM_TOKEN` |
| `publish-pub` | `dart pub publish --force` in the generated Dart package directory | `PUB_CREDENTIALS` (JSON refresh token) |

`publish-crates` inserts a 10-second sleep between crates to allow crates.io's
index propagation before the next dependent is submitted.

The `bindings.yml` CI (ADR-011, triggered on every PR/push to affected paths)
provides the pre-release confidence gate. Pull requests validate that binding
codegen still succeeds without depending on nonessential artifact upload
finalization; pushes to `master` retain the generated binding artifacts for
inspection. `release.yml` only runs on explicit version tags and still uploads
artifacts because later assembly jobs consume them.

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
- The Dart pub.dev and npm packages are thin wrappers around compiled runtime
  artifacts; API surface is defined by `el-ffi` (ADR-001, ADR-024), not
  duplicated here.
- Yanking a broken release requires separate registry-specific commands
  (`cargo yank`, `npm deprecate`, pub.dev retract) — no single-command
  undo exists.

## Links
- Extends: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) (wasm-bindgen / npm surface)
- Partially amended by: [ADR-024](./ADR-024-dart-only-platform-agnostic-pub-dev-sdk.md) (Dart-first pub.dev surface with Flutter mobile runtime packaging)
- Historical extension: [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) (FRB / Flutter framing, now superseded)
- Extends: [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) (Rust workspace / crates.io surface)
- Related: `Makefile`, `.github/workflows/bindings.yml`
- Implements: `crates/adapters/el-ffi`, all `crates/el-*` members
