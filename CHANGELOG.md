# Changelog

## Unreleased (0.5.0)

### Changed

- Added the native React Native asynchronous request surface from ADR-027.
  `EdgeError`, `DomainEvent`, and `Phase` are now non-exhaustive; external Rust
  consumers must include a wildcard arm when matching them.
- Added `EdgeError::Cancelled`, `Phase::Faulted`, and the content-free
  `GenerationCancelled` / `SessionResetFailed` lifecycle events.
- **Breaking for generated native bindings:** `SdkError` now includes `Busy`
  and `Cancelled`. Kotlin and Swift consumers with exhaustive `when`/`switch`
  handling must add those cases. A stateful `EdgeLlm` now returns `Busy` for an
  overlapping `ask`, stream, or `reset` rather than allowing a session race.

Release notes are tracked in Git tags and GitHub releases.
