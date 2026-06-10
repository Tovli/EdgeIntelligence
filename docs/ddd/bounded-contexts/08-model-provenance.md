# Bounded Context 8 — Model Provenance & Security  (Generic)

> Gates every model load on a verified signature, manages encrypted storage, and
> applies signed OTA updates. Source: PRD §"Security, Privacy, and Compliance" →
> "Model Provenance & Updates", §"Memory and Storage".

## Purpose

Ensure only authentic, untampered models are ever loaded, and that updates
arrive through a cryptographically verified channel — the trust boundary in
front of the Core.

## Strategic role

**Generic.** Signing/verification and secure update are standard security
disciplines; use proven primitives (ED25519), don't reinvent.

## Ubiquitous language (context-local)

`Model Artifact`, `Model Signature`, `Model Version`, `Checksum`, `OTA Update`,
`Encrypted Storage`.

## Aggregates

### `ModelArtifact` (Aggregate Root)
A model file plus its provenance metadata.

- **Identity:** `ModelId` + `ModelVersion`.
- **Holds:** flash location, `ModelFormat`, `ModelSignature`, `Checksum`,
  encryption state, and a `VerificationStatus`.
- **Invariants:**
  - An artifact must reach `Verified` **before** the Inference Runtime may load
    it — a failed or missing signature is a hard stop (no load, no fallback).
  - The signature is verified over the *whole* artifact bytes prior to `mmap`.
  - Encrypted artifacts are decrypted only into protected memory, never to a
    durable plaintext file.

### `ModelUpdate` (Aggregate Root)
A pending OTA replacement for an artifact.

- **Identity:** `UpdateId` (targets a `ModelId`).
- **Holds:** the incoming artifact, its signature, and the source channel
  (app-store push / secure channel — never a cloud inference API).
- **Invariants:** an update is applied only after the same signature check
  passes; a rejected update leaves the current artifact untouched.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `ModelSignature` | ED25519 signature bytes | verified pre-load |
| `ModelVersion` | semantic/version id | |
| `Checksum` | content hash | integrity cross-check |
| `VerificationStatus` | `Unverified` \| `Verified` \| `Rejected` | gate state |
| `PublicKeyId` | provider key reference | trust anchor |

## Domain Services

- **`SignatureVerifier`** — verifies the ED25519 `ModelSignature` against the
  provider `PublicKeyId`; emits the verdict.
- **`UpdateApplier`** — swaps in a `ModelUpdate` only on a passing verdict.
- **`StorageProtector`** — manages encrypted-at-rest storage and decryption into
  protected memory.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `CryptoProvider` (ED25519) | `ed25519-dalek` (Rust); platform keystore via FFI | inbound |
| `VerifiedModelSource` | exposed to Inference Runtime (1) | outbound (U, CF) |
| `UpdateChannel` | app-store / secure channel | inbound (no cloud API) |

## Anti-Corruption Layer

`CryptoAcl` wraps the pure-Rust **`ed25519-dalek`** crate (and, for
hardware-backed keys, the platform keystore — Android Keystore / iOS Keychain —
via thin FFI) so the domain deals only in `ModelSignature` /
`VerificationStatus`.

## Domain Events (published)

`ModelSignatureVerified`, `ModelSignatureRejected`, `ModelUpdateApplied`,
`ModelUpdateRejected`. See [domain-events.md](../domain-events.md).

## Relationships

**Upstream Conformist** to Inference Runtime (1): the runtime accepts the
provenance verdict without negotiation. The Rust binary / WASM module and any
platform delegates are additionally OS code-signed (defense in depth) — that
check lives at the platform layer, outside this context's aggregates but noted
in its policy.
