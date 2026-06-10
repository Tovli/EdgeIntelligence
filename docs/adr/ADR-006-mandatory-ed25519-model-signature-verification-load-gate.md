# ADR-006: Mandatory ED25519 model-signature verification as a hard load gate

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: security, provenance, conformist, generic

## Context

Models are loaded from flash and may be updated over time; a tampered or
unauthentic model would compromise both safety and output integrity. The PRD
(§"Security, Privacy, and Compliance" → "Model Provenance & Updates") requires a
digital signature (ED25519) on every model file, verified before loading, with
OTA updates using the same check over a secure channel.

The open question is how strictly the [Inference
Runtime](../ddd/bounded-contexts/01-inference-runtime.md) (Core) should depend on
the [Model Provenance](../ddd/bounded-contexts/08-model-provenance.md) verdict.

## Decision

Make signature verification a **hard load gate**: a `ModelArtifact` must reach
`Verified` before the Inference Runtime may load it; a missing or failing
`ModelSignature` is a **hard stop with no fallback**. The signature is verified
over the whole artifact bytes **before** `mmap`. OTA updates are applied only on
a passing verdict and arrive via a signed out-of-band channel (never a live
cloud API — consistent with
[ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)).

Model the Runtime→Provenance relationship as **Conformist**: the runtime accepts
the provenance verdict without negotiation. WASM engine and delegates are
additionally OS code-signed (defense in depth).

## Consequences

### Positive
- Tampered/unauthentic models cannot be loaded; integrity is enforced before any
  weight is mapped.
- Clear, single trust boundary in front of the Core.

### Negative
- A bad signature renders the SDK unusable until a valid artifact is supplied —
  no degraded "run anyway" mode (by design).
- Key management and rotation (`PublicKeyId` trust anchor) become operational
  responsibilities.

### Neutral
- Verification uses the pure-Rust **`ed25519-dalek`** (RustCrypto) crate, wrapped
  behind `CryptoAcl`; hardware-backed keys in the platform keystore (Android
  Keystore / iOS Keychain) are reached via thin FFI where needed. No C/C++ crypto
  dependency (see [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)),
  and crypto types stay out of the domain.

## Links
- PRD: `docs/prd.md` §"Security, Privacy, and Compliance" → "Model Provenance & Updates"
- DDD: [Model Provenance context](../ddd/bounded-contexts/08-model-provenance.md), [Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md)
- Related: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md), [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
