# el-provenance-ed25519 — real ED25519 signature verifier

The production `SignatureVerifier` for the model-provenance load gate (ADR-006),
implemented over [`ed25519-dalek`](https://crates.io/crates/ed25519-dalek) v2.

It plugs into the gate *logic* in [`el-provenance`](../../el-provenance): this
adapter answers "is this signature valid?", and `el-provenance` decides what to
do about it (issue a `LoadPermit`, or hard-stop). Pure-Rust dependency tree, no
`unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

- **`Ed25519Verifier`** — verifies model signatures against a set of trusted
  provider public keys, keyed by `public_key_id` (the trust-anchor reference
  from ADR-006):
  - `new()` / `Default`
  - `register(id, key_bytes: [u8; 32])` — register a trusted public key
  - implements `el_provenance::SignatureVerifier::verify`, which **rejects**
    (returns `false`) on an unknown key id, a malformed signature, or a
    verification failure — every failure mode is a hard stop upstream.

## Usage

```rust
use el_core::{ModelFormat, ModelId, ModelVersion};
use el_provenance::ModelArtifact;
use el_provenance_ed25519::Ed25519Verifier;

let mut verifier = Ed25519Verifier::new();
verifier.register(/* public_key_id */ 1, trusted_public_key_bytes)?;

let mut artifact = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
artifact.verify(&verifier, model_bytes, signature_bytes, 1);

// Verified → a LoadPermit; tampered bytes, a forged signature, or an unknown
// key id → a hard error with no fallback.
let permit = artifact.ensure_loadable()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Why it's a separate adapter

The core gate lives in `el-provenance` with **zero dependencies** so it
cross-compiles everywhere and stays unit-testable offline with doubles. This
adapter isolates the one crates.io dependency (`ed25519-dalek`) behind the
`SignatureVerifier` seam. It is a regular workspace member.

## Status

Implemented and tested — genuine signatures verify; tampering, forged
signatures, malformed signatures, and unknown key ids are all rejected.

---

Part of the [Edge Intelligence](../../../README.md) workspace. Realizes
[ADR-006](../../../docs/adr/ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md);
see the [Model Provenance](../../../docs/ddd/bounded-contexts/08-model-provenance.md) context.
