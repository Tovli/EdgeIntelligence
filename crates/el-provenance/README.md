# el-provenance — model-signature load gate

The hard model-signature load gate (ADR-006). This crate owns the *decision
logic*: a `ModelArtifact` must reach `Verified` before a `LoadPermit` is issued,
and a missing or failing signature is a **hard stop with no fallback**.

The actual ED25519 maths is abstracted behind the `SignatureVerifier` trait so
the gate is testable offline. The real `ed25519-dalek` implementation lives in
the [`el-provenance-ed25519`](../adapters/el-provenance-ed25519) adapter.

Depends only on `el-core`. No `unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

- **`SignatureVerifier`** — the abstracted primitive:
  `verify(bytes, signature, public_key_id) -> bool`. Implemented for real by the
  ed25519 adapter and by test doubles.
- **`ModelArtifact`** — a model file plus its provenance metadata
  (`id`, `version`, `format`, `status`). `verify(...)` transitions the status
  *before* any load/mmap; `ensure_loadable()` is the gate itself.
- **`LoadPermit`** — the capability token proving an artifact passed the gate.
  It **cannot be constructed except via `ensure_loadable()`**, so a model that
  has not been verified cannot reach the runtime — `el_runtime::InferenceSession`
  requires a `LoadPermit` to be built (the Conformist relationship, enforced in
  the type system).
- **`VerificationStatus`** — `Unverified` | `Verified` | `Rejected`.

## Usage

```rust
use el_core::{ModelFormat, ModelId, ModelVersion};
use el_provenance::{ModelArtifact, SignatureVerifier};

// Plug in a real verifier (el-provenance-ed25519) or a test double.
struct AlwaysOk;
impl SignatureVerifier for AlwaysOk {
    fn verify(&self, _bytes: &[u8], _sig: &[u8], _key_id: u32) -> bool { true }
}

let mut artifact = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
artifact.verify(&AlwaysOk, b"<model-bytes>", b"<signature>", /* public_key_id */ 7);

// No verified signature → no permit → no session.
let permit = artifact.ensure_loadable()?; // Err(UnverifiedModel | SignatureRejected) otherwise
# Ok::<(), el_core::EdgeError>(())
```

## Place in the workspace

`el-runtime` accepts a `LoadPermit` (not raw bytes) to construct a session, so
this gate sits in front of every inference path. The ed25519 adapter provides
the production `SignatureVerifier`.

## Status

Implemented and tested. The gate *logic* is fully covered here; real signature
verification is provided by the ed25519 adapter.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-006](../../docs/adr/ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md);
see the [Model Provenance](../../docs/ddd/bounded-contexts/08-model-provenance.md) context.
