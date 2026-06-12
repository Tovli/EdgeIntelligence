//! Real ED25519 implementation of [`el_provenance::SignatureVerifier`] (ADR-006).
//!
//! Complete and correct against `ed25519-dalek` v2; excluded from the default
//! workspace only because it needs crates.io to fetch the dependency. The gate
//! *logic* it plugs into is already tested in `el-provenance`.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use el_provenance::SignatureVerifier;

/// Verifies model signatures against a set of trusted provider public keys,
/// keyed by `public_key_id` (the trust-anchor reference from ADR-006).
#[derive(Default)]
pub struct Ed25519Verifier {
    keys: HashMap<u32, VerifyingKey>,
}

impl Ed25519Verifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trusted 32-byte public key under `id`.
    pub fn register(
        &mut self,
        id: u32,
        key_bytes: [u8; 32],
    ) -> Result<(), ed25519_dalek::SignatureError> {
        let vk = VerifyingKey::from_bytes(&key_bytes)?;
        self.keys.insert(id, vk);
        Ok(())
    }
}

impl SignatureVerifier for Ed25519Verifier {
    fn verify(&self, bytes: &[u8], signature: &[u8], public_key_id: u32) -> bool {
        let Some(vk) = self.keys.get(&public_key_id) else {
            return false; // unknown key id → reject (hard stop upstream)
        };
        let Ok(sig_bytes) = <[u8; 64]>::try_from(signature) else {
            return false; // malformed signature → reject
        };
        let sig = Signature::from_bytes(&sig_bytes);
        vk.verify(bytes, &sig).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use el_core::{ModelFormat, ModelId, ModelVersion};
    use el_provenance::ModelArtifact;

    /// Deterministic key from fixed secret bytes — no RNG dependency in tests.
    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    #[test]
    fn genuine_signature_verifies_and_tampering_is_rejected() {
        let sk = signing_key();
        let weights = b"quantized-model-weights";
        let sig = sk.sign(weights).to_bytes();

        let mut verifier = Ed25519Verifier::new();
        verifier.register(1, sk.verifying_key().to_bytes()).unwrap();

        assert!(
            verifier.verify(weights, &sig, 1),
            "genuine signature must verify"
        );
        assert!(
            !verifier.verify(b"tampered-weights", &sig, 1),
            "tampered bytes must fail"
        );
        assert!(
            !verifier.verify(weights, &sig, 999),
            "unknown key id must fail"
        );
        assert!(
            !verifier.verify(weights, b"short-sig", 1),
            "malformed signature must fail"
        );
    }

    #[test]
    fn verified_artifact_passes_the_load_gate() {
        let sk = signing_key();
        let weights = b"weights";
        let sig = sk.sign(weights).to_bytes();
        let mut verifier = Ed25519Verifier::new();
        verifier.register(2, sk.verifying_key().to_bytes()).unwrap();

        let mut art = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
        art.verify(&verifier, weights, &sig, 2);
        assert!(
            art.ensure_loadable().is_ok(),
            "verified artifact yields a LoadPermit"
        );
    }

    #[test]
    fn forged_signature_is_a_hard_stop_at_the_gate() {
        let real = signing_key();
        let attacker = SigningKey::from_bytes(&[9u8; 32]);
        let weights = b"weights";
        // Signed by the attacker's key, not the trusted one.
        let forged = attacker.sign(weights).to_bytes();

        let mut verifier = Ed25519Verifier::new();
        verifier
            .register(3, real.verifying_key().to_bytes())
            .unwrap();

        let mut art = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
        art.verify(&verifier, weights, &forged, 3);
        assert!(
            art.ensure_loadable().is_err(),
            "forged signature must block loading"
        );
    }
}
