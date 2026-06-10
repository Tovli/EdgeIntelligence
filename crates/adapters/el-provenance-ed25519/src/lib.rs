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
    pub fn register(&mut self, id: u32, key_bytes: [u8; 32]) -> Result<(), ed25519_dalek::SignatureError> {
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
