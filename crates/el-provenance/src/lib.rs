//! `el-provenance` — the hard model-signature load gate (ADR-006).
//!
//! This crate owns the *decision logic*: a `ModelArtifact` must reach
//! `Verified` before a [`LoadPermit`] is issued; a missing or failing signature
//! is a hard stop with no fallback. The actual ED25519 maths is abstracted
//! behind [`SignatureVerifier`] so the gate is testable offline; the real
//! `ed25519-dalek` implementation lives in the excluded adapter
//! `crates/adapters/el-provenance-ed25519` (needs network to fetch).

#![forbid(unsafe_code)]

use el_core::{EdgeError, ModelFormat, ModelId, ModelVersion, Result};

/// Gate state for an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Unverified,
    Verified,
    Rejected,
}

/// Abstracts the signature primitive. Implemented for real by `ed25519-dalek`
/// (excluded adapter) and by test doubles here.
pub trait SignatureVerifier {
    /// Verify `signature` over the whole artifact `bytes` against the trusted
    /// public key identified by `public_key_id`.
    fn verify(&self, bytes: &[u8], signature: &[u8], public_key_id: u32) -> bool;
}

/// Capability token proving an artifact passed the gate. The Inference Runtime
/// requires one to load a model (Conformist relationship — ADR-006). It cannot
/// be constructed except via [`ModelArtifact::ensure_loadable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadPermit {
    pub model: ModelId,
    pub version: ModelVersion,
    pub format: ModelFormat,
}

/// A model file plus its provenance metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArtifact {
    pub id: ModelId,
    pub version: ModelVersion,
    pub format: ModelFormat,
    pub status: VerificationStatus,
}

impl ModelArtifact {
    pub fn new(id: ModelId, version: ModelVersion, format: ModelFormat) -> Self {
        Self {
            id,
            version,
            format,
            status: VerificationStatus::Unverified,
        }
    }

    /// Verify the whole artifact bytes **before** any load/mmap, transitioning
    /// the status. Returns the resulting status.
    pub fn verify<V: SignatureVerifier>(
        &mut self,
        verifier: &V,
        bytes: &[u8],
        signature: &[u8],
        public_key_id: u32,
    ) -> VerificationStatus {
        self.status = if verifier.verify(bytes, signature, public_key_id) {
            VerificationStatus::Verified
        } else {
            VerificationStatus::Rejected
        };
        self.status
    }

    /// The hard load gate. Issues a [`LoadPermit`] only when `Verified`;
    /// otherwise a hard error with no fallback.
    pub fn ensure_loadable(&self) -> Result<LoadPermit> {
        match self.status {
            VerificationStatus::Verified => Ok(LoadPermit {
                model: self.id,
                version: self.version,
                format: self.format,
            }),
            VerificationStatus::Rejected => Err(EdgeError::SignatureRejected),
            VerificationStatus::Unverified => Err(EdgeError::UnverifiedModel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub(bool);
    impl SignatureVerifier for Stub {
        fn verify(&self, _bytes: &[u8], _sig: &[u8], _key: u32) -> bool {
            self.0
        }
    }

    fn artifact() -> ModelArtifact {
        ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf)
    }

    #[test]
    fn unverified_cannot_load() {
        assert_eq!(
            artifact().ensure_loadable().unwrap_err(),
            EdgeError::UnverifiedModel
        );
    }

    #[test]
    fn rejected_signature_is_a_hard_stop() {
        let mut a = artifact();
        a.verify(&Stub(false), b"weights", b"badsig", 7);
        assert_eq!(a.status, VerificationStatus::Rejected);
        assert_eq!(
            a.ensure_loadable().unwrap_err(),
            EdgeError::SignatureRejected
        );
    }

    #[test]
    fn verified_issues_a_permit() {
        let mut a = artifact();
        a.verify(&Stub(true), b"weights", b"goodsig", 7);
        let permit = a.ensure_loadable().unwrap();
        assert_eq!(permit.model, ModelId(1));
        assert_eq!(permit.format, ModelFormat::Gguf);
    }
}
