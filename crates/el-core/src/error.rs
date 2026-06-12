//! Domain error type.

use core::fmt;

/// Errors surfaced by the SDK. Variants carry only static descriptors and
/// numeric context — never user content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeError {
    /// Attempted to load/use a model that has not reached `Verified`
    /// (ADR-006 hard load gate).
    UnverifiedModel,
    /// Model signature verification failed (ADR-006).
    SignatureRejected,
    /// Operation invalid for the current session phase (ADR-001 state machine).
    InvalidPhase {
        expected: &'static str,
        found: &'static str,
    },
    /// The static memory plan exceeds the configured budget (ADR-003).
    MemoryBudgetExceeded { requested: u64, budget: u64 },
    /// A network egress was attempted while air-gapped (ADR-004).
    AirGapViolation,
    /// Engine/adapter failure (message is a static descriptor, not user data).
    Engine(&'static str),
    /// Cloud request failed (ADR-010). Carries a heap-allocated message so
    /// dynamic error strings (HTTP status, URL) can be included without leaking.
    CloudRequest(Box<str>),
    /// Grammar constraint error (ADR-004). Heap-allocated for the same reason.
    Grammar(Box<str>),
}

impl fmt::Display for EdgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeError::UnverifiedModel => {
                write!(f, "model is not verified; refusing to load (ADR-006)")
            }
            EdgeError::SignatureRejected => write!(f, "model signature rejected (ADR-006)"),
            EdgeError::InvalidPhase { expected, found } => {
                write!(f, "invalid phase: expected {expected}, found {found}")
            }
            EdgeError::MemoryBudgetExceeded { requested, budget } => {
                write!(
                    f,
                    "memory plan needs {requested} bytes > budget {budget} (ADR-003)"
                )
            }
            EdgeError::AirGapViolation => {
                write!(f, "network egress attempted while air-gapped (ADR-004)")
            }
            EdgeError::Engine(msg) => write!(f, "engine error: {msg}"),
            EdgeError::CloudRequest(msg) => write!(f, "cloud request: {msg}"),
            EdgeError::Grammar(msg) => write!(f, "grammar constraint: {msg}"),
        }
    }
}

impl std::error::Error for EdgeError {}

/// SDK result alias.
pub type Result<T> = core::result::Result<T, EdgeError>;
