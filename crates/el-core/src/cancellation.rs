//! Cooperative request cancellation shared by runtime and host bindings.
//!
//! The token is deliberately dependency-free: `el-core` stays usable on every
//! native and WASM target without imposing an async runtime (ADR-027).

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// A one-way, cloneable request-cancellation signal.
///
/// Cancelling a token never force-stops a thread. The runtime observes it at
/// defined prefill, decode, and safety-checkpoint boundaries, then performs its
/// normal session cleanup path.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create an uncancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cooperative cancellation. This operation is idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_observe_the_same_cancellation_request() {
        let token = CancellationToken::new();
        let clone = token.clone();

        clone.cancel();

        assert!(token.is_cancelled());
    }
}
