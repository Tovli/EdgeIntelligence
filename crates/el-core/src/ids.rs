//! Identifier value objects.

use core::fmt;

/// Unique id of an [`crate::SessionId`]-scoped inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

/// Unique id of a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub u64);

/// Semantic version of a model artifact (Model Provenance — ADR-006).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ModelVersion {
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self { major, minor, patch }
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "session#{}", self.0)
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model#{}", self.0)
    }
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_orders_and_displays() {
        assert!(ModelVersion::new(1, 2, 0) > ModelVersion::new(1, 1, 9));
        assert_eq!(ModelVersion::new(0, 1, 0).to_string(), "0.1.0");
        assert_eq!(SessionId(7).to_string(), "session#7");
    }
}
