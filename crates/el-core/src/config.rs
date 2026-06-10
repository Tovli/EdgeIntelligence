//! Per-session configuration (immutable for the life of a session — ADR-001).

use crate::value_objects::{DeviceTarget, ModelFormat, SafetyMode, SpeculationMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionConfig {
    pub format: ModelFormat,
    pub device: DeviceTarget,
    pub safety: SafetyMode,
    pub speculation: SpeculationMode,
    /// Apply LLMLingua-2 prompt compression before prefill (degradable).
    pub compress: bool,
    pub max_tokens: u32,
    /// Hard cap for the static memory plan (ADR-003). Default 1 GiB.
    pub memory_budget_bytes: u64,
    /// Opt-in LAN relay (ADR-004). **Defaults to `false` — air-gapped.**
    pub hybrid_mode: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            format: ModelFormat::Gguf,
            device: DeviceTarget::Auto,
            safety: SafetyMode::Lightweight,
            speculation: SpeculationMode::Off, // safe default off (ADR-002)
            compress: true,
            max_tokens: 512,
            memory_budget_bytes: 1024 * 1024 * 1024, // 1 GiB cap (ADR-003)
            hybrid_mode: false,                      // air-gapped by default (ADR-004)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_air_gapped_and_conservative() {
        let c = SessionConfig::default();
        assert!(!c.hybrid_mode, "must be air-gapped by default (ADR-004)");
        assert_eq!(c.speculation, SpeculationMode::Off, "speculation off by default");
    }
}
