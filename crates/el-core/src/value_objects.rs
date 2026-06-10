//! Core value objects from the ubiquitous language.

/// A vocabulary id produced or consumed by the model — the atomic generation
/// unit.
pub type Token = u32;

/// Supported on-disk model formats (ADR-002). `.pte`/GGUF-via-C++ is gone;
/// Candle reads GGUF and safetensors natively, ONNX via `tract`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Onnx,
}

/// The execution engine that owns a loaded model (ADR-002).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    /// Pure-Rust Candle (primary): GGUF/safetensors, CPU NEON + Metal + WebGPU.
    Candle,
    /// Pure-Rust `tract` for the optional ONNX path.
    Tract,
}

impl ModelFormat {
    /// The engine required by this format. Encodes the ADR-002 compatibility
    /// rule (`GGUF`/`safetensors` → Candle, `ONNX` → tract).
    pub fn runtime(self) -> RuntimeKind {
        match self {
            ModelFormat::Gguf | ModelFormat::Safetensors => RuntimeKind::Candle,
            ModelFormat::Onnx => RuntimeKind::Tract,
        }
    }
}

/// Requested device class; `Auto` is resolved by the Hardware & Delegate
/// context at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTarget {
    Auto,
    MidRange,
    HighEnd,
}

/// Inference session state machine (ADR-001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Initialized,
    Prefilling,
    Decoding,
    Completed,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Initialized => "Initialized",
            Phase::Prefilling => "Prefilling",
            Phase::Decoding => "Decoding",
            Phase::Completed => "Completed",
        }
    }
}

/// Tiered safety strategy (ADR-005), budget-gated by device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyMode {
    Off,
    Lightweight,
    SecDecoding,
    Csd,
}

/// Speculative decoding strategy (ADR-002 / context 3). Default is `Off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationMode {
    Off,
    Draft,
    LeverLite,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    Stopped,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_picks_engine() {
        assert_eq!(ModelFormat::Gguf.runtime(), RuntimeKind::Candle);
        assert_eq!(ModelFormat::Safetensors.runtime(), RuntimeKind::Candle);
        assert_eq!(ModelFormat::Onnx.runtime(), RuntimeKind::Tract);
    }
}
