//! Content-free domain events (ADR-007).
//!
//! **Enforcement:** [`DomainEvent`] and [`EventEnvelope`] derive `Copy`. A
//! `String`, `Vec<u8>`, or any heap-owning field is not `Copy`, so adding one
//! would fail to compile. That makes "no prompt/response content on an event" a
//! *compile-time* guarantee, not a code-review convention. Fixed-point integers
//! (e.g. `*_milli`) stand in for ratios/scores so the type stays `Copy` + `Eq`.

use crate::ids::{ModelId, ModelVersion, SessionId};
use crate::value_objects::{
    DeviceTarget, ModelFormat, RuntimeKind, SafetyMode, SpeculationMode, StopReason,
};

/// Why an optional pipeline stage was degraded (PRD risk policy made
/// observable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeReason {
    Disabled,
    MemoryPressure,
    MidRangeProfile,
}

/// Every fact the pipeline emits. Carries only ids, counts, enums, and
/// fixed-point numbers — never content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainEvent {
    // --- 1. Inference Runtime ---
    SessionInitialized {
        runtime: RuntimeKind,
        device: DeviceTarget,
        safety: SafetyMode,
        speculation: SpeculationMode,
    },
    ModelLoaded {
        model: ModelId,
        version: ModelVersion,
        format: ModelFormat,
    },
    PrefillCompleted {
        prompt_tokens: u32,
        kv_len: u32,
        prefill_tps: u32,
    },
    TokenGenerated {
        sampled: bool,
    },
    TokenCommitted {
        kv_len: u32,
    },
    GenerationCompleted {
        total_tokens: u32,
        stop: StopReason,
    },
    SessionReset,
    HybridRelayConsulted,

    // --- 2. Prompt Compression ---
    PromptCompressed {
        input_tokens: u32,
        output_tokens: u32,
        /// Compression ratio × 1000 (e.g. 250 = 0.25 = 4× shorter).
        ratio_milli: u32,
    },
    CompressionSkipped {
        reason: DegradeReason,
    },

    // --- 3. Speculative Decoding ---
    DraftProposed {
        draft_len: u8,
    },
    DraftVerified {
        accepted: u8,
        first_reject: u8,
    },
    SpeculationDisabled {
        reason: DegradeReason,
    },

    // --- 4. Grammar Constraint ---
    GrammarSwitched {
        from_state: u32,
        to_state: u32,
    },
    TokenMaskApplied {
        allowed: u32,
    },
    GrammarViolationBlocked,

    // --- 5. Safety ---
    SafetyModeSelected {
        mode: SafetyMode,
    },
    LogitsSteered {
        adjustment_norm_milli: u32,
    },
    SafetyViolationDetected {
        score_milli: u16,
        threshold_milli: u16,
    },
    ClaimBacktracked {
        claim_index: u32,
    },
    SafetyDisabled {
        reason: DegradeReason,
    },

    // --- 6. Memory Management ---
    MemoryPlanCreated {
        total_bytes: u64,
        sram_bytes: u64,
        dram_bytes: u64,
    },
    KvCacheCompacted {
        reclaimed: u32,
    },
    MemoryBudgetExceeded {
        requested_bytes: u64,
        budget_bytes: u64,
    },

    // --- 7. Hardware & Delegate ---
    DeviceProfiled {
        profile: DeviceTarget,
        npu_tops: u16,
        bandwidth_gbs: u16,
    },
    DelegateSelected {
        partitions: u8,
    },
    DelegateFellBack,

    // --- 8. Model Provenance ---
    ModelSignatureVerified {
        model: ModelId,
        version: ModelVersion,
    },
    ModelSignatureRejected {
        model: ModelId,
    },

    // --- 9. Telemetry ---
    MetricsSampled {
        decode_tps: u32,
        ttft_ms: u32,
        peak_bytes: u64,
    },

    // --- 10. Frontier LLM (ADR-010 opt-in cloud egress) ---
    /// Emitted when the opt-in cloud backend is consulted (parallel to
    /// `HybridRelayConsulted`). `provider_hash` is a CRC32 of the provider
    /// prefix — not the API key or any content.
    FrontierLlmConsulted {
        provider_hash: u32,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}

/// Standard envelope (`docs/ddd/domain-events.md`): every event is correlated by
/// `SessionId` and ordered by a logical step index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventEnvelope {
    pub session: SessionId,
    pub step: u32,
    pub event: DomainEvent,
}

impl EventEnvelope {
    pub fn new(session: SessionId, step: u32, event: DomainEvent) -> Self {
        Self {
            session,
            step,
            event,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time proof that events are content-free: this only compiles if
    // `DomainEvent: Copy`, which is impossible if any field owns heap memory.
    fn _assert_copy<T: Copy>() {}
    #[test]
    fn events_are_content_free_by_construction() {
        _assert_copy::<DomainEvent>();
        _assert_copy::<EventEnvelope>();
    }

    #[test]
    fn envelope_carries_step_and_session() {
        let e = EventEnvelope::new(
            SessionId(1),
            4,
            DomainEvent::TokenGenerated { sampled: true },
        );
        assert_eq!(e.step, 4);
        assert_eq!(e.session, SessionId(1));
    }
}
