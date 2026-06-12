//! `el-telemetry` — a one-way, downstream subscriber that folds content-free
//! [`el_core::DomainEvent`]s into performance snapshots (ADR-007).
//!
//! It depends on `el-core` and **nothing depends on it**; it has no network
//! channel. Because it can only read the numeric fields of already-content-free
//! events, "no user content in telemetry" is structural.

#![forbid(unsafe_code)]

use el_core::{DomainEvent, EventEnvelope};

/// A content-free sample of session performance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub prefill_tps: u32,
    pub decode_tps: u32,
    pub ttft_ms: u32,
    pub peak_bytes: u64,
    pub tokens_generated: u32,
    pub compressions: u32,
    pub safety_violations: u32,
}

/// Subscribes to the domain-event stream and maintains a running snapshot.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    snap: TelemetrySnapshot,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        self.snap
    }

    /// Fold one event into the snapshot. Reads only numeric/enum fields.
    pub fn observe(&mut self, env: &EventEnvelope) {
        match env.event {
            DomainEvent::PrefillCompleted { prefill_tps, .. } => {
                self.snap.prefill_tps = prefill_tps;
            }
            DomainEvent::TokenCommitted { .. } => {
                self.snap.tokens_generated += 1;
            }
            DomainEvent::PromptCompressed { .. } => {
                self.snap.compressions += 1;
            }
            DomainEvent::SafetyViolationDetected { .. } => {
                self.snap.safety_violations += 1;
            }
            DomainEvent::MemoryPlanCreated { total_bytes, .. } => {
                self.snap.peak_bytes = self.snap.peak_bytes.max(total_bytes);
            }
            DomainEvent::MetricsSampled {
                decode_tps,
                ttft_ms,
                peak_bytes,
            } => {
                self.snap.decode_tps = decode_tps;
                self.snap.ttft_ms = ttft_ms;
                self.snap.peak_bytes = self.snap.peak_bytes.max(peak_bytes);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use el_core::SessionId;

    fn env(step: u32, event: DomainEvent) -> EventEnvelope {
        EventEnvelope::new(SessionId(1), step, event)
    }

    #[test]
    fn folds_events_into_counters() {
        let mut c = MetricsCollector::new();
        c.observe(&env(
            0,
            DomainEvent::PrefillCompleted {
                prompt_tokens: 100,
                kv_len: 100,
                prefill_tps: 480,
            },
        ));
        c.observe(&env(1, DomainEvent::TokenCommitted { kv_len: 101 }));
        c.observe(&env(2, DomainEvent::TokenCommitted { kv_len: 102 }));
        c.observe(&env(
            3,
            DomainEvent::MetricsSampled {
                decode_tps: 55,
                ttft_ms: 180,
                peak_bytes: 600_000_000,
            },
        ));

        let s = c.snapshot();
        assert_eq!(s.prefill_tps, 480);
        assert_eq!(s.tokens_generated, 2);
        assert_eq!(s.decode_tps, 55);
        assert_eq!(s.peak_bytes, 600_000_000);
    }

    #[test]
    fn peak_bytes_is_monotonic() {
        let mut c = MetricsCollector::new();
        c.observe(&env(
            0,
            DomainEvent::MemoryPlanCreated {
                total_bytes: 500,
                sram_bytes: 100,
                dram_bytes: 400,
            },
        ));
        c.observe(&env(
            1,
            DomainEvent::MetricsSampled {
                decode_tps: 1,
                ttft_ms: 1,
                peak_bytes: 200,
            },
        ));
        assert_eq!(c.snapshot().peak_bytes, 500, "high-water mark only rises");
    }
}
