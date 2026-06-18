//! `el-safety` — on-device, tiered, decoder-time safety (ADR-005).
//!
//! The [`SafetyMode`] tier is budget-gated by device profile via
//! [`SafetyModeSelector`]. The `Lightweight` anchor/blacklist filter is fully
//! implemented here. `SecDecoding` (two ~1B models) and `Csd` (claim
//! backtracking) require model assets and are scaffolded as follow-ups
//! ([`SecDecodingSteerer`]). **No safety path touches the network.**
//!
//! ADR-012 adds the runtime-backtracking primitives consumed by the Inference
//! Runtime's decode-time control loop: [`ChunkGuard`]/[`SafetyScore`] scoring,
//! the tier-aware [`RollbackPolicy`] (cadence + bounds), and
//! [`CheckpointManager`]/[`Checkpoint`] safe-prefix snapshots (offsets only —
//! KV payload is never copied).

#![forbid(unsafe_code)]

use el_core::{DeviceTarget, SafetyMode, Token};

/// A vector subtracted from target logits to steer away from unsafe output.
/// Sparse and integer (milli-logits) for deterministic, allocation-light steps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogitAdjustment {
    penalties: Vec<(Token, i32)>,
}

impl LogitAdjustment {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn with_penalties(penalties: Vec<(Token, i32)>) -> Self {
        Self { penalties }
    }

    pub fn is_empty(&self) -> bool {
        self.penalties.is_empty()
    }

    /// The milli-logit delta to add for `token` (0 if unaffected).
    pub fn delta_for(&self, token: Token) -> i32 {
        self.penalties
            .iter()
            .find(|(t, _)| *t == token)
            .map(|(_, d)| *d)
            .unwrap_or(0)
    }

    /// L1 norm in milli-units — what `LogitsSteered.adjustment_norm_milli`
    /// reports to telemetry.
    pub fn l1_norm_milli(&self) -> u32 {
        self.penalties.iter().map(|(_, d)| d.unsigned_abs()).sum()
    }
}

/// Per-step safety intervention. The runtime applies this **after** the grammar
/// mask and **before** sampling.
pub trait SafetySteerer {
    fn adjust(&self, recent_tokens: &[Token]) -> LogitAdjustment;
    fn mode(&self) -> SafetyMode;
}

/// Chooses the affordable mode for the device (ADR-005).
pub struct SafetyModeSelector;

impl SafetyModeSelector {
    /// `SecDecoding` (two ~1B models) is rejected on `MidRange` and downgraded
    /// to `Lightweight`; everything else passes through.
    pub fn resolve(requested: SafetyMode, device: DeviceTarget) -> SafetyMode {
        match (requested, device) {
            (SafetyMode::SecDecoding, DeviceTarget::MidRange) => SafetyMode::Lightweight,
            (m, _) => m,
        }
    }
}

/// `SafetyMode::Off` — a no-op steerer.
pub struct NoSafety;

impl SafetySteerer for NoSafety {
    fn adjust(&self, _recent: &[Token]) -> LogitAdjustment {
        LogitAdjustment::none()
    }
    fn mode(&self) -> SafetyMode {
        SafetyMode::Off
    }
}

/// `SafetyMode::Lightweight` — a training-free blacklist filter (real). Banned
/// tokens receive a very large negative logit so they cannot be sampled.
pub struct LightweightFilter {
    banned: Vec<Token>,
}

impl LightweightFilter {
    pub const HARD_BAN: i32 = -1_000_000;

    pub fn new(banned: Vec<Token>) -> Self {
        Self { banned }
    }
}

impl SafetySteerer for LightweightFilter {
    fn adjust(&self, _recent: &[Token]) -> LogitAdjustment {
        LogitAdjustment::with_penalties(self.banned.iter().map(|&t| (t, Self::HARD_BAN)).collect())
    }
    fn mode(&self) -> SafetyMode {
        SafetyMode::Lightweight
    }
}

/// `SafetyMode::SecDecoding` — base-vs-safety-model logit steering.
///
/// FOLLOW-UP (ADR-005): requires two ~1B models run on Candle. Until model
/// assets are wired, this returns no adjustment and reports its intended mode,
/// so callers can select it without it silently mis-steering.
pub struct SecDecodingSteerer {
    _private: (),
}

impl SecDecodingSteerer {
    pub fn placeholder() -> Self {
        Self { _private: () }
    }
}

impl SafetySteerer for SecDecodingSteerer {
    fn adjust(&self, _recent: &[Token]) -> LogitAdjustment {
        // TODO(adr-005): run base + safety models on Candle, derive adjustment
        // from their divergence.
        LogitAdjustment::none()
    }
    fn mode(&self) -> SafetyMode {
        SafetyMode::SecDecoding
    }
}

// ---------------------------------------------------------------------------
// ADR-012 — checkpointed-rollback control-loop primitives
// ---------------------------------------------------------------------------

/// Risk score in milli-units, `0` (safe) ..= `1000` (max). Integer for
/// deterministic, float-free safety decisions (ADR-008).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct SafetyScore(u16);

impl SafetyScore {
    pub const SAFE: SafetyScore = SafetyScore(0);
    pub const MAX: SafetyScore = SafetyScore(1000);

    /// Clamp a milli-unit value into `[0, 1000]`.
    pub fn from_milli(milli: u16) -> Self {
        Self(milli.min(1000))
    }

    /// The score in milli-units, as reported on `SafetyViolationDetected`.
    pub fn milli(self) -> u16 {
        self.0
    }
}

/// Scores recent generated output for risk (ADR-012 chunk guard). Reuses the
/// active tier's safety model; like every safety path it is deterministic and
/// **never touches the network** (ADR-004).
pub trait ChunkGuard {
    /// Risk of the recent output window. Higher is riskier.
    fn score(&self, recent: &[Token]) -> SafetyScore;
}

/// Cadence and bounds for the checkpointed-rollback control loop (ADR-012),
/// chosen per device tier so cost scales with the hardware budget (ADR-003).
///
/// The early-token *soft-steering window* is intentionally absent: hard bans
/// apply every step, and the windowed SecDecoding-style steering arrives with
/// its model (follow-up). Checkpoint spacing is `guard_every` — checkpoints are
/// only taken at guard-verified-safe boundaries, so there is no separate
/// checkpoint cadence to misconfigure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RollbackPolicy {
    /// Score the output every `guard_every` tokens (`0` = never); also the
    /// spacing of safe-prefix checkpoints.
    pub guard_every: u32,
    /// At/above this score, escalate; do not advance the safe checkpoint.
    pub soft_threshold: SafetyScore,
    /// At/above this score, roll back (or fail closed).
    pub hard_threshold: SafetyScore,
    /// Hard cap on rollbacks before a deterministic refusal (DoS bound).
    pub max_rollbacks: u8,
    /// Bounded checkpoint ring size — fixed memory (ADR-003).
    pub max_checkpoints: u8,
}

impl RollbackPolicy {
    /// Tier-aware policy. `SafetyMode::Off` disables the loop entirely.
    pub fn for_device(device: DeviceTarget, mode: SafetyMode) -> Self {
        if matches!(mode, SafetyMode::Off) {
            return Self::disabled();
        }
        match device {
            DeviceTarget::MidRange => Self {
                guard_every: 16,
                soft_threshold: SafetyScore(600),
                hard_threshold: SafetyScore(800),
                max_rollbacks: 2,
                max_checkpoints: 4,
            },
            DeviceTarget::HighEnd | DeviceTarget::Auto => Self {
                guard_every: 4,
                soft_threshold: SafetyScore(500),
                hard_threshold: SafetyScore(750),
                max_rollbacks: 4,
                max_checkpoints: 8,
            },
        }
    }

    /// A policy that performs no steering, checkpointing, or guarding.
    pub fn disabled() -> Self {
        Self {
            guard_every: 0,
            soft_threshold: SafetyScore::MAX,
            hard_threshold: SafetyScore::MAX,
            max_rollbacks: 0,
            max_checkpoints: 0,
        }
    }

    /// Whether the guard/rollback machinery is active under this policy.
    pub fn guards(&self) -> bool {
        self.guard_every > 0
    }
}

/// A safe-prefix snapshot for rollback (ADR-012). Stores only indices: rollback
/// truncates KV descriptors (`KvRegion::truncate`) and never replays prefill,
/// and the KV payload is never copied (ADR-002/ADR-003).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    /// Committed-token count at the checkpoint.
    pub output_len: u32,
    /// KV-cache length to restore to.
    pub kv_len: u32,
}

/// A bounded ring of guard-verified safe-prefix checkpoints. Fixed memory: the
/// oldest checkpoint is dropped once `cap` is reached (ADR-003). A `cap` of `0`
/// (or [`disable`](Self::disable)) retains nothing — the loop then has no
/// rollback target and fails closed on a breach.
#[derive(Debug, Default)]
pub struct CheckpointManager {
    ring: Vec<Checkpoint>,
    cap: usize,
    enabled: bool,
}

impl CheckpointManager {
    pub fn new(cap: u8) -> Self {
        Self {
            ring: Vec::new(),
            cap: cap as usize,
            enabled: cap > 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Drop all checkpoints and stop retaining new ones (memory-pressure
    /// degradation — ADR-012/ADR-003).
    pub fn disable(&mut self) {
        self.enabled = false;
        self.ring.clear();
    }

    /// Record a safe prefix; evicts the oldest if the ring is full. No-op when
    /// disabled.
    pub fn push(&mut self, checkpoint: Checkpoint) {
        if !self.enabled {
            return;
        }
        if self.ring.len() == self.cap {
            self.ring.remove(0);
        }
        self.ring.push(checkpoint);
    }

    /// The most recent safe prefix, if any.
    pub fn last(&self) -> Option<Checkpoint> {
        self.ring.last().copied()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secdecoding_downgrades_on_midrange() {
        assert_eq!(
            SafetyModeSelector::resolve(SafetyMode::SecDecoding, DeviceTarget::MidRange),
            SafetyMode::Lightweight
        );
        assert_eq!(
            SafetyModeSelector::resolve(SafetyMode::SecDecoding, DeviceTarget::HighEnd),
            SafetyMode::SecDecoding
        );
    }

    #[test]
    fn lightweight_bans_tokens() {
        let f = LightweightFilter::new(vec![42, 99]);
        let adj = f.adjust(&[]);
        assert_eq!(adj.delta_for(42), LightweightFilter::HARD_BAN);
        assert_eq!(adj.delta_for(7), 0);
        assert!(adj.l1_norm_milli() > 0);
    }

    #[test]
    fn safety_score_clamps_and_orders() {
        assert_eq!(SafetyScore::from_milli(5000), SafetyScore::MAX);
        assert!(SafetyScore::SAFE < SafetyScore::MAX);
        assert_eq!(SafetyScore::from_milli(750).milli(), 750);
    }

    #[test]
    fn policy_off_is_disabled() {
        let p = RollbackPolicy::for_device(DeviceTarget::HighEnd, SafetyMode::Off);
        assert!(!p.guards());
        assert_eq!(p.max_rollbacks, 0);
    }

    #[test]
    fn policy_is_tier_aware() {
        let mid = RollbackPolicy::for_device(DeviceTarget::MidRange, SafetyMode::Lightweight);
        let high = RollbackPolicy::for_device(DeviceTarget::HighEnd, SafetyMode::SecDecoding);
        assert!(mid.guards() && high.guards());
        // Stronger hardware guards more often and tolerates more rollbacks.
        assert!(high.guard_every < mid.guard_every);
        assert!(high.max_rollbacks >= mid.max_rollbacks);
    }

    #[test]
    fn checkpoint_ring_is_bounded_and_disablable() {
        let mut m = CheckpointManager::new(2);
        m.push(Checkpoint {
            output_len: 1,
            kv_len: 1,
        });
        m.push(Checkpoint {
            output_len: 2,
            kv_len: 2,
        });
        m.push(Checkpoint {
            output_len: 3,
            kv_len: 3,
        }); // evicts the oldest
        assert_eq!(m.len(), 2);
        assert_eq!(
            m.last(),
            Some(Checkpoint {
                output_len: 3,
                kv_len: 3,
            })
        );
        m.disable();
        assert!(!m.enabled() && m.is_empty());
        m.push(Checkpoint {
            output_len: 9,
            kv_len: 9,
        }); // no-op when disabled
        assert!(m.last().is_none());
    }
}
