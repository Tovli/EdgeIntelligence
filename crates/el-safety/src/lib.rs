//! `el-safety` — on-device, tiered, decoder-time safety (ADR-005).
//!
//! The [`SafetyMode`] tier is budget-gated by device profile via
//! [`SafetyModeSelector`]. The `Lightweight` anchor/blacklist filter is fully
//! implemented here. `SecDecoding` (two ~1B models) and `Csd` (claim
//! backtracking) require model assets and are scaffolded as follow-ups
//! ([`SecDecodingSteerer`]). **No safety path touches the network.**

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
}
