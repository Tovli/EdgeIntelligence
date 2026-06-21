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
    /// reports to telemetry. Saturating, so a large steered set can never
    /// overflow the `u32` aggregate.
    pub fn l1_norm_milli(&self) -> u32 {
        self.penalties
            .iter()
            .fold(0u32, |acc, (_, d)| acc.saturating_add(d.unsigned_abs()))
    }

    /// The `(token, milli-delta)` pairs this adjustment applies. Lets callers
    /// compose adjustments (e.g. hard bans + contrastive steering) without
    /// re-deriving them.
    pub fn penalties(&self) -> &[(Token, i32)] {
        &self.penalties
    }
}

/// Per-step safety intervention. The runtime applies this **after** the grammar
/// mask and **before** sampling.
pub trait SafetySteerer {
    fn adjust(&self, recent_tokens: &[Token]) -> LogitAdjustment;
    fn mode(&self) -> SafetyMode;

    /// Logit-aware steering (ADR-013). Given the base model's next-token logits
    /// for this step, return the adjustment to apply (still after the grammar
    /// mask, before sampling — the ADR-005 order is unchanged).
    ///
    /// The default ignores the logits and delegates to [`adjust`](Self::adjust),
    /// so token-only steerers (hard bans, heuristics) need no change. Model-backed
    /// steerers (e.g. [`ContrastiveSteerer`]) override this to use the base
    /// distribution. The runtime calls this **inside** the early-token
    /// soft-steering window and plain [`adjust`](Self::adjust) outside it, so a
    /// model-backed adjustment costs nothing once the window closes.
    fn adjust_with_logits(&self, recent_tokens: &[Token], _base_logits: &[i32]) -> LogitAdjustment {
        self.adjust(recent_tokens)
    }
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
        // from their divergence. See [`ContrastiveSteerer`] (ADR-013) for the
        // real mechanism, wired once an expert logit source is supplied.
        LogitAdjustment::none()
    }
    fn mode(&self) -> SafetyMode {
        SafetyMode::SecDecoding
    }
}

// ---------------------------------------------------------------------------
// ADR-013 — model-backed (contrastive) steering
// ---------------------------------------------------------------------------

/// Safety-tuned next-token logits for the committed context, in integer
/// milli-logits over the **same vocabulary and tokenizer** as the base
/// generator (the ADR-012 shared-tokenizer invariant — the contrastive direction
/// is only meaningful when base and expert share a token set). The expert weights
/// are integrity-gated on load (ADR-006) by the adapter that constructs the
/// implementor; like every safety path this is deterministic and **never touches
/// the network** (ADR-004).
pub trait ExpertLogits {
    /// Expert next-token milli-logits given the committed (generated) context.
    fn logits(&self, committed: &[Token]) -> Vec<i32>;
}

/// SafeDecoding-style **contrastive adjustment** (ADR-013): steer toward the
/// safety expert and away from the base — `final = base + α·(expert − base)` —
/// returned as additive milli-logit penalties consumed after the grammar mask,
/// before sampling (the ADR-005 order is unchanged).
///
/// `alpha_milli` is the steering strength ×1000 (`1000` = 1.0×). Only the
/// `top_k` highest-base-logit tokens are steered (`0` = all): SafeDecoding
/// restricts contrast to the head of the distribution so long-tail noise is not
/// amplified. Deterministic and integer (ADR-008).
///
/// Safety/robustness invariants: a **negative** `alpha_milli` is clamped to `0`
/// — contrastive steering must never push *toward* the base/unsafe direction —
/// and the delta math is **saturating**, so an extreme strength can neither
/// overflow `i64` nor wrap on the `i32` cast. Plus the usual fail-safes: an
/// **empty** result when `base`/`expert` lengths differ (or are empty), and a
/// natural **no-op** when `expert == base` (every delta is zero).
pub fn contrastive_adjustment(
    base: &[i32],
    expert: &[i32],
    alpha_milli: i32,
    top_k: usize,
) -> LogitAdjustment {
    if base.is_empty() || base.len() != expert.len() {
        return LogitAdjustment::none();
    }
    // Never steer toward the base/unsafe direction (clamp negative to zero).
    let alpha = i64::from(alpha_milli.max(0));
    // Tokens to steer: the top_k by base logit (deterministic tie-break on
    // index), or all of them when top_k is 0 or covers the whole vocab.
    let mut idx: Vec<usize> = (0..base.len()).collect();
    if top_k > 0 && top_k < base.len() {
        idx.sort_unstable_by(|&a, &b| base[b].cmp(&base[a]).then(a.cmp(&b)));
        idx.truncate(top_k);
    }
    let mut penalties: Vec<(Token, i32)> = Vec::new();
    for i in idx {
        let diff = i64::from(expert[i]) - i64::from(base[i]);
        // Saturating throughout: extreme alpha saturates rather than wrapping.
        let delta = (diff.saturating_mul(alpha) / 1000)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        if delta != 0 {
            penalties.push((i as Token, delta));
        }
    }
    LogitAdjustment::with_penalties(penalties)
}

/// `SafetyMode::SecDecoding` (or model-backed `Lightweight`) steerer (ADR-013): a
/// hard-ban layer applied **every step** plus **contrastive soft steering** from
/// an [`ExpertLogits`] source applied only when the runtime supplies base logits
/// (i.e. inside the early-token window). With no expert divergence it reduces to
/// the hard-ban behaviour of [`LightweightFilter`]; with no bans and a flat
/// expert it is a no-op.
pub struct ContrastiveSteerer<E: ExpertLogits> {
    expert: E,
    banned: Vec<Token>,
    alpha_milli: i32,
    top_k: usize,
    mode: SafetyMode,
}

impl<E: ExpertLogits> ContrastiveSteerer<E> {
    /// `mode` is the tier label reported to telemetry (`Lightweight` for a LoRA
    /// expert, `SecDecoding` for a contrastive base+expert pair).
    pub fn new(
        expert: E,
        banned: Vec<Token>,
        alpha_milli: i32,
        top_k: usize,
        mode: SafetyMode,
    ) -> Self {
        Self {
            expert,
            banned,
            alpha_milli,
            top_k,
            mode,
        }
    }

    fn bans(&self) -> Vec<(Token, i32)> {
        self.banned
            .iter()
            .map(|&t| (t, LightweightFilter::HARD_BAN))
            .collect()
    }
}

impl<E: ExpertLogits> SafetySteerer for ContrastiveSteerer<E> {
    /// Out-of-window / token-only path: hard bans only (no expert forward).
    fn adjust(&self, _recent: &[Token]) -> LogitAdjustment {
        LogitAdjustment::with_penalties(self.bans())
    }

    /// In-window path: hard bans **plus** contrastive soft steering. Bans are
    /// listed first, so a token that is both banned and steered keeps the hard
    /// ban (`delta_for` returns the first match).
    fn adjust_with_logits(&self, recent: &[Token], base_logits: &[i32]) -> LogitAdjustment {
        let mut penalties = self.bans();
        let expert = self.expert.logits(recent);
        let contrast = contrastive_adjustment(base_logits, &expert, self.alpha_milli, self.top_k);
        penalties.extend_from_slice(contrast.penalties());
        LogitAdjustment::with_penalties(penalties)
    }

    fn mode(&self) -> SafetyMode {
        self.mode
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

/// `SafetyMode::Lightweight` chunk guard — training-free **token-anchor
/// heuristics** (no weights), the [`ChunkGuard`] counterpart of
/// [`LightweightFilter`]. Each *pattern* is an unsafe token-id n-gram; the guard
/// adds `per_hit` milli-units for every pattern that occurs as a contiguous
/// subslice of the scored window, saturating at [`SafetyScore::MAX`].
///
/// Matching whole token sequences (not loose single ids) is deliberate: a
/// multi-token word like "explosive" matches exactly, so a stray subword shared
/// with benign text does not false-positive. A length-1 pattern is a plain
/// single-token anchor.
///
/// Patterns are token ids, so the caller resolves them from its own tokenizer
/// (the adapter that owns one) — this keeps `el-safety` free of any tokenizer or
/// float dependency, and the guard fully deterministic (ADR-008) and offline
/// (ADR-004).
#[derive(Debug, Clone, Default)]
pub struct AnchorGuard {
    patterns: Vec<Vec<Token>>,
    per_hit: u16,
}

impl AnchorGuard {
    /// A guard adding `per_hit_milli` risk per matched pattern. Use
    /// `per_hit_milli >= hard_threshold` so a single match breaches.
    pub fn new(patterns: Vec<Vec<Token>>, per_hit_milli: u16) -> Self {
        Self {
            patterns,
            per_hit: per_hit_milli,
        }
    }

    /// A guard where any single pattern match saturates the score — every
    /// flagged sequence is treated as a hard breach.
    pub fn hard(patterns: Vec<Vec<Token>>) -> Self {
        Self::new(patterns, SafetyScore::MAX.milli())
    }

    /// Whether the guard carries any non-empty pattern to match.
    pub fn is_empty(&self) -> bool {
        self.patterns.iter().all(Vec::is_empty)
    }
}

impl ChunkGuard for AnchorGuard {
    fn score(&self, recent: &[Token]) -> SafetyScore {
        let mut milli: u32 = 0;
        for pat in &self.patterns {
            if pat.is_empty() || pat.len() > recent.len() {
                continue;
            }
            if recent.windows(pat.len()).any(|w| w == pat.as_slice()) {
                milli = milli.saturating_add(u32::from(self.per_hit));
            }
        }
        SafetyScore::from_milli(milli.min(u32::from(u16::MAX)) as u16)
    }
}

/// Cadence and bounds for the checkpointed-rollback control loop (ADR-012),
/// chosen per device tier so cost scales with the hardware budget (ADR-003).
///
/// `steer_window` is the ADR-012 stage-1 **early-token soft-steering window**
/// (ADR-013): model-backed steering applies for the first `steer_window` output
/// tokens, then decode falls back to hard-bans-only unless the guard
/// re-escalates. Hard bans still apply every step regardless. Checkpoint spacing
/// is `guard_every` — checkpoints are only taken at guard-verified-safe
/// boundaries, so there is no separate checkpoint cadence to misconfigure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RollbackPolicy {
    /// Score the output every `guard_every` tokens (`0` = never); also the
    /// spacing of safe-prefix checkpoints.
    pub guard_every: u32,
    /// Apply model-backed (soft) steering for the first `steer_window` output
    /// tokens (`0` = never — token-only/hard-ban steering at every step).
    pub steer_window: u32,
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
                steer_window: 8,
                soft_threshold: SafetyScore(600),
                hard_threshold: SafetyScore(800),
                max_rollbacks: 2,
                max_checkpoints: 4,
            },
            DeviceTarget::HighEnd | DeviceTarget::Auto => Self {
                guard_every: 4,
                steer_window: 16,
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
            steer_window: 0,
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

    /// Whether any safety stage (guard/rollback or the soft-steering window) is
    /// active — used to gate ingress triage (ADR-013).
    pub fn active(&self) -> bool {
        self.guard_every > 0 || self.steer_window > 0
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

    // ----- ADR-013 model-backed (contrastive) steering -----

    /// Synthetic expert: returns fixed logits, ignoring context — deterministic.
    struct FixedExpert(Vec<i32>);
    impl ExpertLogits for FixedExpert {
        fn logits(&self, _committed: &[Token]) -> Vec<i32> {
            self.0.clone()
        }
    }

    #[test]
    fn contrastive_pushes_toward_expert_and_is_noop_when_equal() {
        let base = [100, 200, 300];
        let expert = [600, 200, 100]; // prefers t0, dislikes t2
        let adj = contrastive_adjustment(&base, &expert, 1000, 0); // alpha 1.0, all
        assert_eq!(adj.delta_for(0), 500); // +500 toward expert
        assert_eq!(adj.delta_for(1), 0); // unchanged → dropped
        assert_eq!(adj.delta_for(2), -200); // away from base preference

        // Half strength halves the deltas.
        assert_eq!(
            contrastive_adjustment(&base, &expert, 500, 0).delta_for(0),
            250
        );
        // expert == base → no-op.
        assert!(contrastive_adjustment(&base, &base, 1000, 0).is_empty());
    }

    #[test]
    fn contrastive_top_k_restricts_to_distribution_head() {
        let base = [10, 50, 40, 5]; // top-2 by base: t1, t2
        let expert = [999, 999, 999, 999];
        let adj = contrastive_adjustment(&base, &expert, 1000, 2);
        assert_ne!(adj.delta_for(1), 0);
        assert_ne!(adj.delta_for(2), 0);
        assert_eq!(adj.delta_for(0), 0); // outside the head → untouched
        assert_eq!(adj.delta_for(3), 0);
    }

    #[test]
    fn contrastive_empty_on_length_mismatch_or_empty() {
        assert!(contrastive_adjustment(&[1, 2, 3], &[1, 2], 1000, 0).is_empty());
        assert!(contrastive_adjustment(&[], &[], 1000, 0).is_empty());
    }

    #[test]
    fn contrastive_clamps_negative_and_saturates_extreme_alpha() {
        let base = [100, 200];
        let expert = [900, 0];
        // Negative strength must NOT reverse the safety direction → clamped to a
        // no-op.
        assert!(contrastive_adjustment(&base, &expert, -5000, 0).is_empty());
        // Extreme strength saturates instead of wrapping (no overflow/panic) and
        // still steers toward the expert-preferred token.
        let adj = contrastive_adjustment(&base, &expert, i32::MAX, 0);
        assert!(adj.delta_for(0) > 0);
        assert!(adj.delta_for(1) < 0);
        let _ = adj.l1_norm_milli(); // saturating; must not overflow/panic
    }

    #[test]
    fn contrastive_steerer_layers_bans_and_contrast() {
        let base = [100, 100, 100];
        let expert = FixedExpert(vec![100, 400, 100]); // prefers token 1
        let steerer = ContrastiveSteerer::new(expert, vec![0], 1000, 0, SafetyMode::SecDecoding);

        // Token-only path (out of window): hard ban only, no contrast.
        let banned_only = steerer.adjust(&[]);
        assert_eq!(banned_only.delta_for(0), LightweightFilter::HARD_BAN);
        assert_eq!(banned_only.delta_for(1), 0);

        // In-window path: ban (token 0) + contrastive steer toward token 1.
        let full = steerer.adjust_with_logits(&[], &base);
        assert_eq!(full.delta_for(0), LightweightFilter::HARD_BAN); // ban dominates
        assert_eq!(full.delta_for(1), 300); // 400 - 100
        assert_eq!(steerer.mode(), SafetyMode::SecDecoding);
    }

    #[test]
    fn policy_has_tier_aware_steer_window() {
        let mid = RollbackPolicy::for_device(DeviceTarget::MidRange, SafetyMode::Lightweight);
        let high = RollbackPolicy::for_device(DeviceTarget::HighEnd, SafetyMode::SecDecoding);
        assert!(mid.steer_window > 0 && high.steer_window > 0);
        assert!(high.steer_window >= mid.steer_window);
        let off = RollbackPolicy::for_device(DeviceTarget::Auto, SafetyMode::Off);
        assert_eq!(off.steer_window, 0);
        assert!(!off.active());
    }

    #[test]
    fn anchor_guard_matches_token_ngrams_exactly() {
        // Two patterns: a single-token anchor (9) and a 2-token n-gram (40,41).
        let guard = AnchorGuard::hard(vec![vec![9], vec![40, 41]]);

        // No anchor present → safe.
        assert_eq!(guard.score(&[1, 2, 3]), SafetyScore::SAFE);
        // The single-token anchor anywhere in the window → hard breach.
        assert_eq!(guard.score(&[1, 9, 2]), SafetyScore::MAX);
        // The full 2-gram present (contiguous) → match.
        assert_eq!(guard.score(&[7, 40, 41, 8]), SafetyScore::MAX);
        // The 2-gram's tokens present but NOT contiguous → no false positive.
        assert_eq!(guard.score(&[40, 99, 41]), SafetyScore::SAFE);
    }

    #[test]
    fn anchor_guard_accumulates_below_max_and_is_empty_safe() {
        // per-hit below MAX: one hit is soft, two hits saturate.
        let guard = AnchorGuard::new(vec![vec![1], vec![2]], 600);
        assert_eq!(guard.score(&[1, 5]).milli(), 600);
        assert_eq!(guard.score(&[1, 2]), SafetyScore::MAX); // 1200 clamps to 1000

        // No patterns (or only empty ones) → always safe, never matches.
        assert!(AnchorGuard::hard(vec![]).is_empty());
        assert_eq!(
            AnchorGuard::hard(vec![]).score(&[1, 2, 3]),
            SafetyScore::SAFE
        );
        assert_eq!(
            AnchorGuard::hard(vec![vec![]]).score(&[1]),
            SafetyScore::SAFE
        );
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
