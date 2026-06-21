//! The `InferenceSession` aggregate root and decode-loop orchestrator.

use crate::ports::{InferenceEngine, Ports};
use el_core::{
    DegradeReason, DomainEvent, EdgeError, EventEnvelope, Phase, Result, SessionConfig, SessionId,
    StopReason, Token,
};
use el_memory::KvRegion;
use el_provenance::LoadPermit;
use el_safety::{
    Checkpoint, CheckpointManager, LogitAdjustment, RollbackPolicy, SafetyModeSelector,
};

/// Below this static-memory budget there is no room to retain rollback
/// checkpoints, so the control loop degrades to guard-only (no rollback) and
/// emits `SafetyDisabled` (ADR-012 tier-aware degradation; ADR-003 budget).
const MIN_CHECKPOINT_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// One live generation. Constructing it requires a [`LoadPermit`], so a model
/// that has not passed the provenance gate (ADR-006) cannot reach the runtime —
/// the Conformist relationship is enforced in the type system.
pub struct InferenceSession<E: InferenceEngine> {
    id: SessionId,
    config: SessionConfig,
    phase: Phase,
    engine: E,
    kv: KvRegion,
    permit: LoadPermit,
    /// The prompt fed at `load_prompt`, retained for ADR-013 ingress triage
    /// (scored before generation).
    prompt: Vec<Token>,
    output: Vec<Token>,
    step: u32,
    events: Vec<EventEnvelope>,
}

/// Outcome of one chunk-guard evaluation in the control loop (ADR-012).
enum GuardVerdict {
    /// Below the hard threshold — safe (safe checkpoint advanced) or tolerated.
    Pass,
    /// Hard breach rolled back to a safe checkpoint; decoding should resume.
    RolledBack,
    /// Hard breach with no rollback budget or target — refuse (fail closed).
    FailClosed,
}

/// Mutable bookkeeping for the checkpointed-rollback loop, threaded through both
/// the cadence guard check and the mandatory final guard check.
struct GuardState {
    checkpoints: CheckpointManager,
    rollback_count: u8,
    banned: Vec<Token>,
    /// The post-prefill safe baseline `(output_len, kv_len)` — the fail-closed
    /// restore target when no checkpoint exists, so prompt prefill KV survives.
    start_out: u32,
    start_kv: u32,
}

impl<E: InferenceEngine> InferenceSession<E> {
    pub fn new(id: SessionId, config: SessionConfig, engine: E, permit: LoadPermit) -> Self {
        let mut s = Self {
            id,
            config,
            phase: Phase::Initialized,
            engine,
            kv: KvRegion::new(),
            permit,
            prompt: Vec::new(),
            output: Vec::new(),
            step: 0,
            events: Vec::new(),
        };
        s.emit(DomainEvent::SessionInitialized {
            runtime: config.format.runtime(),
            device: config.device,
            safety: config.safety,
            speculation: config.speculation,
        });
        s.emit(DomainEvent::ModelLoaded {
            model: permit.model,
            version: permit.version,
            format: permit.format,
        });
        s
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn output(&self) -> &[Token] {
        &self.output
    }
    pub fn kv_len(&self) -> u32 {
        self.kv.len()
    }
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }
    /// The load permit this session was constructed with — evidence the model
    /// passed the provenance gate (ADR-006).
    pub fn permit(&self) -> LoadPermit {
        self.permit
    }
    /// Take the buffered domain events (a real build would stream these to the
    /// Telemetry subscriber).
    pub fn drain_events(&mut self) -> Vec<EventEnvelope> {
        std::mem::take(&mut self.events)
    }

    fn emit(&mut self, event: DomainEvent) {
        self.events
            .push(EventEnvelope::new(self.id, self.step, event));
    }

    /// Compress (optional) → prefill → build KV. Valid only from `Initialized`.
    pub fn load_prompt(&mut self, ports: &Ports, prompt: &[Token]) -> Result<()> {
        if self.phase != Phase::Initialized {
            return Err(EdgeError::InvalidPhase {
                expected: "Initialized",
                found: self.phase.as_str(),
            });
        }

        // Retain the raw prompt for ADR-013 ingress triage (scored in
        // `generate_with_policy` before any token is generated).
        self.prompt = prompt.to_vec();

        let compressed = if self.config.compress {
            ports.compressor.compress(prompt)
        } else {
            prompt.to_vec()
        };
        if compressed.len() < prompt.len() {
            let ratio_milli =
                ((compressed.len() as u64 * 1000) / (prompt.len().max(1) as u64)) as u32;
            self.emit(DomainEvent::PromptCompressed {
                input_tokens: prompt.len() as u32,
                output_tokens: compressed.len() as u32,
                ratio_milli,
            });
        }

        self.phase = Phase::Prefilling;
        let kv_len = self.engine.prefill(&compressed)?;
        for _ in 0..kv_len {
            let off = self.kv.len() as u64;
            self.kv.push(off);
        }
        self.emit(DomainEvent::PrefillCompleted {
            prompt_tokens: compressed.len() as u32,
            kv_len,
            prefill_tps: 0,
        });
        self.phase = Phase::Decoding;
        Ok(())
    }

    /// Run the decode loop until EOS or `max_tokens`, deriving the rollback
    /// policy from the session's device tier and safety mode (ADR-005/ADR-012).
    pub fn generate(&mut self, ports: &Ports, max_tokens: u32) -> Result<StopReason> {
        // Resolve the *effective* tier in the decode path (ADR-013): a tier the
        // device cannot run (e.g. `SecDecoding` on `MidRange`) is downgraded
        // here, and the effective mode — not just the requested one — is what
        // drives the policy and is recorded for telemetry.
        let effective = SafetyModeSelector::resolve(self.config.safety, self.config.device);
        self.emit(DomainEvent::SafetyModeSelected { mode: effective });
        let policy = RollbackPolicy::for_device(self.config.device, effective);
        self.generate_with_policy(ports, max_tokens, policy)
    }

    /// The checkpointed-rollback safety control loop (ADR-012).
    ///
    /// Every step preserves the invariant order **grammar mask → safety adjust →
    /// sample → commit** (ADR-005). When a [`ChunkGuard`](el_safety::ChunkGuard)
    /// is wired and the policy enables guarding, the loop additionally captures a
    /// checkpoint at each guard-verified-safe boundary, scores recent output
    /// every `guard_every` tokens, and on a hard-threshold breach rolls the KV
    /// cache *and* the committed output back to the last safe checkpoint —
    /// banning the offending token through the grammar mask so the resumed
    /// decode necessarily diverges. Rollbacks are bounded by `max_rollbacks`; on
    /// exhaustion — or with no checkpoint (e.g. under memory pressure) — the loop
    /// fails closed with a deterministic refusal (`StopReason::Stopped`).
    ///
    /// Termination (EOS or `max_tokens`) is gated behind a **mandatory final
    /// guard check**: the loop scores the trailing chunk before honouring either
    /// stop condition, so no completion is ever returned unscored — including one
    /// shorter than `guard_every` or whose unsafe tail ends in EOS. A final check
    /// coincident with a cadence boundary is idempotent (re-scoring identical
    /// output yields the same verdict).
    pub fn generate_with_policy(
        &mut self,
        ports: &Ports,
        max_tokens: u32,
        policy: RollbackPolicy,
    ) -> Result<StopReason> {
        if self.phase != Phase::Decoding {
            return Err(EdgeError::InvalidPhase {
                expected: "Decoding",
                found: self.phase.as_str(),
            });
        }

        // ---- ingress / prompt-risk triage (ADR-013) ----
        // Score the prompt before generating anything. A hard breach fails
        // closed deterministically — no unsafe trajectory is ever started. This
        // is the heterogeneous monitor's ingress layer, distinct from the
        // output-side chunk guard below.
        if policy.active() {
            if let Some(ingress) = ports.ingress.as_deref() {
                let score = ingress.score(&self.prompt);
                if score >= policy.hard_threshold {
                    self.emit(DomainEvent::SafetyViolationDetected {
                        score_milli: score.milli(),
                        threshold_milli: policy.hard_threshold.milli(),
                    });
                    self.phase = Phase::Completed;
                    self.emit(DomainEvent::GenerationCompleted {
                        total_tokens: self.output.len() as u32,
                        stop: StopReason::Stopped,
                    });
                    return Ok(StopReason::Stopped);
                }
            }
        }

        let eos = self.engine.eos_token();
        let guarding = policy.guards() && ports.guard.is_some();

        // Tier-aware degradation (ADR-003/ADR-012): without budget for
        // checkpoints, run guard-only with no rollback capability.
        let checkpoints = if guarding {
            if self.config.memory_budget_bytes < MIN_CHECKPOINT_BUDGET_BYTES {
                self.emit(DomainEvent::SafetyDisabled {
                    reason: DegradeReason::MemoryPressure,
                });
                CheckpointManager::new(0)
            } else {
                CheckpointManager::new(policy.max_checkpoints)
            }
        } else {
            CheckpointManager::new(0)
        };
        let mut state = GuardState {
            checkpoints,
            rollback_count: 0,
            banned: Vec::new(),
            // The post-prefill baseline: the safe prefix to restore to when no
            // checkpoint exists (e.g. checkpointing disabled under memory
            // pressure). Captured as (output, KV) so fail-closed never drops
            // prompt prefill KV.
            start_out: self.output.len() as u32,
            start_kv: self.kv.len(),
        };
        // Seed the safe prefix at generation start (an empty continuation is safe).
        if state.checkpoints.enabled() {
            state.checkpoints.push(Checkpoint {
                output_len: state.start_out,
                kv_len: state.start_kv,
            });
        }

        let stop = loop {
            // A *candidate* termination for this iteration: the token cap is
            // reached (checked before generating), or — set below — the model
            // emitted EOS. With a guard active, neither is honoured until the
            // final chunk passes the mandatory guard check, so a short or
            // EOS-terminated tail cannot bypass scoring.
            let mut terminating: Option<StopReason> = None;

            if self.output.len() as u32 >= max_tokens {
                terminating = Some(StopReason::MaxTokens);
            } else {
                // 2. next-token logits (drafting off by default).
                let logits = self.engine.next_logits(&self.output);
                let vocab = logits.len();

                // 3. grammar mask (BEFORE safety). Rollback bans ride the mask so
                //    the resumed decode cannot re-pick the off-trajectory token.
                let mut mask = ports.grammar.mask(&self.output, vocab);
                for &t in &state.banned {
                    if let Some(slot) = mask.get_mut(t as usize) {
                        *slot = false;
                    }
                }
                let allowed = mask.iter().filter(|b| **b).count() as u32;
                self.emit(DomainEvent::TokenMaskApplied { allowed });

                // 4. safety adjust (AFTER mask, BEFORE sampling). Inside the
                //    early-token soft-steering window (ADR-013) the steerer is
                //    given the base logits so a model-backed (contrastive)
                //    steerer can run; outside the window it is token-only (hard
                //    bans every step). For token-only steerers the two paths are
                //    identical (the default `adjust_with_logits` delegates).
                let adj = if (self.output.len() as u32) < policy.steer_window {
                    // Hide grammar-illegal tokens from the steerer so a top-K
                    // model-backed steerer ranks only legal candidates — otherwise
                    // the whole top-K could be illegal and legal tokens get no
                    // adjustment. Skip the copy when the grammar allows everything.
                    if mask.iter().any(|&legal| !legal) {
                        let legal_logits: Vec<i32> = logits
                            .iter()
                            .zip(mask.iter())
                            .map(|(&l, &legal)| if legal { l } else { i32::MIN })
                            .collect();
                        ports.safety.adjust_with_logits(&self.output, &legal_logits)
                    } else {
                        ports.safety.adjust_with_logits(&self.output, &logits)
                    }
                } else {
                    ports.safety.adjust(&self.output)
                };
                if !adj.is_empty() {
                    self.emit(DomainEvent::LogitsSteered {
                        adjustment_norm_milli: adj.l1_norm_milli(),
                    });
                }

                // 5. sample (greedy argmax over legal, steered logits). If grammar
                //    + rollback bans leave no legal token, fail closed rather than
                //    emit a masked/banned token.
                let token = match pick(&logits, &mask, &adj) {
                    Some(t) => t,
                    None => {
                        self.emit(DomainEvent::GrammarViolationBlocked);
                        break StopReason::Stopped;
                    }
                };
                self.emit(DomainEvent::TokenGenerated { sampled: false });

                // 6. commit.
                self.output.push(token);
                self.kv.push(self.output.len() as u64);
                self.step += 1;
                self.emit(DomainEvent::TokenCommitted {
                    kv_len: self.kv.len(),
                });

                if token == eos {
                    terminating = Some(StopReason::Eos);
                }
            }

            // ---- chunk guard + checkpointed rollback (ADR-012) ----
            // Score at each `guard_every` cadence boundary AND before any
            // termination (the mandatory final check). This closes the bypass
            // where EOS or the token cap returned a tail shorter than
            // `guard_every` unscored.
            if guarding {
                let guard = ports
                    .guard
                    .as_deref()
                    .expect("guarding implies a guard is wired");
                let at_boundary = (self.output.len() as u32).is_multiple_of(policy.guard_every);
                if terminating.is_some() || at_boundary {
                    match self.guard_chunk(guard, &policy, &mut state) {
                        // Fail closed: no checkpoint, or rollback budget spent.
                        GuardVerdict::FailClosed => break StopReason::Stopped,
                        // Rolled back: the candidate termination (if any) was
                        // undone with it, so resume decoding from the safe prefix.
                        GuardVerdict::RolledBack => continue,
                        GuardVerdict::Pass => {}
                    }
                }
            }

            if let Some(reason) = terminating {
                break reason;
            }
        };

        self.phase = Phase::Completed;
        self.emit(DomainEvent::GenerationCompleted {
            total_tokens: self.output.len() as u32,
            stop,
        });
        Ok(stop)
    }

    /// Score the committed output and apply the ADR-012 rollback policy:
    /// advance the safe checkpoint when verified safe, roll back (banning the
    /// divergence token) on a hard breach within budget, or fail closed.
    ///
    /// Invoked both at `guard_every` cadence boundaries and as the mandatory
    /// final check before termination, so no completion is returned unscored.
    /// On [`GuardVerdict::RolledBack`]/[`GuardVerdict::FailClosed`] the output
    /// **and** KV are truncated together to the safe prefix (or the post-prefill
    /// baseline) so prompt prefill descriptors are never dropped (AC-5). A
    /// rollback also restores the *engine's* internal state via
    /// [`InferenceEngine::rollback`] — a stateful engine (real KV cache) that
    /// kept the abandoned branch would otherwise serve logits from the unsafe
    /// path and skip the replacement tokens.
    fn guard_chunk(
        &mut self,
        guard: &dyn el_safety::ChunkGuard,
        policy: &RollbackPolicy,
        state: &mut GuardState,
    ) -> GuardVerdict {
        let score = guard.score(&self.output);
        if score >= policy.hard_threshold {
            self.emit(DomainEvent::SafetyViolationDetected {
                score_milli: score.milli(),
                threshold_milli: policy.hard_threshold.milli(),
            });
            match state.checkpoints.last() {
                Some(cp) if state.rollback_count < policy.max_rollbacks => {
                    // Restore the engine's internal state (real KV cache /
                    // position) to the checkpoint too. If it cannot, fail closed
                    // rather than resume decoding on an inconsistent cache.
                    if self.engine.rollback(cp.output_len).is_err() {
                        self.output.truncate(cp.output_len as usize);
                        self.kv.truncate(cp.kv_len);
                        return GuardVerdict::FailClosed;
                    }
                    // Ban the token that began the unsafe span → divergence.
                    if let Some(&bad) = self.output.get(cp.output_len as usize) {
                        state.banned.push(bad);
                    }
                    self.output.truncate(cp.output_len as usize);
                    self.kv.truncate(cp.kv_len);
                    state.rollback_count += 1;
                    self.emit(DomainEvent::ClaimBacktracked {
                        claim_index: cp.output_len,
                    });
                    GuardVerdict::RolledBack
                }
                _ => {
                    let (safe_out, safe_kv) = state
                        .checkpoints
                        .last()
                        .map_or((state.start_out, state.start_kv), |c| {
                            (c.output_len, c.kv_len)
                        });
                    self.output.truncate(safe_out as usize);
                    self.kv.truncate(safe_kv);
                    GuardVerdict::FailClosed
                }
            }
        } else if score < policy.soft_threshold {
            // Verified safe: advance the last-safe checkpoint, drop bans.
            if state.checkpoints.enabled() {
                state.checkpoints.push(Checkpoint {
                    output_len: self.output.len() as u32,
                    kv_len: self.kv.len(),
                });
            }
            state.banned.clear();
            GuardVerdict::Pass
        } else {
            // soft ≤ score < hard: tolerated but not checkpointed (still risky).
            GuardVerdict::Pass
        }
    }

    /// Clear KV/output for a fresh conversation (volatile memory only).
    pub fn reset(&mut self) {
        self.kv = KvRegion::new();
        self.prompt.clear();
        self.output.clear();
        self.step = 0;
        self.phase = Phase::Initialized;
        self.emit(DomainEvent::SessionReset);
    }

    /// Consult the opt-in LAN relay. Hard-fails with [`EdgeError::AirGapViolation`]
    /// unless `hybrid_mode` is enabled AND a relay is wired (ADR-004).
    pub fn consult_relay(&mut self, ports: &Ports, query: &[Token]) -> Result<Vec<Token>> {
        if !self.config.hybrid_mode {
            return Err(EdgeError::AirGapViolation);
        }
        match &ports.relay {
            Some(relay) => {
                let out = relay.consult(query);
                self.emit(DomainEvent::HybridRelayConsulted);
                Ok(out)
            }
            None => Err(EdgeError::AirGapViolation),
        }
    }
}

/// Greedy pick over legal, safety-steered logits. Masked-out tokens are skipped
/// entirely; the safety delta is added to surviving logits before argmax.
/// Returns `None` when no token is legal (every token masked out or banned), so
/// the caller fails closed instead of emitting a rejected token.
fn pick(logits: &[i32], mask: &[bool], adj: &LogitAdjustment) -> Option<Token> {
    let mut best: Option<Token> = None;
    let mut best_val = i32::MIN;
    for (i, &l) in logits.iter().enumerate() {
        if mask.get(i).copied() == Some(false) {
            continue;
        }
        let v = l.saturating_add(adj.delta_for(i as Token));
        if v > best_val {
            best_val = v;
            best = Some(i as Token);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::NullEngine;
    use crate::ports::{GrammarMasker, Ports};
    use el_core::{ModelFormat, ModelId, ModelVersion};
    use el_provenance::{ModelArtifact, SignatureVerifier};
    use el_safety::LightweightFilter;

    struct OkVerifier;
    impl SignatureVerifier for OkVerifier {
        fn verify(&self, _b: &[u8], _s: &[u8], _k: u32) -> bool {
            true
        }
    }

    fn permit() -> LoadPermit {
        let mut a = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
        a.verify(&OkVerifier, b"weights", b"sig", 1);
        a.ensure_loadable().expect("verified artifact loads")
    }

    /// A deterministic engine returning fixed logits; eos out of vocab range so
    /// it never self-terminates (used for the composition-order test).
    struct FixedEngine {
        logits: Vec<i32>,
    }
    impl InferenceEngine for FixedEngine {
        fn prefill(&mut self, t: &[Token]) -> Result<u32> {
            Ok(t.len() as u32)
        }
        fn next_logits(&mut self, _c: &[Token]) -> Vec<i32> {
            self.logits.clone()
        }
        fn eos_token(&self) -> Token {
            9999
        }
        fn rollback(&mut self, _keep: u32) -> Result<()> {
            Ok(()) // stateless
        }
    }

    // Grammar masker that disallows specific token ids.
    struct DisallowMasker(Vec<Token>);
    impl GrammarMasker for DisallowMasker {
        fn mask(&self, _recent: &[Token], vocab: usize) -> Vec<bool> {
            (0..vocab as Token).map(|t| !self.0.contains(&t)).collect()
        }
    }

    #[test]
    fn full_lifecycle_init_prefill_decode_complete_reset() {
        let mut s = InferenceSession::new(
            SessionId(1),
            SessionConfig::default(),
            NullEngine::new(3, 8),
            permit(),
        );
        assert_eq!(s.phase(), Phase::Initialized);

        let ports = Ports::permissive();
        s.load_prompt(&ports, &[10, 11, 12]).unwrap();
        assert_eq!(s.phase(), Phase::Decoding);

        let stop = s.generate(&ports, 16).unwrap();
        assert_eq!(stop, StopReason::Eos); // NullEngine emits EOS first step
        assert_eq!(s.output(), &[3]);
        assert_eq!(s.phase(), Phase::Completed);

        s.reset();
        assert_eq!(s.phase(), Phase::Initialized);
        assert!(s.output().is_empty());
    }

    #[test]
    fn decode_applies_grammar_before_safety_before_sampling() {
        // logits favour token 0 (10), then 1 (9), then 2 (8), then 3 (7).
        let engine = FixedEngine {
            logits: vec![10, 9, 8, 7],
        };
        let mut s = InferenceSession::new(SessionId(2), SessionConfig::default(), engine, permit());

        let ports = Ports {
            compressor: Box::new(crate::defaults::IdentityCompressor),
            grammar: Box::new(DisallowMasker(vec![0])), // grammar removes the top token
            safety: Box::new(LightweightFilter::new(vec![1])), // safety bans the next-best
            guard: None,
            ingress: None,
            relay: None,
        };
        s.load_prompt(&ports, &[1]).unwrap();
        let stop = s.generate(&ports, 1).unwrap();

        assert_eq!(stop, StopReason::MaxTokens);
        // Token 0 removed by grammar, token 1 banned by safety → token 2 wins.
        // Proves order: mask → adjust → sample.
        assert_eq!(s.output(), &[2]);
    }

    #[test]
    fn generate_before_load_prompt_is_invalid_phase() {
        let mut s = InferenceSession::new(
            SessionId(3),
            SessionConfig::default(),
            NullEngine::new(0, 4),
            permit(),
        );
        let ports = Ports::permissive();
        let err = s.generate(&ports, 4).unwrap_err();
        assert!(matches!(err, EdgeError::InvalidPhase { .. }));
    }

    #[test]
    fn relay_is_blocked_unless_hybrid_mode_opted_in() {
        struct EchoRelay;
        impl crate::ports::HybridRelay for EchoRelay {
            fn consult(&self, q: &[Token]) -> Vec<Token> {
                q.to_vec()
            }
        }

        // Air-gapped by default: even with a relay wired, consulting fails.
        let mut s = InferenceSession::new(
            SessionId(4),
            SessionConfig::default(),
            NullEngine::new(0, 4),
            permit(),
        );
        let ports = Ports {
            relay: Some(Box::new(EchoRelay)),
            ..Ports::permissive()
        };
        assert_eq!(
            s.consult_relay(&ports, &[1, 2]).unwrap_err(),
            EdgeError::AirGapViolation
        );

        // Opt in → allowed.
        let cfg = SessionConfig {
            hybrid_mode: true,
            ..SessionConfig::default()
        };
        let mut s2 = InferenceSession::new(SessionId(5), cfg, NullEngine::new(0, 4), permit());
        assert_eq!(s2.consult_relay(&ports, &[1, 2]).unwrap(), vec![1, 2]);

        // Opted in but no relay wired → still air-gapped.
        let no_relay = Ports::permissive();
        assert_eq!(
            s2.consult_relay(&no_relay, &[1]).unwrap_err(),
            EdgeError::AirGapViolation
        );
    }

    #[test]
    fn first_events_are_init_then_model_loaded() {
        let mut s = InferenceSession::new(
            SessionId(6),
            SessionConfig::default(),
            NullEngine::new(0, 4),
            permit(),
        );
        let evs = s.drain_events();
        assert!(matches!(
            evs[0].event,
            DomainEvent::SessionInitialized { .. }
        ));
        assert!(matches!(evs[1].event, DomainEvent::ModelLoaded { .. }));
    }

    // ----- ADR-012 checkpointed-rollback control loop -----

    use el_safety::{ChunkGuard, SafetyScore};

    /// Hard-unsafe whenever the given token appears in the output.
    struct BanToken(Token);
    impl ChunkGuard for BanToken {
        fn score(&self, recent: &[Token]) -> SafetyScore {
            if recent.contains(&self.0) {
                SafetyScore::MAX
            } else {
                SafetyScore::SAFE
            }
        }
    }

    /// Always hard-unsafe — exercises the rollback bound and fail-closed path.
    struct AlwaysHot;
    impl ChunkGuard for AlwaysHot {
        fn score(&self, _recent: &[Token]) -> SafetyScore {
            SafetyScore::MAX
        }
    }

    fn tiny_policy(max_rollbacks: u8) -> RollbackPolicy {
        RollbackPolicy {
            guard_every: 1,
            steer_window: 0,
            soft_threshold: SafetyScore::from_milli(500),
            hard_threshold: SafetyScore::from_milli(800),
            max_rollbacks,
            max_checkpoints: 8,
        }
    }

    /// Grammar masker that disallows every token — exercises the no-legal-token
    /// fail-closed path.
    struct DenyAllMasker;
    impl GrammarMasker for DenyAllMasker {
        fn mask(&self, _recent: &[Token], vocab: usize) -> Vec<bool> {
            vec![false; vocab]
        }
    }

    /// Engine that always prefers token 0, then 1, 2, 3 — so banning the
    /// chosen token forces the next-best, giving deterministic divergence.
    fn descending_engine() -> FixedEngine {
        FixedEngine {
            logits: vec![5, 4, 3, 2],
        }
    }

    /// Emits the unsafe token `0` first, then EOS — a completion shorter than a
    /// large `guard_every`, so only the *final* mandatory guard check can catch
    /// it. Banning token `0` forces the next-best (a safe token), then EOS.
    struct UnsafeThenEos {
        eos: Token,
        vocab: usize,
    }
    impl InferenceEngine for UnsafeThenEos {
        fn prefill(&mut self, t: &[Token]) -> Result<u32> {
            Ok(t.len() as u32)
        }
        fn next_logits(&mut self, ctx: &[Token]) -> Vec<i32> {
            let mut v = vec![0i32; self.vocab];
            if ctx.is_empty() {
                v[0] = 10; // unsafe token 0 wins the first step
            } else {
                v[self.eos as usize] = 10; // then terminate with EOS
            }
            v
        }
        fn eos_token(&self) -> Token {
            self.eos
        }
        fn rollback(&mut self, _keep: u32) -> Result<()> {
            Ok(()) // stateless: logits depend only on the passed ctx
        }
    }

    /// `guard_every` larger than any completion here, so the *cadence* check
    /// never fires — isolating the mandatory final guard check.
    fn coarse_policy(max_rollbacks: u8) -> RollbackPolicy {
        RollbackPolicy {
            guard_every: 16,
            steer_window: 0,
            soft_threshold: SafetyScore::from_milli(500),
            hard_threshold: SafetyScore::from_milli(800),
            max_rollbacks,
            max_checkpoints: 8,
        }
    }

    #[test]
    fn eos_terminated_short_completion_is_scored_not_bypassed() {
        // Regression (P1): EOS was handled before guard evaluation, so an unsafe
        // tail ending in EOS within < guard_every tokens escaped scoring.
        let mut s = InferenceSession::new(
            SessionId(26),
            SessionConfig::default(),
            UnsafeThenEos { eos: 5, vocab: 8 },
            permit(),
        );
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        // No rollback budget → the final check must refuse, not return EOS.
        let stop = s.generate_with_policy(&ports, 8, coarse_policy(0)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty()); // unsafe EOS-terminated tail not emitted
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. })));
    }

    #[test]
    fn max_tokens_partial_chunk_is_scored_not_bypassed() {
        // Regression (P1): hitting max_tokens exited before flushing a partial
        // chunk, so a completion shorter than guard_every was never scored.
        let mut s = InferenceSession::new(
            SessionId(27),
            SessionConfig::default(),
            descending_engine(), // always prefers unsafe token 0
            permit(),
        );
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        // 2 tokens < guard_every (16): only the final check can catch the breach.
        let stop = s.generate_with_policy(&ports, 2, coarse_policy(0)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty());
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. })));
    }

    #[test]
    fn eos_unsafe_tail_rolls_back_and_recovers() {
        // With rollback budget, an unsafe EOS-terminated tail is rolled back to
        // the seeded safe prefix, the offending token banned, and decoding
        // resumes to a safe completion that then terminates cleanly.
        let mut s = InferenceSession::new(
            SessionId(28),
            SessionConfig::default(),
            UnsafeThenEos { eos: 5, vocab: 8 },
            permit(),
        );
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        let stop = s.generate_with_policy(&ports, 8, coarse_policy(1)).unwrap();

        assert_eq!(stop, StopReason::Eos);
        assert!(!s.output().contains(&0)); // unsafe token banned out of the result
        assert_eq!(s.output().last(), Some(&5)); // ends on EOS, scored safe
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::ClaimBacktracked { .. })));
    }

    /// Mirrors `QwenEngine`'s statefulness: tracks how many committed tokens it
    /// has "fed" into its (mock) KV cache. If the session truncated its own
    /// output without telling the engine, `committed` would shrink below `fed` —
    /// the desync the real engine hits as serving stale logits and never
    /// re-feeding. Shared cells let the test observe behaviour after the engine
    /// is moved into the session.
    struct StatefulEngine {
        fed: usize,
        logits: Vec<i32>,
        eos: Token,
        rollbacks: std::rc::Rc<std::cell::Cell<u32>>,
        last_keep: std::rc::Rc<std::cell::Cell<u32>>,
        desynced: std::rc::Rc<std::cell::Cell<bool>>,
    }
    impl InferenceEngine for StatefulEngine {
        fn prefill(&mut self, t: &[Token]) -> Result<u32> {
            self.fed = 0;
            Ok(t.len() as u32)
        }
        fn next_logits(&mut self, committed: &[Token]) -> Vec<i32> {
            // The engine should never be "ahead" of the committed context; if it
            // is, a rollback was not propagated here (the bug).
            if self.fed > committed.len() {
                self.desynced.set(true);
            }
            while self.fed < committed.len() {
                self.fed += 1;
            }
            self.logits.clone()
        }
        fn eos_token(&self) -> Token {
            self.eos
        }
        fn rollback(&mut self, keep_committed: u32) -> Result<()> {
            self.fed = keep_committed as usize; // re-sync to the retained prefix
            self.rollbacks.set(self.rollbacks.get() + 1);
            self.last_keep.set(keep_committed);
            Ok(())
        }
    }

    #[test]
    fn rollback_restores_engine_state_not_just_session_metadata() {
        // Regression (P1): the loop truncated only the session's output + KV
        // descriptors; a stateful engine kept the abandoned branch. The session
        // must drive `InferenceEngine::rollback` on every backtrack.
        let rollbacks = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let last_keep = std::rc::Rc::new(std::cell::Cell::new(u32::MAX));
        let desynced = std::rc::Rc::new(std::cell::Cell::new(false));
        let engine = StatefulEngine {
            fed: 0,
            logits: vec![5, 4, 3, 2], // prefers the unsafe token 0
            eos: 9999,
            rollbacks: rollbacks.clone(),
            last_keep: last_keep.clone(),
            desynced: desynced.clone(),
        };
        let mut s =
            InferenceSession::new(SessionId(29), SessionConfig::default(), engine, permit());
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[7, 8]).unwrap(); // non-empty prompt

        let stop = s.generate_with_policy(&ports, 3, tiny_policy(3)).unwrap();

        assert_eq!(stop, StopReason::MaxTokens);
        assert_eq!(s.output(), &[1, 1, 1]); // recovered safe completion
                                            // The engine was told to roll back — not just the session metadata...
        assert!(
            rollbacks.get() >= 1,
            "session must propagate the backtrack to the engine"
        );
        // ...to a real prefix, and the cache never desynced from `committed`.
        assert!(last_keep.get() < 3);
        assert!(
            !desynced.get(),
            "engine cache must track the session rollback"
        );
    }

    fn guarded_ports(guard: Box<dyn ChunkGuard>) -> Ports {
        Ports {
            compressor: Box::new(crate::defaults::IdentityCompressor),
            grammar: Box::new(crate::defaults::AllowAllMasker),
            safety: Box::new(el_safety::NoSafety),
            guard: Some(guard),
            ingress: None,
            relay: None,
        }
    }

    #[test]
    fn hard_breach_rolls_back_kv_and_recovers() {
        let mut s = InferenceSession::new(
            SessionId(20),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        let stop = s.generate_with_policy(&ports, 3, tiny_policy(3)).unwrap();

        assert_eq!(stop, StopReason::MaxTokens);
        // Token 0 is unsafe; each occurrence is rolled back and banned, so the
        // recovered output contains only the safe next-best token.
        assert_eq!(s.output(), &[1, 1, 1]);
        assert!(!s.output().contains(&0));
        // KV rewound in lock-step with the committed output (AC-5).
        assert_eq!(s.kv_len(), s.output().len() as u32);

        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::ClaimBacktracked { .. })));
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. })));
    }

    #[test]
    fn fail_closed_refusal_when_no_rollback_budget() {
        let mut s = InferenceSession::new(
            SessionId(21),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        // No rollback budget → first hard breach refuses deterministically.
        let stop = s.generate_with_policy(&ports, 5, tiny_policy(0)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty());
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. })));
        assert!(!evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::ClaimBacktracked { .. })));
    }

    #[test]
    fn rollbacks_are_bounded_then_fail_closed() {
        let mut s = InferenceSession::new(
            SessionId(22),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = guarded_ports(Box::new(AlwaysHot));
        s.load_prompt(&ports, &[]).unwrap();

        let stop = s.generate_with_policy(&ports, 8, tiny_policy(2)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        let evs = s.drain_events();
        let rollbacks = evs
            .iter()
            .filter(|e| matches!(e.event, DomainEvent::ClaimBacktracked { .. }))
            .count();
        // Exactly max_rollbacks attempts, then a deterministic refusal (AC-6).
        assert_eq!(rollbacks, 2);
    }

    #[test]
    fn memory_pressure_disables_checkpoints_and_fails_closed() {
        let cfg = SessionConfig {
            memory_budget_bytes: 1024, // far below MIN_CHECKPOINT_BUDGET_BYTES
            ..SessionConfig::default()
        };
        let mut s = InferenceSession::new(SessionId(23), cfg, descending_engine(), permit());
        let ports = guarded_ports(Box::new(BanToken(0)));
        s.load_prompt(&ports, &[]).unwrap();

        let stop = s.generate_with_policy(&ports, 5, tiny_policy(3)).unwrap();

        assert_eq!(stop, StopReason::Stopped); // no checkpoint to roll back to
        let evs = s.drain_events();
        assert!(evs.iter().any(|e| matches!(
            e.event,
            DomainEvent::SafetyDisabled {
                reason: DegradeReason::MemoryPressure
            }
        )));
    }

    #[test]
    fn no_legal_token_fails_closed() {
        // Grammar disallows everything → pick() has no legal token. The loop must
        // fail closed, not commit token 0 (which could be EOS).
        let mut s = InferenceSession::new(
            SessionId(24),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = Ports {
            compressor: Box::new(crate::defaults::IdentityCompressor),
            grammar: Box::new(DenyAllMasker),
            safety: Box::new(el_safety::NoSafety),
            guard: None,
            ingress: None,
            relay: None,
        };
        s.load_prompt(&ports, &[]).unwrap();

        let stop = s.generate(&ports, 4).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty()); // no illegal token committed
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::GrammarViolationBlocked)));
    }

    #[test]
    fn fail_closed_preserves_prefill_kv() {
        // Non-empty prompt → prefill KV descriptors must survive a
        // budget-exhausted refusal (regression: fail-closed once truncated KV to
        // output_len, dropping prefill).
        let mut s = InferenceSession::new(
            SessionId(25),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = guarded_ports(Box::new(AlwaysHot));
        s.load_prompt(&ports, &[7, 8, 9]).unwrap(); // prefill KV length = 3
        assert_eq!(s.kv_len(), 3);

        let stop = s.generate_with_policy(&ports, 8, tiny_policy(1)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty()); // refused back to the post-prefill prefix
        assert_eq!(s.kv_len(), 3); // prefill KV intact, not truncated to 0
    }

    // ----- ADR-013 model-backed steering: window, ingress, mode selector -----

    use el_core::SafetyMode;
    use el_safety::SafetySteerer;

    /// Records, per step, whether the logit-aware path was taken and the output
    /// length at that step — so a test can prove the early-token window gating.
    struct RecordingSteerer {
        log: std::rc::Rc<std::cell::RefCell<Vec<(bool, usize)>>>,
    }
    impl SafetySteerer for RecordingSteerer {
        fn adjust(&self, recent: &[Token]) -> LogitAdjustment {
            self.log.borrow_mut().push((false, recent.len()));
            LogitAdjustment::none()
        }
        fn adjust_with_logits(&self, recent: &[Token], _base: &[i32]) -> LogitAdjustment {
            self.log.borrow_mut().push((true, recent.len()));
            LogitAdjustment::none()
        }
        fn mode(&self) -> SafetyMode {
            SafetyMode::SecDecoding
        }
    }

    #[test]
    fn soft_steer_applies_only_inside_the_early_token_window() {
        // AC-1: adjust_with_logits runs for output positions < steer_window, and
        // plain token-only adjust() afterwards.
        let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut s = InferenceSession::new(
            SessionId(30),
            SessionConfig::default(),
            descending_engine(), // never EOS (eos 9999)
            permit(),
        );
        let ports = Ports {
            compressor: Box::new(crate::defaults::IdentityCompressor),
            grammar: Box::new(crate::defaults::AllowAllMasker),
            safety: Box::new(RecordingSteerer { log: log.clone() }),
            guard: None,
            ingress: None,
            relay: None,
        };
        s.load_prompt(&ports, &[]).unwrap();
        let policy = RollbackPolicy {
            guard_every: 0,
            steer_window: 2,
            soft_threshold: SafetyScore::MAX,
            hard_threshold: SafetyScore::MAX,
            max_rollbacks: 0,
            max_checkpoints: 0,
        };
        s.generate_with_policy(&ports, 4, policy).unwrap();

        let calls = log.borrow();
        assert_eq!(calls.len(), 4);
        for &(with_logits, len) in calls.iter() {
            assert_eq!(
                with_logits,
                len < 2,
                "window gate wrong at output len {len}"
            );
        }
    }

    #[test]
    fn ingress_triage_fails_closed_before_generation() {
        // AC-3: a prompt scored at/above the hard threshold refuses with no decode.
        let mut s = InferenceSession::new(
            SessionId(31),
            SessionConfig::default(),
            descending_engine(),
            permit(),
        );
        let ports = Ports {
            compressor: Box::new(crate::defaults::IdentityCompressor),
            grammar: Box::new(crate::defaults::AllowAllMasker),
            safety: Box::new(el_safety::NoSafety),
            guard: None,
            ingress: Some(Box::new(AlwaysHot)), // prompt scores MAX
            relay: None,
        };
        s.load_prompt(&ports, &[1, 2, 3]).unwrap();

        let stop = s.generate_with_policy(&ports, 8, coarse_policy(0)).unwrap();

        assert_eq!(stop, StopReason::Stopped);
        assert!(s.output().is_empty());
        let evs = s.drain_events();
        assert!(evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. })));
        // Fail-closed at ingress means nothing was ever generated/committed.
        assert!(!evs
            .iter()
            .any(|e| matches!(e.event, DomainEvent::TokenCommitted { .. })));
    }

    #[test]
    fn generate_applies_safety_mode_selector_and_records_effective_mode() {
        // AC-4: SecDecoding on MidRange downgrades to Lightweight in the decode
        // path, and the effective mode is what gets recorded.
        let cfg = SessionConfig {
            device: el_core::DeviceTarget::MidRange,
            safety: SafetyMode::SecDecoding,
            ..SessionConfig::default()
        };
        let mut s = InferenceSession::new(SessionId(32), cfg, NullEngine::new(0, 4), permit());
        let ports = Ports::permissive();
        s.load_prompt(&ports, &[1]).unwrap();
        s.generate(&ports, 4).unwrap();

        let evs = s.drain_events();
        assert!(evs.iter().any(|e| matches!(
            e.event,
            DomainEvent::SafetyModeSelected {
                mode: SafetyMode::Lightweight
            }
        )));
    }
}
