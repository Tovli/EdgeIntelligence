//! The `InferenceSession` aggregate root and decode-loop orchestrator.

use crate::ports::{InferenceEngine, Ports};
use el_core::{
    DomainEvent, EdgeError, EventEnvelope, Phase, Result, SessionConfig, SessionId, StopReason,
    Token,
};
use el_memory::KvRegion;
use el_provenance::LoadPermit;
use el_safety::LogitAdjustment;

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
    output: Vec<Token>,
    step: u32,
    events: Vec<EventEnvelope>,
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

    /// Run the decode loop until EOS or `max_tokens`. Each step composes
    /// collaborators in the invariant order: grammar mask → safety adjust →
    /// sample → commit.
    pub fn generate(&mut self, ports: &Ports, max_tokens: u32) -> Result<StopReason> {
        if self.phase != Phase::Decoding {
            return Err(EdgeError::InvalidPhase {
                expected: "Decoding",
                found: self.phase.as_str(),
            });
        }
        let eos = self.engine.eos_token();

        let stop = loop {
            if self.output.len() as u32 >= max_tokens {
                break StopReason::MaxTokens;
            }

            // 2. verify / next-token logits (1. drafting is off by default)
            let logits = self.engine.next_logits(&self.output);
            let vocab = logits.len();

            // 3. grammar mask (BEFORE safety)
            let mask = ports.grammar.mask(&self.output, vocab);
            let allowed = mask.iter().filter(|b| **b).count() as u32;
            self.emit(DomainEvent::TokenMaskApplied { allowed });

            // 4. safety adjust (AFTER mask, BEFORE sampling)
            let adj = ports.safety.adjust(&self.output);
            if !adj.is_empty() {
                self.emit(DomainEvent::LogitsSteered {
                    adjustment_norm_milli: adj.l1_norm_milli(),
                });
            }

            // 5. sample (greedy argmax over legal, steered logits)
            let token = pick(&logits, &mask, &adj);
            self.emit(DomainEvent::TokenGenerated { sampled: false });

            // 6. commit
            self.output.push(token);
            self.kv.push(self.output.len() as u64);
            self.step += 1;
            self.emit(DomainEvent::TokenCommitted {
                kv_len: self.kv.len(),
            });

            if token == eos {
                break StopReason::Eos;
            }
        };

        self.phase = Phase::Completed;
        self.emit(DomainEvent::GenerationCompleted {
            total_tokens: self.output.len() as u32,
            stop,
        });
        Ok(stop)
    }

    /// Clear KV/output for a fresh conversation (volatile memory only).
    pub fn reset(&mut self) {
        self.kv = KvRegion::new();
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
fn pick(logits: &[i32], mask: &[bool], adj: &LogitAdjustment) -> Token {
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
    best.unwrap_or(0)
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
}
