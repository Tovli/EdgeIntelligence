//! Port traits the Inference Runtime depends on (the collaborator contexts).

use el_core::{Result, Token};
use el_safety::{ChunkGuard, SafetySteerer};

/// The inference engine adapter (`RuntimeAcl`). Implemented for real by Candle
/// in the excluded adapter `el-engine-candle` (ADR-002).
///
/// Logits are integer milli-logits to keep the orchestrator deterministic and
/// float-free; a real engine quantises its float logits at the ACL boundary.
pub trait InferenceEngine {
    /// Encode the (compressed) prompt; returns the resulting KV length.
    fn prefill(&mut self, tokens: &[Token]) -> Result<u32>;
    /// Produce next-token logits given the committed context.
    fn next_logits(&mut self, committed: &[Token]) -> Vec<i32>;
    /// The end-of-sequence token id.
    fn eos_token(&self) -> Token;

    /// Roll the engine's internal state back so its context is exactly the
    /// prompt plus `keep_committed` generated tokens.
    ///
    /// The ADR-012 control loop truncates the session's committed output and KV
    /// descriptors on a safety backtrack. A **stateful** engine (one holding a
    /// real KV cache and position counters, e.g. a transformer) must mirror that
    /// truncation here — otherwise it keeps serving logits from the abandoned
    /// (unsafe) branch and never re-feeds the replacement tokens, so the rollback
    /// is silently a no-op at the engine level.
    ///
    /// After `Ok(())`, the next [`next_logits`](Self::next_logits) call — passed a
    /// `committed` slice of length `keep_committed` — must produce logits
    /// consistent with that prefix. Returning `Err` makes the loop fail closed
    /// rather than resume on an inconsistent cache.
    ///
    /// This method is **required, with no default**, deliberately: a default
    /// no-op would let a stateful adapter that forgot to override silently resume
    /// on a stale KV cache — a safety bug that fails *open*. Every engine must
    /// make the choice explicit. A stateless engine whose `next_logits`
    /// recomputes purely from `committed` implements it as `Ok(())`.
    fn rollback(&mut self, keep_committed: u32) -> Result<()>;
}

/// Prompt Compression port (LLMLingua-2 — context 2).
pub trait PromptCompressor {
    fn compress(&self, tokens: &[Token]) -> Vec<Token>;
}

/// Grammar Constraint port (llguidance — context 4). Returns a per-token allow
/// mask of length `vocab`; `true` = legal this step.
pub trait GrammarMasker {
    fn mask(&self, recent: &[Token], vocab: usize) -> Vec<bool>;
}

/// Opt-in LAN relay (ADR-004 HybridMode). Implementations MUST stay on the local
/// network — there is no cloud variant.
pub trait HybridRelay {
    fn consult(&self, query_tokens: &[Token]) -> Vec<Token>;
}

/// The collaborator ports bound to a session. `relay` is `None` by default —
/// air-gapped.
pub struct Ports {
    pub compressor: Box<dyn PromptCompressor>,
    pub grammar: Box<dyn GrammarMasker>,
    pub safety: Box<dyn SafetySteerer>,
    /// Optional chunk guard for the checkpointed-rollback control loop
    /// (ADR-012). `None` runs the plain single-pass decode.
    pub guard: Option<Box<dyn ChunkGuard>>,
    /// Optional **ingress / prompt-risk triage** (ADR-013): scores the prompt
    /// before generation. Reuses the [`ChunkGuard`] contract — it scores a token
    /// window for risk — applied to the prompt rather than the output. `None`
    /// runs no ingress check.
    pub ingress: Option<Box<dyn ChunkGuard>>,
    pub relay: Option<Box<dyn HybridRelay>>,
}

impl Ports {
    /// Defaults: identity compression, all-tokens-allowed grammar, no safety
    /// steering, no guard, no ingress, no relay (air-gapped).
    pub fn permissive() -> Self {
        Self {
            compressor: Box::new(super::defaults::IdentityCompressor),
            grammar: Box::new(super::defaults::AllowAllMasker),
            safety: Box::new(el_safety::NoSafety),
            guard: None,
            ingress: None,
            relay: None,
        }
    }
}
