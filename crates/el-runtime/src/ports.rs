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

    /// Return the engine to its **pristine, pre-prefill** state so the same
    /// loaded weights can serve a *new conversation* without being reloaded
    /// (ADR-018). After `Ok(())`, the next [`prefill`](Self::prefill) must build a
    /// KV cache from scratch as if the engine had just been constructed.
    ///
    /// This is distinct from [`rollback`](Self::rollback): `rollback(keep)` rewinds
    /// *within* a generation to a safe prefix of length `keep` (ADR-012);
    /// `reset_cache` discards the whole conversation. It is what lets a provider
    /// hold one resident model and reuse it across turns instead of re-reading the
    /// weights from disk every call.
    ///
    /// Like `rollback`, it is **required with no default**: a stateful adapter that
    /// forgot to override would otherwise carry a stale cache into the next
    /// conversation. A stateless engine whose `next_logits` recomputes purely from
    /// `committed` implements it as `Ok(())`.
    fn reset_cache(&mut self) -> Result<()>;

    /// Prefill `full_context`, **reusing the KV already cached for its longest
    /// matching prefix** and feeding only the divergent suffix — cross-turn
    /// incremental prefill (ADR-018 AC-3). Returns the resulting KV length.
    ///
    /// This is the engine half of [`InferenceSession::continue_prompt`]: on a
    /// follow-up turn the whole conversation is re-rendered and re-tokenized, but a
    /// stateful engine that still holds the prior turn's KV can skip re-encoding the
    /// unchanged prefix. The token-level prefix match against the live cache **is**
    /// the tokenizer-round-trip guard — if the re-tokenized context diverges from
    /// what was cached (decode→encode is not always identity), reuse stops at the
    /// divergence and the suffix is fed fresh.
    ///
    /// **Soundness contract.** After `Ok`, the engine MUST be in the exact state a
    /// `reset_cache()` + `prefill(full_context)` would have left it — identical
    /// logits for any subsequent [`next_logits`](Self::next_logits). Reuse is purely
    /// a compute optimisation; it must never change *what* the cache represents, so
    /// the runtime's safety checks (which re-run over `full_context` every turn) see
    /// identical data.
    ///
    /// Unlike [`rollback`](Self::rollback)/[`reset_cache`](Self::reset_cache), a wrong
    /// implementation here is a correctness/perf regression, not a safety
    /// fail-*open* — so this has a **safe default**: discard the cache and re-prefill
    /// the whole context (no reuse). Stateful engines override it for the fast path;
    /// stateless engines (whose `next_logits` recomputes from `committed`) inherit
    /// the default unchanged.
    fn prefill_reuse(&mut self, full_context: &[Token]) -> Result<u32> {
        self.reset_cache()?;
        self.prefill(full_context)
    }
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
