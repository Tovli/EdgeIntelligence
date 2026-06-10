//! Port traits the Inference Runtime depends on (the collaborator contexts).

use el_core::{Result, Token};
use el_safety::SafetySteerer;

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
    pub relay: Option<Box<dyn HybridRelay>>,
}

impl Ports {
    /// Defaults: identity compression, all-tokens-allowed grammar, no safety
    /// steering, no relay (air-gapped).
    pub fn permissive() -> Self {
        Self {
            compressor: Box::new(super::defaults::IdentityCompressor),
            grammar: Box::new(super::defaults::AllowAllMasker),
            safety: Box::new(el_safety::NoSafety),
            relay: None,
        }
    }
}
