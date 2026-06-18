//! Default, pure-Rust port implementations (no-op / identity) so a session can
//! be built without any external adapter.

use crate::ports::{GrammarMasker, InferenceEngine, PromptCompressor};
use el_core::{Result, Token};

/// Identity compressor — passes the prompt through unchanged.
pub struct IdentityCompressor;

impl PromptCompressor for IdentityCompressor {
    fn compress(&self, tokens: &[Token]) -> Vec<Token> {
        tokens.to_vec()
    }
}

/// Allow-all grammar — every token legal (used when no schema is registered).
pub struct AllowAllMasker;

impl GrammarMasker for AllowAllMasker {
    fn mask(&self, _recent: &[Token], vocab: usize) -> Vec<bool> {
        vec![true; vocab]
    }
}

/// A trivial engine that emits EOS immediately after prefill. Lets you exercise
/// the full session lifecycle without the Candle adapter (ADR-002 is the real
/// engine). Not for production inference.
pub struct NullEngine {
    pub eos: Token,
    pub vocab: usize,
}

impl NullEngine {
    pub fn new(eos: Token, vocab: usize) -> Self {
        Self { eos, vocab }
    }
}

impl InferenceEngine for NullEngine {
    fn prefill(&mut self, tokens: &[Token]) -> Result<u32> {
        Ok(tokens.len() as u32)
    }

    fn next_logits(&mut self, _committed: &[Token]) -> Vec<i32> {
        let mut v = vec![0i32; self.vocab];
        if let Some(slot) = v.get_mut(self.eos as usize) {
            *slot = 1;
        }
        v
    }

    fn eos_token(&self) -> Token {
        self.eos
    }

    /// Stateless: `next_logits` ignores `committed`, so there is nothing to undo.
    fn rollback(&mut self, _keep_committed: u32) -> Result<()> {
        Ok(())
    }
}
