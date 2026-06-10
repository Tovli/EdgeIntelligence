//! `el-grammar-llguidance` — the real grammar masker over **llguidance** (the
//! Rust successor to XGrammar-style tag dispatch), implementing
//! [`el_runtime::GrammarMasker`] (ADR-004 grammar).
//!
//! SKELETON / FOLLOW-UP. Sketches the seam: register a JSON schema → maintain an
//! llguidance token parser → emit a per-step allow mask. llguidance's parser/FSM
//! internals never escape this crate.

#![forbid(unsafe_code)]

use el_core::Token;
use el_runtime::GrammarMasker;

/// Grammar masker backed by llguidance.
pub struct LlguidanceMasker {
    vocab: usize,
    // TODO(adr-004): hold an llguidance `TokenParser` built from the JSON schema
    // + the tokenizer's vocab, advancing it as tokens are committed.
}

impl LlguidanceMasker {
    /// Build from a developer-supplied JSON schema.
    ///
    /// TODO(adr-004): compile the schema into an llguidance grammar
    /// (`llguidance::api::TopLevelGrammar` / `GrammarBuilder`), instantiate a
    /// `TokenParser`, and cache compiled FSM fragments (cross-grammar cache).
    pub fn from_json_schema(_schema_json: &str, vocab: usize) -> Self {
        Self { vocab }
    }
}

impl GrammarMasker for LlguidanceMasker {
    fn mask(&self, _recent: &[Token], vocab: usize) -> Vec<bool> {
        // TODO(adr-004): return llguidance's `compute_mask()` bitset for the
        // current parser state. Until then, allow-all (no constraint) so the
        // pipeline is correct, just unconstrained.
        debug_assert_eq!(vocab, self.vocab, "vocab must match the compiled grammar");
        vec![true; vocab]
    }
}
