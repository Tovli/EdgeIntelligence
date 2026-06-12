//! `el-grammar` — pure-Rust **DFA token masking** for grammar-constrained
//! decoding (Grammar Constraint context, `docs/ddd/bounded-contexts/04`).
//!
//! A grammar is compiled to a deterministic automaton over **token ids** (the
//! alphabet). Each decode step, the masker replays the committed tokens to find
//! the current state and allows only tokens with a valid transition — the exact
//! token-level masking mechanism that XGrammar/llguidance implement, here for
//! regular grammars in pure Rust.
//!
//! Scope: this covers regular grammars over an explicit token alphabet. Full
//! context-free / JSON-schema grammars + a real tokenizer environment are the
//! `el-grammar-llguidance` production scale-up.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use el_core::Token;
use el_runtime::GrammarMasker;

pub type StateId = u32;

/// A deterministic finite automaton over token ids.
#[derive(Debug, Clone, Default)]
pub struct Dfa {
    start: StateId,
    transitions: BTreeMap<(StateId, Token), StateId>,
    accepting: BTreeSet<StateId>,
}

impl Dfa {
    pub fn new(start: StateId) -> Self {
        Self {
            start,
            transitions: BTreeMap::new(),
            accepting: BTreeSet::new(),
        }
    }

    /// Builder: add a transition `from --token--> to`.
    pub fn transition(mut self, from: StateId, token: Token, to: StateId) -> Self {
        self.transitions.insert((from, token), to);
        self
    }

    /// Builder: mark a state accepting.
    pub fn accept(mut self, state: StateId) -> Self {
        self.accepting.insert(state);
        self
    }

    /// The next state for `(state, token)`, if any.
    pub fn step(&self, state: StateId, token: Token) -> Option<StateId> {
        self.transitions.get(&(state, token)).copied()
    }

    /// Replay `tokens` from the start; `None` if any token is invalid (dead).
    pub fn run(&self, tokens: &[Token]) -> Option<StateId> {
        let mut s = self.start;
        for &t in tokens {
            s = self.step(s, t)?;
        }
        Some(s)
    }

    pub fn is_accepting(&self, state: StateId) -> bool {
        self.accepting.contains(&state)
    }
}

/// A [`GrammarMasker`] backed by a [`Dfa`].
pub struct DfaMasker {
    dfa: Dfa,
}

impl DfaMasker {
    pub fn new(dfa: Dfa) -> Self {
        Self { dfa }
    }

    /// Whether the committed sequence is in an accepting state (i.e. it would be
    /// valid to stop now).
    pub fn accepts(&self, committed: &[Token]) -> bool {
        self.dfa
            .run(committed)
            .is_some_and(|s| self.dfa.is_accepting(s))
    }
}

impl GrammarMasker for DfaMasker {
    fn mask(&self, recent: &[Token], vocab: usize) -> Vec<bool> {
        let mut mask = vec![false; vocab];
        // If the sequence so far is invalid (dead state), nothing is legal.
        if let Some(state) = self.dfa.run(recent) {
            for t in 0..vocab as Token {
                if self.dfa.step(state, t).is_some() {
                    mask[t as usize] = true;
                }
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grammar accepting exactly the token sequence `5, 5, 9`.
    fn grammar_5_5_9() -> Dfa {
        Dfa::new(0)
            .transition(0, 5, 1)
            .transition(1, 5, 2)
            .transition(2, 9, 3)
            .accept(3)
    }

    #[test]
    fn mask_allows_only_valid_next_tokens_per_state() {
        let m = DfaMasker::new(grammar_5_5_9());
        let only = |mask: Vec<bool>| -> Vec<usize> {
            mask.iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i)
                .collect()
        };
        assert_eq!(only(m.mask(&[], 10)), vec![5]);
        assert_eq!(only(m.mask(&[5], 10)), vec![5]);
        assert_eq!(only(m.mask(&[5, 5], 10)), vec![9]);
        assert_eq!(only(m.mask(&[5, 5, 9], 10)), Vec::<usize>::new()); // accepting, no continuation
        assert_eq!(only(m.mask(&[6], 10)), Vec::<usize>::new()); // dead state → nothing legal
    }

    #[test]
    fn accepts_reports_completion() {
        let m = DfaMasker::new(grammar_5_5_9());
        assert!(!m.accepts(&[5, 5]));
        assert!(m.accepts(&[5, 5, 9]));
    }

    #[test]
    fn grammar_constrains_real_decoding_in_the_runtime() {
        use el_core::{
            ModelFormat, ModelId, ModelVersion, SessionConfig, SessionId, StopReason, Token,
        };
        use el_provenance::{ModelArtifact, SignatureVerifier};
        use el_runtime::{InferenceEngine, InferenceSession, Ports};

        // Engine with uniform logits — left to itself it would emit token 0;
        // the grammar must override that and force 5, 5, 9.
        struct UniformEngine {
            vocab: usize,
        }
        impl InferenceEngine for UniformEngine {
            fn prefill(&mut self, t: &[Token]) -> el_core::Result<u32> {
                Ok(t.len() as u32)
            }
            fn next_logits(&mut self, _c: &[Token]) -> Vec<i32> {
                vec![0; self.vocab]
            }
            fn eos_token(&self) -> Token {
                99
            }
        }

        struct OkVerifier;
        impl SignatureVerifier for OkVerifier {
            fn verify(&self, _: &[u8], _: &[u8], _: u32) -> bool {
                true
            }
        }
        let mut art = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
        art.verify(&OkVerifier, b"w", b"s", 1);
        let permit = art.ensure_loadable().unwrap();

        let mut session = InferenceSession::new(
            SessionId(1),
            SessionConfig::default(),
            UniformEngine { vocab: 10 },
            permit,
        );
        let ports = Ports {
            grammar: Box::new(DfaMasker::new(grammar_5_5_9())),
            ..Ports::permissive()
        };
        session.load_prompt(&ports, &[1]).unwrap();
        let stop = session.generate(&ports, 3).unwrap();

        assert_eq!(stop, StopReason::MaxTokens);
        // Despite uniform logits, output is grammar-forced.
        assert_eq!(session.output(), &[5, 5, 9]);
    }
}
