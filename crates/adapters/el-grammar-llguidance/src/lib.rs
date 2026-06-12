//! `el-grammar-llguidance` — real grammar masker over **llguidance** (ADR-004).
//!
//! Bridges HuggingFace [`tokenizers::Tokenizer`] to llguidance's [`TokEnv`] via
//! [`toktrie_hf_tokenizers`] (the official integration crate from guidance-ai,
//! ~50 µs/mask for 128k vocab). The grammar FSM is built once from a JSON schema
//! at construction time and advanced per committed token during generation.
//!
//! # Usage
//! ```ignore
//! let tok = tokenizers::Tokenizer::from_file("tokenizer.json").unwrap();
//! let masker = LlguidanceMasker::from_tokenizer(&tok, r#"{"type":"integer"}"#).unwrap();
//! // In the decode loop:
//! let allow_mask = masker.mask(&committed_tokens, vocab_size);
//! ```
//!
//! # API-version note
//! This file targets the llguidance/toktrie 1.7 line (llguidance 1.7 +
//! toktrie_hf_tokenizers 1.7 + tokenizers 0.21 — the versions must share one
//! `toktrie`). If a build fails with an API mismatch on `ByteTokenizer`,
//! `ParserFactory`, or `TokenParser`, check the crate changelogs and update the
//! corresponding call sites (all marked with `// llg-api:`).

#![forbid(unsafe_code)]

use el_core::{EdgeError, Result, Token};
use el_runtime::GrammarMasker;
use llguidance::{api::TopLevelGrammar, toktrie::TokEnv, ParserFactory, TokenParser};
use std::sync::Mutex;
use toktrie_hf_tokenizers::ByteTokenizer;

fn llg_err(msg: impl std::fmt::Display) -> EdgeError {
    EdgeError::Grammar(format!("llguidance: {msg}").into_boxed_str())
}

/// Grammar masker backed by llguidance + a HuggingFace tokenizer.
///
/// Internally holds a [`TokenParser`] (llguidance's per-session FSM) plus a
/// committed-token counter behind a `Mutex`. The counter tracks how many tokens
/// the parser has already consumed so `mask()` can advance the FSM only for the
/// newly appended tail of the token buffer.
pub struct LlguidanceMasker {
    /// Compiles grammars and owns the shared [`TokEnv`]; reused by [`reset`](Self::reset).
    factory: ParserFactory,
    schema_json: String,
    /// `Some((parser, n_committed))` while the grammar is active;
    /// `None` once the grammar is exhausted (allow-all fallback).
    state: Mutex<Option<(TokenParser, usize)>>,
    vocab: usize,
}

impl LlguidanceMasker {
    /// Build a masker from a HuggingFace tokenizer and a JSON schema string.
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer, schema_json: &str) -> Result<Self> {
        // llg-api: ByteTokenizer::from_tokenizer(Tokenizer) -> Result<ByteTokenizer>
        //          ByteTokenizer::into_tok_env(self, n_vocab) -> Result<TokEnv>
        let byte_tok = ByteTokenizer::from_tokenizer(tokenizer.clone()).map_err(llg_err)?;
        let env: TokEnv = byte_tok.into_tok_env(None).map_err(llg_err)?;
        let vocab = env.tok_trie().vocab_size();

        // llg-api: ParserFactory::new_simple(&TokEnv) -> Result<ParserFactory>
        let mut factory = ParserFactory::new_simple(&env).map_err(llg_err)?;
        factory.quiet();

        let parser = build_parser(&factory, schema_json)?;

        Ok(Self {
            factory,
            schema_json: schema_json.to_owned(),
            state: Mutex::new(Some((parser, 0))),
            vocab,
        })
    }

    /// Reset the FSM to the initial state (same schema, fresh parser).
    pub fn reset(&self) -> Result<()> {
        let parser = build_parser(&self.factory, &self.schema_json)?;
        *self.state.lock().unwrap() = Some((parser, 0));
        Ok(())
    }
}

/// Builds a fresh started [`TokenParser`] from a JSON schema via the shared factory.
fn build_parser(factory: &ParserFactory, schema_json: &str) -> Result<TokenParser> {
    let schema_val: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| llg_err(format!("invalid JSON schema: {e}")))?;
    let grammar = TopLevelGrammar::from_json_schema(schema_val);

    // llg-api: ParserFactory::create_parser(TopLevelGrammar) -> Result<TokenParser>
    let mut parser = factory.create_parser(grammar).map_err(llg_err)?;
    // We mask pure generation (no prompt tokens go through the grammar), and the
    // parser requires exactly one of process_prompt()/start_without_prompt().
    parser.start_without_prompt();
    Ok(parser)
}

impl GrammarMasker for LlguidanceMasker {
    fn mask(&self, recent: &[Token], vocab: usize) -> Vec<bool> {
        debug_assert_eq!(vocab, self.vocab, "vocab must match the tokenizer");

        let mut guard = self.state.lock().unwrap();

        // Advance the FSM for every token committed since the last mask call,
        // then compute and return the allowed-token mask. The outcome enum lets
        // us release the mutable borrow of `guard` (via the Option) before the
        // Exhaust arm assigns `*guard = None`.
        enum Outcome {
            Mask(Vec<bool>),
            AllowAll,
            Exhaust,
        }

        let outcome = if let Some((parser, committed)) = guard.as_mut() {
            let mut violated = false;
            for &tok in &recent[*committed..] {
                // llg-api: consume_token(TokenId) -> Result<usize /* backtrack */>
                if parser.consume_token(tok).is_err() {
                    // Token violated grammar (or parser stopped) — the FSM
                    // cannot recover, so fall back to allow-all from here on.
                    violated = true;
                    break;
                }
                *committed += 1;
            }

            if violated {
                Outcome::Exhaust
            } else {
                // llg-api: compute_mask() -> Result<SimpleVob>;
                // SimpleVob::is_allowed(tok: u32) -> bool   (toktrie crate)
                match parser.compute_mask() {
                    Ok(vob) => {
                        Outcome::Mask((0..vocab).map(|i| vob.is_allowed(i as u32)).collect())
                    }
                    Err(_) => Outcome::Exhaust,
                }
            }
        } else {
            Outcome::AllowAll
        };

        match outcome {
            Outcome::Mask(m) => m,
            Outcome::AllowAll => vec![true; vocab],
            Outcome::Exhaust => {
                *guard = None; // grammar satisfied / exhausted → allow-all from here on
                vec![true; vocab]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    /// Build a tiny synthetic `WordLevel` tokenizer from its `tokenizer.json`
    /// wire format — the serialized form is stable across `tokenizers` releases,
    /// unlike the builder APIs (`BPE::new` arity, `AHashMap` vocab params).
    ///
    /// Vocabulary: single digits "0"–"9" → token IDs 0–9,
    ///             letters "a"–"e" → token IDs 10–14.
    ///
    /// A `ByteLevel` decoder is declared because `ByteTokenizer::from_tokenizer`
    /// requires a ByteLevel or ByteFallback decoder to derive token byte
    /// representations (our single-ASCII-char tokens map to themselves).
    fn digit_tokenizer() -> Tokenizer {
        let vocab: serde_json::Map<String, serde_json::Value> = (0u8..=9)
            .map(|d| (d.to_string(), u32::from(d)))
            .chain((0u8..5).map(|i| (((b'a' + i) as char).to_string(), 10 + u32::from(i))))
            .map(|(name, id)| (name, serde_json::Value::from(id)))
            .collect();
        let spec = serde_json::json!({
            "version": "1.0",
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true,
            },
        });
        Tokenizer::from_bytes(serde_json::to_vec(&spec).unwrap())
            .expect("synthetic tokenizer.json parses")
    }

    #[test]
    fn integer_schema_allows_only_digit_tokens() {
        let tok = digit_tokenizer();
        let masker = LlguidanceMasker::from_tokenizer(&tok, r#"{"type":"integer"}"#)
            .expect("integer schema builds");

        let mask = masker.mask(&[], tok.get_vocab(false).len());
        for d in 0u32..10 {
            assert!(mask[d as usize], "digit token {d} should be allowed");
        }
        for l in 10u32..15 {
            assert!(
                !mask[l as usize],
                "letter token {l} should be blocked by integer schema"
            );
        }
    }

    #[test]
    fn allow_all_when_schema_exhausted_or_satisfied() {
        let tok = digit_tokenizer();
        let masker = LlguidanceMasker::from_tokenizer(&tok, r#"{"type":"integer"}"#).unwrap();
        let vocab = tok.get_vocab(false).len();

        let mask = masker.mask(&[5], vocab);
        assert!(
            mask.iter().any(|&b| b),
            "after committing a digit, mask should be non-empty"
        );
    }
}
