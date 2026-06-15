# el-grammar-llguidance — JSON-schema grammar masking over llguidance

The production grammar masker (ADR-004): real **JSON-schema** token masking over
[llguidance](https://github.com/guidance-ai/llguidance), bridged to a HuggingFace
tokenizer. It is the scale-up of the pure-Rust, regular-grammar
[`el-grammar`](../../el-grammar) — both implement the same
`el_runtime::GrammarMasker` port, so the runtime is agnostic to which is wired.

The grammar FSM is built once from a JSON schema at construction time and
advanced per committed token during generation (~50 µs/mask for a 128k vocab).
The HuggingFace `tokenizers::Tokenizer` is bridged to llguidance's `TokEnv` via
`toktrie_hf_tokenizers` (the official guidance-ai integration crate). No
`unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

- **`LlguidanceMasker`** — implements `el_runtime::GrammarMasker`:
  - `from_tokenizer(&tokenizer, schema_json)` — build a masker from a HF
    tokenizer and a JSON-schema string.
  - `mask(recent, vocab)` — advances the FSM over newly committed tokens and
    returns the allowed-token mask. Once the grammar is satisfied/exhausted (or
    a token violates it and the parser cannot recover) it falls back to
    allow-all from there on.
  - `reset()` — rebuild a fresh parser for the same schema.

## Usage

```rust
use el_grammar_llguidance::LlguidanceMasker;
use el_runtime::GrammarMasker;

let tokenizer = tokenizers::Tokenizer::from_file("tokenizer.json").unwrap();
let masker = LlguidanceMasker::from_tokenizer(&tokenizer, r#"{"type":"integer"}"#)?;

// In the decode loop, wired into the session's Ports:
let allow_mask = masker.mask(&committed_tokens, vocab_size);
# Ok::<(), el_core::EdgeError>(())
```

## Building (workspace-excluded)

This crate depends on `llguidance` + `toktrie_hf_tokenizers` (which require
crates.io) and a native tokenizer build, so it is **excluded from the offline
workspace** and declares its own empty `[workspace]` table to build standalone:

```sh
cargo build  --manifest-path crates/adapters/el-grammar-llguidance/Cargo.toml
cargo test   --manifest-path crates/adapters/el-grammar-llguidance/Cargo.toml
```

### API-version note

The call sites target the llguidance / toktrie **1.7** line (llguidance 1.7 +
toktrie_hf_tokenizers 1.7 + tokenizers 0.21 — they must share one `toktrie`). If
a build fails with an API mismatch on `ByteTokenizer`, `ParserFactory`, or
`TokenParser`, the relevant call sites are marked with `// llg-api:` comments.

> Note: this crate uses the `onig` tokenizer backend (unlike `el-engine-candle`,
> which uses the pure-Rust `fancy-regex` backend per ADR-008).

## Status

Implemented and tested (integer-schema masking constrains digit tokens;
allow-all fallback on exhaustion). Lark/CFG grammars are tracked follow-up work.

---

Part of the [Edge Intelligence](../../../README.md) workspace (excluded build).
Realizes [ADR-004](../../../docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md);
see the [Grammar Constraint](../../../docs/ddd/bounded-contexts/04-grammar-constraint.md) context.
