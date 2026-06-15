# el-grammar — pure-Rust DFA token masking

Grammar-constrained decoding via **DFA token masking**, in pure Rust (the
Grammar Constraint context). A grammar is compiled to a deterministic automaton
over **token ids** (the alphabet). Each decode step, the masker replays the
committed tokens to find the current state and allows only tokens with a valid
transition — the exact token-level masking mechanism that XGrammar/llguidance
implement, here for *regular* grammars with no external dependencies.

Depends on `el-core` and `el-runtime`. No `unsafe` (`#![forbid(unsafe_code)]`).

## Scope

This crate covers **regular grammars over an explicit token alphabet**. Full
context-free / JSON-schema grammars with a real tokenizer environment are the
production scale-up in
[`el-grammar-llguidance`](../adapters/el-grammar-llguidance). Both implement the
same `el_runtime::GrammarMasker` port, so the runtime is agnostic to which one
is wired.

## What it provides

- **`Dfa`** — a deterministic finite automaton over `Token` ids, with a builder
  API: `Dfa::new(start).transition(from, token, to).accept(state)`. Plus
  `step`, `run` (replay a token sequence), and `is_accepting`.
- **`DfaMasker`** — wraps a `Dfa` and implements `el_runtime::GrammarMasker`.
  `mask(recent, vocab)` returns a per-token allow mask; a dead (invalid) state
  yields an all-`false` mask (nothing legal). `accepts(committed)` reports
  whether stopping now would be valid.
- **`StateId`** — `u32` alias for automaton states.

## Usage

```rust
use el_grammar::{Dfa, DfaMasker};
use el_runtime::GrammarMasker;

// A grammar that accepts exactly the token sequence 5, 5, 9.
let dfa = Dfa::new(0)
    .transition(0, 5, 1)
    .transition(1, 5, 2)
    .transition(2, 9, 3)
    .accept(3);
let masker = DfaMasker::new(dfa);

// At the start, only token 5 is legal.
let mask = masker.mask(&[], 10);
assert!(mask[5] && !mask[9]);

// Wired into a session's Ports, this forces grammar-valid output even when the
// engine's raw logits would prefer something else.
assert!(masker.accepts(&[5, 5, 9]));
```

Drop it into a session via `Ports { grammar: Box::new(masker), ..Ports::permissive() }`.

## Status

Implemented and tested, including an end-to-end test that constrains real
decoding inside `el_runtime::InferenceSession`.

---

Part of the [Edge Intelligence](../../README.md) workspace; see the
[Grammar Constraint](../../docs/ddd/bounded-contexts/04-grammar-constraint.md)
context and [ADR-004](../../docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md).
