# Bounded Context 4 — Grammar Constraint  (Supporting)

> Guarantees structured output (JSON / tool-calls) by masking illegal tokens
> each step, using **llguidance** (Rust) for tag dispatch + FSM caching.
> Source: PRD §"Grammar-Constrained Decoding (XGrammar-2 → llguidance)",
> §Decoding Loop → Grammar Masking.

## Purpose

Let developers supply schemas (e.g. tool-call JSON) and have the decoder emit
only schema-valid tokens, at near-zero per-step overhead.

## Strategic role

**Supporting.** Essential for agentic apps but built on the established
**llguidance** engine (the Rust successor to XGrammar-style tag dispatch). Off
when no schema is registered.

## Ubiquitous language (context-local)

`Grammar Ruleset`, `Tag`, `TagDispatch`, `Token Mask`, `FSM State`, `Cross-Grammar
Cache`, `Partial JIT`.

## Aggregates

### `GrammarSession` (Aggregate Root)
The grammar state bound to one `InferenceSession`.

- **Identity:** mirrors the owning `SessionId`.
- **Holds:** the registered `GrammarRuleset`, the active grammar stack (a stack
  of `GrammarContext`s), the `FSM` cursor, and the cross-grammar cache handle.
- **Invariants:**
  - The emitted `TokenMask` always reflects the **current top** of the grammar
    stack; sampling outside the mask is forbidden.
  - A `Tag` match pushes a sub-grammar; the matching close pops it (balanced
    stack — never popped below the root grammar).
  - Mask generation is O(1) per step via cached FSM fragments; large array
    bounds use compressed/repetition states (no linear blow-up).

### `GrammarRuleset` (Entity)
- **Holds:** the set of registered grammars (root JSON grammar + tag→sub-grammar
  map) and their compiled `FSM State`s.
- **Invariant:** frequent grammars are precompiled at startup; the rest JIT on
  first encounter (Partial JIT budget of K largest states).

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `Tag` | registered marker string | detected via Aho–Corasick |
| `TokenMask` | boolean vector over vocabulary | the per-step output |
| `JsonSchema` | developer-supplied schema | source of a grammar |
| `GrammarId` | opaque id | |
| `FsmStateId` | compiled-state id | may be JIT-built lazily |
| `BacktrackWindow` | ≤256 chars | how far TagDispatch rescans |

## Domain Services

- **`TagDispatcher`** — Aho–Corasick scan over the recent output
  (`BacktrackWindow`) detecting registered `Tag`s and switching grammar context.
- **`MaskGenerator`** — produces the `TokenMask` for the active FSM state,
  consulting the cross-grammar cache.
- **`GrammarCompiler`** — startup precompilation + Partial JIT on first
  encounter; populates the cache.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `GrammarEngine` | llguidance (Rust) via ACL | inbound |
| consumed by `GrammarMasker` | Inference Runtime (1) | outbound (C/S) |

## Anti-Corruption Layer

`LlguidanceAcl` wraps the llguidance engine (tag-dispatch automaton, FSM cache,
repetition-compression). It exposes only `register(JsonSchema)`,
`mask_for(state)`, and `advance(token)` in domain terms; llguidance's internal
parser/FSM structures never leak.

## Domain Events (published)

`GrammarRegistered`, `GrammarCompiled`, `GrammarSwitched` (tag matched, sub-grammar
pushed), `TokenMaskApplied`, `GrammarViolationBlocked`. Structural metadata only.
See [domain-events.md](../domain-events.md).

## Relationships

Customer/Supplier with Inference Runtime (1): each decode step the orchestrator
requests a `TokenMask` to apply **before** sampling and **before** the safety
adjustment is combined. Independent of Safety (5) and Speculative Decoding (3).
