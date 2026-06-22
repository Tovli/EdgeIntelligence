# ADR-020: Batched (single-pass) prefill

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: runtime, performance, on-device, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) §P1.8,
pulled into roadmap Phase 1 / EPIC-2) identifies token-by-token prefill as a
removable orchestration cost: a prompt of length `N` should be processed as a
single `(1, N)` forward, not `N` separate forwards.

The current engine does exactly the slow thing:

- `QwenEngine::prefill` loops `self.forward_one(t)` once **per prompt token**,
  each call running a `(1, 1)` candle forward and advancing `index_pos` by one.
- `candle-transformers`' `quantized_qwen2::ModelWeights::forward(&input, index_pos)`
  already accepts a multi-token `input` tensor, so the batched shape is available
  at the seam — only the adapter drives it one token at a time.
- `CandleEngine::prefill` (the linear engine-seam proof) returns `tokens.len()`
  without a real forward, so the cost lives entirely in the real `QwenEngine`.
- The `InferenceEngine` port defines `prefill(&mut self, tokens) -> Result<u32>`
  with no batched variant; `next_logits` is the separate decode step.

Prefill dominates first-response latency for long prompts (system prompt + RAG
context + history), so this is a direct lever on TTFT and on the per-turn cost
that ADR-018 reduces by reuse.

## Decision

Process the prompt in a single batched forward, keeping prefill and decode
distinct, and preserving exact equivalence with sequential prefill.

1. **Batched prefill at the seam.** Either add `InferenceEngine::prefill_batch`
   or make `prefill` itself issue one `(1, N)` forward over the whole (compressed)
   prompt. `decode_step`/`next_logits` stays a separate single-token path.

2. **Equivalence requirement.** Batched prefill must leave the engine in the
   **same final KV state** and yield the **same** post-prefill logits as the
   current sequential loop (a regression test asserts batched == sequential on the
   toy and Qwen engines). Determinism (ADR-002) is preserved.

3. **Chunked prefill for memory-bound devices.** Devices that cannot fit a full
   `(1, N)` activation batch process the prompt in chunks of size drawn from the
   ADR-003 memory plan, accumulating KV across chunks to the identical end state.
   Chunk size is a planned quantity, not a magic constant.

4. **Honest prefill telemetry.** `load_prompt` currently emits
   `PrefillCompleted { prefill_tps: 0 }` — a placeholder. With batched prefill,
   emit a **measured** `prefill_tps` (wired through ADR-023), so the speedup is
   observable in CI rather than asserted.

5. **No safety/grammar change.** Prefill does not sample tokens, so the ADR-005
   decode-order invariant and the ADR-012/013 guard/steer/ingress logic are
   untouched; ingress triage still scores the prompt before generation.

## Consequences

### Positive
- Prompt-processing latency falls from `N` forwards to one batched forward (plus
  chunking only where memory forces it) — the plan's named win.
- Compounds with ADR-018: a persistent session re-prefills only the new suffix,
  and that suffix is itself batched.
- Turns the `prefill_tps: 0` placeholder into a real, CI-gated metric.

### Negative
- A `(1, N)` forward needs a larger transient activation buffer than `(1, 1)`;
  on tight devices this forces chunking, partially eroding the gain — bounded by
  the ADR-003 plan.
- `prefill_batch` widens the `InferenceEngine` contract; every engine adapter must
  provide it (a sequential fallback satisfies the contract for engines that can't
  batch).

### Neutral
- The candle path already supports multi-token `forward`, so this is an adapter
  change, not an upstream dependency change.
- Speculative decoding (P1, §9; events `DraftProposed`/`DraftVerified` already
  exist) reuses the same prefill/decode split but is out of scope for this
  milestone.

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) §P1.8 (Phase 1 / EPIC-2).
- Builds on: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (candle multi-token forward), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md) (suffix-only prefill on reuse).
- Constrained by: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (chunk size from the memory plan), [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md)/[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (decode/guard semantics unchanged).
- Implementation seams: `crates/el-runtime` (`ports::InferenceEngine::prefill`/`prefill_batch`, `InferenceSession::load_prompt`), `crates/adapters/el-engine-candle` (`QwenEngine::prefill`).
- Measured by: [ADR-023](./ADR-023-baseline-performance-instrumentation.md) (measured `prefill_tps`, prompt latency).
