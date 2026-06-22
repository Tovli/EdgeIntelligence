# ADR-019: In-loop incremental decoding and real token streaming

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: runtime, performance, bindings, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.2,
roadmap Phase 1, EPIC-2) calls for **true in-loop streaming**: the first token
should be emitted after model load + prefill + **one** decode step, not after the
whole response is finished.

The `LlmProvider` trait (ADR-010) already advertises a streaming entry point —
`chat_stream(req, on_token: &mut dyn FnMut(ChatToken))` — and `ChatToken { text,
is_final }` exists. But the implementations only *simulate* streaming:

- `QwenChatProvider::chat_stream` and `LocalLlmProvider::chat_stream` both call
  `self.chat(req)?` to completion, then re-emit the finished string character by
  character. The comment is explicit: *"The runtime decode loop runs to completion
  internally (no per-token hook), so … we stream the finished reply out character
  by character."*
- `InferenceSession::generate_with_policy` is the decode loop, but it commits
  tokens into `self.output` and returns only a `StopReason` — there is **no
  per-token emit hook**.
- Consequently time-to-first-token equals total generation time, and
  `TelemetrySnapshot.ttft_ms` is never populated with a real measurement.

A real-streaming design must respect the ADR-012 control loop: committed tokens
can be **rolled back** by the chunk guard. Emitting a token the instant it is
sampled would risk surfacing an unsafe span that the guard later rewinds — a
fail-*open* leak. So "stream each token immediately" is not safe as-is; emission
must be gated on the safety loop.

## Decision

Thread a per-token emit callback through the decode loop and drive real streaming
from it, with emission gated by the safety control loop.

1. **Emit hook in the loop.** Add an optional sink to
   `generate_with_policy` (e.g. `on_token: Option<&mut dyn FnMut(Token)>`, or an
   `emit` closure on a small context) called when a token becomes **eligible to
   surface** — see §3. The invariant decode order is unchanged: grammar mask →
   safety adjust → sample → commit, then *consider emit*.

2. **Provider streams from the hook.** `QwenChatProvider`/`LocalLlmProvider`
   `chat_stream` detokenize emitted token ids incrementally and call `on_token`,
   ending with `ChatToken { is_final: true }`. The character-replay shim is
   removed.

3. **Safety-gated emission (the key constraint).** A token is surfaced only once
   it is **guard-verified safe** — i.e. at the next checkpoint/guard-pass boundary
   (ADR-012), not at the instant of commit. Tokens still inside the in-flight,
   not-yet-scored window are buffered; a rollback discards the buffer instead of
   un-saying already-streamed text. This trades a little TTFT granularity (first
   emit lands at the first guard-verified boundary, not literally token 1) for
   never streaming a span the guard will rewind. With `SafetyMode::Off` or no
   guard wired, tokens surface immediately after commit.

4. **Cancellation.** The sink can request stop; the loop must halt promptly,
   release temporary buffers, and leave the session in a consistent `Phase`.

5. **Binding parity.** Because the seam is a plain `FnMut` callback (no async
   runtime in `el-core`), each binding wraps it natively: FRB → Dart `Stream`,
   uniffi → async callback, wasm-bindgen → `ReadableStream`. The `el-ffi` adapter
   carries this through to all surfaces.

## Consequences

### Positive
- TTFT drops from "whole response" to "first guard-verified boundary," the
  headline UX win the plan targets, and becomes a *measurable* number (ADR-023).
- One streaming path through the real decode loop replaces the per-provider
  character-replay shims.
- Streaming composes with persistent sessions (ADR-018): tokens stream as the
  reused-prefix continuation is generated.

### Negative
- Emission gated on guard boundaries means first-token latency is bounded by
  `guard_every` under an active guard — slightly coarser than per-token; documented
  as the safety/latency trade-off.
- The loop grows an emit/cancel path and a small surface buffer; more state than a
  fire-and-forget loop.
- Detokenization becomes incremental (partial multi-byte / multi-token glyphs must
  not be split mid-emit) — the tokenizer-owning adapter handles boundary buffering.

### Neutral
- `LlmProvider::chat` (non-streaming) is unchanged; it can be expressed as
  `chat_stream` drained to a string.
- The ADR-007 content-free telemetry guarantee is untouched: streamed text goes to
  the host callback, never onto a `DomainEvent`.

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.2, EPIC-2.
- Builds on: [ADR-010](./ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md) (`chat_stream`/`ChatToken` seam), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md) (persistent session to stream over), [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md) (Dart stream binding).
- Constrained by: [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (emit only guard-verified tokens — no streaming a rolled-back span), [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md) (decode order invariant).
- Implementation seams: `crates/el-runtime` (`generate_with_policy` emit hook), `crates/adapters/el-engine-candle` (`chat_stream` incremental detokenize), `crates/adapters/el-ffi` (per-binding stream wrappers).
- Measured by: [ADR-023](./ADR-023-baseline-performance-instrumentation.md) (real `ttft_ms`).
