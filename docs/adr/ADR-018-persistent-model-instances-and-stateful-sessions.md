# ADR-018: Persistent model instances and stateful inference sessions

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: runtime, performance, on-device, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.1,
roadmap Phase 1, EPIC-1) identifies model-and-session lifecycle as the single
highest-value foundation: weights should load **once**, and conversation state
(committed tokens + KV) should persist and be **reused across turns** instead of
re-prefilling the whole history.

What the codebase does today contradicts that on the real backend:

- `QwenChatProvider::chat` builds a **fresh `QwenEngine` every turn** and
  **reloads the GGUF from disk each call** (`QwenEngine::from_path(&self.model_path, …)`),
  then re-prefills the entire rendered conversation. The inline comment is
  explicit: *"Fresh engine + session each turn (candle KV cache has no public
  reset); the full conversation is re-prefilled."*
- `QwenEngine` holds candle's stateful KV cache (`index_pos`, `fed`) but
  `candle-transformers`' `quantized_qwen2` exposes no public cache reset/truncate,
  so a *new conversation* can only be served by constructing a new engine.
- `LocalLlmProvider` keeps one `InferenceSession` behind a `Mutex` but calls
  `session.reset()` at the start of every `chat`, clearing KV — so even there,
  prefix reuse never happens.
- `InferenceSession` (the ADR-001 aggregate root) already owns the session state
  machine (`Phase`), the `KvRegion`, the retained `prompt`, and a `reset()` that
  clears volatile memory — but it conflates *model lifetime* (the `engine`, moved
  in by value) with *conversation lifetime*.

Net effect: per-turn latency pays a full model load **and** a full prefill of the
growing history — exactly the orchestration cost the plan calls out first.

## Decision

Separate **model lifecycle** from **conversation lifecycle**, keeping
`el-runtime` as the session authority (the plan's explicit constraint — ruvLLM's
`RuvLLMEngine` is a reference, not an adopted dependency).

1. **Persistent model handle.** Introduce a loaded-model handle (e.g.
   `LoadedModel` / `StatefulLlmProvider`) that owns the weights and the
   `LoadPermit`, constructed once. Provider `chat`/`chat_stream` calls borrow it;
   weights are never re-read from disk per turn. (Mmap loading — ADR-021 — makes
   that one load cheap and shareable.)

2. **Multiple isolated sessions over one model.** A session is a conversation:
   its own `KvRegion`, committed output, `prompt`, `Phase`, and event buffer,
   bound to the shared immutable model. `SessionConfig` stays immutable for the
   session's life (ADR-001).

3. **Engine cache ownership.** Because the per-conversation KV cache must be
   resettable/clonable without reloading weights, the `InferenceEngine` seam gains
   an explicit conversation-reset (e.g. `reset_cache()` / a per-session cache
   handle) distinct from the ADR-012 `rollback`. The candle `quantized_qwen2`
   limitation (no public cache reset) is recorded under Consequences and is the
   reason this is an engine-seam change, not a pure runtime change.

4. **Prefix reuse across turns.** On a follow-up turn, prefill only the **new**
   suffix (new user message + assistant turn opener), reusing the KV for the
   unchanged prefix, rather than re-prefilling the whole transcript.

5. **Explicit lifecycle ops.** Every session exposes `reset`, `close`, and
   eviction; session memory is measurable and explicitly releasable (the telemetry
   to prove this comes from ADR-023).

## Consequences

### Positive
- Eliminates per-turn model reload and whole-history re-prefill — the largest
  measured hot-path cost named in the plan.
- Multi-turn conversations scale with *new* tokens per turn, not total transcript
  length.
- Clean separation lets one resident model back many lightweight sessions
  (foundation for the later policy/adapter work, P1+).

### Negative
- Requires an engine-seam change: candle's `quantized_qwen2` has no public
  KV-cache reset, so faithful per-conversation reuse needs either a forked/custom
  attention cache or an engine wrapper that manages cache lifetime — the same
  engine-internals constraint ADR-012 hit for `rollback`.
- Resident weights held for the model's lifetime raise steady-state memory vs.
  load-per-call (mitigated by mmap, ADR-021, and the ADR-003 budget).
- Prefix reuse interacts with the ADR-012 rollback/checkpoint invariants — a
  retained prefix must remain a guard-verified-safe prefix.

### Neutral
- `InferenceSession::reset()` and the `Phase` machine remain; this ADR re-homes
  the engine from "owned by the session" to "shared by sessions," which is
  additive to the existing lifecycle.
- The cloud `LlmProvider` (ADR-010) is unaffected — it is already stateless per
  call.

## Implementation status

Landed via a SPARC pass (scope: AC-1/AC-2/AC-4 + the engine seam; AC-3 deferred):

- **Engine seam** — added required `InferenceEngine::reset_cache(&mut self) ->
  Result<()>` (`el-runtime::ports`), distinct from `rollback`: it returns the
  engine to a pristine pre-prefill state so resident weights serve a new
  conversation. Implemented for every engine (`NullEngine`, `CandleEngine`
  stateless `Ok(())`; `QwenEngine` resets `index_pos`/`fed`/`prompt`/`last_logits`)
  and all test engines. No default — a forgotten override can't silently carry a
  stale cache.
- **Verified enabler** — candle `quantized_qwen2` attention discards its KV cache
  on a forward at `index_pos == 0`, so a re-`prefill` rebuilds it; the engine is
  reusable without reloading weights. The prior "candle has no public cache reset"
  caveat is satisfied this way.
- **Session lifecycle (conversation vs. model)** — `InferenceEngine::reset_cache`
  *actually releases the current conversation's KV while keeping weights loaded*:
  for `QwenEngine` it runs one position-0 forward over a benign token, which
  candle's attention uses to overwrite (and thus drop) the prior user K/V tensors
  — freeing that memory and clearing user data — without touching the weights.
  `reset()` (reuse) and `close(&mut self)` (end-of-conversation: also frees buffer
  capacity + discards events) both build on it; both keep the model resident and
  the session reusable, and both propagate a `reset_cache` failure (state untouched
  on error). `reset()` preserves undrained events (generic semantics). `close`
  takes `&mut self`, **not** `self`: consuming would also drop the expensive
  weights — the opposite of "load once, reuse." To free the weights too, drop the
  session/provider (ownership). `kv_len()` exposes the measurable footprint (AC-4).
- **Resident model + explicit release (AC-1/AC-2/AC-4)** — `QwenChatProvider` loads
  the `QwenEngine` **once** in `from_paths` and holds it in a `Mutex<ChatSession>`
  (`Loaded` → `Active`, promoted lazily on the first `chat` so builder config is
  final), reusing one session per turn via `reset()` + `load_prompt()` +
  `generate()`. The per-turn `QwenEngine::from_path` disk reload is gone.
  `QwenChatProvider::end_session()` exposes the explicit conversation release
  (releases KV/output/prompt/events, keeps the model resident). Turn-level event
  isolation lives in the providers (drain/discard at turn start), not in generic
  `reset()`.
- **Tests** — runtime tests prove `reset()` resets the engine and preserves
  undrained events; one engine serves multiple conversations without
  reconstruction; `close()` releases the conversation, keeps the engine resident
  (Drop-counter shows zero drops), and leaves the session reusable; and a fallible
  `reset_cache` is propagated, not swallowed. Full workspace suite green;
  `cargo fmt --all -- --check` and `clippy -D warnings` clean.

Still **deferred** (own increment, per the spec's scope decision):

- **AC-3 — cross-turn prefix reuse / incremental prefill.** Each turn still
  re-prefills the whole conversation (no weight reload). Reusing the unchanged
  prefix's KV needs an engine "extend-without-reset" contract, a session "continue"
  transition, and care around the ADR-012 safe-prefix invariant and tokenizer
  round-trip stability. The persistent-session architecture here is its foundation.
- **Expert persistence.** With `--expert-model`, the safety expert GGUF is still
  loaded per turn; persisting it like the base model is a follow-up. The default
  (no-expert) path is fully resident.
- **Concurrent multi-session weight sharing.** candle couples weights + KV cache,
  so this realizes *serial* conversation reuse; true N-sessions-over-one-model
  needs an engine that separates weights from cache (same root constraint as
  ADR-022).

### Review fixes (post-implementation hardening)

**Round 1.** (1) `reset()`/`close()` silently ignored `reset_cache()` errors → they
now return `Result<()>` and propagate, leaving session state untouched on failure
so an empty session can't desync from a stale engine cache; (2) stray empty root
artifacts from a botched shell redirect were removed.

**Round 2** (correcting round 1's overreach + remaining gaps). (1) **KV not freed
on session end (PRD line 131).** Documenting candle's limitation was insufficient —
the PRD requires user K/V cleared on session end via ownership. `close` now takes
`self` **by value**, so ending a session drops the engine and frees its KV
tensors; round 1's `&mut self` reset could not. Proven by a Drop-counting engine
test. (2) **`reset()` overreached by clearing events.** Round 1 fixed the
provider's event leak by clearing events in generic `reset()`, which silently drops
a consumer's undrained telemetry. Reverted — `reset()` preserves events; turn-level
isolation now lives in the providers (drain/discard at turn start). (3) **Release
gate.** `cargo fmt --all -- --check` now passes (it was failing in `el-runtime` /
`el-engine-candle`). (4) A further stray root artifact (`0)`) was removed.

**Round 3** (correcting round 2's over-correction). Making `close(self)` consume
the session freed the KV but **also dropped the resident weights** — defeating the
ADR's separation of conversation lifecycle from model lifecycle (a caller could not
release conversation memory while keeping the model loaded), and `QwenChatProvider`
exposed no release at all. Fixes: (1) `reset_cache` now genuinely frees the
conversation's KV in place (the position-0 benign-forward overwrite for candle), so
KV is released **without** dropping weights; (2) `close` is back to `&mut self` —
releases the conversation, keeps the model resident, session reusable; (3)
`QwenChatProvider::end_session()` exposes that release to provider callers; (4)
further stray artifacts (`0`, `1`) from shell-redirect mishaps removed, and that
redirect pattern abandoned. Regression test:
`close_releases_conversation_but_keeps_engine_resident` (asserts KV freed, engine
**not** dropped, session reusable). Full gate green: tests,
`cargo fmt --all -- --check`, `clippy -D warnings`.

**Round 4.** (1) **Partial-eviction failure could later report success.**
`reset_cache` keyed its skip on `index_pos`, which it zeroed *before* the fallible
forward — so a forward that failed after candle replaced some layers left
`index_pos == 0`, and the next `reset_cache` skipped clearing while user K/V was
still resident. Now `QwenEngine` tracks a `cache_dirty` flag set by every
`forward_one` (before the fallible call) and cleared **only after** a fully
successful eviction forward; a failed eviction stays dirty and is retried.
Regression test `reset_retries_clearing_after_a_failed_attempt`. (2) **App reset
paths didn't release.** `el-chat`'s `/reset`, `/system`, and generation-error
recovery now call `QwenChatProvider::end_session()` (via a `release_session`
helper) instead of only discarding chat history, so a discarded conversation
doesn't linger in the engine until the next turn. (3) **Buffers retained
allocations.** `reset_cache` used `Vec::clear()` (keeps capacity + stale token
bytes); it now assigns `Vec::new()` to **release** the prompt and logits
allocations. (Guaranteed zeroize-before-free would need the `zeroize` crate —
out of scope; the owned allocation is dropped.)

**Round 5.** (1) **CLI cleanup failed open.** `release_session` logged an
`end_session` failure but the caller still cleared history and reported a
successful reset, so user KV could remain resident silently. It now **retries
once** and returns `false` on failure; `/reset`, `/system`, and error recovery
release *before* reporting success and, on unrecoverable failure, **stop** (break
the loop → `main` returns → the provider drops → KV freed via ownership) instead of
failing open. (2) **Stale public docs.** The adapter `README` and the `QwenEngine`
doc still described the old "fresh engine per `chat` (Candle can't reset its
cache)" behavior; both now describe the ADR-018 resident model + in-place
`reset_cache` eviction, and the `EL_BENCH` section reflects the once-at-startup
load ("session setup", not a per-chat "model load").

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.1, EPIC-1.
- Builds on: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) (session immutability), [ADR-010](./ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md) (`LlmProvider` seam), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (engine).
- Constrained by: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (memory budget for resident weights), [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (retained prefix must stay a safe prefix; cache-reset is distinct from `rollback`).
- Enables: [ADR-021](./ADR-021-memory-mapped-verified-gguf-loading.md) (cheap shared load), [ADR-019](./ADR-019-in-loop-incremental-decoding-and-token-streaming.md) (streaming over a persistent session), [ADR-022](./ADR-022-two-tier-quantized-kv-cache-with-attention-aware-eviction.md) (per-session KV budgets).
- Implementation seams: `crates/el-runtime` (`InferenceSession`, `ports::InferenceEngine`), `crates/adapters/el-engine-candle` (`QwenEngine`, `QwenChatProvider`, `LocalLlmProvider`).
- Measured by: [ADR-023](./ADR-023-baseline-performance-instrumentation.md) (warm-turn latency, prefill-tokens-saved).
