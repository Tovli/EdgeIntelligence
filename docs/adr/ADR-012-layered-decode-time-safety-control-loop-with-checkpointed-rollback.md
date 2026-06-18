# ADR-012: Layered decode-time safety control loop with checkpointed rollback

- **Status**: accepted
- **Date**: 2026-06-16
- **Deciders**:
- **Tags**: safety, security, on-device, runtime, supporting

## Context

[ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md) decided *which*
safety mode runs (`Off | Lightweight | SecDecoding | CSD`) and *where* the
`LogitAdjustment` sits in a decode step (after the grammar mask, before
sampling). It did not decide *how* steering recovers once generation has already
drifted into an unsafe trajectory. Treating safety as a one-shot per-token gate
is the weak point: once an autoregressive model commits to a bad prefix,
post-hoc repair is expensive and often too late.

The `SecDecoding alternatives` research
([docs/research/SecDecoding alternatives research.md](../research/SecDecoding%20alternatives%20research.md))
surveys the field and concludes that the strongest practical edge architecture is
*not* any single technique but a **hybrid control loop**: selective
model-backed steering (SafeDecoding-style, concentrated on the early-token
window) wrapped in **checkpointed rollback to the last safe prefix**, with a
chunk-level guard and deterministic ingress rules, degrading gracefully across
device tiers. Runtime backtracking ([RESET]-token, RoCode, Hydra) is identified
as the missing piece that turns steering into a recoverable loop. The research
also warns that a single safety judge is itself an attack surface, so the monitor
must be heterogeneous.

## Decision

Wrap the ADR-005 steering in a **recoverable decode-time control loop** with four
on-device, air-gapped stages (no stage may reach the network, per
[ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)):

1. **Selective steering window.** Apply model-backed `LogitAdjustment` on prompt
   ingress and the **first 8–32 output tokens**, then fall back to normal decode
   unless the guard re-escalates. This captures most early-trajectory risk at the
   lowest sidecar cost.
2. **Checkpoint the KV-cache, not just text.** Snapshot generation state
   (emitted token IDs, sampler RNG state, safety-control state, and KV-cache
   handle / block references) every **K = 8–16 tokens** and at semantic
   boundaries (newline, sentence break, closing brace, tool-call delimiter).
   Rollback **restores cache state** rather than replaying text *where the engine
   supports in-place cache restoration*; the realized per-engine cost (some
   engines must replay) is recorded under Consequences.
3. **Chunk-guard cadence.** Score recent output every **4–16 tokens** (not every
   token); where hardware allows, run the guard **asynchronously** with
   generation (Hydra-style) so valid output pays near-zero cost.
4. **Bounded rollback, fail-closed.** On a hard-threshold breach, restore the last
   safe checkpoint, raise `alpha` / lower temperature / inject refusal bias, and
   retry. After `max_rollbacks` (or with no checkpoint), emit a **deterministic
   hard refusal**. Rollback loops are always bounded.

The monitor stays **heterogeneous**: the model-backed signal is one layer
alongside deterministic ingress controls and CSD/grammar constraints on any
machine-consumable channel — never a single judge. Cadence, checkpoint density,
and steering width are **tier-aware**, selected from the `DeviceProfile` and
subject to the [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
memory budget. Safety adapters, guard weights, and policy files are integrity-gated
on load per [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md).

### Model inventory

One generator is mandatory; the safety signal's form follows the active
`SafetyMode`. The CSD layer is a grammar engine, not a model.

| Mode | On-device model artifacts |
|---|---|
| `Off` | Main generator only |
| `Lightweight` | Main **+ one of**: same-family safety **LoRA adapter** (shares base weights), a ~0.1B refusal classifier, or token-anchor heuristics (no weights) |
| `SecDecoding` | Main **+** small **base** **+** safety-tuned **expert** (contrastive pair) — accelerator-class HW only |
| `CSD` | Main **+** grammar/completion engine (**llguidance** — code, not weights) |

Defaults and invariants:
- **Recommended baseline:** a quantized `Qwen2.5-0.5B-Instruct` / `Qwen3-0.6B`
  generator (mixed-INT4 on GPU/NPU, INT8 on CPU) **plus one safety LoRA adapter**
  — two artifacts cover the `Lightweight` default. The contrastive second model is
  shipped only with accelerator headroom.
- **Single shared tokenizer.** Steering intersects base/expert token sets, so the
  adapter / expert / classifier must use the **identical** tokenizer (the reason
  for a same-family Qwen choice).
- The chunk guard and prompt-risk triage **reuse** the active tier's safety model;
  the backtracking loop adds **no** new weights.
- Every weight file (base, adapter, expert, classifier) is ED25519-signed and
  integrity-gated on load (ADR-006).

## Consequences

### Positive
- Safety becomes recoverable, not one-shot: drift is caught mid-generation and
  rewound to a safe prefix instead of repaired after the fact.
- Worst-case behaviour is bounded and deterministic (capped rollbacks → hard
  refusal), which suits an air-gapped device with no field re-tuning.
- Defence-in-depth: no single guard model is the whole safety case.

### Negative
- Rollback and replay add **latency variance**; pathological inputs can trigger
  repeated rewinds (a local DoS vector that the rollback cap must contain). On
  engines without in-place cache truncation each rewind replays prompt + kept
  prefix (see *Realized* below), so a single capped rollback can cost up to a full
  prefill — `max_rollbacks` is what keeps the worst case finite.
- Checkpoints cost memory (KV-cache handles/block refs + sampler/control state),
  competing with the ADR-003 budget on low-end devices.
- Loop, checkpoint, and async-guard state machine is materially more complex than
  a stateless per-token adjustment.

### Neutral
- Under memory pressure the loop degrades gracefully: checkpoint density and guard
  cadence thin out (and ultimately `SafetyDisabled`) per the ADR-003 policy.
- **As built:** hard bans apply every step; the early-token *soft-steering window*
  ships with the SecDecoding steerer (follow-up), and checkpoint spacing is the
  guard cadence (`guard_every`) — checkpoints are taken only at guard-verified-safe
  boundaries, so there is no separate checkpoint-cadence knob to misconfigure.
- Feasibility of cheap cache-handle checkpoints depends on the inference engine
  exposing KV-cache references; on [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)
  (Candle) this constrains how checkpoints are represented. **Realized — in two
  layers, with different cost models:**
  - *Session layer.* The descriptor-only `el-memory::KvRegion` (ADR-003) makes the
    session's rollback an `O(dropped)` `truncate` of KV descriptors — no payload
    copy — so the token-buffer fallback was not needed.
  - *Engine layer.* Restoring the engine's own KV cache is delegated to
    `InferenceEngine::rollback`, whose cost is adapter-dependent. A stateless
    engine restores in `O(1)`. But an engine whose cache is **append-only with no
    truncation API** cannot restore in place — candle 0.8.4's `quantized_qwen2`
    holds a *private* per-layer cache and exposes no `clear`/`truncate`, so
    `QwenEngine` rebuilds the safe prefix by **replaying prompt + kept tokens**
    (`O(prompt + kept_prefix)` per rollback). For that engine the "restores cache
    state, not a full replay" goal does **not** hold; the replay-free model is
    contingent on an engine that exposes true cache truncation. The replay is
    bounded by `max_rollbacks` (each capped rewind costs up to a prefill — see
    *Negative*), and a future engine with a truncation API can override
    `rollback` to recover the `O(dropped)` behaviour without any change to the
    control loop.

## Links
- Implementation: `crates/el-safety` (`ChunkGuard`, `SafetyScore`, `RollbackPolicy`, `CheckpointManager`/`Checkpoint`), `crates/el-runtime` (`InferenceSession::generate_with_policy` control loop), `crates/el-memory` (`KvRegion::truncate`)
- Research: [docs/research/SecDecoding alternatives research.md](../research/SecDecoding%20alternatives%20research.md)
- PRD: `docs/prd.md` §"Safety Guardrails (SecDecoding, CSD)", §Decoding Loop → Safety Adjustment
- DDD: [Safety context](../ddd/bounded-contexts/05-safety.md), [domain-events](../ddd/domain-events.md)
- Related: [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md), [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)
