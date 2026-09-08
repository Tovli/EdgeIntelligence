# ADR-017: Soft-steering window gate

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: safety, runtime, follow-up

## Context

[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
defines a **selective early-token soft-steering window** (8–32 tokens) as the
primary cost lever: apply model-backed `LogitAdjustment` only early in generation
(and on guard re-escalation), then fall back to normal decode. ADR-012 §Neutral
explicitly states: "the early-token soft-steering window ships with the
`SecDecoding` steerer (follow-up)."

[ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md)
added `RollbackPolicy::steer_window` and `generate_with_policy` window gating,
but `SecDecodingSteerer::adjust_with_logits` remains a placeholder returning
`LogitAdjustment::none()`. The window opens and closes correctly; the soft
adjustment it gates is a no-op. The `soft ≤ score < hard` branch in `RollbackPolicy`
is likewise a no-op — there is no `guard_mode` re-escalation bit.

This ADR is the **heaviest of the four follow-ups** (it needs the trained safety
assets from [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md) and the
Candle ACL from [ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md))
and should be implemented last.

Two gaps to close:

1. **Real soft steerer.** `SecDecodingSteerer` must produce a non-trivial
   `LogitAdjustment` — either contrastive (base vs. base+safety-LoRA) on
   accelerator HW, or a safety-LoRA direct adapter on CPU. ADR-013 already
   specified the contrastive arrangement; the missing piece is wiring it into the
   steering window and making the adjustment non-zero.

2. **Guard-mode re-escalation.** When `soft ≤ score < hard` the loop should
   re-open the steering window (`guard_mode = true`) so soft-steering continues
   past the nominal window end. Today that branch is a no-op, so elevated-risk
   generations receive no soft correction.

## Decision

Complete the soft-steering window gate by wiring the real steerer and adding a
`guard_mode` control bit.

1. **Split the port into two channels.** Keep `safety` as the always-on hard
   constraint port (fires every step, never gated). Add `steerer` as the
   soft-adjustment port (fires only within window or on `guard_mode`). This mirrors
   the ADR-012 architecture explicitly: hard ban ≠ soft steer. No existing test
   changes because `safety` is unchanged.

2. **Real `SecDecodingSteerer`.** Replace `adjust_with_logits` placeholder with an
   implementation that:
   - On **accelerator HW**: applies contrastive steering
     (`logits_expert − logits_base`, then `base + α·steer`) using
     `ContrastiveSteerer<QwenExpert>` from ADR-013's R3. Bounded to `steer_window`
     and top-K restricted; grammar mask applied before ranking.
   - On **CPU / Lightweight**: applies the safety LoRA adapter directly as a
     logit delta (single forward pass; cheaper than contrastive pair).
   - Returns `LogitAdjustment::none()` outside the window unless `guard_mode` is
     set.

3. **`guard_mode` bit.** Add `guard_mode: bool` to the loop's per-step control
   state. The `soft ≤ score < hard` branch in `RollbackPolicy` sets `guard_mode =
   true`. The steerer checks `guard_mode || step < steer_window` to decide whether
   to apply the adjustment. On the next hard-threshold-passing guard check,
   `guard_mode` is cleared.

4. **`LogitAdjustment::merge`.** Add a `merge` combinator so the hard-ban
   adjustment (`safety` channel) and the soft steering adjustment (`steerer`
   channel) are combined before sampling:
   ```
   final_adjustment = hard_ban.merge(soft_steer)
   ```
   Merge semantics: hard bans override; soft logit deltas accumulate (saturating
   milli-arithmetic). The combined adjustment is what `generate_with_policy` passes
   to the sampler — one call site, same as today.

5. **`SecDecodingAcl` boundary.** The steerer returns a **sparse milli-logit
   delta** (token index → i32 milli-logit) from the base/expert divergence. The
   adapter quantizes this to the `SecDecodingAcl` representation so the core stays
   float-free (the ADR-013 design). The quantization step lives in the adapter, not
   in `el-safety`.

6. **Mode surface.** Expose `secdecoding` in `el-chat --safety` and `el-bench
   --safety` only after the steerer returns a real adjustment and
   `SafetyModeSelector` correctly gates it on device class. Until then the flag
   remains hidden (matching the ADR-013 §7 "Faithful mode surface" rule).

## Consequences

### Positive
- The early-token steering window becomes a real cost-safety trade-off rather than
  a structural placeholder; the ADR-012 rationale ("captures most early-trajectory
  risk at the lowest sidecar cost") is finally realized.
- `guard_mode` re-escalation closes the `soft ≤ score < hard` no-op gap: elevated
  but sub-hard-threshold risk now receives a soft corrective signal.
- `LogitAdjustment::merge` makes the hard + soft combination a single, auditable
  step rather than two separate calls.

### Negative
- Heaviest of the four follow-ups: requires trained safety assets (ADR-014), the
  Candle ACL (ADR-013), and two forward passes per steered token on accelerator HW.
- `guard_mode` adds per-step mutable state to the loop; the state machine is
  correspondingly more complex.
- CPU hosts still pay single-LoRA forward-pass cost during the window — not free,
  but cheaper than contrastive.

### Neutral
- Hard-ban path (`LightweightFilter`) is completely unchanged — the `safety` port
  fires every step regardless.
- Extends ADR-013; supersedes nothing.

## Links
- Builds on: [ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md) (contrastive steerer R3, `steer_window` R1, `SecDecodingAcl`), [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md) (trained safety asset; soft/hard thresholds calibrated), [ADR-015](./ADR-015-semantic-boundary-checkpoints.md) (boundary checkpoints raise rollback precision under re-escalation), [ADR-016](./ADR-016-async-hydra-style-chunk-guard.md) (async guard reduces re-escalation latency on accelerator HW).
- Constrained by: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) (on-device, air-gapped), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) (sign steerer weights), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (memory budget).
- Implementation seams: `el-safety::SafetySteerer`/`SecDecodingSteerer` (real adjustment), `el-safety::LogitAdjustment::merge` (new combinator), `el-runtime::generate_with_policy` (`guard_mode` bit, dual-channel combine), `el-engine-candle::ContrastiveSteerer` / LoRA adapter path, `el-chat` + `el-bench` (`--safety secdecoding` flag).
- Source: `docs/followups.md` §2 "Soft-steering window gate".
- Sequence: implement last, after [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md), [ADR-015](./ADR-015-semantic-boundary-checkpoints.md), [ADR-016](./ADR-016-async-hydra-style-chunk-guard.md).
