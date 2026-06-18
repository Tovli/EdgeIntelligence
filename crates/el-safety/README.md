# el-safety — on-device decoder-time safety

On-device, tiered, decoder-time safety (ADR-005). Safety is applied as a
per-step logit adjustment during decoding — **after** the grammar mask and
**before** sampling — so it only ever steers over already-legal tokens. **No
safety path touches the network.**

Depends only on `el-core`. No `unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

- **`SafetySteerer`** — the per-step intervention trait: `adjust(recent_tokens)
  -> LogitAdjustment` and `mode()`. The runtime calls this each decode step.
- **`LogitAdjustment`** — a sparse, integer (milli-logit) vector subtracted from
  target logits. Sparse + integer keeps steering deterministic and
  allocation-light. `delta_for(token)`, `l1_norm_milli()` (what the
  `LogitsSteered` telemetry event reports), `is_empty()`.
- **`SafetyModeSelector::resolve(requested, device)`** — budget-gates the tier
  by device profile: `SecDecoding` (two ~1B models) is downgraded to
  `Lightweight` on a `MidRange` device.
- **Steerers per `SafetyMode`:**
  - `NoSafety` (`Off`) — a no-op.
  - `LightweightFilter` (`Lightweight`) — a training-free blacklist filter
    (**fully implemented**). Banned tokens receive `HARD_BAN = -1_000_000`
    milli-logits so they cannot be sampled.
  - `SecDecodingSteerer` (`SecDecoding`) — base-vs-safety-model steering.
    **Scaffolded** follow-up: it requires two ~1B models on Candle, so until the
    assets are wired it returns no adjustment while honestly reporting its mode
    (so callers can select it without it silently mis-steering).
- **ADR-012 control-loop primitives** (consumed by `el-runtime`'s decode loop):
  - `ChunkGuard::score(recent) -> SafetyScore` — risk of the recent output window.
  - `SafetyScore` — integer milli-units `0..=1000` (deterministic, float-free).
  - `RollbackPolicy::for_device(device, mode)` — tier-aware bounds (guard cadence
    `guard_every`, soft/hard thresholds, `max_rollbacks`, checkpoint-ring size).
    The early-token soft-steering window ships with the SecDecoding steerer
    (follow-up); hard bans apply every step.
  - `CheckpointManager` / `Checkpoint` — a bounded ring of guard-verified
    safe-prefix snapshots (offsets only; KV payload is never copied).

## Usage

```rust
use el_core::{DeviceTarget, SafetyMode};
use el_safety::{LightweightFilter, SafetyModeSelector, SafetySteerer};

// On a mid-range device, SecDecoding is downgraded to a tier it can afford.
let mode = SafetyModeSelector::resolve(SafetyMode::SecDecoding, DeviceTarget::MidRange);
assert_eq!(mode, SafetyMode::Lightweight);

// Lightweight bans specific token ids outright.
let filter = LightweightFilter::new(vec![42, 99]);
let adj = filter.adjust(&[]);
assert_eq!(adj.delta_for(42), LightweightFilter::HARD_BAN);
assert_eq!(adj.delta_for(7), 0);
```

## Place in the workspace

Re-exported by `el-runtime` (`el_runtime::{SafetySteerer, LogitAdjustment,
ChunkGuard, SafetyScore, RollbackPolicy, Checkpoint, CheckpointManager}`) so
callers wire a single type system. The session applies the chosen steerer in the
invariant decode order `grammar mask → safety adjust → sample → commit`, wrapped
by the ADR-012 checkpointed-rollback control loop.

## Status

Partial by design: the `Lightweight` blacklist path is real and tested;
`SecDecoding`/`Csd` model-backed steering is a tracked follow-up that needs
model assets. The ADR-012 checkpointed-rollback control-loop primitives
(`ChunkGuard`, `RollbackPolicy`, `CheckpointManager`) are implemented and tested,
and the `el-runtime` session drives them (hard bans every step, chunk-guard
cadence with a mandatory final check, safe-prefix checkpoints, bounded
fail-closed rollback). Selective soft-steering over an early-token window is part
of the deferred `SecDecoding` follow-up above, not current behavior. A trained
`ChunkGuard` model is the remaining follow-up — the loop runs against any
`ChunkGuard` implementation.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-005](../../docs/adr/ADR-005-on-device-only-tiered-decoder-time-safety.md) and
[ADR-012](../../docs/adr/ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md);
see the [Safety](../../docs/ddd/bounded-contexts/05-safety.md) context.
