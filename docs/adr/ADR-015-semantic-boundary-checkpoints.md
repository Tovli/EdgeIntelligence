# ADR-015: Semantic-boundary checkpoints

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: safety, runtime, follow-up

## Context

[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
specifies checkpointing at **semantic boundaries** (newline, sentence break,
closing brace, tool-call delimiter) in addition to the fixed `guard_every` cadence.
Today the control loop in `generate_with_policy` places checkpoints only at the
guard cadence — semantic-boundary checkpointing is absent from the implementation.

Two consequences:

1. **Rollback targets land mid-clause.** Rolling back to the last checkpoint may
   truncate to a position inside a sentence or structured token, producing
   grammatically broken output even after a clean rollback.
2. **Guard checks are cadence-only.** The guard does not fire at natural output
   boundaries where a safety-relevant turn is most likely complete.

Boundary detection requires knowing which token IDs correspond to boundary surfaces
(newline, `\n`, `.`, `!`, `?`, `}`, tool-call delimiters). This is inherently
tokenizer-specific — the same surface character maps to different IDs across
tokenizer families — so detection cannot live in the engine-agnostic `el-safety`
crate. It belongs in the tokenizer-owning adapter.

The fix is a new port (`BoundaryDetector`) provided by the adapter and consumed by
the runtime loop. Absent an adapter implementation, the default is "never" —
current behaviour — so no existing test is broken.

## Decision

Add a `BoundaryDetector` port to the decode loop and integrate boundary checks
with the existing guard cadence.

1. **New port.** Define in `el-safety` (or `el-runtime`):
   ```rust
   pub trait BoundaryDetector: Send + Sync {
       fn is_boundary(&self, token: Token) -> bool;
   }
   ```
   Add `boundary: Option<Box<dyn BoundaryDetector>>` to `Ports` (same pattern as
   `guard`). Default: `None` → never fires (backward-compatible).

2. **Precomputed token-id set.** The adapter implements `BoundaryDetector` by
   scanning the vocabulary at load time for IDs whose surface string contains
   `\n`, sentence terminators (`. ! ?`), closing delimiters (`} ] )`), and
   tool-call delimiters. Store as a `HashSet<u32>`. `is_boundary` is then an
   `O(1)` set membership test — no string comparison in the hot loop;
   fully deterministic.

3. **Integrated loop behaviour.** In `generate_with_policy`, at each step:
   - if `ports.boundary.is_some()` and `is_boundary(token)` → fire a guard check
     immediately (in addition to the `guard_every` cadence check);
   - if the guard passes, take a checkpoint at this boundary step.
   - when cadence and boundary coincide (E3), the check is idempotent (a single
     guard call, a single checkpoint).

4. **Rollback precision.** Because checkpoints are only taken after a
   guard-verified-safe step (ADR-012 invariant), a boundary checkpoint is by
   definition a safe rollback target. No change to rollback logic — the improved
   rollback precision is a consequence of denser safe-boundary checkpoints.

## Consequences

### Positive
- Rollback targets land on natural output boundaries (clause/sentence starts)
  rather than arbitrary token offsets, producing coherent output after recovery.
- Guard checks at boundaries catch safety-relevant completions promptly without
  increasing the base cadence or the per-token overhead.
- E3 idempotency means no double guard cost when cadence and boundary coincide.

### Negative
- Adapter must scan its vocabulary at load time (one-time `O(|vocab|)` pass); adds
  a small start-up cost and requires the boundary surface list to be maintained per
  tokenizer family.
- `Ports` grows a new optional field; adapters not implementing `BoundaryDetector`
  silently get current behaviour (no regression, but the safety benefit is absent).

### Neutral
- Depends on [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md) only in the
  sense that both guard at boundary and guard at cadence trigger the same
  `ChunkGuard` implementation; a weights-free `AnchorGuard` also benefits.
- No change to `CheckpointManager`, `RollbackPolicy`, or the loop's control flow.

## Links
- Builds on: [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (semantic-boundary checkpointing specified), [ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md) (loop skeleton).
- Constrained by: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (tokenizer lives in the adapter).
- Implementation seams: `el-safety::BoundaryDetector` (new trait), `el-runtime::Ports` (new field), `generate_with_policy` (boundary check + checkpoint branch), `el-engine-candle` (vocab scan + `BoundaryDetector` impl).
- Source: `docs/followups.md` §3 "Semantic-boundary checkpoints".
- Sequence: implement after [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md), before [ADR-016](./ADR-016-async-hydra-style-chunk-guard.md).
