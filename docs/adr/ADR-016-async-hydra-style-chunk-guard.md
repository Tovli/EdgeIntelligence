# ADR-016: Async (Hydra-style) chunk guard

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: safety, runtime, performance, follow-up

## Context

[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
names an async Hydra-style guard — running the chunk guard concurrently with
generation so valid output pays near-zero guard cost — but notes it is
**accelerator-class HW only** and defers it as an optimization.
[ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md)
lists it in §"Still deferred". Today the guard is **synchronous**: each
`guard_every`-token check blocks the decode loop until the guard returns.

With the trained classifier from [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md)
in place, the sync guard's latency is non-trivial on accelerator hardware (Jetson,
Core Ultra — ADR-012 §Heterogeneity). On those platforms the async pattern allows
generation to continue while the previous chunk is being scored, cutting per-token
guard overhead to near zero for non-breaching chunks.

Correctness constraint: output must remain a deterministic function of
`(input, model, policy)`. Rollback targets must depend on the **logical chunk
index**, never on real-time arrival order of guard results. The checkpoint ring
must span the in-flight window so a delayed breach still has a safe checkpoint to
roll back to.

This ADR is a performance optimization; it adds no new safety capability. It must
not be implemented before ADR-014 (no trained guard to overlap) or ADR-015 (the
boundary checkpoints establish the rollback-target density).

## Decision

Make the `ChunkGuard` port non-blocking via an async interface pushed to the
adapter; keep `el-safety` and `el-runtime` sync and dependency-free.

1. **Extended port.** Add an optional non-blocking method to `ChunkGuard`:
   ```rust
   pub trait ChunkGuard: Send + Sync {
       fn score(&self, recent: &[Token]) -> SafetyScore;          // sync; always required
       fn score_async(&self, recent: &[Token]) -> Option<Ticket>; // non-blocking; default None
   }
   ```
   `Ticket` is an opaque handle the loop passes back to `try_recv(ticket) ->
   Option<SafetyScore>`. Adapters that implement only `score` opt out of async
   automatically.

2. **Worker thread in the adapter.** `el-guard-candle`'s `AsyncChunkGuard`
   owns a background thread holding the classifier. `score_async` sends the
   recent-token window over a channel and returns a `Ticket`; the worker scores
   and sends back a `SafetyScore`. The thread and classifier are fully contained in
   the adapter — no threads cross into `el-safety` or `el-runtime`.

3. **Loop changes.** In `generate_with_policy`:
   - On async adapters: at each guard cadence point, call `score_async`, store the
     `Ticket`, and continue decoding.
   - Each subsequent step: call `try_recv(ticket)`. On `Some(score)`: apply
     `RollbackPolicy`. On `None`: continue.
   - On breach: roll back to `CheckpointManager::last_safe_at_or_before(scored_index)`
     (see §4). On sync adapters: behaviour is unchanged.

4. **`CheckpointManager::last_safe_at_or_before(index)`.** Today
   `CheckpointManager` exposes `last()`. Add
   `last_safe_at_or_before(output_len: usize) -> Option<&Checkpoint>` that returns
   the newest checkpoint whose `output_len` ≤ the scored window end. This is the
   E4 behaviour already specified in ADR-012 §Bounded rollback.

5. **Determinism guard-rail.** The rollback target is `last_safe_at_or_before(
   scored_chunk_end_index)` where `scored_chunk_end_index` is the **logical token
   offset** at which the chunk was submitted for scoring — fixed at submission,
   independent of when the result arrives. The checkpoint ring is sized to cover
   at least one full async in-flight window (≥ `guard_every` tokens of history) so
   the safe checkpoint is never evicted before the result lands.

6. **Tier guard.** `score_async` is only activated on `DeviceProfile::Accelerator`
   class HW. On CPU / low-power tiers the loop calls `score` synchronously — same
   as today. The tier selector is in the adapter factory, not in `el-runtime`.

## Consequences

### Positive
- On accelerator hardware, non-breaching chunks cost near-zero guard latency;
  generation throughput approaches the unguarded baseline.
- Determinism is preserved: rollback targets are a pure function of logical token
  offsets, not scheduling.
- No change to `el-safety` or `el-runtime` safety semantics; the sync path remains
  the universal fallback.

### Negative
- Adds concurrency to the adapter layer; the worker thread and channel must be
  well-tested for shutdown and panic propagation.
- The checkpoint ring must be sized conservatively (≥ in-flight window) — under
  memory pressure the async guard degrades to sync before the ring shrinks, to
  avoid losing the rollback anchor.
- A delayed breach means tokens generated *after* the flagged window have already
  been emitted (or buffered). The rollback discards them but the caller may have
  already partially consumed them — callers must be prepared for truncation.

### Neutral
- Optimization only; does not change the safety guarantee level (no new model
  signal, no new policy). ADR-012 and ADR-013 remain the safety-substance ADRs.
- CPU hosts are unaffected; this ADR changes nothing for the default device tier.

## Links
- Builds on: [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md) (trained guard to overlap), [ADR-015](./ADR-015-semantic-boundary-checkpoints.md) (boundary checkpoints provide rollback-target density), [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (async guard specified; E4 rollback).
- Constrained by: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (checkpoint ring must fit memory budget), [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) (no network).
- Implementation seams: `el-safety::ChunkGuard` (extend with `score_async`/`Ticket`), `el-safety::CheckpointManager` (`last_safe_at_or_before`), `el-runtime::generate_with_policy` (async ticket loop), `el-guard-candle::AsyncChunkGuard` (worker thread).
- Source: `docs/followups.md` §4 "Async (Hydra-style) guard".
- Sequence: implement after [ADR-014](./ADR-014-trained-model-backed-chunk-guard.md) and [ADR-015](./ADR-015-semantic-boundary-checkpoints.md), before [ADR-017](./ADR-017-soft-steering-window-gate.md).
