# ADR-022: Two-tier quantized KV cache with attention-aware eviction

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: memory, runtime, performance, on-device, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.3 +
§P0.4, roadmap Phase 2 opener / EPIC-4 — the stated first-milestone cutoff)
combines two KV concerns: a **two-tier quantized KV cache** (recent tokens at
high precision, older tokens quantized to Q8→Q4→Q3) and **attention-aware
eviction** (sliding-window first, then H2O/PyramidKV-style retention) with pinned
ranges for system prompts and tool definitions. KV memory grows with context and
is the dominant *mutable* cost after weights, so this is the milestone's
memory-reduction payload.

The codebase has KV in two layers, neither of which quantizes or evicts:

- **Session layer.** `el-memory::KvRegion` is **descriptor-only**: `KvSlot {
  token_index, offset, valid }` — it tracks *where* KV lives, not the K/V tensors
  themselves (ADR-003). It offers `push`, `truncate` (the ADR-012 rollback
  primitive), `mark_pruned`, and `compact` (for rejected drafts) — but no
  precision tiers and no eviction policy. `compact` emits `KvCacheCompacted`.
- **Engine layer.** The real K/V tensors live inside candle's `quantized_qwen2`,
  which is **append-only with no public truncation** (the same limitation ADR-012
  hit for rollback, forcing prompt replay).
- `SessionConfig.memory_budget_bytes` exists (ADR-003), and
  `MemoryBudgetExceeded` is defined, but nothing enforces a per-session KV ceiling
  or migrates old KV to a cheaper tier.

So there is a budget concept and descriptor bookkeeping, but no policy that bounds
or compresses the actual KV growth.

## Decision

Introduce a pluggable KV policy in `el-memory` and an eviction strategy layered on
it, degrading deterministically under the ADR-003 budget.

1. **`KvCachePolicy` port.** Extend `el-memory` from descriptors to a policy over
   real KV storage (the plan's interface):
   ```rust
   pub trait KvCachePolicy {
       fn append(&mut self, layer: usize, key: TensorView, value: TensorView) -> Result<(), MemoryError>;
       fn view_for_attention(&self, layer: usize, range: TokenRange) -> Result<KvView<'_>, MemoryError>;
       fn memory_usage(&self) -> KvMemoryStats;
   }
   ```
   Recent tokens stay FP16/BF16; older blocks quantize to **Q8 first**, then Q4/Q3
   as evaluated. Blocks are **dequantized only when attention needs them**. The
   uncompressed policy remains the reference implementation.

2. **Per-session KV budget.** The policy enforces a hard per-session KV ceiling
   derived from `memory_budget_bytes`; exceeding it triggers tiering/eviction, and
   `MemoryBudgetExceeded` / `KvCacheCompacted` make the action observable
   (content-free, ADR-007).

3. **Eviction, staged.** Start with **deterministic sliding-window** eviction; add
   **pinned ranges** (system prompt, tool definitions) that are never evicted; put
   **H2O/PyramidKV** attention-scored retention behind an experimental feature
   flag. Eviction must never remove tokens required by the grammar/tool-call state
   (ADR / grammar constraint) — those are implicitly pinned.

4. **Engine-capability honesty.** True per-block KV quantization/eviction requires
   an engine that exposes its KV cache. candle's `quantized_qwen2` does not, so on
   the **current** engine this ADR lands as: (a) the `KvCachePolicy` port +
   reference impl + budget accounting at the session/descriptor layer, and (b)
   deterministic sliding-window/pinning where the descriptor layer can express it.
   Full tiered-tensor quantization is realized only with an engine that owns a
   truncatable/inspectable cache (a custom attention kernel — see the deferred
   flash/sparse-attention work, plan §P1.7/§P2.17). This contingency is recorded
   exactly as ADR-012 recorded the rollback-replay contingency.

5. **Determinism.** In deterministic mode, eviction decisions are reproducible
   (sliding-window and pinning are pure functions of token positions; the
   experimental attention-scored path is excluded from deterministic builds).

## Consequences

### Positive
- Bounds the dominant mutable memory cost: longer conversations or smaller budgets
  on phones/embedded targets without blunt whole-history truncation.
- Pinned system/tool ranges preserve instruction-following while trimming
  mid-history — better long-context quality than FIFO.
- Pluggable policy keeps an uncompressed reference for A/B quality regression.

### Negative
- Real tensor-level tiering needs engine KV access that candle's
  `quantized_qwen2` doesn't expose — full realization waits on a custom kernel;
  until then the win is partial (budget + descriptor-layer eviction).
- Quantization adds dequant overhead on the attention read path; if it erases the
  decode-time gain it is not worth shipping — gated by the ADR-023 benchmark.
- Q4/Q3 tiers risk quality regression; each tier must pass deterministic quality
  tests before promotion past Q8.

### Neutral
- `KvRegion`'s descriptor model, `truncate` (ADR-012 rollback), and `compact`
  remain; this ADR adds a policy *above* them, not a replacement.
- Interacts with ADR-021: mapping quantized weights and on-demand dequant share
  the same "decompress only what's needed" discipline.

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.3 + §P0.4, EPIC-4 (milestone cutoff).
- Builds on: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (memory budget, `KvRegion` descriptors), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md) (per-session KV lifecycle).
- Constrained by: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (engine KV access — candle has no cache truncation), [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (rollback `truncate` must keep working under any policy), [ADR-007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md) (eviction events stay content-free).
- Pairs with: [ADR-021](./ADR-021-memory-mapped-verified-gguf-loading.md) (on-demand dequant discipline).
- Implementation seams: `crates/el-memory` (`KvRegion`, new `KvCachePolicy`/`KvView`/`KvMemoryStats`), `crates/adapters/el-engine-candle` (`QwenEngine` cache access — contingent on a truncatable cache), `crates/el-core` (`MemoryBudgetExceeded`/`KvCacheCompacted` events).
- Measured by: [ADR-023](./ADR-023-baseline-performance-instrumentation.md) (peak KV bytes, decode tok/s with vs. without tiering, long-context quality).
