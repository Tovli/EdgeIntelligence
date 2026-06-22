# ADR-023: Baseline performance instrumentation

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: telemetry, performance, testing, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) roadmap
Phase 1 §6, drawing on §P2.13 evaluation/ablation) makes baseline performance
instrumentation a Phase-1 deliverable: every optimization in this milestone
(ADR-018..022) must ship with a **measured** before/after, or the speedups are
unfalsifiable. The plan warns the ruvLLM performance figures are *claims* to be
independently benchmarked inside EdgeIntelligence before they become acceptance
criteria.

Instrumentation today is split and largely unpopulated:

- **`el-telemetry`** has the right shape — `MetricsCollector` folds content-free
  `DomainEvent`s into `TelemetrySnapshot { prefill_tps, decode_tps, ttft_ms,
  peak_bytes, tokens_generated, … }` (ADR-007). But the **runtime never feeds it
  real numbers**: `load_prompt` emits `PrefillCompleted { prefill_tps: 0 }` (a
  placeholder), and `MetricsSampled { decode_tps, ttft_ms, peak_bytes }` is
  defined but **never emitted** by `generate_with_policy`. `ttft_ms` only moves in
  unit tests.
- **`el-engine-candle`** has a *second*, disjoint path: an `EL_BENCH`-gated `bench`
  module that prints a per-phase wall-clock breakdown (load / tokenize / prefill /
  decode / detok, plus model-vs-seam-vs-loop forward attribution and tok/s) to
  **stderr**. It is useful but ad hoc, adapter-local, and not machine-readable.
- **`apps/el-bench`** exists as the harness but has no standardized, CI-consumable
  metric output to gate regressions.

So there are two measurement systems, neither of which produces a structured
baseline a CI gate can read.

## Decision

Make `el-telemetry` the single instrumentation authority, populate the existing
events with real measurements, and emit a machine-readable benchmark record from
`apps/el-bench`.

1. **Populate the existing event schema.** Compute and emit a **measured**
   `prefill_tps` from `load_prompt` (real once ADR-020 batches prefill), and emit
   `MetricsSampled { decode_tps, ttft_ms, peak_bytes }` from the decode loop —
   `ttft_ms` taken at the first **emitted** token (ADR-019). No new content on
   events; the ADR-007 `Copy`-only guarantee is preserved.

2. **One measurement path.** Fold the `EL_BENCH` stderr breakdown into a *view*
   over the same `TelemetrySnapshot` / event stream rather than a parallel timing
   system — the pretty stderr report becomes a renderer of the canonical numbers,
   not an independent source of them.

3. **Machine-readable benchmark output.** `apps/el-bench` emits a structured
   (JSON) record per run covering the plan's metric list where feasible on the
   host: binary size, cold/warm startup, peak RSS, time-to-first-token, decode
   tokens/s, prefill tokens/s, quality/safety scores, determinism, storage growth.
   Each record is tagged with **device, model, build flags, and context size** so
   numbers are never quoted context-free.

4. **Ablation per optimization.** Each P0 optimization ships behind a switch and
   with a baseline-vs-enabled comparison captured through this path: persistent
   sessions (ADR-018), streaming TTFT (ADR-019), batched prefill (ADR-020), mmap
   load (ADR-021), quantized/bounded KV (ADR-022).

5. **Regression gates.** Safety/quality regression thresholds (reusing the
   clinical-safety benchmark) and key performance thresholds **block merges** in
   CI; benchmark output is the artifact those gates read.

## Consequences

### Positive
- Establishes the measured baseline the whole milestone is judged against; ruvLLM
  claims become locally verified numbers or get rejected.
- Collapses two instrumentation paths into one content-free, CI-readable source of
  truth.
- Makes every P0 optimization independently switchable and provable — no "it feels
  faster."

### Negative
- Taking timings/RSS on the hot path has a (small) cost; sampling must stay cheap
  and off the per-token critical section where possible.
- Cross-platform metrics (energy, peak RSS) are not uniformly available; the
  record marks unavailable fields rather than faking them.
- A real CI perf gate needs stable hardware or tolerance bands to avoid flakiness.

### Neutral
- This is logically the *first* thing to land (instrument before optimizing), even
  though it is numbered last in this batch — ADR-018..022 each depend on it for
  their acceptance evidence.
- No change to the content-free telemetry guarantee (ADR-007); only numeric/enum
  fields are populated.

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) Phase 1 §6 (and §P2.13 evaluation/ablation).
- Builds on: [ADR-007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md) (`DomainEvent`/`MetricsCollector`/`TelemetrySnapshot`).
- Measures: [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md), [ADR-019](./ADR-019-in-loop-incremental-decoding-and-token-streaming.md), [ADR-020](./ADR-020-batched-single-pass-prefill.md), [ADR-021](./ADR-021-memory-mapped-verified-gguf-loading.md), [ADR-022](./ADR-022-two-tier-quantized-kv-cache-with-attention-aware-eviction.md).
- Implementation seams: `crates/el-telemetry` (`MetricsCollector`/`TelemetrySnapshot`), `crates/el-runtime` (emit measured `prefill_tps` + `MetricsSampled`), `crates/adapters/el-engine-candle` (fold `EL_BENCH` into the event view), `apps/el-bench` (machine-readable record + CI gate).
- Related: [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)/[ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md) (safety/quality regression gates reuse the clinical-safety benchmark).
