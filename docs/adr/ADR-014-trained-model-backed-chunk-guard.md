# ADR-014: Trained model-backed chunk guard

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: safety, security, on-device, runtime, follow-up

## Context

[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
specified a **chunk guard** that scores recent output every 4–16 tokens and feeds
`RollbackPolicy`; it states the guard *reuses the active tier's safety model*.
[ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md)
wired the full loop skeleton including `SecDecodingSteerer`, contrastive steering,
and ingress triage, but its §"Still deferred" explicitly lists **the safety
LoRA / classifier weights themselves** as unbuilt. Today `ChunkGuard` is
implemented only as `AnchorGuard` — token n-gram anchor heuristics — which can be
evaded by sub-word fragments, morphological variants, or cross-lingual paraphrase.
`RollbackPolicy`'s soft/hard milli-thresholds are guessed constants, not
empirically calibrated values.

Consequence: the `SafetyScore` that gates rollback decisions is a coarse,
bypassable signal, and the loop's FPR/FNR (over-refusal vs. adversarial slip-rate)
is unknown. [Clinical-safety benchmark context](../../.claude/projects/D--Code-Tovli-EdgeIntelligence/memory/edge-llm-sdk-clinical-safety-benchmark.md)
records the base model as crisis-unsafe, making this the highest-priority gap.

The `ChunkGuard` trait and `SecDecodingAcl` boundary already exist; a real
implementation adds a model-backed adapter *behind the existing port* — the
control loop is unchanged.

## Decision

Ship a trained, quantized **chunk-guard adapter** that implements the existing
`ChunkGuard` port, plus a calibration pass through `apps/el-bench` to set the
policy thresholds empirically.

1. **Adapter crate.** Create `crates/adapters/el-guard-candle` (or fold into
   `el-engine-candle` to share its tokenizer/runtime), implementing
   `el_safety::ChunkGuard`. The adapter must **not** live in `el-safety` (keeps
   `el-safety` dependency-free).

2. **Model.** A small (~0.1 B parameter) distilled binary classifier operating over
   the **recent token window** (avoids detokenisation; stays deterministic).
   Quantize to **INT8** for the Candle CPU path. The classifier emits an integer
   logit mapped to `SafetyScore` via `SafetyScore::from_milli`.

3. **Load gate.** The adapter constructor takes a `LoadPermit`
   (mirroring `InferenceSession::new` and `QwenExpert::from_path_primed`).
   Safety-classifier weights are ED25519-signed and integrity-gated exactly like
   the generator, per [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md).
   This turns AC-10 ("inherited" load gate on the guard) into **enforced**.

4. **Heterogeneity preserved.** The trained guard is one signal. `AnchorGuard`
   (deterministic blacklist) and ingress rules remain active alongside it — the
   ADR-012 heterogeneity requirement. Under memory pressure the loop degrades to
   `AnchorGuard`-only, per [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md).

5. **Calibration pass.** Wire `apps/el-bench` against
   `CounselBench` / `MindEval` / `VERA-MH` and sweep the soft/hard
   milli-thresholds over a validation split. Report **ASR** (adversarial success
   rate) and **over-refusal rate** for each candidate threshold pair. The chosen
   thresholds are recorded in `RollbackPolicy`'s default constants with their
   calibration source committed alongside.

6. **Seam choice.** Prefer the guard running its own small forward pass over the
   recent-token window (modular, seam unchanged). Only if latency measurements
   force it should the adapter extend `InferenceEngine` with a `last_hidden()`
   hook and a cheap linear probe head — that couples guard to engine internals and
   is the optimization, not the starting point.

## Consequences

### Positive
- `ChunkGuard` becomes a real safety signal; the rollback loop gains discriminating
  power and the el-bench A/B comparison becomes meaningful.
- Threshold calibration turns a guessed constant into an empirically validated
  FPR/FNR trade-off, surfacing the current baseline before further investment.
- No change to the control loop — swappable behind the existing port.

### Negative
- Ships a new model artifact: added memory (INT8 ~0.1 B ≈ 100 MB), load time, and
  signing/distribution overhead ([ADR-011](./ADR-011-multi-registry-release-ci-crates-io-npm-pub-dev.md),
  ADR-006).
- Classifier training and benchmark calibration are pre-coding work; the ADR
  records the architecture decision but not the training pipeline.
- A small INT8 classifier will have its own failure modes (false negatives on
  novel phrasing, false positives on medical terminology); performance monitoring
  via el-bench is the mitigation.

### Neutral
- Low-end devices without the 100 MB budget degrade to `AnchorGuard` heuristics —
  same behaviour as today.
- The adapter extends [ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md)
  without superseding it; the loop skeleton, steerer, and ingress triage remain
  unchanged.

## Links
- Builds on: [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (loop + chunk-guard cadence), [ADR-013](./ADR-013-model-backed-steering-layers-for-the-hybrid-safety-control-loop.md) (model inventory, still-deferred list).
- Constrained by: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) (on-device, air-gapped), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) (sign classifier weights), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (memory budget / degradation), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (Candle INT8 inference).
- Implementation seams: `el-safety::ChunkGuard`, `el-safety::SafetyScore::from_milli`, `el-safety::RollbackPolicy` (thresholds), `crates/adapters/el-guard-candle` (new), `apps/el-bench` (calibration gate).
- Source: `docs/followups.md` §1 "Trained ChunkGuard model".
- Sequence: implement before [ADR-015](./ADR-015-semantic-boundary-checkpoints.md), [ADR-016](./ADR-016-async-hydra-style-chunk-guard.md), [ADR-017](./ADR-017-soft-steering-window-gate.md).
