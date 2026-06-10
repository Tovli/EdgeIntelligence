# ADR-005: On-device-only, tiered decoder-time safety (no cloud moderation)

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: safety, compliance, on-device, supporting

## Context

Safety enforcement cannot rely on cloud moderation given the air-gap invariant
([ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)), yet the
strongest technique in the PRD (SecDecoding, two ~1B models) is too heavy for a
0.5B-on-mid-range scenario. The PRD (§"Safety Guardrails", §Decoding Loop →
Safety Adjustment) proposes a spectrum: SecDecoding logit steering, a lightweight
classifier / ~0.1B safety expert, and Claim-Based Streaming Decoding (CSD) with
conformal guarantees for regulated use-cases.

The challenge is fitting safety to wildly varying device budgets without ever
leaving the device.

## Decision

Enforce safety **entirely on-device at decode time**, via a **tiered
`SafetyMode`**: `Off | Lightweight | SecDecoding | CSD`. The
[Safety](../ddd/bounded-contexts/05-safety.md) context's `SafetyModeSelector`
chooses the affordable mode from the `DeviceProfile`: `SecDecoding` is rejected
on `MidRange` (falls back to the `Lightweight` ~0.1B classifier or token-anchor
heuristics); `CSD` is offered as a reference implementation for stringent
domains. No safety path may reach the network — this context is stricter than the
SDK-wide air-gap default.

Within a decode step, the safety `LogitAdjustment` is applied **after** the
grammar mask and **before** sampling, so steering operates only over
already-legal tokens (see decode-step order in [domain-events](../ddd/domain-events.md)).

## Consequences

### Positive
- Safety works offline and scales down to mid-range hardware.
- Compliance posture (partial regulatory claims, auditable thresholds) without a
  cloud dependency.

### Negative
- The strongest guarantees (full SecDecoding, CSD) are unavailable or costly on
  low-end devices; protection quality varies by tier.
- CSD backtracking can rewind generation, adding latency variance.

### Neutral
- Under memory pressure, Safety is degraded after compression and speculation per
  the [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
  budget policy, emitting `SafetyDisabled`.

## Links
- PRD: `docs/prd.md` §"Safety Guardrails (SecDecoding, CSD)", §"Safety Guardrails" (specs)
- DDD: [Safety context](../ddd/bounded-contexts/05-safety.md), [domain-events](../ddd/domain-events.md)
- Related: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
