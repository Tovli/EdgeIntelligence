# ADR-004: Air-gapped by default with opt-in local-network HybridMode

- **Status**: accepted — partially amended by [ADR-010](./ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md)
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: privacy, networking, compliance, invariant

## Context

The product promise is a private, offline-capable LLM with "zero trust in the
cloud" (PRD Executive Summary, §"Air-Gap Compliance"). All inference, safety, and
storage must work with no internet or cell data. The PRD also allows an optional
`HybridMode` that consults a *local-network* relay (e.g. Frontier) for retrieval —
but never a cloud API.

Without an explicit architectural rule, network access tends to creep in across
contexts (telemetry, model updates, retrieval), eroding the core guarantee.

## Decision

Treat **air-gap as a system-wide invariant**, not a feature: no bounded context
may open a network egress. The **only** permitted outbound network edge is a
single, **explicitly opt-in `HybridRelayPort` on the
[Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md) context**,
and it may reach a **LAN relay only — never a cloud endpoint**. Model updates
arrive out-of-band through a signed channel (see
[ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md)),
not a live API. Use of the relay emits a `HybridRelayConsulted` event for
auditability.

## Consequences

### Positive
- The privacy/offline promise is enforced structurally — there is exactly one
  auditable network seam to review.
- Satisfies strict offline/regulated deployments (GDPR/CCPA: all processing
  local).

### Negative
- Features that would naturally use the network (cloud retrieval, server-side
  moderation) are foreclosed or pushed to the opt-in LAN relay only.
- Developers wanting a cloud upgrade path must build it outside the SDK boundary.

### Neutral
- Telemetry is independently constrained to be content-free and has no network
  channel of its own (see [ADR-007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md)).

## Links
- PRD: `docs/prd.md` Executive Summary, §"Air-Gap Compliance", §"Privacy"
- DDD: [Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md), [Telemetry & Privacy](../ddd/bounded-contexts/09-telemetry-privacy.md), [context-map](../ddd/context-map.md)
- Partially amended by: [ADR-010](./ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md) — adds a
  second opt-in egress for cloud LLM APIs (distinct from `HybridRelayPort`);
  the "LAN relay only" restriction applies to the default path only
- Related: [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md), [ADR-007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md)
