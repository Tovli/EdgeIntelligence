# ADR-007: Content-free domain events for privacy-by-construction telemetry

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: privacy, telemetry, events, generic

## Context

Developers need performance visibility (tokens/sec, latency, memory high-water
mark, utilization), but the product guarantees that user data never leaves the
device and is not logged (PRD §"Telemetry & Privacy", §"Privacy"). A telemetry
subsystem that can *see* prompt/response content is a standing liability even if
policy forbids logging it — the guarantee should be structural, not procedural.

This shapes the [Telemetry & Privacy](../ddd/bounded-contexts/09-telemetry-privacy.md)
context and the schema of every domain event across the system.

## Decision

Design **all domain events to be content-free** — payloads carry only ids,
counts, scores, timings, and enums, never prompt/response text or tokens-as-text.
Make **Telemetry a one-way, downstream Published-Language subscriber**: it
consumes event metadata only, nothing depends on it, and it has **no network
channel of its own**. Snapshots live in volatile memory and surface solely via an
opt-in host callback. Any analytics a developer adds (optionally
differentially-private) lives outside the SDK boundary.

## Consequences

### Positive
- "No user data leaves the pipeline" is enforced by the event schema and
  dependency direction — not merely by a logging policy.
- Telemetry can evolve independently without coupling to any context's internals.

### Negative
- Debugging cannot rely on captured content; only counters/timings are available,
  which can make field diagnosis harder.
- Every new event must be reviewed to confirm it carries no user-derived content.

### Neutral
- The degradation policy (`MemoryBudgetExceeded` → `CompressionSkipped` /
  `SpeculationDisabled` / `SafetyDisabled`) is itself expressed as content-free
  events, making it observable without inspecting payloads.

## Links
- PRD: `docs/prd.md` §"Telemetry & Privacy", §"Privacy"
- DDD: [Telemetry & Privacy context](../ddd/bounded-contexts/09-telemetry-privacy.md), [domain-events catalog](../ddd/domain-events.md)
- Related: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)
