# Bounded Context 9 — Telemetry & Privacy  (Generic)

> A one-way, content-free observer that samples performance counters for
> optimization and exposes a housekeeping report — and structurally guarantees
> no user content is ever captured. Source: PRD §"Telemetry & Privacy",
> §"Privacy", §"Air-Gap Compliance".

## Purpose

Give developers performance visibility (tokens/sec, latency, memory) without
ever recording prompts, responses, or any user content, and without any cloud
call.

## Strategic role

**Generic**, but it encodes a hard product invariant: **privacy by
construction**. It is a pure downstream subscriber — nothing depends on it, and
it depends on no other context's internals.

## Ubiquitous language (context-local)

`Telemetry Snapshot`, `Housekeeping Report`, `Throughput Metric`, `Latency
Metric`, `Memory High-Water Mark`, `Air-Gap`.

## Aggregates

### `TelemetrySnapshot` (Aggregate Root)
A content-free sample of the running session's performance.

- **Identity:** `SnapshotId` (+ `SessionId` correlation, no content).
- **Holds:** `ThroughputMetric` (prefill & decode t/s), `LatencyMetric` (TTFT,
  per-token), `MemoryHighWaterMark`, and CPU/GPU/NPU utilization samples.
- **Invariants:**
  - **No field may contain prompt or response text, tokens-as-text, or any
    user-derived content** — only numeric counters and opaque ids. This is the
    defining invariant of the context.
  - Snapshots live in volatile memory and are surfaced only via an opt-in
    callback; nothing is persisted by default.
  - No outbound network channel exists here; if a developer adds analytics, it
    must go through their own (optionally differentially-private) channel,
    outside this context.

### `HousekeepingReport` (Entity)
- **Holds:** aggregate session stats — tokens generated, memory high-water mark,
  utilization timing samples.
- **Invariant:** aggregate and anonymized; contains no user content.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `ThroughputMetric` | { prefillTps, decodeTps } | |
| `LatencyMetric` | { ttftMs, perTokenMs } | |
| `MemoryHighWaterMark` | peak bytes | sourced from Memory (6) events |
| `UtilizationSample` | { cpu%, gpu%, npu%, t } | timed sample |
| `SnapshotId` | opaque id | no content |

## Domain Services

- **`MetricsCollector`** — subscribes to domain events from all contexts and
  folds them into `TelemetrySnapshot`s. It reads only event metadata
  (counts/scores/timings), never payloads.
- **`ReportPublisher`** — emits snapshots/reports through the opt-in callback.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `DomainEventStream` | all contexts (PL) | inbound, subscribe-only |
| `MetricsCallback` | host app | outbound, opt-in |

## Anti-Corruption Layer

None needed inbound (it consumes already-domain events). A thin outbound adapter
shapes snapshots for the host callback. The privacy invariant is enforced at the
event schema level — events are *designed* to be content-free (see
[domain-events.md](../domain-events.md)).

## Domain Events (published)

`MetricsSampled`, `HousekeepingReportReady`. (This context mostly *consumes*
events.) See [domain-events.md](../domain-events.md).

## Relationships

**Published-Language subscriber, one-way.** It is downstream of every other
context and upstream of nothing. Because no context calls into Telemetry and
Telemetry reads only content-free event metadata, the "no user data leaves the
pipeline" privacy guarantee is structural, not merely procedural.
