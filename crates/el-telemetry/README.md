# el-telemetry — content-free metrics collector

A one-way, downstream subscriber that folds content-free
[`el_core::DomainEvent`](../el-core)s into performance snapshots (ADR-007).

It depends on `el-core` and **nothing depends on it**, and it has no network
channel of its own. Because it can only ever read the numeric and enum fields of
events that are *already* content-free by construction, "no user content in
telemetry" is structural — there is no code path by which a prompt or response
could reach a metric.

## What it provides

- **`MetricsCollector`** — subscribes to the domain-event stream and maintains a
  running snapshot. Call `observe(&envelope)` per event and `snapshot()` to read.
- **`TelemetrySnapshot`** — a `Copy` struct of counters and gauges:
  `prefill_tps`, `decode_tps`, `ttft_ms`, `peak_bytes`, `tokens_generated`,
  `compressions`, `safety_violations`.

`peak_bytes` is a monotonic high-water mark — it only ever rises across the
events it observes.

## Usage

```rust
use el_core::{DomainEvent, EventEnvelope, SessionId};
use el_telemetry::MetricsCollector;

let mut collector = MetricsCollector::new();

collector.observe(&EventEnvelope::new(
    SessionId(1),
    0,
    DomainEvent::PrefillCompleted { prompt_tokens: 100, kv_len: 100, prefill_tps: 480 },
));
collector.observe(&EventEnvelope::new(
    SessionId(1),
    1,
    DomainEvent::TokenCommitted { kv_len: 101 },
));

let snap = collector.snapshot();
assert_eq!(snap.prefill_tps, 480);
assert_eq!(snap.tokens_generated, 1);
```

## Place in the workspace

In a full build, the runtime drains `EventEnvelope`s from an `InferenceSession`
(or a cloud `CloudProvider`'s event sink) and feeds them here. This crate is the
read-only end of that pipeline.

## Status

Implemented and tested.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-007](../../docs/adr/ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md);
see the [Telemetry & Privacy](../../docs/ddd/bounded-contexts/09-telemetry-privacy.md) context.
