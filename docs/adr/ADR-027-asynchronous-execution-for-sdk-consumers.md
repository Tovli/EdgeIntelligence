# ADR-027: Asynchronous execution for SDK consumers

- **Status**: accepted — staged native React Native rollout; Rust/Dart/WASM follow-ups remain open
- **Date**: 2026-08-12
- **Deciders**:
- **Tags**: runtime, async, bindings, cancellation, concurrency, performance, follow-up

## Context

Inference is a long-running, CPU/accelerator-bound operation. SDK consumers
must be able to start a request without blocking a UI thread, JavaScript event
loop, Dart isolate, or browser main thread.

The current repository does not provide one consistent execution contract:

- `el-core::LlmProvider` and `InferenceSession` are deliberately synchronous
  and deterministic.
- `el-ffi::EdgeLlm::ask` and `ask_stream_cb` invoke the provider directly. The
  React Native facade documents that these legacy compatibility methods execute
  on the JavaScript thread and therefore applies a 64-token cap to them.
- The Dart facade returns `Future`/`Stream`, but its FRB functions still run the
  provider call inline in the generated task. Cancelling the Dart stream closes
  the sink while the underlying generation continues because the provider has
  no cancellation signal.
- The WASM facade exposes a blocking `ask_wasm` method. A browser consumer
  cannot safely use that method on the main thread, and the synchronous provider
  seam cannot await browser APIs.

Persistent model instances and stateful sessions (ADR-018) make execution
ownership important: two turns must not mutate the same session concurrently,
and reset/close must not race an active generation. Real streaming (ADR-019)
also requires cancellation and safe handling of tokens that have not yet passed
the safety checkpoint.

Making `LlmProvider` itself async would push an executor choice into the core
domain and complicate native, WASM, and test implementations. The problem is
not that the inference contract needs an async state machine; it is that host
consumers need an asynchronous execution boundary around the synchronous
inference contract.

## Decision

Add a worker-backed asynchronous execution boundary at the SDK facade/binding
layer while keeping the Rust inference core synchronous.

### 1. Separate inference from execution

`LlmProvider` and `InferenceSession` remain synchronous, deterministic seams.
The host-facing facade adds an asynchronous request executor (the concrete
type may be named `AsyncEdgeLlm`, `RequestExecutor`, or an equivalent adapter)
that runs those calls on SDK-owned workers and returns a future, promise, or
stream immediately.

The core must not acquire a mandatory Tokio or other async-runtime dependency.
Native bindings may use a platform executor, Rust consumers may inject a
blocking-spawn/executor adapter, and WASM uses a worker-compatible adapter.
The executor owns request scheduling; the inference loop remains the owner of
model, KV-cache, grammar, and safety state.

### Staged implementation scope

This decision is accepted, but its platform delivery is deliberately staged:

- The native React Native binding implements per-handle bounded worker-backed
  `askAsync`/`askStreamAsync` callback APIs and a TypeScript `Promise` helper.
- The current local Candle and Qwen providers still generate a complete safe
  reply before replaying fragments. The worker boundary removes JavaScript-thread
  blocking, but true in-loop safe-token streaming remains ADR-019 work.
- Rust `Future` APIs, a Dart request handle/backpressure surface, and WASM
  `Promise`/`ReadableStream` worker entry points are deferred. Dart
  subscription cancellation currently only closes the binding sink; it does
  not promise to interrupt inference or transport. `ask_wasm` remains a
  blocking compatibility API and is not an approved browser-main-thread
  application path.

### 2. Define one request lifecycle

Every asynchronous call has a request identity and an explicit lifecycle:

1. The host submits an owned prompt and request options to a bounded queue.
2. The executor accepts the request or returns `Busy`/`QueueFull` without
   starting hidden work.
3. A worker runs the synchronous provider call and publishes completion, error,
   or safe token events to the host-facing future/stream.
4. Completion closes the stream exactly once and releases request-local state.

Token events are delivered outside the provider/session lock. Host callbacks
must not be able to deadlock the inference worker by synchronously re-entering
the same SDK handle.

### 3. Make cancellation cooperative and safe

The implemented native asynchronous API exposes a cancellation handle or
cancellation signal:

- React Native request cancellation maps to the cooperative signal. Rust
  request handles, Dart cancellation propagation, and WASM stream cancellation
  are follow-up work; the existing Dart subscription cancellation only stops
  consumer delivery and must not be represented as inference cancellation.
- The runtime checks the signal at prefill boundaries, between decode steps,
  and at safety-guard/checkpoint boundaries. A provider that cannot stop at a
  finer boundary must still stop at the next defined boundary.
- Cancellation discards un-emitted token buffers and never surfaces a span that
  the ADR-012 safety loop could later roll back. The session returns to an
  explicit reusable terminal state; it must not remain wedged in `Prefilling`
  or `Generating`.
- Cancellation is not implemented by forcibly killing a thread. The binding
  may report a typed `Cancelled` result promptly after it detaches host event
  delivery, while the worker finishes cleanup and retains its session/worker
  reservation. A stateful handle remains `Busy` until that cleanup exits; it is
  never reused while an uncooperative backend might still mutate its session.

### 4. Serialize stateful sessions and bound concurrency

An `EdgeLlm` handle with a persistent conversation admits at most one active
turn and fails a second request with `Busy`; it does not queue hidden session
work. Reset, close, and model/session eviction are scheduling barriers and
cannot race an active request. Providers explicitly declare whether they need
this exclusive-turn reservation, so stateless cloud/relay providers may accept
concurrent calls on one handle.

Native asynchronous workers are bounded per handle (two accepted workers; a
stateful handle is tighter because its exclusive turn admits one). A stalled or
uncooperative provider can therefore exhaust only its own handle, not a global
process-wide slot that would block unrelated SDK consumers. Token delivery uses
a bounded event queue, and capacity pressure is reported as `Busy` rather than
creating an unbounded thread or event backlog.

### 5. Project the boundary idiomatically to each consumer

| Consumer | Target asynchronous contract | Current staging |
|---|---|---|
| Rust | `Future<Result<...>>` plus a cancellable token stream/request handle | Deferred; synchronous provider API remains for CLI/tests. |
| Dart/Flutter | `Future<String>` and `Stream<String>` | Deferred request-handle/backpressure work; subscription cancellation closes delivery only and does not promise inference cancellation. |
| React Native | Promise/future and cancellable async stream/callback surface | Implemented with per-handle bounded native workers; true inference-time token streaming is deferred to ADR-019. |
| Web/WASM | `Promise` and `ReadableStream` | Deferred; blocking `ask_wasm` remains a compatibility seam, not a main-thread application API. |

The binding-specific event adapters may differ, but they must preserve the
same completion, error, cancellation, ordering, and per-session serialization
semantics. ADR-019 remains the authority for when a token is eligible to be
emitted.

### 6. Measure queueing separately from inference (deferred)

The current native executor does not yet publish queue wait, prefill,
time-to-first-safe-token, decode, cancellation, or completion timing. That
executor-level instrumentation is deferred until it has a content-free metrics
sink compatible with ADR-007 and ADR-023; it must distinguish scheduler latency
from model latency without recording prompt or output content.

## Consequences

### Positive

- SDK consumers can use local inference without blocking UI or event-loop
  threads.
- Dart, React Native, Rust, and WASM consumers receive one coherent lifecycle
  for completion, streaming, errors, cancellation, and backpressure.
- Stateful sessions remain safe because scheduling, reset, and close are
  serialized explicitly rather than relying on callers to coordinate locks.
- The synchronous core stays portable and testable without coupling it to a
  particular async runtime.
- The deferred executor metrics will make queue delay and model latency
  separately measurable, including TTFT and cancellation behavior on real
  devices.

### Negative

- The SDK gains executor, queue, request-handle, and cancellation state that
  must be tested across native and WASM targets.
- A single stateful session cannot parallelize turns; consumers that need
  concurrency must create separate sessions and accept the memory cost.
- Worker lifecycle, platform event dispatch, and stream backpressure add
  binding-specific failure modes.
- A cooperative cancellation request cannot interrupt an individual backend
  kernel; stop latency is bounded by the next supported cancellation boundary.
- The public FFI error and scheduling contract requires a 0.5.0-or-later minor
  release.
- Release builds use Cargo's workspace-wide `panic = "unwind"` profile because
  Cargo cannot scope panic strategy to only the FFI cdylib. Non-FFI release
  binaries therefore accept the associated binary-size and unwind-code cost in
  exchange for containing provider and foreign-callback panics at the FFI
  boundary.

### Neutral

- Existing synchronous Rust APIs remain valid for command-line tools,
  benchmarks, and applications that already own a worker thread.
- This ADR does not change model loading, KV-cache ownership, safety policy, or
  the content of generated responses.
- Cloud providers use the same host execution boundary, but their transport
  may have an additional network cancellation mechanism.

## Implementation seams and staged acceptance criteria

- Implemented: `crates/el-ffi` owns the React Native request executor, request
  handle, cancellation state, per-handle worker capacity, and stateful-session
  serialization; the runtime resets after every mid-flight cancellation rather
  than replaying a potentially long prompt for decode rollback, intentionally
  sacrificing prefix reuse for bounded cancellation cleanup. It otherwise
  exposes explicit `Faulted` state rather than leaving an in-progress phase.
- Implemented: the React Native adapter proves non-blocking submission,
  one-active-turn serialization, per-handle capacity rejection, exactly-once
  terminal delivery, cancellation during prefill/decode, callback failures, and
  reset races.
- Deferred: `packaging/dart` request-handle/backpressure integration and the
  WASM/npm worker-safe `Promise`/`ReadableStream` surface.
- Deferred: executor-level, content-free queue wait, inference phase,
  cancellation, and completion timing metrics (ADR-023).
- Deferred: provider-level in-loop safe-token emission and its corresponding
  time-to-first-token/backpressure measurements (ADR-019 and ADR-023).

## Links

- Builds on: [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md)
  (persistent state and session ownership), [ADR-019](./ADR-019-in-loop-incremental-decoding-and-token-streaming.md)
  (safe token streaming and cancellation), [ADR-024](./ADR-024-dart-only-platform-agnostic-pub-dev-sdk.md)
  (Dart/Flutter surface), and [ADR-025](./ADR-025-react-native-expo-autolink-ready-native-distribution.md)
  (React Native distribution).
- Constrained by: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md)
  (native/WASM parity), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
  (memory budget), [ADR-007](./ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md)
  (content-free telemetry), and [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
  (safe-prefix and rollback invariants).
- Future measurement target: [ADR-023](./ADR-023-baseline-performance-instrumentation.md)
  (queue wait, TTFT, and cancellation latency).
