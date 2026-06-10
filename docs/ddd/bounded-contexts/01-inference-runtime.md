# Bounded Context 1 — Inference Runtime  ⭐ CORE

> The orchestration heart of the SDK: it owns the session lifecycle, prefill, the
> decode loop, the KV-cache, engine selection, and the public Rust/WASM interface.
> Source: PRD §"Proposed Edge-Native Pipeline", §"API/SDK Design".

## Purpose

Drive a model from a compressed prompt to committed tokens, composing the
collaborator contexts (compression, speculation, grammar, safety) under a fixed
memory plan and a chosen delegate plan, fully on-device.

## Strategic role

**Core domain.** This is where heterogeneous scheduling, KV-cache lifecycle, and
zero-allocation decoding create defensible value. Invest the most design rigor
here.

## Ubiquitous language (context-local)

`Inference Session`, `Prefill`, `Decode Loop`, `Commit`, `Runtime` (Candle),
`Build Target`, `Progressive Graph Scheduling`, `Hybrid Mode`. See
[glossary](../ubiquitous-language.md).

## Aggregates

### `InferenceSession` (Aggregate Root)
The consistency boundary for one live generation.

- **Identity:** `SessionId`.
- **Holds:** the active `Model` reference, the `KVCache`, the bound `MemoryPlan`
  handle (from context 6), the chosen `DelegatePlan` handle (from context 7),
  `SessionConfig` (format, safety mode, speculation mode, compression on/off),
  and the current `Phase` (`Initialized → Prefilling → Decoding → Completed`).
- **Invariants:**
  - A session may **not** enter `Prefilling` until a `Model` is loaded and its
    signature has been verified (conformist to context 8).
  - The KV-cache must always be a single contiguous region; pruning triggers
    compaction (delegated to context 6), never a realloc.
  - **Zero heap allocation during `Decoding`** — all working tensors come from
    the pre-allocated arena.
  - At most one decode step is in flight per session (single-writer to KV-cache).
  - `reset()` returns the session to `Initialized` and clears KV/activations
    from volatile memory.

### `Model` (Aggregate Root)
The loaded, runnable model within a session.

- **Identity:** `ModelId` (+ `ModelVersion` from context 8).
- **Holds:** `ModelFormat`, the Candle model-graph handle (built from
  GGUF/safetensors), vocabulary, and the `RuntimeKind` that owns it.
- **Invariants:** weights are `memmap2`-mapped read-only; a `Model` is immutable
  once loaded; format and engine must be compatible (`GGUF`/`safetensors`→Candle,
  `ONNX`→`tract`).

### `KVCache` (Entity within `InferenceSession`)
- **Holds:** contiguous key/value tensors, logical length, and tensor
  descriptors (offsets, not data).
- **Invariants:** physically contiguous for the GPU/Metal/WebGPU graph;
  compaction changes descriptors only.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `SessionId` | opaque id | |
| `Token` | vocabulary id (int) | atomic generation unit |
| `RuntimeKind` | `Candle` \| `Tract` (ONNX) | selects the engine + ACL |
| `ModelFormat` | `Gguf` \| `Safetensors` \| `Onnx` | |
| `DeviceTarget` | `Auto` \| `MidRange` \| `HighEnd` | request hint; resolved by context 7 |
| `SessionConfig` | { format, safetyMode, speculationMode, compress:bool, maxTokens } | immutable per session |
| `Phase` | `Initialized` \| `Prefilling` \| `Decoding` \| `Completed` | state machine |
| `GraphOp` | ACL-translated op | never a Candle/`ggml` type |
| `GenerationResult` | { token, isFinal, stopReason } | per `generate()` return |

## Domain Services

- **`DecodeLoopOrchestrator`** — runs one decode step: requests a draft set
  (context 3), runs verification on the target model, asks Grammar (context 4)
  for a `TokenMask`, asks Safety (context 5) for a `LogitAdjustment`, composes
  (mask then adjust then sample), then commits. Pure orchestration; holds no
  state beyond the session it acts on.
- **`PrefillService`** — chunks the compressed prompt (e.g. 1024 tokens),
  applies **Progressive Graph Scheduling**, and produces the initial KV-cache.
- **`RuntimeSelector`** — picks `RuntimeKind` from `ModelFormat` + `DeviceTarget`
  and binds the build target (native ARM, or `wasm32` via Wasmtime).

## Ports (interfaces this context depends on)

| Port | Provided by | Direction |
|------|-------------|-----------|
| `VerifiedModelSource` | Model Provenance (8) | inbound (CF) |
| `DelegatePlanProvider` | Hardware & Delegate (7) | inbound (C/S) |
| `MemoryPlanProvider` / `Arena` | Memory Management (6) | inbound (SK) |
| `PromptCompressor` | Prompt Compression (2) | inbound (C/S) |
| `Speculator` | Speculative Decoding (3) | inbound (C/S) |
| `GrammarMasker` | Grammar Constraint (4) | inbound (C/S) |
| `SafetySteerer` | Safety (5) | inbound (C/S) |
| `HybridRelayPort` | Local relay (Frontier) | **outbound, opt-in only** |
| `TelemetrySink` | Telemetry & Privacy (9) | outbound, event publish |

## Anti-Corruption Layer

`RuntimeAcl` wraps **Candle** (GGUF/safetensors loading, quantized kernels,
`Tensor`/`Device`, Metal backend), the optional **`tract`** ONNX path, and the
**Wasmtime** host (`Instance`/`Memory` + linear memory) on the `wasm32` target.
All inbound library structures are translated to `GraphOp`, `Model`, and arena
`Tensor` before reaching the domain — no Candle/`ggml` type crosses the boundary.

## Domain Events (published)

`SessionInitialized`, `ModelLoaded`, `PrefillCompleted`, `TokenGenerated`,
`TokenCommitted`, `GenerationCompleted`, `SessionReset`, `HybridRelayConsulted`.
See [domain-events.md](../domain-events.md).

## Public interface (Published Language / OHS)

Maps directly to PRD §API: `init(model_uri, options)`, `load_prompt(&str)`,
`generate(max_tokens)`, `commit()`, `reset()`. These are the only stable contract
with host apps, surfaced via **UniFFI** (Kotlin/Swift) on native and
**`wasm-bindgen`/`wit-bindgen`** on the web — no hand-written JNI/Obj-C.

## Notable design constraints (from PRD)

- Threading: a dedicated inference thread runs `generate()` in a loop; a second
  thread runs compression/safety; threads coordinate via Rust channels
  (`std::sync::mpsc`/`crossbeam`). Data moves via shared linear memory (WASM) or
  shared `Arc<[u8]>` buffers (native), avoiding marshalling overhead. Thread
  priorities bound to platform schedulers (Android RT
  FIFO / iOS QoS) to avoid UI jitter.
- Throughput targets: prefill >1000 t/s (high-end) / ~200 t/s (mid-range);
  decode 50–100 t/s (high-end) / 10–20 t/s (mid-range); TTFT <200 ms / <500 ms.
