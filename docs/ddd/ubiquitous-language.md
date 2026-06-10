# Ubiquitous Language

> The shared vocabulary for the Edge-Native LLM SDK. Every term here means
> exactly one thing across code, docs, and conversation. Where a vendor uses a
> different word for the same concept, the vendor term is noted as an *alias to
> translate at the ACL boundary* — it must not appear in domain code.

## Core pipeline terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Inference Session** | A single live conversation/generation with its own KV-cache, memory plan, and configuration. Created by `init`, ended by `reset`. | Inference Runtime |
| **Prefill** | The phase that encodes the (compressed) prompt through the model in large chunks, producing the initial KV-cache. | Inference Runtime |
| **Decode Loop** | The iterative per-token phase: draft → verify → mask → steer → commit, repeating until EOS or max tokens. | Inference Runtime |
| **Token** | A vocabulary id produced or consumed by the model. The atomic unit of generation. | (shared) |
| **KV-Cache** | The contiguous key/value tensor block accumulated during prefill and extended each decode step. Lives only in volatile memory. | Inference Runtime |
| **Commit** | Appending an accepted token to the output and writing it into the KV-cache. | Inference Runtime |
| **Calibration** | Building a DFS token-tree from final prefill logits to align drafts to the context distribution. | Speculative Decoding |
| **Runtime / Engine** | The execution backend that runs the model graph — **Candle** (pure Rust), behind `RuntimeAcl`. | Inference Runtime |
| **Build Target** | The compiled artifact: a **native ARM** library or a **`wasm32`** module (run via Wasmtime), built from one Rust codebase, exposing `init/load_prompt/generate/commit/reset`. | Inference Runtime |
| **Time-to-First-Token (TTFT)** | Latency from request to first emitted token; a primary performance target. | Telemetry & Privacy |

## Model & format terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Model Artifact** | A signed, possibly encrypted, quantized model file on flash. | Model Provenance |
| **Model Format** | `GGUF` or `safetensors` (read natively by Candle), or `ONNX` (via the pure-Rust `tract`). | Model Provenance / Inference Runtime |
| **Quantization** | Weight precision scheme; default **4-bit groupwise with BF16 scales**. | Model Provenance |
| **Model Graph** | The runnable model graph built by Candle from the loaded weights. Translate Candle's `Tensor`/op types → domain `GraphOp` at the ACL. | Inference Runtime |
| **Model Signature** | An ED25519 signature over the artifact, verified before any load. | Model Provenance |

## Memory terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Memory Plan** | The ahead-of-time assignment of every tensor to a fixed offset in a contiguous arena. | Memory Management |
| **Arena** | The single pre-allocated contiguous buffer used for all activations; no `malloc` during inference. | Memory Management |
| **Memory Tier** | `SRAM` (hot, e.g. KV-cache) vs `DRAM` (constant weights). | Memory Management |
| **Zero-Copy Load** | `mmap`-ing constant weight pages from the model file instead of copying them. | Memory Management |
| **KV Compaction** | A pointer/descriptor shuffle (no data copy) that removes holes after pruning. | Memory Management |

## Optimization terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Prompt Compression** | Reducing prompt length 3–6× by dropping low-information tokens. | Prompt Compression |
| **Budget Controller** | The policy that assigns a token-preservation budget per segment (instructions/demos/query). | Prompt Compression |
| **Prompt Segment** | A classified span of the prompt: `Instruction`, `Demonstration`, or `Query`. | Prompt Compression |
| **Compression Ratio** | Output-tokens ÷ input-tokens after trimming (target 1/3–1/6). | Prompt Compression |
| **Speculation Mode** | `Off` \| `Draft` \| `LeverLite` — how aggressively to speculate. Default `Off`. | Speculative Decoding |
| **Draft Token** | A speculatively proposed token awaiting verification. | Speculative Decoding |
| **Token Tree** | A DFS tree of candidate continuations built at calibration. | Speculative Decoding |
| **Verification** | Running the target model to accept/reject draft tokens. | Speculative Decoding |
| **Intermediate Predictor** | A mid-layer linear head that prunes unlikely draft branches early (Lever technique). | Speculative Decoding |
| **Progressive Graph Scheduling** | Loading decoding-optimized graphs block-by-block, overlapping prefill (CoordGen technique). | Inference Runtime |

## Constraint & safety terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Grammar Ruleset** | The set of grammars (JSON schemas, tool-call shapes) a session may enforce. | Grammar Constraint |
| **Tag** | A registered marker (e.g. an opening `{`) whose appearance switches the active grammar. | Grammar Constraint |
| **TagDispatch** | The Aho–Corasick scan that detects tags and switches grammar context. | Grammar Constraint |
| **Token Mask** | The per-step boolean vector of which vocabulary tokens are currently legal. | Grammar Constraint |
| **FSM State** | A compiled grammar automaton state; may be JIT-compiled on first encounter. | Grammar Constraint |
| **Safety Mode** | `Off` \| `Lightweight` \| `SecDecoding` \| `CSD`. | Safety |
| **Logit Adjustment** | A vector subtracted from target logits to steer away from unsafe output (SecDecoding). | Safety |
| **Safety Score** | A scalar risk estimate for a token, claim, or hidden state. | Safety |
| **Claim** | A semantic span delimited by termination tokens, scored as a unit (CSD). | Safety |
| **Backtrack** | Discarding a flagged claim and resampling from before it (CSD). | Safety |

## Hardware terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Device Profile** | `MidRange` or `HighEnd` — RAM, bandwidth, and NPU TOPS class. | Hardware & Delegate |
| **Delegate** | A compute backend: `Cpu` (Candle NEON/SIMD, always available), `Metal` (Apple), `WebGpu` (`wgpu`); optionally `Nnapi`/`Qnn` or `CoreMl` reached via thin OS FFI where present. | Hardware & Delegate |
| **Delegate Plan** | The ordered partition of the graph across delegates, with CPU fallback. | Hardware & Delegate |
| **Capability Detection** | Runtime probing of available NPUs/GPUs to choose a delegate plan. | Hardware & Delegate |
| **Graph Partition** | A subgraph routed to a single delegate. | Hardware & Delegate |

## Compliance & ops terms

| Term | Definition | Bounded context |
|------|------------|-----------------|
| **Air-Gap** | Operating with zero network egress. The default and an invariant. | (cross-cutting) |
| **Hybrid Mode** | Optional, opt-in consultation of a *local-network* relay (e.g. Frontier). Never a cloud API. | Inference Runtime |
| **Telemetry Snapshot** | A content-free sample of perf counters (tokens/sec, latency, memory high-water mark). | Telemetry & Privacy |
| **Housekeeping Report** | Aggregate session stats containing no user content. | Telemetry & Privacy |

## Banned/translated library terms

These belong to the Rust crates we depend on (or to C/C++ research systems cited
only as technique lineage) and must **not** leak into domain code; wrap at the
ACL:

- Candle `Tensor` / `Device` / `QTensor` → domain `GraphOp`, `Delegate`, arena `Tensor`.
- `ggml`/`GGUF` block types, `Core-ATen op` (GGUF import lineage) → `GraphOp` / `Model Format`.
- `llguidance` parser/FSM internals → `Grammar Ruleset`, `FSM State`, `Token Mask`.
- `tract` ONNX node types (optional path) → `GraphOp`.
- SecDecoding base-vs-safety model divergence → `Logit Adjustment`.
- Wasmtime `Instance`/`Memory`, `wasm-bindgen`/UniFFI glue → kept in the host
  adapter, never in domain types.
