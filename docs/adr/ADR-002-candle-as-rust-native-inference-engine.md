# ADR-002: Candle as the Rust-native inference engine

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: runtime, model-format, candle, rust, core

## Context

The decode loop needs an engine that delivers quantized inference, a workable
memory model, and GPU acceleration on mobile — *without* C/C++
([ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)). The
originally-chosen engines, **ExecuTorch and llama.cpp, are C++** and are
therefore out. The PRD (§"Background and Validation", §"Model Format and
Compilation") still values their *techniques* — AOT static memory planning,
quantized ARM CPU kernels — which we reproduce in Rust.

Among Rust-native engines (Candle, Burn+wgpu, ratchet, tract), **Candle** is the
most mature for LLM inference: it reads GGUF and safetensors, supports 4-bit
groupwise quantization, has CPU (SIMD/NEON), Metal, and CUDA backends, and
compiles to `wasm32`.

This choice is the basis of the
[Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md) `RuntimeAcl`.

## Decision

Adopt **Candle** (HuggingFace's pure-Rust ML framework) as the **primary and only
required** inference engine. Supported model formats: **GGUF** and
**safetensors** (read natively by Candle); **ONNX** is supported optionally via
**`tract`** (also pure Rust). Default quantization is **4-bit groupwise with
BF16 scales**.

Acceleration: **CPU via Candle's NEON/SIMD kernels everywhere** (the universal
fallback), **Metal** on Apple, and **WebGPU via `wgpu`** for portable GPU
compute. Dedicated **NPUs (NNAPI/QNN, CoreML ANE)** are *optional*, reached
through thin OS FFI (`ndk`, `objc2`) only where present and beneficial. Build
with Rust ARM target-features (`-C target-feature=+neon,+dotprod,+i8mm`, plus
`+sve2` where available). Candle sits behind `RuntimeAcl` so its tensor/types
never leak into the domain.

## Consequences

### Positive
- Single-language, memory-safe inference path that compiles to both native ARM
  and `wasm32`.
- GGUF/safetensors + 4-bit quant keep artifacts small (<300 MB on disk for a
  0.5B model) and enable `memmap2` zero-copy loading
  ([ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)).
- `tract` gives a pure-Rust ONNX option without pulling in C++ ONNX Runtime.

### Negative
- **No default NPU acceleration**: throughput leans on CPU(NEON)+GPU(Metal/
  WebGPU). Mobile-NPU peak performance from the original design is only available
  via optional FFI. Decode targets are revised down accordingly (high-end
  ~30–80 t/s rather than 50–100 t/s).
- Candle's Android GPU story is weaker than its Apple/Metal path; Android often
  runs CPU-primary unless WebGPU/NNAPI is wired up.
- Candle is younger than llama.cpp/ExecuTorch; some ops/kernels may need
  contribution or workarounds.

### Neutral
- Quantization is fixed offline; the SDK assumes pre-quantized GGUF/safetensors
  artifacts (see [Model Provenance](../ddd/bounded-contexts/08-model-provenance.md)).

## Links
- PRD: `docs/prd.md` §"Background and Validation of Claims", §"Model Format and Compilation"
- DDD: [Inference Runtime](../ddd/bounded-contexts/01-inference-runtime.md), [Hardware & Delegate](../ddd/bounded-contexts/07-hardware-delegate.md)
- Driven by: [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
- Related: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
