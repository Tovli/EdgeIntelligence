# ADR-021: Memory-mapped verified GGUF loading

- **Status**: proposed
- **Date**: 2026-06-22
- **Deciders**:
- **Tags**: runtime, performance, provenance, on-device, follow-up, P0

## Context

The improvements plan
([docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.5,
roadmap Phase 1, EPIC-3) wants memory-mapped GGUF loading to cut startup latency,
avoid copying the whole model into anonymous heap, and let the OS page unused
weights — **without** weakening the provenance gate.

Today loading is fully buffered and re-done per turn:

- `QwenEngine::from_path` opens the file, reads the GGUF header with
  `gguf_file::Content::read`, then calls `Qwen2Weights::from_gguf(content, &mut
  file, &device)`, materializing weights into resident tensors on the CPU device.
  `CandleEngine` additionally `dequantize`s `token_embd.weight` / `output.weight`
  into full `f32` tensors.
- `from_bytes` is documented as the "WASM / memory-mapped scenario," but it is a
  `std::io::Cursor` over an in-memory `&[u8]` — **not** an actual `mmap`.
- Because `QwenChatProvider::chat` rebuilds the engine each turn (ADR-018), this
  buffered load is paid **per call**.
- Provenance ordering matters: `local_load_permit` issues a `LoadPermit` through
  the ADR-006 gate, but it is currently a *trust-the-local-file* verifier (no
  cryptographic check of the GGUF bytes — see ADR-013 §Implementation status).
  The load order must remain: file → signature verification → verified permit →
  tensor/mmap construction → session.

## Decision

Add a memory-mapped model source beneath `el-engine-candle` that maps the GGUF
file and constructs tensor views over the mapping, with verification strictly
**before** any view is built.

1. **`MmapModelSource`.** A loader that `mmap`s the GGUF and hands candle tensor
   views backed by the mapping instead of heap copies. The mapping is kept alive
   for the model's entire lifetime (tensor views must not outlive it).

2. **Verify-before-map ordering (ADR-006).** Provenance verification reads/digests
   the file and issues the `LoadPermit` **before** executable tensor views are
   constructed. A corrupt or altered file fails at verification, never at first
   inference. This preserves the type-level gate: a session still cannot be built
   without a `LoadPermit`. (Replacing the trust-the-file verifier with real
   whole-artifact signature verification is tracked under ADR-006/ADR-013 and is a
   precondition for production signed assets, not introduced here.)

3. **Quantized stays mapped.** Where the GGUF is already quantized, prefer mapping
   the quantized bytes and dequantizing **on demand** (pairs with ADR-022's
   tiered KV / on-demand dequant) rather than eagerly materializing full-precision
   tensors — that is where the resident-memory win comes from. The realized saving
   is engine-dependent and recorded as a measured result, not assumed.

4. **Buffered fallback.** Targets without usable `mmap` (notably wasm32) fall back
   to the existing buffered/`from_bytes` path. The public engine API is unchanged;
   only the load strategy is selected per target.

5. **Per-OS validation.** Android, iOS, Linux, macOS, and Windows are tested
   separately — `mmap` semantics, file-locking, and page behavior differ across
   them.

## Consequences

### Positive
- Lower cold-start latency and lower peak RSS: the OS pages weights on demand
  instead of a full read-into-heap (+ dequantize) up front.
- Combined with ADR-018 (load once, share across sessions), the per-turn load cost
  disappears and the mapped pages are shared.
- Provenance is preserved exactly: verification still gates session construction.

### Negative
- `mmap` + tensor-views-over-mapping needs careful lifetime handling; the mapping
  must outlive every view (a use-after-unmap would be a soundness bug — note the
  crate is `#![forbid(unsafe_code)]`, so the mapping must come from a safe `mmap`
  abstraction).
- Real integrity (vs. trust-the-file) means hashing/verifying the whole artifact
  before mapping, which costs a one-time read — it trims the *copy*, not the
  *verify*. Honest accounting required.
- Platform variance (esp. wasm has no `mmap`; iOS file protections) means two code
  paths to maintain and test.

### Neutral
- The buffered loader remains the reference and the wasm path; this is an additive
  strategy selection.
- Tensor numerics are unchanged — mapping changes *where bytes live*, not the
  forward result; determinism (ADR-002) holds.

## Links
- Source: [docs/research/improvements-plan.md](../research/improvements-plan.md) §P0.5, EPIC-3.
- Builds on: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (GGUF/candle), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md) (load once, share).
- Constrained by: [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) (verify → permit → map ordering; real whole-artifact verification still owed), [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md) (no `unsafe` — safe mmap abstraction), [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md) (wasm has no mmap → fallback).
- Pairs with: [ADR-022](./ADR-022-two-tier-quantized-kv-cache-with-attention-aware-eviction.md) (map quantized bytes, dequantize on demand).
- Implementation seams: `crates/adapters/el-engine-candle` (`QwenEngine`/`CandleEngine` load path, new `MmapModelSource`), `crates/el-provenance` (verify-before-map ordering).
- Measured by: [ADR-023](./ADR-023-baseline-performance-instrumentation.md) (cold start, peak RSS vs. buffered).
