Below is a prioritized backlog based on **value to EdgeIntelligence’s edge-device goals**, not ruvLLM’s marketing order. I weighted memory reduction, startup time, latency, mobile usefulness, architectural fit, and implementation risk.

The performance figures are ruvLLM project claims and should be independently benchmarked inside EdgeIntelligence before becoming acceptance criteria.

## P0 — Highest-value foundation

### 1. Persistent model instances and stateful inference sessions

**Description:** Keep model weights loaded across requests and maintain a session-specific inference state, including conversation tokens and KV-cache references. `RuvLLMEngine` includes session lifecycle management, persistent session indexing, and KV-cache references for multi-turn interaction. ([GitHub][1])

**Value to EdgeIntelligence:** This directly addresses EdgeIntelligence’s largest identified orchestration issue: it currently needs to load weights once, reset only the KV cache when appropriate, and reuse KV state across conversation turns instead of re-prefilling the entire history. ([GitHub][2])

**Implementation direction:**

* Add a `StatefulLlmProvider` or `InferenceSessionHandle`.
* Separate model lifecycle from conversation lifecycle.
* Maintain one shared model instance with multiple isolated sessions.
* Give every session explicit `reset`, `close`, and eviction operations.
* Keep `el-runtime` as the session authority; use ruvLLM’s session structures as an implementation reference or optional adapter.

**First acceptance criteria:**

* Model weights load once per provider instance.
* A second turn does not reload the model.
* Unchanged conversation prefixes are not re-prefilled.
* Session memory can be measured and explicitly released.

---

### 2. Incremental decoding and real token streaming

**Description:** ruvLLM exposes incremental token-generation concepts, token streams, stream events, and a `decode_step()` design for advancing generation one token at a time. Its Node interface also exposes generation as an asynchronous token stream. ([GitHub][3])

**Value to EdgeIntelligence:** EdgeIntelligence currently identifies true in-loop streaming as a major opportunity: the first token should be emitted after model load, prefill, and one decode step—not after the full response has completed. ([GitHub][2])

**Implementation direction:**

```rust
pub trait StreamingLlmProvider {
    type Stream: Iterator<Item = Result<TokenEvent, SdkError>>;

    fn chat_stream(
        &self,
        session: &SessionId,
        request: ChatRequest,
    ) -> Result<Self::Stream, SdkError>;
}
```

The decode pipeline should remain:

```text
model logits
→ grammar mask
→ safety adjustment
→ sampling
→ KV commit
→ emit token
```

**First acceptance criteria:**

* Time-to-first-token is independently measured.
* Cancellation stops generation and releases temporary buffers.
* Streaming works through Rust, UniFFI, Flutter, and Web bindings.
* Grammar and safety still run before every emitted token.

---

### 3. Two-tier and quantized KV-cache storage

**Description:** ruvLLM implements a two-tier KV cache in which recent tokens remain at higher precision while older tokens migrate to a quantized tier. It also contains TurboQuant 2–4-bit KV compression and reports configurations for 2-, 3-, 4-, and 8-bit storage. ([GitHub][4])

**Value to EdgeIntelligence:** KV memory grows with context length and often becomes the main mutable memory cost after model weights. Quantizing old KV entries could enable longer conversations or smaller memory budgets on phones and embedded devices.

**Implementation direction:**

* Extend `el-memory` from KV descriptors to an actual pluggable KV storage policy.
* Preserve recent tokens in FP16/BF16.
* Quantize older blocks to Q8 first, then evaluate Q4 and Q3.
* Dequantize only blocks required by attention.
* Add per-session KV memory limits.

Suggested interface:

```rust
pub trait KvCachePolicy {
    fn append(&mut self, layer: usize, key: TensorView, value: TensorView)
        -> Result<(), MemoryError>;

    fn view_for_attention(
        &self,
        layer: usize,
        range: TokenRange,
    ) -> Result<KvView<'_>, MemoryError>;

    fn memory_usage(&self) -> KvMemoryStats;
}
```

**First acceptance criteria:**

* Q8 cache passes deterministic quality regression tests.
* Peak KV memory decreases by a defined percentage.
* Quantization overhead does not erase decode-time gains.
* The uncompressed implementation remains available as a reference.

---

### 4. Intelligent KV eviction for long contexts

**Description:** ruvLLM includes H2O and PyramidKV-style eviction approaches that retain tokens considered important based on attention behavior rather than discarding tokens solely by age. It also supports sliding-window and tiered cache configurations. ([GitHub][4])

**Value to EdgeIntelligence:** A fixed-size context currently forces a choice between high memory use and blunt truncation. Attention-aware eviction can preserve system instructions, important facts, and salient earlier messages while removing less valuable cache entries.

**Implementation direction:**

* Start with deterministic sliding-window eviction.
* Add pinned ranges for system prompts and tool definitions.
* Implement H2O-style scoring behind an experimental feature.
* Expose eviction events as content-free telemetry.
* Do not allow eviction to remove tokens required by grammar or tool-call state.

**First acceptance criteria:**

* Hard memory budgets are never exceeded.
* System prompt tokens can be pinned.
* Long-context benchmark quality exceeds simple FIFO truncation.
* Eviction decisions are reproducible in deterministic mode.

---

### 5. Memory-mapped GGUF model loading

**Description:** ruvLLM supports GGUF loading with memory mapping, quantized model formats, metadata inspection, and optional checksum validation. Its documentation presents memory mapping as a way to improve loading speed and reduce resident memory pressure. ([GitHub][4])

**Value to EdgeIntelligence:** This can reduce model startup latency, avoid copying the entire model into anonymous heap memory, and allow the operating system to page unused weights.

**Implementation direction:**

* Add an `MmapModelSource` beneath `el-engine-candle`.
* Preserve the existing signature verification gate.
* Verify the file before constructing executable tensor views.
* Keep the mapped file alive for the model’s full lifetime.
* Add fallback buffered loading for unsupported targets.

The load order must remain:

```text
model file
→ signature verification
→ verified load permit
→ mmap/tensor construction
→ inference session
```

EdgeIntelligence requires verified provenance before a session may be constructed. ([GitHub][2])

**First acceptance criteria:**

* No mmap occurs before provenance validation.
* Startup time and peak RSS are compared with buffered loading.
* Corrupt or altered model files fail before inference.
* Android, iOS, Linux, macOS, and Windows behavior is tested separately.

---

## P1 — Major performance and product capabilities

### 6. Hardware capability detection and backend selection

**Description:** ruvLLM contains platform and capability-detection types for CPU features, GPU backends, compute capabilities, and inference configuration. It offers CPU, Metal, CUDA, Apple Neural Engine/Core ML, and hybrid execution paths. ([GitHub][1])

**Value to EdgeIntelligence:** A single static backend configuration is unlikely to be optimal across iPhone, Apple Silicon, Android ARM, desktop CUDA, and WebAssembly. Capability detection allows the SDK to select the best supported path while keeping one public API.

**Implementation direction:**

```rust
pub struct DeviceCapabilities {
    pub cpu_threads: usize,
    pub simd: SimdLevel,
    pub available_memory_bytes: u64,
    pub metal: bool,
    pub core_ml: bool,
    pub cuda: bool,
    pub wasm_simd: bool,
}
```

Then derive an explicit execution plan:

```rust
pub struct ExecutionPlan {
    pub backend: BackendKind,
    pub model_quantization: QuantizationKind,
    pub kv_policy: KvPolicyKind,
    pub thread_count: usize,
    pub context_limit: usize,
}
```

**First acceptance criteria:**

* Selection is deterministic and inspectable.
* Applications can override every automatic decision.
* Unsupported accelerators fail over cleanly.
* No runtime network access is introduced.

---

### 7. Flash Attention and memory-efficient attention kernels

**Description:** ruvLLM includes Flash Attention 2 concepts with block sizing and online softmax, as well as optimized Metal kernels. The project describes Flash Attention as reducing attention memory complexity and improving throughput. ([GitHub][4])

**Value to EdgeIntelligence:** Attention cost becomes significant as prompt length grows. A memory-efficient attention implementation can reduce temporary allocations and improve prefill speed, particularly for longer prompts.

**Implementation direction:**

* Add an attention-kernel port beneath `InferenceEngine`.
* Keep a simple reference implementation.
* Add tiled CPU/SIMD implementation.
* Add Metal implementation for supported Apple targets.
* Validate numerical differences against the reference kernel.

**First acceptance criteria:**

* No per-token heap allocation in the optimized path.
* Output error remains within an agreed tolerance.
* Prefill latency improves for representative context lengths.
* Kernel selection remains target- and feature-gated.

---

### 8. Batched prefill

**Description:** ruvLLM’s serving components support batched requests and separate prefill/decode scheduling. Although continuous batching mainly targets concurrent serving, the same separation can be used to process a single prompt as a matrix rather than one forward pass per input token. ([GitHub][4])

**Value to EdgeIntelligence:** EdgeIntelligence’s own benchmark identifies token-by-token prefill as a removable orchestration cost and recommends feeding the prompt in one `(1, prompt_len)` forward operation. ([GitHub][2])

**Implementation direction:**

* Extend `InferenceEngine` with `prefill_batch`.
* Keep `decode_step` separate from prefill.
* Produce the same final KV state as sequential prefill.
* Add chunked prefill for devices that cannot fit the full prompt batch.

**First acceptance criteria:**

* Batched and sequential prefill produce equivalent logits/KV state.
* Prompt processing latency falls substantially.
* Chunk size can be selected from the memory plan.
* No change to grammar or safety semantics during decoding.

---

### 9. Speculative decoding

**Description:** ruvLLM includes speculative decoding in which a smaller draft model proposes several tokens and the target model verifies them together. Its API records acceptance rate and realized speedup. ([GitHub][4])

**Value to EdgeIntelligence:** This can materially improve tokens per second when a small draft model closely predicts the target model. It is particularly useful when target-model compute dominates runtime.

**Implementation direction:**

* Add `DraftEngine` and `SpeculativeDecoder` abstractions.
* Verify candidate tokens through the existing grammar and safety pipeline.
* Load the draft model only where memory permits.
* Support self-speculation or early-exit speculation later.

**Important limitation:** Two model instances may increase footprint enough to make this unsuitable for smaller mobile devices.

**First acceptance criteria:**

* Generated output distribution remains correct.
* Safety and grammar checks cannot be skipped for accepted draft tokens.
* Report acceptance rate, extra memory, and actual speedup.
* Automatically disable speculation when acceptance is poor.

---

### 10. LoRA adapter loading and hot swapping

**Description:** ruvLLM contains an `AdapterManager`, LoRA adapter registry/pool, task-specific adapters, hot swapping, and several adapter-composition strategies. ([GitHub][4])

**Value to EdgeIntelligence:** One base model could support different products or tasks without shipping several full models. Examples include medical terminology, coding, customer service, language specialization, or organization-specific behavior.

**Implementation direction:**

* Create an `AdapterProvider` port.
* Associate adapters with verified manifests.
* Apply adapters at session creation or request boundaries.
* Cache only a small number of active adapters.
* Keep the base model immutable.

Suggested API:

```rust
pub trait AdapterProvider {
    fn load_verified(
        &self,
        permit: AdapterLoadPermit,
    ) -> Result<AdapterId, AdapterError>;

    fn activate(
        &self,
        session: &SessionId,
        adapter: &AdapterId,
    ) -> Result<(), AdapterError>;
}
```

**First acceptance criteria:**

* Every adapter is signed and tied to a compatible base-model hash.
* Adapter switching does not reload the base model.
* Adapter memory is bounded and observable.
* Deterministic mode pins an exact adapter version.

---

### 11. Semantic runtime-policy selection

**Description:** `RuvLLMEngine` includes a RuVector-backed policy store. It can semantically retrieve policies such as quantization settings and routing rules based on a request-context embedding. ([GitHub][1])

**Value to EdgeIntelligence:** Different workloads may need different execution strategies. A short classification request might use a tiny model and aggressive quantization, while a structured generation task might use stronger grammar constraints and a larger context allocation.

Potential policy decisions:

* Model selection
* KV compression level
* Context limit
* Adapter selection
* Local backend selection
* Speculative decoding enablement
* Safety tier
* Cloud fallback eligibility

**Implementation direction:**

* Begin with static rule-based policies.
* Define a serializable, versioned `ExecutionPolicy`.
* Add semantic retrieval only after deterministic rules are stable.
* Require applications to supply or explicitly enable an embedding provider.
* Record the selected policy ID in content-free telemetry.

**First acceptance criteria:**

* Policies cannot bypass safety, provenance, or air-gap requirements.
* Deterministic mode pins policy versions.
* Rules have a fallback when embeddings are unavailable.
* Policy lookup overhead is measured separately.

---

### 12. Structured witness records and searchable audit history

**Description:** ruvLLM includes a witness log for recording latency breakdowns, routing decisions, and quality scores, with optional HNSW semantic search over records. ([GitHub][1])

**Value to EdgeIntelligence:** This can improve diagnosis of poor performance and routing decisions, especially across heterogeneous devices. EdgeIntelligence already has content-free telemetry, so the useful part is a richer structured record—not necessarily prompt indexing.

**Implementation direction:**

* Extend `el-telemetry` with an optional local execution journal.
* Store IDs, numeric metrics, configuration hashes, and outcomes.
* Do not store prompts, responses, embeddings, or user identifiers by default.
* Make retention and deletion explicit.
* Add semantic search only in a separate opt-in privacy mode.

**First acceptance criteria:**

* Default witness events remain content-free.
* Storage has a hard size and retention limit.
* Records can be fully deleted.
* Sensitive modes require explicit host configuration.

---

## P2 — Differentiating intelligence features

### 13. Evaluation harness and ablation testing

**Description:** ruvLLM includes an evaluation harness with quality metrics and multiple ablation modes, allowing comparison of baseline inference, retrieval, adapters, and full adaptive configurations. ([GitHub][3])

**Value to EdgeIntelligence:** This is essential for determining whether optimizations actually help. EdgeIntelligence already has runtime and clinical/safety benchmark harnesses; adding standardized ablation support would make feature decisions evidence-based. ([GitHub][2])

**Implementation direction:**

Define feature profiles:

```text
baseline
+ persistent sessions
+ batched prefill
+ quantized KV
+ speculative decoding
+ adapters
+ semantic policy routing
+ adaptive learning
```

Measure:

* Binary size
* Cold startup
* Warm startup
* Peak RSS
* Time to first token
* Decode tokens per second
* Energy consumption
* Quality/safety scores
* Determinism
* Storage growth

**First acceptance criteria:**

* Every major optimization ships with an ablation benchmark.
* Performance claims include device, model, build flags, and context size.
* Safety/quality regression thresholds block merges.
* Benchmark outputs are machine-readable for CI.

---

### 14. SONA three-tier adaptive learning

**Description:** ruvLLM’s SONA system provides instant, background, and deeper learning loops. The instant path uses per-request adaptation, while later loops consolidate accumulated feedback. ([GitHub][4])

**Value to EdgeIntelligence:** This could allow on-device personalization and automatic optimization based on user feedback or measured outcomes, without sending private data to a server.

Potential uses:

* Learn preferred response style
* Select the best adapter for a task
* Tune routing thresholds
* Adjust generation settings
* Improve repeated workflows

**Implementation direction:**

* Do not begin with weight adaptation.
* Start by learning bounded policy values.
* Require explicit feedback rather than treating every response as successful.
* Keep learning state separate from signed model assets.
* Add export, reset, and deletion APIs.
* Disable learning in deterministic and regulated modes.

**First acceptance criteria:**

* Learning is opt-in.
* Users can inspect and delete learned state.
* Adaptation cannot modify safety boundaries.
* Regression tests prove that deterministic mode remains unchanged.
* Learned policy rollback is supported.

---

### 15. MicroLoRA per-request adaptation

**Description:** ruvLLM exposes low-rank per-request adaptation through MicroLoRA and related configuration and feedback types. The project positions it as lightweight personalization without full model retraining. ([GitHub][4])

**Value to EdgeIntelligence:** It could personalize a shared base model to a user or temporary task while keeping the base weights unchanged.

**Implementation direction:**

* Treat MicroLoRA as a later experimental extension of the adapter system.
* Apply updates only from trusted feedback.
* Bound rank, memory, update magnitude, and lifetime.
* Separate temporary session adapters from persistent user adapters.
* Sign or integrity-protect persisted adapter state.

**First acceptance criteria:**

* Maximum memory and latency overhead are enforced.
* Bad feedback cannot permanently corrupt the base behavior.
* Adaptation can be rolled back atomically.
* Safety evaluation runs before promoting an adapted state.

---

### 16. Quantized semantic-memory and embedding storage

**Description:** ruvLLM includes a TurboQuant embedding store that keeps vectors in compressed form and performs similarity scoring without fully decompressing every stored vector. ([GitHub][4])

**Value to EdgeIntelligence:** This could support compact local RAG, semantic session recall, policy lookup, or tool-result caching without requiring a large full-precision vector database.

**Implementation direction:**

* Introduce a separate optional `el-semantic-memory` crate.
* Do not make vector storage part of the core inference binary.
* Encrypt persisted stores where platform facilities allow.
* Use namespace and retention controls.
* Benchmark recall degradation at each quantization level.

**First acceptance criteria:**

* Feature is fully optional.
* No embedding model is silently loaded.
* Memory savings and retrieval recall are reported together.
* Data can be partitioned and deleted by application/user namespace.

---

### 17. Sparse attention for very long contexts

**Description:** ruvLLM includes a sparse-attention kernel based on local windows, logarithmic strides, and landmarks, together with incremental decoding support. It is presented as a small, dependency-light component for embedded and WebAssembly targets. ([GitHub][4])

**Value to EdgeIntelligence:** Sparse attention could allow much longer contexts on constrained hardware, especially where dense attention is impractical.

**Implementation direction:**

* Integrate as a separate attention-kernel experiment.
* Limit initial support to compatible architectures.
* Compare against sliding-window attention.
* Require model-specific validation because changing the attention pattern may reduce quality if the model was not trained for it.

**First acceptance criteria:**

* Context-length gains are demonstrated on edge hardware.
* Quality is compared against dense attention.
* Kernel remains optional and has no effect on unsupported models.
* Incremental decoding preserves bounded memory use.

---

## P3 — Valuable mainly for servers or specialized products

### 18. PagedAttention

**Description:** ruvLLM provides page-table-based KV management and describes a mistral-rs backend using PagedAttention and prefix caching for higher concurrency and GPU-memory utilization. ([GitHub][4])

**Value to EdgeIntelligence:** This is highly useful when one process serves many simultaneous sessions. It is less valuable for a single-user phone application, where its additional machinery may outweigh the gains.

**Implementation direction:**

* Target desktop, gateway, or local-network server deployments.
* Place it behind a separate `server` feature.
* Reuse the same `LlmProvider` API.
* Do not include it in default mobile artifacts.

---

### 19. Continuous batching

**Description:** ruvLLM includes dynamic request scheduling, token budgets, prefill/decode task separation, preemption, and batch-utilization statistics. ([GitHub][4])

**Value to EdgeIntelligence:** It improves aggregate throughput when multiple clients share one model. It offers limited benefit in a normal single-user mobile application.

**Implementation direction:**

* Add only to a server-oriented provider.
* Keep request isolation and cancellation explicit.
* Ensure batching never mixes grammar, safety, or adapter state.
* Measure tail latency as well as aggregate throughput.

---

### 20. Hugging Face model acquisition and distribution

**Description:** ruvLLM includes model download/upload and model-registry components for Hugging Face Hub integration. ([GitHub][4])

**Value to EdgeIntelligence:** It could simplify development, model provisioning, and cache management.

**Why it ranks last:** EdgeIntelligence is air-gapped by default, uses explicit opt-in egress, and requires signed model provenance. Automatic downloads conflict with these defaults unless carefully isolated. ([GitHub][2])

**Implementation direction:**

* Build it as a separate development/provisioning CLI.
* Never include network download code in the default runtime.
* Download to a staging path.
* Verify signature and compatibility before moving into the trusted model store.
* Keep production devices capable of operating entirely offline.

---

# Recommended implementation roadmap

## Phase 1 — Fix the existing hot path

Implement these without introducing the full `RuvLLMEngine`:

1. Persistent model instance
2. Stateful sessions
3. Batched prefill
4. Real token streaming
5. Memory-mapped model loading
6. Baseline performance instrumentation

This phase directly addresses EdgeIntelligence’s measured bottlenecks and carries the lowest architectural risk.

## Phase 2 — Reduce runtime memory

1. Q8 two-tier KV cache
2. Q4/Q3 experimental KV cache
3. Sliding-window eviction
4. H2O/PyramidKV-style eviction
5. Hardware capability detection
6. Flash/memory-efficient attention

This phase should establish device-specific memory budgets and quality thresholds.

## Phase 3 — Increase generation speed

1. Optimized CPU/SIMD kernels
2. Metal acceleration
3. Batched/chunked prefill tuning
4. Speculative decoding
5. Sparse attention experiments

Every item should be independently switchable for ablation testing.

## Phase 4 — Add specialization

1. Signed LoRA adapter format
2. Adapter loading and lifecycle
3. Hot swapping
4. Adapter composition
5. Task-based adapter selection

This introduces strong product value without making the base model adaptive.

## Phase 5 — Add the ruvLLM intelligence layer

1. Versioned execution-policy type
2. Static policy engine
3. Optional semantic policy store
4. Privacy-safe witness journal
5. Quantized semantic-memory store
6. Session indexing

At this point, wrapping selected parts of `RuvLLMEngine` becomes more appropriate.

## Phase 6 — Adaptive learning

1. Explicit feedback API
2. Learned routing thresholds
3. Temporary MicroLoRA
4. Persistent user adapters
5. SONA background consolidation
6. Deep optimization only in controlled environments

Adaptive behavior should remain opt-in and excluded from deterministic builds.

# Recommended initial GitHub epics

```text
EPIC-1: Stateful local inference
EPIC-2: Streaming and batched prefill
EPIC-3: Memory-mapped verified model loading
EPIC-4: Quantized and bounded KV cache
EPIC-5: Hardware-aware inference kernels
EPIC-6: Speculative decoding
EPIC-7: Signed LoRA adapter lifecycle
EPIC-8: Execution policy and routing
EPIC-9: Privacy-safe witness and evaluation framework
EPIC-10: Optional semantic memory
EPIC-11: Opt-in adaptive learning
EPIC-12: Multi-user serving backend
```

The first release milestone should stop after **EPIC-4**. Those features produce the clearest improvement to EdgeIntelligence’s latency and footprint while preserving its existing provenance, safety, privacy, and deterministic-runtime architecture.

[1]: https://raw.githubusercontent.com/ruvnet/RuVector/main/crates/ruvllm/src/lib.rs "raw.githubusercontent.com"
[2]: https://github.com/Tovli/EdgeIntelligence "GitHub - Tovli/EdgeIntelligence · GitHub"
[3]: https://github.com/ruvnet/RuVector/tree/main/npm/packages/ruvllm "RuVector/npm/packages/ruvllm at main · ruvnet/RuVector · GitHub"
[4]: https://github.com/ruvnet/RuVector/tree/main/crates/ruvllm "RuVector/crates/ruvllm at main · ruvnet/RuVector · GitHub"
