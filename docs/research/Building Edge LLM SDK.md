**Prompt**: Research how to build pipeline/flow that utilize local llm models and optimize the prompt and response to achieve more accurate results. Find SOTA solutions and explore latest peer review academic work.

The goal is to create sdk to use in an edge device like smartphone

**Response**:
# **Technical Design of an Edge-Native LLM SDK Pipeline: Optimizing Execution Runtimes, Prompt Constraints, Speculative Verification, and Structured Generation on Smartphone Hardware**

Deploying large language models (LLMs) directly on resource-constrained smartphone hardware represents a paradigm shift away from cloud-based machine learning serving. This shift is motivated by the demands for strict user data privacy, guaranteed offline operation, reduced latency, and eliminated server-side API operational expenses.1 To achieve these goals, an edge-native Software Development Kit (SDK) must resolve the severe physical limitations of smartphone platforms.

## **Core Hardware Bottlenecks and SDK Optimization Requirements**

Modern smartphone System-on-Chips (SoCs) possess highly capable Neural Processing Units (NPUs) and Graphics Processing Units (GPUs).4 For example, the Apple A19 Pro Neural Engine delivers approximately 35 Tera Operations Per Second (TOPS), while the Qualcomm Snapdragon 8 Elite Gen 5 offers up to 60 TOPS.5 However, these processing capabilities are severely throttled by mobile memory architectures.5
While datacenter accelerators feature memory bandwidths ranging from 2 to 3 Terabytes per second (TB/s), modern smartphone SoCs operate within a constrained envelope of 50 to 90 Gigabytes per second (GB/s).5 This 30x to 50x bandwidth deficit creates a profound bottleneck during the autoregressive decoding phase of language model inference.5 During decoding, the entire weight tensor of the model must be loaded from physical memory to the processor core to generate a single token, leaving high-performance tensor cores idle while waiting for memory transfers.5
An edge-native SDK must therefore optimize the entire computational pipeline, from prompt ingestion to final token output. This report details the design of a unified, high-performance edge execution pipeline, synthesizing state-of-the-art (SOTA) research across model compilation, dynamic context pruning, hardware-cooperative speculative decoding, grammar-constrained generation, and real-time safety enforcement.

## **Co-Design and Performance Evaluation of Mobile Runtimes**

The foundation of the edge SDK is the selection and configuration of a low-level execution engine. The engine must compile model graphs, manage memory, and map operators directly to the heterogeneous compute units of the device.

| Runtime Framework | Model Support and Portability | Memory Allocation Strategy | Accelerator Delegation | Key Engineering Trade-offs |
| :---- | :---- | :---- | :---- | :---- |
| **ExecuTorch** | PyTorch-exclusive via torch.export; generates unified .pte binaries 4 | Static ahead-of-time (AOT) memory planning; zero-heap runtime 4 | Pluggable Delegates (Apple CoreML, Qualcomm QNN, Arm Vulkan/XNNPACK) 4 | Minimal runtime binary footprint (KB core); steep learning curve; strict export compliance 6 |
| **ONNX Runtime Mobile** | Universal format; accepts PyTorch, TensorFlow, and Scikit-Learn 6 | Dynamic allocation with execution-provider-specific optimizations 6 | Execution Providers (CoreML, NNAPI, QNN, DirectML) 6 | Broad model compatibility; heavier runtime footprint; potential semantic conversion gaps 4 |
| **llama.cpp** | Focused on GGUF format; highly optimized CPU implementations 6 | Contiguous buffer pooling; direct memory mapping (mmap) 9 | GGML Backend supporting Armv9 KleidiAI, SVE2, and SME 9 | Ultra-fast single-batch CPU execution; lacks sophisticated multi-accelerator partitioning 6 |
| **Google MediaPipe** | Curated task pipelines and Gemma models via Tasks API 6 | Solution-specific managed pipeline buffers 6 | GPU/NPU acceleration via Google LiteRT (formerly TensorFlow Lite) 4 | Extremely fast integration for standard vision/text tasks; highly rigid; limited custom LLM flexibility 6 |
| **Cactus SDK** | Unified API wrapper over native engines 6 | Managed abstract buffers with automatic host recycling 6 | Native framework delegates with automated fallback 6 | Simplest developer experience; unique Hybrid Cloud \+ On-Device routing; proprietary 6 |

### **ExecuTorch Memory and Delegate Architecture**

ExecuTorch eliminates the execution overhead associated with dynamic interpreters by utilizing ahead-of-time (AOT) graph-level compilation.4 Models are exported into an intermediate representation called Edge Dialect, which converts scalar types into tensors and flattens operators into the standardized Core ATen operator set.11 This standardization minimizes the operator surface that third-party hardware compilers must implement, enabling robust execution across diverse mobile chipsets.11
Memory optimization within ExecuTorch is handled by the MemoryManager and MemoryAllocator abstractions.7 During AOT compilation, a static memory planner maps the lifespans of all intermediate tensors and packs them into a contiguous, user-allocated buffer.4 This approach eliminates the need for dynamic heap allocations (malloc or new) during the inference loop, preventing memory fragmentation and potential application crashes on mobile operating systems.7
ExecuTorch supports memory hierarchy planning, enabling developers to map high-frequency, mutable intermediate tensors (such as Key-Value caches) to fast on-chip Static Random-Access Memory (SRAM), while constant weights are stored in slower Dynamic Random-Access Memory (DRAM).7 Constant weights are memory-mapped directly from the compiled .pte file using the DataLoader interface, avoiding redundant copying and minimizing the system memory footprint.7
To leverage mobile NPUs, ExecuTorch uses a delegate system that partitions the computational graph.4 Subgraphs containing compatible operators are delegated to hardware-specific compilation binaries (e.g., Qualcomm QNN or Apple CoreML), while incompatible operators fall back to ExecuTorch's portable C++ operator library.4
On Arm Cortex-A architectures, ExecuTorch integrates the Arm KleidiAI library via the XNNPACK delegate.10 This integration provides highly optimized kernel routines for quantized models executing on Armv9 CPUs with the i8mm ISA extension.10 On consumer mobile hardware, this co-designed software stack achieves speeds exceeding 350 tokens per second during the prefill stage for 4-bit block-quantized Llama models.10

### **llama.cpp and Snapdragon Compilation Architecture**

For deployments targeting CPU-bound execution, llama.cpp remains a highly optimized framework.9 It achieves high efficiency by utilizing direct C/C++ implementations, custom quantization schemes, and assembly-level hardware instructions.8
When targeting ARMv9-compatible mobile processors (such as the Snapdragon 8 Gen 5), compilation flags must be explicitly configured to leverage KleidiAI routines and specialized vector extensions.9 The table below outlines the compilation variables required to compile llama.cpp for modern Android platforms using the Android NDK.9

| Compilation Parameter | Value Configuration | Target Functionality |
| :---- | :---- | :---- |
| **CMAKE\_TOOLCHAIN\_FILE** | $NDK\_PATH/build/cmake/android.toolchain.cmake | Configures the cross-compilation environment for Android platforms 9 |
| **ANDROID\_ABI** | arm64-v8a | Restricts the target architecture to 64-bit ARM mobile processors 9 |
| **ANDROID\_PLATFORM** | android-29 | Sets the minimum Android API level (Android 10\) to balance compatibility and features 9 |
| **GGML\_CPU\_KLEIDIAI** | ON | Integrates Arm KleidiAI optimized mathematical kernel routines 9 |
| **GGML\_SYSTEM\_ARCH** | ARM | Directs the build system to generate ARM-specific instruction sets 9 |
| **GGML\_CPU\_AARCH64** | ON | Enables 64-bit execution optimizations for ARM architectures 9 |
| **GGML\_CPU\_ARM\_ARCH** | armv9.2-a+sve2+sme+dotprod+i8mm | Targets ARMv9 extensions, including Scalable Vector Extension (SVE2), Scalable Matrix Extension (SME), dot-product instructions, and 8-bit integer matrix multiplication 9 |
| **CMAKE\_C\_FLAGS / CMAKE\_CXX\_FLAGS** | \-march=armv9.2-a+sve2+sme+dotprod+i8mm | Passes machine-level architecture parameters to the compiler to generate optimized machine code 9 |

## **SOTA On-Device Foundation Model Architectures**

In addition to runtime optimizations, deploying language models on edge devices requires utilizing foundation models specifically co-designed for mobile hardware constraints.14 Recent peer-reviewed academic work has introduced architectural designs that maximize performance-per-parameter under tight latency constraints.14

### **MobileLLM-Flash**

The MobileLLM-Flash family of foundation models (available in 350M, 650M, and 1.4B parameter variants) is designed using a hardware-in-the-loop architecture search under strict mobile latency constraints.14 Rather than relying on specialized attention mechanisms that require custom hardware kernels, MobileLLM-Flash uses standard attention mechanisms combined with attention skipping for long-context acceleration.14
By treating candidate architectures as pruned variants of a larger pretrained backbone with inherited weights, the search space jointly optimizes layer dimensions, depth, and attention patterns.14 This optimization yields models compatible with standard mobile runtimes, such as ExecuTorch, delivering up to a 1.8x speedup in prefill latency and a 1.6x speedup in decoding throughput on mobile CPUs while supporting context windows up to 8k tokens.14

### **MobileLLM-Pro**

The MobileLLM-Pro architecture addresses the challenge of running highly capable 1-billion-parameter models on mobile hardware while supporting context windows up to 128,000 tokens.15 To achieve this capability without exceeding memory limits, the model integrates two main innovations 15:

* **Implicit Positional Distillation**: This technique transfers long-context capabilities from a larger teacher model to the compact student model during training, minimizing performance degradation at extended sequence lengths.15
* **Specialist Model Merging**: This framework merges multiple domain-specific expert models into a single, compact foundation model without increasing overall parameter size.15

Furthermore, MobileLLM-Pro shows minimal performance degradation when subjected to 4-bit quantization, allowing it to maintain reasoning and comprehension capabilities within a small physical memory footprint.15

## **Context Pruning and Prompt Compression Frameworks**

To further reduce latency and minimize the memory footprint of the Key-Value (KV) cache, prompt compression is employed as a key pre-processing step.16 Prompt compression is defined as mapping an original input prompt of length to a compressed prompt of length , minimizing semantic divergence under a fixed budget constraint.16

In soft compression paradigms, segments of the prompt are mapped to continuous token vectors via a frozen encoder and a trainable bridge projection 16:

The model is optimized using a joint reconstruction and alignment loss 16:

For mobile pipelines, hard compression is preferred because it produces standard text tokens that are fully compatible with black-box and localized edge-native LLMs.17

### **The LLMLingua Family**

The LLMLingua framework uses a small, well-trained auxiliary language model (such as GPT-2 Small or a 1B-parameter mobile model) to compute the perplexity and conditional probabilities of sentences and individual tokens within a long prompt.17 Since natural language contains significant redundancy, tokens that contribute minimally to the overall information density have low perplexity and can be safely pruned.17

Raw Prompt \---\> \---\> Coarse-Grained Sentence Pruning (PPL-based)
                       |
                       v
                \---\> Fine-Grained Segment Compression
                       |
                       v
                \---\> Target LLM Semantic Mapping \---\> Compressed Prompt

1. **Budget Controller**: This module dynamically distributes the target compression ratio across different segments of the prompt.18 It prioritizes instructions and final questions, forcing higher compression rates onto intermediate few-shot demonstrations.20
2. **Iterative Token-Level Compression**: Simple token pruning ignores the changing conditional probabilities of downstream tokens as upstream tokens are removed.18 LLMLingua resolves this by dividing the prompt into segments and iteratively updating conditional probabilities, preserving linguistic coherence and preventing critical context loss.18
3. **Distribution Alignment**: Small compression models exhibit semantic gaps when compared to massive target black-box or local edge models.17 LLMLingua introduces an alignment mechanism that forces the perplexity estimation of the small model to closely match the probability distribution of the target model, preserving reasoning and in-context learning capabilities up to a 20x compression ratio with only a 1.5% drop in performance.17

| Prompt Compression Engine | Primary Compression Mechanism | Optimization Focus | Speed and Throughput Characteristics |
| :---- | :---- | :---- | :---- |
| **LLMLingua** | Iterative token-level perplexity estimation and distribution alignment 17 | Retains in-context learning and multi-step reasoning capabilities 17 | Moderate latency; requires multiple forward passes of the small model 17 |
| **LongLLMLingua** | Query-aware token routing and key information reorganization 18 | Long-context retrieval-augmented generation (RAG) tasks 18 | Reduces context size up to 4x while improving key information retrieval by 17.1% 19 |
| **LLMLingua-2** | Token classification using a bidirectional BERT encoder 19 | Task-agnostic, low-latency, real-time edge processing 19 | **3x to 6x faster** than original LLMLingua, making it ideal for edge devices 19 |

LLMLingua-2 is the most suitable compression engine for mobile pipelines.19 By formulating compression as a binary token classification task, it determines which tokens to preserve in a single forward pass.19 This approach eliminates the computational overhead of iterative perplexity calculations, allowing on-device prompt optimization to run in real time.19

## **Heterogeneous and Speculative Decoding Systems**

Speculative decoding is an optimization technique that accelerates autoregressive generation by predicting and verifying multiple tokens simultaneously.22 A small, fast draft model proposes a sequence of candidate tokens (typically 3 to 12 tokens), which are then verified by a larger target model in a single, parallel forward pass.22 Rejection sampling determines which candidates are accepted or rejected.22 This shift from a memory-bound sequential bottleneck to a compute-bound parallel operation dramatically reduces inference latency on modern hardware.5

Draft Stage (SSM on DRAM/NPU) \---\> Proposes 3-12 Candidate Tokens
                                          |
                                          v
Verification Stage (Target on NPU) \---\> Parallel Single-Pass Validation
                                          |
                                          v
Rejection Sampling \---\> Commits Accepted Prefix / Prunes Rejected Tail

While speculative decoding works well on high-power server accelerators, standard speculative frameworks degrade on mobile hardware because of the overheads associated with running multiple active models in RAM and switching graphs on the NPU.23 Two state-of-the-art frameworks, sd.npu and Lever, address these limitations through hardware-cooperative optimizations.24

### **sd.npu (CoordGen): NPU-Centric Speculative Orchestration**

Mobile Neural Processing Units (NPUs) are domain-specific architectures optimized for massive tensor operations.24 They operate with high efficiency when execution graphs have static, predictable shapes and input batches are large.23 However, speculative decoding introduces dynamic, fragmented shapes that leave up to 75% of the NPU's compute capacity underutilized.24
The sd.npu framework (also known as CoordGen) maximizes NPU efficiency through three synergistic components 1:

* **Progressive Graph Scheduling**: Traditional setups keep only one static graph resident in the NPU memory, forcing a high-latency teardown-and-load cycle when transitioning from the long-sequence prefill graph to the short-sequence decoding graph.23 sd.npu partitions the model and progressively switches prefill graphs to decoding graphs block-by-block.23 This transition is overlapped with chunked prefill operations, eliminating graph-switching latency entirely.24
* **In-Context Distribution Calibration**: Retrieval-based drafts (which build drafts from local historical context or database documents) often suffer from lexical divergence, causing high rejection rates during verification.1 sd.npu leverages the logits generated during the prefill phase to construct a model-calibrated token tree using a lightweight depth-first search (DFS).1 Retrieving from this calibrated tree ensures that draft sequences match the target model's output distribution without requiring additional training or runtime execution.1
* **NPU-Optimized Draft Reuse**: NPUs are weight-stationary architectures that operate efficiently only when sequence lengths are sufficiently large (tokens) to saturate parallel compute lanes.23 Since retrieval drafts are often short (tokens), over 70% of NPU capacity sits idle.24 sd.npu selectively identifies and reuses high-confidence rejected tokens from prior steps, extending draft length and converting underutilized NPU capacity into valid generation throughput.23

### **Lever: Speculative Decoding with DRAM-Flash Heterogeneous Storage**

When target models exceed smartphone DRAM limits, they must reside in flash storage, leaving only a tiny speculative draft model resident in DRAM.2 Flash-backed autoregressive decoding is slow because loading target weights for every sequential step triggers sustained, high-latency flash I/O operations.27 The Lever architecture utilizes the DRAM-resident draft model to generate candidate token trees, ensuring that the target model in flash is invoked only once per multiple tokens, thereby amortizing I/O costs.2
To optimize this process, Lever implements three main components 2:

* **Mobile-Optimized Drafting Construction**: Instead of building linear sequences, the DRAM-resident draft model constructs token trees using an I/O- and compute-aware gain-cost objective.2 It estimates the marginal gain of candidate tokens and balances this against mobile-specific verification costs, prioritizing high-value branches.2
* **Predictor-Based Verification Pruning**: To prevent the flash-backed model from performing unnecessary computations, Lever inserts a lightweight predictor at an intermediate target layer.2 This predictor evaluates early hidden states to predict and prune low-value draft branches before they traverse the remaining target layers, saving compute cycles.2
* **Hardware-Hybrid Execution**: Lever maps computations across hardware based on workload profiles.2 Draft tree generation is scheduled in batches on the NPU to maximize parallel efficiency.25 During verification, the target transformer layers are executed on the NPU, but final output projection and logit calculation are offloaded to the CPU to be executed on demand only along the accepted path, reducing redundant NPU verification calculations.2

## **Grammar-Constrained Token Masking and Agentic Tool Routing**

On-device agent applications require language models to produce highly structured output formats (such as JSON schemas or dynamic tool calls) to reliably interact with local system APIs, databases, and UI components.28 Freeform natural language introductions introduce the risk of syntax and parsing failures, which can break downstream application execution.28
Grammar-constrained decoding solves this formatting challenge at the generation layer.28 At each decoding step, the generation engine uses a formal grammar (such as GBNF or a JSON schema) to identify valid next tokens and masks out invalid tokens by setting their logits to negative infinity () before sampling.28 This approach guarantees 100% compliance with the specified schema, eliminating the need for post-generation parsing or retry-on-invalid-JSON routines.28

Target Model Logits \---\> \[ Grammar-Constrained Mask Filter \] \---\> \[ Masked Logits \] \---\> Token Sampler
                                     ^
                             (Dynamic FSM State)

Traditional grammar engines are static, requiring substantial compilation time (often several seconds) to preprocess schemas and generate state-machine lookup tables.32 This latency is unacceptable for mobile agents, where schemas change dynamically on a per-request basis.32

### **XGrammar-2: High-Performance Grammar Engine**

XGrammar-2 is a high-performance structured generation engine designed specifically for dynamic agentic workloads.32 It addresses the computational overhead of dynamic grammars through four key optimizations 32:

* **TagDispatch**: Instead of compiling monolithic grammars that represent all possible tool schemas simultaneously, TagDispatch treats structured generation as a dynamic, tag-triggered dispatching process.32 Parameterized as a triple containing tag strings, sub-grammars, and stop strings, it runs in a lightweight "Dispatching Mode" using an Aho-Corasick automaton to scan for tags in the output stream.33 Once a tag is matched, the engine switches to "Dispatched Mode" and applies the corresponding sub-grammar constraint until a stop string is reached, avoiding the memory and latency overhead of compiling and tracking monolithic grammars.32
* **Cross-Grammar Cache**: XGrammar-2 observes that different JSON schemas share identical sub-structures (such as generic string patterns, nested objects, or arrays).32 The engine compiles grammars into Finite State Machines (FSMs), hashes their acyclic and cyclic portions, and stores them in a global cross-grammar cache.35 By reusing these precompiled FSM sub-structures across different requests, the engine drops compilation latency to single-digit milliseconds.33
* **Repetition State Compression**: Large arrays or repeating structures (e.g., {"type": "array", "maxItems": 1000000}) typically scale preprocessing time linearly with the repetition count.36 XGrammar-2 introduces a specialized repetition grammar primitive that compresses the state space.36 This primitive reduces preprocessing complexity to scaling time, preventing runtime execution pauses.32
* **Partial Just-in-Time (JIT) Mask Compilation**: Building a complete token-prefix-to-allowed-token mask cache for large schemas can block inference for several seconds.33 XGrammar-2 employs a partial-JIT approach.33 During the prefill phase, the engine analyzes grammar states and compiles only the top\-most computationally expensive states within a set time budget.33 The remaining states are compiled on-demand during decoding.33 Because compile-on-demand steps are lightweight, their execution is masked by the model's forward inference pass, eliminating runtime latency spikes.33

## **Localized Decoding-Time Safety Guardrails and Content Policy Enforcement**

Deploying models directly on user devices requires robust, localized safety guardrails to prevent the generation of harmful, toxic, or policy-violating content.37 Calling cloud-based moderation APIs is not feasible due to offline requirements and latency constraints, while hosting full-sized classification models on-device introduces prohibitive memory overhead.40 SOTA solutions integrate safety enforcement directly into the decoding loop, altering token selection at runtime.37

### **SecDecoding: Dual-Contrastive Logit Steering**

SecDecoding is a modular, decoding-time defense framework that blocks jailbreaks and adversarial prompt-injection attacks without degrading downstream model performance.37 Rather than running an expensive classifier, SecDecoding utilizes a pair of lightweight, auxiliary 1B-parameter models: a standard base model and a safety-fine-tuned expert.37
During generation, the divergence between the output distributions of these two auxiliary models is calculated to isolate a token-level safety signal.37 This signal is applied as a dynamic probabilistic constraint, modifying the target model's logits and steering the generation trajectory away from unsafe paths while preserving the helpfulness of safe responses.37 Because it operates on hidden states, SecDecoding is compatible with speculative decoding pipelines, providing up to a 1.5x inference speedup by overlapping safety checks with speculative verification.37


                                       |
                                       v  (Measure Divergence)
Target Model Logits \---\> \---\> Dynamic Logit Adjustment \---\> Safe Output
                                       ^
                                       |


### **Claim-Based Stream Decoding (CSD)**

For strict regulatory compliance, Claim-Based Stream Decoding (CSD) provides provable, certifiable safety bounds via conformal analysis.39 CSD divides the target model’s sequential output stream into discrete, semantic claims using a lightweight parsing model.39 A streaming guardrail model then evaluates each completed claim, calculating safety risk.39
If a claim's safety risk exceeds a mathematically certified threshold, the system initiates a backtracking routine.39 By rewinding the KV-cache to the start of the flagged claim, the system blocks the unsafe output and resamples a safe alternative sequence, ensuring grammatical fluency while maintaining theoretical safety guarantees.39

### **Training-Free and Multi-Task Guardrails**

When auxiliary models are unavailable, developers can implement training-free methods like **Gradient-Controlled Decoding** (GCD), which utilizes fixed anchor tokens—such as the acceptance anchor "Sure" and the refusal anchor "Sorry"—to dynamically tighten decision boundaries, significantly reducing false-positive over-refusals on benign queries.44
For applications requiring explainable moderation, the **Lightweight Explainable Guardrail** (LEG) utilizes a multi-task learning architecture to classify prompt safety while simultaneously highlighting the specific words triggering the decision.41 This design matches or exceeds the detection accuracy of larger models while operating at minimal computational cost.41

## **Architectural Implementation Synthesis for the SDK Pipeline**

To build a high-performance on-device LLM SDK, developers must integrate these optimization layers into a unified pipeline. The pipeline should be written in clean, compile-optimized C++ wrapping native mobile delegates, ensuring seamless operation across both iOS and Android platforms.4

\[ User Input / Context \]
          |
          v
\========================================= PRE-PROCESSING PHASE \=========================================

  \--\> Compresses context/history in a single bidirectional pass (3x-6x faster than original LLMLingua)
  \--\> Reduces prefill latency and initial KV-cache footprint
          |
          v
\=========================================== PREFILL PHASE \===========================================

  \--\> Progressive Graph Scheduling loads the decoding graph block-by-block
  \--\> Switches graphs during chunked prefill computation to eliminate switching latency
  \--\> Generates initial logits from the compressed prompt
          |
          v
\======================================== CALIBRATION PHASE \========================================

  \--\> Uses prefill logits to construct a model-calibrated DFS token tree
  \--\> Aligns context semantics with model expectations to reduce draft rejections
          |
          v
\=========================================== DECODING PHASE \===========================================
Loop until End-of-Sequence (EOS) or stop criteria met:

  1\.
     \--\> Generates draft sequences from the calibrated DFS tree on the NPU
     \--\> Lengthens drafts via NPU-Optimized Draft Reuse, reclaiming prior rejected tokens

  2\. \[ Parallel Verification \]
     \--\> Verifies draft sequences in parallel on the NPU
     \--\> Applies intermediate predictor pruning to abort unpromising branches early

  3\.
     \--\> Employs TagDispatch to run in lightweight Aho-Corasick dispatching mode \[32, 33, 35\]
     \--\> On tag match, switches to Dispatched Mode, applying the schema constraint
     \--\> Uses Cross-Grammar Cache & partial-JIT to generate token masks with zero latency overhead \[33, 36\]

  4\.
     \--\> Adjusts target model logits using dual-contrastive auxiliary steering (Base vs. Expert 1B)
     \--\> Redirects output trajectories away from jailbreaks in real-time

  5\.
     \--\> Commits accepted, safe, schema-compliant tokens to the active KV-cache \[22, 24, 39\]
\========================================================================================================
          |
          v

### **Actionable Integration Directives for SDK Construction**

To execute this architectural design, developers must configure the following low-level system settings within the C++ library implementation:

#### **1\. Implement a Zero-Heap Allocation Memory Strategy**

Within the ExecuTorch runtime configuration, standard memory allocation mechanisms must be completely bypassed during the active inference loop.7 Developers should instantiate a contiguous, pre-planned memory buffer using a static hierarchy, mapping fast-changing tensors (such as KV-cache blocks) directly to the SoC's high-speed SRAM.7 Constant weights must be mapped using mmap to allow multiple processes to reference the model binary directly from storage without causing memory inflation or triggering Out-of-Memory (OOM) process termination.7

#### **2\. Optimize CPU-NPU Threading and Queue Management**

To minimize execution bubbles, speculative drafting on the host CPU and verification on the NPU must be managed asynchronously.25 The C++ SDK should maintain parallel circular queues where the NPU processes verification batches for sequence segment , while the CPU concurrently runs draft tree generation for segment .25 Thread priorities should be bound to native platform scheduling engines (such as Android's pthread\_setschedparam or iOS's Grand Central Dispatch QoS) to prevent background UI rendering threads from interrupting inference cycles.4

#### **3\. Dynamically Compact and Re-index the KV-Cache**

When using LLMLingua-2 for context pruning, the SDK must dynamically compact and re-index active KV-cache allocations.7 Pruning tokens creates fragmented holes in the cache memory.4 The C++ pipeline must execute a quick, memory-copy-free tensor re-indexing operation at the start of the prefill stage, ensuring that contiguous memory blocks are presented to the NPU's static verification graphs.7

#### **4\. Compile Target-Specific Hardware Instructions**

When building GGUF or portable operator libraries, compilation must target ARMv9 extensions to leverage SVE2 and SME instruction sets.9 On devices with the Snapdragon 8 Gen 5 or equivalent SoCs, these vectorized kernels execute matrix operations up to 20% faster than standard Neon vector code, allowing the CPU to execute auxiliary safety and grammar engines with minimal latency overhead.9
By implementing this integrated architectural pipeline, the edge-native SDK overcomes the memory and processing limitations of smartphone hardware.4 It delivers a private, reliable, and highly responsive user experience, enabling complex agentic behaviors directly on user devices.1

#### **עבודות שצוטטו**

1. Accelerating Mobile Language Model Generation via Hybrid Context and Hardware Coordination \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2510.15312v1](https://arxiv.org/html/2510.15312v1)
2. Lever: Speculative LLM Inference on Smartphones \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/pdf/2605.16786](https://arxiv.org/pdf/2605.16786)
3. \[2510.15312\] Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/abs/2510.15312](https://arxiv.org/abs/2510.15312)
4. ExecuTorch \-- A Unified PyTorch Solution to Run AI Models On-Device \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/pdf/2605.08195](https://arxiv.org/pdf/2605.08195)
5. On-Device LLMs: State of the Union, 2026 \- Vikas Chandra, נרשמה גישה בתאריך יוני 10, 2026, [https://v-chandra.github.io/on-device-llms/](https://v-chandra.github.io/on-device-llms/)
6. ExecuTorch vs ONNX Runtime: PyTorch Native vs Universal Model ..., נרשמה גישה בתאריך יוני 10, 2026, [https://cactuscompute.com/compare/executorch-vs-onnx-runtime](https://cactuscompute.com/compare/executorch-vs-onnx-runtime)
7. ExecuTorch Runtime Overview \- PyTorch documentation, נרשמה גישה בתאריך יוני 10, 2026, [https://docs.pytorch.org/executorch/0.4/runtime-overview.html](https://docs.pytorch.org/executorch/0.4/runtime-overview.html)
8. Accelerating Phi-2, CodeLlama, Gemma and other Gen AI models with ONNX Runtime, נרשמה גישה בתאריך יוני 10, 2026, [https://onnxruntime.ai/blogs/accelerating-phi-2](https://onnxruntime.ai/blogs/accelerating-phi-2)
9. Accelerating LLAMA inference on mobile CPUs using Qualcomm Matrix Extensions, נרשמה גישה בתאריך יוני 10, 2026, [https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx)
10. Unleashing the Power of AI on Mobile: LLM Inference for Llama 3.2 Quantized Models with ExecuTorch and KleidiAI \- Arm Developer, נרשמה גישה בתאריך יוני 10, 2026, [https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
11. How ExecuTorch Works \- PyTorch documentation, נרשמה גישה בתאריך יוני 10, 2026, [https://docs.pytorch.org/executorch/stable/intro-how-it-works.html](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html)
12. Concepts — ExecuTorch 1.0 documentation, נרשמה גישה בתאריך יוני 10, 2026, [https://docs.pytorch.org/executorch/1.0/concepts.html](https://docs.pytorch.org/executorch/1.0/concepts.html)
13. Understanding Backends and Delegates — ExecuTorch 1.3 documentation, נרשמה גישה בתאריך יוני 10, 2026, [https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)
14. \[2603.15954\] MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale Deployment \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/abs/2603.15954](https://arxiv.org/abs/2603.15954)
15. \[2511.06719\] MobileLLM-Pro Technical Report \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/abs/2511.06719](https://arxiv.org/abs/2511.06719)
16. Prompt Compression for LLMs \- Emergent Mind, נרשמה גישה בתאריך יוני 10, 2026, [https://www.emergentmind.com/topics/prompt-compression-for-large-language-models](https://www.emergentmind.com/topics/prompt-compression-for-large-language-models)
17. Compressing Prompts for Accelerated Inference of Large Language Models \- LLMLingua, נרשמה גישה בתאריך יוני 10, 2026, [https://llmlingua.com/llmlingua.html](https://llmlingua.com/llmlingua.html)
18. LLMLingua: Innovating LLM efficiency with prompt compression \- Microsoft Research, נרשמה גישה בתאריך יוני 10, 2026, [https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)
19. Prompt Compression in Large Language Models (LLMs): Making Every Token Count | by Sahin Ahmed(Data Scientist/MLE) | Medium, נרשמה גישה בתאריך יוני 10, 2026, [https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03](https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03)
20. Compressing Prompts with LLMLingua: Reduce Costs, Retain Performance \- PromptHub, נרשמה גישה בתאריך יוני 10, 2026, [https://www.prompthub.us/blog/compressing-prompts-with-llmlingua-reduce-costs-retain-performance](https://www.prompthub.us/blog/compressing-prompts-with-llmlingua-reduce-costs-retain-performance)
21. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2310.05736v2](https://arxiv.org/html/2310.05736v2)
22. An Introduction to Speculative Decoding for Reducing Latency in AI Inference | NVIDIA Technical Blog, נרשמה גישה בתאריך יוני 10, 2026, [https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
23. Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2510.15312v3](https://arxiv.org/html/2510.15312v3)
24. Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2510.15312v4](https://arxiv.org/html/2510.15312v4)
25. Lever: Speculative LLM Inference on Smartphones \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2605.16786v1](https://arxiv.org/html/2605.16786v1)
26. Accelerating Mobile Language Model via Speculative Decoding and NPU-Coordinated Execution \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/pdf/2510.15312](https://arxiv.org/pdf/2510.15312)
27. \[2605.16786\] Lever: Speculative LLM Inference on Smartphones \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/abs/2605.16786](https://arxiv.org/abs/2605.16786)
28. LM-Kit.NET Structured Output: Constrained JSON Generation in C\# .NET, נרשמה גישה בתאריך יוני 10, 2026, [https://docs.lm-kit.com/lm-kit-net/guides/glossary/structured-output.html](https://docs.lm-kit.com/lm-kit-net/guides/glossary/structured-output.html)
29. \[RFC\] Constrained decoding for extension/llm — is it worth doing? · Issue \#19215 · pytorch/executorch \- GitHub, נרשמה גישה בתאריך יוני 10, 2026, [https://github.com/pytorch/executorch/issues/19215](https://github.com/pytorch/executorch/issues/19215)
30. Structured Output Generation in LLMs: JSON Schema and Grammar-Based Decoding | by Emre Karatas | Medium, נרשמה גישה בתאריך יוני 10, 2026, [https://medium.com/@emrekaratas-ai/structured-output-generation-in-llms-json-schema-and-grammar-based-decoding-6a5c58b698a6](https://medium.com/@emrekaratas-ai/structured-output-generation-in-llms-json-schema-and-grammar-based-decoding-6a5c58b698a6)
31. Structured Output of Large Language Models | Niklas Heidloff, נרשמה גישה בתאריך יוני 10, 2026, [https://heidloff.net/article/llm-structured-output/](https://heidloff.net/article/llm-structured-output/)
32. XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2601.04426v2](https://arxiv.org/html/2601.04426v2)
33. XGrammar 2: High-Performance Grammar Systems \- Emergent Mind, נרשמה גישה בתאריך יוני 10, 2026, [https://www.emergentmind.com/topics/xgrammar-2](https://www.emergentmind.com/topics/xgrammar-2)
34. 1 Introduction \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2601.04426v1](https://arxiv.org/html/2601.04426v1)
35. XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs \- arXiv, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2601.04426v3](https://arxiv.org/html/2601.04426v3)
36. XGrammar-2: Fast and Customizable Structured Generation for Tool Calling and Agents, נרשמה גישה בתאריך יוני 10, 2026, [https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)
37. SecDecoding: Steerable Decoding for Safer LLM Generation \- ACL Anthology, נרשמה גישה בתאריך יוני 10, 2026, [https://aclanthology.org/2025.findings-emnlp.1118/](https://aclanthology.org/2025.findings-emnlp.1118/)
38. Strengthening LLM guardrails with synthetic data generation \- JPMorganChase, נרשמה גישה בתאריך יוני 10, 2026, [https://www.jpmorganchase.com/about/technology/blog/fence-framework](https://www.jpmorganchase.com/about/technology/blog/fence-framework)
39. C-SafeGen: Certified Safe LLM Generation with Claim-Based Streaming Guardrails, נרשמה גישה בתאריך יוני 10, 2026, [https://neurips.cc/virtual/2025/poster/116139](https://neurips.cc/virtual/2025/poster/116139)
40. LLM Guardrails in Production: Input, Output, and Runtime Checks That Actually Work, נרשמה גישה בתאריך יוני 10, 2026, [https://www.kalviumlabs.ai/blog/guardrails-for-llm-applications/](https://www.kalviumlabs.ai/blog/guardrails-for-llm-applications/)
41. A Lightweight Explainable Guardrail for Prompt Safety \- OpenReview, נרשמה גישה בתאריך יוני 10, 2026, [https://openreview.net/forum?id=M4He5YzG44](https://openreview.net/forum?id=M4He5YzG44)
42. SecDecoding: Steerable Decoding for Safer LLM Generation \- ACL Anthology, נרשמה גישה בתאריך יוני 10, 2026, [https://aclanthology.org/2025.findings-emnlp.1118.pdf](https://aclanthology.org/2025.findings-emnlp.1118.pdf)
43. C-SafeGen: Certified Safe LLM Generation with Claim-Based Streaming Guardrails | OpenReview, נרשמה גישה בתאריך יוני 10, 2026, [https://openreview.net/forum?id=nOsEyBGk1I](https://openreview.net/forum?id=nOsEyBGk1I)
44. Gradient-Controlled Decoding: A Safety Guardrail for LLMs with Dual-Anchor Steering, נרשמה גישה בתאריך יוני 10, 2026, [https://arxiv.org/html/2604.05179v1](https://arxiv.org/html/2604.05179v1)
