# ADR-027: Official DictaLM 3.0 Qwen3 GGUF model profile

- **Status**: accepted
- **Date**: 2026-08-11
- **Deciders**:
- **Tags**: runtime, qwen3, dictalm, gguf, hebrew, supported-model, high-memory

## Context

Edge Intelligence will officially support
[`EMD123/DictaLM-3.0-1.7B-Instruct-Q4_K_M-GGUF`](https://huggingface.co/EMD123/DictaLM-3.0-1.7B-Instruct-Q4_K_M-GGUF),
a pinned community GGUF conversion of Dicta's Hebrew/English 1.7B instruct model,
as its first named Qwen3 model profile. This ADR distinguishes three different
claims:

1. Candle can parse the GGUF container and its Q4_K_M tensor types.
2. The SDK has a numerically correct Qwen3 transformer path for this model.
3. The public SDK can run that path within a declared device, memory, quality,
   safety, and provenance envelope.

This ADR accepts the model profile and commits the project to implementing all
three claims. SDK v0.3.14 predates that implementation and remains incompatible;
that historical fact is an implementation-status note, not a rejection of the
support decision. The first release that advertises DictaLM availability must
satisfy the release gates in this ADR.

### Supported artifact profile

Evidence was collected on 2026-08-05. Validation must use immutable revisions,
not either repository's moving `main` branch.

| Item | Pinned identity |
|---|---|
| Community quant repository | revision [`2f41ecf9bdeb2d75ff4dd85266970402c59c14d8`](https://huggingface.co/EMD123/DictaLM-3.0-1.7B-Instruct-Q4_K_M-GGUF/tree/2f41ecf9bdeb2d75ff4dd85266970402c59c14d8) |
| GGUF | `dictalm-3.0-1.7b-instruct-q4_k_m.gguf`; 1,107,404,704 bytes (1.031 GiB); SHA-256 `68998faec0aee53a93ba116dd0a89e9857092a3e693578aff52502148a3d6707` |
| Official source model | [`dicta-il/DictaLM-3.0-1.7B-Instruct`](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct/tree/5add44d6941d6ffb9eb6cc6b516f9fb4fd472494), revision `5add44d6941d6ffb9eb6cc6b516f9fb4fd472494` |
| External tokenizer required by this SDK | `tokenizer.json`; 11,422,648 bytes; SHA-256 `3c3dfe474a8bbe89b0e83627fd9ff784ad71027f12fd8c618708c818e808789d` |
| License claim | Apache-2.0 in the source model card and GGUF metadata; the community quant repository does not include a license text |

The GGUF v3 header has 310 tensors and declares `general.architecture=qwen3`,
`general.file_type=15` (mostly Q4_K_M), 28 blocks, a 2,048-wide embedding, a
6,144-wide feed-forward layer, 16 query heads, 8 KV heads, 128-wide keys and
values, RoPE base 1,000,000, vocabulary size 151,936, and a 62,080-token context.
The official [`config.json`](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct/blob/5add44d6941d6ffb9eb6cc6b516f9fb4fd472494/config.json)
corroborates the architecture and context ceiling. The tokenizer's 131,072
`model_max_length` must not be mistaken for a validated model context.

### Implementation baseline at acceptance

| Surface | Result | Evidence |
|---|---|---|
| GGUF v3 and Q4_K tensor decoding | compatible | Candle 0.8.4 has Q4_K storage support. Quantization is not the blocker. |
| Transformer architecture | **incompatible** | `el-engine-candle` imports only `quantized_qwen2::ModelWeights`; its loader requires `qwen2.*` metadata and Q/K/V bias tensors. This artifact has `qwen3.*` metadata, Q/K RMSNorm tensors, and no attention biases. |
| Empirical load | **fails** | `el-chat` was run with the authentic first 16 MiB of the GGUF (enough to contain the complete header) and the pinned source tokenizer. It exited with `GGUF: failed to load Qwen2 weights` before tensor data was read. This proves the architecture rejection only; it is not a full-model inference test. |
| Tokenizer | incomplete packaging | The SDK tokenizes outside GGUF and therefore needs the separately pinned official `tokenizer.json`; the community quant repository does not ship it. |
| Prompt contract | incomplete | The GGUF has no `tokenizer.chat_template`. The provider hard-codes a Qwen2.5 ChatML subset. Dicta ships a separate [`chat_template.jinja`](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct/blob/5add44d6941d6ffb9eb6cc6b516f9fb4fd472494/chat_template.jinja) with default-system and tool-call behavior that is not represented by the current renderer. |
| Termination | incomplete | The provider stops only on `<|im_end|>` (151645). Dicta's pinned [`generation_config.json`](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct/blob/5add44d6941d6ffb9eb6cc6b516f9fb4fd472494/generation_config.json) declares both 151645 and `<|endoftext|>` (151643) as EOS; the GGUF exposes only 151643 as its single EOS metadata value. |
| Default memory envelope | **incompatible** | The 1.031 GiB file already exceeds the default 1 GiB session budget before runtime storage. Candle-style F32 dequantization of the tied `[151936, 2048]` embedding alone is 1,244,659,712 bytes; retaining quantized weights and a 2,048-token F32 KV cache gives an analytical lower bound above 2.5 GiB before activations and allocator overhead. This must be measured, not treated as the final RSS. |
| Public SDK bindings | **incompatible** | `EdgeLlm::local` routes non-empty GGUF paths through `LocalLlmProvider`/`CandleEngine`, the two-tensor seam proof, rather than the full-transformer `QwenChatProvider`. The wasm path uses `EchoProvider`. |
| Production provenance | **incomplete** | The artifact is an un-signed third-party conversion. Its card does not pin the source revision, converter/llama.cpp revision, conversion command, or quantization-quality result. The current local verifier is permissive/trust-the-file. |

Candle 0.8.4 has no Qwen3 module. Candle 0.11 exposes a
[`quantized_qwen3`](https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/quantized_qwen3.rs)
loader with the required `qwen3.*` metadata and Q/K normalization. That makes
support technically feasible, but an engine upgrade alone does not satisfy the
prompt, termination, memory, binding, safety, or provenance contracts.

### Initial implementation (2026-08-11)

The first implementation changes are now present: Candle is upgraded to 0.11,
the engine preflights and dispatches `qwen2`/`qwen3` rather than shimming their
metadata, and Qwen3 uses Candle's explicit KV-cache clear. `from_paths` remains
the Qwen2 compatibility path; `from_dictalm_bundle` is the only Qwen3 entry
point and verifies the pinned GGUF, tokenizer, and exact upstream template
sizes/digests before parsing tensors. A detached Ed25519 signature over the
canonical whole-bundle manifest must verify against an Edge build-time trust
anchor; builds without that key reject the profile. It compiles DictaLM's
plain-chat template policy (with tool calling still excluded), uses both stop
IDs, imposes the initial 2,048 total-token cap, and requires an explicit 4 GiB
HighEnd capability before load. Rust CLI/bench and native FFI/RN have distinct
DictaLM constructors; FFI additionally rejects non-arm64 hosts.

This is an implementation baseline, **not release certification**. AC-1's
signed-bundle distribution requirement and AC-2 through AC-10's real-artifact,
physical-device, numerical-parity, safety, RSS, thermal, and offline evidence
remain mandatory before public availability is advertised.

The release pipeline must set `EDGE_DICTALM_ED25519_PUBLIC_KEY_HEX` to the
64-hex-character Edge trust-anchor public key at compile time and ship the
corresponding detached manifest signature with the bundle. The key is public;
the signing secret stays outside the repository and release artifact. A build
without this trust anchor rejects DictaLM before opening the GGUF.

## Decision

Officially adopt the pinned DictaLM 3.0 1.7B Instruct Q4_K_M bundle as a
**first-class supported model profile** for Rust and native arm64 Android/iOS on
the high-memory device tier. DictaLM support is part of the product architecture,
not an optional research outcome. The project must implement, test, package, and
maintain the following contract.

Until the implementation lands, older SDK versions must reject the model with an
architecture-specific error rather than silently passing Qwen3 metadata to the
Qwen2 loader. That fail-closed behavior does not remove the model from the
official support policy.

1. **Architecture preflight and dispatch.** Read `general.architecture` before
   allocating model tensors. Dispatch explicitly to Qwen2 or Qwen3 and preserve
   a diagnostic containing the detected architecture and supported set. Unknown
   or mismatched tensor schemas fail closed.

2. **Correct Qwen3 implementation.** Pin and dependency-audit a Candle release
   with `quantized_qwen3` (0.11 is the first evaluated candidate), or backport
   the equivalent implementation. A metadata-prefix alias or Qwen2 tensor shim
   is forbidden: Qwen3 Q/K RMSNorm and bias-free attention are semantic model
   differences. The same engine abstraction must preserve Qwen2.5 behavior.

3. **Pinned model bundle contract.** Treat weights, tokenizer, chat template,
   special-token policy, and generation policy as one versioned artifact manifest.
   Pin their repository revisions and SHA-256 digests. Set `add_bos=false`, render
   prompts from the pinned Dicta template (or a byte-for-byte equivalent compiled
   representation), and stop on both 151645 and 151643. A tokenizer/vocabulary or
   architecture mismatch is rejected before a session is created.

4. **High-memory profile with an enforced context cap.** The existing 1 GiB
   profile must reject this model before a large allocation. Official DictaLM
   support starts with a 2,048-token total prompt-plus-generation cap and a
   separate, explicit high-memory budget no greater than 4 GiB. This ADR amends
   ADR-003 by adding that opt-in profile; the 1 GiB default remains unchanged.
   Do not advertise the GGUF's 62,080-token architectural ceiling as a supported
   runtime window. Raising the cap requires new on-device RSS, latency, thermal,
   and stability evidence.

5. **Native full-transformer routing.** Expose the architecture-aware DictaLM
   provider through the public Rust and native arm64 Android/iOS binding
   constructors, including
   the real grammar and active safety/guard/rollback ports. A seam-proof
   projection using `Ports::permissive()`, echo backend, CLI-only success, or raw
   GGUF parse does not count as model support. wasm remains unsupported until it
   independently passes the same resource and correctness gates. Mid-range and
   wasm availability are not included in the initial official support matrix.

6. **Deterministic SDK behavior.** The SDK may retain greedy decoding even though
   Dicta publishes sampling defaults (`temperature=0.6`, `top_k=20`, `top_p=0.95`).
   Documentation must state that this is an SDK policy and that output parity with
   the sampled upstream demo is not expected. Tool calling is a separate API
   capability; plain-chat support must not imply tool-call support.

7. **Provenance before distribution.** Development may validate the pinned
   community blob by digest. Production distribution requires the ADR-006
   signature gate over the entire bundle and either (a) a reproducible internal
   conversion from the pinned official BF16 source or (b) an explicit supply-chain
   acceptance of the third-party conversion. Preserve Apache-2.0 notices and
   complete attribution; model licensing does not resolve training-data legal or
   privacy review.

## Release gates

The architecture decision is accepted. All rows remain mandatory before an SDK
release may advertise its DictaLM profile as available on a particular surface or
device tier.

| ID | Gate | Pass condition |
|---|---|---|
| AC-1 | Artifact integrity | The full 1,107,404,704-byte GGUF and external bundle files match the manifest's pinned revisions, sizes, and SHA-256 digests; one-bit mutation and cross-version tokenizer tests fail before model construction. |
| AC-2 | Architecture load | The full artifact loads through the Qwen3 dispatcher on CPU; logits are finite with width 151,936. A Qwen3 file sent to the Qwen2 loader and an unknown architecture return explicit, tested errors. |
| AC-3 | Prompt parity | For system/user, multi-turn Hebrew/English, and tool-shaped fixtures, SDK-rendered token IDs match the pinned Dicta template. Plain-chat mode documents any deliberately unsupported template branches. |
| AC-4 | Termination | Generation stops cleanly on either 151645 or 151643, never emits a following role, honors `add_bos=false`, and enforces prompt + completion <= 2,048 tokens. |
| AC-5 | Numerical behavior | Fixed Hebrew and English prompts are compared with a pinned llama.cpp Q4_K_M development oracle and the official BF16 model. Differences are reviewed at token/logit and task-quality levels; load success alone cannot pass. No C/C++ oracle becomes a shipped dependency (ADR-008). |
| AC-6 | Stateful runtime and safety | Multi-turn prefix reuse, true Qwen3 KV clear, rollback, refusal, `end_session`, ingress triage, and same-tokenizer safety-expert compatibility pass. No previous-conversation token or cache state survives reset. |
| AC-7 | Performance and memory | Release builds record cold load, peak RSS, time to first token, prefill tok/s, decode tok/s, and thermal/stability behavior at 512, 1,024, and 2,048 tokens. Peak RSS stays within the declared high-memory budget and the default 1 GiB profile rejects early. The model meets the then-current SLO for every claimed device tier (today: high-end >500 prefill tok/s, 30–80 decode tok/s, <200 ms TTFT; mid-range ~200 prefill tok/s, 10–20 decode tok/s, <500 ms TTFT), or that tier remains unsupported. A 30-minute soak retains at least 80% of initial throughput with no OOM, ANR, or thermal shutdown. Batched/chunked prefill is required if one-token prefill misses the SLO. |
| AC-8 | Product surfaces and offline operation | End-to-end Hebrew and English smoke chats use the same full-transformer and active-safety path through Rust and each claimed native binding on recorded physical arm64 Android and arm64 iOS devices. Load plus a 100-turn run under denied network access attempts no egress. Unsupported wasm and device tiers return a capability error, not a fallback response. |
| AC-9 | Quality and safety | The existing 66-task CounselBench/MindEval/VERA-MH harness plus versioned Hebrew, mixed RTL/LTR, niqqud, code-switching, and Hebrew-safety slices complete with zero engine errors under both raw-model and product-safety configurations. Q4 quality loses no more than two absolute points against the pinned reference on agreed metrics. A clinical/mental-health claim additionally requires zero VERA-MH Red conversations; otherwise support is explicitly non-clinical. |
| AC-10 | Regression, privacy, and operations | Existing Qwen2.5, malformed-GGUF, streaming, reset, provenance, and safety suites remain green. Logs and telemetry contain model identity and numeric timings but no prompt or completion content. Forced safety intervention is observable through each host binding, and a rejected update leaves the prior verified model untouched. |

## Rollout and fallback

1. Land the official profile implementation behind a temporary
   `experimental-qwen3` development flag for internal CLI validation. The flag is
   a rollout control, not a reversal of the support decision.
2. Admit only signed, allowlisted manifests to high-memory native dogfood. Keep
   Qwen2.5-0.5B as the default and reference profile.
3. Promote through opt-in 5%, 25%, and 100% cohorts only after content-free local
   metrics and every applicable release gate remain green.
4. On verification, capability, or budget failure, keep or restart with a
   previously installed and verified local model at a session boundary. Never
   switch models mid-generation, accept an unsigned artifact, or fall back to the
   cloud automatically.
5. Signature/manifest mismatch, crash/OOM, any safety-gate regression, or a >10%
   p95 latency regression removes the Qwen3 manifest from the allowlist and
   restores the previous verified profile.

## Consequences

### Positive

- The project has a clear, affirmative support commitment for a capable
  Hebrew/English edge model rather than treating it as an indefinite experiment.
- We avoid claiming release availability based only on container parsing or a
  CLI demo.
- Correct architecture dispatch creates a reusable path for later Qwen3 GGUFs.
- The pinned bundle makes tokenizer, template, stop-token, and supply-chain drift
  testable instead of implicit.
- DictaLM can serve its intended Hebrew/English edge use case without weakening
  the default 1 GiB mobile invariant.

### Negative

- Candle 0.8 to 0.11 is a workspace-wide dependency upgrade with build, target,
  numerical, and performance regression risk.
- This 1.7B model needs a materially larger memory/device tier than the project's
  0.5B default and may still miss interactive TTFT on CPU until batched prefill
  (ADR-020) lands.
- Supporting the full Dicta prompt contract and multi-EOS termination expands the
  runtime/model-bundle interface.
- A community quant either adds ongoing supply-chain review or must be replaced by
  a reproducible internal conversion.

### Neutral

- Apache-2.0 permits use under its terms but does not itself prove artifact
  integrity, conversion quality, training-data clearance, or product safety.
- Tool calling and the full 62,080-token architectural window remain out of scope
  until separately accepted.
- llama.cpp may serve as a development oracle, but ADR-008 still forbids making
  its C/C++ implementation a shipped SDK dependency.

## Rejected alternatives

- **Treat Qwen3 as Qwen2 by renaming metadata.** Rejected because the attention
  implementations differ; it would produce either load errors or incorrect logits.
- **Limit DictaLM to an experimental research candidate.** Rejected because the
  project is committing to first-class support; implementation gates control
  release readiness rather than whether support will be pursued.
- **Declare release availability after a successful GGUF parse/load.** Rejected
  because prompt, EOS, cache, binding, memory, quality, safety, and provenance
  failures can remain.
- **Adopt llama.cpp as the production fallback.** Rejected by ADR-008's Rust-only
  implementation decision.
- **Enable the model under the default 1 GiB profile.** Rejected by the artifact
  size and analytical resident-memory lower bound.
- **Consume latest upstream files at runtime.** Rejected because independently
  moving weights, tokenizer, template, or config can silently change behavior and
  violates the air-gapped, signed-bundle model.

## Links

- Candidate: [community GGUF repository](https://huggingface.co/EMD123/DictaLM-3.0-1.7B-Instruct-Q4_K_M-GGUF), [GGUF metadata API](https://huggingface.co/api/models/EMD123/DictaLM-3.0-1.7B-Instruct-Q4_K_M-GGUF?expand=gguf).
- Upstream: [official model card](https://huggingface.co/dicta-il/DictaLM-3.0-1.7B-Instruct), [DictaLM 3 release](https://dicta.org.il/dicta-lm-3), [technical report](https://www.dicta.org.il/publications/DictaLM_3_0___Techincal_Report.pdf).
- Builds on: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md), [ADR-023](./ADR-023-baseline-performance-instrumentation.md).
- Constrained by: [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md), [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md), [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md), [ADR-020](./ADR-020-batched-single-pass-prefill.md), [ADR-021](./ADR-021-memory-mapped-verified-gguf-loading.md), [ADR-022](./ADR-022-two-tier-quantized-kv-cache-with-attention-aware-eviction.md).
- Implementation seams: `crates/adapters/el-engine-candle` (architecture dispatch, Qwen3 engine, tokenizer/template/stop policy), `crates/el-runtime` (multi-EOS and context enforcement), `crates/el-core` (capability and high-memory profile), `crates/adapters/el-ffi` (public native routing), `apps/el-chat` / `apps/el-bench` (end-to-end validation).
