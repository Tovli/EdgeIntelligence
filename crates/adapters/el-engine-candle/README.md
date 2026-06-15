# el-engine-candle — Candle inference engine adapter

The inference-engine adapter over [Candle](https://github.com/huggingface/candle)
(ADR-002). It implements the runtime's `InferenceEngine` port (`RuntimeAcl`) and
the `LlmProvider` trait (ADR-010), so the same `el_runtime::InferenceSession`
decode loop drives everything — nothing in the SDK pipeline is bypassed.

Float logits are quantised to integer milli-logits at the anti-corruption-layer
boundary, so Candle's `Tensor`/`Device` types never cross into the domain. No
`unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

This crate ships two engines and two providers, from "seam proof" to
"real on-device chat":

| Type | What it is |
|------|------------|
| `CandleEngine` | The **engine-seam proof**: one real Candle forward, `embed[last] · w_out` — a single linear projection. `toy()` builds deterministic synthetic weights; `from_path`/`from_bytes` load `token_embd.weight` + `output.weight` from a GGUF. Transformer blocks, attention, RoPE, and norms are *ignored* — logits won't match a real model. |
| `LocalLlmProvider` | Wraps `CandleEngine` behind `LlmProvider` with a byte-level tokenizer. Good for exercising the binding layer end-to-end. |
| `QwenEngine` | A **real Qwen2 transformer** `InferenceEngine` via `candle-transformers`, holding Candle's stateful KV cache. |
| `QwenChatProvider` | A **real local chat backend**: a Qwen2 GGUF + its `tokenizer.json`, rendered to Qwen2.5 ChatML and driven through the standard provenance-gated session. This is what powers [`apps/el-chat`](../../../apps/el-chat). |

### Expected GGUF tensor names (`CandleEngine`)

- `token_embd.weight` — embedding table `[vocab, dim]`
- `output.weight` or `lm_head.weight` — lm-head `[vocab, dim]` (standard Llama layout)

Mismatched shapes are rejected at load time, not silently at inference.

## Usage

```rust
use el_core::{ChatMessage, ChatRequest, LlmProvider};
use el_engine_candle::QwenChatProvider;

// Real on-device chat: a local Qwen2 GGUF + its tokenizer (no network egress).
let provider = QwenChatProvider::from_paths(
    "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "models/qwen2.5-0.5b-instruct.tokenizer.json",
)?;

let req = ChatRequest::new("local", vec![ChatMessage::user("What is the capital of France?")])
    .with_max_tokens(128);
let reply = provider.chat(&req)?;
println!("{}", reply.content);
# Ok::<(), el_core::EdgeError>(())
```

Each `chat` call builds a fresh `QwenEngine` (Candle exposes no public KV-cache
reset) and runs the standard SDK path: provenance permit → `load_prompt`
(prefill) → `generate` (grammar mask → safety steer → greedy commit). Decoding
is deterministic greedy argmax, so replies are reproducible.

## Benchmark instrumentation

Setting `EL_BENCH=1` makes `QwenChatProvider::chat` print a per-phase breakdown
(model load / tokenize / prefill / decode / detokenize) plus per-forward
attribution (model compute vs. seam quantisation vs. runtime loop) to stderr.
It is zero-cost when unset and is a diagnostic only — not part of public behaviour.

## Features & dependencies

- `candle-core` + `candle-transformers` (0.8), and `tokenizers` 0.21 with the
  **pure-Rust `fancy-regex`** backend (no C/C++ `onig`/`esaxx`, per ADR-008).
- `metal` feature → enables `candle-core/metal` for Apple GPUs.

## Status

Implemented; runs real on-device chat. The `CandleEngine` linear projection is
the ADR-002 engine-seam proof; `QwenEngine`/`QwenChatProvider` are the real
transformer path.

---

Part of the [Edge Intelligence](../../../README.md) workspace. Realizes
[ADR-002](../../../docs/adr/ADR-002-candle-as-rust-native-inference-engine.md)
and [ADR-010](../../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md);
see the [Inference Runtime](../../../docs/ddd/bounded-contexts/01-inference-runtime.md) context.
