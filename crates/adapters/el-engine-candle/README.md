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

The model weights load **once** in `from_paths` and stay resident; each `chat`
reuses one persistent provenance-gated session (ADR-018) — reset (which evicts the
previous conversation's KV from Candle's cache via a position-0 forward, keeping
the weights loaded) → `load_prompt` (prefill) → `generate` (grammar mask → safety
steer → chunk-guard + checkpointed rollback → greedy commit). No per-turn reload
from disk. Decoding is deterministic greedy argmax, so replies are reproducible.
`end_session()` releases a conversation's memory while keeping the model resident;
dropping the provider frees the weights too.

### On-device safety (ADR-005 + ADR-012)

`QwenChatProvider` is **secure by default**: `from_paths` resolves a conservative
built-in unsafe-word list against the model's tokenizer into the token-id data
the runtime's float-free safety loop consumes — a `LightweightFilter` hard-ban
steerer plus an `AnchorGuard` (weights-free token-anchor chunk guard) — and wires
them into the session's `Ports`. The ADR-012 control loop then runs on every
reply: guard the trajectory, roll the KV cache + output back to the last safe
checkpoint on a hard breach, and fail closed with a deterministic refusal once
`max_rollbacks` is exhausted. The owning app does **not** depend on `el-safety`;
the adapter is the one layer that bridges tokenizer text ↔ token ids.

```rust
use el_core::SafetyMode;
# use el_engine_candle::QwenChatProvider;
# fn demo(provider: QwenChatProvider) {
let provider = provider
    .with_safety(SafetyMode::Off)          // opt out → plain single-pass decode
    .with_extra_guard_words(["banana"]);   // test hook: trip the guard on a benign word
# }
```

`SecDecoding`/`Csd` need model assets not shipped here and fall back to the
`Lightweight` wiring. Token-anchor heuristics match whole token n-grams, so they
are a defence-in-depth net, not a complete guard — production swaps in the active
tier's real safety model (ADR-012 model inventory).

**ADR-013 model-backed layers.** The **built-in** anchor patterns drive both the
output chunk guard and **ingress triage** (the prompt is scored before
generation; a hard breach fails closed with no decode). `--guard-word` extras are
guard-only — they never trigger an ingress refusal. `with_expert_model(path)`
enables **contrastive steering**: a second `QwenExpert` (any same-tokenizer Qwen
GGUF) implements `ExpertLogits` and a `ContrastiveSteerer` applies
`base + α·(expert − base)` over the early-token window only; supplying an expert
promotes the session to `SecDecoding` so `SafetyModeSelector` gates it on device
class. The expert re-primes to the prompt on a base rollback (no stale-branch
logits). It loads through the ADR-006 `LoadPermit` gate using the local
trust-the-file path for user GGUFs; this is not cryptographic integrity over the
weights. Production signed weights must verify the whole artifact plus detached
signature before issuing the permit. The chat model as its own expert is a
≈no-op; a safety-tuned Qwen gives real steering.

## Benchmark instrumentation

Setting `EL_BENCH=1` makes `QwenChatProvider::chat` print a per-phase breakdown
(session setup / tokenize / prefill / decode / detokenize) plus per-forward
attribution (model compute vs. seam quantisation vs. runtime loop) to stderr. The
weights load once at startup (ADR-018), so per-turn "session setup" is near-zero —
not a per-chat model load. Zero-cost when unset; a diagnostic only, not part of
public behaviour.

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
and [ADR-010](../../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md),
and wires the on-device safety of
[ADR-005](../../../docs/adr/ADR-005-on-device-only-tiered-decoder-time-safety.md) /
[ADR-012](../../../docs/adr/ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
into the chat provider; see the
[Inference Runtime](../../../docs/ddd/bounded-contexts/01-inference-runtime.md) context.
