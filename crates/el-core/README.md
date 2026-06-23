# el-core — shared domain vocabulary

The foundational crate of the Edge Intelligence SDK: the *ubiquitous language*
of the project turned into Rust types. Every other crate speaks in terms of the
ids, value objects, errors, events, and the provider trait defined here.

`el-core` has **zero external dependencies** — pure `std`, so it compiles
offline on any target including `wasm32` (ADR-008). It contains no I/O, no
network, and no `unsafe` (`#![forbid(unsafe_code)]`).

## What it provides

| Module | Key types | Purpose |
|--------|-----------|---------|
| `ids` | `SessionId`, `ModelId`, `ModelVersion` | Identifier value objects |
| `value_objects` | `Token`, `ModelFormat`, `RuntimeKind`, `DeviceTarget`, `Phase`, `SafetyMode`, `SpeculationMode`, `StopReason` | Core enums and the `Token = u32` alias |
| `config` | `SessionConfig` | Immutable per-session configuration |
| `error` | `EdgeError`, `Result<T>` | The SDK-wide error type |
| `events` | `DomainEvent`, `EventEnvelope`, `DegradeReason` | Content-free domain events |
| `provider` | `LlmProvider`, `ChatRequest`, `ChatResponse`, `ChatMessage`, `ChatRole`, `ChatToken`, `CredentialRef` | The unified backend abstraction |

All names are re-exported at the crate root, e.g. `use el_core::{LlmProvider, ChatRequest};`.

## Cross-cutting invariants encoded here

These are enforced by the type system, not by convention:

- **Content-free events (ADR-007).** `DomainEvent` and `EventEnvelope` derive
  `Copy`. A `String`/`Vec`/heap field is not `Copy`, so adding one fails to
  compile — "no prompt or response content on an event" is a *compile-time*
  guarantee. Ratios and scores are carried as fixed-point integers (`*_milli`).
- **Air-gap by default (ADR-004).** `SessionConfig::default().hybrid_mode` is
  `false`. The only network seam is an explicit opt-in.
- **Unified provider (ADR-010).** `LlmProvider` covers both the local Candle
  engine and cloud frontier backends behind one trait, so host apps can swap
  local ↔ frontier without touching their UI.
- **Redacted credentials.** `CredentialRef`'s `Debug` output is
  `CredentialRef([REDACTED])`, so bearer keys cannot leak into logs or panic
  messages.

## Usage

This is the crate-local part of a real local SDK call. In an app, the request
can be served by `el_engine_candle::QwenChatProvider::from_paths(...)` with
`models/qwen2.5-0.5b-instruct-q4_k_m.gguf` and
`models/qwen2.5-0.5b-instruct.tokenizer.json`; `el-core` only defines the
backend-agnostic contract.

```rust
use el_core::{ChatMessage, ChatRequest, LlmProvider, SessionConfig};

// Configuration is air-gapped by default (ADR-004).
let cfg = SessionConfig::default();
assert!(!cfg.hybrid_mode);

// A backend-agnostic request. `model` is a routing hint:
//   "local"/"" → local engine, "openai/…", "anthropic/…", "ollama/…", "gemini/…"
let req = ChatRequest::new(
    "local/qwen2.5-0.5b-instruct-q4_k_m",
    vec![ChatMessage::user("Summarize edge inference in one sentence.")],
)
    .with_max_tokens(256)
    .with_temperature(700); // 700 = 0.7 (milli to keep the type Eq-able)

// Any backend is reached through the same trait.
fn run(provider: &dyn LlmProvider, req: &ChatRequest) -> el_core::Result<String> {
    Ok(provider.chat(req)?.content)
}
```

## Place in the workspace

`el-core` is the root of the dependency graph: `el-memory`, `el-telemetry`,
`el-provenance`, `el-safety`, `el-runtime`, and every adapter depend on it, and
it depends on nothing. Keep it dependency-free — that property is what lets the
local core cross-compile to WASM and mobile targets.

## Status

Implemented and tested.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-004](../../docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md),
[ADR-007](../../docs/adr/ADR-007-content-free-domain-events-privacy-by-construction-telemetry.md),
[ADR-008](../../docs/adr/ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md),
and [ADR-010](../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md).
The vocabulary mirrors [`docs/ddd/ubiquitous-language.md`](../../docs/ddd/ubiquitous-language.md).
