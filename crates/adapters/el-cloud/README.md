# el-cloud — opt-in frontier LLM cloud backend

The opt-in frontier cloud backend (ADR-010). It implements
`el_core::LlmProvider` over [`reqwest`](https://crates.io/crates/reqwest) using
the OpenAI Chat Completions API — which OpenAI, Anthropic (compat), Gemini
(compat), Ollama, and any OpenAI-compatible endpoint all speak.

## Air-gap is preserved (ADR-004)

This is the project's **only** outbound network surface, and it is opt-in at the
API level:

- Outbound calls happen **only** when an app *explicitly constructs* a
  `CloudProvider`. Apps that never build this type have zero outbound surface,
  even though the crate compiles as part of the workspace.
- The `reqwest` network dependency is gated to non-wasm32 targets (see the
  wasm32 note below), so the web surface has no blocking HTTP transport at all.
- Every consultation emits a content-free `DomainEvent::FrontierLlmConsulted`
  (the `provider_hash` is a CRC32 of the provider prefix — never the API key or
  any content).

## What it provides

- **`CloudProvider`** — owns a pooled `reqwest::blocking::Client` with explicit
  timeouts (10 s connect, 60 s idle/per-read, 120 s total per non-streaming
  request) so a stalled provider can never block an FFI caller indefinitely.
  - `new()` / `Default`
  - `with_event_sink(|event| …)` — register a callback for `DomainEvent`s (e.g.
    to feed an [`el_telemetry::MetricsCollector`](../../el-telemetry)).
  - implements `LlmProvider::chat` (blocking) and `chat_stream` (SSE).

### Provider routing (by model prefix)

| Model string | Base URL |
|--------------|----------|
| `openai/<model>` | `https://api.openai.com/v1` |
| `anthropic/<model>` | `https://api.anthropic.com/v1` (compat) |
| `gemini/<model>` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `ollama/<model>` | `http://localhost:11434/v1` (no key required) |
| `http(s)://…/<model>` | custom base URL |

The streaming path parses Server-Sent Events strictly: provider error payloads
and malformed chunks fail the call (a half-finished stream is never reported as
a clean completion), and errors carry only parse category/position/size —
**never** echoed content, per the el-core error contract.

## Usage

```rust
use el_core::{ChatMessage, ChatRequest, CredentialRef, LlmProvider};
use el_cloud::CloudProvider;

let provider = CloudProvider::new();

// The credential is resolved at runtime from the platform keystore — never embedded.
let req = ChatRequest::new("openai/gpt-4o", vec![ChatMessage::user("Hello!")])
    .with_credential(CredentialRef::new(api_key_from_keystore))
    .with_max_tokens(256);

let resp = provider.chat(&req)?;
println!("{}", resp.content);
# Ok::<(), el_core::EdgeError>(())
```

## wasm32 note

`reqwest::blocking` is unavailable on `wasm32-unknown-unknown` (no threads), and
the synchronous `LlmProvider::chat` cannot await the browser's `fetch`. The
network modules are therefore gated to non-wasm32, and the web binding
(`el-ffi`) exposes a throwing `cloud` constructor instead of silently degrading
(ADR-010 amendment).

## Status

Implemented as an explicit egress adapter and a regular workspace member — the
opt-in is enforced at construction, not by excluding it from the build. Uses
`rustls-tls` so Android cross-compiles need no target OpenSSL sysroot.

---

Part of the [Edge Intelligence](../../../README.md) workspace. Realizes
[ADR-010](../../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md)
while preserving [ADR-004](../../../docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md).
