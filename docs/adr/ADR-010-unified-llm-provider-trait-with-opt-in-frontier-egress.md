# ADR-010: Unified LlmProvider trait with opt-in frontier LLM cloud egress

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**:
- **Tags**: llm, cloud, networking, trait, architecture

## Context

The SDK was originally designed as air-gapped by default
([ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)), with all
inference running on-device via Candle ([ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)).
Flutter and React Native are now confirmed deployment targets
([ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md),
[ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md)), and
the requirement has been extended: **the SDK must support frontier LLMs (OpenAI,
Anthropic, Gemini, Ollama, and compatible providers) as an opt-in backend**
alongside the primary local Candle path.

ADR-004's "LAN relay only — never a cloud endpoint" restriction was written for
the default, fully air-gapped use case. It does not account for apps that
explicitly choose to use a cloud LLM. A new opt-in egress path is needed that:

- Does not violate the privacy guarantee for apps that stay air-gapped
- Is explicit and auditable (no accidental network calls)
- Shares the same host-facing API as the local path so mobile apps can switch
  backends without code changes

## Decision

### 1 — Add `LlmProvider` trait to `el-core`

A single trait covers both local and cloud backends:

```rust
pub trait LlmProvider: Send + Sync {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse>;
    /// Calls `on_token` per fragment; the last call has `is_final == true`.
    fn chat_stream(
        &self,
        req: &ChatRequest,
        on_token: &mut dyn FnMut(ChatToken),
    ) -> Result<()>;
}
```

Streaming is **callback-based, not an async `Stream`**: the SDK is synchronous
end-to-end (no async runtime in the core, ADR-008) and the FFI surfaces bridge
callbacks naturally (UniFFI callback interfaces, FRB closures).

`ChatRequest` carries messages, model name, temperature, max_tokens, and
an optional `CredentialRef` (a handle, not an inline key). `ChatToken` is
a single decoded text fragment for streaming.

### 2 — Add `el-cloud` adapter crate (excluded from offline workspace)

A new crate `crates/adapters/el-cloud` implements `LlmProvider` over HTTP
using **`reqwest`** (blocking client) with the OpenAI-compatible Chat
Completions API.

> **Amendment (2026-06-11)**: cloud egress is **native-only for now**.
> `reqwest`'s blocking client does not exist on `wasm32-unknown-unknown` —
> only the async `fetch`-backed surface does — and the synchronous
> `LlmProvider` trait cannot await it. The npm/web binding therefore exposes
> an explicit "unavailable" error from its `cloud` constructor instead of a
> working cloud path (see Negative consequences). Web support requires an
> async `LlmProvider` variant and is deferred.

Supported providers out of the box (via model-name prefix routing, same
pattern as `genai`):
- `openai/*` → OpenAI (api.openai.com)
- `anthropic/*` → Anthropic Messages API
- `gemini/*` → Google Generative AI
- `ollama/*` → local Ollama (OpenAI-compat, no key required)
- Any `http(s)://…` base URL → custom OpenAI-compat endpoint

### 3 — `el-engine-candle` also implements `LlmProvider`

The existing `CandleEngine` is wrapped to implement `LlmProvider` so the
host API is identical regardless of backend. Local path: no network, no key.
Cloud path: explicit `CredentialRef` required at `start_session()`.

### 4 — Credential handling

API keys are passed in at runtime via `CredentialRef` — never embedded in
the binary. On mobile, `CredentialRef` is a platform-keystore handle
(Android Keystore / iOS Keychain) resolved by the host app before calling
`start_session()`. The SDK never logs or persists credentials.

### 5 — Partial amendment to ADR-004

ADR-004's "no cloud endpoint" rule applies to the **default, air-gapped path**.
The `el-cloud` backend is a second, explicitly opt-in egress: the app must
construct an `ElCloud::new(provider, credential)` and pass it to `start_session`.
An app that never constructs `el-cloud` has zero network surface — the
air-gap guarantee is preserved by default. Apps using `el-cloud` emit a
`FrontierLlmConsulted` domain event for auditability (parallel to
`HybridRelayConsulted` in ADR-004).

## Consequences

### Positive
- Single `LlmProvider` trait: mobile apps swap local ↔ frontier backend
  with one constructor change, no API surface difference.
- Frontier LLMs work from both native binding surfaces (React Native,
  Flutter); the npm/web surface fails explicitly rather than silently
  (see Negative).
- Ollama as a provider means local models served by Ollama (on a dev machine
  or LAN server) are also reachable — useful for development and privacy-first
  enterprise deployments.
- Air-gap guarantee preserved for apps that don't opt in.

### Negative
- **No cloud on any wasm target yet.** Browser (`wasm32-unknown-unknown`):
  the sync `LlmProvider` trait cannot drive the async `fetch` API, so the
  el-ffi web `cloud` constructor throws an explicit error instead of
  connecting (amendment above). Server-side WASM (`wasm32-wasip1`): no
  network capability in Wasmtime's `wasip1` at all. Unlocking the browser
  path requires an async `LlmProvider` variant (or a worker-based blocking
  shim) — tracked as follow-up work.
- Grammar-constrained decoding (llguidance) against frontier LLMs is
  client-side only: sample → check mask → reject and resample. This adds
  latency per constrained token and is impractical for heavily constrained
  grammars with large vocabs (128k). Acceptable for simple JSON-output schemas.
- Adds `reqwest` (+ TLS) to the adapter dependency tree; `el-core` and the
  7 pure-Rust crates remain dependency-free.

### Neutral
- `tiktoken-rs` is used in `el-cloud` for token counting against OpenAI
  model vocabs (prompt compression budget controller from LLMLingua-2).

## Links
- Partially amends: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)
- Depends on: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (local path),
  [ADR-008](./ADR-008-implement-the-sdk-in-rust-instead-of-c-cpp.md)
- Related: [ADR-001](./ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md),
  [ADR-009](./ADR-009-flutter-rust-bridge-for-dart-bindings.md)
- Crates: `el-core` (trait), `crates/adapters/el-cloud` (new)
