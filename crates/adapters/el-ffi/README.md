# el-ffi — host bindings (React Native, Dart, Web)

One Rust API surface exported three ways, so mobile and web apps can call the
SDK in their native idiom (ADR-001, ADR-009, ADR-010):

| Surface | Tool | Output |
|---------|------|--------|
| **React Native** | `uniffi-bindgen-react-native` | TypeScript + JSI C++ + Turbo Module |
| **Dart / Flutter / pub.dev** | `flutter_rust_bridge` v2 codegen | Dart opaque handle, `Future`/`Stream` |
| **Web / npm** | `wasm-bindgen` | ESM TypeScript package via `wasm-pack` |

Hand-written code denies `unsafe`; generated flutter_rust_bridge glue is the
only module allowed to contain FFI `unsafe`. The crate is `cdylib` +
`staticlib` + `lib` so each toolchain can link the form it needs.

## What it provides

- **`EdgeLlm`** — the flat, FFI-friendly facade, annotated for all three
  surfaces at once:
  - `EdgeLlm::local_qwen(model_uri, tokenizer_uri)` — native-only, real local
    Qwen2/Qwen2.5 chat (ADR-026). Both caller-provided paths are required: the
    provider renders ChatML and decodes IDs with `tokenizer.json`.
  - `EdgeLlm::local(model_uri)` — local Candle engine, air-gapped (ADR-002/004).
    An empty `model_uri` uses a deterministic toy model for development;
    a path loads a byte-level test/demo GGUF; it is not a Qwen production API.
  - `EdgeLlm::cloud(model, api_key)` — frontier cloud backend (opt-in, ADR-010).
    **Native only** — see the web limitation below. `api_key` must come from the
    platform keystore, never embedded.
  - `ask(prompt) -> Result<String, SdkError>` — blocking compatibility chat.
  - `ask_stream_cb(prompt, handler)` — `StreamHandler` callback streaming
    (React Native; UniFFI cannot export `impl FnMut`).
  - `ask_async(prompt, handler) -> AsyncRequest` and
    `ask_stream_async(prompt, handler) -> AsyncRequest` — bounded native-worker
    requests with cooperative cancellation (ADR-027).
  - `reset() -> Result<(), SdkError>` — clears a stateful session; on failure,
    discard or rebuild the handle rather than reuse a possibly stale KV cache.
- **React Native** — generated UniFFI bindings expose `EdgeLlm.localQwen`
  (production Qwen), `EdgeLlm.local` (development seam), `EdgeLlm.cloud`,
  blocking compatibility methods, and the `askAsync` / `askStreamAsync`
  native-worker surface.
- **Dart / Flutter / pub.dev wrappers** — `edge_llm_*` FRB functions wrapped by the Dart
  facade as `EdgeLlm.local`, `EdgeLlm.cloud`, `ask`, `askStream`, and `reset`.
- **`SdkError`** — a thin, FFI-safe projection of `el_core::EdgeError`
  (`el-core`'s `Box<str>`/Rust-specific variants are not FFI-safe). Projects to
  the host language's exception type, or a JS exception on wasm.
- **`AsyncRequest`** — an idempotent React Native cancellation handle; terminal
  callbacks are exactly-once. A cancelled callback can arrive before a backend
  that ignores cancellation has drained; that handle remains `Busy` until its
  provider exits, so synchronous re-entry never races a stateful session.

## Usage (Rust side)

Use the Qwen2.5 0.5B GGUF and its matching `tokenizer.json` when constructing a
native Qwen facade. React Native bindings use the same two paths after copying
both assets into app storage. For full Rust ChatML/tokenizer control, use
`el_engine_candle::QwenChatProvider` directly.

```rust
use el_ffi::EdgeLlm;

const QWEN_0_5B_GGUF: &str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const QWEN_0_5B_TOKENIZER: &str = "models/qwen2.5-0.5b-instruct.tokenizer.json";

let sdk = EdgeLlm::local_qwen(QWEN_0_5B_GGUF.into(), QWEN_0_5B_TOKENIZER.into())?;
let reply = sdk.ask("Summarize edge inference in one sentence.".into())?;
assert!(!reply.is_empty());
# Ok::<(), el_ffi::SdkError>(())
```

## Usage (npm / web)

The npm package exposes the wasm-bindgen browser surface. The local web path
currently exercises the generated API shape while Candle-on-wasm is being wired;
it does not load a caller-supplied Qwen GGUF.

```ts
import init, { EdgeLlm } from "edge-intelligence-sdk";

await init();

const sdk = new EdgeLlm("/models/development-placeholder.gguf");
const reply = sdk.ask_wasm("Summarize edge inference in one sentence.");

console.log(reply);
```

## Usage (React Native)

```ts
import { askAsync, askStreamAsync, localEdgeLlm } from "edge-intelligence-sdk";

const qwen05b = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const qwenTokenizer = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct.tokenizer.json";
const sdk = localEdgeLlm(qwen05b, qwenTokenizer);

const completion = askAsync(sdk, "Summarize edge inference in one sentence.");
const reply = await completion.response;
let streamed = "";
const stream = askStreamAsync(sdk, "Give me two deployment tips.", {
  onToken(token) {
    streamed += token;
  },
  onComplete() {},
  onError(error) { console.error(error); },
  onCancelled() {},
});
// stream.cancel(); // requests cleanup at the next runtime boundary
```

`ask` and `askStreamCb` remain synchronous compatibility seams. Use
`askAsync`/`askStreamAsync` for application requests: stateful providers accept
one active turn per handle, while stateless providers accept up to two native
async requests per handle. Capacity is isolated to that handle and returns a
typed `Busy` error rather than queuing unbounded work. A cancellation request is
cooperative, so its latency is bounded by the next prefill, decode, or safety
checkpoint boundary. `onCancelled` is delivered promptly, but a backend that
does not return at that boundary keeps its handle `Busy` until cleanup has
finished. The opt-in cloud transport bounds a stalled connection to 10 seconds,
an idle stream read to 60 seconds, and a non-streaming request to 120 seconds,
so it cannot hold that handle indefinitely. The legacy synchronous Qwen methods
retain their 64-token UI-thread cap; native-worker calls use the provider's
normal generation default. If cancellation wins the race, an async stream can already have
delivered partial tokens and its `onCancelled` callback is authoritative. A
queued provider error retains that outcome, but a queued completion is changed
to cancellation when buffered token fragments are discarded. The current local Candle and Qwen providers replay completed
text fragments for both callback stream APIs, so `askStreamAsync` moves work off
the host thread without yet improving time-to-first-token. ADR-019 remains the
authority for true in-loop safe-token streaming.

## Dart and Flutter

The existing Dart `EdgeLlm.local(modelUri)` binding remains a development/test
surface and does not yet expose the tokenizer-aware Qwen constructor. It must
not be used for Qwen GGUF chat until that binding receives a separately scoped
ADR-026 compatibility update.

Dart's `ask` and `askStream` execute as Flutter Rust Bridge asynchronous tasks,
not on the UI isolate. Cancelling a Dart subscription closes its FRB sink; the
runtime token is cancelled only after the sink refuses an append. The current
local Candle and Qwen providers infer the full reply before replaying it, so
this cannot interrupt inference or transport. Dart has no request-handle
cancellation yet. A concurrent `ask`, `askStream`, or `reset` on the same
handle receives `Busy` rather than racing the session.

## Native Qwen integration fixture

The ignored `native_qwen_integration_decodes_and_streams_english_text` test
verifies the real FFI facade against caller-provisioned Qwen assets. It exercises
`ask`, `reset`, and the React Native callback-shaped `ask_stream_cb` API,
requiring the requested `ready` completion, rejecting an all-`?` response, and
rejecting the deterministic safety-refusal fallback. CI downloads the official
Qwen2.5 0.5B GGUF and matching tokenizer from pinned revisions, verifies their
SHA-256 digests, and caches the validated fixture. To run it locally:

```sh
EDGE_INTELLIGENCE_QWEN_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
EDGE_INTELLIGENCE_QWEN_TOKENIZER=/path/to/tokenizer.json \
cargo test -p el-ffi native_qwen_integration -- --ignored
```

## Building the bindings

The Rust binding *surfaces* compile on the host; the cross-target builds and
codegen run via the [`Makefile`](../../../Makefile):

```sh
make build-android    # cargo build --target aarch64-linux-android  (shared lib)
make build-ios        # cargo build --target aarch64-apple-ios       (static lib)
make build-ios-xcframework # device + simulator framework for Flutter iOS
make build-wasm       # wasm-pack build → out/web ESM package

make codegen-rn       # React Native JSI bindings (needs build-android)
make codegen-dart     # flutter_rust_bridge v2 Dart bindings
make bindings         # all three surfaces
```

Prerequisites (rustup targets, NDK linker, `wasm-pack`,
`uniffi-bindgen-react-native`, `flutter_rust_bridge_codegen`) are documented in
the Makefile header.

## Web limitations

On `wasm32` the local path currently uses a dev-stage echo placeholder until
Candle-on-wasm is wired, and the **cloud backend is not available on web**
(ADR-010 amendment): `el-cloud`'s blocking HTTP transport has no wasm
implementation, so `EdgeLlm.cloud` throws an explicit error there instead of
silently degrading. Use a native binding (React Native / Dart native) for cloud
access.

## Status

Implemented and tested (native + `wasm32` compile). As a workspace member, the
host-target Rust surfaces build and test with the rest of the workspace
(`cargo test --workspace`); the Android / iOS / wasm cross-builds and binding
codegen run separately via the Makefile because those toolchains are installed
out-of-band.

---

Part of the [Edge Intelligence](../../../README.md) workspace. Realizes
[ADR-001](../../../docs/adr/ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md),
[ADR-024](../../../docs/adr/ADR-024-dart-only-platform-agnostic-pub-dev-sdk.md),
and [ADR-010](../../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md).
