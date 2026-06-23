# el-ffi — host bindings (React Native, Flutter, Web)

One Rust API surface exported three ways, so mobile and web apps can call the
SDK in their native idiom (ADR-001, ADR-009, ADR-010):

| Surface | Tool | Output |
|---------|------|--------|
| **React Native** | `uniffi-bindgen-react-native` | TypeScript + JSI C++ + Turbo Module |
| **Flutter** | `flutter_rust_bridge` v2 | Dart opaque handle, `Future`/`Stream` |
| **Web / npm** | `wasm-bindgen` | ESM TypeScript package via `wasm-pack` |

No `unsafe` (`#![forbid(unsafe_code)]`). The crate is `cdylib` + `staticlib` +
`lib` so each toolchain can link the form it needs.

## What it provides

- **`EdgeLlm`** — the flat, FFI-friendly facade, annotated for all three
  surfaces at once:
  - `EdgeLlm::local(model_uri)` — local Candle engine, air-gapped (ADR-002/004).
    An empty `model_uri` uses a deterministic toy model for development;
    a path loads a consumer-supplied GGUF.
  - `EdgeLlm::cloud(model, api_key)` — frontier cloud backend (opt-in, ADR-010).
    **Native only** — see the web limitation below. `api_key` must come from the
    platform keystore, never embedded.
  - `ask(prompt) -> Result<String, SdkError>` — blocking chat.
  - `ask_stream(prompt, |token| …)` — closure streaming (Flutter / FRB v2).
  - `ask_stream_cb(prompt, handler)` — `StreamHandler` callback streaming
    (React Native; UniFFI cannot export `impl FnMut`).
  - `reset()`.
- **`SdkError`** — a thin, FFI-safe projection of `el_core::EdgeError`
  (`el-core`'s `Box<str>`/Rust-specific variants are not FFI-safe). Projects to
  the host language's exception type, or a JS exception on wasm.
- **`StreamHandler`** — the React Native streaming callback interface.

## Usage (Rust side)

Use the Qwen2.5 0.5B GGUF as the local model URI when constructing the facade.
Native package bindings use the same model path after copying the file into app
storage. For full Rust ChatML/tokenizer control, use
`el_engine_candle::QwenChatProvider` directly.

```rust
use el_ffi::EdgeLlm;

const QWEN_0_5B_GGUF: &str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

let sdk = EdgeLlm::local(QWEN_0_5B_GGUF.into())?;
let reply = sdk.ask("Summarize edge inference in one sentence.".into())?;
assert!(!reply.is_empty());

// Streaming (Flutter / closure form):
sdk.ask_stream("Give me two deployment tips.".into(), |fragment| print!("{fragment}"))?;
# Ok::<(), el_ffi::SdkError>(())
```

## Usage (npm / web)

The npm package exposes the wasm-bindgen browser surface. The local web path
currently exercises the generated API shape while Candle-on-wasm is being wired;
native React Native builds load the GGUF through the same package.

```ts
import init, { EdgeLlm } from "edge-intelligence-sdk";

await init();

const qwen05b = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const sdk = new EdgeLlm(qwen05b);
const reply = sdk.ask_wasm("Summarize edge inference in one sentence.");

console.log(reply);
```

## Usage (React Native)

```ts
import { EdgeLlm } from "edge-intelligence-sdk";

const qwen05b = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const sdk = EdgeLlm.local(qwen05b);

const reply = sdk.ask("Summarize edge inference in one sentence.");
let streamed = "";
sdk.ask_stream_cb("Give me two deployment tips.", {
  on_token(token) {
    streamed += token;
  },
});
```

## Usage (pub.dev / Flutter)

```dart
import "package:edge_intelligence/edge_intelligence.dart";

final qwen05b = "/path/to/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
final sdk = await EdgeLlm.local(qwen05b);

final reply = await sdk.ask("Summarize edge inference in one sentence.");
await for (final token in sdk.askStream("Give me two deployment tips.")) {
  stdout.write(token);
}
```

## Building the bindings

The Rust binding *surfaces* compile on the host; the cross-target builds and
codegen run via the [`Makefile`](../../../Makefile):

```sh
make build-android    # cargo build --target aarch64-linux-android  (shared lib)
make build-ios        # cargo build --target aarch64-apple-ios       (static lib)
make build-wasm       # wasm-pack build → out/web ESM package

make codegen-rn       # React Native JSI bindings (needs build-android)
make codegen-flutter  # flutter_rust_bridge v2 Dart bindings
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
silently degrading. Use a native binding (React Native / Flutter) for cloud
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
[ADR-009](../../../docs/adr/ADR-009-flutter-rust-bridge-for-dart-bindings.md),
and [ADR-010](../../../docs/adr/ADR-010-unified-llm-provider-trait-with-opt-in-frontier-egress.md).
