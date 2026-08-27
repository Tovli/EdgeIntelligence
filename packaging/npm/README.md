# edge-intelligence-sdk

Web and React Native bindings for the Edge Intelligence SDK.

This package contains the WASM/TypeScript browser surface and a React Native
Turbo Module backed by generated UniFFI JSI bindings and native Android/iOS
libraries. The React Native package declares the generator runtime dependencies
it needs; applications should not install `@ubjs/core` separately.

## Usage

React Native Qwen inference requires two caller-provided local assets: the
Qwen2.5 GGUF and its matching official `tokenizer.json`. The package never
downloads either asset or falls back to a byte-level decoder.

Browser / WASM:

```ts
import init, { EdgeLlm } from "edge-intelligence-sdk";

await init();

const sdk = new EdgeLlm("/models/development-placeholder.gguf");
const reply = sdk.ask_wasm("Summarize edge inference in one sentence.");

console.log(reply);
```

The browser/WASM local path is still a development placeholder; it does not yet
run a caller-supplied Qwen GGUF.

React Native:

```ts
import { askAsync, askStreamAsync, localEdgeLlm } from "edge-intelligence-sdk";

const qwen05b = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const qwenTokenizer = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct.tokenizer.json";
const sdk = localEdgeLlm(qwen05b, qwenTokenizer);

const request = askAsync(sdk, "Summarize edge inference in one sentence.");
const reply = await request.response;
let streamed = "";
const stream = askStreamAsync(sdk, "Give me two deployment tips.", {
  onToken(token) {
    streamed += token;
  },
  onComplete() {},
  onError(error) { console.error(error); },
  onCancelled() {},
});
// stream.cancel();
```

`askAsync` and `askStreamAsync` use SDK-owned native workers, so generation
never runs on the JavaScript thread. A conversational handle admits one active
turn; a second turn or reset returns `Busy` instead of racing the session.
Stateless providers may run concurrently, up to two async requests per handle;
that capacity is not shared process-wide. Call `cancel()` to
request cooperative cancellation at the next prefill, decode, or safety
checkpoint boundary. The cancellation callback/promise rejection is prompt, but
a backend that does not return at that boundary keeps its handle `Busy` until
its stateful cleanup finishes. A cancellation that wins the race may follow
partial tokens; treat `onCancelled` as the terminal outcome, not `onComplete`.
A queued provider error retains that terminal outcome. A queued completion is
changed to cancellation when buffered token fragments are discarded.

`askAsync` never throws synchronously for native submission or binding-version
errors: its `response` promise rejects instead. `askStreamAsync` likewise
returns an inert request and invokes `onError` asynchronously for submission or
binding-version errors.

`ask` and `askStreamCb` remain synchronous compatibility calls. The current
local Candle and Qwen providers produce a complete safe reply before replaying
its fragments for both callback stream APIs, so `askStreamAsync` removes JS-thread
blocking but does not yet reduce time-to-first-token. ADR-019 tracks true in-loop
safe-token streaming. The legacy `ask` and `askStreamCb` Qwen replies are capped
at 64 generated tokens; callers should treat only those replies as
potentially length-limited. `askAsync` and `askStreamAsync` use the provider's
normal generation default rather than the JavaScript-thread compatibility cap.

`reset()` throws if the provider cannot clear its session cache. Treat that as
terminal for the handle and construct a new session instead of issuing another
prompt against possibly stale conversation state.

For an opt-in cloud session, use the guarded `cloudEdgeLlm(model, apiKey)`
factory. React Native TypeScript consumers can import `EdgeLlmLike` and
`SdkError` as types when their resolver selects the `react-native` export
condition. Native UniFFI bindings and this JavaScript wrapper must be released
and rebuilt together: the wrapper detects a native build without the async
methods and throws an explicit version-mismatch error instead of calling an
undefined native function.

Migrating from the published one-path `localEdgeLlm(modelPath)` API: its
deprecated overload remains available so existing TypeScript builds continue to
compile, but it throws a migration error before creating a session. Pass the
matching tokenizer as `localEdgeLlm(modelPath, tokenizerPath)`. The two-path
factory constructs the native Qwen provider, renders Qwen2.5 ChatML, and
decodes generated IDs through that tokenizer. `EdgeLlmLike` is the TypeScript
session interface; `EdgeLlm` is also available as a class type, not a runtime
export. Because the one-path call changes runtime behavior, this migration
ships in the 0.4.0-or-later minor release rather than a 0.3.x patch release.

## React Native and Expo setup

Bare React Native applications install the package normally and use React
Native autolinking:

```sh
npm install edge-intelligence-sdk
cd ios && pod install && cd .. # iOS only
```

Expo applications are supported through a custom development build or a
production build. Expo and React Native discover the package through native
autolinking metadata, so no Expo config plugin entry is required. Install the
package, then rebuild the native app:

```sh
npx expo prebuild
npx expo run:android # or: npx expo run:ios
```

Expo Go is not supported because it cannot load third-party native JSI modules.
After adding or upgrading this package, rebuild the development client before
running `npx expo start` again.

The verified support target is Expo SDK 57 / React Native 0.86 with React 19.2.
The React Native peer range retains 0.85 as expected compatibility, but release
CI does not currently exercise that older host version. Expo itself is not a
peer dependency because the package uses React Native's native autolinking.

Node.js and SSR are not supported runtimes. Use the browser/WASM export in a
browser-aware bundler or the React Native export in a native application.

The release supports Android `armeabi-v7a`, `arm64-v8a`, and `x86_64`, plus iOS
device and simulator slices in `el_ffi.xcframework`. Android x86 is not
supported. Store the GGUF and matching tokenizer in the application's writable
documents/files directory and pass both platform-local paths to `localEdgeLlm`.

Source, crate documentation, and release notes live in the Edge Intelligence
repository.
