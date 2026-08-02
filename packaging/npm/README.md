# edge-intelligence-sdk

Web and React Native bindings for the Edge Intelligence SDK.

This package contains the WASM/TypeScript browser surface and a React Native
Turbo Module backed by generated UniFFI JSI bindings and native Android/iOS
libraries. The React Native package declares the generator runtime dependencies
it needs; applications should not install `@ubjs/core` separately.

## Usage

Copy the Qwen2.5 0.5B GGUF into app storage and pass that local path to the SDK
facade.

Browser / WASM:

```ts
import init, { EdgeLlm } from "edge-intelligence-sdk";

await init();

const qwen05b = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const sdk = new EdgeLlm(qwen05b);
const reply = sdk.ask_wasm("Summarize edge inference in one sentence.");

console.log(reply);
```

React Native:

```ts
import { localEdgeLlm } from "edge-intelligence-sdk";

const qwen05b = "/data/user/0/com.example.app/files/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const sdk = localEdgeLlm(qwen05b);

const reply = sdk.ask("Summarize edge inference in one sentence.");
let streamed = "";
sdk.askStreamCb("Give me two deployment tips.", {
  onToken(token) {
    streamed += token;
  },
});
```

For an opt-in cloud session, use the guarded `cloudEdgeLlm(model, apiKey)`
factory. React Native TypeScript consumers can import `EdgeLlmLike` and
`SdkError` as types when their resolver selects the `react-native` export
condition.

Migrating from `EdgeLlm.local()`: prerelease React Native consumers should
replace that value import and constructor call with `localEdgeLlm(modelPath)`.
`EdgeLlmLike` is the TypeScript session interface; `EdgeLlm` is also available
as a class type, not a runtime export.

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
supported. Store GGUF files in the application's writable documents/files
directory and pass that platform-local path to `localEdgeLlm`.

Source, crate documentation, and release notes live in the Edge Intelligence
repository.
