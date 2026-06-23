# edge-intelligence-sdk

Generated Web and React Native bindings for the Edge Intelligence SDK.

This package contains the WASM/TypeScript browser surface, React Native
TypeScript bindings, and native Android/iOS libraries assembled by the release
pipeline.

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

Source, crate documentation, and release notes live in the Edge Intelligence
repository.
