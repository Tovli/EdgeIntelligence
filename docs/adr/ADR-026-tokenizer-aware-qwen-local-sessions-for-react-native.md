# ADR-026: Tokenizer-aware Qwen local sessions for React Native

- **Status**: proposed
- **Date**: 2026-08-06
- **Deciders**:
- **Tags**: react-native, npm, qwen, tokenizer, bindings, on-device, api

## Context

The published React Native factory, `localEdgeLlm(modelUri)`, passes its single
GGUF path to `EdgeLlm::local`. That facade constructs
`LocalLlmProvider::from_path`, which is the byte-level Candle engine seam used
for development and binding tests. It formats messages as plain role-prefixed
text and reduces generated token IDs to bytes; non-printable bytes become `?`.

That behavior is unsuitable for a Qwen GGUF. Qwen token IDs must be decoded by
the tokenizer that belongs to the model, and instruction requests must be
rendered with Qwen2.5 ChatML. The repository already has the appropriate
provider: `QwenChatProvider::from_paths(model_path, tokenizer_path)` loads a
Qwen2 GGUF and its `tokenizer.json`, renders ChatML, tokenizes the prompt, and
decodes generated IDs through that tokenizer. It also retains the model and
session according to ADR-018.

The npm README nevertheless demonstrates `localEdgeLlm` with the official
Qwen2.5 0.5B Q4_K_M GGUF. A React Native application has no way to select the
Qwen provider or pass its matching tokenizer, so the documented example can
finish generation while producing an all-`?` response. This breaks the native
product boundary defined by ADR-025: a shipped, documented local session must
be usable without application-specific native patches.

## Decision

Make the production React Native local-inference path explicitly Qwen and
tokenizer-aware.

1. The native FFI facade exposes a Qwen-specific constructor, such as
   `EdgeLlm::local_qwen(model_uri, tokenizer_uri)`, backed by
   `QwenChatProvider::from_paths`.
2. The React Native public factory is
   `localEdgeLlm(modelUri, tokenizerUri)` and calls that constructor. Both paths
   are required, platform-local files; the production overload must reject an
   empty or unavailable path instead of silently choosing a toy or byte-level
   provider. The published one-path `localEdgeLlm(modelUri)` overload remains
   as deprecated source compatibility and throws a directive migration error.
3. The Qwen path renders Qwen2.5 ChatML and uses the supplied tokenizer for both
   prompt encoding and generated-token decoding. Construction fails clearly if
   the tokenizer cannot be read or lacks the required Qwen ChatML control tokens;
   it must not fall back to byte decoding.
4. `LocalLlmProvider` remains a test/demo engine seam. It is not a production
   React Native backend and must be removed from public Qwen examples and
   package documentation. Any compatibility-only `EdgeLlm::local(model_uri)`
   entrypoint is similarly not selected by `localEdgeLlm` for a user-supplied
   GGUF.
5. The model and tokenizer remain caller-provided local assets. The constructor
   adds no download, network lookup, or cloud fallback, preserving ADR-004's
   air-gapped default and the existing local load-gate behavior.
6. CI provisions the official Qwen2.5 0.5B GGUF and matching `tokenizer.json`
   for a native FFI fixture on Linux and macOS, which proves `ask`, `reset`,
   and `askStreamCb` produce the explicitly requested `ready` completion and
   reject both all-`?` output and the deterministic safety-refusal fallback.
   Android release validation additionally launches the packed Expo app against
   those assets, proving the generated React Native constructor and callback
   reach a real Qwen session. The iOS release job verifies the packaged
   framework's autolinking and simulator build; iOS real-model execution needs
   an app-sandbox asset fixture and is deferred to a separately scoped mobile
   test decision. Test assets are CI-provisioned rather than committed to the
   package. The download uses immutable Hugging Face revisions and verifies
   committed SHA-256 digests before a fixture is used.
7. The React Native compatibility `ask` and `askStreamCb` calls are synchronous.
   `askStreamCb` replays a completed reply because the runtime does not yet
   expose a per-token decode hook. Those legacy calls are capped at 64 generated
   tokens. ADR-027 subsequently added `askAsync` and `askStreamAsync`, which use
   a bounded native worker and the provider's normal generation default; true
   in-loop incremental streaming remains deferred to ADR-019.

The browser/WASM local placeholder is outside this decision; this ADR governs
the native React Native surface only.

## Consequences

### Positive

- The documented React Native Qwen example exercises a real chat provider,
  rather than a byte-level test seam.
- Host applications supply the model/tokenizer pair needed for correct ChatML
  prompting and token decoding without bespoke native code.
- Invalid model setup fails at construction with an actionable error instead of
  returning plausible-looking but corrupted output.
- The solution reuses the existing Qwen provider, its on-device safety pipeline,
  and ADR-018 resident-session behavior.

### Negative

- Production Qwen callers must migrate from the deprecated one-path overload
  to the two-path API and copy both local assets into app storage. The old
  overload remains source-compatible but throws a directive migration error.
- Because that one-path call changes runtime behavior, this migration must ship
  in the 0.4.0-or-later minor release rather than a 0.3.x patch release; release
  CI rejects older tags.
- Native host and Android release validation now have real-model integration
  fixtures, which add artifact provisioning time and platform test maintenance.
- The React Native factory has a deliberately bounded synchronous execution
  model until asynchronous, incremental delivery is separately designed.
- This initial factory intentionally supports Qwen2/Qwen2.5-compatible local
  assets; a future multi-model API needs its own model-family selection and
  compatibility decision.

### Neutral

- `LlmProvider`, `InferenceSession`, and the generated UniFFI/JSI bridge remain
  the ownership boundaries; only the provider selected by the public local
  factory changes.
- Dart and browser bindings are not changed by this ADR. They may adopt the
  Qwen-specific FFI constructor in a separately scoped compatibility update.
- Local model files continue to be owned and placed by the host application;
  the npm package does not distribute model weights or tokenizer assets.

## Links

- Issue: [#18](https://github.com/Tovli/EdgeIntelligence/issues/18)
- Builds on: [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md)
  (Candle/Qwen inference), [ADR-018](./ADR-018-persistent-model-instances-and-stateful-sessions.md)
  (resident Qwen provider), and [ADR-025](./ADR-025-react-native-expo-autolink-ready-native-distribution.md)
  (React Native package boundary).
- Constrained by: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md)
  (local-only assets and no implicit egress).
- Implementation seams: `crates/adapters/el-ffi` (`EdgeLlm`),
  `crates/adapters/el-engine-candle` (`QwenChatProvider`), and
  `packaging/npm/src/rn` (`localEdgeLlm`).
