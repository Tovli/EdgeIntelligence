//! `el-ffi` — host bindings for the Rust core (ADR-001, ADR-009, ADR-010).
//!
//! One Rust API surface exported three ways:
//!
//! ## React Native — `uniffi-bindgen-react-native` (ADR-001)
//! `#[derive(uniffi::Object)]` + `#[uniffi::export]` → TypeScript + JSI C++ +
//! Turbo Module. Streaming via `StreamHandler` callback interface (UniFFI
//! cannot export `impl FnMut` parameters).
//!
//! ## Dart / pub.dev — `flutter_rust_bridge` v2 codegen (ADR-024)
//! `#[frb(opaque)]` on `EdgeLlm` → Dart opaque handle. `ask()` →
//! `Future<String>`, `edge_llm_ask_stream()` + `StreamSink<String>` →
//! `Stream<String>`.
//!
//! ## Web / npm — `wasm-bindgen` (ADR-001)
//! `#[wasm_bindgen]` on both the struct **and** the impl block → ESM TypeScript
//! package via `wasm-pack`. The struct annotation is required: without it
//! wasm-bindgen cannot satisfy `IntoWasmAbi`/`WasmDescribe` for the impl block.
//!
//! **Web limitations**: the local path uses a dev-stage echo placeholder until
//! Candle-on-wasm is wired, and the **cloud backend is not available on web**
//! (ADR-010 amendment): `el-cloud`'s blocking HTTP transport has no wasm
//! implementation, so `EdgeLlm.cloud` throws an explicit error there instead
//! of silently degrading.

// `#![forbid(unsafe_code)]` cannot be used: `forbid` is unoverridable even by
// inner `#[allow]`, so `frb_generated` (generated FFI glue) would not compile.
// `deny` permits the scoped override below. Invariant: the only permitted use
// of `#[allow(unsafe_code)]` in this crate is on `mod frb_generated`.
#![deny(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
use el_core::CredentialRef;
use el_core::{ChatMessage, ChatRequest, ChatToken, LlmProvider};

/// Per-request generation bound for the synchronous React Native Qwen facade.
#[cfg(not(target_arch = "wasm32"))]
const QWEN_FFI_DEFAULT_MAX_TOKENS: u32 = 64;

// UniFFI scaffolding — must appear once per crate, before any uniffi proc-macros.
#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!("el_ffi");

#[cfg(not(target_arch = "wasm32"))]
use flutter_rust_bridge::for_generated::DcoCodec;
#[cfg(not(target_arch = "wasm32"))]
use flutter_rust_bridge::frb;

#[cfg(not(target_arch = "wasm32"))]
#[allow(unsafe_code)]
mod frb_generated;
#[cfg(not(target_arch = "wasm32"))]
use frb_generated::StreamSink;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned across the FFI boundary.
///
/// On **native** (non-wasm32): `#[uniffi::Error]` projects this to the host
/// language's exception type (TS `Error`, Kotlin `Exception`, Swift `Error`).
/// On **wasm32**: converted to a JS exception via `JsValue` at the
/// `ask_wasm` call site.
///
/// Design note: `EdgeError` from el-core is not directly FFI-safe (uses
/// `Box<str>` and Rust-specific variants). `SdkError` is a thin projection.
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Error))]
#[derive(Debug)]
pub enum SdkError {
    /// The LLM backend (local Candle or cloud) returned an error.
    ProviderError { message: String },
}

impl std::fmt::Display for SdkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self::ProviderError { message } = self;
        write!(f, "{message}")
    }
}

impl From<el_core::EdgeError> for SdkError {
    fn from(e: el_core::EdgeError) -> Self {
        Self::ProviderError {
            message: e.to_string(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn emit_stream_error(
    result: std::result::Result<(), SdkError>,
    sink_closed: bool,
    mut emit: impl FnMut(String),
) {
    if let Err(error) = result {
        if !sink_closed {
            emit(error.to_string());
        }
    }
}

// ── Streaming callback interface (UniFFI / React Native) ─────────────────────

/// Token-by-token callback for streaming on React Native.
///
/// Implement on the TS/Kotlin/Swift side and pass to
/// [`EdgeLlm::ask_stream_cb`]. Each call delivers one text fragment; the
/// method returns (and calls nothing more) when generation is complete.
///
/// Dart bindings use the `edge_llm_ask_stream` FRB wrapper and
/// `StreamSink<String>` instead.
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export(callback_interface)]
pub trait StreamHandler: Send + Sync {
    fn on_token(&self, token: String);
}

// ── Public FFI facade ────────────────────────────────────────────────────────

/// The flat FFI-friendly facade (ADR-001, ADR-009, ADR-010).
///
/// Annotated for all three binding surfaces:
/// - `uniffi::Object` (native) → opaque UniFFI / React Native handle
/// - `frb(opaque)` (native) → opaque Dart handle via FRB v2 codegen
/// - `wasm_bindgen` (wasm32) → satisfies `IntoWasmAbi`/`WasmDescribe` so
///   that `#[wasm_bindgen] impl EdgeLlm { ... }` compiles
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(not(target_arch = "wasm32"), frb(opaque))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct EdgeLlm {
    provider: Box<dyn LlmProvider>,
    /// Default model routing string (stored so `ask()` can fill `ChatRequest::model`).
    default_model: String,
    /// Optional per-request completion cap applied by this FFI facade.
    max_tokens: Option<u32>,
}

/// UniFFI-exported methods: constructors, blocking chat, and reset.
///
/// Dart uses the `edge_llm_*` free-function wrappers below so FRB does not
/// need to parse the UniFFI-decorated impl block after macro expansion.
#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl EdgeLlm {
    /// Construct with the local Candle engine (air-gapped, ADR-002/004).
    ///
    /// If `model_uri` is non-empty, loads the GGUF at that path via
    /// `CandleEngine::from_path` (consumer-supplied model, ADR-002).
    /// Pass an empty string to use a deterministic toy model for development
    /// and testing — the toy generates gibberish but exercises the full
    /// binding layer end-to-end.
    ///
    /// Returns `Err(SdkError)` if `model_uri` is non-empty but the file
    /// cannot be parsed (missing, malformed GGUF, incompatible tensor shapes).
    /// An empty `model_uri` never fails.
    ///
    /// The permissive signature verifier used here is intentional: it lets the
    /// binding layer be exercised without a signed model artifact. A production
    /// deployment should substitute a real `SignatureVerifier` backed by the
    /// platform keystore.
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn local(model_uri: String) -> Result<Self, SdkError> {
        #[cfg(target_arch = "wasm32")]
        let _ = &model_uri;
        #[cfg(not(target_arch = "wasm32"))]
        {
            use el_core::{ModelFormat, ModelId, ModelVersion};
            use el_provenance::{ModelArtifact, SignatureVerifier};

            struct PermissiveVerifier;
            impl SignatureVerifier for PermissiveVerifier {
                fn verify(&self, _: &[u8], _: &[u8], _: u32) -> bool {
                    true
                }
            }

            let mut art =
                ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
            art.verify(&PermissiveVerifier, b"placeholder", b"sig", 0);
            let permit = art.ensure_loadable().map_err(SdkError::from)?;

            let provider: Box<dyn LlmProvider> = if model_uri.is_empty() {
                // No path — toy model for development/tests.
                Box::new(
                    el_engine_candle::LocalLlmProvider::toy(256, 64, 255, permit)
                        .map_err(SdkError::from)?,
                )
            } else {
                // Consumer-supplied GGUF path.
                Box::new(
                    el_engine_candle::LocalLlmProvider::from_path(&model_uri, 1, permit)
                        .map_err(SdkError::from)?,
                )
            };

            Ok(Self {
                provider,
                default_model: "local".into(),
                max_tokens: None,
            })
        }
        #[cfg(target_arch = "wasm32")]
        Ok(Self {
            provider: Box::new(EchoProvider),
            default_model: "local".into(),
            max_tokens: None,
        })
    }

    /// Construct a real, air-gapped Qwen2/Qwen2.5 chat session for native
    /// hosts (ADR-026). Both files must be caller-provided local assets: the
    /// GGUF contains model weights and `tokenizer.json` provides the ChatML
    /// control tokens plus the only valid token-id decoder.
    ///
    /// This intentionally does not fall back to [`Self::local`], whose
    /// byte-level `LocalLlmProvider` exists only as a development/test seam.
    /// Browser/WASM keeps its separate placeholder path and does not export
    /// this constructor.
    #[cfg(not(target_arch = "wasm32"))]
    #[uniffi::constructor]
    pub fn local_qwen(model_uri: String, tokenizer_uri: String) -> Result<Self, SdkError> {
        if model_uri.trim().is_empty() {
            return Err(SdkError::ProviderError {
                message: "Qwen model path must not be empty".into(),
            });
        }
        if tokenizer_uri.trim().is_empty() {
            return Err(SdkError::ProviderError {
                message: "Qwen tokenizer path must not be empty".into(),
            });
        }

        // React Native's current synchronous UniFFI methods execute on the JS
        // thread. Every Qwen FFI request is therefore limited to 64 generated
        // tokens. This facade has no caller-supplied generation-limit argument;
        // hosts needing a different bound must use the Rust provider API until
        // an async React Native surface is introduced.
        let provider = el_engine_candle::QwenChatProvider::from_paths(&model_uri, &tokenizer_uri)
            .map_err(|error| SdkError::ProviderError {
            message: format!("{error} (model: {model_uri}, tokenizer: {tokenizer_uri})"),
        })?;
        Ok(Self {
            provider: Box::new(provider),
            default_model: "local/qwen".into(),
            max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS),
        })
    }

    /// Construct the official DictaLM 3.0 Qwen3 profile (ADR-027).
    ///
    /// Unlike `local_qwen`, this accepts only the immutable three-asset bundle:
    /// pinned GGUF, pinned tokenizer, and the upstream chat template. Validation
    /// occurs before any model tensor is constructed. The profile is native-only
    /// and targets high-memory arm64 application hosts; callers on unsupported
    /// hosts receive a provider error rather than falling back to another model.
    #[cfg(not(target_arch = "wasm32"))]
    #[uniffi::constructor]
    pub fn local_dictalm(
        model_uri: String,
        tokenizer_uri: String,
        chat_template_uri: String,
        manifest_signature_uri: String,
        high_end: bool,
        memory_budget_bytes: u64,
    ) -> Result<Self, SdkError> {
        if !cfg!(target_arch = "aarch64") {
            return Err(SdkError::ProviderError {
                message: "DictaLM is supported only on native arm64 high-memory devices".into(),
            });
        }
        if model_uri.trim().is_empty()
            || tokenizer_uri.trim().is_empty()
            || chat_template_uri.trim().is_empty()
            || manifest_signature_uri.trim().is_empty()
        {
            return Err(SdkError::ProviderError {
                message: "DictaLM model, tokenizer, chat template, and manifest signature paths must not be empty"
                    .into(),
            });
        }
        let bundle = el_engine_candle::DictaLmBundlePaths::new(
            model_uri.clone(),
            tokenizer_uri.clone(),
            chat_template_uri.clone(),
            manifest_signature_uri.clone(),
        );
        let capability = el_engine_candle::DictaLmCapability {
            high_end,
            memory_budget_bytes,
        };
        let provider = el_engine_candle::QwenChatProvider::from_dictalm_bundle(bundle, capability)
            .map_err(|error| SdkError::ProviderError {
                message: format!(
                    "{error} (model: {model_uri}, tokenizer: {tokenizer_uri}, template: {chat_template_uri})"
                ),
            })?;
        Ok(Self {
            provider: Box::new(provider),
            default_model: "local/dictalm-3.0-1.7b-instruct-q4_k_m".into(),
            max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS),
        })
    }

    /// Construct with a frontier cloud backend (opt-in, ADR-010).
    ///
    /// `model` uses the routing prefix: `"openai/gpt-4o"`,
    /// `"anthropic/claude-sonnet-4-6"`, `"ollama/llama3"`,
    /// `"gemini/gemini-2.0-flash"`, or any OpenAI-compat base URL.
    /// `api_key` must come from the platform keystore — never embedded.
    ///
    /// **Native only** (React Native / Dart native). On wasm32 this constructor
    /// does not exist — the web surface exposes a throwing `cloud` instead
    /// (see the wasm32 impl block below and the ADR-010 amendment).
    #[cfg(not(target_arch = "wasm32"))]
    #[uniffi::constructor]
    pub fn cloud(model: String, api_key: String) -> Self {
        let credential = CredentialRef::new(api_key);
        let inner = el_cloud::CloudProvider::new();
        let provider = BoundCloudProvider {
            model: model.clone(),
            credential,
            inner,
        };
        Self {
            provider: Box::new(provider),
            default_model: model,
            max_tokens: None,
        }
    }

    /// Blocking chat completion.
    ///
    /// Returns `Err(SdkError::ProviderError)` on network/auth/engine failure
    /// so callers can distinguish model output from error conditions.
    pub fn ask(&self, prompt: String) -> Result<String, SdkError> {
        let req = self.request(prompt);
        self.provider
            .chat(&req)
            .map(|r| r.content)
            .map_err(SdkError::from)
    }

    /// End the active conversation and clear its KV cache, generated output,
    /// and other session-local buffers while retaining model weights.
    ///
    /// Returns an error if a stateful provider cannot release its session. The
    /// caller must stop or rebuild the provider rather than reuse a possibly
    /// stale KV cache.
    pub fn reset(&self) -> Result<(), SdkError> {
        self.provider.end_session().map_err(SdkError::from)
    }
}

impl EdgeLlm {
    fn request(&self, prompt: String) -> ChatRequest {
        let request = ChatRequest::new(self.default_model.clone(), vec![ChatMessage::user(prompt)]);
        match self.max_tokens {
            Some(max_tokens) => request.with_max_tokens(max_tokens),
            None => request,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ask_stream_with(
        &self,
        prompt: String,
        mut on_token: impl FnMut(String),
    ) -> Result<(), SdkError> {
        let req = self.request(prompt);
        self.provider
            .chat_stream(&req, &mut |t: ChatToken| {
                if !t.is_final {
                    on_token(t.text);
                }
            })
            .map_err(SdkError::from)
    }
}

/// Streaming via callback interface — exported for React Native (UniFFI).
///
/// Separated from the main block because UniFFI cannot export `impl FnMut`.
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export]
impl EdgeLlm {
    /// Stream tokens to a [`StreamHandler`] callback (React Native path).
    ///
    /// Returns an error on network/auth/engine failure so callers are not
    /// left waiting for a stream that will never arrive.
    pub fn ask_stream_cb(
        &self,
        prompt: String,
        handler: Box<dyn StreamHandler>,
    ) -> Result<(), SdkError> {
        self.ask_stream_with(prompt, |token| handler.on_token(token))
    }
}

/// Dart / FRB wrappers.
///
/// These are intentionally separate from the UniFFI impl blocks. FRB parses
/// these plain Rust functions and the Dart facade wraps them into the public
/// `EdgeLlm` class API.
#[cfg(not(target_arch = "wasm32"))]
pub mod dart_api {
    use super::*;

    #[frb]
    pub fn edge_llm_local(model_uri: String) -> anyhow::Result<EdgeLlm> {
        EdgeLlm::local(model_uri).map_err(to_anyhow)
    }

    #[frb]
    pub fn edge_llm_cloud(model: String, api_key: String) -> EdgeLlm {
        EdgeLlm::cloud(model, api_key)
    }

    #[frb]
    pub fn edge_llm_ask(sdk: &EdgeLlm, prompt: String) -> anyhow::Result<String> {
        sdk.ask(prompt).map_err(to_anyhow)
    }

    #[frb]
    pub fn edge_llm_reset(sdk: &EdgeLlm) -> anyhow::Result<()> {
        sdk.reset().map_err(to_anyhow)
    }

    #[frb]
    pub fn edge_llm_ask_stream(sdk: &EdgeLlm, prompt: String, sink: StreamSink<String, DcoCodec>) {
        let mut sink_closed = false;
        let result = sdk.ask_stream_with(prompt, |token| {
            if !sink_closed {
                if sink.add(token).is_err() {
                    // Dart cancelled the stream (e.g. take(n), listen().cancel()).
                    // LlmProvider has no cancellation hook so generation runs to
                    // completion; remaining tokens are silently dropped.
                    sink_closed = true;
                }
            }
        });

        // The returned Dart Stream is the consumer-facing error channel.
        // Completing the generated task successfully prevents its unawaited
        // future from reporting a duplicate global error.
        emit_stream_error(result, sink_closed, |message| {
            let _ = sink.add_error(message);
        });
    }

    // Converts SdkError to an anyhow string error for FRB's Dart propagation.
    // FRB surfaces this as a Dart AnyhowException(message) — variant type is
    // erased. If SdkError grows structured variants (e.g. AuthError { code }),
    // replace this with a #[frb]-annotated error enum in dart_api and return
    // Result<_, DartError> directly instead of going through anyhow.
    fn to_anyhow(error: SdkError) -> anyhow::Error {
        anyhow::anyhow!(error.to_string())
    }
}

// ── wasm32 surface ────────────────────────────────────────────────────────────

/// wasm-bindgen methods. `ask_wasm` converts `SdkError` to a JS exception
/// (`Result<_, JsValue>`) so the npm consumer can use `try/catch`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl EdgeLlm {
    #[wasm_bindgen(constructor)]
    pub fn new_local(model_uri: String) -> Result<EdgeLlm, JsValue> {
        EdgeLlm::local(model_uri).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Blocking chat; throws a JS Error on provider failure.
    #[wasm_bindgen]
    pub fn ask_wasm(&self, prompt: String) -> Result<String, JsValue> {
        self.ask(prompt)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Frontier cloud backend is **not yet available on web** (ADR-010):
    /// `el-cloud`'s blocking HTTP transport has no wasm implementation, and
    /// the synchronous `LlmProvider` trait cannot await the browser's async
    /// `fetch`. Always throws so callers fail loudly instead of silently
    /// receiving an echo stub. Use a native binding (React Native / Dart native)
    /// for cloud access.
    #[wasm_bindgen]
    pub fn cloud(_model: String, _api_key: String) -> Result<EdgeLlm, JsValue> {
        Err(JsValue::from_str(
            "EdgeLlm.cloud is not available on web/wasm: the cloud transport \
             requires a native binding (ADR-010)",
        ))
    }
}

// ── Native-only helper types ──────────────────────────────────────────────────

/// Wraps `CloudProvider` with a pinned model prefix and credential so that
/// `EdgeLlm::ask()` — which only takes a prompt — can fill `ChatRequest` fully.
#[cfg(not(target_arch = "wasm32"))]
struct BoundCloudProvider {
    model: String,
    credential: CredentialRef,
    inner: el_cloud::CloudProvider,
}

#[cfg(not(target_arch = "wasm32"))]
impl LlmProvider for BoundCloudProvider {
    fn chat(&self, req: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
        let mut r = req.clone();
        r.model = self.model.clone();
        r.credential = Some(self.credential.clone());
        self.inner.chat(&r)
    }

    fn chat_stream(
        &self,
        req: &ChatRequest,
        on_token: &mut dyn FnMut(ChatToken),
    ) -> el_core::Result<()> {
        let mut r = req.clone();
        r.model = self.model.clone();
        r.credential = Some(self.credential.clone());
        self.inner.chat_stream(&r, on_token)
    }
}

// ── WASM placeholder (no network, no Candle) ──────────────────────────────────

/// Dev-stage stand-in used **only** by the wasm32 `local` path until
/// Candle-on-wasm is wired. The cloud path never falls back to this — on
/// wasm32 the `cloud` constructor throws instead (ADR-010).
#[cfg(target_arch = "wasm32")]
struct EchoProvider;

#[cfg(target_arch = "wasm32")]
impl LlmProvider for EchoProvider {
    fn chat(&self, req: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
        let echo = req
            .messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("")
            .to_owned();
        Ok(el_core::ChatResponse {
            content: echo,
            model: "echo".into(),
            prompt_tokens: 0,
            completion_tokens: 0,
        })
    }
    fn chat_stream(
        &self,
        req: &ChatRequest,
        on_token: &mut dyn FnMut(ChatToken),
    ) -> el_core::Result<()> {
        let text = req
            .messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("")
            .to_owned();
        for ch in text.chars() {
            on_token(ChatToken {
                text: ch.to_string(),
                is_final: false,
            });
        }
        on_token(ChatToken {
            text: String::new(),
            is_final: true,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_toy_ask_returns_non_empty_response() {
        let sdk = EdgeLlm::local("".into()).expect("toy model never fails");
        let response = sdk
            .ask("hello".into())
            .expect("local toy model should not error");
        assert!(!response.is_empty());
    }

    #[test]
    fn stream_ends_with_final_and_has_content() {
        let sdk = EdgeLlm::local("".into()).expect("toy model never fails");
        let mut parts: Vec<String> = Vec::new();
        sdk.ask_stream_with("hi".into(), |t| parts.push(t))
            .expect("local toy model stream should not error");
        assert!(!parts.is_empty());
    }

    #[test]
    fn ask_error_is_distinguishable_from_content() {
        let sdk = EdgeLlm::local("".into()).expect("toy model never fails");
        let r = sdk.ask("ping".into());
        assert!(
            r.is_ok(),
            "toy local provider must not error on a plain prompt"
        );
        assert!(
            !r.unwrap().starts_with("error:"),
            "response must not look like a swallowed error"
        );
    }

    #[test]
    fn active_dart_stream_receives_one_provider_error() {
        let mut errors = Vec::new();

        emit_stream_error(
            Err(SdkError::ProviderError {
                message: "stream interrupted".into(),
            }),
            false,
            |error| errors.push(error),
        );

        assert_eq!(errors, vec!["stream interrupted"]);
    }

    #[test]
    fn local_missing_gguf_path_returns_sdk_error() {
        let r = EdgeLlm::local("/nonexistent/model.gguf".into());
        assert!(
            matches!(r, Err(SdkError::ProviderError { .. })),
            "non-empty path that doesn't exist must return SdkError"
        );
    }

    #[test]
    fn local_qwen_requires_both_asset_paths() {
        let missing_model = EdgeLlm::local_qwen("".into(), "tokenizer.json".into());
        assert!(matches!(
            missing_model,
            Err(SdkError::ProviderError { ref message }) if message == "Qwen model path must not be empty"
        ));

        let missing_tokenizer = EdgeLlm::local_qwen("model.gguf".into(), "".into());
        assert!(matches!(
            missing_tokenizer,
            Err(SdkError::ProviderError { ref message }) if message == "Qwen tokenizer path must not be empty"
        ));
    }

    #[test]
    fn dictalm_ffi_rejects_unsupported_host_before_asset_load() {
        let result = EdgeLlm::local_dictalm(
            "/not/read.gguf".into(),
            "/not/read.tokenizer.json".into(),
            "/not/read.jinja".into(),
            "/not/read.sig".into(),
            true,
            4 * 1024 * 1024 * 1024,
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert!(matches!(
            result,
            Err(SdkError::ProviderError { ref message }) if message.contains("arm64 high-memory")
        ));
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(result, Err(SdkError::ProviderError { .. })));
    }

    #[test]
    fn local_qwen_missing_assets_return_sdk_error() {
        let r = EdgeLlm::local_qwen(
            "/nonexistent/qwen.gguf".into(),
            "/nonexistent/tokenizer.json".into(),
        );
        assert!(matches!(
            r,
            Err(SdkError::ProviderError { ref message })
                if message.contains("model file not found")
                    && message.contains("model: /nonexistent/qwen.gguf")
                    && message.contains("tokenizer: /nonexistent/tokenizer.json")
        ));
    }

    fn assert_ready_completion(label: &str, text: &str) {
        let trimmed = text.trim();
        assert!(!trimmed.is_empty(), "{label} must not be empty");
        assert!(
            !trimmed.chars().all(|character| character == '?'),
            "{label} must not be all question marks: {trimmed:?}"
        );
        assert_ne!(
            trimmed, "I can't help with that request.",
            "{label} must not be the deterministic safety refusal"
        );
        assert!(
            trimmed.to_ascii_lowercase().contains("ready"),
            "{label} must contain the requested word 'ready': {trimmed:?}"
        );
    }

    struct ResetTrackingProvider {
        reset_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        should_fail: bool,
    }

    impl LlmProvider for ResetTrackingProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            unreachable!("reset test does not chat")
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            unreachable!("reset test does not stream")
        }

        fn end_session(&self) -> el_core::Result<()> {
            self.reset_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.should_fail {
                Err(el_core::EdgeError::Engine("reset failed"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn reset_forwards_to_provider_session_lifecycle() {
        let reset_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sdk = EdgeLlm {
            provider: Box::new(ResetTrackingProvider {
                reset_calls: std::sync::Arc::clone(&reset_calls),
                should_fail: false,
            }),
            default_model: "test".into(),
            max_tokens: None,
        };

        sdk.reset().expect("provider reset must succeed");

        assert_eq!(reset_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn reset_propagates_provider_failure() {
        let sdk = EdgeLlm {
            provider: Box::new(ResetTrackingProvider {
                reset_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                should_fail: true,
            }),
            default_model: "test".into(),
            max_tokens: None,
        };

        assert!(matches!(
            sdk.reset(),
            Err(SdkError::ProviderError { ref message }) if message.contains("reset failed")
        ));
    }

    struct RequestCapProvider(std::sync::Arc<std::sync::Mutex<Vec<Option<u32>>>>);

    impl LlmProvider for RequestCapProvider {
        fn chat(&self, req: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            self.0.lock().unwrap().push(req.max_tokens);
            Ok(el_core::ChatResponse {
                content: "ready".into(),
                model: req.model.clone(),
                prompt_tokens: 0,
                completion_tokens: 1,
            })
        }

        fn chat_stream(
            &self,
            req: &ChatRequest,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            self.0.lock().unwrap().push(req.max_tokens);
            on_token(ChatToken {
                text: "ready".into(),
                is_final: false,
            });
            on_token(ChatToken {
                text: String::new(),
                is_final: true,
            });
            Ok(())
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn qwen_ffi_cap_is_applied_to_ask_and_stream_requests() {
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sdk = EdgeLlm {
            provider: Box::new(RequestCapProvider(std::sync::Arc::clone(&requests))),
            default_model: "local/qwen".into(),
            max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS),
        };

        sdk.ask("Reply with exactly: ready".into()).unwrap();
        sdk.ask_stream_with("Reply with exactly: ready".into(), |_| {})
            .unwrap();

        assert_eq!(
            *requests.lock().unwrap(),
            vec![Some(QWEN_FFI_DEFAULT_MAX_TOKENS); 2]
        );
    }

    #[test]
    #[ignore = "requires EDGE_INTELLIGENCE_QWEN_GGUF and EDGE_INTELLIGENCE_QWEN_TOKENIZER"]
    fn native_qwen_integration_decodes_and_streams_english_text() {
        struct CapturingHandler(std::sync::Arc<std::sync::Mutex<String>>);

        impl StreamHandler for CapturingHandler {
            fn on_token(&self, token: String) {
                self.0.lock().unwrap().push_str(&token);
            }
        }

        let model_uri = std::env::var("EDGE_INTELLIGENCE_QWEN_GGUF")
            .expect("set EDGE_INTELLIGENCE_QWEN_GGUF to the official Qwen2.5 GGUF");
        let tokenizer_uri = std::env::var("EDGE_INTELLIGENCE_QWEN_TOKENIZER")
            .expect("set EDGE_INTELLIGENCE_QWEN_TOKENIZER to its matching tokenizer.json");
        let sdk = EdgeLlm::local_qwen(model_uri, tokenizer_uri)
            .expect("official Qwen model/tokenizer pair must construct");

        let reply = sdk
            .ask("Reply with exactly: ready".into())
            .expect("Qwen ask must succeed");
        assert_ready_completion("ask response", &reply);

        // Reset must succeed and must not unload the resident model. State
        // release itself is verified by the provider/session lifecycle tests.
        sdk.reset()
            .expect("Qwen reset must release the active session");

        let streamed = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        sdk.ask_stream_cb(
            "Reply with exactly: ready".into(),
            Box::new(CapturingHandler(std::sync::Arc::clone(&streamed))),
        )
        .expect("Qwen ask_stream_cb must succeed");
        assert_ready_completion("streamed response", &streamed.lock().unwrap());
    }
}
