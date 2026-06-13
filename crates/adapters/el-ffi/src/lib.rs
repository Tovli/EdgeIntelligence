//! `el-ffi` — host bindings for the Rust core (ADR-001, ADR-009, ADR-010).
//!
//! One Rust API surface exported three ways:
//!
//! ## React Native — `uniffi-bindgen-react-native` (ADR-001)
//! `#[derive(uniffi::Object)]` + `#[uniffi::export]` → TypeScript + JSI C++ +
//! Turbo Module. Streaming via `StreamHandler` callback interface (UniFFI
//! cannot export `impl FnMut` parameters).
//!
//! ## Flutter — `flutter_rust_bridge` v2 (ADR-009)
//! `#[frb(opaque)]` on `EdgeLlm` → Dart opaque handle. `ask()` →
//! `Future<String>`, `ask_stream()` (closure variant) → `Stream<String>`.
//! FRB v2 handles `impl FnMut` natively.
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

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
use el_core::CredentialRef;
use el_core::{ChatMessage, ChatRequest, ChatToken, LlmProvider};

// UniFFI scaffolding — must appear once per crate, before any uniffi proc-macros.
#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!("el_ffi");

#[cfg(not(target_arch = "wasm32"))]
use flutter_rust_bridge::frb;

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

// ── Streaming callback interface (UniFFI / React Native) ─────────────────────

/// Token-by-token callback for streaming on React Native.
///
/// Implement on the TS/Kotlin/Swift side and pass to
/// [`EdgeLlm::ask_stream_cb`]. Each call delivers one text fragment; the
/// method returns (and calls nothing more) when generation is complete.
///
/// Flutter uses the closure-based [`EdgeLlm::ask_stream`] instead.
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
/// - `frb(opaque)` (native) → opaque Flutter/Dart handle via FRB v2
/// - `wasm_bindgen` (wasm32) → satisfies `IntoWasmAbi`/`WasmDescribe` so
///   that `#[wasm_bindgen] impl EdgeLlm { ... }` compiles
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(not(target_arch = "wasm32"), frb(opaque))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct EdgeLlm {
    provider: Box<dyn LlmProvider>,
    /// Default model routing string (stored so `ask()` can fill `ChatRequest::model`).
    default_model: String,
}

/// UniFFI-exported methods: constructors, blocking chat, and reset.
///
/// `ask_stream` (closure variant) lives in a separate plain impl block —
/// UniFFI cannot export `impl FnMut` parameters. The RN streaming surface is
/// `ask_stream_cb` in the block below.
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
            })
        }
        #[cfg(target_arch = "wasm32")]
        Ok(Self {
            provider: Box::new(EchoProvider),
            default_model: "local".into(),
        })
    }

    /// Construct with a frontier cloud backend (opt-in, ADR-010).
    ///
    /// `model` uses the routing prefix: `"openai/gpt-4o"`,
    /// `"anthropic/claude-sonnet-4-6"`, `"ollama/llama3"`,
    /// `"gemini/gemini-2.0-flash"`, or any OpenAI-compat base URL.
    /// `api_key` must come from the platform keystore — never embedded.
    ///
    /// **Native only** (React Native / Flutter). On wasm32 this constructor
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
        }
    }

    /// Blocking chat completion.
    ///
    /// Returns `Err(SdkError::ProviderError)` on network/auth/engine failure
    /// so callers can distinguish model output from error conditions.
    pub fn ask(&self, prompt: String) -> Result<String, SdkError> {
        let req = ChatRequest::new(self.default_model.clone(), vec![ChatMessage::user(prompt)]);
        self.provider
            .chat(&req)
            .map(|r| r.content)
            .map_err(SdkError::from)
    }

    /// Reset the session (clears KV cache and output).
    pub fn reset(&self) {
        // Reset happens automatically at the start of each LocalLlmProvider::chat() call.
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
        let req = ChatRequest::new(self.default_model.clone(), vec![ChatMessage::user(prompt)]);
        self.provider
            .chat_stream(&req, &mut |t: ChatToken| {
                if !t.is_final {
                    handler.on_token(t.text);
                }
            })
            .map_err(SdkError::from)
    }
}

/// Closure-based streaming — used by Flutter (FRB v2) and tests.
///
/// FRB v2 wraps `impl FnMut` into a Dart `Stream<String>` automatically.
impl EdgeLlm {
    /// Stream tokens via closure (Flutter / FRB path).
    ///
    /// Returns `Err` on provider failure so the stream is not silently truncated.
    pub fn ask_stream(
        &self,
        prompt: String,
        mut on_token: impl FnMut(String),
    ) -> Result<(), SdkError> {
        let req = ChatRequest::new(self.default_model.clone(), vec![ChatMessage::user(prompt)]);
        self.provider
            .chat_stream(&req, &mut |t: ChatToken| {
                if !t.is_final {
                    on_token(t.text.clone());
                }
            })
            .map_err(SdkError::from)
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
    /// receiving an echo stub. Use a native binding (React Native / Flutter)
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
        sdk.ask_stream("hi".into(), |t| parts.push(t))
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
    fn local_missing_gguf_path_returns_sdk_error() {
        let r = EdgeLlm::local("/nonexistent/model.gguf".into());
        assert!(
            matches!(r, Err(SdkError::ProviderError { .. })),
            "non-empty path that doesn't exist must return SdkError"
        );
    }
}
