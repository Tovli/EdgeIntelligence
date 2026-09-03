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
use el_core::CancellationToken;
#[cfg(not(target_arch = "wasm32"))]
use el_core::CredentialRef;
use el_core::{ChatMessage, ChatRequest, ChatToken, EdgeError, LlmProvider};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicUsize;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// Release mobile cdylibs rely on `catch_unwind` around both provider work and
// foreign callbacks. Keep this compile-time guard coupled to Cargo's release
// panic strategy so a future profile change cannot silently turn containment
// into a process abort.
#[cfg(all(not(target_arch = "wasm32"), not(panic = "unwind")))]
compile_error!("native el-ffi requires panic = \"unwind\" for FFI panic containment");

/// Per-request generation bound for the synchronous React Native Qwen facade.
#[cfg(not(target_arch = "wasm32"))]
const QWEN_FFI_DEFAULT_MAX_TOKENS: u32 = 64;

/// Backpressure bound between an inference worker and host token delivery.
#[cfg(not(target_arch = "wasm32"))]
const ASYNC_STREAM_EVENT_BUFFER: usize = 32;
/// Native-worker cap scoped to one SDK handle. It protects a stateless handle
/// from unbounded thread creation without letting an uncooperative provider on
/// one handle starve unrelated handles.
#[cfg(not(target_arch = "wasm32"))]
const MAX_ASYNC_REQUESTS_PER_HANDLE: usize = 2;

/// A fail-fast permit for one stateful `EdgeLlm` operation.
///
/// Holding this across a provider call serializes `ask`, streaming, reset, and
/// asynchronous worker requests on a single conversation handle.
struct OperationPermit {
    active: Arc<AtomicBool>,
}

impl Drop for OperationPermit {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

/// A per-handle permit for native asynchronous workers.
#[cfg(not(target_arch = "wasm32"))]
struct AsyncWorkerPermit {
    active: Arc<AtomicUsize>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AsyncWorkerPermit {
    fn acquire(active: &Arc<AtomicUsize>) -> Result<Self, SdkError> {
        loop {
            let current = active.load(Ordering::Acquire);
            if current >= MAX_ASYNC_REQUESTS_PER_HANDLE {
                return Err(SdkError::Busy {
                    message: "async request capacity reached for this SDK handle; retry after an active request completes".into(),
                });
            }
            if active
                .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(Self {
                    active: Arc::clone(active),
                });
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for AsyncWorkerPermit {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

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
    /// This stateful SDK handle or its per-handle async capacity is busy.
    Busy { message: String },
    /// The consumer cancelled a request at a cooperative runtime boundary.
    Cancelled { message: String },
}

impl std::fmt::Display for SdkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProviderError { message }
            | Self::Busy { message }
            | Self::Cancelled { message } => write!(f, "{message}"),
        }
    }
}

impl From<el_core::EdgeError> for SdkError {
    fn from(e: el_core::EdgeError) -> Self {
        match e {
            EdgeError::Cancelled => Self::Cancelled {
                message: "request cancelled".into(),
            },
            other => Self::ProviderError {
                message: other.to_string(),
            },
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

/// Completion callbacks for non-blocking single-response requests.
///
/// Exactly one terminal callback is delivered: `on_complete`, `on_error`, or
/// `on_cancelled`. A completion or error already queued before consumer
/// cancellation retains that terminal outcome.
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export(callback_interface)]
pub trait AsyncCompletionHandler: Send + Sync {
    fn on_complete(&self, response: String);
    fn on_error(&self, error: String);
    fn on_cancelled(&self);
}

/// Token and terminal callbacks for non-blocking streaming requests.
///
/// `on_token` is never called after a terminal callback. If cancellation wins
/// the race after one or more token callbacks, those fragments are a partial
/// response and `on_cancelled` is the terminal outcome; callers must not treat
/// them as a completed answer. A queued provider error retains its error
/// outcome. A queued completion is retained only when no buffered token
/// fragments are discarded; otherwise `on_cancelled` remains authoritative.
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export(callback_interface)]
pub trait AsyncStreamHandler: Send + Sync {
    fn on_token(&self, token: String);
    fn on_complete(&self);
    fn on_error(&self, error: String);
    fn on_cancelled(&self);
}

/// Cooperative cancellation handle for an FFI-owned asynchronous request.
///
/// Cancellation is observed by the runtime at prefill and decode boundaries;
/// it is idempotent and does not interrupt unsafe model-engine internals.
#[cfg(not(target_arch = "wasm32"))]
#[derive(uniffi::Object)]
pub struct AsyncRequest {
    /// The consumer-visible request signal. Internal delivery failures must not
    /// change this value, or `is_cancelled` could contradict an error callback.
    cancellation: CancellationToken,
    /// The token passed to the provider. It is also tripped for internal
    /// delivery failures so a cooperative provider can stop promptly.
    runtime_cancellation: CancellationToken,
    /// Wakes the delivery worker without polling when the consumer cancels.
    /// A full bounded token queue is handled on its next receive instead.
    cancel_notifier: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AsyncRequest {
    fn new(cancel_notifier: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            cancellation: CancellationToken::new(),
            runtime_cancellation: CancellationToken::new(),
            cancel_notifier: Mutex::new(Some(cancel_notifier)),
        }
    }

    fn runtime_cancellation(&self) -> CancellationToken {
        self.runtime_cancellation.clone()
    }

    /// Release the host-owned sender once delivery has a terminal outcome or
    /// the producer exits. This lets the receiver observe a genuine worker
    /// disconnect instead of keeping it alive for the FFI handle's lifetime.
    fn disarm_cancel_notifier(&self) {
        self.cancel_notifier
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export]
impl AsyncRequest {
    /// Request cooperative cancellation. Safe to call more than once.
    pub fn cancel(&self) {
        self.cancellation.cancel();
        self.runtime_cancellation.cancel();
        let notifier = self
            .cancel_notifier
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(Arc::clone);
        if let Some(notifier) = notifier {
            notifier();
        }
    }

    /// Returns whether cancellation has been requested by the consumer.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

/// Events crossing the bounded boundary between synchronous inference and host
/// callbacks. The delivery loop owns the host callback side so provider/session
/// code never invokes foreign code while it owns its state.
#[cfg(not(target_arch = "wasm32"))]
enum AsyncStreamEvent {
    Token(String),
    Complete,
    Cancelled,
    Error(String),
    CancellationRequested,
}

/// Terminal result for a non-streaming asynchronous request.
#[cfg(not(target_arch = "wasm32"))]
enum AsyncCompletionEvent {
    Complete(String),
    Cancelled,
    Error(String),
    CancellationRequested,
}

/// Consume already-buffered stream fragments after consumer cancellation without
/// invoking foreign callbacks. A queued error remains factual. A queued
/// completion cannot be delivered after silently discarding token fragments, so
/// it is downgraded to cancellation in that case. This preserves the terminal
/// contract without waiting for an uncooperative producer.
#[cfg(not(target_arch = "wasm32"))]
fn queued_stream_terminal(
    receiver: &std::sync::mpsc::Receiver<AsyncStreamEvent>,
) -> Option<AsyncStreamEvent> {
    let mut discarded_token = false;
    while let Ok(event) = receiver.try_recv() {
        match event {
            AsyncStreamEvent::Token(_) => discarded_token = true,
            AsyncStreamEvent::CancellationRequested => {}
            AsyncStreamEvent::Complete if discarded_token => {
                return Some(AsyncStreamEvent::Cancelled);
            }
            terminal => return Some(terminal),
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn cancellation_stream_terminal(
    receiver: &mut Option<std::sync::mpsc::Receiver<AsyncStreamEvent>>,
) -> (AsyncStreamEvent, bool) {
    let terminal = receiver.as_ref().and_then(queued_stream_terminal);
    drop(receiver.take());
    match terminal {
        Some(terminal) => (terminal, true),
        None => (AsyncStreamEvent::Cancelled, false),
    }
}

/// Consume a completion terminal already queued after a cancellation sentinel.
/// As with streams, an established provider result is factual and therefore
/// wins over a later cancellation without waiting for an uncooperative worker.
#[cfg(not(target_arch = "wasm32"))]
fn queued_completion_terminal(
    receiver: &std::sync::mpsc::Receiver<AsyncCompletionEvent>,
) -> Option<AsyncCompletionEvent> {
    while let Ok(event) = receiver.try_recv() {
        match event {
            AsyncCompletionEvent::CancellationRequested => {}
            terminal => return Some(terminal),
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn cancellation_completion_terminal(
    receiver: &mut Option<std::sync::mpsc::Receiver<AsyncCompletionEvent>>,
) -> (AsyncCompletionEvent, bool) {
    let terminal = receiver.as_ref().and_then(queued_completion_terminal);
    drop(receiver.take());
    match terminal {
        Some(terminal) => (terminal, true),
        None => (AsyncCompletionEvent::Cancelled, false),
    }
}

/// Foreign callback failures must not unwind an SDK worker or suppress its
/// terminal lifecycle attempt. There is no safe recovery after a terminal
/// callback itself panics, but catching it keeps permit cleanup deterministic.
#[cfg(not(target_arch = "wasm32"))]
fn invoke_host_callback(callback: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)).is_ok()
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
    provider: Arc<dyn LlmProvider>,
    /// Default model routing string (stored so `ask()` can fill `ChatRequest::model`).
    default_model: String,
    /// Optional per-request completion cap applied by this FFI facade.
    max_tokens: Option<u32>,
    /// One active turn per conversational handle. Contention is reported as
    /// `SdkError::Busy` rather than queued behind an unbounded backlog.
    operation_active: Arc<AtomicBool>,
    /// Number of native asynchronous requests currently accepted for this
    /// handle. Unlike session serialization, this also bounds stateless calls.
    #[cfg(not(target_arch = "wasm32"))]
    async_requests_active: Arc<AtomicUsize>,
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

            let provider: Arc<dyn LlmProvider> = if model_uri.is_empty() {
                // No path — toy model for development/tests.
                Arc::new(
                    el_engine_candle::LocalLlmProvider::toy(256, 64, 255, permit)
                        .map_err(SdkError::from)?,
                )
            } else {
                // Consumer-supplied GGUF path.
                Arc::new(
                    el_engine_candle::LocalLlmProvider::from_path(&model_uri, 1, permit)
                        .map_err(SdkError::from)?,
                )
            };

            Ok(Self {
                provider,
                default_model: "local".into(),
                max_tokens: None,
                operation_active: Arc::new(AtomicBool::new(false)),
                async_requests_active: Arc::new(AtomicUsize::new(0)),
            })
        }
        #[cfg(target_arch = "wasm32")]
        Ok(Self {
            provider: Arc::new(EchoProvider),
            default_model: "local".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
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

        // Legacy synchronous methods retain their bounded 64-token behavior.
        // SDK consumers that need non-blocking execution use `ask_async` or
        // `ask_stream_async`, which run on native worker threads.
        let provider = el_engine_candle::QwenChatProvider::from_paths(&model_uri, &tokenizer_uri)
            .map_err(|error| SdkError::ProviderError {
            message: format!("{error} (model: {model_uri}, tokenizer: {tokenizer_uri})"),
        })?;
        Ok(Self {
            provider: Arc::new(provider),
            default_model: "local/qwen".into(),
            max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS),
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
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
            provider: Arc::new(provider),
            default_model: model,
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Blocking chat completion.
    ///
    /// Returns `Err(SdkError::ProviderError)` on network/auth/engine failure
    /// so callers can distinguish model output from error conditions.
    pub fn ask(&self, prompt: String) -> Result<String, SdkError> {
        let _operation = self.try_start_operation()?;
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
        let _operation = self.try_start_operation()?;
        self.provider.end_session().map_err(SdkError::from)
    }
}

impl EdgeLlm {
    fn try_start_operation(&self) -> Result<Option<OperationPermit>, SdkError> {
        if !self.provider.requires_exclusive_turn() {
            return Ok(None);
        }
        self.operation_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| SdkError::Busy {
                message: "an operation is already active for this SDK handle".into(),
            })?;
        Ok(Some(OperationPermit {
            active: Arc::clone(&self.operation_active),
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn try_start_async_worker(&self) -> Result<AsyncWorkerPermit, SdkError> {
        AsyncWorkerPermit::acquire(&self.async_requests_active)
    }

    fn request(&self, prompt: String) -> ChatRequest {
        self.request_with_max_tokens(prompt, self.max_tokens)
    }

    /// The legacy synchronous facade is capped to avoid blocking a host UI for
    /// an unbounded reply. Native-worker requests do not inherit that transport
    /// workaround; the provider's own default generation policy remains active.
    #[cfg(not(target_arch = "wasm32"))]
    fn async_request(&self, prompt: String) -> ChatRequest {
        self.request_with_max_tokens(prompt, None)
    }

    fn request_with_max_tokens(&self, prompt: String, max_tokens: Option<u32>) -> ChatRequest {
        let request = ChatRequest::new(self.default_model.clone(), vec![ChatMessage::user(prompt)]);
        match max_tokens {
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
        let _operation = self.try_start_operation()?;
        let req = self.request(prompt);
        self.provider
            .chat_stream(&req, &mut |t: ChatToken| {
                if !t.is_final {
                    on_token(t.text);
                }
            })
            .map_err(SdkError::from)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ask_stream_with_cancellation(
        &self,
        prompt: String,
        cancellation: &CancellationToken,
        mut on_token: impl FnMut(String),
    ) -> Result<(), SdkError> {
        let _operation = self.try_start_operation()?;
        let req = self.request(prompt);
        self.provider
            .chat_stream_cancellable(&req, cancellation, &mut |t: ChatToken| {
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

    /// Begin a non-blocking completion on a native FFI worker.
    ///
    /// The request is rejected with `SdkError::Busy` when this conversational
    /// stateful handle already has an active turn. Stateless providers may
    /// accept concurrent calls on one SDK handle. Exactly one terminal callback
    /// is delivered after acceptance.
    /// If a backend cannot stop immediately, `on_cancelled` is delivered
    /// promptly but this handle remains Busy until its provider exits and its
    /// stateful session is safe to reuse.
    pub fn ask_async(
        &self,
        prompt: String,
        handler: Box<dyn AsyncCompletionHandler>,
    ) -> Result<Arc<AsyncRequest>, SdkError> {
        let operation = self.try_start_operation()?;
        let worker = self.try_start_async_worker()?;
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let cancellation_sender = sender.clone();
        let request = Arc::new(AsyncRequest::new(Arc::new(move || {
            let _ = cancellation_sender.try_send(AsyncCompletionEvent::CancellationRequested);
        })));
        let producer_request = Arc::clone(&request);
        let delivery_request = Arc::clone(&request);
        let runtime_cancellation = request.runtime_cancellation();
        let provider = Arc::clone(&self.provider);
        let chat_request = self.async_request(prompt);

        std::thread::Builder::new()
            .name("edge-intelligence-request".into())
            .spawn(move || {
                let (ack_sender, ack_receiver) = std::sync::mpsc::sync_channel(0);
                let producer = std::thread::Builder::new()
                    .name("edge-intelligence-request-producer".into())
                    .spawn(move || {
                        // The provider owns these per-handle permits until it
                        // actually exits. A cancellation callback may be delivered
                        // first, but later calls still fail fast with Busy
                        // until stateful cleanup has completed.
                        let operation_permit = operation;
                        let worker_permit = worker;
                        let mut response = String::new();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            provider.chat_stream_cancellable(
                                &chat_request,
                                &runtime_cancellation,
                                &mut |token: ChatToken| {
                                    if !token.is_final {
                                        response.push_str(&token.text);
                                    }
                                },
                            )
                        }));
                        let terminal = match result {
                            Ok(Ok(())) if runtime_cancellation.is_cancelled() => {
                                AsyncCompletionEvent::Cancelled
                            }
                            Ok(Ok(())) => AsyncCompletionEvent::Complete(response),
                            Ok(Err(EdgeError::Cancelled)) => AsyncCompletionEvent::Cancelled,
                            Ok(Err(error)) => {
                                AsyncCompletionEvent::Error(SdkError::from(error).to_string())
                            }
                            Err(_) => {
                                AsyncCompletionEvent::Error("async request worker panicked".into())
                            }
                        };
                        producer_request.disarm_cancel_notifier();
                        if sender.send(terminal).is_ok() {
                            // Keep the permits through terminal callback
                            // delivery; dropping the receiver/ack sender after
                            // early cancellation lets this exit without a join.
                            let _ = ack_receiver.recv();
                        }
                        drop(operation_permit);
                        drop(worker_permit);
                    });

                if let Err(error) = producer {
                    delivery_request.disarm_cancel_notifier();
                    let _ = invoke_host_callback(|| {
                        handler.on_error(format!("failed to start async request worker: {error}"))
                    });
                    return;
                }

                let mut receiver = Some(receiver);
                let (terminal, needs_ack) = match receiver
                    .as_ref()
                    .expect("receiver lives until a terminal event")
                    .recv()
                {
                    Ok(AsyncCompletionEvent::CancellationRequested) => {
                        cancellation_completion_terminal(&mut receiver)
                    }
                    Ok(event) => (event, true),
                    Err(_) => (
                        AsyncCompletionEvent::Error(
                            "async request worker exited without a terminal event".into(),
                        ),
                        false,
                    ),
                };

                match terminal {
                    AsyncCompletionEvent::Complete(response) => {
                        let _ = invoke_host_callback(|| handler.on_complete(response));
                    }
                    AsyncCompletionEvent::Cancelled => {
                        let _ = invoke_host_callback(|| handler.on_cancelled());
                    }
                    AsyncCompletionEvent::Error(error) => {
                        let _ = invoke_host_callback(|| handler.on_error(error));
                    }
                    AsyncCompletionEvent::CancellationRequested => {
                        unreachable!("cancellation notifications are handled in the receive loop")
                    }
                }
                if needs_ack {
                    let _ = ack_sender.send(());
                }
                delivery_request.disarm_cancel_notifier();
            })
            .map_err(|error| SdkError::ProviderError {
                message: format!("failed to start async request worker: {error}"),
            })?;

        Ok(request)
    }

    /// Begin a non-blocking token stream on native FFI workers.
    ///
    /// A bounded channel forwards provider-emitted tokens to a delivery worker,
    /// so host callbacks never run under a provider/session lock. Slow callbacks
    /// apply backpressure instead of accumulating an unbounded token buffer. The
    /// host binding is responsible for dispatching callbacks onto any UI-specific
    /// executor. Providers that only replay a completed response (including the
    /// current local Candle/Qwen adapters) do not improve time-to-first-token;
    /// ADR-019 owns true in-loop safe-token streaming. If a backend cannot stop
    /// immediately after cancellation, `on_cancelled` is delivered promptly but
    /// the handle remains Busy until provider cleanup is complete.
    pub fn ask_stream_async(
        &self,
        prompt: String,
        handler: Box<dyn AsyncStreamHandler>,
    ) -> Result<Arc<AsyncRequest>, SdkError> {
        let operation = self.try_start_operation()?;
        let worker = self.try_start_async_worker()?;
        let (sender, receiver) = std::sync::mpsc::sync_channel(ASYNC_STREAM_EVENT_BUFFER);
        let cancellation_sender = sender.clone();
        let request = Arc::new(AsyncRequest::new(Arc::new(move || {
            let _ = cancellation_sender.try_send(AsyncStreamEvent::CancellationRequested);
        })));
        let producer_request = Arc::clone(&request);
        let delivery_request = Arc::clone(&request);
        let user_cancellation = request.cancellation.clone();
        let runtime_cancellation = request.runtime_cancellation();
        let provider = Arc::clone(&self.provider);
        let chat_request = self.async_request(prompt);

        std::thread::Builder::new()
            .name("edge-intelligence-stream".into())
            .spawn(move || {
                let (ack_sender, ack_receiver) = std::sync::mpsc::sync_channel(0);
                let producer_cancellation = runtime_cancellation.clone();
                let producer = std::thread::Builder::new()
                    .name("edge-intelligence-stream-producer".into())
                    .spawn(move || {
                        // Keep the stateful session and this handle's worker
                        // slot owned by inference until it exits. Cancellation
                        // can notify the host promptly, but safe reuse remains
                        // unavailable until this cleanup completes.
                        let operation_permit = operation;
                        let worker_permit = worker;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            provider.chat_stream_cancellable(
                                &chat_request,
                                &producer_cancellation,
                                &mut |token: ChatToken| {
                                    if !token.is_final
                                        && sender.send(AsyncStreamEvent::Token(token.text)).is_err()
                                    {
                                        producer_cancellation.cancel();
                                    }
                                },
                            )
                        }));
                        let terminal = match result {
                            Ok(Ok(())) if producer_cancellation.is_cancelled() => {
                                AsyncStreamEvent::Cancelled
                            }
                            Ok(Ok(())) => AsyncStreamEvent::Complete,
                            Ok(Err(EdgeError::Cancelled)) => AsyncStreamEvent::Cancelled,
                            Ok(Err(error)) => {
                                AsyncStreamEvent::Error(SdkError::from(error).to_string())
                            }
                            Err(_) => {
                                AsyncStreamEvent::Error("async stream worker panicked".into())
                            }
                        };
                        producer_request.disarm_cancel_notifier();
                        if sender.send(terminal).is_ok() {
                            let _ = ack_receiver.recv();
                        }
                        drop(operation_permit);
                        drop(worker_permit);
                    });

                match producer {
                    Ok(_) => {}
                    Err(error) => {
                        delivery_request.disarm_cancel_notifier();
                        let _ = invoke_host_callback(|| {
                            handler
                                .on_error(format!("failed to start async stream worker: {error}"))
                        });
                        return;
                    }
                };

                let mut receiver = Some(receiver);
                let (terminal, needs_ack) = loop {
                    match receiver
                        .as_ref()
                        .expect("receiver lives until a terminal event")
                        .recv()
                    {
                        Ok(AsyncStreamEvent::Token(token)) => {
                            // If the token queue was full when cancellation was
                            // requested, the notifier could not enqueue its
                            // sentinel. Observe the atomic before invoking host
                            // code so that the next receive still cancels
                            // promptly without polling.
                            if user_cancellation.is_cancelled() {
                                break cancellation_stream_terminal(&mut receiver);
                            }
                            if !invoke_host_callback(|| handler.on_token(token)) {
                                runtime_cancellation.cancel();
                                drop(receiver.take());
                                break (
                                    AsyncStreamEvent::Error(
                                        "async stream token callback panicked".into(),
                                    ),
                                    false,
                                );
                            }
                        }
                        Ok(AsyncStreamEvent::CancellationRequested) => {
                            break cancellation_stream_terminal(&mut receiver);
                        }
                        Ok(event) => {
                            break (event, true);
                        }
                        Err(_) => {
                            runtime_cancellation.cancel();
                            break (
                                AsyncStreamEvent::Error(
                                    "async stream worker exited without a terminal event".into(),
                                ),
                                false,
                            );
                        }
                    }
                };

                match terminal {
                    AsyncStreamEvent::Complete => {
                        let _ = invoke_host_callback(|| handler.on_complete());
                    }
                    AsyncStreamEvent::Cancelled => {
                        let _ = invoke_host_callback(|| handler.on_cancelled());
                    }
                    AsyncStreamEvent::Error(error) => {
                        let _ = invoke_host_callback(|| handler.on_error(error));
                    }
                    AsyncStreamEvent::Token(_) => unreachable!("token events are handled above"),
                    AsyncStreamEvent::CancellationRequested => {
                        unreachable!("cancellation notifications are handled in the receive loop")
                    }
                }
                if needs_ack {
                    let _ = ack_sender.send(());
                }
                delivery_request.disarm_cancel_notifier();
            })
            .map_err(|error| SdkError::ProviderError {
                message: format!("failed to start async stream worker: {error}"),
            })?;

        Ok(request)
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
        let cancellation = CancellationToken::new();
        let sink_cancellation = cancellation.clone();
        let result = sdk.ask_stream_with_cancellation(prompt, &cancellation, |token| {
            if !sink_closed {
                if sink.add(token).is_err() {
                    // Dart cancelled the stream (e.g. take(n), listen().cancel()).
                    // This only stops delivery through the current FRB surface.
                    // Replay-style local providers have already finished
                    // inference before their first sink write, so it is not an
                    // inference-cancellation guarantee.
                    sink_closed = true;
                    sink_cancellation.cancel();
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
    fn requires_exclusive_turn(&self) -> bool {
        false
    }

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
    fn requires_exclusive_turn(&self) -> bool {
        false
    }

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

    static ASYNC_WORKER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const ASYNC_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

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
    fn queued_stream_error_beats_a_later_consumer_cancellation() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        sender
            .send(AsyncStreamEvent::Token("partial".into()))
            .unwrap();
        sender
            .send(AsyncStreamEvent::Error("provider failed".into()))
            .unwrap();
        let mut receiver = Some(receiver);

        let (terminal, needs_ack) = cancellation_stream_terminal(&mut receiver);

        assert!(needs_ack, "the producer is waiting for terminal delivery");
        assert!(matches!(
            terminal,
            AsyncStreamEvent::Error(error) if error == "provider failed"
        ));
        assert!(receiver.is_none());
    }

    #[test]
    fn queued_stream_completion_after_discarded_tokens_is_cancelled() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        sender
            .send(AsyncStreamEvent::Token("partial".into()))
            .unwrap();
        sender.send(AsyncStreamEvent::Complete).unwrap();
        let mut receiver = Some(receiver);

        let (terminal, needs_ack) = cancellation_stream_terminal(&mut receiver);

        assert!(needs_ack, "the producer is waiting for terminal delivery");
        assert!(matches!(terminal, AsyncStreamEvent::Cancelled));
        assert!(receiver.is_none());
    }

    #[test]
    fn queued_completion_error_beats_a_later_consumer_cancellation() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        sender
            .send(AsyncCompletionEvent::CancellationRequested)
            .unwrap();
        sender
            .send(AsyncCompletionEvent::Error("provider failed".into()))
            .unwrap();
        let mut receiver = Some(receiver);

        let (terminal, needs_ack) = cancellation_completion_terminal(&mut receiver);

        assert!(needs_ack, "the producer is waiting for terminal delivery");
        assert!(matches!(
            terminal,
            AsyncCompletionEvent::Error(error) if error == "provider failed"
        ));
        assert!(receiver.is_none());
    }

    struct StatelessTestProvider;

    impl LlmProvider for StatelessTestProvider {
        fn requires_exclusive_turn(&self) -> bool {
            false
        }

        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            Ok(el_core::ChatResponse {
                content: "ready".into(),
                model: "test".into(),
                prompt_tokens: 0,
                completion_tokens: 1,
            })
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
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

    #[test]
    fn stateless_provider_skips_the_exclusive_turn_reservation() {
        let sdk = EdgeLlm {
            provider: Arc::new(StatelessTestProvider),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        assert!(sdk.try_start_operation().unwrap().is_none());
        assert!(!sdk.operation_active.load(Ordering::Acquire));
    }

    struct BlockingCancellableProvider {
        started: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl LlmProvider for BlockingCancellableProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            Ok(el_core::ChatResponse {
                content: "ready".into(),
                model: "test".into(),
                prompt_tokens: 0,
                completion_tokens: 1,
            })
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
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

        fn chat_stream_cancellable(
            &self,
            _: &ChatRequest,
            cancellation: &CancellationToken,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            let (started, ready) = &*self.started;
            *started.lock().unwrap() = true;
            ready.notify_all();
            while !cancellation.is_cancelled() {
                let (lock, _) = ready
                    .wait_timeout(
                        started.lock().unwrap(),
                        std::time::Duration::from_millis(10),
                    )
                    .unwrap();
                drop(lock);
            }
            Err(EdgeError::Cancelled)
        }
    }

    struct TerminalRecordingStreamHandler {
        terminal: std::sync::Arc<(std::sync::Mutex<Option<String>>, std::sync::Condvar)>,
        terminal_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl AsyncStreamHandler for TerminalRecordingStreamHandler {
        fn on_token(&self, _: String) {
            panic!("a cancelled provider must not emit a token")
        }

        fn on_complete(&self) {
            self.record("complete");
        }

        fn on_error(&self, error: String) {
            self.record(&format!("error:{error}"));
        }

        fn on_cancelled(&self) {
            self.record("cancelled");
        }
    }

    impl TerminalRecordingStreamHandler {
        fn record(&self, outcome: &str) {
            self.terminal_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (terminal, done) = &*self.terminal;
            *terminal.lock().unwrap() = Some(outcome.into());
            done.notify_all();
        }
    }

    struct CancelsBeforeSuccessProvider;

    impl LlmProvider for CancelsBeforeSuccessProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            unreachable!("the async completion test uses the cancellable stream seam")
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            unreachable!("the async completion test uses the cancellable stream seam")
        }

        fn chat_stream_cancellable(
            &self,
            _: &ChatRequest,
            cancellation: &CancellationToken,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            // Models/providers that observe cancellation only between their
            // internal units may still finish with Ok. The FFI must project
            // that race as Cancelled, not as a successful response.
            cancellation.cancel();
            Ok(())
        }
    }

    struct CompletionRecordingHandler {
        terminal: std::sync::Arc<(std::sync::Mutex<Option<String>>, std::sync::Condvar)>,
    }

    impl AsyncCompletionHandler for CompletionRecordingHandler {
        fn on_complete(&self, response: String) {
            self.record(&format!("complete:{response}"));
        }

        fn on_error(&self, error: String) {
            self.record(&format!("error:{error}"));
        }

        fn on_cancelled(&self) {
            self.record("cancelled");
        }
    }

    impl CompletionRecordingHandler {
        fn record(&self, outcome: &str) {
            let (terminal, done) = &*self.terminal;
            *terminal.lock().unwrap() = Some(outcome.into());
            done.notify_all();
        }
    }

    fn wait_for_terminal(
        terminal: &std::sync::Arc<(std::sync::Mutex<Option<String>>, std::sync::Condvar)>,
    ) -> String {
        let (terminal_lock, terminal_done) = &**terminal;
        let mut outcome = terminal_lock.lock().unwrap();
        while outcome.is_none() {
            let (next, timeout) = terminal_done
                .wait_timeout(outcome, ASYNC_TEST_TIMEOUT)
                .unwrap();
            assert!(!timeout.timed_out(), "async request did not finish");
            outcome = next;
        }
        outcome.clone().unwrap()
    }

    fn wait_until_idle(sdk: &EdgeLlm) {
        let deadline = std::time::Instant::now() + ASYNC_TEST_TIMEOUT;
        while sdk
            .operation_active
            .load(std::sync::atomic::Ordering::Acquire)
            || sdk
                .async_requests_active
                .load(std::sync::atomic::Ordering::Acquire)
                != 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "async provider did not release its per-handle permits"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn async_completion_projects_post_provider_cancellation_as_cancelled() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let terminal =
            std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
        let sdk = EdgeLlm {
            provider: Arc::new(CancelsBeforeSuccessProvider),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        let request = sdk
            .ask_async(
                "cancel".into(),
                Box::new(CompletionRecordingHandler {
                    terminal: std::sync::Arc::clone(&terminal),
                }),
            )
            .unwrap();

        assert_eq!(wait_for_terminal(&terminal), "cancelled");
        wait_until_idle(&sdk);
        assert!(
            !request.is_cancelled(),
            "provider-level cancellation must not masquerade as consumer cancellation"
        );
    }

    struct IgnoresCancellationProvider {
        started: Signal,
        release: Signal,
    }

    impl LlmProvider for IgnoresCancellationProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            unreachable!("the cancellation-liveness test uses the stream seam")
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            unreachable!("the cancellation-liveness test uses the stream seam")
        }

        fn chat_stream_cancellable(
            &self,
            _: &ChatRequest,
            _: &CancellationToken,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            let (started, started_ready) = &*self.started;
            *started.lock().unwrap() = true;
            started_ready.notify_all();

            let (released, release_ready) = &*self.release;
            let mut is_released = released.lock().unwrap();
            while !*is_released {
                let (next, timeout) = release_ready
                    .wait_timeout(is_released, ASYNC_TEST_TIMEOUT)
                    .unwrap();
                if timeout.timed_out() {
                    return Err(EdgeError::Engine("test provider was never released"));
                }
                is_released = next;
            }
            drop(is_released);

            on_token(ChatToken {
                text: "late".into(),
                is_final: false,
            });
            Ok(())
        }
    }

    fn wait_for_signal(signal: &Signal, message: &str) {
        let (state, ready) = &**signal;
        let mut set = state.lock().unwrap();
        while !*set {
            let (next, timeout) = ready.wait_timeout(set, ASYNC_TEST_TIMEOUT).unwrap();
            assert!(!timeout.timed_out(), "{message}");
            set = next;
        }
    }

    #[test]
    fn consumer_cancellation_notifies_promptly_while_a_noncooperative_provider_drains() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let started =
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let release =
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let terminal =
            std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
        let sdk = EdgeLlm {
            provider: Arc::new(IgnoresCancellationProvider {
                started: std::sync::Arc::clone(&started),
                release: std::sync::Arc::clone(&release),
            }),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        let request = sdk
            .ask_async(
                "cancel".into(),
                Box::new(CompletionRecordingHandler {
                    terminal: std::sync::Arc::clone(&terminal),
                }),
            )
            .unwrap();
        wait_for_signal(&started, "provider did not start");

        request.cancel();
        assert_eq!(wait_for_terminal(&terminal), "cancelled");
        assert!(request.is_cancelled());
        assert!(matches!(sdk.reset(), Err(SdkError::Busy { .. })));

        let (released, release_ready) = &*release;
        *released.lock().unwrap() = true;
        release_ready.notify_all();
        wait_until_idle(&sdk);
        sdk.reset()
            .expect("the handle is reusable after the provider drains");
    }

    #[test]
    fn async_stream_returns_immediately_rejects_overlap_and_reports_cancelled_once() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let started =
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let terminal =
            std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
        let terminal_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sdk = EdgeLlm {
            provider: Arc::new(BlockingCancellableProvider {
                started: std::sync::Arc::clone(&started),
            }),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        let sdk = Arc::new(sdk);
        let (submission_sender, submission_receiver) = std::sync::mpsc::sync_channel(1);
        let submitting_sdk = Arc::clone(&sdk);
        let submitting_terminal = std::sync::Arc::clone(&terminal);
        let submitting_terminal_calls = std::sync::Arc::clone(&terminal_calls);
        std::thread::spawn(move || {
            let _ = submission_sender.send(submitting_sdk.ask_stream_async(
                "wait".into(),
                Box::new(TerminalRecordingStreamHandler {
                    terminal: submitting_terminal,
                    terminal_calls: submitting_terminal_calls,
                }),
            ));
        });
        let request = submission_receiver
            .recv_timeout(ASYNC_TEST_TIMEOUT)
            .expect("submission must not wait for provider work")
            .expect("submission must be accepted");

        let (started_lock, started_ready) = &*started;
        let mut is_started = started_lock.lock().unwrap();
        while !*is_started {
            let (next, timeout) = started_ready
                .wait_timeout(is_started, ASYNC_TEST_TIMEOUT)
                .unwrap();
            assert!(!timeout.timed_out(), "async worker did not start");
            is_started = next;
        }
        drop(is_started);

        assert!(matches!(sdk.reset(), Err(SdkError::Busy { .. })));
        request.cancel();

        let (terminal_lock, terminal_done) = &*terminal;
        let mut outcome = terminal_lock.lock().unwrap();
        while outcome.is_none() {
            let (next, timeout) = terminal_done
                .wait_timeout(outcome, ASYNC_TEST_TIMEOUT)
                .unwrap();
            assert!(!timeout.timed_out(), "async cancellation did not finish");
            outcome = next;
        }
        assert_eq!(outcome.as_deref(), Some("cancelled"));
        drop(outcome);
        wait_until_idle(&sdk);
        assert_eq!(
            terminal_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a request emits exactly one terminal callback"
        );

        sdk.reset()
            .expect("the released operation permit must allow a reset");
    }

    type Signal = std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

    struct PauseAfterFirstTokenProvider {
        release: Signal,
    }

    impl LlmProvider for PauseAfterFirstTokenProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            unreachable!("the async stream test uses the cancellable stream seam")
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            unreachable!("the async stream test uses the cancellable stream seam")
        }

        fn chat_stream_cancellable(
            &self,
            _: &ChatRequest,
            cancellation: &CancellationToken,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            on_token(ChatToken {
                text: "first".into(),
                is_final: false,
            });

            let (released, ready) = &*self.release;
            let mut is_released = released.lock().unwrap();
            while !*is_released && !cancellation.is_cancelled() {
                let (next, timeout) = ready.wait_timeout(is_released, ASYNC_TEST_TIMEOUT).unwrap();
                if timeout.timed_out() {
                    return Err(EdgeError::Engine(
                        "host did not receive the first token incrementally",
                    ));
                }
                is_released = next;
            }
            drop(is_released);

            if cancellation.is_cancelled() {
                return Err(EdgeError::Cancelled);
            }
            on_token(ChatToken {
                text: "second".into(),
                is_final: false,
            });
            on_token(ChatToken {
                text: String::new(),
                is_final: true,
            });
            Ok(())
        }
    }

    struct IncrementalRecordingHandler {
        sdk: Arc<EdgeLlm>,
        release: Signal,
        tokens: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        reset_was_busy: std::sync::Arc<std::sync::Mutex<Option<bool>>>,
        terminal: std::sync::Arc<(std::sync::Mutex<Option<String>>, std::sync::Condvar)>,
    }

    impl AsyncStreamHandler for IncrementalRecordingHandler {
        fn on_token(&self, token: String) {
            *self.reset_was_busy.lock().unwrap() =
                Some(matches!(self.sdk.reset(), Err(SdkError::Busy { .. })));
            self.tokens.lock().unwrap().push(token);
            let (released, ready) = &*self.release;
            *released.lock().unwrap() = true;
            ready.notify_all();
        }

        fn on_complete(&self) {
            self.record("complete");
        }

        fn on_error(&self, error: String) {
            self.record(&format!("error:{error}"));
        }

        fn on_cancelled(&self) {
            self.record("cancelled");
        }
    }

    impl IncrementalRecordingHandler {
        fn record(&self, outcome: &str) {
            let (terminal, done) = &*self.terminal;
            *terminal.lock().unwrap() = Some(outcome.into());
            done.notify_all();
        }
    }

    #[test]
    fn async_stream_delivers_incrementally_and_keeps_the_turn_busy_during_callbacks() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let release =
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let tokens = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let reset_was_busy = std::sync::Arc::new(std::sync::Mutex::new(None));
        let terminal =
            std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
        let sdk = Arc::new(EdgeLlm {
            provider: Arc::new(PauseAfterFirstTokenProvider {
                release: std::sync::Arc::clone(&release),
            }),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        });

        sdk.ask_stream_async(
            "stream".into(),
            Box::new(IncrementalRecordingHandler {
                sdk: Arc::clone(&sdk),
                release,
                tokens: std::sync::Arc::clone(&tokens),
                reset_was_busy: std::sync::Arc::clone(&reset_was_busy),
                terminal: std::sync::Arc::clone(&terminal),
            }),
        )
        .unwrap();

        assert_eq!(wait_for_terminal(&terminal), "complete");
        wait_until_idle(&sdk);
        assert_eq!(
            *reset_was_busy.lock().unwrap(),
            Some(true),
            "the turn permit stays held through callback delivery"
        );
        assert_eq!(tokens.lock().unwrap().as_slice(), ["first", "second"]);
    }

    struct ImmediateTokenProvider;

    impl LlmProvider for ImmediateTokenProvider {
        fn chat(&self, _: &ChatRequest) -> el_core::Result<el_core::ChatResponse> {
            unreachable!("the callback-panic test uses the cancellable stream seam")
        }

        fn chat_stream(
            &self,
            _: &ChatRequest,
            _: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            unreachable!("the callback-panic test uses the cancellable stream seam")
        }

        fn chat_stream_cancellable(
            &self,
            _: &ChatRequest,
            _: &CancellationToken,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> el_core::Result<()> {
            on_token(ChatToken {
                text: "token".into(),
                is_final: false,
            });
            Ok(())
        }
    }

    struct PanicThenRecordHandler {
        terminal: std::sync::Arc<(std::sync::Mutex<Option<String>>, std::sync::Condvar)>,
    }

    impl AsyncStreamHandler for PanicThenRecordHandler {
        fn on_token(&self, _: String) {
            panic!("foreign token callback failure");
        }

        fn on_complete(&self) {
            self.record("complete");
        }

        fn on_error(&self, error: String) {
            self.record(&format!("error:{error}"));
        }

        fn on_cancelled(&self) {
            self.record("cancelled");
        }
    }

    impl PanicThenRecordHandler {
        fn record(&self, outcome: &str) {
            let (terminal, done) = &*self.terminal;
            *terminal.lock().unwrap() = Some(outcome.into());
            done.notify_all();
        }
    }

    #[test]
    fn token_callback_panic_still_attempts_one_terminal_error() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let terminal =
            std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
        let sdk = EdgeLlm {
            provider: Arc::new(ImmediateTokenProvider),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        sdk.ask_stream_async(
            "panic".into(),
            Box::new(PanicThenRecordHandler {
                terminal: std::sync::Arc::clone(&terminal),
            }),
        )
        .unwrap();

        assert_eq!(
            wait_for_terminal(&terminal),
            "error:async stream token callback panicked"
        );
        wait_until_idle(&sdk);
    }

    #[test]
    fn async_worker_capacity_is_scoped_per_handle() {
        let _capacity_test_guard = ASYNC_WORKER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let first_handle = Arc::new(AtomicUsize::new(0));
        let second_handle = Arc::new(AtomicUsize::new(0));
        let first = AsyncWorkerPermit::acquire(&first_handle).expect("first worker slot");
        let second = AsyncWorkerPermit::acquire(&first_handle).expect("second worker slot");

        assert!(matches!(
            AsyncWorkerPermit::acquire(&first_handle),
            Err(SdkError::Busy { .. })
        ));

        let independent = AsyncWorkerPermit::acquire(&second_handle)
            .expect("a busy handle must not exhaust another handle's capacity");

        drop(independent);
        drop(second);
        drop(first);
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
            provider: Arc::new(ResetTrackingProvider {
                reset_calls: std::sync::Arc::clone(&reset_calls),
                should_fail: false,
            }),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        sdk.reset().expect("provider reset must succeed");

        assert_eq!(reset_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn reset_propagates_provider_failure() {
        let sdk = EdgeLlm {
            provider: Arc::new(ResetTrackingProvider {
                reset_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                should_fail: true,
            }),
            default_model: "test".into(),
            max_tokens: None,
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
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
            provider: Arc::new(RequestCapProvider(std::sync::Arc::clone(&requests))),
            default_model: "local/qwen".into(),
            max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS),
            operation_active: Arc::new(AtomicBool::new(false)),
            async_requests_active: Arc::new(AtomicUsize::new(0)),
        };

        sdk.ask("Reply with exactly: ready".into()).unwrap();
        sdk.ask_stream_with("Reply with exactly: ready".into(), |_| {})
            .unwrap();

        assert_eq!(
            *requests.lock().unwrap(),
            vec![Some(QWEN_FFI_DEFAULT_MAX_TOKENS); 2]
        );
        assert_eq!(
            sdk.async_request("Reply with exactly: ready".into())
                .max_tokens,
            None,
            "the compatibility cap must not silently constrain native-worker calls"
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
