//! `el-ffi` — host bindings for the Rust core (ADR-001).
//!
//! SKELETON / FOLLOW-UP. One Rust surface exported two ways:
//! - **Native (Android/iOS):** UniFFI generates Kotlin & Swift bindings —
//!   `start_session()`, `ask(prompt) -> String`, `on_token` callback — with no
//!   hand-written JNI/Objective-C.
//! - **Web (`wasm32`):** `wasm-bindgen` exposes the same surface to JS/TS; the
//!   module is run under Wasmtime (ADR-001) or in the browser.
//!
//! Cross-compilation to `wasm32`/`aarch64-mobile` and binding generation can't
//! be exercised in this environment, so this stays a documented seam.

#![forbid(unsafe_code)]

/// The flat, FFI-friendly facade over `el_runtime::InferenceSession`.
///
/// TODO(adr-001):
/// - `#[derive(uniffi::Object)]` + `#[uniffi::export]` on the methods below.
/// - `#[cfg(target_arch = "wasm32")] #[wasm_bindgen]` mirror for the web.
/// - Marshal a real engine (el-engine-candle) and a callback channel for tokens.
pub struct EdgeLlm {
    _private: (),
}

impl EdgeLlm {
    /// `init(model_uri, options)` → load + verify (ADR-006) + allocate arena.
    pub fn start_session(_model_uri: &str) -> Self {
        Self { _private: () }
    }

    /// `ask` → load_prompt + generate, returning decoded text.
    pub fn ask(&self, _prompt: &str) -> String {
        // TODO(adr-001): drive el_runtime::InferenceSession and detokenize.
        String::new()
    }
}
