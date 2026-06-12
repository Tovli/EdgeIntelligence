//! `el-cloud` — opt-in frontier LLM cloud backend (ADR-010).
//!
//! Implements [`el_core::LlmProvider`] over `reqwest` using the OpenAI Chat
//! Completions API, which is supported natively by OpenAI, Anthropic (via
//! compatibility layer), Ollama, and any OpenAI-compatible endpoint.
//!
//! Provider routing is by model prefix:
//! - `"openai/<model>"` → `https://api.openai.com/v1`
//! - `"anthropic/<model>"` → `https://api.anthropic.com/v1` (compat)
//! - `"gemini/<model>"` → `https://generativelanguage.googleapis.com/v1beta/openai`
//! - `"ollama/<model>"` → `http://localhost:11434/v1` (no key required)
//! - `"http(s)://…/<model>"` → custom base URL
//!
//! The air-gap guarantee (ADR-004) is preserved: this crate is excluded from
//! the default workspace build. Only apps that explicitly construct a
//! [`CloudProvider`] opt into outbound network calls.
//!
//! # wasm32 note
//! `reqwest::blocking` is unavailable on `wasm32-unknown-unknown` (no threads).
//! The async reqwest surface exists there, but `LlmProvider::chat()` is
//! synchronous and cannot await the browser's `fetch`. The el-ffi wasm32
//! `cloud` constructor therefore throws an explicit "unavailable" error
//! (ADR-010 amendment), and this crate's network modules are gated to
//! non-wasm32.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod openai_wire;
#[cfg(not(target_arch = "wasm32"))]
mod routing;

#[cfg(not(target_arch = "wasm32"))]
pub use routing::CloudProvider;
