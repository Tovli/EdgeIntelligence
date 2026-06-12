//! `el-core` — the shared, dependency-free domain vocabulary for the
//! Edge-Native LLM SDK.
//!
//! Everything here is the *ubiquitous language* made into Rust types
//! (`docs/ddd/ubiquitous-language.md`). It has **no external dependencies** so
//! it compiles offline on any target, including `wasm32` (ADR-008).
//!
//! Cross-cutting invariants encoded here:
//! - **Content-free events (ADR-007):** [`events::DomainEvent`] derives `Copy`,
//!   which makes the compiler reject any `String`/`Vec`/heap field — so no
//!   prompt or response content can ever ride on an event.
//! - **Air-gap by default (ADR-004):** [`config::SessionConfig::hybrid_mode`]
//!   defaults to `false`; the only network seam is an explicit opt-in.
//! - **Unified LLM provider (ADR-010):** [`provider::LlmProvider`] covers both
//!   local Candle and cloud frontier backends behind one trait.

#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod events;
pub mod ids;
pub mod provider;
pub mod value_objects;

pub use config::SessionConfig;
pub use error::{EdgeError, Result};
pub use events::{DegradeReason, DomainEvent, EventEnvelope};
pub use ids::{ModelId, ModelVersion, SessionId};
pub use provider::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ChatToken, CredentialRef, LlmProvider,
};
pub use value_objects::{
    DeviceTarget, ModelFormat, Phase, RuntimeKind, SafetyMode, SpeculationMode, StopReason, Token,
};
