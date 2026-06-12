//! OpenAI Chat Completions wire types (serde).

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct WireRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub temperature: f32,
    pub stream: bool,
}

#[derive(Serialize)]
pub struct WireMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[derive(Deserialize)]
pub struct WireResponse {
    pub choices: Vec<WireChoice>,
    pub usage: Option<WireUsage>,
    pub model: String,
}

#[derive(Deserialize)]
pub struct WireChoice {
    pub message: WireChoiceMessage,
}

#[derive(Deserialize)]
pub struct WireChoiceMessage {
    pub content: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct WireUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// SSE stream delta for streaming completions.
#[derive(Deserialize)]
pub struct WireStreamChunk {
    pub choices: Vec<WireStreamChoice>,
}

/// Provider mid-stream error payload: `data: {"error":{"message":…}}`.
/// OpenAI-compatible endpoints send this instead of a chunk when the request
/// fails after streaming has begun (quota exhausted, server error, …).
#[derive(Deserialize)]
pub struct WireStreamError {
    pub error: WireErrorDetail,
}

#[derive(Deserialize)]
pub struct WireErrorDetail {
    pub message: Option<String>,
}

#[derive(Deserialize)]
pub struct WireStreamChoice {
    pub delta: WireStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct WireStreamDelta {
    pub content: Option<String>,
}
