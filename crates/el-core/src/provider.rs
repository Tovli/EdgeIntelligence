//! Unified LLM provider abstraction (ADR-010).
//!
//! One trait covers both local Candle inference and cloud frontier LLMs. The
//! host picks a backend at session construction time — the rest of the SDK
//! sees only `LlmProvider`. This is the seam that lets mobile apps swap
//! local ↔ frontier without touching their UI code.
//!
//! Design notes:
//! - All types are plain `std` (no async runtime dep in this crate).
//! - `chat_stream` uses a callback so each binding surface wraps it in its
//!   own async/stream primitive (FRB → Dart `Stream`, uniffi → async callback,
//!   wasm-bindgen → `ReadableStream`).
//! - `CredentialRef` is a runtime value from the host — never embedded.

/// Which role a message belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// One turn in the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Runtime API credential. The host resolves this from platform keystore
/// (Android Keystore / iOS Keychain) before calling `start_session`. The SDK
/// never logs or persists the value.
///
/// # Security
/// `Debug` output is redacted so that `{:?}` in logs and assertion failures
/// cannot expose bearer keys. If you need to verify a credential is present,
/// use `CredentialRef::is_empty()`.
#[derive(Clone, PartialEq, Eq)]
pub struct CredentialRef(String);

impl CredentialRef {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for CredentialRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CredentialRef([REDACTED])")
    }
}

/// A chat completion request. `model` is a routing hint:
/// - `"local"` or `""` → local Candle engine
/// - `"openai/<model>"` → OpenAI Chat Completions
/// - `"anthropic/<model>"` → Anthropic Messages
/// - `"ollama/<model>"` → local Ollama (OpenAI-compat)
/// - `"gemini/<model>"` → Google Generative AI
///
/// The `credential` field's `Debug` output is redacted; the rest of the struct
/// derives a normal `Debug` impl.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    /// Temperature × 1000 (integer to keep the type `Eq`-able; 1000 = 1.0).
    pub temperature_milli: u32,
    pub credential: Option<CredentialRef>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            max_tokens: None,
            temperature_milli: 700,
            credential: None,
        }
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_temperature(mut self, t_milli: u32) -> Self {
        self.temperature_milli = t_milli;
        self
    }

    pub fn with_credential(mut self, cred: CredentialRef) -> Self {
        self.credential = Some(cred);
        self
    }
}

/// A single streamed token fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToken {
    pub text: String,
    pub is_final: bool,
}

/// A completed chat response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    pub content: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// The unified backend trait (ADR-010). Implemented by:
/// - `LocalLlmProvider` in `el-runtime` (wraps `InferenceSession` + Candle)
/// - `CloudProvider` in `el-cloud` (wraps `reqwest` + OpenAI-compat API)
pub trait LlmProvider: Send + Sync {
    /// Blocking, non-streaming chat completion.
    fn chat(&self, req: &ChatRequest) -> crate::Result<ChatResponse>;

    /// Streaming chat: calls `on_token` for each fragment as it arrives.
    /// Returns when generation is complete or on error. The final call will
    /// have `ChatToken::is_final == true`.
    fn chat_stream(
        &self,
        req: &ChatRequest,
        on_token: &mut dyn FnMut(ChatToken),
    ) -> crate::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;

    struct EchoProvider;
    impl LlmProvider for EchoProvider {
        fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
            let echo = req
                .messages
                .last()
                .map(|m| m.content.as_str())
                .unwrap_or("")
                .to_owned();
            Ok(ChatResponse {
                content: echo.clone(),
                model: req.model.clone(),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }
        fn chat_stream(
            &self,
            req: &ChatRequest,
            on_token: &mut dyn FnMut(ChatToken),
        ) -> Result<()> {
            let text = req
                .messages
                .last()
                .map(|m| m.content.as_str())
                .unwrap_or("");
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

    #[test]
    fn chat_request_builder() {
        let req = ChatRequest::new("local", vec![ChatMessage::user("hello")])
            .with_max_tokens(256)
            .with_temperature(500);
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.temperature_milli, 500);
    }

    #[test]
    fn echo_provider_round_trips() {
        let p = EchoProvider;
        let req = ChatRequest::new("test", vec![ChatMessage::user("ping")]);
        let resp = p.chat(&req).unwrap();
        assert_eq!(resp.content, "ping");
    }

    #[test]
    fn stream_delivers_all_chars_then_final() {
        let p = EchoProvider;
        let req = ChatRequest::new("test", vec![ChatMessage::user("hi")]);
        let mut tokens: Vec<ChatToken> = Vec::new();
        p.chat_stream(&req, &mut |t| tokens.push(t)).unwrap();
        assert!(tokens.last().unwrap().is_final);
        let text: String = tokens
            .iter()
            .filter(|t| !t.is_final)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(text, "hi");
    }
}
