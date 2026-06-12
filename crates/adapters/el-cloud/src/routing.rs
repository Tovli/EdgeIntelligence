//! Provider routing and `LlmProvider` implementation for cloud backends.

use crate::openai_wire::{
    WireMessage, WireRequest, WireResponse, WireStreamChunk, WireStreamError,
};
use el_core::{
    ChatRequest, ChatResponse, ChatRole, ChatToken, DomainEvent, EdgeError, LlmProvider, Result,
};

/// Resolves provider base URL and strips the prefix from the model name.
fn resolve(model: &str) -> (&str, String) {
    if let Some(m) = model.strip_prefix("openai/") {
        return ("https://api.openai.com/v1", m.to_owned());
    }
    if let Some(m) = model.strip_prefix("anthropic/") {
        return ("https://api.anthropic.com/v1", m.to_owned());
    }
    if let Some(m) = model.strip_prefix("gemini/") {
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            m.to_owned(),
        );
    }
    if let Some(m) = model.strip_prefix("ollama/") {
        return ("http://localhost:11434/v1", m.to_owned());
    }
    // Custom base URL: "https://my.server/v1/llama3" → split on last "/"
    if model.starts_with("http://") || model.starts_with("https://") {
        if let Some(pos) = model.rfind('/') {
            return (&model[..pos], model[pos + 1..].to_owned());
        }
    }
    // Fallback: treat as an OpenAI model name
    ("https://api.openai.com/v1", model.to_owned())
}

fn wire_role(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

fn make_err(msg: impl Into<String>) -> EdgeError {
    EdgeError::CloudRequest(msg.into().into_boxed_str())
}

/// Cap on the TCP + TLS connect phase.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Blocking-client operation timeout. It bounds the wait for response headers,
/// and — because `reqwest::blocking` applies it to **every body `read()`** —
/// acts as an *idle* timeout between SSE chunks while streaming: a long
/// generation streams indefinitely as long as chunks keep arriving, but a
/// stalled provider unblocks the caller within this window.
const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// Total wall-clock cap for non-streaming `chat()` requests (headers + body).
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Frontier LLM cloud backend. Construct one per session; it owns a
/// `reqwest::blocking::Client` that reuses the TCP connection pool.
///
/// # Air-gap guarantee (ADR-004)
/// A [`CloudProvider`] is only ever created if the host app explicitly opts
/// in. Apps that never construct this type have zero outbound network surface.
///
/// # Event emission (ADR-010)
/// After each successful `chat()` or `chat_stream()` call, emits
/// [`DomainEvent::FrontierLlmConsulted`] via the optional sink registered
/// with [`CloudProvider::with_event_sink`].
pub struct CloudProvider {
    client: reqwest::blocking::Client,
    event_sink: Option<Box<dyn Fn(DomainEvent) + Send + Sync>>,
}

impl CloudProvider {
    /// Builds the provider with explicit network timeouts: [`CONNECT_TIMEOUT`]
    /// for the handshake, [`IDLE_TIMEOUT`] per read (stream-friendly), and a
    /// [`REQUEST_TIMEOUT`] total applied per non-streaming request. A stalled
    /// provider can therefore never block an FFI caller indefinitely.
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(IDLE_TIMEOUT)
            .build()
            .expect("static client configuration is valid");
        Self {
            client,
            event_sink: None,
        }
    }

    /// Register a callback that receives [`DomainEvent`]s (e.g. to feed a
    /// [`el_telemetry::MetricsCollector`]).
    pub fn with_event_sink(mut self, sink: impl Fn(DomainEvent) + Send + Sync + 'static) -> Self {
        self.event_sink = Some(Box::new(sink));
        self
    }

    fn emit(&self, event: DomainEvent) {
        if let Some(sink) = &self.event_sink {
            sink(event);
        }
    }

    /// Consumes an SSE response body line-by-line, forwarding text deltas to
    /// `on_token`. Factored out of [`LlmProvider::chat_stream`] so the
    /// protocol handling is testable without a live HTTP connection.
    ///
    /// # Error contract
    /// Every `data:` payload that is neither `[DONE]` nor a well-formed
    /// [`WireStreamChunk`] fails the call with [`EdgeError::CloudRequest`] —
    /// provider error objects (`{"error":{…}}`) and protocol corruption must
    /// never be mistaken for a clean (possibly empty) completion. Malformed
    /// payloads are reported by parse category/position/size only — never
    /// echoed — per the el-core rule that errors carry no content. Non-`data`
    /// SSE lines (comments/keepalives starting with `:`, `event:`/`id:`/
    /// `retry:` fields, blank separators) are skipped per the SSE spec.
    fn pump_sse(
        &self,
        reader: impl std::io::BufRead,
        model: &str,
        on_token: &mut dyn FnMut(ChatToken),
    ) -> Result<()> {
        for line in reader.lines() {
            let line = line.map_err(|e| make_err(format!("cloud stream read: {e}")))?;
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let data = line["data:".len()..].trim();
            if data == "[DONE]" {
                self.emit(DomainEvent::FrontierLlmConsulted {
                    provider_hash: provider_hash(model),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                });
                on_token(ChatToken {
                    text: String::new(),
                    is_final: true,
                });
                return Ok(());
            }
            let chunk = match serde_json::from_str::<WireStreamChunk>(data) {
                Ok(chunk) => chunk,
                Err(e) => {
                    if let Ok(err) = serde_json::from_str::<WireStreamError>(data) {
                        return Err(make_err(format!(
                            "cloud stream provider error: {}",
                            err.error.message.as_deref().unwrap_or("unknown")
                        )));
                    }
                    // EdgeError must never carry payload content (el-core
                    // error contract) — a malformed chunk may embed generated
                    // text. Report only parse category, position, and size.
                    let category = match e.classify() {
                        serde_json::error::Category::Io => "io",
                        serde_json::error::Category::Syntax => "syntax",
                        serde_json::error::Category::Data => "data",
                        serde_json::error::Category::Eof => "eof",
                    };
                    return Err(make_err(format!(
                        "cloud stream decode: {category} error at line {} column {} \
                         in {}-byte payload (content withheld)",
                        e.line(),
                        e.column(),
                        data.len()
                    )));
                }
            };
            for choice in chunk.choices {
                if let Some(text) = choice.delta.content {
                    if !text.is_empty() {
                        on_token(ChatToken {
                            text,
                            is_final: false,
                        });
                    }
                }
                if choice.finish_reason.is_some() {
                    self.emit(DomainEvent::FrontierLlmConsulted {
                        provider_hash: provider_hash(model),
                        prompt_tokens: 0,
                        completion_tokens: 0,
                    });
                    on_token(ChatToken {
                        text: String::new(),
                        is_final: true,
                    });
                    return Ok(());
                }
            }
        }
        // Stream ended without [DONE] — emit final anyway.
        on_token(ChatToken {
            text: String::new(),
            is_final: true,
        });
        // provider_hash is available but token counts aren't in streaming path;
        // emit with zeros so the audit trail still fires.
        self.emit(DomainEvent::FrontierLlmConsulted {
            provider_hash: provider_hash(model),
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        Ok(())
    }
}

impl Default for CloudProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for CloudProvider {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let (base_url, model_name) = resolve(&req.model);
        let api_key = req
            .credential
            .as_ref()
            .map(|c| c.as_str().to_owned())
            .unwrap_or_default();

        let messages: Vec<WireMessage> = req
            .messages
            .iter()
            .map(|m| WireMessage {
                role: wire_role(m.role),
                content: &m.content,
            })
            .collect();

        let body = WireRequest {
            model: &model_name,
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature_milli as f32 / 1000.0,
            stream: false,
        };

        let mut builder = self
            .client
            .post(format!("{base_url}/chat/completions"))
            // Non-streaming: the body arrives in one read, so a total
            // request deadline is appropriate (overrides the idle default).
            .timeout(REQUEST_TIMEOUT)
            .json(&body);

        if !api_key.is_empty() {
            builder = builder.bearer_auth(&api_key);
        }

        let resp = builder
            .send()
            .map_err(|e| make_err(format!("cloud send: {e}")))?
            .error_for_status()
            .map_err(|e| make_err(format!("cloud status: {e}")))?
            .json::<WireResponse>()
            .map_err(|e| make_err(format!("cloud decode: {e}")))?;

        let content = resp
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let usage = resp.usage.unwrap_or_default();

        self.emit(DomainEvent::FrontierLlmConsulted {
            provider_hash: provider_hash(&req.model),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        });

        Ok(ChatResponse {
            content,
            model: resp.model,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        })
    }

    fn chat_stream(&self, req: &ChatRequest, on_token: &mut dyn FnMut(ChatToken)) -> Result<()> {
        let (base_url, model_name) = resolve(&req.model);
        let api_key = req
            .credential
            .as_ref()
            .map(|c| c.as_str().to_owned())
            .unwrap_or_default();

        let messages: Vec<WireMessage> = req
            .messages
            .iter()
            .map(|m| WireMessage {
                role: wire_role(m.role),
                content: &m.content,
            })
            .collect();

        let body = WireRequest {
            model: &model_name,
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature_milli as f32 / 1000.0,
            stream: true,
        };

        let mut builder = self
            .client
            .post(format!("{base_url}/chat/completions"))
            .json(&body);

        if !api_key.is_empty() {
            builder = builder.bearer_auth(&api_key);
        }

        let resp = builder
            .send()
            .map_err(|e| make_err(format!("cloud stream send: {e}")))?
            .error_for_status()
            .map_err(|e| make_err(format!("cloud stream status: {e}")))?;

        // SSE body: "data: {json}", "data: [DONE]", ":keepalive", blank separators.
        self.pump_sse(std::io::BufReader::new(resp), &req.model, on_token)
    }
}

/// Simple CRC32-style hash for provider name (for the `FrontierLlmConsulted`
/// domain event — content-free audit trail).
pub fn provider_hash(model: &str) -> u32 {
    let (base, _) = resolve(model);
    base.bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_resolves_known_prefixes() {
        assert_eq!(resolve("openai/gpt-4o").0, "https://api.openai.com/v1");
        assert_eq!(resolve("openai/gpt-4o").1, "gpt-4o");
        assert_eq!(resolve("ollama/llama3").0, "http://localhost:11434/v1");
        assert_eq!(resolve("ollama/llama3").1, "llama3");
        assert_eq!(
            resolve("anthropic/claude-sonnet-4-6").1,
            "claude-sonnet-4-6"
        );
        assert_eq!(resolve("gemini/gemini-2.0-flash").1, "gemini-2.0-flash");
    }

    #[test]
    fn routing_custom_url() {
        let (base, model) = resolve("https://my.llm.server/v1/mistral-7b");
        assert_eq!(base, "https://my.llm.server/v1");
        assert_eq!(model, "mistral-7b");
    }

    #[test]
    fn provider_hash_is_deterministic() {
        let h1 = provider_hash("openai/gpt-4o");
        let h2 = provider_hash("openai/gpt-4o-mini");
        assert_eq!(h1, h2, "same provider, different model → same hash");
        let h3 = provider_hash("anthropic/claude-3-5-sonnet");
        assert_ne!(h1, h3, "different providers → different hashes");
    }

    #[test]
    fn cloud_provider_constructs() {
        let _ = CloudProvider::new();
    }

    #[test]
    fn event_sink_receives_frontier_consulted_on_successful_chat() {
        use el_core::DomainEvent;
        use std::sync::{Arc, Mutex};

        let events: Arc<Mutex<Vec<DomainEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        // Build a provider with an event sink but don't make a real HTTP call —
        // call emit() directly to verify the sink wiring is correct.
        let provider = CloudProvider::new().with_event_sink(move |ev| {
            events_clone.lock().unwrap().push(ev);
        });

        // Simulate the event that chat() would emit after a real HTTP response.
        provider.emit(DomainEvent::FrontierLlmConsulted {
            provider_hash: provider_hash("openai/gpt-4o"),
            prompt_tokens: 10,
            completion_tokens: 5,
        });

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!(matches!(
            captured[0],
            DomainEvent::FrontierLlmConsulted {
                prompt_tokens: 10,
                completion_tokens: 5,
                ..
            }
        ));
    }

    #[test]
    fn provider_hash_same_for_same_provider_different_models() {
        // Proves the hash is content-free: same base URL regardless of model suffix.
        assert_eq!(
            provider_hash("openai/gpt-4o"),
            provider_hash("openai/gpt-4o-mini"),
        );
    }

    // ── SSE protocol handling (pump_sse) ──────────────────────────────────

    fn run_sse(body: &str) -> (Result<()>, Vec<ChatToken>) {
        let provider = CloudProvider::new();
        let mut tokens = Vec::new();
        let result = provider.pump_sse(
            std::io::Cursor::new(body.to_owned()),
            "openai/gpt-4o",
            &mut |t| tokens.push(t),
        );
        (result, tokens)
    }

    #[test]
    fn stream_chunks_until_done_yield_tokens_and_final() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"He\"},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":null}]}\n\n\
                    data: [DONE]\n\n";
        let (result, tokens) = run_sse(body);
        result.expect("well-formed stream must succeed");
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text, "Hello");
        assert!(
            tokens.last().unwrap().is_final,
            "stream must end with a final token"
        );
    }

    #[test]
    fn stream_provider_error_payload_fails_the_call() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"He\"},\"finish_reason\":null}]}\n\n\
                    data: {\"error\":{\"message\":\"insufficient quota\",\"type\":\"insufficient_quota\"}}\n\n";
        let (result, tokens) = run_sse(body);
        let err = result.expect_err("provider error payload must fail the stream");
        match err {
            EdgeError::CloudRequest(msg) => {
                assert!(
                    msg.contains("insufficient quota"),
                    "error must carry the provider message, got: {msg}"
                );
            }
            other => panic!("expected CloudRequest, got {other:?}"),
        }
        assert!(
            tokens.iter().all(|t| !t.is_final),
            "a failed stream must not signal a clean completion"
        );
    }

    #[test]
    fn stream_malformed_payload_fails_the_call_without_echoing_content() {
        let body = "data: {this is not json\n\n";
        let (result, tokens) = run_sse(body);
        let err = result.expect_err("malformed payload must fail the stream");
        match err {
            EdgeError::CloudRequest(msg) => {
                assert!(
                    msg.contains("decode"),
                    "error must identify a decode failure, got: {msg}"
                );
                assert!(
                    !msg.contains("this is not json"),
                    "raw payload must never leak into EdgeError (el-core contract), got: {msg}"
                );
            }
            other => panic!("expected CloudRequest, got {other:?}"),
        }
        assert!(tokens.iter().all(|t| !t.is_final));
    }

    #[test]
    fn stream_keepalives_comments_and_events_are_skipped() {
        let body = ": keep-alive\n\
                    event: ping\n\
                    id: 42\n\
                    \n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n\
                    data: [DONE]\n\n";
        let (result, tokens) = run_sse(body);
        result.expect("SSE comments/keepalives/fields must not fail the stream");
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text, "ok");
    }

    #[test]
    fn stream_eof_without_done_still_finalizes() {
        // Some OpenAI-compat servers close the connection after finish_reason
        // without sending [DONE]; that is a complete (non-error) stream.
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let (result, tokens) = run_sse(body);
        result.expect("EOF without [DONE] is not a protocol error");
        assert!(tokens.last().unwrap().is_final);
    }

    #[test]
    fn stream_finish_reason_completes_without_done() {
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n";
        let (result, tokens) = run_sse(body);
        result.expect("finish_reason terminates the stream cleanly");
        assert!(tokens.last().unwrap().is_final);
    }
}
