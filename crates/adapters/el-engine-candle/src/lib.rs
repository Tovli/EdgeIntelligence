//! `el-engine-candle` — the inference engine adapter over **Candle** (ADR-002),
//! implementing [`el_runtime::InferenceEngine`] / `RuntimeAcl`.
//!
//! This runs a **real Candle CPU forward** (embedding lookup → linear projection
//! → logits) over a tiny model built *in code*, so the adapter is exercised
//! end-to-end with genuine Candle tensors without a downloaded GGUF. Float logits
//! are quantised to integer milli-logits **at this ACL boundary**, so Candle's
//! `Tensor`/`Device` types never reach the domain.
//!
//! FOLLOW-UP ([`CandleEngine::load`]): production GGUF/safetensors loading +
//! transformer forward + KV-cache wiring against the `el-memory` arena.

#![forbid(unsafe_code)]

use candle_core::{Device, Tensor};
use el_core::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ChatToken, EdgeError, LlmProvider, Result,
    SessionConfig, SessionId, Token,
};
use el_provenance::LoadPermit;
use el_runtime::{InferenceEngine, InferenceSession, Ports};

/// Candle-backed engine. The toy model is `embed: [vocab, dim]` and
/// `w_out: [dim, vocab]`; next-token logits = `embed[last] · w_out`.
pub struct CandleEngine {
    embed: Tensor,
    w_out: Tensor,
    vocab: usize,
    eos: Token,
}

impl CandleEngine {
    /// Build a deterministic toy model on the CPU (Candle NEON/SIMD on ARM,
    /// scalar/SIMD on x86). Real Candle tensors and matmul — just trivial,
    /// fixed weights so tests are deterministic.
    pub fn toy(vocab: usize, dim: usize, eos: Token) -> Result<Self> {
        let device = Device::Cpu;

        let embed_data: Vec<f32> = (0..vocab * dim)
            .map(|k| {
                let (i, j) = (k / dim, k % dim);
                (((i + j) % 7) as f32) * 0.1
            })
            .collect();
        let wout_data: Vec<f32> = (0..dim * vocab)
            .map(|k| {
                let (a, b) = (k / vocab, k % vocab);
                ((((a * 31 + b * 17) % 13) as f32) * 0.1) - 0.6
            })
            .collect();

        let embed = Tensor::from_vec(embed_data, (vocab, dim), &device)
            .map_err(|_| EdgeError::Engine("candle: embed tensor build failed"))?;
        let w_out = Tensor::from_vec(wout_data, (dim, vocab), &device)
            .map_err(|_| EdgeError::Engine("candle: w_out tensor build failed"))?;

        Ok(Self {
            embed,
            w_out,
            vocab,
            eos,
        })
    }

    /// Production model loading — the ADR-002 follow-up.
    ///
    /// TODO(adr-002): pick `Device::new_metal(0)` on Apple else `Device::Cpu`;
    /// read GGUF (`candle_core::quantized::gguf_file`) or safetensors; build the
    /// transformer; map constant weights via the el-memory `WeightMapper`.
    pub fn load(_model_path: &str) -> Result<Self> {
        Err(EdgeError::Engine(
            "CandleEngine::load (GGUF/safetensors) not yet implemented — ADR-002 follow-up; use ::toy",
        ))
    }

    /// One real Candle forward: `embed[last] · w_out` → length-`vocab` logits.
    fn forward(&self, last: usize) -> candle_core::Result<Vec<f32>> {
        let row = self.embed.narrow(0, last, 1)?; // [1, dim]
        let logits = row.matmul(&self.w_out)?; // [1, vocab]
        Ok(logits.to_vec2::<f32>()?.remove(0))
    }
}

impl InferenceEngine for CandleEngine {
    fn prefill(&mut self, tokens: &[Token]) -> Result<u32> {
        // TODO(adr-002): chunked prefill filling the KV cache; the toy model is
        // stateless, so prefill just reports the context length.
        Ok(tokens.len() as u32)
    }

    fn next_logits(&mut self, committed: &[Token]) -> Vec<i32> {
        let last = committed
            .last()
            .copied()
            .unwrap_or(0)
            .min(self.vocab as u32 - 1) as usize;
        match self.forward(last) {
            // Quantise float logits → integer milli-logits at the ACL boundary.
            Ok(logits) => logits.iter().map(|x| (x * 1000.0).round() as i32).collect(),
            Err(_) => vec![0; self.vocab],
        }
    }

    fn eos_token(&self) -> Token {
        self.eos
    }
}

// ── LlmProvider (text-level) wrapper (ADR-010) ───────────────────────────────

/// Wraps an `InferenceSession<CandleEngine>` behind the `LlmProvider` trait,
/// bridging the text API down to token IDs via a byte-level tokenizer.
///
/// The byte tokenizer maps each UTF-8 byte → `byte % vocab` so it works with
/// any vocab size. This is intentionally minimal — a production build would
/// swap in a HuggingFace `tokenizers::Tokenizer` loaded from the model file.
pub struct LocalLlmProvider {
    session: std::sync::Mutex<InferenceSession<CandleEngine>>,
    vocab: usize,
}

impl LocalLlmProvider {
    /// Build a toy provider for testing — same fixed weights as [`CandleEngine::toy`].
    pub fn toy(vocab: usize, dim: usize, eos: Token, permit: LoadPermit) -> Result<Self> {
        let engine = CandleEngine::toy(vocab, dim, eos)?;
        let session = InferenceSession::new(SessionId(1), SessionConfig::default(), engine, permit);
        Ok(Self {
            session: std::sync::Mutex::new(session),
            vocab,
        })
    }

    /// Encode text → token IDs using a byte-level mapping (byte % vocab).
    fn encode(&self, text: &str) -> Vec<Token> {
        text.bytes()
            .map(|b| (b as Token) % self.vocab as Token)
            .collect()
    }

    /// Decode token IDs → text: printable ASCII bytes pass through; others → '?'.
    fn decode(tokens: &[Token]) -> String {
        tokens
            .iter()
            .map(|&t| {
                let b = (t & 0xFF) as u8;
                if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else {
                    '?'
                }
            })
            .collect()
    }

    /// Format messages into a single prompt string (role: content\n).
    fn format_messages(messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                };
                format!("{role}: {}", m.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl LlmProvider for LocalLlmProvider {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let prompt = Self::format_messages(&req.messages);
        let prompt_tokens = self.encode(&prompt);
        let prompt_len = prompt_tokens.len() as u32;
        let max = req.max_tokens.unwrap_or(64);

        let mut session = self.session.lock().unwrap();
        session.reset();
        let ports = Ports::permissive();
        session.load_prompt(&ports, &prompt_tokens)?;
        session.generate(&ports, max)?;

        let output = session.output().to_vec();
        let completion_len = output.len() as u32;

        Ok(ChatResponse {
            content: Self::decode(&output),
            model: "local/candle".into(),
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
        })
    }

    fn chat_stream(&self, req: &ChatRequest, on_token: &mut dyn FnMut(ChatToken)) -> Result<()> {
        let resp = self.chat(req)?;
        // The Candle session generates all tokens in one pass; simulate streaming
        // by emitting one character at a time so the callback contract holds.
        for ch in resp.content.chars() {
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
    use el_runtime::InferenceEngine;

    #[test]
    fn real_candle_forward_is_deterministic_and_right_shape() {
        let mut eng = CandleEngine::toy(8, 4, 7).unwrap();
        let a = eng.next_logits(&[2]);
        let b = eng.next_logits(&[2]);
        assert_eq!(a.len(), 8, "logits length == vocab");
        assert_eq!(a, b, "fixed weights → deterministic real-tensor forward");
        // Different context token generally yields different logits.
        let c = eng.next_logits(&[5]);
        assert_ne!(a, c);
    }

    #[test]
    fn drives_the_runtime_end_to_end() {
        use el_core::{ModelFormat, ModelId, ModelVersion, SessionConfig, SessionId, StopReason};
        use el_provenance::{ModelArtifact, SignatureVerifier};
        use el_runtime::{InferenceSession, Ports};

        struct OkVerifier;
        impl SignatureVerifier for OkVerifier {
            fn verify(&self, _: &[u8], _: &[u8], _: u32) -> bool {
                true
            }
        }
        let mut art = ModelArtifact::new(
            ModelId(1),
            ModelVersion::new(0, 1, 0),
            ModelFormat::Safetensors,
        );
        art.verify(&OkVerifier, b"w", b"s", 1);
        let permit = art.ensure_loadable().unwrap();

        // eos out of range → runs to max_tokens, producing real Candle tokens.
        let eng = CandleEngine::toy(16, 8, 9999).unwrap();
        let mut session =
            InferenceSession::new(SessionId(1), SessionConfig::default(), eng, permit);
        let ports = Ports::permissive();
        session.load_prompt(&ports, &[1, 2, 3]).unwrap();

        let stop = session.generate(&ports, 4).unwrap();
        assert_eq!(stop, StopReason::MaxTokens);
        assert_eq!(
            session.output().len(),
            4,
            "4 tokens decoded via a real Candle forward"
        );
    }

    fn ok_permit() -> LoadPermit {
        use el_core::{ModelFormat, ModelId, ModelVersion};
        use el_provenance::{ModelArtifact, SignatureVerifier};
        struct OkV;
        impl SignatureVerifier for OkV {
            fn verify(&self, _: &[u8], _: &[u8], _: u32) -> bool {
                true
            }
        }
        let mut a = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
        a.verify(&OkV, b"w", b"s", 0);
        a.ensure_loadable().unwrap()
    }

    #[test]
    fn local_provider_chat_returns_response() {
        let p = LocalLlmProvider::toy(32, 8, 31, ok_permit()).unwrap();
        let req = el_core::ChatRequest::new("local", vec![el_core::ChatMessage::user("hello")])
            .with_max_tokens(4);
        let resp = p.chat(&req).unwrap();
        assert_eq!(resp.model, "local/candle");
        assert_eq!(resp.completion_tokens, 4);
        assert!(!resp.content.is_empty());
    }

    #[test]
    fn local_provider_stream_ends_with_final_token() {
        let p = LocalLlmProvider::toy(32, 8, 31, ok_permit()).unwrap();
        let req = el_core::ChatRequest::new("local", vec![el_core::ChatMessage::user("hi")])
            .with_max_tokens(3);
        let mut tokens: Vec<el_core::ChatToken> = Vec::new();
        p.chat_stream(&req, &mut |t| tokens.push(t)).unwrap();
        assert!(tokens.last().unwrap().is_final, "last token must be final");
        assert!(tokens.len() > 1, "at least one content token before final");
    }

    #[test]
    fn local_provider_session_resets_between_calls() {
        let p = LocalLlmProvider::toy(32, 8, 31, ok_permit()).unwrap();
        let req = el_core::ChatRequest::new("local", vec![el_core::ChatMessage::user("a")])
            .with_max_tokens(4);
        let r1 = p.chat(&req).unwrap();
        let r2 = p.chat(&req).unwrap();
        assert_eq!(
            r1.content, r2.content,
            "same input → same output (deterministic weights)"
        );
    }
}
