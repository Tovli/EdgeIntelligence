//! `el-engine-candle` — inference engine adapter over **Candle** (ADR-002),
//! implementing [`el_runtime::InferenceEngine`] / `RuntimeAcl`.
//!
//! Consumers supply their own model file; see [`CandleEngine::from_path`] and
//! [`CandleEngine::from_bytes`].  For tests that need a working engine without
//! a model asset, use [`CandleEngine::toy`].
//!
//! Expected GGUF tensor names:
//! - `token_embd.weight`  — embedding table  `[vocab, dim]`
//! - `output.weight` or `lm_head.weight` — lm-head  `[vocab, dim]`  (standard Llama layout)
//!
//! Float logits are quantised to integer milli-logits at the ACL boundary, so
//! Candle's `Tensor`/`Device` types never cross into the domain.

#![forbid(unsafe_code)]

use candle_core::{Device, Tensor};
use el_core::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ChatToken, EdgeError, LlmProvider, Result,
    SessionConfig, SessionId, Token,
};
use el_provenance::LoadPermit;
use el_runtime::{InferenceEngine, InferenceSession, Ports};

/// Candle-backed inference engine.
pub struct CandleEngine {
    embed: Tensor,
    w_out: Tensor,
    vocab: usize,
    eos: Token,
}

impl CandleEngine {
    /// Build a deterministic toy model on the CPU — no model file required.
    ///
    /// Uses fixed synthetic weights so tests are deterministic.
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

    /// Load `token_embd.weight` and `output.weight` from a consumer-supplied GGUF file.
    ///
    /// # Limitations
    /// This engine's forward pass is `embed[last_token] · w_out` — a single linear
    /// projection.  Only these two tensors are used; transformer blocks, attention,
    /// RoPE, and norms present in the GGUF are ignored.  Logits will not match a
    /// real Llama/Mistral/etc. forward.  This is the ADR-002 engine-seam proof; for
    /// a full transformer forward implement a separate [`InferenceEngine`] using
    /// `candle-transformers`.
    pub fn from_path(path: impl AsRef<std::path::Path>, eos: Token) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|_| EdgeError::Engine("model file not found or not readable"))?;
        Self::load_gguf(&mut std::io::BufReader::new(file), eos)
    }

    /// Load from raw bytes (WASM / memory-mapped scenarios).
    ///
    /// Same limitations as [`Self::from_path`]: only `token_embd.weight` and
    /// `output.weight` are used; the forward is `embed[last] · w_out`.
    pub fn from_bytes(data: &[u8], eos: Token) -> Result<Self> {
        Self::load_gguf(&mut std::io::Cursor::new(data), eos)
    }

    fn load_gguf<R: std::io::Read + std::io::Seek>(reader: &mut R, eos: Token) -> Result<Self> {
        use candle_core::quantized::gguf_file;

        let content = gguf_file::Content::read(reader)
            .map_err(|_| EdgeError::Engine("GGUF: invalid or unrecognised file"))?;
        let device = Device::Cpu;

        let embed = content
            .tensor(reader, "token_embd.weight", &device)
            .map_err(|_| EdgeError::Engine("GGUF: missing 'token_embd.weight'"))?
            .dequantize(&device)
            .map_err(|_| EdgeError::Engine("GGUF: cannot dequantize embed tensor"))?;

        let (vocab, dim) = match embed.shape().dims() {
            [v, d] => (*v, *d),
            _ => return Err(EdgeError::Engine("GGUF: 'token_embd.weight' must be 2-D")),
        };

        let raw_w_q = match content.tensor(reader, "output.weight", &device) {
            Ok(t) => t,
            Err(_) => content
                .tensor(reader, "lm_head.weight", &device)
                .map_err(|_| {
                    EdgeError::Engine("GGUF: missing 'output.weight' / 'lm_head.weight'")
                })?,
        };
        let raw_w = raw_w_q
            .dequantize(&device)
            .map_err(|_| EdgeError::Engine("GGUF: cannot dequantize output weight"))?;

        // Standard GGUF / Llama convention: output.weight is [vocab, dim].
        // We need [dim, vocab] so that embed_row [1,dim] × w_out [dim,vocab] → logits [1,vocab].
        let w_out = match raw_w.shape().dims() {
            [v, _d] if *v == vocab => raw_w
                .t()
                .map_err(|_| EdgeError::Engine("GGUF: failed to transpose output weight"))?,
            _ => raw_w,
        };

        // Validate that the output weight's inner dimension matches the embedding dimension.
        // A mismatch would silently produce all-zero logits at inference time.
        match w_out.shape().dims() {
            [d, v] if *d == dim && *v == vocab => {}
            _ => return Err(EdgeError::Engine(
                "GGUF: output weight shape incompatible with embed dim — expected [dim, vocab] after transpose",
            )),
        }

        Ok(Self {
            embed,
            w_out,
            vocab,
            eos,
        })
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
        Ok(tokens.len() as u32)
    }

    fn next_logits(&mut self, committed: &[Token]) -> Vec<i32> {
        let last = committed
            .last()
            .copied()
            .unwrap_or(0)
            .min(self.vocab as u32 - 1) as usize;
        match self.forward(last) {
            Ok(logits) => logits.iter().map(|x| (x * 1000.0).round() as i32).collect(),
            Err(_) => vec![0; self.vocab],
        }
    }

    fn eos_token(&self) -> Token {
        self.eos
    }
}

// ── LlmProvider (text-level) wrapper (ADR-010) ───────────────────────────────

/// Wraps a `CandleEngine` behind the `LlmProvider` trait using a byte-level
/// tokenizer.  A production build would swap in a HuggingFace tokenizer loaded
/// from the model file.
pub struct LocalLlmProvider {
    session: std::sync::Mutex<InferenceSession<CandleEngine>>,
    vocab: usize,
}

impl LocalLlmProvider {
    /// Load from a consumer-supplied GGUF file.
    pub fn from_path(
        path: impl AsRef<std::path::Path>,
        eos: Token,
        permit: LoadPermit,
    ) -> Result<Self> {
        let engine = CandleEngine::from_path(path, eos)?;
        let vocab = engine.vocab;
        let session = InferenceSession::new(SessionId(1), SessionConfig::default(), engine, permit);
        Ok(Self {
            session: std::sync::Mutex::new(session),
            vocab,
        })
    }

    /// Build a toy provider for testing.
    pub fn toy(vocab: usize, dim: usize, eos: Token, permit: LoadPermit) -> Result<Self> {
        let engine = CandleEngine::toy(vocab, dim, eos)?;
        let session = InferenceSession::new(SessionId(1), SessionConfig::default(), engine, permit);
        Ok(Self {
            session: std::sync::Mutex::new(session),
            vocab,
        })
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        text.bytes()
            .map(|b| (b as Token) % self.vocab as Token)
            .collect()
    }

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

    // ── helpers ──────────────────────────────────────────────────────────────

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

    /// Build a minimal but spec-compliant GGUF v3 file in memory.
    ///
    /// Layout:  no KV metadata, two F32 tensors:
    ///   `token_embd.weight`  [vocab, dim]  at offset 0
    ///   `output.weight`      [vocab, dim]  at offset vocab*dim*4
    ///
    /// GGUF stores dimensions innermost-first; candle reverses them on read.
    fn make_minimal_gguf(vocab: usize, dim: usize) -> Vec<u8> {
        let mut w: Vec<u8> = Vec::new();

        // Header
        w.extend_from_slice(b"GGUF");
        w.extend_from_slice(&3u32.to_le_bytes()); // version 3
        w.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        w.extend_from_slice(&0u64.to_le_bytes()); // n_kv (none)

        let tensor_bytes = (vocab * dim * 4) as u64;

        // token_embd.weight: [vocab, dim] → GGUF dims [dim, vocab]
        let name = b"token_embd.weight";
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name);
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&(dim as u64).to_le_bytes()); // innermost
        w.extend_from_slice(&(vocab as u64).to_le_bytes()); // outermost
        w.extend_from_slice(&0u32.to_le_bytes()); // F32
        w.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // output.weight: [vocab, dim] → GGUF dims [dim, vocab]; loader will transpose
        let name = b"output.weight";
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name);
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&(dim as u64).to_le_bytes());
        w.extend_from_slice(&(vocab as u64).to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&tensor_bytes.to_le_bytes()); // offset after embed

        // Pad to 32-byte alignment
        let pad = (32usize.wrapping_sub(w.len() % 32)) % 32;
        w.resize(w.len() + pad, 0u8);

        // Tensor data (both tensors, row-major f32)
        for i in 0..(vocab * dim * 2) {
            w.extend_from_slice(&(i as f32 * 0.1f32).to_le_bytes());
        }

        w
    }

    // ── toy-model tests (unchanged) ──────────────────────────────────────────

    #[test]
    fn real_candle_forward_is_deterministic_and_right_shape() {
        let mut eng = CandleEngine::toy(8, 4, 7).unwrap();
        let a = eng.next_logits(&[2]);
        let b = eng.next_logits(&[2]);
        assert_eq!(a.len(), 8, "logits length == vocab");
        assert_eq!(a, b, "fixed weights → deterministic real-tensor forward");
        let c = eng.next_logits(&[5]);
        assert_ne!(a, c);
    }

    #[test]
    fn drives_the_runtime_end_to_end() {
        use el_core::{ModelFormat, ModelId, ModelVersion, SessionConfig, SessionId, StopReason};
        use el_provenance::{ModelArtifact, SignatureVerifier};

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

        let eng = CandleEngine::toy(16, 8, 9999).unwrap();
        let mut session =
            InferenceSession::new(SessionId(1), SessionConfig::default(), eng, permit);
        let ports = Ports::permissive();
        session.load_prompt(&ports, &[1, 2, 3]).unwrap();

        let stop = session.generate(&ports, 4).unwrap();
        assert_eq!(stop, StopReason::MaxTokens);
        assert_eq!(session.output().len(), 4);
    }

    // ── GGUF loading tests ───────────────────────────────────────────────────

    #[test]
    fn from_bytes_rejects_invalid_magic() {
        let r = CandleEngine::from_bytes(b"not a gguf file", 0);
        assert!(matches!(r, Err(EdgeError::Engine(_))));
    }

    #[test]
    fn from_bytes_loads_minimal_gguf_and_forward_has_correct_vocab() {
        let vocab = 8;
        let dim = 4;
        let gguf = make_minimal_gguf(vocab, dim);
        let mut engine = CandleEngine::from_bytes(&gguf, 7).unwrap();

        let logits = engine.next_logits(&[0]);
        assert_eq!(logits.len(), vocab, "logit vec width == vocab from GGUF");
        assert_eq!(engine.eos_token(), 7);
    }

    #[test]
    fn from_bytes_gguf_forward_is_deterministic() {
        let gguf = make_minimal_gguf(8, 4);
        let mut eng = CandleEngine::from_bytes(&gguf, 0).unwrap();
        assert_eq!(eng.next_logits(&[3]), eng.next_logits(&[3]));
    }

    /// Same as `make_minimal_gguf` but `output.weight` has `wrong_dim` instead of `dim`,
    /// so the embed / output dimensions are incompatible.
    fn make_mismatched_gguf(vocab: usize, embed_dim: usize, output_dim: usize) -> Vec<u8> {
        let mut w: Vec<u8> = Vec::new();
        w.extend_from_slice(b"GGUF");
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes());

        let embed_bytes = (vocab * embed_dim * 4) as u64;

        let name = b"token_embd.weight";
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name);
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&(embed_dim as u64).to_le_bytes());
        w.extend_from_slice(&(vocab as u64).to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes());

        let name = b"output.weight";
        w.extend_from_slice(&(name.len() as u64).to_le_bytes());
        w.extend_from_slice(name);
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&(output_dim as u64).to_le_bytes()); // wrong dim
        w.extend_from_slice(&(vocab as u64).to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&embed_bytes.to_le_bytes());

        let pad = (32usize.wrapping_sub(w.len() % 32)) % 32;
        w.resize(w.len() + pad, 0u8);

        for i in 0..(vocab * embed_dim + vocab * output_dim) {
            w.extend_from_slice(&(i as f32 * 0.1f32).to_le_bytes());
        }
        w
    }

    #[test]
    fn from_path_missing_file_returns_engine_error() {
        let r = CandleEngine::from_path(std::path::Path::new("/nonexistent/model.gguf"), 0);
        assert!(matches!(r, Err(EdgeError::Engine(_))));
    }

    #[test]
    fn from_bytes_rejects_mismatched_output_dim_at_load_time() {
        // embed dim=4, output dim=7 — incompatible; must error at load, not silently at forward.
        let gguf = make_mismatched_gguf(8, 4, 7);
        let r = CandleEngine::from_bytes(&gguf, 0);
        assert!(
            matches!(r, Err(EdgeError::Engine(_))),
            "mismatched output weight dim must be rejected at load time"
        );
    }

    // ── LocalLlmProvider tests (unchanged + new from_path error path) ────────

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
        assert!(tokens.last().unwrap().is_final);
        assert!(tokens.len() > 1);
    }

    #[test]
    fn local_provider_session_resets_between_calls() {
        let p = LocalLlmProvider::toy(32, 8, 31, ok_permit()).unwrap();
        let req = el_core::ChatRequest::new("local", vec![el_core::ChatMessage::user("a")])
            .with_max_tokens(4);
        let r1 = p.chat(&req).unwrap();
        let r2 = p.chat(&req).unwrap();
        assert_eq!(r1.content, r2.content);
    }

    #[test]
    fn local_provider_from_path_missing_file_returns_error() {
        let r = LocalLlmProvider::from_path(
            std::path::Path::new("/nonexistent/model.gguf"),
            0,
            ok_permit(),
        );
        assert!(matches!(r, Err(EdgeError::Engine(_))));
    }
}
