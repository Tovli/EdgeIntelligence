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
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ChatToken, DomainEvent, EdgeError,
    LlmProvider, Result, SafetyMode, SessionConfig, SessionId, StopReason, Token,
};
use el_provenance::LoadPermit;
use el_runtime::{
    AnchorGuard, ContrastiveSteerer, ExpertLogits, InferenceEngine, InferenceSession,
    LightweightFilter, NoSafety, Ports, SafetyModeSelector, SafetySteerer,
};

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

    /// Stateless: this engine's forward is `embed[committed.last()] · w_out`, so
    /// it holds no KV cache to restore — a rollback is a no-op.
    fn rollback(&mut self, _keep_committed: u32) -> Result<()> {
        Ok(())
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

// ── Real Qwen2 transformer engine + chat provider (ADR-002 + ADR-010) ────────
//
// Unlike `CandleEngine` (a single linear projection used as the engine-seam
// proof) this runs a genuine Qwen2 transformer forward via `candle-transformers`
// with a real HuggingFace tokenizer, so it produces coherent chat. It plugs into
// the SAME `el_runtime::InferenceSession` decode loop as every other engine —
// nothing in the SDK pipeline is bypassed.

use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use el_core::{ModelId, ModelVersion};
use el_provenance::{ModelArtifact, SignatureVerifier};
use tokenizers::Tokenizer;

// ── Opt-in benchmark instrumentation (EL_BENCH=1) ────────────────────────────
//
// Zero-cost when `EL_BENCH` is unset: `enabled()` short-circuits and no timing
// is taken. When set, `QwenChatProvider::chat` prints a per-phase breakdown and
// per-forward attribution (model compute vs. seam quantisation vs. runtime loop)
// to stderr. Diagnostics only — not part of the SDK's public behaviour.
mod bench {
    use std::cell::Cell;
    use std::sync::OnceLock;
    use std::time::Duration;

    static ENABLED: OnceLock<bool> = OnceLock::new();

    /// True iff the `EL_BENCH` environment variable is present (read once).
    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var_os("EL_BENCH").is_some())
    }

    thread_local! {
        static FWD_TOTAL: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        static FWD_MODEL: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        static FWD_CALLS: Cell<u64> = const { Cell::new(0) };
    }

    /// Accumulate one `forward_one` sample: `total` is the whole seam call,
    /// `model` is just the candle transformer forward inside it.
    pub fn record(total: Duration, model: Duration) {
        FWD_TOTAL.with(|c| c.set(c.get() + total));
        FWD_MODEL.with(|c| c.set(c.get() + model));
        FWD_CALLS.with(|c| c.set(c.get() + 1));
    }

    /// Read and reset the forward accumulators: `(total, model, calls)`.
    pub fn take() -> (Duration, Duration, u64) {
        (
            FWD_TOTAL.replace(Duration::ZERO),
            FWD_MODEL.replace(Duration::ZERO),
            FWD_CALLS.replace(0),
        )
    }
}

/// A real Qwen2 transformer `InferenceEngine`.
///
/// Holds candle's stateful KV cache. Within one generation it is fed
/// incrementally (prefill, then one new token per `next_logits` call); candle
/// exposes no public cache reset, so a *fresh conversation* builds a new engine.
///
/// A *within-generation* safety backtrack (ADR-012) is supported via
/// [`InferenceEngine::rollback`]: candle's attention discards its cache when a
/// forward runs at `index_pos == 0`, so we retain the prompt and replay it from
/// position 0 to rebuild the cache for the safe prefix (the session then
/// re-feeds the retained committed tokens). Float logits are quantised to
/// integer milli-logits at the seam, exactly like [`CandleEngine`], so the
/// runtime stays float-free.
pub struct QwenEngine {
    model: Qwen2Weights,
    device: Device,
    /// Absolute KV position written so far (candle's `index_pos`).
    index_pos: usize,
    /// How many of the runtime-`committed` tokens have already been fed.
    fed: usize,
    /// The prefill prompt, retained so a rollback can replay it from position 0
    /// to rebuild candle's KV cache (which has no public truncation).
    prompt: Vec<Token>,
    /// Milli-logits produced after the most recent forward.
    last_logits: Vec<i32>,
    vocab: usize,
    eos: Token,
}

impl QwenEngine {
    /// Load Qwen2 weights from a consumer-supplied GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>, eos: Token) -> Result<Self> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|_| EdgeError::Engine("model file not found or not readable"))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|_| EdgeError::Engine("GGUF: invalid or unrecognised file"))?;
        let device = Device::Cpu;
        let model = Qwen2Weights::from_gguf(content, &mut file, &device)
            .map_err(|_| EdgeError::Engine("GGUF: failed to load Qwen2 weights"))?;
        Ok(Self {
            model,
            device,
            index_pos: 0,
            fed: 0,
            prompt: Vec::new(),
            last_logits: Vec::new(),
            vocab: 0,
            eos,
        })
    }

    /// One forward over a single token at the current position; advances the KV
    /// cache and returns milli-logits for the next token.
    fn forward_one(&mut self, token: Token) -> Result<Vec<i32>> {
        let t_total = bench::enabled().then(std::time::Instant::now);

        let input = Tensor::from_vec(vec![token], (1, 1), &self.device)
            .map_err(|_| EdgeError::Engine("candle: input tensor build failed"))?;

        let t_model = bench::enabled().then(std::time::Instant::now);
        let logits = self
            .model
            .forward(&input, self.index_pos)
            .map_err(|_| EdgeError::Engine("candle: Qwen2 forward failed"))?;
        let model_dur = t_model.map(|t| t.elapsed()).unwrap_or_default();

        self.index_pos += 1;
        let row = logits
            .squeeze(0)
            .map_err(|_| EdgeError::Engine("candle: squeeze logits failed"))?;
        let floats = row
            .to_vec1::<f32>()
            .map_err(|_| EdgeError::Engine("candle: logits to vec failed"))?;
        let out: Vec<i32> = floats.iter().map(|x| (x * 1000.0).round() as i32).collect();

        if let Some(t) = t_total {
            bench::record(t.elapsed(), model_dur);
        }
        Ok(out)
    }
}

impl InferenceEngine for QwenEngine {
    fn prefill(&mut self, tokens: &[Token]) -> Result<u32> {
        self.index_pos = 0;
        self.fed = 0;
        self.prompt = tokens.to_vec(); // retained for rollback replay
        for &t in tokens {
            self.last_logits = self.forward_one(t)?;
        }
        self.vocab = self.last_logits.len();
        Ok(tokens.len() as u32)
    }

    fn next_logits(&mut self, committed: &[Token]) -> Vec<i32> {
        // Feed any newly committed (generated) tokens beyond what we've seen.
        // `committed` grows by exactly one per decode step, so this feeds the
        // token the runtime just sampled and returns the next distribution.
        while self.fed < committed.len() {
            let t = committed[self.fed];
            match self.forward_one(t) {
                Ok(l) => self.last_logits = l,
                Err(_) => return vec![0; self.vocab.max(1)],
            }
            self.fed += 1;
        }
        self.last_logits.clone()
    }

    fn eos_token(&self) -> Token {
        self.eos
    }

    fn rollback(&mut self, _keep_committed: u32) -> Result<()> {
        // candle's KV cache is append-only with no public truncation, but its
        // attention discards the cache on a forward at `index_pos == 0` (see
        // quantized_qwen2). So rebuild deterministically: replay the prompt from
        // position 0 — the first forward resets the cache, the rest re-append it —
        // leaving the engine in its exact post-prefill state. We reset `fed` to 0
        // so the session's next `next_logits` re-feeds the retained committed
        // prefix (already truncated to `keep_committed`) on top. Cost is bounded
        // by `max_rollbacks` (ADR-012).
        self.index_pos = 0;
        self.fed = 0;
        for i in 0..self.prompt.len() {
            let t = self.prompt[i];
            self.last_logits = self.forward_one(t)?;
        }
        Ok(())
    }
}

// ── On-device safety wiring (ADR-005 tier + ADR-012 control loop) ────────────
//
// The runtime ships the *primitives* (steerer, chunk guard, checkpointed
// rollback). They only engage when a session is given a real steerer + guard in
// its `Ports` — `Ports::permissive()` wires neither, so a provider must opt in.
// This adapter does: it owns the tokenizer, so it is the one place that can turn
// a human-readable unsafe-word list into the token-id patterns the runtime's
// float-free guard consumes. The resolved patterns/bans then drive the standard
// `InferenceSession::generate` control loop — nothing in the SDK is bypassed.

/// A small, conservative built-in `Lightweight` safety list (ADR-005). These are
/// unambiguous weapons/mass-harm manufacture terms — content the decode-time
/// guard should never let the model emit. It is intentionally narrow to avoid
/// false positives in ordinary chat; production swaps in the active tier's real
/// safety model (the LoRA adapter / classifier of ADR-012's model inventory).
const DEFAULT_UNSAFE_WORDS: &[&str] = &[
    "bomb",
    "explosive",
    "detonator",
    "methamphetamine",
    "ricin",
    "anthrax",
    "sarin",
    "nerve agent",
];

/// Deterministic hard refusal emitted when the control loop fails closed —
/// rollbacks exhausted, no safe checkpoint, or refused at ingress (ADR-012
/// §"Bounded rollback, fail-closed"; ADR-013 ingress triage).
const SAFETY_REFUSAL: &str = "I can't help with that request.";

/// Contrastive steering is restricted to the top-K base-logit tokens (ADR-013 /
/// SafeDecoding): it keeps the per-step adjustment small (so the runtime's
/// `pick` stays linear in the vocab) and avoids amplifying long-tail noise.
const CONTRASTIVE_TOP_K: usize = 64;

/// Encode a word to the token-id sequence(s) the model may actually emit for
/// it. A word tokenizes differently depending on what precedes it, so both the
/// **leading-space** form (mid-sentence, after another token) and the **bare**
/// form (start of a line/turn or after punctuation) are returned — each as a
/// distinct anchor n-gram. Empty/duplicate encodings are dropped, so the caller
/// gets only matchable patterns.
fn word_to_patterns(tokenizer: &Tokenizer, word: &str) -> Vec<Vec<Token>> {
    let mut out: Vec<Vec<Token>> = Vec::new();
    for variant in [format!(" {word}"), word.to_string()] {
        if let Ok(enc) = tokenizer.encode(variant, false) {
            let seq = enc.get_ids().to_vec();
            if !seq.is_empty() && !out.contains(&seq) {
                out.push(seq);
            }
        }
    }
    out
}

/// Resolved safety wiring for the provider, derived from the tokenizer once at
/// construction. Holds only token-id data, so rebuilding a turn's `Ports` is a
/// cheap clone with no tokenizer access on the hot path.
#[derive(Debug, Clone)]
struct SafetyConfig {
    /// The ADR-005 tier. `Off` runs the plain single-pass decode (legacy path).
    mode: SafetyMode,
    /// Hard-banned single tokens — the always-on per-step `LightweightFilter`
    /// layer (only words that encode to exactly one token; banning a shared
    /// subword would be too blunt).
    banned: Vec<Token>,
    /// Built-in unsafe token-id n-grams — drive **both** the ADR-012 output
    /// chunk guard and the ADR-013 prompt ingress triage.
    patterns: Vec<Vec<Token>>,
    /// Caller-supplied `--guard-word` n-grams — a **guard-only** demo/test hook.
    /// Deliberately excluded from ingress so the documented rollback demo
    /// (`--guard-word banana --prompt "…banana…"`) fires the *trajectory* loop
    /// instead of refusing the prompt before decoding.
    extra_guard_patterns: Vec<Vec<Token>>,
}

impl SafetyConfig {
    /// Resolve the built-in `Lightweight` list against `tokenizer`.
    fn lightweight(tokenizer: &Tokenizer) -> Self {
        let mut banned = Vec::new();
        let mut patterns = Vec::new();
        for &word in DEFAULT_UNSAFE_WORDS {
            for seq in word_to_patterns(tokenizer, word) {
                // A single-token form is safe to hard-ban per step; multi-token
                // forms are caught by the guard (banning a shared subword could
                // hurt benign text).
                if seq.len() == 1 && !banned.contains(&seq[0]) {
                    banned.push(seq[0]);
                }
                if !patterns.contains(&seq) {
                    patterns.push(seq);
                }
            }
        }
        Self {
            mode: SafetyMode::Lightweight,
            banned,
            patterns,
            extra_guard_patterns: Vec::new(),
        }
    }

    /// Build the per-turn safety `Ports` (steerer + chunk guard) for this tier.
    /// `Off`, or an empty list, yields no steering/guarding.
    fn ports(&self) -> Ports {
        let mut ports = Ports::permissive();
        if matches!(self.mode, SafetyMode::Off) {
            return ports;
        }
        let steerer: Box<dyn SafetySteerer> = if self.banned.is_empty() {
            Box::new(NoSafety)
        } else {
            Box::new(LightweightFilter::new(self.banned.clone()))
        };
        ports.safety = steerer;
        // Output chunk guard (ADR-012): built-in unsafe patterns + caller's
        // --guard-word extras.
        let guard_patterns: Vec<Vec<Token>> = self
            .patterns
            .iter()
            .chain(self.extra_guard_patterns.iter())
            .cloned()
            .collect();
        if !guard_patterns.is_empty() {
            ports.guard = Some(Box::new(AnchorGuard::hard(guard_patterns)));
        }
        // Prompt ingress triage (ADR-013): built-in patterns ONLY — the
        // --guard-word extras are a trajectory demo hook, not ingress refusals.
        if !self.patterns.is_empty() {
            ports.ingress = Some(Box::new(AnchorGuard::hard(self.patterns.clone())));
        }
        ports
    }
}

/// A safety **expert** logit source for contrastive steering (ADR-013): a second
/// Qwen engine — in production base + a safety LoRA; here any same-tokenizer Qwen
/// GGUF — loaded through the ADR-006 provenance gate and **primed with the turn's
/// prompt** so its logits align with the base engine's. The session feeds it the
/// committed tokens via [`ExpertLogits::logits`]; interior mutability is required
/// because each forward advances candle's KV cache.
///
/// Steering is bounded to the early-token window (ADR-013), so the expert runs
/// only for the first `steer_window` tokens. When the base engine rolls back
/// (committed output shrinks), the expert **re-primes to the prompt** and
/// re-feeds the retained prefix, so its contrastive context stays aligned with
/// the base rather than serving logits from the abandoned branch. Pointing this
/// at the chat model itself yields ~zero contrast (a no-op); a safety-tuned Qwen
/// GGUF gives real steering.
pub struct QwenExpert {
    engine: std::cell::RefCell<QwenEngine>,
    /// How many committed tokens the expert has fed since its last prime — used
    /// to detect a base rollback (committed shrinks below this) and re-sync.
    fed: std::cell::Cell<usize>,
    /// Evidence the expert weights passed the ADR-006 load gate (R5). Held for
    /// the engine's lifetime; never used after construction.
    _permit: LoadPermit,
}

impl QwenExpert {
    /// Load the expert GGUF, gate it (ADR-006 — `permit` is required, not
    /// optional), and prime it with `prompt` so its KV state matches the base
    /// engine's post-prefill state.
    pub fn from_path_primed(
        path: impl AsRef<std::path::Path>,
        eos: Token,
        prompt: &[Token],
        permit: LoadPermit,
    ) -> Result<Self> {
        let mut engine = QwenEngine::from_path(path, eos)?;
        engine.prefill(prompt)?;
        Ok(Self {
            engine: std::cell::RefCell::new(engine),
            fed: std::cell::Cell::new(0),
            _permit: permit,
        })
    }
}

impl ExpertLogits for QwenExpert {
    fn logits(&self, committed: &[Token]) -> Vec<i32> {
        let mut engine = self.engine.borrow_mut();
        // Base rolled back? `committed` shrank below what we've fed. Re-prime the
        // expert to the prompt (QwenEngine::rollback replays the prompt and
        // resets its feed cursor) so it re-feeds the retained prefix from a clean
        // state — keeping the contrastive context aligned with the base. Cost is
        // bounded by `max_rollbacks` (ADR-012), same as the base engine.
        if committed.len() < self.fed.get() {
            if engine.rollback(0).is_err() {
                return Vec::new();
            }
            self.fed.set(0);
        }
        let out = engine.next_logits(committed);
        self.fed.set(committed.len());
        out
    }
}

/// A real local chat backend: a Qwen2 GGUF model + its tokenizer, driven
/// through [`el_runtime::InferenceSession`].
///
/// Each `chat` call renders the whole conversation to Qwen2.5 ChatML, builds a
/// fresh [`QwenEngine`] (candle has no public KV-cache reset), then runs the
/// SDK's standard provenance-gated session: `load_prompt` (prefill) →
/// `generate` (grammar mask → safety steer → guard + checkpointed rollback →
/// greedy commit). On-device safety (ADR-005 `Lightweight` tier + the ADR-012
/// control loop) is **on by default**; see [`with_safety`](Self::with_safety).
/// The provider holds no mutable session state, so it is `Send + Sync` without
/// locking.
pub struct QwenChatProvider {
    model_path: std::path::PathBuf,
    tokenizer: Tokenizer,
    permit: LoadPermit,
    eos: Token,
    default_max_tokens: u32,
    model_label: String,
    safety: SafetyConfig,
    /// Optional safety **expert** GGUF for ADR-013 contrastive steering. `None`
    /// runs the token-only `Lightweight` steerer.
    expert_model: Option<std::path::PathBuf>,
    /// Contrastive steering strength ×1000 (1000 = 1.0×).
    steer_alpha_milli: i32,
}

impl QwenChatProvider {
    /// Load a Qwen2 GGUF model and its `tokenizer.json` from local paths.
    pub fn from_paths(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(EdgeError::Engine("model file not found"));
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|_| EdgeError::Engine("failed to load tokenizer.json"))?;

        // Stop token: Qwen2.5 ChatML turn terminator (fallback to its known id).
        let eos = tokenizer.token_to_id("<|im_end|>").unwrap_or(151_645);

        let model_label = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("local/{s}"))
            .unwrap_or_else(|| "local/qwen2".to_string());

        let safety = SafetyConfig::lightweight(&tokenizer);

        let permit = local_load_permit(&model_path)?;
        Ok(Self {
            model_path,
            tokenizer,
            permit,
            eos,
            default_max_tokens: 512,
            model_label,
            safety,
            expert_model: None,
            steer_alpha_milli: 1000,
        })
    }

    /// Select the on-device safety tier (ADR-005). [`SafetyMode::Off`] disables
    /// the steerer and the ADR-012 control loop (the plain single-pass decode);
    /// [`SafetyMode::Lightweight`] (the default) runs the token-anchor guard +
    /// hard-ban steerer + checkpointed rollback. `SecDecoding`/`Csd` need model
    /// assets not shipped here and fall back to the `Lightweight` wiring.
    pub fn with_safety(mut self, mode: SafetyMode) -> Self {
        self.safety.mode = mode;
        self
    }

    /// Add extra words to the chunk guard's unsafe patterns (resolved to token
    /// ids via this model's tokenizer). Primarily a **test/demo hook**: e.g.
    /// `--guard-word banana` lets you watch the ADR-012 rollback / fail-closed
    /// refusal fire on a benign word, without needing the model to emit genuinely
    /// harmful content. Guard-only — these are not added to the hard-ban list, so
    /// the trajectory loop (not silent suppression) is what engages.
    pub fn with_extra_guard_words<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for word in words {
            for seq in word_to_patterns(&self.tokenizer, word.as_ref()) {
                if !self.safety.extra_guard_patterns.contains(&seq) {
                    self.safety.extra_guard_patterns.push(seq);
                }
            }
        }
        self
    }

    /// Enable model-backed **contrastive** steering (ADR-013) with a safety
    /// **expert** GGUF (same tokenizer/family as the chat model). Steering runs
    /// only inside the early-token window. Pointing this at the chat model itself
    /// gives ~zero contrast (a no-op); a safety-tuned Qwen GGUF gives real
    /// steering. No effect under `--safety off`.
    pub fn with_expert_model(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.expert_model = Some(path.as_ref().to_path_buf());
        self
    }

    /// Contrastive steering strength ×1000 (`1000` = 1.0×). Only meaningful with
    /// [`with_expert_model`](Self::with_expert_model).
    pub fn with_steer_alpha(mut self, alpha_milli: i32) -> Self {
        self.steer_alpha_milli = alpha_milli;
        self
    }

    fn encode(&self, text: &str) -> Result<Vec<Token>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|_| EdgeError::Engine("tokenizer encode failed"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[Token]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|_| EdgeError::Engine("tokenizer decode failed"))
    }
}

impl LlmProvider for QwenChatProvider {
    fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let prompt = render_chatml(&req.messages);

        let t_encode = bench::enabled().then(std::time::Instant::now);
        let prompt_tokens = self.encode(&prompt)?;
        let d_encode = t_encode.map(|t| t.elapsed()).unwrap_or_default();

        // Fresh engine + session each turn (candle KV cache has no public reset);
        // the full conversation is re-prefilled. This is the standard SDK path —
        // provenance permit, session lifecycle, decode loop — not a shortcut.
        let t_load = bench::enabled().then(std::time::Instant::now);
        let engine = QwenEngine::from_path(&self.model_path, self.eos)?;
        let d_load = t_load.map(|t| t.elapsed()).unwrap_or_default();

        // Carry the active safety tier on the session config so the runtime
        // derives the tier-aware ADR-012 `RollbackPolicy` and records the true
        // mode. A supplied expert promotes the tier to `SecDecoding`, so the
        // runtime's `SafetyModeSelector` can gate it on device class instead of
        // it masquerading as `Lightweight`.
        let requested = requested_session_safety(self.safety.mode, self.expert_model.is_some());
        let cfg = SessionConfig {
            safety: requested,
            ..SessionConfig::default()
        };
        // Resolve the same effective tier the runtime will: only install the
        // contrastive steerer if `SecDecoding` survives device selection (it
        // downgrades to `Lightweight` on non-accelerator devices, where the
        // expert is dropped — honest tier-aware behaviour).
        let effective = SafetyModeSelector::resolve(requested, cfg.device);
        let mut session = InferenceSession::new(SessionId(1), cfg, engine, self.permit);
        let mut ports = self.safety.ports();

        if matches!(effective, SafetyMode::SecDecoding) {
            if let Some(expert_path) = &self.expert_model {
                let expert = QwenExpert::from_path_primed(
                    expert_path,
                    self.eos,
                    &prompt_tokens,
                    local_load_permit(expert_path)?,
                )?;
                ports.safety = Box::new(ContrastiveSteerer::new(
                    expert,
                    self.safety.banned.clone(),
                    self.steer_alpha_milli,
                    CONTRASTIVE_TOP_K,
                    effective,
                ));
            }
        }

        let _ = bench::take(); // clear forward accumulators before prefill
        let t_prefill = bench::enabled().then(std::time::Instant::now);
        session.load_prompt(&ports, &prompt_tokens)?;
        let d_prefill = t_prefill.map(|t| t.elapsed()).unwrap_or_default();
        let (pf_total, pf_model, pf_calls) = bench::take();

        let max = req.max_tokens.unwrap_or(self.default_max_tokens);
        let t_decode = bench::enabled().then(std::time::Instant::now);
        let stop = session.generate(&ports, max)?;
        let d_decode = t_decode.map(|t| t.elapsed()).unwrap_or_default();
        let (dc_total, dc_model, dc_calls) = bench::take();

        let out = session.output().to_vec();
        let completion_tokens = out.len() as u32;

        let t_detok = bench::enabled().then(std::time::Instant::now);
        let decoded = self.decode(&out)?.trim().to_string();
        let d_detok = t_detok.map(|t| t.elapsed()).unwrap_or_default();

        // ADR-012: surface what the decode-time control loop did. A fail-closed
        // stop (rollbacks exhausted / no safe checkpoint) returns the
        // deterministic refusal rather than the truncated unsafe prefix; any
        // intervention is reported on stderr so the test client can show the
        // guard working without corrupting the reply on stdout.
        let safety_active = !matches!(self.safety.mode, SafetyMode::Off);
        let content = if safety_active {
            let events = session.drain_events();
            let violations = events
                .iter()
                .filter(|e| matches!(e.event, DomainEvent::SafetyViolationDetected { .. }))
                .count();
            let rollbacks = events
                .iter()
                .filter(|e| matches!(e.event, DomainEvent::ClaimBacktracked { .. }))
                .count();
            let refused = stop == StopReason::Stopped && violations > 0;
            if violations > 0 || rollbacks > 0 {
                eprintln!(
                    "[safety] {violations} violation(s), {rollbacks} rollback(s){}",
                    if refused {
                        " → refused (fail-closed)"
                    } else {
                        " → recovered"
                    }
                );
            }
            if refused {
                SAFETY_REFUSAL.to_string()
            } else {
                decoded
            }
        } else {
            decoded
        };

        if bench::enabled() {
            report_breakdown(
                prompt_tokens.len() as u32,
                completion_tokens,
                d_load,
                d_encode,
                d_prefill,
                d_decode,
                d_detok,
                (pf_total, pf_model, pf_calls),
                (dc_total, dc_model, dc_calls),
            );
        }

        Ok(ChatResponse {
            content,
            model: self.model_label.clone(),
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens,
        })
    }

    fn chat_stream(&self, req: &ChatRequest, on_token: &mut dyn FnMut(ChatToken)) -> Result<()> {
        // The runtime decode loop runs to completion internally (no per-token
        // hook), so — like the toy `LocalLlmProvider` — we stream the finished
        // reply out character by character.
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

/// Print an `EL_BENCH` per-phase + per-forward breakdown for one `chat()` call.
#[allow(clippy::too_many_arguments)]
fn report_breakdown(
    prompt_tokens: u32,
    completion_tokens: u32,
    d_load: std::time::Duration,
    d_encode: std::time::Duration,
    d_prefill: std::time::Duration,
    d_decode: std::time::Duration,
    d_detok: std::time::Duration,
    prefill_fwd: (std::time::Duration, std::time::Duration, u64),
    decode_fwd: (std::time::Duration, std::time::Duration, u64),
) {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let total = d_load + d_encode + d_prefill + d_decode + d_detok;
    let pct = |d: std::time::Duration| {
        if total.as_secs_f64() > 0.0 {
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        } else {
            0.0
        }
    };
    let tps = |n: u32, d: std::time::Duration| {
        if d.as_secs_f64() > 0.0 {
            n as f64 / d.as_secs_f64()
        } else {
            0.0
        }
    };

    let (pf_total, pf_model, pf_calls) = prefill_fwd;
    let (dc_total, dc_model, dc_calls) = decode_fwd;
    let dc_loop = d_decode.saturating_sub(dc_total);
    let dc_seam = dc_total.saturating_sub(dc_model);
    let per_tok = |d: std::time::Duration, n: u64| if n > 0 { ms(d) / n as f64 } else { 0.0 };

    eprintln!("\n┌─ EL_BENCH chat() breakdown ───────────────────────────────");
    eprintln!("│ prompt_tokens={prompt_tokens}  completion_tokens={completion_tokens}");
    eprintln!("│ phase           wall(ms)    %total   throughput");
    eprintln!(
        "│ model load    {:>9.1}  {:>6.1}%   (read+dequantize GGUF)",
        ms(d_load),
        pct(d_load)
    );
    eprintln!(
        "│ tokenize       {:>9.2}  {:>6.1}%",
        ms(d_encode),
        pct(d_encode)
    );
    eprintln!(
        "│ prefill       {:>9.1}  {:>6.1}%   {:>7.1} tok/s",
        ms(d_prefill),
        pct(d_prefill),
        tps(prompt_tokens, d_prefill)
    );
    eprintln!(
        "│ decode        {:>9.1}  {:>6.1}%   {:>7.1} tok/s",
        ms(d_decode),
        pct(d_decode),
        tps(completion_tokens, d_decode)
    );
    eprintln!(
        "│ detokenize     {:>9.2}  {:>6.1}%",
        ms(d_detok),
        pct(d_detok)
    );
    eprintln!("│ TOTAL         {:>9.1}", ms(total));
    eprintln!("│ ─ forward attribution (where prefill+decode time goes) ─");
    eprintln!(
        "│ prefill: {} fwd calls, model {:.1}ms, seam {:.1}ms, loop {:.1}ms",
        pf_calls,
        ms(pf_model),
        ms(pf_total.saturating_sub(pf_model)),
        ms(d_prefill.saturating_sub(pf_total)),
    );
    eprintln!(
        "│ decode : {} fwd calls, model {:.1}ms, seam {:.1}ms, loop {:.1}ms",
        dc_calls,
        ms(dc_model),
        ms(dc_seam),
        ms(dc_loop),
    );
    eprintln!(
        "│ per decoded token: {:.2}ms total = model {:.2} + seam {:.2} + loop {:.2}",
        per_tok(d_decode, dc_calls),
        per_tok(dc_model, dc_calls),
        per_tok(dc_seam, dc_calls),
        per_tok(dc_loop, dc_calls),
    );
    eprintln!("└───────────────────────────────────────────────────────────");
}

/// Render a conversation as Qwen2.5 ChatML and open an assistant turn.
fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        let role = match m.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        s.push_str("<|im_start|>");
        s.push_str(role);
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

fn requested_session_safety(configured: SafetyMode, has_expert: bool) -> SafetyMode {
    match (configured, has_expert) {
        (SafetyMode::Off, _) => SafetyMode::Off,
        // A supplied expert is the only backed SecDecoding implementation in
        // this adapter. Promote any non-Off configured tier to that concrete
        // model-backed path so the runtime selector can gate it by device.
        (_, true) => SafetyMode::SecDecoding,
        // These public enum variants are not backed here without an expert.
        // Keep telemetry/policy honest by reflecting the lightweight ports that
        // will actually be installed.
        (SafetyMode::SecDecoding | SafetyMode::Csd, false) => SafetyMode::Lightweight,
        (mode, false) => mode,
    }
}

/// Obtain a [`LoadPermit`] through the ADR-006 gate for a user-supplied local
/// model. There is no detached signature to check for a file the user downloaded
/// themselves, so this uses a trust-the-local-file verifier. This is explicitly
/// **not** cryptographic integrity over the GGUF bytes; production signed assets
/// must use a separate verifier path that reads the whole artifact and verifies
/// its detached signature before issuing a permit.
fn local_load_permit(path: &std::path::Path) -> Result<LoadPermit> {
    struct LocalFileTrust;
    impl SignatureVerifier for LocalFileTrust {
        fn verify(&self, _bytes: &[u8], _sig: &[u8], _key: u32) -> bool {
            true
        }
    }
    // Keep the local-trust path cheap: it proves callers go through the permit
    // gate, while deliberately avoiding fake "verification" of path strings or
    // header fragments that could be mistaken for artifact integrity.
    let _ = path;
    let mut artifact = ModelArtifact::new(
        ModelId(1),
        ModelVersion::new(0, 1, 0),
        el_core::ModelFormat::Gguf,
    );
    artifact.verify(&LocalFileTrust, b"local-trust", b"", 0);
    artifact.ensure_loadable()
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

    // ── Qwen provider helpers ─────────────────────────────────────────────────

    #[test]
    fn render_chatml_wraps_each_turn_and_opens_assistant() {
        let msgs = vec![
            ChatMessage::system("be nice"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
            ChatMessage::user("bye"),
        ];
        let got = render_chatml(&msgs);
        let want = "<|im_start|>system\nbe nice<|im_end|>\n\
                    <|im_start|>user\nhi<|im_end|>\n\
                    <|im_start|>assistant\nhello<|im_end|>\n\
                    <|im_start|>user\nbye<|im_end|>\n\
                    <|im_start|>assistant\n";
        assert_eq!(got, want);
    }

    #[test]
    fn local_load_permit_passes_the_provenance_gate() {
        // The runtime requires a LoadPermit; the local-trust path must yield one
        // for a GGUF artifact (ADR-006 gate exercised, not bypassed).
        let permit = local_load_permit(std::path::Path::new("models/qwen.gguf"))
            .expect("local permit issued");
        assert_eq!(permit.format, el_core::ModelFormat::Gguf);
    }

    #[test]
    fn requested_safety_matches_the_backed_steerer_surface() {
        assert_eq!(
            requested_session_safety(SafetyMode::Off, true),
            SafetyMode::Off,
            "Off must stay off even if an expert path is configured"
        );
        assert_eq!(
            requested_session_safety(SafetyMode::Lightweight, true),
            SafetyMode::SecDecoding,
            "an expert promotes the concrete model-backed path"
        );
        assert_eq!(
            requested_session_safety(SafetyMode::SecDecoding, false),
            SafetyMode::Lightweight,
            "unbacked SecDecoding must not be reported as active"
        );
        assert_eq!(
            requested_session_safety(SafetyMode::Csd, false),
            SafetyMode::Lightweight,
            "unbacked Csd must not be reported as active"
        );
    }

    #[test]
    fn qwen_provider_from_paths_missing_model_errors() {
        let r = QwenChatProvider::from_paths(
            std::path::Path::new("/nonexistent/model.gguf"),
            std::path::Path::new("/nonexistent/tokenizer.json"),
        );
        assert!(matches!(r, Err(EdgeError::Engine(_))));
    }

    // ── safety wiring (ADR-005 tier + ADR-012 control loop) ──────────────────

    #[test]
    fn safety_off_wires_no_guard_or_steering() {
        // Off → the plain single-pass decode: `Ports::permissive()` semantics
        // regardless of any resolved bans/patterns.
        let cfg = SafetyConfig {
            mode: SafetyMode::Off,
            banned: vec![1],
            patterns: vec![vec![2]],
            extra_guard_patterns: vec![],
        };
        let ports = cfg.ports();
        assert!(ports.guard.is_none(), "Off must not wire the chunk guard");
        assert!(ports.ingress.is_none(), "Off must not wire ingress triage");
        assert_eq!(
            ports.safety.mode(),
            SafetyMode::Off,
            "Off must keep the no-op steerer"
        );
    }

    #[test]
    fn lightweight_wires_guard_and_hard_ban_steerer() {
        let cfg = SafetyConfig {
            mode: SafetyMode::Lightweight,
            banned: vec![1],
            patterns: vec![vec![2, 3]],
            extra_guard_patterns: vec![],
        };
        let ports = cfg.ports();
        assert!(
            ports.guard.is_some(),
            "Lightweight must wire the chunk guard"
        );
        assert!(
            ports.ingress.is_some(),
            "Lightweight must wire prompt ingress triage (ADR-013)"
        );
        assert_eq!(
            ports.safety.mode(),
            SafetyMode::Lightweight,
            "a non-empty ban list selects the LightweightFilter steerer"
        );
    }

    #[test]
    fn lightweight_without_patterns_has_no_guard_or_ingress() {
        // No resolvable unsafe patterns (e.g. all multi-token and tokenizer
        // produced nothing) → guard/ingress stay off; the per-step ban can still
        // apply.
        let cfg = SafetyConfig {
            mode: SafetyMode::Lightweight,
            banned: vec![7],
            patterns: vec![],
            extra_guard_patterns: vec![],
        };
        let ports = cfg.ports();
        assert!(ports.guard.is_none());
        assert!(ports.ingress.is_none());
    }

    #[test]
    fn extra_guard_words_drive_guard_but_not_ingress() {
        // Regression (review P2): --guard-word extras must NOT trigger ingress
        // refusal, or the documented rollback demo would refuse before decoding.
        let cfg = SafetyConfig {
            mode: SafetyMode::Lightweight,
            banned: vec![],
            patterns: vec![],                     // no built-in unsafe terms
            extra_guard_patterns: vec![vec![42]], // a --guard-word trip token
        };
        let ports = cfg.ports();
        assert!(
            ports.guard.is_some(),
            "extra guard words must drive the output guard"
        );
        assert!(
            ports.ingress.is_none(),
            "extra guard words must NOT drive ingress (trajectory demo, not refusal)"
        );
    }

    #[test]
    fn qwen_expert_missing_file_errors_and_is_permit_gated() {
        // R5: the expert load requires an ADR-006 permit (required arg) and a
        // missing file is rejected, not silently ignored.
        let r = QwenExpert::from_path_primed(
            std::path::Path::new("/nonexistent/expert.gguf"),
            0,
            &[1, 2],
            ok_permit(),
        );
        assert!(matches!(r, Err(EdgeError::Engine(_))));
    }
}
