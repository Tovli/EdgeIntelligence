//! `el-engine-candle` — the real inference engine adapter over **Candle**
//! (ADR-002), implementing [`el_runtime::InferenceEngine`] / `RuntimeAcl`.
//!
//! SKELETON / FOLLOW-UP. The structure and load path are sketched against
//! Candle's API; the prefill/decode bodies are TODOs requiring a quantized
//! model (GGUF/safetensors) and KV-cache wiring. None of Candle's `Tensor`/
//! `Device` types escape this crate — only `el_core` domain types do.

#![forbid(unsafe_code)]

use candle_core::{Device, Tensor};
use el_core::{EdgeError, Result, Token};
use el_runtime::InferenceEngine;

/// Inference engine backed by Candle.
pub struct CandleEngine {
    device: Device,
    eos: Token,
    vocab: usize,
    // TODO(adr-002): hold the loaded model (e.g. quantized_llama::ModelWeights),
    // the tokenizer, and a KV-cache handle bound to the el-memory arena.
    _last_logits: Option<Tensor>,
}

impl CandleEngine {
    /// Load a quantized GGUF/safetensors model.
    ///
    /// TODO(adr-002):
    /// 1. Pick device: `Device::new_metal(0)` on Apple, else `Device::Cpu`
    ///    (Candle NEON). WebGPU via a future `wgpu` backend.
    /// 2. `gguf_file::Content::read` (GGUF) or `candle_core::safetensors::load`.
    /// 3. Build the transformer (e.g. `candle_transformers::models::
    ///    quantized_llama::ModelWeights::from_gguf`).
    /// 4. Map constant weights via the el-memory `WeightMapper` (memmap2).
    pub fn load(_model_path: &str) -> Result<Self> {
        Err(EdgeError::Engine("CandleEngine::load not yet implemented (ADR-002 follow-up)"))
    }
}

impl InferenceEngine for CandleEngine {
    fn prefill(&mut self, _tokens: &[Token]) -> Result<u32> {
        // TODO(adr-002): chunked prefill (Progressive Graph Scheduling), fill KV.
        Err(EdgeError::Engine("prefill not yet implemented"))
    }

    fn next_logits(&mut self, _committed: &[Token]) -> Vec<i32> {
        // TODO(adr-002): forward one step, quantise float logits → milli-logits
        // at this ACL boundary so the orchestrator stays float-free.
        Vec::new()
    }

    fn eos_token(&self) -> Token {
        self.eos
    }
}
