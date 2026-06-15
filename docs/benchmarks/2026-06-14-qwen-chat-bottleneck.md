# SDK Benchmark — Bottleneck Report (el-chat / Qwen2.5-0.5B)

**Date:** 2026-06-14
**Subject:** End-to-end latency of the local inference path exercised by the
`el-chat` test client (`el_core::LlmProvider` → `el_engine_candle::QwenChatProvider`
→ `el_runtime::InferenceSession`).
**Question:** Where does the time go, and what is the dominant bottleneck?

---

## 1. Executive summary

Measured on an Intel i5-14500 (14C/20T, 32 GB) with the release build
(`opt-level=3`, LTO, `codegen-units=1`), the **`el-chat` reply latency is
dominated by three structural costs, in priority order**:

| # | Bottleneck | Cost (measured) | Share | Lever |
|---|------------|-----------------|-------|-------|
| **1** | **Prefill is not batched** — the prompt is fed one token at a time | ~65 ms **per prompt token** (209 tok → **14.3 s**) | up to **90%** on long prompts / later turns | SDK fix, ~10× win |
| **2** | **Full model reload every turn** — the 491 MB GGUF is re-read + re-parsed on *every* `chat()` call | **~1.2 s warm** (1.5 s cold) **per turn** | 20–33% on short turns | SDK fix, removes a flat tax |
| **3** | **Decode compute floor** — candle quantized Qwen2 forward | ~65 ms/token → **~15 tok/s** | 60–75% on long replies | kernel-level, hard |

**What is *not* a bottleneck:** the SDK's own per-token glue — the runtime
decode loop, the `vec![true; vocab]` grammar mask, the full-vocab logits clone,
the argmax, event emission, and the float→milli-logit quantization at the engine
seam — together account for **< 1.2% of decode time (~0.8 ms of ~65 ms per
token)**. Optimizing these would not move the needle.

The single highest-impact change is **#1 (batch the prefill)**: it is a pure
SDK-side defect (the engine issues `prompt_len` separate forwards instead of
one), it scales the worst (linear in prompt length, and prompt length grows
every conversation turn), and candle already supports the batched call.

A closely related compounding effect: because the engine is rebuilt every turn
(#2), each turn **re-prefills the entire growing conversation from scratch** —
turn 2 in the measured run re-processed 56 tokens even though 32 of them were
already prefilled in turn 1. Fixing #2 enables KV reuse, which removes that
redundant re-prefill entirely.

---

## 2. Method

### 2.1 Harness

The benchmark drives the **actual** `el-chat` test client binary — no synthetic
harness — so every number reflects the real public SDK path
(`LlmProvider::chat_stream` → `chat` → `QwenEngine` + `InferenceSession`).

Phase timing was obtained by adding **opt-in, env-gated instrumentation**
(`EL_BENCH=1`) inside the engine adapter (`crates/adapters/el-engine-candle/src/lib.rs`):

- `QwenChatProvider::chat` is timed per phase: **model load → tokenize →
  prefill → decode → detokenize**.
- `QwenEngine::forward_one` is timed to split each forward into **model**
  (the candle transformer `forward`) vs **seam** (tensor build + `to_vec1` +
  float→milli-logit conversion). The remainder of each phase's wall time
  (`wall − Σ forward`) is the **loop** overhead (runtime decode loop:
  logits clone, mask alloc, argmax, KV push, event emit).

The instrumentation is **zero-cost when `EL_BENCH` is unset** (a single
`OnceLock<bool>` short-circuits all timers) and is behavior-preserving — the
crate's 14 tests pass unchanged. It can be removed or kept as a diagnostic.

### 2.2 Configuration

- **Model:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB) + `tokenizer.json` (7 MB), loaded from local `models/` (air-gapped, ADR-004).
- **Build:** `cargo build --release -p el-chat`. (Release matters: the runtime/engine SDK crates are `opt-level=0` in dev, which would overstate the SDK glue cost. Release optimizes everything — the fair, production-representative measurement.)
- **Decoding:** deterministic greedy argmax (the SDK local path does not sample), so runs are reproducible.
- **CPU:** Intel i5-14500, 6 P-cores + 8 E-cores (20 threads). candle uses the `gemm`/rayon CPU backend; no MKL/Accelerate; no GPU.

### 2.3 Runs

| Run | Prompt | max-tokens | Purpose |
|-----|--------|-----------:|---------|
| A | "Hello! Who are you?" | 16 | baseline, cold load |
| B | TCP question | 128 | decode throughput |
| C | TCP question | 64 | decode cross-check |
| D | ~190-token system prompt | 4 | **prefill scaling** |
| F | 2-turn REPL | 24 | **multi-turn compounding** |
| — | "Count slowly." | 24 | thread sensitivity (1 / 6 / 20) |

---

## 3. Raw results

Phase wall times in ms; `tok/s` is that phase's throughput.

| Run | prompt tok | compl. tok | model load | tokenize | **prefill** | **decode** | detok | TOTAL | prefill tok/s | decode tok/s |
|-----|-----------:|-----------:|-----------:|---------:|------------:|-----------:|------:|------:|------:|------:|
| A | 31 | 16 | 1504.1 | 1.8 | 2071.1 | 985.2 | 0.1 | 4562.3 | 15.0 | 16.2 |
| B | 29 | 128 | 1191.9 | 1.3 | 1771.2 | 8788.4 | 0.1 | 11752.9 | 16.4 | 14.6 |
| C | 29 | 64 | 1186.2 | 1.3 | 1808.3 | 4133.6 | 0.1 | 7129.5 | 16.0 | 15.5 |
| D | 209 | 3 | 1325.2 | 2.0 | **14299.2** | 203.4 | 0.0 | 15829.8 | 14.6 | 14.8 |
| F·t1 | 32 | 8 | 1174.6 | 1.3 | 1989.8 | 460.2 | 0.0 | 3626.0 | 16.1 | 17.4 |
| F·t2 | 56 | 15 | 1239.0 | 0.2 | 3749.0 | 1130.9 | 0.0 | 6119.1 | 14.9 | 13.3 |

**Per-forward attribution** (representative — Run B decode, 127 forwards):

```
decode : 127 fwd calls, model 8694.4ms, seam 54.1ms, loop 39.9ms
per decoded token: 69.20ms total = model 68.46 + seam 0.43 + loop 0.31
```

→ **model 98.9%, seam 0.6%, loop 0.5%.** The same split holds in every run.

**Thread sensitivity** (per-token model time, "Count slowly.", 24 tokens):

| Threads | ms/token | speedup vs 1 |
|--------:|---------:|-------------:|
| 1 | 311.7 | 1.0× |
| 6 (P-cores) | 77.8 | 4.0× |
| 20 (default) | 66.4 | 4.7× |

---

## 4. Analysis

### 4.1 Bottleneck #1 — Prefill is not batched *(highest impact)*

Prefill throughput (14.6–16.4 tok/s) is **identical to decode throughput**
(13.3–17.4 tok/s), and per-forward time is ~65 ms whether the call happens
during prefill or decode. That is the signature of **no prefill batching**: the
prompt is processed as `prompt_len` independent single-token forwards rather
than one batched forward over the whole prompt.

Run D makes it undeniable: **209 prompt tokens cost 14.3 s of prefill — 90% of
the entire request** — for a 3-token reply.

Root cause — `crates/adapters/el-engine-candle/src/lib.rs`:

```rust
// QwenEngine::prefill  (lib.rs:419)
for &t in tokens {
    self.last_logits = self.forward_one(t)?;   // one forward PER prompt token
}
```

```rust
// QwenEngine::forward_one  (lib.rs:389)
let input = Tensor::from_vec(vec![token], (1, 1), &self.device)  // shape (1,1) — single token
```

candle's `quantized_qwen2::ModelWeights::forward(input, index_pos)` accepts a
`(batch, seq_len)` input — a real batched prefill is one call with the whole
prompt as `(1, prompt_len)`. A batched prefill reads each weight tensor **once**
for the whole prompt (compute-bound, parallelizes well) instead of once *per
token* (memory-bandwidth-bound, ×`prompt_len`). Expected effect: prefill drops
from `O(prompt_len × 65 ms)` to roughly a single forward's compute over
`prompt_len` positions — an **order-of-magnitude reduction** on non-trivial
prompts.

This is the worst-scaling cost because prompt length grows every turn (§4.4).

### 4.2 Bottleneck #2 — Full model reload every turn

"loading … ready (0.2 s)" is **misleading**: `QwenChatProvider::from_paths` only
loads the *tokenizer*. The 491 MB GGUF weights are (re)loaded **inside every
`chat()` call**:

```rust
// QwenChatProvider::chat  (lib.rs:522)
let engine = QwenEngine::from_path(&self.model_path, self.eos)?;  // re-reads + re-parses 491 MB, every turn
```

Measured cost: **~1.2 s warm (OS page cache hot), ~1.5 s cold** — paid on *every*
reply. On a short turn that is 20–33% of total latency (Runs A, F·t1); across a
REPL session it is a flat per-turn tax (Run F: 1.17 s on turn 1, 1.24 s on turn 2).

The code does this deliberately because **candle's quantized model exposes no
public KV-cache reset**, and the weights and the KV cache live in the same
`ModelWeights` object — so to get a clean cache the whole engine (weights
included) is rebuilt. The fix is to **separate the immutable weights (load once,
keep/`mmap`) from the per-conversation KV state** (the only thing that must
reset). That removes the reload tax and is the prerequisite for KV reuse (§4.4).

### 4.3 Bottleneck #3 — Decode compute floor (~15 tok/s)

Decode is **98.9% candle transformer forward** (Run B: model 68.46 ms of
69.20 ms/token). At ~15 tok/s for a 0.5B Q4 model on a 14-core CPU, this is
several times slower than hand-tuned stacks (llama.cpp-class kernels reach
50–100+ tok/s on comparable hardware).

Thread sensitivity explains why this is a *floor*, not a tuning miss: 1→6
threads gives 4.0×, 6→20 only adds 17% more. Throughput saturates at ~6 cores —
the classic profile of **memory-bandwidth-bound batch-1 decode** (each token
must stream the full quantized weight set from RAM; 15 tok/s ≈ ~5 GB/s effective,
far below the platform's DRAM bandwidth, indicating the q4_k_m matmul kernel
under-utilizes SIMD/cache rather than saturating the bus). Closing this gap
requires a faster quantized CPU kernel — an engine/kernel change, not an SDK
orchestration change. **Thread tuning is not a lever** (default 20 threads is
already within 17% of the best observed; ~6–8 P-cores is near-optimal).

### 4.4 Multi-turn compounding (interaction of #1 + #2)

Run F shows the two structural costs compounding across a conversation:

- **Turn 1:** 32 prompt tokens → reload 1.17 s + prefill 1.99 s.
- **Turn 2:** 56 prompt tokens → reload 1.24 s **again** + prefill **3.75 s**.

Turn 2 re-prefills the *entire* history (system + user₁ + assistant₁ + user₂),
including the 32 tokens already prefilled in turn 1 — pure redundant work caused
by rebuilding the engine each turn (no KV carried over). As history grows, every
turn re-prefills everything from scratch, so per-turn latency grows roughly
linearly across the session. (The `el-chat` README already warns "per-turn cost
grows"; this quantifies *why* and shows it is fixable, not inherent.)

### 4.5 What is NOT a bottleneck (red herrings)

The per-token forward attribution shows the SDK's own work is negligible
(~0.8 ms of ~65 ms, < 1.2%). The following are *real allocations* but
**immaterial to latency** and should not be prioritized for performance
(only for tidiness/correctness):

- `el-runtime/src/session.rs:139` — `next_logits` returns a fresh ~151 K-wide
  `Vec<i32>` (~600 KB) every step; `el-engine-candle` `next_logits`
  additionally `.clone()`s it (`lib.rs:438`).
- `el-runtime/src/session.rs:143` + `el-runtime/src/defaults.rs:21` —
  `AllowAllMasker` allocates `vec![true; vocab]` (~152 KB) and `pick` scans the
  full vocab each step, despite no grammar constraint being active on the chat
  path.
- The engine seam re-allocates two vocab-sized vectors per forward (`to_vec1`
  then the milli-logit `Vec<i32>`).
- `chat_stream` is **not** real streaming: it runs `chat()` to completion, then
  replays the finished string char-by-char (`lib.rs:575`). This does not affect
  throughput, but it means perceived **time-to-first-token = full generation
  time**. With a per-token callback wired into `InferenceSession::generate`,
  TTFT would drop to `load + prefill + 1 token` instead of waiting for the whole
  reply. Worth fixing for UX, separately from the three throughput bottlenecks.

---

## 5. Recommendations (prioritized)

1. **Batch the prefill** *(biggest win, pure SDK fix).* Replace the per-token
   loop in `QwenEngine::prefill` with a single `forward` over the whole prompt
   `(1, prompt_len)`; track `index_pos` accordingly. Expected: long-prompt /
   later-turn latency down ~10×; Run D's 14.3 s prefill → low single-digit
   seconds. *(lib.rs:419, 389)*

2. **Load weights once; reset only the KV cache** *(removes the flat ~1.2 s/turn
   tax).* Keep the parsed `Qwen2Weights` (or at least `mmap` the GGUF) in
   `QwenChatProvider` and add a KV-cache reset path rather than calling
   `QwenEngine::from_path` per turn. Unblocks #3. *(lib.rs:522)*

3. **Reuse KV across turns** *(removes redundant re-prefill; depends on #1+#2).*
   Once weights persist and the cache survives, prefill only the **new** tokens
   each turn instead of the whole conversation. Turns become near-constant cost
   instead of linearly growing.

4. **Real token streaming** *(UX / TTFT).* Add a per-token callback to
   `InferenceSession::generate` and have `chat_stream` emit from inside the
   decode loop instead of replaying a completed string. *(session.rs:124,
   lib.rs:575)*

5. **Faster quantized decode kernel** *(largest absolute ceiling, hardest).* The
   ~15 tok/s decode floor is candle's q4_k_m CPU matmul. Revisit only after
   1–4; consider a tuned kernel or delegate. Thread tuning is **not** needed
   (default is near-optimal).

6. **(Low priority, non-perf) Trim per-step allocations** — reuse logit/mask
   buffers and skip mask allocation when grammar is permissive. Cleanliness, not
   speed (< 1.2% of decode).

### Rough projected impact

For a typical second turn (~56 prompt tok, ~30 reply tok), today ≈
`1.2 (reload) + 3.7 (prefill) + 2.0 (decode) ≈ 6.9 s`. With #1+#2+#3:
`~0 (reload) + ~0.3 (prefill new tokens, batched) + 2.0 (decode) ≈ 2.3 s` —
roughly **3× faster**, and the saving grows with conversation length and prompt
size. Decode (#5) remains the residual floor.

---

## 6. Reproduce

```sh
cargo build --release -p el-chat

# Per-phase breakdown for any invocation:
EL_BENCH=1 ./target/release/el-chat.exe --prompt "Hello!" --once --max-tokens 64

# Prefill scaling (long prompt, tiny reply):
EL_BENCH=1 ./target/release/el-chat.exe --system "$(cat long_prompt.txt)" \
  --prompt "Say OK." --once --max-tokens 4

# Multi-turn compounding:
printf 'What is 2+2?\nMultiply that by 3.\n/exit\n' | \
  EL_BENCH=1 ./target/release/el-chat.exe --max-tokens 24
```

`EL_BENCH` prints the phase table + forward attribution to stderr; unset, it is
inert (no timing taken).
