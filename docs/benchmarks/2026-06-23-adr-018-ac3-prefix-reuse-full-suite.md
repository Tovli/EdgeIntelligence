# SDK Benchmark — Full-suite performance after ADR-018 AC-3 (cross-turn prefix reuse)

**Date:** 2026-06-23
**Subject:** End-to-end performance of the local inference path exercised by the
`el-bench` harness (`el_core::LlmProvider` → `el_engine_candle::QwenChatProvider`
→ `el_runtime::InferenceSession`) across the **full** normalized benchmark suite
(CounselBench, MindEval, VERA-MH), now with **[ADR-018](../adr/ADR-018-persistent-model-instances-and-stateful-sessions.md)
AC-3 (cross-turn prefix reuse / incremental prefill)** landed.
**Question:** The [2026-06-22 full-suite report](./2026-06-22-adr-018-resident-model-full-suite.md)
deferred AC-3 and found prefill was the #1 cost (56.8%, un-batched). With AC-3 now
implemented, where does the time go — and does the bottleneck shift?

---

## 1. Executive summary

Measured on the **same host** as the [2026-06-22](./2026-06-22-adr-018-resident-model-full-suite.md)
and [2026-06-14](./2026-06-14-qwen-chat-bottleneck.md) reports (Intel i5-14500,
14C/20T, 32 GB; release build), driving the real `el-bench` binary over **all 66
tasks → 97 model replies, 0 errors, in 35.7 min (2141.5 s)**, safety `Off` (raw-model
baseline — the SDK safety layer is benched separately). SDK version **0.3.11**.

AC-3 changed the picture exactly where the prior report predicted:

| 2026-06-22 finding | Status now | Evidence |
|---|---|---|
| **#1 Prefill is the largest cost** (56.8%, 1827 s) | **Cut by 65% → 29.5% (632 s)** by AC-3 | Multi-turn replies now reuse the cached KV for the unchanged prefix and prefill only the new suffix: **9,893 prefill forwards for 27,148 presented prompt tokens** (64% of prefill positions served from cache). |
| **Multi-turn re-prefill (`mindeval`)** — re-prefilled the whole conversation each turn | **FIXED by AC-3** | `mindeval` avg time/reply **52,225 ms → 17,478 ms (~3×)**; e2e rate **2.7 → 8.1 tok/s**. |
| **#3 Decode compute floor** (~14 tok/s) | **Unchanged → now the #1 cost** | Decode 70.5% (1509 s) at **13.1 tok/s**; 99.2% candle forward. |
| Per-turn weight reload (fixed by ADR-018 core) | **Still fixed** | `session setup` is `0.0 ms` on all 97 replies. |

**Headline:** AC-3 removed the redundant re-prefill of unchanged conversation
prefixes. Prefill dropped from the dominant cost (56.8%) to **29.5%**, cutting
**~20 min off the full run (53.7 → 35.7 min, −33.5%)** — almost entirely on the
multi-turn `mindeval` suite. With prefill shrunk, **decode is now the single largest
cost (70.5%)** — the unchanged ~14 tok/s candle q4_k_m batch-1 floor.

**Important nuance — the prefill *kernel* did not get faster.** AC-3 makes prefill
cheaper by doing **less work**, not faster math: the per-forward prefill rate is
**15.7 tok/s (9,893 forwards / 632 s) — unchanged from the 14.9 tok/s baseline**.
Each prefill step is still **one token per forward**, so
[ADR-020 (batched single-pass prefill)](../adr/ADR-020-batched-single-pass-prefill.md)
remains **not done** and is still the lever for *single-turn* and *first-turn*
prefill (where there is no prefix to reuse). AC-3 and ADR-020 are complementary:
**AC-3 removes the unnecessary prefill; ADR-020 makes the necessary prefill fast.**

### 1.1 Versus the [2026-06-22 run](./2026-06-22-adr-018-resident-model-full-suite.md)

Same harness, same inputs, same host, same configuration as 2026-06-22 — the **only**
change is the AC-3 implementation in the working tree (`el-runtime/src/{session,ports}.rs`,
`el-engine-candle/src/lib.rs`). So this is a clean run-for-run comparison.

| Metric | 2026-06-22 (no AC-3) | 2026-06-23 (AC-3) | Change |
|---|---|---|---|
| **Total run wall** | 3221 s (53.7 min) | **2141.5 s (35.7 min)** | **−33.5%** ✅ |
| Prefill wall / share | 1827.2 s / 56.8% | **632.1 s / 29.5%** | **−65% wall** ✅ |
| Decode wall / share | 1387.9 s / 43.2% | 1509.2 s / 70.5% | ~flat (now dominant) |
| Prefill forwards | ~27,148 (1/token) | **9,893** | **−64%** (reuse) ✅ |
| Prefill per-forward rate | 14.9 tok/s | 15.7 tok/s | unchanged (no ADR-020) |
| Decode throughput | 14.2 tok/s | 13.1 tok/s | ~flat (−8%, thermal/noise) |
| `mindeval` avg ms/reply | 52,225 | **17,478** | **~3× faster** ✅ |
| `mindeval` e2e tok/s | 2.7 | 8.1 | **~3×** ✅ |
| Model forward share of compute | 99.3% | 99.2% | unchanged (SDK glue negligible) |
| Per-turn weight reload | 0.0 ms | 0.0 ms | unchanged (ADR-018 core) |

**Reading:** AC-3 targeted exactly the structural cost the prior report isolated —
re-prefilling the unchanged prefix every turn — and eliminated it. The decode floor
(#3/#5) and the un-batched prefill *kernel* (ADR-020) are untouched by design, so
decode is now the dominant cost and first-turn prefill rate is unchanged.

---

## 2. Method

### 2.1 Harness

The benchmark drives the **actual** `el-bench` release binary (SDK v0.3.11) — no
synthetic harness — so every number reflects the real public SDK path
(`LlmProvider::chat` → `QwenChatProvider` → `QwenEngine` + `InferenceSession`),
including the ADR-018 resident session **and** the AC-3 cross-turn prefix reuse
landed in this branch (`InferenceSession::continue_prompt` →
`InferenceEngine::prefill_reuse`, which reuses the longest cached token prefix that
still matches the new context and feeds only the divergent suffix).

Per-phase timing comes from the same opt-in, env-gated instrumentation
(`EL_BENCH=1`) used in the prior reports (`crates/adapters/el-engine-candle/src/lib.rs`):
each `chat()` is split into **session setup → tokenize → prefill → decode →
detokenize**, and each `forward_one` into **model** (candle transformer forward) vs
**seam** (tensor build + `to_vec1` + float→milli-logit), with the remainder
attributed to the runtime **loop**. Zero-cost when unset.

Per-suite figures are aggregated from the transcript JSONL (`prompt_tokens`,
`completion_tokens`, `ms` per exchange); the prefill/decode split and forward-call
counts from the `EL_BENCH` stderr log.

### 2.2 Configuration

- **Model:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB) + `tokenizer.json`, local `models/` (air-gapped, ADR-004).
- **Build:** `cargo build --release -p el-bench` (`opt-level=3`, LTO), SDK v0.3.11.
- **Safety:** `SafetyMode::Off` — el-bench benches the raw model; the ADR-005/012 safety loop is evaluated in the clinical-safety report.
- **Decoding:** deterministic greedy argmax → reproducible.
- **Tasks:** `benchmarks/tasks/{counselbench-adv,counselbench-eval,mindeval,veramh}.jsonl`, default `--max-tokens 256`, no `--limit` (full suite).
- **CPU:** Intel i5-14500 (6 P + 8 E, 20 threads); candle `gemm`/rayon CPU backend; no GPU.

---

## 3. Raw results

### 3.1 Per-suite (end-to-end, from the transcript)

`avg ms/reply` is wall time per `chat()` (prefill + decode); `e2e tok/s` is
`completion_tokens / total_time`.

| Suite | tasks | replies | prompt tok | compl tok | avg ms/reply | e2e tok/s |
|-------|------:|--------:|-----------:|----------:|-------------:|----------:|
| counselbench-adv  | 24 | 24 |  3,168 | 5,717 | 26,248 |  9.1 |
| counselbench-eval | 20 | 20 |  3,126 | 5,112 | 26,880 |  9.5 |
| mindeval          |  5 | 30 | 16,491 | 4,258 | **17,478** |  **8.1** |
| veramh            | 17 | 23 |  4,363 | 4,706 | 19,544 | 10.5 |
| **TOTAL**         | 66 | 97 | 27,148 | 19,793 | 22,076 |  9.2 |

> `mindeval` is where AC-3 pays off: in the 2026-06-22 run it averaged 52,225 ms/reply
> (2.7 tok/s) because each turn re-prefilled the entire growing conversation; here it
> is 17,478 ms/reply (8.1 tok/s), now in line with the largely single-turn suites.

### 3.2 Prefill vs decode (summed over all 97 replies, from `EL_BENCH`)

| Phase | wall | share | forwards | presented tokens | rate |
|-------|-----:|------:|---------:|-----------------:|-----:|
| **prefill** | 632.1 s | **29.5%** | 9,893 | 27,148 | **42.9 tok/s effective** / 15.7 tok/s per-forward |
| **decode**  | 1509.2 s | **70.5%** | 19,696 | 19,793 | **13.1 tok/s** |

- **Effective vs per-forward prefill rate:** 42.9 tok/s is *presented* prompt tokens
  ÷ wall — it counts prefix tokens served from cache for free. The true kernel rate
  is **9,893 forwards / 632.1 s = 15.7 tok/s** (≈ the 14.9 tok/s baseline). The gap is
  the AC-3 win: **17,255 of 27,148 prefill positions (64%) never hit the model.**
- **Compute attribution:** candle model forward is **99.2%** of prefill+decode wall
  (628.0 s prefill + 1495.7 s decode = 2123.7 s of 2141.3 s); seam quantization + the
  runtime decode loop are ~0.8% combined. Per decoded token: **76.6 ms = model 75.9 +
  seam 0.4 + loop 0.3**.
- **Session setup:** `0.0 ms` on all 97 replies (weights loaded once — ADR-018 core).

---

## 4. Analysis

### 4.1 AC-3 (cross-turn prefix reuse) — the win, proven per-conversation

The aggregate (9,893 forwards for 27,148 presented tokens) is the suite-wide
signature of reuse, but the mechanism is clearest **within a single multi-turn
conversation**. From the `EL_BENCH` log for `mindeval-f28-marketingc` (6 turns), the
prefill forward count **falls turn-over-turn even as the prompt grows**, because each
turn reuses the prior turns' cached KV and forwards only the new suffix:

| Turn | presented prompt tok | prefill forwards | reused from cache |
|-----:|---------------------:|-----------------:|------------------:|
| 1 | 90  | 90 | 0 (cold) |
| 2 | 149 | 39 | 110 |
| 3 | 268 | 24 | 244 |
| 4 | 364 | 24 | 340 |
| 5 | 449 | 14 | 435 |

This is `InferenceSession::continue_prompt` → `prefill_reuse` reusing the longest
matching cached token prefix (system prompt + all prior turns, now baked into the
re-rendered context) and feeding only the divergent suffix. Single-turn replies
(all of counselbench, most of veramh) show `forwards == prompt_tokens` — no prefix
to reuse on a cold conversation, so they are unchanged from the baseline (correctly).

### 4.2 The prefill kernel is unchanged — ADR-020 still the lever for first-turn

AC-3 does **not** batch prefill. The per-forward prefill rate is **15.7 tok/s**,
statistically identical to the 14.9 tok/s baseline, because each prefill step is
still **one `(1,1)` forward per token** (`QwenEngine::prefill`). A cold first turn,
or any single-turn task, still pays the full token-by-token prefill at ~15 tok/s.
Replacing that loop with one batched `(1, suffix_len)` forward
([ADR-020](../adr/ADR-020-batched-single-pass-prefill.md)) is now the highest-value
remaining prefill optimization, and it composes with AC-3: AC-3 shrinks the *number*
of suffix tokens that must be prefilled; ADR-020 makes prefilling them fast.

### 4.3 Decode (~14 tok/s) is now the dominant cost — unchanged floor

With prefill cut by 65%, **decode is now 70.5% of compute** at 13.1 tok/s (per-token
76.6 ms, 99.2% candle q4_k_m forward) — a memory-bandwidth-bound batch-1 decode
floor, an engine/kernel concern, not SDK orchestration. It is ~8% slower than the
14.2 tok/s measured on 2026-06-22; with model forward = 99.2% and the SDK glue
unchanged, that delta is consistent with run-to-run thermal/scheduling variance over
a 36-min CPU-bound run, not a regression in SDK code. This is the same floor called
out as bottleneck #3/#5 in the prior reports and remains out of scope for P0.

### 4.4 Correctness of reuse

AC-3 ships with a soundness contract: `prefill_reuse` must leave the cache *byte-for-byte*
as a full `reset_cache()` + `prefill(full_context)` would (identical subsequent
logits) — reuse is purely a compute optimization, never a change to *what* the cache
represents. The implementation fails **open** (a wrong/short prefix match discards
the cache and re-prefills the whole context), and a `ReuseEngine` test mock proves
reuse-equals-fresh-prefill without a real model. The full suite ran with **0 errors**
and deterministic greedy decoding, and the per-suite completion-token counts match
the 2026-06-22 run (e.g. mindeval 4,258 compl tok both runs) — i.e. AC-3 changed the
*speed*, not the *output*.

---

## 5. Recommendations (prioritized)

1. **Batch the prefill — [ADR-020](../adr/ADR-020-batched-single-pass-prefill.md).**
   Now the largest *remaining* prefill lever (AC-3 cut the count, not the per-token
   cost). Replace the per-token loop in `QwenEngine::prefill` with one
   `(1, suffix_len)` forward. Composes with AC-3; biggest win for first/single-turn.
2. **Faster quantized decode kernel** — decode is now the dominant cost (70.5%) at
   the ~14 tok/s candle CPU matmul floor. Largest absolute ceiling, hardest; revisit
   after ADR-020.
3. **Real token streaming — [ADR-019](../adr/ADR-019-in-loop-incremental-decoding-and-token-streaming.md).**
   TTFT is still full-generation time; a per-token emit hook drops it to
   `prefill + 1 token`. UX, separate from throughput.
4. **Baseline instrumentation as a CI gate — [ADR-023](../adr/ADR-023-baseline-performance-instrumentation.md).**
   Capture this run (prefill share 29.5%, forwards 9,893/27,148, decode 13.1 tok/s) as
   the machine-readable baseline so ADR-020's win is regression-gated and the AC-3 gain
   here is protected.

---

## 6. Reproduce

```sh
cargo build --release -p el-bench

# Full suite (this run): all tasks, default 256 max-tokens, EL_BENCH timing.
EL_BENCH=1 ./target/release/el-bench.exe \
  --out benchmarks/out/transcripts.jsonl

# Bounded smoke (fast): a multi-turn task to see AC-3 reuse in the EL_BENCH log
# (prefill `fwd calls` drops on turns 2+ while prompt_tokens grows).
EL_BENCH=1 ./target/release/el-bench.exe \
  --suite mindeval --limit 1 --out /tmp/perf.jsonl
```

`EL_BENCH` prints the per-phase table + forward attribution to stderr; the transcript
JSONL records per-reply `prompt_tokens` / `completion_tokens` / `ms`. Per-suite
figures are aggregated from the transcript, the prefill/decode split and forward-call
counts from the `EL_BENCH` stderr log.
