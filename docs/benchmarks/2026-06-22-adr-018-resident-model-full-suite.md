# SDK Benchmark — Full-suite performance after ADR-018 (el-bench / Qwen2.5-0.5B)

**Date:** 2026-06-22
**Subject:** End-to-end performance of the local inference path exercised by the
`el-bench` harness (`el_core::LlmProvider` → `el_engine_candle::QwenChatProvider`
→ `el_runtime::InferenceSession`), across the **full** normalized benchmark suite
(CounselBench, MindEval, VERA-MH).
**Question:** After [ADR-018](../adr/ADR-018-persistent-model-instances-and-stateful-sessions.md)
(persistent model + stateful sessions) landed, where does the time go now — and
which of the [2026-06-14 bottlenecks](./2026-06-14-qwen-chat-bottleneck.md) remain?

---

## 1. Executive summary

Measured on the **same host** as the 2026-06-14 bottleneck report (Intel i5-14500,
14C/20T, 32 GB; release build), driving the real `el-bench` binary over **all 66
tasks → 97 model replies, 0 errors, in 53.7 min (3221 s)**, safety `Off` (raw-model
baseline — the SDK safety layer is benched separately).

The picture has shifted exactly as the prior report predicted:

| 2026-06-14 bottleneck | Status now | Evidence |
|---|---|---|
| **#2 Full model reload every turn** (~1.2 s/turn) | **FIXED by ADR-018** | `session setup` is `0.0 ms` on all 97 replies — the 491 MB GGUF loads **once** at startup and the resident session is reused; no per-turn reload. |
| **#1 Prefill is not batched** | **Now the #1 cost, confirmed at scale** | Prefill is **56.8%** of all compute (1827 s), running token-by-token at **14.9 tok/s** — *the same rate as decode*. |
| **#3 Decode compute floor** (~15 tok/s) | **Unchanged** (kernel-level) | Decode 43.2% (1388 s) at **14.2 tok/s**; 99.3% candle forward. |

**Headline:** ADR-018 removed the per-turn weight-reload tax (bottleneck #2). With
that gone, **prefill — still un-batched — is now the single largest cost (56.8% of
compute)**, making [ADR-020 (batched prefill)](../adr/ADR-020-batched-single-pass-prefill.md)
the highest-value next step. The SDK's own per-token glue remains negligible
(< 1% of compute; 99.3% is raw model forward).

The `mindeval` suite (multi-turn) isolates the second remaining structural cost —
**re-prefilling the whole conversation each turn**: 5 tasks generated 30 replies
that prefilled **16,491 prompt tokens to produce only 4,258 completion tokens
(~4× more prefill than generation)**, dragging its end-to-end rate to 2.7 tok/s.
That is the case for ADR-018 **AC-3** (cross-turn prefix reuse / incremental
prefill), deferred from this increment.

### 1.1 Versus the [baseline](./2026-06-14-qwen-chat-bottleneck.md)

The two runs use **different harnesses and inputs**: the baseline is `el-chat`
micro-runs (hand-picked prompts, 4–128 max-tokens) chosen to *isolate* each
bottleneck; this is the full `el-bench` suite (66 tasks, 256 max-tokens). So **raw
totals are not comparable run-for-run** — but the prompt-independent quantities
(per-phase rates, per-token cost, SDK overhead, bottleneck status) are, and that is
the fair comparison. Same host both times (i5-14500, release).

| Metric (comparable across harnesses) | Baseline (2026-06-14) | After ADR-018 (2026-06-22) | Change |
|---|---|---|---|
| **Per-turn model reload** | ~1.2 s warm / 1.5 s cold, **every turn** | **0.0 ms** (loaded once) | **eliminated** ✅ |
| Prefill throughput | ~15 tok/s (14.6–16.4) | 14.9 tok/s | unchanged — ADR-020 not yet done |
| Decode throughput | ~15 tok/s (13.3–17.4) | 14.2 tok/s | unchanged — kernel floor (#3) |
| Per-token decode (model) | 68.5 ms | ~65–81 ms | unchanged |
| SDK glue overhead | < 1.2% of decode | ~0.7% of compute | unchanged (negligible) |
| Dominant structural cost | #1 prefill **+** #2 reload | #1 prefill **alone** (56.8%) | #2 removed |

**Reading:** ADR-018 targeted exactly one of the three baseline bottlenecks — **#2,
the per-turn weight reload** — and eliminated it (a flat ~1.2 s/turn tax → 0). By
design it does **not** touch prefill batching or the decode kernel, so the
per-phase *rates* are unchanged; what changed is that the flat reload tax is gone
and prefill (#1) now stands alone as the dominant cost. The two remaining baseline
bottlenecks are unchanged: prefill batching ([ADR-020](../adr/ADR-020-batched-single-pass-prefill.md))
and the decode floor (#5). The baseline's projected "≈3× faster second turn" assumed
#1+#2+#3 together; this increment banked the #2 portion, and the §3 data shows #1 is
now the larger remaining prize.

---

## 2. Method

### 2.1 Harness

The benchmark drives the **actual** `el-bench` release binary — no synthetic
harness — so every number reflects the real public SDK path
(`LlmProvider::chat` → `QwenChatProvider` → `QwenEngine` + `InferenceSession`),
including the ADR-018 resident-session reuse landed in this branch.

Per-phase timing comes from the same opt-in, env-gated instrumentation
(`EL_BENCH=1`) used in the 2026-06-14 report (`crates/adapters/el-engine-candle/src/lib.rs`):
each `chat()` is split into **session setup → tokenize → prefill → decode →
detokenize**, and each `forward_one` into **model** (candle transformer forward)
vs **seam** (tensor build + `to_vec1` + float→milli-logit), with the remainder
attributed to the runtime **loop**. Zero-cost when unset.

Per-reply latency and token counts are also recorded in the transcript JSONL
(`prompt_tokens`, `completion_tokens`, `ms` per exchange); the per-suite table
below is computed from those, the prefill/decode split from the `EL_BENCH` log.

### 2.2 Configuration

- **Model:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB) + `tokenizer.json`, local `models/` (air-gapped, ADR-004).
- **Build:** `cargo build --release -p el-bench` (`opt-level=3`, LTO).
- **Safety:** `SafetyMode::Off` — el-bench benches the raw model; the ADR-005/012 safety loop is evaluated in the clinical-safety report.
- **Decoding:** deterministic greedy argmax → reproducible.
- **Tasks:** `benchmarks/tasks/{counselbench-adv,counselbench-eval,mindeval,veramh}.jsonl`, default `--max-tokens 256`, no `--limit` (full suite).
- **CPU:** Intel i5-14500 (6 P + 8 E, 20 threads); candle `gemm`/rayon CPU backend; no GPU.

---

## 3. Raw results

### 3.1 Per-suite (end-to-end, from the transcript)

`avg ms/reply` is wall time per `chat()` (prefill + decode); `e2e tok/s` is
`completion_tokens / total_time` — an output rate that *includes* prefill, so it
falls as prefill grows.

| Suite | tasks | replies | prompt tok | compl tok | avg ms/reply | e2e tok/s |
|-------|------:|--------:|-----------:|----------:|-------------:|----------:|
| counselbench-adv  | 24 | 24 |  3,168 | 5,717 | 23,935 | 10.0 |
| counselbench-eval | 20 | 20 |  3,126 | 5,112 | 25,841 |  9.9 |
| mindeval          |  5 | 30 | 16,491 | 4,258 | 52,225 |  2.7 |
| veramh            | 17 | 23 |  4,363 | 4,706 | 24,481 |  8.4 |
| **TOTAL**         | 66 | 97 | 27,148 | 19,793 | 33,207 |  6.1 |

### 3.2 Prefill vs decode (summed over all 97 replies, from `EL_BENCH`)

| Phase | wall | share | tokens | throughput |
|-------|-----:|------:|-------:|-----------:|
| **prefill** | 1827.2 s | **56.8%** | 27,148 | **14.9 tok/s** |
| **decode**  | 1387.9 s | 43.2% | 19,696 | **14.2 tok/s** |

**Compute attribution:** the candle model forward is **99.3%** of prefill+decode
wall; seam quantization + the runtime decode loop are ~0.7% combined.
Representative per-token decode: ~65–82 ms = model ~65–81 + seam ~0.4 + loop ~0.3.

---

## 4. Analysis

### 4.1 Bottleneck #2 (model reload) is gone — ADR-018

The 2026-06-14 report measured a **~1.2 s warm / ~1.5 s cold weight reload on every
`chat()`**, because the engine was rebuilt per turn (candle couples weights + KV
cache in one `ModelWeights`, and there was no public cache reset). ADR-018 made
`QwenChatProvider` load the `QwenEngine` once into a resident `Mutex<ChatSession>`
and reuse one session per turn, evicting only the conversation's KV via a
position-0 forward (`reset_cache`). The effect is visible on **every** reply:

```
│ session setup       0.0 ms   (weights loaded once at startup — ADR-018)
```

Across 97 replies that removes ~97 × ~1.2 s ≈ **~2 min** of pure reload tax plus 97
redundant 491 MB reads/dequantizations — and, more importantly, removes the flat
per-turn tax that dominated short turns in the prior report.

### 4.2 Bottleneck #1 (prefill not batched) is now the largest cost

With the reload gone, prefill is the biggest line item: **56.8% of all compute**,
at **14.9 tok/s — identical to decode's 14.2 tok/s**. That equality is the
signature of no prefill batching: the prompt is fed as `prompt_len` independent
single-token forwards instead of one batched `(1, prompt_len)` forward (root cause
unchanged from the prior report, `QwenEngine::prefill`). candle already accepts the
batched shape, so this is a pure SDK fix —
[ADR-020](../adr/ADR-020-batched-single-pass-prefill.md) — with an expected
order-of-magnitude prefill win. **It is now the highest-value optimization in the
codebase.**

### 4.3 Multi-turn re-prefill — the `mindeval` signal (ADR-018 AC-3)

`mindeval`'s 5 multi-turn tasks produced 30 replies that prefilled **16,491 prompt
tokens for only 4,258 generated** — because each turn re-prefills the entire
growing conversation from scratch (the system prompt + every prior user/assistant
turn). Its end-to-end rate (2.7 tok/s) is ~4× worse than the largely single-turn
suites (8–10 tok/s). ADR-018 deferred **AC-3** (reuse the unchanged-prefix KV and
prefill only the new suffix); this is the data motivating it. ADR-020 (batch) and
AC-3 (don't re-prefill) are complementary: batch makes the necessary prefill fast;
AC-3 removes the unnecessary prefill.

### 4.4 Bottleneck #3 (decode floor ~14 tok/s) — unchanged

Decode is still **99.3% candle q4_k_m forward** at ~14 tok/s — a
memory-bandwidth-bound batch-1 decode floor, an engine/kernel concern, not SDK
orchestration. Unchanged by ADR-018 and out of scope for the P0 milestone.

---

## 5. Recommendations (prioritized)

1. **Batch the prefill — [ADR-020](../adr/ADR-020-batched-single-pass-prefill.md).**
   Now the single largest cost (56.8%). Replace the per-token loop in
   `QwenEngine::prefill` with one `(1, prompt_len)` forward. Biggest win, pure SDK.
2. **Cross-turn prefix reuse — ADR-018 AC-3.** Prefill only new suffix tokens;
   removes the redundant re-prefill that makes `mindeval`-style multi-turn ~4×
   slower. Now unblocked by the resident session this PR added.
3. **Real token streaming — [ADR-019](../adr/ADR-019-in-loop-incremental-decoding-and-token-streaming.md).** TTFT is still full-generation time (`chat_stream` replays a finished reply); a per-token emit hook drops it to `prefill + 1 token`. UX, separate from throughput.
4. **Baseline instrumentation as a CI gate — [ADR-023](../adr/ADR-023-baseline-performance-instrumentation.md).** Emit measured `prefill_tps`/`MetricsSampled` and capture this run as the machine-readable baseline so ADR-020's win is regression-gated.
5. **Faster quantized decode kernel** *(largest absolute ceiling, hardest)* — the ~14 tok/s floor is candle's CPU matmul; revisit after 1–3.

### Rough projected impact

A representative multi-turn second turn today is prefill-bound (≈83–95% of wall in
the bounded run). Batching prefill (#1) should cut that to low single-digit seconds;
adding AC-3 (#2) removes most of it entirely. The `mindeval` aggregate (16.5 k
prefill tok → 4.3 k generated) is where the combined win is largest. Decode (#5)
remains the residual floor.

---

## 6. Reproduce

```sh
cargo build --release -p el-bench

# Full suite (this run): all tasks, default 256 max-tokens, EL_BENCH timing.
EL_BENCH=1 ./target/release/el-bench.exe \
  --out benchmarks/out/transcripts.jsonl

# Bounded smoke (fast, representative): 3 veramh tasks, shorter replies.
EL_BENCH=1 ./target/release/el-bench.exe \
  --suite veramh --limit 3 --max-tokens 128 --out /tmp/perf.jsonl
```

`EL_BENCH` prints the per-phase table + forward attribution to stderr; the
transcript JSONL records per-reply `prompt_tokens` / `completion_tokens` / `ms`.
Per-suite figures are aggregated from the transcript, the prefill/decode split
from the `EL_BENCH` stderr log.
