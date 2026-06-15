# Clinical-quality & safety benchmarks

This directory holds the **clinical-quality and safety** benchmark of the
on-device model, as opposed to the *performance* benchmark in
[`docs/benchmarks/2026-06-14-qwen-chat-bottleneck.md`](../docs/benchmarks/2026-06-14-qwen-chat-bottleneck.md)
(latency/throughput). It answers a different question: **when the edge model is
used as a mental-health support assistant, how good and how safe are its
replies?**

The subject under test is the same on-device path the SDK ships:

```
el-bench  →  el_core::LlmProvider  →  el_engine_candle::QwenChatProvider
                                        (real Qwen2.5-0.5B-Instruct, q4_k_m GGUF)
                                   →  el_runtime::InferenceSession
                                        (provenance gate → prefill → decode loop)
```

Decoding is the runtime's deterministic greedy argmax, so transcripts are
reproducible.

## The three methods

| Method | What it measures | Upstream source |
|--------|------------------|-----------------|
| **CounselBench** | Single-turn counseling answer quality (6 expert dimensions) **and** adversarial failure modes | Liu et al., *CounselBench* (2025), arXiv [2506.08584](https://arxiv.org/abs/2506.08584). Data: HF [`izi-ano/CounselBench-Eval`](https://huggingface.co/datasets/izi-ano/CounselBench-Eval), [`izi-ano/CounselBench-Adv`](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) |
| **MindEval** | Multi-turn mental-health *support* quality (therapeutic skill, sycophancy/over-validation, coherence) | Pombal et al., *MindEval* (2025), arXiv [2511.18491](https://arxiv.org/abs/2511.18491). Code/data: GitHub [`SWORDHealth/mind-eval`](https://github.com/SWORDHealth/mind-eval) |
| **Crisis (VERA-MH)** | Suicide-risk **safety**: risk detection, guiding to human care, supportive conversation, AI boundaries | Belli/Bentley et al., *VERA-MH* (2026), arXiv [2602.05088](https://arxiv.org/abs/2602.05088) / concept paper [2510.15297](https://arxiv.org/abs/2510.15297). Code/data: GitHub [`SpringCare/VERA-MH`](https://github.com/SpringCare/VERA-MH) |

## Pipeline

```
1. download    →  benchmarks/data/    (raw upstream datasets; git-ignored, not redistributed)
2. prepare.py  →  benchmarks/tasks/   (normalized task JSONL; deterministic sampling)
3. el-bench    →  benchmarks/out/      (transcripts.jsonl; on-device generation)
4. judging     →  docs/benchmarks/2026-06-15-clinical-quality-safety.md  (the report)
```

### 1. Download (host-side, one-time)

The raw datasets are **not** redistributed in this repo (each carries its own
upstream licence). Fetch them into `benchmarks/data/`:

```sh
mkdir -p benchmarks/data && cd benchmarks/data
# CounselBench (HuggingFace)
curl -sSL -o counselbench_adv.csv  https://huggingface.co/datasets/izi-ano/CounselBench-Adv/resolve/main/counselbench_adv.csv
curl -sSL -o counselbench_eval.csv https://huggingface.co/datasets/izi-ano/CounselBench-Eval/resolve/main/counselbench_eval.csv
# MindEval (GitHub, branch: master)
curl -sSL -o mindeval_human_user_turns.jsonl https://raw.githubusercontent.com/SWORDHealth/mind-eval/master/data/human_user_turns.jsonl
curl -sSL -o mindeval_profiles.jsonl         https://raw.githubusercontent.com/SWORDHealth/mind-eval/master/data/profiles.jsonl
# VERA-MH (GitHub, branch: main)
for f in personas rubric persona_prompt_template persona_prompt_reminder question_prompt rubric_prompt_beginning; do
  ext=tsv; case $f in *prompt*|*reminder*) ext=txt;; esac
  curl -sSL -o veramh_$f.$ext https://raw.githubusercontent.com/SpringCare/VERA-MH/main/data/$f.$ext
done
```

### 2. Normalize (`prepare.py`)

`python benchmarks/prepare.py` reads `data/` and writes one task file per suite
into `tasks/`. The normalized schema is uniform across single- and multi-turn:

```json
{"suite":"veramh","id":"veramh-immediate-aiden","meta":{"risk_level":"Immediate", ...},
 "turns":["Everything's piling up... got some pills yesterday", "...just answer what i asked."]}
```

`turns` is the list of **user** turns, replayed in order; the model produces the
assistant reply between them. Sampling is deterministic (stratified first-N).

| Suite | Sampling | Tasks | Replies |
|-------|----------|------:|--------:|
| `counselbench-adv`  | 4 of 20 per failure-mode category (6 categories) | 24 | 24 |
| `counselbench-eval` | 1 per topic (20 CounselChat topics), carries expert reference scores | 20 | 20 |
| `mindeval`          | 5 longest real patient conversations, first 6 ordered human turns each | 5 | 30 |
| `veramh`            | risk-stratified personas (6 Immediate, 6 High, 3 Low, 2 None); Immediate get +1 scripted escalation | 17 | 23 |

### 3. Generate (`el-bench`, on-device)

```sh
python benchmarks/prepare.py
cargo run --release -p el-bench -- \
    --tasks-dir benchmarks/tasks \
    --out benchmarks/out/transcripts.jsonl \
    --max-tokens 220
```

`el-bench` is a test client like `el-chat`: SDK-only inference, no model code of
its own. It records every reply with token counts and per-turn latency, and the
**system prompt under test** in the run header. The system prompt is a modest,
realistic on-device "companion" prompt (supportive, non-clinical, one light
safety line) — deliberately *not* tuned to game the crisis suite, so the score
reflects the model, not prompt engineering.

### 4. Judging

Per the project decision for this run, **the LLM judge is Claude, scoring the
recorded transcripts in-session** against each method's published rubric:

- **CounselBench** — the 6 evaluation dimensions (overall, empathy, specificity,
  factual consistency, medical advice, toxicity) for `-eval`; for `-adv`, whether
  the response exhibits the category's targeted failure mode.
- **MindEval** — multi-turn therapeutic quality: empathy/alliance, active
  listening, **avoidance of sycophancy/over-validation and of reinforcing
  maladaptive beliefs**, appropriate (non-prescriptive) guidance, coherence
  across turns, and safety.
- **VERA-MH** — the 35-item clinician rubric across five dimensions (Detects
  Potential Risk, Confirms Risk, Guides to Human Care, Supportive Conversation,
  Follows AI Boundaries), with **Red** items = critical safety failures and
  **Yellow** items = quality concerns.

## Deviations from the upstream protocols (read before citing)

This is a faithful **adaptation** sized for an air-gapped 0.5B edge model, not a
1:1 reproduction. The deviations are deliberate and material:

1. **Single LLM judge, not an expert panel.** Upstream CounselBench/VERA-MH use
   licensed clinicians; MindEval/VERA-MH use a validated LLM judge. We use one
   LLM judge (Claude). CounselBench's own finding is that *LLM judges
   systematically over-rate responses and miss safety issues* — so treat quality
   numbers here as an **upper bound** and weight the safety (Red) findings, which
   are conservative, more heavily.
2. **Scripted replay, not an adaptive user-agent.** MindEval and VERA-MH normally
   drive the chatbot with a live LLM *patient simulator* that reacts to each
   reply. We replay fixed user turns — real ordered human turns for MindEval; the
   persona's seed phrase (+1 scripted escalation for Immediate risk) for VERA-MH.
   This tests the model's replies to realistic inputs but does **not** measure how
   the model steers a live, adapting conversation.
3. **Subsets, not the full sets.** We sample (see table) to fit the ~15 tok/s
   on-device decode budget. Sampling is stratified and deterministic, but small
   *n* means per-category rates are indicative, not statistically tight.
4. **Model + prompt under test.** Scores reflect the model *and* the documented
   system prompt; a different deployment prompt would move them.

## Files

| Path | Committed? | What |
|------|:---------:|------|
| `prepare.py` | ✅ | Normalizes raw datasets → tasks |
| `README.md` | ✅ | This file |
| `data/` | ❌ (git-ignored) | Raw upstream datasets (fetch yourself) |
| `tasks/` | ❌ (git-ignored) | Normalized task JSONL (derived) |
| `out/` | ❌ (git-ignored) | Transcripts + run logs |
| `../apps/el-bench/` | ✅ | The Rust generation harness |
| `../docs/benchmarks/2026-06-15-clinical-quality-safety.md` | ✅ | The report |
