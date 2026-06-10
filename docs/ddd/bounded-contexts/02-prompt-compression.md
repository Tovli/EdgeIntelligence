# Bounded Context 2 — Prompt Compression  (Supporting)

> Trims a raw prompt to a shorter token stream before prefill, using an
> LLMLingua-2-style BERT token classifier with a per-segment budget.
> Source: PRD §"Prompt Compression (LLMLingua / LLMLingua-2)", §"Prompt
> Compression Details".

## Purpose

Reduce prompt length 3–6× with minimal quality loss so that latency and KV-cache
size both shrink, while emitting a *standard text/token prompt* (not soft
vectors) so any target model stays compatible.

## Strategic role

**Supporting.** Valuable and on the critical path, but the technique
(LLMLingua-2) is established. Must be degradable: under memory pressure the
runtime can disable it (PRD §Risks → "fallbacks disable compression first").

## Ubiquitous language (context-local)

`Prompt Compression`, `Prompt Segment`, `Budget Controller`, `Compression Ratio`.

## Aggregates

### `CompressionRequest` (Aggregate Root)
One compression of one prompt.

- **Identity:** `CompressionId` (correlates to the owning session).
- **Holds:** the ordered list of `PromptSegment`s, the assigned `TokenBudget`,
  and the resulting `CompressedPrompt`.
- **Invariants:**
  - Segments classified as `Query` and `Instruction` receive higher preservation
    than `Demonstration` (questions are protected from over-trimming).
  - The output is always valid standard tokens — never a soft/embedding vector.
  - Achieved `CompressionRatio` must respect the configured floor (developer may
    cap the ratio to protect quality; PRD §Risks "Latency vs Quality").

### `PromptSegment` (Entity)
- **Holds:** `SegmentKind`, raw span, per-token keep/drop decisions.
- **Invariant:** keep/drop decisions come only from the classifier; no segment
  is dropped wholesale unless its budget is zero.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `CompressionId` | opaque id | |
| `SegmentKind` | `Instruction` \| `Demonstration` \| `Query` | drives budget weighting |
| `TokenBudget` | per-segment max keep counts | from `BudgetController` |
| `CompressionRatio` | output ÷ input (target 1/3–1/6) | reported, floored by config |
| `CompressedPrompt` | token list + length | handed to Inference Runtime |
| `KeepDecision` | per-token bool + confidence | classifier output |

## Domain Services

- **`BudgetController`** — assigns `TokenBudget` per segment from the global
  budget and segment kinds. The policy core of the context.
- **`TokenClassifier`** (port + ACL) — the LLMLingua-2 forward pass (a ~100M
  BERT/XLM-R model run on **Candle**) producing `KeepDecision`s. Target: <50 ms
  for 1k tokens, a single Candle forward.
- **`Segmenter`** — splits a raw prompt into `PromptSegment`s.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `TokenClassifierModel` | LLMLingua-2 (XLM-R/mBERT ~100M) via ACL | inbound |
| consumed by `PromptCompressor` | Inference Runtime (1) | outbound (C/S) |

## Anti-Corruption Layer

`LLMLingua2Acl` wraps the BERT-style classifier: it accepts tokenized segments
and returns domain `KeepDecision`s. The classifier's tensor/logit shapes never
escape the ACL.

## Domain Events (published)

`PromptCompressed` (carries input/output token counts and ratio — **never the
text**), `CompressionSkipped` (when disabled under memory pressure). See
[domain-events.md](../domain-events.md).

## Relationships

Customer/Supplier **upstream** of Inference Runtime: it produces a
`CompressedPrompt` consumed by `loadPrompt()`. It does not call any other
context. Privacy note: only token counts/ratios are emitted to Telemetry.
