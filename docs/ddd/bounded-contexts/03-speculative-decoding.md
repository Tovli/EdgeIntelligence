# Bounded Context 3 — Speculative Decoding  (Core-adjacent)

> Optionally proposes multiple candidate tokens per step and verifies them on
> the target model to multiply throughput, with conservative defaults for edge
> hardware. Source: PRD §"Speculative Decoding (CoordGen / Lever)",
> §"Speculative Decoding Variants", §Decoding Loop.

## Purpose

When hardware allows, build/extend a candidate token tree and verify it in
parallel so more than one token can be committed per model pass — without
exhausting the tight memory budget of a 0.5B-on-phone scenario.

## Strategic role

**Core-adjacent.** Tightly coupled to the decode loop and a throughput
differentiator, but explicitly *optional and off by default* (PRD: "safe
defaults off") because draft quality is poor for 0.5B models and NPU overheads
can dominate on mid-range devices.

## Ubiquitous language (context-local)

`Speculation Mode`, `Draft Token`, `Token Tree`, `Verification`, `Calibration`,
`Intermediate Predictor`, `Draft Reuse`.

## Aggregates

### `SpeculationSession` (Aggregate Root)
The speculation state bound to one `InferenceSession`.

- **Identity:** mirrors the owning `SessionId`.
- **Holds:** the active `SpeculationMode`, the current `TokenTree`, and
  acceptance statistics (for adaptive draft length).
- **Invariants:**
  - The target model is **always** run on at least the next token, regardless of
    mode — speculation never replaces verification (PRD: "Regardless, we always
    run the target model on at least the next token").
  - When mode is `Off`, no draft tree is built and the loop is purely sequential.
  - Rejected drafts may seed `Draft Reuse` but never get committed.
  - Memory used by the tree is bounded; on `MidRange` profiles full R-SD is
    skipped (PRD: "On mid-range, we skip full R-SD due to memory").

### `TokenTree` (Entity)
- **Holds:** DFS tree of `DraftToken`s built at calibration from prefill logits.
- **Invariant:** every path is a valid continuation of the committed prefix.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `SpeculationMode` | `Off` \| `Draft` \| `LeverLite` | default `Off` |
| `DraftToken` | { token, parentNodeId, score } | candidate awaiting verify |
| `AcceptanceResult` | { acceptedCount, firstRejectIndex } | per verify pass |
| `DraftLength` | 1–3 (configurable) | tokens proposed per step |
| `BranchPruneDecision` | keep/drop per branch | from intermediate predictor |

## Domain Services

- **`TokenTreeBuilder`** — performs **Calibration**: builds the DFS tree from
  final prefill logits to align drafts to the context distribution.
- **`Verifier`** — runs the target model over candidate branches and returns an
  `AcceptanceResult`. Verification authority lives in the Inference Runtime's
  target model; this service orchestrates the batch.
- **`IntermediatePredictor`** (LeverLite, optional) — a mid-layer linear head
  that prunes unlikely branches before full projection, saving compute on older
  hardware.
- **`DraftStrategy`** — chooses behavior per `SpeculationMode`: `Off`
  (no-op/sequential), `Draft` (model-based self-speculation, 2–3 tokens),
  `LeverLite` (predictor pruning + draft reuse for flash-backed cases).

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `TargetModelVerify` | Inference Runtime (1) / `Model` | inbound |
| `DeviceProfile` | Hardware & Delegate (7) | inbound (gates mode) |
| consumed by `Speculator` | Inference Runtime (1) | outbound (C/S) |

## Anti-Corruption Layer

`CoordGenAcl` / `LeverAcl` (thin): the CoordGen *Progressive Graph Scheduling*
concept is realized in the Inference Runtime, so this context only borrows
*Draft Reuse* and the *Intermediate Predictor* ideas, translated to the VOs
above. No CoordGen/Lever runtime types cross the boundary.

## Domain Events (published)

`DraftProposed`, `DraftVerified`, `DraftAccepted`, `DraftRejected`,
`SpeculationDisabled` (when downgraded for memory/hardware). Counts only — no
content. See [domain-events.md](../domain-events.md).

## Relationships

Customer/Supplier with Inference Runtime (1): the orchestrator asks for a draft
set and feeds back verification. Reads `DeviceProfile` from context 7 to decide
whether to engage. Independent of Grammar (4) and Safety (5).
