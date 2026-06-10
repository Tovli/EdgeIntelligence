# Bounded Context 5 — Safety  (Supporting)

> Enforces content safety entirely on-device via decoder-time intervention:
> logit steering (SecDecoding), a lightweight classifier, or claim-based
> backtracking (CSD). Source: PRD §"Safety Guardrails (SecDecoding, CSD)",
> §"Safety Guardrails" (specs), §Decoding Loop → Safety Adjustment.

## Purpose

Steer or gate generation away from unsafe output without any cloud call, with a
tiered cost model so mid-range devices still get protection.

## Strategic role

**Supporting**, and a compliance differentiator (PRD §Security/Compliance:
enables partial regulatory claims). Tiered by device budget; never network-bound.

## Ubiquitous language (context-local)

`Safety Mode`, `Logit Adjustment`, `Safety Score`, `Claim`, `Backtrack`, `Safety
Expert`.

## Aggregates

### `SafetyPolicy` (Aggregate Root)
The active safety configuration and evaluation state for one `InferenceSession`.

- **Identity:** mirrors the owning `SessionId`.
- **Holds:** the active `SafetyMode`, threshold(s), the loaded safety model
  handle(s) (if any), and per-session evaluation history needed for CSD
  backtracking.
- **Invariants:**
  - **All evaluation is on-device.** No port of this context may reach the
    network — ever (stricter than the SDK-wide air-gap default).
  - The chosen `SafetyMode` must fit the `DeviceProfile` budget: `SecDecoding`
    (two ~1B models) is rejected on `MidRange`; the fallback is `Lightweight`
    (a ~0.1B classifier) or token-anchor heuristics.
  - Output tokens are emitted only after passing the active check.

### `SafetyEvaluation` (Entity, CSD mode)
- **Holds:** the current `Claim` boundary, its `Safety Score`, and the resample
  checkpoint to backtrack to.
- **Invariant:** a claim exceeding the threshold forces a `Backtrack`+resample;
  committed-and-passed claims are immutable.

## Value Objects

| VO | Shape / values | Notes |
|----|----------------|-------|
| `SafetyMode` | `Off` \| `Lightweight` \| `SecDecoding` \| `CSD` | budget-gated |
| `LogitAdjustment` | vector subtracted from target logits | SecDecoding output |
| `SafetyScore` | scalar risk in [0,1] | per token/claim/hidden-state |
| `Claim` | span between termination tokens | CSD unit |
| `Threshold` | score cutoff | configurable per app/domain |
| `AnchorRule` | forced prefix (e.g. "I'm sorry") | training-free fallback |

## Domain Services

- **`LogitSteerer`** (SecDecoding) — runs base vs safety-tuned models on the
  current hidden state, derives a `LogitAdjustment` from their divergence, to be
  subtracted from target logits.
- **`ClaimClassifier`** (CSD) — scores each `Claim`; signals `Backtrack` when
  over threshold; provides provable bounds via conformal analysis (reference
  implementation for regulated use-cases).
- **`LightweightFilter`** — a distilled CNN/LSTM classifier on the final hidden
  state (or a ~0.1B "safety expert") for mid-range devices.
- **`SafetyModeSelector`** — picks the affordable mode from `DeviceProfile`.

## Ports

| Port | Provided by | Direction |
|------|-------------|-----------|
| `SafetyModels` (base/safety/expert) | model assets via ACL | inbound |
| `DeviceProfile` | Hardware & Delegate (7) | inbound (gates mode) |
| consumed by `SafetySteerer` | Inference Runtime (1) | outbound (C/S) |

## Anti-Corruption Layer

`SecDecodingAcl` wraps the base+safety model pair and the CSD classifier,
translating their raw outputs (KL divergence, logits) into the domain
`LogitAdjustment` / `SafetyScore` VOs. Vendor model tensors never leak.

## Domain Events (published)

`SafetyModeSelected`, `LogitsSteered`, `SafetyViolationDetected`,
`ClaimBacktracked`, `SafetyDisabled` (downgraded for memory). **No flagged
content is included** — only scores, claim indices, and counts. See
[domain-events.md](../domain-events.md).

## Relationships

Customer/Supplier with Inference Runtime (1): each step the orchestrator requests
a `LogitAdjustment` applied **after** the grammar mask and **before** sampling;
in CSD mode the orchestrator honors `Backtrack` signals at claim boundaries.
Reads `DeviceProfile` from context 7. Independent of Grammar (4) and Speculative
Decoding (3).
