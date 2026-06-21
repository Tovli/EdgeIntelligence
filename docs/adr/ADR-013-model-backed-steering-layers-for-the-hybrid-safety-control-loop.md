# ADR-013: Model-backed steering layers for the hybrid safety control loop

- **Status**: proposed
- **Date**: 2026-06-19
- **Deciders**:
- **Tags**: safety, security, on-device, runtime, follow-up

## Context

[ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md)
decided a **hybrid control loop** — *heterogeneous* by design: model-backed
steering + checkpointed rollback + a chunk guard + deterministic ingress rules,
degrading across device tiers. What actually shipped is the **recoverable loop
skeleton**: checkpointed rollback (`InferenceSession::generate_with_policy`,
`CheckpointManager`, `RollbackPolicy`, `el-memory::KvRegion::truncate`, per-engine
`InferenceEngine::rollback`), a chunk-guard cadence (`guard_every` + a mandatory
final check), every-step hard-ban steering (`LightweightFilter`), and bounded
fail-closed refusal.

The loop's **model-backed half was deliberately deferred** (ADR-012 §Consequences
"As built"). The gaps:

- **No selective early-token soft-steering window.** ADR-012 states the window
  "ships with the SecDecoding steerer (follow-up)". Today steering is hard-ban
  every step, not a windowed soft steer over the first 8–32 tokens.
- **No model-backed steering signal.** `SecDecodingSteerer::adjust()` is a
  placeholder returning `LogitAdjustment::none()`; no same-family safety **LoRA
  adapter**, **base+expert contrastive pair**, or **refusal classifier** is wired.
- **The chunk guard is weights-free.** It ships as token-anchor heuristics
  (`AnchorGuard`) matching whole token n-grams — a model can evade via sub-word
  fragments or inflections — instead of *reusing the active tier's safety model*
  as the ADR-012 model inventory specifies.
- **No deterministic ingress / prompt-risk triage.** The guard scores only
  generated output (`&self.output`), never the prompt, so the ADR-012 "ingress"
  layer is absent.
- **The guard is synchronous.** No Hydra-style async overlap with generation.

Consequence today: only `Off` and `Lightweight` are faithfully backed.
`SecDecoding`/`Csd` fall back to the Lightweight wiring with a *misleading label*,
so the `el-chat` and `el-bench` CLIs deliberately expose only `off|lightweight`
(see those apps' `--safety` flags). The loop is a real defence-in-depth net, but
provides little measurable safety lift on adversarial suites (e.g.
CounselBench-adversarial) because the discriminating signal — a real safety model
— is missing.

## Decision

Complete the hybrid control loop by implementing the deferred **model-backed
layers behind the existing ports** (`SafetySteerer`, `ChunkGuard`), with **no
change to the proven loop control flow**. All stages remain on-device and
air-gapped (ADR-004); every safety weight is integrity-gated on load (ADR-006).

1. **Real model-backed steerer (Lightweight upgrade + SecDecoding).** Replace the
   `SecDecodingSteerer` placeholder with a steerer that derives a real
   `LogitAdjustment` from a same-family safety signal — a Qwen safety **LoRA
   adapter** (shares base weights; the `Lightweight` default) and, on
   accelerator-class HW, **contrastive** base-vs-expert steering (`SecDecoding`).

   Contrastive steering needs two *distributions* that differ in safety, **not
   two new models**. The cheapest faithful arrangement (classic SafeDecoding) is:
   - **base** = the main generator itself, *reused* (no new artifact);
   - **expert** = the same base **+ the safety LoRA adapter** (a small delta that
     shares base weights, not a second full model);
   - `steer = logits_expert − logits_base`, then `final = logits_base + α·steer`.

   So the *minimum* new artifact for `SecDecoding` is **one** same-family safety
   LoRA — the same adapter that backs `Lightweight`, used contrastively rather
   than directly. A **separate small base+expert pair** is warranted only when the
   main generator is **large** and running it twice per token is too costly; with
   a small generator (e.g. Qwen2.5-0.5B) reuse it as the base. Either way, base
   and expert must share the **single tokenizer** with the generator (ADR-012
   invariant — steering intersects token sets), and steering costs two forward
   passes per steered token, bounded to the early-token window (§2) and to
   accelerator-class HW.

2. **Selective early-token soft-steering window.** Apply the model-backed
   adjustment on prompt ingress and the first **8–32 output tokens**, then fall
   back to normal decode unless the chunk guard re-escalates. Add a tier-aware
   window width to `RollbackPolicy` and `generate_with_policy`; hard bans remain
   every-step.

3. **Deterministic ingress / prompt-risk triage.** Score the prompt at/before
   prefill (the tier's safety model + deterministic rules) and raise steering
   strength / inject refusal bias / fail closed *before* an unsafe trajectory
   begins — the heterogeneous ingress layer ADR-012 names.

4. **Model-backed chunk guard.** Have the guard reuse the active tier's safety
   model (classifier/adapter score) instead of — or alongside — `AnchorGuard`
   heuristics, keeping the weights-free path as the degraded fallback under
   memory pressure.

5. **Async guard (Hydra-style).** Where HW allows, run the chunk guard
   concurrently with generation so valid output pays near-zero cost; synchronous
   remains the low-tier fallback.

6. **Integrity-gate safety weights (ADR-006).** ED25519-sign and load-gate every
   safety artifact (adapter, expert, classifier) exactly like the generator.

7. **Faithful mode surface.** Once backed: construct `SecDecodingSteerer` for
   `SecDecoding` mode in `QwenChatProvider`; apply `SafetyModeSelector` in the
   decode path so `SecDecoding` downgrades on non-accelerator HW; record the
   **effective** mode on events / in benchmark run-headers; and only then expose
   `secdecoding`/`csd` in the `el-chat` and `el-bench` CLIs.

Tier-awareness and the [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md)
memory budget continue to govern which layers are active; under pressure the loop
degrades to the weights-free `Lightweight` heuristics and ultimately
`SafetyDisabled`.

## Consequences

### Positive
- Real, measurable safety lift on adversarial suites — the loop gains the
  discriminating signal it currently lacks (the el-bench A/B becomes meaningful).
- The early-token window captures most trajectory risk at the lowest sidecar
  cost — the ADR-012 rationale finally realized.
- The monitor becomes genuinely heterogeneous (ingress + model steer + chunk
  guard + grammar), not a single judge / single attack surface.

### Negative
- Ships model assets (adapter / expert / classifier): added memory, load time,
  and signing/distribution infrastructure (ADR-006 / [ADR-011](./ADR-011-multi-registry-release-ci-crates-io-npm-pub-dev.md)).
- Model-backed steering and a synchronous guard add latency; the contrastive
  pair is accelerator-class-HW only, so CPU hosts cannot run `SecDecoding`.
- The tokenizer-identity constraint couples safety assets to the generator
  family (the reason for a same-family Qwen choice).

### Neutral
- CPU / low-tier hosts stay on the weights-free `Lightweight` heuristics; the
  loop's control flow is unchanged — only the ports' *implementations* and the
  steering-window knob are added.
- Extends ADR-012; supersedes nothing.

## Links
- Builds on: [ADR-012](./ADR-012-layered-decode-time-safety-control-loop-with-checkpointed-rollback.md) (the loop), [ADR-005](./ADR-005-on-device-only-tiered-decoder-time-safety.md) (tiers + `LogitAdjustment` placement).
- Constrained by: [ADR-004](./ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md) (no safety stage touches the network), [ADR-006](./ADR-006-mandatory-ed25519-model-signature-verification-load-gate.md) (sign safety weights), [ADR-003](./ADR-003-static-memory-planning-with-zero-allocation-arena.md) (memory budget / degradation), [ADR-002](./ADR-002-candle-as-rust-native-inference-engine.md) (Candle engine / KV-cache rollback cost).
- Research: [docs/research/SecDecoding alternatives research.md](../research/SecDecoding%20alternatives%20research.md)
- Implementation seams: `crates/el-safety` (`SecDecodingSteerer`, `SafetySteerer`, `ChunkGuard`, `AnchorGuard`, `RollbackPolicy`, `SafetyModeSelector`), `crates/el-runtime` (`generate_with_policy`), `crates/adapters/el-engine-candle` (`QwenChatProvider`), `apps/el-chat` + `apps/el-bench` (the `--safety` surface).

## Implementation status

Landed via a SPARC pass ("full plumbing + port evolution" scope):

- **Port evolution** — defaulted `SafetySteerer::adjust_with_logits(recent, base_logits)`; the decode loop calls it window-gated (`el-safety`, `el-runtime`). Additive, no control-flow change; all prior tests stay green.
- **R1 early-token window** — `RollbackPolicy::steer_window` (tier-aware) gates soft steering in `generate_with_policy`.
- **R2 ingress triage** — `Ports::ingress` + retained session prompt; a hard breach fails closed before any decode. Wired in the adapter via `AnchorGuard` over the prompt.
- **R3 contrastive steering** — `el_safety::contrastive_adjustment` + `ContrastiveSteerer<E: ExpertLogits>`; `el_engine_candle::QwenExpert` (second Qwen engine) implements `ExpertLogits`. Enabled by `--expert-model` (top-K restricted; α via `--steer-alpha`).
- **R4 effective mode** — `generate` applies `SafetyModeSelector` in the decode path and emits `SafetyModeSelected{effective}`.
- **R5 load gate** — `QwenExpert::from_path_primed` requires a `LoadPermit` (ADR-006). For a user-supplied local GGUF there is no detached signature, so this is the same *trust-the-local-file* verifier the base model uses: it enforces the load-permit protocol and the loader rejects malformed GGUFs, but it is **not** cryptographic integrity over the weights. A signed safety artifact (production) needs a separate signed-artifact path that verifies the whole GGUF bytes plus detached signature before issuing the permit.
- **Expert rollback alignment** — when the base engine rolls back, `QwenExpert` re-primes to the prompt and re-feeds the retained prefix, so the contrastive context stays aligned (no stale-branch logits). Cost is bounded by `max_rollbacks`, like the base engine.

Still **deferred** (need real trained assets / HW): the safety LoRA / classifier weights themselves, accelerator-class true `SecDecoding`, the async (Hydra-style) guard, and **cryptographic verification of user-supplied weights** (no signature exists for local files). The `Lightweight` tier remains weights-free token-anchor heuristics.

### Review fixes (post-implementation hardening)

A code review surfaced six issues, all fixed: (1) the expert permit used a dummy local verifier with an overbroad integrity claim → the claim now explicitly says local GGUFs are trust-the-file only and production signed assets need whole-artifact verification; (2) the expert kept the abandoned branch after a rollback → now re-primes; (3) an expert installed a `SecDecoding` steerer while the session stayed `Lightweight` → the session is now promoted to `SecDecoding` so `SafetyModeSelector` gates it on device class and telemetry is honest; (4) top-K contrastive ranking ignored the grammar mask → grammar-illegal tokens are now hidden from the steerer; (5) `--guard-word` extras leaked into ingress and refused the rollback demo → ingress now uses built-in patterns only; (6) `--steer-alpha` was unchecked → negatives are clamped (never reverse the safety direction), the milli math saturates, and the CLI validates `0..=4000`.
