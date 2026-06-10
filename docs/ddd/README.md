# Domain-Driven Design — Edge-Native LLM SDK

> Derived from [`docs/prd.md`](../prd.md). This folder holds the **strategic** and
> **tactical** DDD model for the edge-native, on-device LLM SDK. It is design
> documentation, not generated code — the product is a **Rust** SDK targeting
> native ARM and WASM (see `docs/adr/ADR-008`), so these documents define the
> ubiquitous language, bounded contexts, aggregates, and domain events that the
> implementation should honor.

## How to read this folder

| Document | Purpose |
|----------|---------|
| [`ubiquitous-language.md`](./ubiquitous-language.md) | Shared glossary. The single source of truth for every term. |
| [`context-map.md`](./context-map.md) | The nine bounded contexts and the relationships between them. |
| [`domain-events.md`](./domain-events.md) | End-to-end event catalog and the request → token event flow. |
| [`bounded-contexts/`](./bounded-contexts/) | One tactical model per bounded context (aggregates, VOs, events, services, ACLs). |

## Strategic distillation

The product's defensible value is **orchestrating heterogeneous on-device
inference within tight memory/compute budgets, fully air-gapped**. Everything
else either feeds that loop or guards it. Contexts are classified accordingly so
that engineering effort is invested where it differentiates.

| # | Bounded Context | Classification | Why |
|---|-----------------|---------------|-----|
| 1 | [Inference Runtime](./bounded-contexts/01-inference-runtime.md) | **Core** | The decode-loop orchestration, KV-cache lifecycle, and heterogeneous scheduling are the product. |
| 2 | [Prompt Compression](./bounded-contexts/02-prompt-compression.md) | Supporting | LLMLingua-2 token trimming; shrinks latency and KV-cache but built on a known technique. |
| 3 | [Speculative Decoding](./bounded-contexts/03-speculative-decoding.md) | Core-adjacent | Draft/verify throughput multiplier; tightly coupled to the decode loop. |
| 4 | [Grammar Constraint](./bounded-contexts/04-grammar-constraint.md) | Supporting | llguidance (Rust) structured-output masking for agentic tool calls. |
| 5 | [Safety](./bounded-contexts/05-safety.md) | Supporting | On-device logit steering / claim backtracking; a compliance differentiator. |
| 6 | [Memory Management](./bounded-contexts/06-memory-management.md) | Supporting | Static arena planning; foundational to the Core but a well-understood discipline. |
| 7 | [Hardware & Delegate](./bounded-contexts/07-hardware-delegate.md) | Generic | Device profiling and delegate routing; commodity capability detection. |
| 8 | [Model Provenance & Security](./bounded-contexts/08-model-provenance.md) | Generic | Signing, encryption, OTA update gating. |
| 9 | [Telemetry & Privacy](./bounded-contexts/09-telemetry-privacy.md) | Generic | Privacy-preserving performance metrics; pure downstream observer. |

## Bounded-context principles for this SDK

- **Air-gap is an invariant, not a feature.** No context may depend on a network
  port. The optional `HybridMode` relay is modeled as an explicit, opt-in
  outbound port on the Inference Runtime context only.
- **External crates are wrapped, never leaked.** Candle, llguidance, the
  LLMLingua-2 / SecDecoding models (run on Candle), Wasmtime, and `ed25519-dalek`
  each sit behind an Anti-Corruption Layer so library types never become domain
  types.
- **Privacy by construction.** Telemetry is a one-way subscriber to domain
  events and may never carry prompt/response content (see context 9).
- **Memory is planned, not allocated.** The decode loop performs zero heap
  allocation; the Memory Management context owns the static plan that makes this
  an enforceable invariant.

## Status & next steps

This is the initial model from the PRD. Suggested follow-ups (ask if you want
them):

1. Generate per-context Rust crate/module scaffolding (Rust → native ARM +
   `wasm32` per `docs/adr/ADR-008`, not the skill's default TypeScript layout).
2. Capture cross-context decisions as ADRs (`/ruflo-adr:adr-create`).
3. Run boundary validation (`/ruflo-ddd:ddd validate`) after code exists.
