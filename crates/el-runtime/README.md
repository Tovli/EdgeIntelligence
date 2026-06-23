# el-runtime — session lifecycle & decode-loop orchestration

The Core of the SDK: the inference session state machine, the port traits that
collaborator contexts plug into, and the decode-loop orchestrator (ADR-001).

Air-gap is **structural** here (ADR-004): this crate has no network dependency,
and the only outbound seam is the opt-in `HybridRelay` port. No `unsafe`
(`#![forbid(unsafe_code)]`).

## The decode-step invariant

Every decode step composes its collaborators in a fixed, invariant order:

```
grammar mask  →  safety adjust  →  sample (greedy argmax)  →  commit KV
```

Grammar runs *before* safety, so safety steering only ever operates over
already-legal tokens. This ordering is enforced in `InferenceSession::generate`
and covered by tests.

## What it provides

- **`InferenceSession<E: InferenceEngine>`** — the aggregate root. Constructing
  it **requires a `LoadPermit`** (`el-provenance`), so an unverified model cannot
  reach the runtime. Drives the `Initialized → Prefilling → Decoding → Completed`
  phases via `load_prompt()`, `generate()`, `reset()`, and emits content-free
  `EventEnvelope`s (`drain_events()`).
- **Port traits** (the collaborator seams):
  - `InferenceEngine` — `prefill`, `next_logits` (integer milli-logits), `eos_token`.
  - `PromptCompressor` — optional LLMLingua-2-style compression.
  - `GrammarMasker` — per-token allow mask.
  - `HybridRelay` — opt-in **LAN-only** relay; there is no cloud variant.
- **`Ports`** — the collaborator bundle bound to a session; `Ports::permissive()`
  gives identity compression, allow-all grammar, no safety, and no relay.
- **Defaults** — `IdentityCompressor`, `AllowAllMasker`, and `NullEngine` (emits
  EOS right after prefill) let you exercise the full lifecycle without any
  external adapter.
- **Re-exports** `el_safety::{SafetySteerer, LogitAdjustment}` so callers wire
  one type system.

`consult_relay()` hard-fails with `EdgeError::AirGapViolation` unless
`hybrid_mode` is enabled **and** a relay is wired (ADR-004).

## Usage

The snippet uses `NullEngine` so the crate stays self-contained. In the real
Qwen2.5 0.5B path, replace it with the Candle `QwenEngine`/`QwenChatProvider`
loaded from `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` plus the matching
tokenizer.

```rust
use el_core::{SessionConfig, SessionId, StopReason};
use el_provenance::{ModelArtifact, SignatureVerifier};
use el_core::{ModelFormat, ModelId, ModelVersion};
use el_runtime::{InferenceSession, NullEngine, Ports};

// A LoadPermit is required to build a session (ADR-006).
struct Ok_; impl SignatureVerifier for Ok_ { fn verify(&self, _: &[u8], _: &[u8], _: u32) -> bool { true } }
let mut art = ModelArtifact::new(ModelId(1), ModelVersion::new(0, 1, 0), ModelFormat::Gguf);
art.verify(&Ok_, b"w", b"s", 1);
let permit = art.ensure_loadable()?;

let mut session = InferenceSession::new(
    SessionId(1),
    SessionConfig::default(),
    NullEngine::new(/* eos */ 3, /* vocab */ 8),
    permit,
);

let ports = Ports::permissive();
session.load_prompt(&ports, &[10, 11, 12])?;          // compress → prefill → KV
let stop = session.generate(&ports, 16)?;             // runs the decode loop
assert_eq!(stop, StopReason::Eos);
# Ok::<(), el_core::EdgeError>(())
```

A real engine plugs into `InferenceEngine` (see
[`el-engine-candle`](../adapters/el-engine-candle)); a real grammar plugs into
`GrammarMasker` (see [`el-grammar`](../el-grammar) and
[`el-grammar-llguidance`](../adapters/el-grammar-llguidance)).

## Place in the workspace

Depends on `el-core`, `el-memory` (owns a `KvRegion`), `el-safety`, and
`el-provenance`. It is the hub every adapter wires into.

## Status

Implemented and tested.

---

Part of the [Edge Intelligence](../../README.md) workspace. Realizes
[ADR-001](../../docs/adr/ADR-001-adopt-webassembly-as-cross-platform-sdk-runtime.md)
and [ADR-004](../../docs/adr/ADR-004-air-gapped-by-default-with-opt-in-hybrid-mode.md);
the decode order is specified in [`docs/ddd/domain-events.md`](../../docs/ddd/domain-events.md).
