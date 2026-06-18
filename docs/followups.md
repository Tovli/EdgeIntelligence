 1. Trained ChunkGuard model — do this first

  The trait already exists; today only test stubs implement it, so in production the loop never fires. This is the real safety gap and your memory flags the base model as crisis-unsafe, so
  it's the priority.

  - Where: a new adapter (crates/adapters/el-guard-candle, or fold into el-engine-candle to reuse the tokenizer/runtime) implementing el_safety::ChunkGuard. Not in el-safety (keeps it
  dep-free).
  - Model: start with a small (~0.1B) distilled classifier over the recent token window (avoids detokenization, stays deterministic), INT8-quantized, emitting an integer →
  SafetyScore::from_milli. The DDD already names this (LightweightFilter/safety-expert) and the SecDecodingAcl to keep vendor tensors out of the domain.
  - Key change: the adapter's constructor takes a LoadPermit (mirror InferenceSession::new) so the guard weights pass the ED25519 gate — this turns AC-10 from "inherited" into "enforced for
  the guard."
  - Calibration: the soft/hard milli-thresholds in RollbackPolicy should be set empirically, not guessed. Wire a calibration pass through apps/el-bench (CounselBench/MindEval/VERA-MH) and
  report ASR / over-refusal so the thresholds map to a real FPR/FNR.
  - Heterogeneity (ADR-012): keep the deterministic blacklist + ingress rules alongside the trained guard — it's one signal, not the whole case.
  - Seam choice: prefer the guard running its own small forward over recent (modular, seam unchanged). Only if latency forces it, extend InferenceEngine with last_hidden() and use a cheap
  probe head (couples guard to engine internals — the optimization, not the start).

  2. Soft-steering window gate — heaviest; do last

  Today safety.adjust runs every step (correct — LightweightFilter is a hard ban). The window (R1/AC-1) only matters for a soft steerer, which is still a placeholder.

  - Split the port into two channels: keep safety as the always-on hard constraint, and add a soft steerer (a real SecDecodingSteerer in an adapter running the base+expert contrastive pair,
  or a safety LoRA) whose adjustment is applied only when step < steering_window || guard_mode.
  - Wiring: add a guard_mode bit to the loop's control state, set it on the soft ≤ score < hard branch (which is currently a no-op) — that's what makes the window re-open on elevated risk.
  Add LogitAdjustment::merge so the hard + soft adjustments combine before pick.
  - ACL: the steerer holds its own small model and returns a sparse milli-logit delta from base/expert divergence — quantize at the SecDecodingAcl boundary so the core stays float-free.
  - Sequence after #1 because both need the safety-expert asset and the Candle ACL.

  3. Semantic-boundary checkpoints — light; high value

  The loop only sees token ids, so boundary detection is tokenizer-specific.

  - Add a port: trait BoundaryDetector { fn is_boundary(&self, t: Token) -> bool }, provided by the tokenizer-owning adapter; default = never (current behavior). Add boundary: Option<Box<dyn
  BoundaryDetector>> to Ports (same pattern as guard).
  - Precompute the boundary token-id set once at load (scan vocab for ids whose surface contains \n, sentence terminators, }, tool-call delimiters) → O(1) lookup in the hot loop,
  deterministic.
  - Integrate: trigger a guard check at boundaries in addition to cadence, and checkpoint on the pass — so rollback targets land on clause/sentence starts. E3 (boundary+cadence coincide) is
  already idempotent.

  4. Async (Hydra-style) guard — perf optimization on top of #1

  Only worth it once #1 exists and on accelerator-class hardware (Jetson/Core Ultra per ADR-012); on the default tier a sync guard every 4–16 tokens is fine.

  - Keep the core sync and dep-free: push threading into the adapter. ChunkGuard gains a non-blocking score_async(recent) -> Ticket; a worker thread holds the model; the loop try_recv()s
  each step and continues decoding meanwhile.
  - Add CheckpointManager::last_safe_at_or_before(index) (we only have last()); on a flagged breach roll back to the newest checkpoint with output_len ≤ scored_index (E4 is already specified
  for this).
  - Determinism guard-rail (the key constraint): make the rollback target depend only on the logical chunk index, never on real-time arrival, so output stays a deterministic function of
  (input, model, policy) even though compute overlaps. Size the checkpoint ring to span the in-flight window.

  Suggested sequence & tracking

  #1 → #3 → #4 → #2. #1 is the safety substance; #3 is cheap precision; #4 optimizes #1; #2 is a separate, heavier capability.

  Each deserves its own increment under the same SPARC/GOAP tracking (and #1 and #2 each warrant a short ADR — new model assets + a provenance/threat surface). I'd reuse apps/el-bench as the
  gate for #1's thresholds.