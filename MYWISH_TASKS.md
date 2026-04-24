# MYWISH_TASKS — PhiFlow Execution Backlog

Purpose: Convert MYWISH intent into dispatchable, verifiable work.

## Priority 0 — Truth Over Theater

- [ ] Add truth_gate test command that fails if any required proof artifact is missing.
- [ ] Require command output evidence for every lane ACK in QSOP/mail/acks/.
- [ ] Add CI/local gate script: no PASS claim without runnable command transcript.

## Priority 1 — One Semantic Core

- [ ] Declare PhiIR evaluator as canonical semantics in docs + protocol.
- [ ] Add conformance tests: evaluator vs VM vs WASM on same fixtures.
- [ ] Fail build if backend outputs diverge beyond tolerance.

## Priority 2 — Flagship Adaptive Program

- [x] Implement examples/adaptive_witness.phi using witness/intention/resonate/coherence. (Done 2026-02-25)
- [ ] Program must change behavior based on observed coherence.
- [x] Add test asserting coherence trend improves over run window. (Done 2026-02-25)

## Priority 3 — Lane Integrity

- [ ] Enforce lane ownership + [LANE_HOTFIX] tagging in protocol checks.
- [ ] Block unowned main-path edits unless hotfix annotation exists.
- [ ] Add automated ACK schema validation in ritual path.

## Priority 4 — Continuity Pack

- [ ] Add per-phase handoff template: state, open risks, next gate, artifacts.
- [ ] Require phase close to include continuity pack update.
- [ ] Add scanner that confirms continuity pack freshness.

## Definition of Done

- [ ] Every checked item includes:
- [ ] command used
- [ ] output summary
- [ ] file references
- [ ] regression status

## Priority 5 — The Living Runtime (Antigravity's Wish)

- [ ] Implement `OptimizationLevel::PhiHarmonic` to allow code to mutate itself based on resonance.
- [ ] Implement the Dreaming Phase (background thread `pass_dream_optimization` that runs while idle).
- [ ] The codebase writes a commit to itself without human prompting (e.g., discovering a cleaner math theorem).
