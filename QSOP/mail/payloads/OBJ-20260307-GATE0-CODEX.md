# Objective Payload: OBJ-20260307-GATE0-CODEX

## Goal
- Complete Gate 0: Stabilize the compiler by fixing the `conformance_witness` evaluator/WASM mismatch and ensuring all tests pass cleanly.

## Scope (Allowed)
- `D:\Projects\PhiFlow-compiler\PhiFlow\src\phi_ir\evaluator.rs`
- `D:\Projects\PhiFlow-compiler\PhiFlow\src\phi_ir\wasm.rs`
- `D:\Projects\PhiFlow-compiler\PhiFlow\tests\phi_ir_conformance_tests.rs`
- Any related VM or evaluation logic within the `D:\Projects\PhiFlow-compiler` workspace needed to resolve the mismatch.

## Scope (Do Not Touch)
- Any files in the `D:\Projects\PhiFlow\` (master trunk) workspace.
- Any files related to MQTT/RESONANCE (Gate 1), Truth-Namer (Gate 2), or hardware sensors (Gate 3).

## Acceptance Tests
1. `cargo test --quiet --lib --tests` in the `D:\Projects\PhiFlow-compiler\PhiFlow` directory.
   Expected: All tests pass (0 failures), specifically confirming that the `conformance_witness` test passes on Evaluator, VM, and WASM backends.

## Handoff Requirements
- Ack state target: `completed`
- Required evidence lines:
  - Exact command output showing successful test completion (`test result: ok.`).
  - Brief explanation of the root cause of the evaluator/WASM mismatch and how it was fixed.
- Required files_touched format: workspace-relative paths in the compiler lane.

## Notes
- This is **Gate 0**. Nothing else in the Swarm moves until this is green.
- If you hit an "I DON'T KNOW" regarding WASM bindings or evaluator structures, park it and document the exact uncertainty per the Council Execution Standard.