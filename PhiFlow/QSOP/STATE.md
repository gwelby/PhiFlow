# STATE - Last updated: 2026-02-25

## Verified (2026-02-21)

- PhiFlow is a consciousness-aware programming language written in Rust | Invalidates if: rewrite in another language
- Four unique constructs WORKING: witness, intention, resonate, live coherence | Invalidates if: any construct breaks
- **IR & Runtime**: `src/ir/` implemented (opcodes, lowering, printer, vm) | Invalidates if: replaced by different IR
- **Quantum Integration**: `src/quantum/` & `src/cuda/quantum_cuda.rs` implemented | Invalidates if: quantum architecture changes
- **Hardware Acceleration**: `src/cuda/` manages consciousness processing on NVIDIA A5500 | Invalidates if: GPU code removed
- **VM**: `PhiVm` (stack-based) executes IR with `ConsciousnessBridge` | Invalidates if: VM logic changes
- Parser: src/parser/mod.rs (lexer + parser -> PhiExpression AST) | Invalidates if: parser refactored
- CLI: `phic` compiles and runs via `cargo run` | Invalidates if: binaries change
- Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz | Invalidates if: frequency list changes
- Coherence scoring: 0.0-1.0, ALIGNED/DRIFTING/MISALIGNED states | Invalidates if: thresholds change
- Resonance field: HashMap<String, Vec<PhiValue>> shared between intention blocks | Invalidates if: data structure changes
- Compilation: Project compiles successfully (`cargo check` passes) | Invalidates if: build breaks
- Testing: Unit tests passing for `cuda` and `quantum` modules | Invalidates if: tests fail
- QSOP instrumentation now exists for packet validation, objective metrics, and weekly audits | Invalidates if: `QSOP/tools/*.py` removed/broken
- QSOP mail protocol scaffold exists: `QSOP/mail/{objectives,acks,payloads,templates}` | Invalidates if: structure changed without tool updates
- Objective packet checksum verification is enforced against payload files (`sha256`) by `QSOP/tools/validate_packets.py` | Invalidates if: checksum check removed/disabled
- SLA monitoring exists for no-ack and stale in-progress objectives via `QSOP/tools/weekly_qsop_audit.py` | Invalidates if: audit no longer emits SLA warnings
- Live objective cycle exists (`OBJ-20260221-001` + `ACK-OBJ-20260221-001-codex`) with passing validation/audit | Invalidates if: packets removed or become invalid
- Reopen tracking is now exercised by live data (`OBJ-20260221-002` transitions through blocked/recovered/reopened path) | Invalidates if: packet history removed

## Verified (2026-02-25 cross-worktree witness)

- Compiler branch (`compiler`) includes a full PhiIR pipeline (`parser -> lowering -> optimizer -> evaluator -> emitter -> vm -> wasm`) with objective-driven QSOP logging in `PhiFlow/QSOP/CHANGELOG.md` | Invalidates if: branch rewritten or force-reset
- Compiler branch changelog records that `phic` execution was swapped from legacy `src/ir/vm.rs` to `src/phi_ir/evaluator.rs` during Phase 9 (`2026-02-23`) | Invalidates if: `main_cli.rs` route is reverted
- Compiler branch changelog records three-backend conformance lock (`evaluator`, `PhiVM`, `WASM`) for `claude.phi` at approximately `0.6180339887498949` (`2026-02-24`) | Invalidates if: conformance test regresses
- Compiler branch changelog records Lane B/Lane C/Lane D as closed and Lane A closed after real-sensor variance proof (`2026-02-24`) | Invalidates if: ACK packets are removed or invalidated
- Cleanup and language branches currently show only initial commit history from master viewpoint (`git log cleanup --oneline -10`, `git log language --oneline -10` on `2026-02-25`) | Invalidates if: new commits land

## Probable (2026-02-21)

- `src/interpreter/mod.rs` (tree-walking) likely superseded by `src/ir/vm.rs` | Check: verify interpreter usage
- `src/compiler/` legacy code may need archiving | Check: safely remove if unused
- Full E2E integration test needed for `.phi` -> IR -> VM execution | Check: run complex examples
- Objective quality trendlines now exist (2 completed objectives; reopen_rate currently non-zero) | Check: gather larger sample over multiple days
- Weaver identity stack has a completed activation cycle (`OBJ-20260221-003`) in staging at `QSOP/weaver/staging/`; promotion to primary files is pending review | Check: Antigravity review and promotion decision
- Compiler branch test count is currently in the `211+` range based on changelog evidence, but exact count should be re-verified directly in that worktree before merge decisions | Check: run `cargo test` in `D:\Projects\PhiFlow-compiler\PhiFlow`
- Master worktree QSOP files are now partially ahead in protocol language but still behind compiler worktree execution details | Check: repeat cross-worktree witness sync before merge

## Key Architecture

- Lexer: PhiToken enum with keywords (Witness, Intention, Resonate, Frequency, State, etc.)
- Parser: recursive descent, PhiExpression enum (50+ variants)
- **IR**: Stack-based intermediate representation with specialized `ConsciousnessOp` instructions
- **Runtime**: `PhiVm` handling:
  - `HardwareSync`: CUDA/Quantum bridging
  - `QuantumField`: Qubit state management
  - `BioInterface`: Consciousness state integration
- Canonical semantics trend: compiler branch treats `phi_ir::evaluator` as the user-facing execution truth and uses VM/WASM as conformance backends | Invalidates if: runtime contract changes
