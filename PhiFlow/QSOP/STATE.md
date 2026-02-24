# STATE - Last updated: 2026-02-21

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

## Probable (2026-02-21)

- `src/interpreter/mod.rs` (tree-walking) likely superseded by `src/ir/vm.rs` | Check: verify interpreter usage
- `src/compiler/` legacy code may need archiving | Check: safely remove if unused
- Full E2E integration test needed for `.phi` -> IR -> VM execution | Check: run complex examples
- Objective quality trendlines now exist (2 completed objectives; reopen_rate currently non-zero) | Check: gather larger sample over multiple days
- Weaver identity stack has a completed activation cycle (`OBJ-20260221-003`) in staging at `QSOP/weaver/staging/`; promotion to primary files is pending review | Check: Antigravity review and promotion decision

## Key Architecture

- Lexer: PhiToken enum with keywords (Witness, Intention, Resonate, Frequency, State, etc.)
- Parser: recursive descent, PhiExpression enum (50+ variants)
- **IR**: Stack-based intermediate representation with specialized `ConsciousnessOp` instructions
- **Runtime**: `PhiVm` handling:
  - `HardwareSync`: CUDA/Quantum bridging
  - `QuantumField`: Qubit state management
  - `BioInterface`: Consciousness state integration
