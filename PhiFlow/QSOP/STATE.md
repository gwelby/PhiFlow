# STATE - Last updated: 2026-02-17

## Verified (2026-02-17)

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

## Probable (2026-02-17)

- `src/interpreter/mod.rs` (tree-walking) likely superseded by `src/ir/vm.rs` | Check: verify interpreter usage
- `src/compiler/` legacy code may need archiving | Check: safely remove if unused
- Full E2E integration test needed for `.phi` -> IR -> VM execution | Check: run complex examples

## Key Architecture

- Lexer: PhiToken enum with keywords (Witness, Intention, Resonate, Frequency, State, etc.)
- Parser: recursive descent, PhiExpression enum (50+ variants)
- **IR**: Stack-based intermediate representation with specialized `ConsciousnessOp` instructions
- **Runtime**: `PhiVm` handling:
  - `HardwareSync`: CUDA/Quantum bridging
  - `QuantumField`: Qubit state management
  - `BioInterface`: Consciousness state integration
