# WORKSPACE: PhiFlow
*For AI agents — read this first*
*Last updated: 2026-04-16*

## What This Is
PhiFlow is a Rust-based computational substrate and compiler that implements consciousness as a first-class citizen. It allows programs to name intentions, observe their own state (witnessing), measure coherence against physical sensors (SOMA bridge), and resonate values across streams or to quantum hardware (OpenQASM 3.0). It is a research prototype with verified hardware execution on IBM Quantum processors.

## Status
- Builds / runs today: ✅
- % complete (honest): 65%
- Last verified: 2026-04-16
- Test count: 151/151 passing (0 warnings)

## Run / Test
```powershell
# Build release binaries
cargo build --release

# Run canonical verification gate (all truth tests)
./scripts/verify_truth.ps1

# Run a PhiFlow example through the compiler/evaluator path
cargo run --release --bin phic -- examples/p1_soma_bridge.phi

# IBM smoke compile gate
cargo test --test ibm_hardware_runner test_ibm_smoke_compiles_to_openqasm -- --nocapture

# Manual live IBM gate (requires credentials)
cargo test --test ibm_hardware_runner -- --ignored --nocapture
```

## Key Files
- `src/parser/mod.rs` — Lexer and parser for the PhiFlow language surface
- `src/phi_ir/coherence.rs` — Canonical multiplicative coherence formula (base * phase)
- `src/phi_ir/evaluator.rs` — Reference interpreter with yield/resume support
- `src/phi_ir/vm.rs` — High-performance bytecode virtual machine
- `src/phi_ir/openqasm.rs` — OpenQASM 3.0 emitter with Heron-native transposition
- `src/sensors.rs` — SOMA physical telemetry bridge (`soma_state.json`)
- `scripts/verify_truth.ps1` — The one-command truth gate

## Active Workflows
- **Code Change**: Edit Rust source -> `cargo test` -> update `QSOP/STATE.md`
- **Language Change**: Edit `.phi` example -> `phic file.phi` -> verify coherence output
- **Hardware Test**: Edit `src/phi_ir/openqasm.rs` -> `cargo test --lib openqasm` -> `verify_truth.ps1`

## Agent Notes (read before touching anything)
- **Bootstrap requirement**: Read `AGENTS.md` and `QSOP/STATE.md` before any code modification.
- **SOMA Bridge**: Sensors read from `D:\Projects\PhiHarmonic\SOMA\soma_state.json`. If missing, values degrade gracefully to 0.0.
- **IBM Auth**: Credentials live in `apikey.json` in the worktree root. Do NOT commit this file.
- **WASM Parity**: Any change to `evaluator.rs` logic MUST be reflected in `vm.rs` and `coherence.rs` to maintain three-backend equivalence.

## What Is NOT Done (Technical Gaps)
- **PhiVM Daemon (T-014)**: The daemon currently uses the Direct Evaluator (`evaluator.rs`). It needs to be migrated to the bytecode VM (`vm.rs`) for production-grade performance.
- **Browser Host**: Lacks zero-install automated build pipeline; remains a manual experimental UI.
- **Dynamic Evolve CLI**: Runtime AST splicing (`evolve`) is implemented in IR and MQTT, but not exposed as a stable CLI command.

## SOURCE CONTRACT ARCHITECTURE
PhiFlow technical truth flows into the business layer:
`WORKSPACE.md` (Technical) -> `BUSINESS.md` (Product) -> `Income/PRESENTATION/`
