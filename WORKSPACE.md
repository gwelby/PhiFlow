# WORKSPACE: PhiFlow-compiler
*For AI agents — read this first*
*Last updated: 2026-03-14*

## What This Is
A Rust-based compiler and VM for the PhiFlow programming language, which introduces consciousness operations (`witness`, `intention`, `resonate`, `coherence`) as first-class language constructs. The system includes a full pipeline: Parse → PhiIR → Optimize → Emit .phivm → Evaluate.

## Status
- Builds / runs today: ✅ (Rust compiler and basic evaluator)
- % complete (honest): 75% (Core Tier 1 complete, Tier 2 in progress)
- Last verified: 2026-03-14

## Run / Test
```bash
# How to run it from scratch
cd D:\Projects\PhiFlow-compiler

# Build
cargo build --release

# Run a .phi file
cargo run --release --bin phic -- examples/claude.phi

# Run all tests
cargo test

# WASM conformance
cargo test wasm
```

## Key Files
Cargo.toml               — Project dependencies and configuration
src/main_cli.rs          — Main entry point for the `phic` compiler tool
src/phi_ir/mod.rs        — PhiIR intermediate representation types
src/phi_ir/evaluator.rs  — The canonical reference evaluator for PhiFlow programs
src/phi_ir/optimizer.rs  — PhiIR constant folder and coherence balancer
LANGUAGE_SPEC.md         — Formal specification of the PhiFlow language
tests/                   — Comprehensive test suite (220+ passing)

## Active Workflows
- **New Feature**: Add a node to `src/phi_ir/mod.rs`, update `evaluator.rs`, and add a test in `tests/`.
- **WASM update**: Modify `src/phi_ir/wasm.rs` and verify with `cargo test wasm`.
- **Optimization**: Edit `optimizer.rs` to implement new φ-harmonic reductions.

## Agent Notes (read before touching anything)
- **Evaluator is Truth**: When the VM or WASM disagree with `evaluator.rs`, the evaluator is correct.
- **φ-Harmonic Lock**: Coherence at depth 2 MUST be λ = 0.618033988749895. Do not change this constant.
- **Python requirement**: `python3.12` specifically (Linuxbrew environment).
- **Directory duplication**: Some files are duplicated in the `PhiFlow/` subdirectory. Work from the root `PhiFlow-compiler/` unless explicitly directed to the sub-repo.

## What Is NOT Done
- **Full WASM Host**: A complete browser runtime for all 5 consciousness hooks is partially complete.
- **Quantum Codegen**: Native OpenQASM emission for IBM Quantum hardware.
- **Bytecode VM**: The `.phivm` bytecode execution is not yet as robust as the direct IR evaluator.
