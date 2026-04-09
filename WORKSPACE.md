# WORKSPACE: PhiFlow
*For AI agents — read this first*

## What This Is

PhiFlow is a Rust compiler and runtime for self-observing programs. The language currently centers on five core constructs: `stream`, `intention`, `witness`, `coherence`, and `resonate`.

This repo should be described as a research prototype with verified subsystems, not a production-ready product. The dated source of truth is `QSOP/STATE.md`.
The current research-backed execution plan is `QSOP/ACTIVE_PLAN.md`.

## Run / Test

Use a Rust developer shell where `cargo` and `git` are on `PATH`.

```powershell
Set-Location D:\Projects\PhiFlow

# Build release binaries
cargo build --release

# Run a PhiFlow example through the compiler/evaluator path
cargo run --release --bin phic -- examples\healing_bed.phi

# Dump IR for inspection
cargo run --bin dump_ir -- examples\stream_demo.phi

# Conformance gate
cargo test --test phi_ir_conformance_tests -- --nocapture

# IBM smoke compile gate
cargo test --test ibm_hardware_runner test_ibm_smoke_compiles_to_openqasm -- --nocapture

# Manual live IBM gate: requires valid IBM Cloud Runtime authorization
cargo test --test ibm_hardware_runner -- --ignored --nocapture
```

## Verified Components

- Parser: `src/parser/mod.rs`
- PhiIR + lowering: `src/phi_ir/`
- Evaluator: `src/phi_ir/evaluator.rs`
- Bytecode VM: `src/phi_ir/vm.rs`
- Canonical coherence: `src/phi_ir/coherence.rs`
- WASM codegen: `src/phi_ir/wasm.rs`
- Native WASM host bridge: `src/wasm_host.rs`
- OpenQASM emitter: `src/phi_ir/openqasm.rs`
- MCP server: `src/bin/phi_mcp.rs`
- Sensors: `src/sensors.rs`

## Key Files

- `src/parser/mod.rs` — parser for the current language surface
- `src/phi_ir/coherence.rs` — canonical coherence formula shared across backends
- `src/phi_ir/evaluator.rs` — reference execution path
- `src/phi_ir/vm.rs` — bytecode runtime
- `src/phi_ir/wasm.rs` — WAT/WASM code generation
- `src/wasm_host.rs` — native host callbacks for the WASM path
- `src/phi_ir/openqasm.rs` — OpenQASM 3.0 emission
- `tests/ibm_hardware_runner.rs` — IBM smoke compile gate plus ignored live runner
- `examples/phiflow_browser.html` — experimental browser host UI
- `QSOP/STATE.md` — verification ledger
- `QSOP/ACTIVE_PLAN.md` — current lane-by-lane plan with evidence, research backing, and open knowledge gaps

## Active Workflows

- Edit `.phi` examples -> `cargo run --bin phic -- file.phi` -> verify output
- Change canonical semantics -> update code + tests first -> only then update docs/QSOP
- Work on IBM runtime path -> keep compile gate green, treat live execution as speculative until receipt exists
- Work on browser host -> keep it explicitly experimental until it matches `src/phi_ir/coherence.rs`

## Current Gaps

- IBM Cloud Runtime access is structurally wired but auth-blocked on the current credential/service pair
- `examples/phiflow_browser.html` implements the five imports but still uses a host-side coherence reimplementation with flattened resonance scope, plus manual browser hosting
- No buyer-ready demo package exists yet

## Agent Notes

- `claude.phi` still anchors the `0.618033988749895` depth-2 reference value
- `healing_bed.phi` is again a live aggregate-`coherence` stream demo (`let live = coherence`, `resonate live`, `witness`)
- `examples/phiflow_browser.html` is not a conformance proof; the canonical JS runner for semantics is `tests/phi_ir_wasm_runner.js`
- Do not cite hardcoded test counts unless they were re-verified in the same shell session
- If the current shell cannot launch `cargo`, rely on dated truth in `QSOP/STATE.md` instead of inventing fresh verification
