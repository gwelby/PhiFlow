# PhiFlow - Code That Lives

## Read First
- **QSOP/STATE.md** - Current project state (what works, what doesn't, architecture)
- **QSOP/PATTERNS.md** - Known pitfalls and what works
- **LANGUAGE.md** - What makes PhiFlow unique (the four constructs)

## What Is PhiFlow

A programming language where code observes itself, declares its purpose, communicates internally, and measures its own alignment with reality. Written in Rust.

Four constructs that exist in no other language:
1. `witness` - program pauses to observe its own state
2. `intention "name" { }` - program declares WHY before HOW
3. `resonate` - intention blocks share state through resonance
4. Live coherence - program measures alignment 0.0 to 1.0

## Architecture

```
.phi file -> Parser (PhiToken -> AST) -> PhiIR Lowering -> Evaluator/VM/WASM -> Output + Coherence Report
```

Backend equivalence: `tests/phi_ir_conformance_tests.rs` has 12 tests (12/12 pass, 2026-11-11). 8 compare Evaluator == VM == WASM by value on core constructs (incl. `three_backend_depth2_coherence` = 1 − φ⁻²); 2 compare Evaluator == WASM only (the legacy VM returns Void for user-defined functions); 1 runs shared examples on Evaluator + WASM without comparing values; 1 is a WASM NaN guard. `tests/phi_ir_full_conformance_probe.rs` is a *diagnostic* over 8 v0.3+ constructs that never fails on divergence — it currently prints 0 divergences (observed, not asserted). Self-correction loop: **structural plumbing only** — `run_self_correction_loop()` (src/quantum_feedback.rs) re-measures with hardcoded mocks (Decoherent 0.0 → Coherent 1.0), so the "improvement" is simulated and does not depend on the correction; no real-hardware correction run is recorded. CLI output tests: 5 tests verify what the user sees. Full suite: **501 passed, 0 failed, 5 ignored** (`cargo test`, 2026-11-11, HEAD `bcff85b`, producer-tested; +1 conformance test added after that run) + 43 Python tests passed. Codex audit 2026-07-31 found and fixed the coherence reporting bug (every program reported 0.0000 while tests passed green).

Key files:
- `src/parser/mod.rs` - Lexer + Parser
- `src/phi_ir/` - PhiIR (lowering, evaluator, VM, WASM codegen, OpenQASM)
- `src/main_cli.rs` - CLI binary (`phic`)
- `src/wasm_host.rs` - WASM runtime host (provides phi namespace imports)
- `src/metrics/` - Consciousness metrics (L_self, C_PF, R_in, R_out)
- `src/quantum/` - IBM Quantum backend topology and transpilation
- `src/consciousness/` - Consciousness math, sacred geometry, bridge
- `src/mcp_server/` - MCP stdio server for AI assistant integration
- `src/visualization/` - SVG generation from sacred geometry patterns
- `examples/` - Working .phi programs

Legacy modules archived in `src/_archive/` (compiler, vm, interpreter, main.rs).

## Build & Run

```bash
cargo build --release
cargo run --release --bin phic -- examples/code_that_resonates.phi
cargo test                                    # 501 passed, 0 failed, 5 ignored at bcff85b (2026-11-11); +1 test since
cargo test --test phi_ir_full_conformance_probe -- --nocapture  # prints per-construct backend comparison table (diagnostic; does not fail on divergence)
cargo run --release --bin phic -- --measure examples/type4_trace_benchmark.phi
cargo run --release --bin phic -- --sacred-geometry flower_of_life | sed -n '/<svg/,$p' > pattern.svg   # stdout has log lines before the SVG
cargo run --release --bin phic -- --consciousness-info | jq .
cargo run --release --bin phic -- --mcp-serve  # MCP stdio server
```

## Rules for Contributing

1. **Test after every change** - Run `cargo build --release` and test with example .phi files
2. **Keyword collision** - If you add a keyword, update `expect_identifier()` in parser to accept it as a variable name too (Pattern P-1)
3. **Bare keyword forms** - If a keyword can be bare (no arguments), check what IMMEDIATELY follows before consuming newlines (Pattern P-2)
4. **Coherence math** - Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz. Tolerance: +/-5Hz. Only check phi-harmonic ratios between sacred frequencies.
5. **QSOP** - Update QSOP/STATE.md when you change architecture. Update QSOP/PATTERNS.md when you find recurring issues or successes.

## Agent Team

Four specialized agents defined in `.claude/agents/`:
- **wasm-backend** - WebAssembly compilation target
- **quantum-backend** - IBM Quantum circuit generation
- **hardware-backend** - ESP32/P1 firmware target
- **docs-specialist** - Documentation and examples

## What's Done

- ✅ PhiIR (intermediate representation) — lowering, evaluator, VM, WASM codegen
- ✅ Backend equivalence — 12/12 core conformance tests (8 three-backend, 2 Evaluator==WASM only); full probe diagnostic shows 0 divergences over 8 constructs
- ✅ WASM codegen — all 15 phi namespace imports (incl. `actuate`, added da61589) defined in the Rust wasmtime host and the JS test runner; the JS runner's `actuate` only records in memory (no motor-bus write)
- ✅ OpenQASM 3.0 codegen — phic-emitted QASM run on IBM Heron r2 per project records (job `d7euddh5a5qc73drdosg`, ibm_fez, 1 qubit, 2026-04-14; job `d941s54ql68s73c909fg`, ibm_marrakesh, 3 qubits, 2026-07-03; CLAIMS.md C-10/C-24). **Producer-recorded, not independently confirmed:** no raw IBM job export is saved (receipt outside the repo: `D:\CosmicFamily\EVIDENCE\PHIFLOW_IBM_HERON_20260414.md`)
- ✅ Consciousness metrics — L_self, C_PF, R_in, R_out, D_int, C_coh
- ✅ SOMA sensor bridge — live telemetry
- ✅ Singularity daemon — persistent execution with state save/resume
- ✅ MCP server — stdio JSON-RPC for AI assistant integration
- ✅ Sacred geometry SVG generation — 6 patterns
- ✅ Topology-aware quantum transpilation — layout-aware qubit routing
- ✅ Metrics handoff — `phic --measure` appends to `/tmp/phiflow_daemon_metrics.jsonl` (only when metrics + self-correlation are present); the external bridge `/mnt/d/System/tools/phiflow_metrics_bridge.py` (not in this repo) serves it on :18030

## What's NOT Done Yet

- No hardware firmware generation (ESP32/P1 target)
- bio_compute is archived, not compiled (`src/_archive/speculative/bio_compute/`; `pub mod bio_compute` commented out in src/lib.rs)
- Type 4 observer status not established: CLAIMS.md C-21/C-23 PARTIAL-POSITIVE (observable self-model, scope-limited); natural actuator→sensor coupling falsified (R4, 2026-10-10); engineered-limb R5 did not close (OPEN, L_self=0.0 degenerate, 2026-11-07); ENGINEERED-SIGNAL lane CLOSED at engineered weight only (2026-11-07), never pooled with natural traces — see QSOP/STATE.md
- Self-correction loop on real hardware: `scripts/self_correction_real.py` written, no run recorded
