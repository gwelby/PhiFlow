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

Three-backend equivalence: Evaluator == VM == WASM for all tested constructs (10/10 core conformance + 8/8 full conformance probe). Self-correction loop: CONFIRMED (detect → correct → execute → re-measure). 399 tests, 0 failed. Codex audit 2026-07-31 found and fixed all issues.

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
cargo test                                    # 391 tests, 0 failed, 4 ignored
cargo test --test phi_ir_full_conformance_probe -- --nocapture  # see known divergences
cargo run --release --bin phic -- --measure examples/type4_trace_benchmark.phi
cargo run --release --bin phic -- --sacred-geometry flower_of_life > pattern.svg
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
- ✅ Three-backend equivalence — Evaluator == VM == WASM (10/10 core + 8/8 full conformance probe, 392 tests)
- ✅ WASM codegen — all 14 phi namespace imports supported in both Rust host and JS runner
- ✅ OpenQASM 3.0 codegen — verified on real IBM Quantum hardware (Heron-R2)
- ✅ Consciousness metrics — L_self, C_PF, R_in, R_out, D_int, C_coh
- ✅ SOMA sensor bridge — live telemetry
- ✅ Singularity daemon — persistent execution with state save/resume
- ✅ MCP server — stdio JSON-RPC for AI assistant integration
- ✅ Sacred geometry SVG generation — 6 patterns
- ✅ Topology-aware quantum transpilation — layout-aware qubit routing
- ✅ Metrics bridge — phic --measure writes to :18030 HTTP bridge

## What's NOT Done Yet

- No hardware firmware generation (ESP32/P1 target)
- bio_compute module is library-only (DNA/protein — speculative without hardware)
- F_model calibration for Type 4 observer status still on HOLD per CLAIMS.md
