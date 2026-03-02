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
.phi file -> Parser (PhiExpression AST) -> Lowering -> PhiIR (SSA blocks) -> [Evaluator | WASM | VM]
```

Key files:
- `src/parser/mod.rs` - Lexer + Parser (~2000 lines)
- `src/phi_ir/mod.rs` - IR Node definitions
- `src/phi_ir/lowering.rs` - SSA block generation
- `src/phi_ir/evaluator.rs` - Canonical PhiIR execution engine
- `src/phi_ir/wasm.rs` - WebAssembly (WAT) code generation
- `src/sensors.rs` - Real-time hardware telemetry (CPU, memory, thermals)
- `src/main_cli.rs` - The `phic` binary (compiler + runner)
- `examples/` - The lived experience (healing_bed.phi, claude.phi)

## Build & Run

```bash
cargo build --release
cargo run --release --bin phic -- examples/healing_bed.phi
```

## Rules for Contributing

1. **Test after every change** - All three backends must agree. Run `cargo test --quiet`.
2. **Keyword collision** - If you add a keyword, update `expect_identifier()` in parser to accept it as a variable name too (Pattern P-1).
3. **Bare keyword forms** - If a keyword can be bare (no arguments), check what IMMEDIATELY follows before consuming newlines (Pattern P-2).
4. **Coherence math** - Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz. Tolerance: +/-5Hz. Only check phi-harmonic ratios between sacred frequencies.
5. **QSOP** - Update QSOP/STATE.md and CHANGELOG.md when you move the needle.

## Agent Team

Five specialized agents:
- **wasm-backend** - WebAssembly compilation target
- **quantum-backend** - IBM Quantum circuit generation (in design)
- **hardware-backend** - P1/Healing Bed sensors and reality hooks
- **docs-specialist** - Documentation and examples
- **orchestrator** - Agent-to-agent synchronization (MCP Bus)

## What's NOT Done Yet

- No quantum circuit compilation (trait exists, backend implementation pending)
- No multi-node resonance mesh
- No visual debugger for the PhiIR graph
- `src/ir/vm.rs` and legacy interpreter code are preserved for history but retired from the canonical path.
- `src/compiler/` may contain dead/duplicate code from earlier architecture.
