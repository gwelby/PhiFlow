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
.phi file -> Lexer (PhiToken) -> Parser (PhiExpression AST) -> Interpreter -> Output + Coherence Report
```

Key files:
- `src/parser/mod.rs` - Lexer + Parser (~1800 lines)
- `src/interpreter/mod.rs` - Tree-walking interpreter with coherence tracking
- `src/main.rs` - Test suite runner (binary: phi)
- `src/main_cli.rs` - File runner (binary: phic)
- `examples/` - Five working .phi programs

## Build & Run

```bash
cargo build --release
cargo run --release --bin phic -- examples/code_that_resonates.phi
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

## What's NOT Done Yet

- No intermediate representation (IR) - AST goes straight to interpreter
- No WASM codegen
- No quantum circuit compilation (trait exists, codegen doesn't)
- No hardware firmware generation
- No bytecode VM
- src/compiler/ and src/vm/ may contain dead/duplicate code from earlier architecture
