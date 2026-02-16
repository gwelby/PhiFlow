# STATE - Last updated: 2026-02-10

## Verified (2026-02-10)
- PhiFlow is a consciousness-aware programming language written in Rust (~17K lines) | Invalidates if: rewrite in another language
- Four unique constructs WORKING: witness, intention, resonate, live coherence | Invalidates if: any construct breaks
- Parser: src/parser/mod.rs (lexer + parser -> PhiExpression AST) | Invalidates if: parser refactored to separate files
- Interpreter: src/interpreter/mod.rs (tree-walking, coherence tracking, resonance field) | Invalidates if: replaced by bytecode VM
- CLI: two binaries - `phi` (test suite), `phic` (file runner via clap) | Invalidates if: binaries change
- Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz | Invalidates if: frequency list changes
- Coherence scoring: 0.0-1.0, ALIGNED/DRIFTING/MISALIGNED states | Invalidates if: thresholds change
- Resonance field: HashMap<String, Vec<PhiValue>> shared between intention blocks | Invalidates if: data structure changes
- Five example programs in examples/ all passing | Invalidates if: examples break
- LANGUAGE.md documents all four unique constructs | Invalidates if: constructs change
- Comments supported (// line comments only) | Invalidates if: block comments added
- No WASM backend yet - exists only as agent definition | Invalidates if: wasm module created
- No quantum compilation yet - src/quantum/ has trait + IBM stub only | Invalidates if: quantum codegen added
- No hardware compilation yet - src/hardware/ has consciousness detection only | Invalidates if: firmware codegen added
- src/compiler/ has separate lexer/parser/ast - NOT connected to main parser | Invalidates if: unified
- Built with `cargo build --release`, runs with `cargo run --release --bin phic -- file.phi` | Invalidates if: build changes

## Probable (2026-02-10)
- src/vm/interpreter.rs may duplicate src/interpreter/mod.rs functionality | Check: compare both files
- src/compiler/ may be dead code from earlier architecture | Check: see if anything imports it
- Cargo.toml dependencies include quantum/async crates that may not all be needed | Check: cargo tree

## Key Architecture
- Lexer: PhiToken enum with keywords (Witness, Intention, Resonate, Frequency, State, etc.)
- Parser: recursive descent, PhiExpression enum (50+ variants)
- Interpreter: PhiInterpreter struct with coherence tracking, intention stack, resonance field
- No IR/bytecode layer yet - AST directly interpreted
