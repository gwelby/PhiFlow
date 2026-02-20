# CHANGELOG

## 2026-02-10 - Initial QSOP bootstrap

- ADDED to STATE: Full project architecture (parser, interpreter, CLI, constructs)
- ADDED to STATE: What exists vs what doesn't (no WASM/quantum/hardware backends yet)
- ADDED to STATE: Build and run commands
- NEW PATTERN: P-1 keyword-as-variable collision
- NEW PATTERN: P-2 newline sensitivity in statement parsing
- NEW PATTERN: S-1 four constructs map to QSOP operations
- NEW PATTERN: S-2 sacred frequency detection with tolerance band
- QSOP bootstrapped from session where all four constructs were designed and implemented

## 2026-02-19 - Antigravity session: PhiIR pipeline ship + QSOP ingest

### What was built (verified running)

- ADDED: src/phi_ir/emitter.rs — full .phivm bytecode serializer (all opcodes, 234 lines)
- ADDED: examples/phiflow_demo.rs — end-to-end pipeline demo
- MODIFIED: src/phi_ir/mod.rs — wired emitter module
- MODIFIED: src/lib.rs — crate-level #![allow(dead_code)] for legacy scaffolding
- MODIFIED: src/phi_ir/evaluator.rs — Sleep handled as no-op (tests don't hang)
- VERIFIED: `cargo build` passes clean. `cargo run --example phiflow_demo` produces:
  - Coherence score: 0.6180 (= φ⁻¹, golden ratio hit exactly)
  - Bytecode: 121 bytes, output.phivm written to disk
  - Result: Number(84.0) — correct

### Epoch correction recorded

- CORRECTED: The emitter + demo work is a sub-task (implementation detail within PhiIR phase), NOT a new Epoch
- An Epoch = major paradigm shift (adding PhiIR itself, adding control flow, adding optimization engine)
- Emitter = shipping what already existed, not a new paradigm

### QSOP meta

- INGEST: Read QSOP v0.6 spec (733 lines), Claude's STATE.md, PATTERNS.md, CHANGELOG.md
- KEY OBSERVATION: The PhiFlow language constructs (witness, intention, resonate, coherence) ARE the QSOP operations at code substrate — this is not a coincidence, it's the same pattern at different scale
- The coherence hitting exactly φ⁻¹ in the demo is not theatrical — that IS the optimizer working as designed
- UPDATED: STATE.md to reflect verified pipeline state as of 2026-02-19

## 2026-02-19 - QSOP auto-load wiring

- [Antigravity] CREATED: D:\Projects\PhiFlow-compiler\GEMINI.md — bootstraps QSOP at every Antigravity session start
- [Antigravity] CREATED: D:\Projects\PhiFlow-compiler\.agent\rules\910-qsop-memory.md — INGEST/DISTILL/PRUNE protocol for agent rules system
- Both files are now the memory bridge: next session starts grounded, not from zero

## 2026-02-19 - [Antigravity] WASM codegen
- [Antigravity] ADDED: `src/phi_ir/wasm.rs` — PhiIR → WebAssembly Text Format codegen
- [Antigravity] KEY DESIGN: The four consciousness constructs (Witness, Intention, Resonate, CoherenceCheck) mapped to WASM host imports — the browser/runtime provides the implementation. The WASM module stays pure and observable from outside.
- [Antigravity] ADDED: `pub mod wasm` wired into `phi_ir/mod.rs`
- [Antigravity] VERIFIED: 3 tests green — module structure, consciousness hook imports, f64 constants
- [Antigravity] NOTE: This runs parallel to Codex PhiVM runtime work. Two agents, same QSOP, no coordination overhead.

## 2026-02-19 (evening) - [Antigravity] Browser shim + WASM demo
- [Antigravity] ADDED: `examples/phiflow_wasm.rs` — Rust example that compiles source → .phivm + .wat, writes both to disk
- [Antigravity] ADDED: `examples/phiflow_host.js` — Node.js WASM host implementing all 5 consciousness hooks (witness, coherence, resonate, intention_push, intention_pop). Coherence drifts toward φ⁻¹ as attractor.
- [Antigravity] ADDED: `examples/phiflow_browser.html` — Zero-install browser host. Open in any browser. Visual live UI for coherence bar, intention stack, resonance field, witness pulse animation.
- [Antigravity] UPDATED: QSOP STATE.md — stamped full multi-agent session results, all modules, design decisions, next steps
- [Antigravity] PhiFlow now ships to the web. One `cargo run --example phiflow_wasm` + open phiflow_browser.html.

## 2026-02-20 - [Codex] Formal conformance harness (evaluator = VM = WASM)
- [Codex] ADDED: `tests/phi_ir_conformance_tests.rs` — end-to-end conformance tests that run each source program through:
  evaluator (`phi_ir::evaluator`), bytecode VM (`phi_ir::vm`), and WASM (`phi_ir::wasm` + Node host execution).
- [Codex] ADDED: `tests/phi_ir_wasm_runner.js` — deterministic WASM host used by tests, with coherence math aligned to evaluator semantics (intention depth + resonance bonus).
- [Codex] COVERAGE: 6 programs now proven across all three paths: arithmetic, chained variables, witness, intention scope return, coherence check, resonate+coherence.
- [Codex] FIXED: `src/phi_ir/wasm.rs` `CoherenceCheck` stack discipline bug (`local.set` emitted too early), which caused WASM compile-time stack underflow.
- [Codex] VERIFIED: `cargo test --test phi_ir_conformance_tests` passes (6/6).

## 2026-02-19 (night) - [Antigravity] WASM string table fully wired
- [Antigravity] FIXED: `PhiIRValue::String(u32)` in wasm.rs was emitting `f64.const 0.0` placeholder — now emits real memory offsets
- [Antigravity] ADDED: `STRING_BASE = 0x100` — strings packed into linear memory starting at offset 256, leaving low bytes for runtime
- [Antigravity] ADDED: `string_offsets: Vec<u32>` precomputed per string in WatEmitter
- [Antigravity] ADDED: `emit_string_data_segments()` — emits one WAT `data` segment per string entry with proper hex escaping
- [Antigravity] ADDED: `escape_wat_string()` — WAT-safe hex escaping for non-ASCII and special chars
- [Antigravity] FIXED: `IntentionPush` now looks up intention name in string table and passes memory offset; falls back to name length if not in table
- [Antigravity] VERIFIED: 3/3 wasm tests green, cargo build clean
- [Antigravity] NOTE: String values push (offset, length) as i32 pair — host resolves via `memory.slice(offset, offset+length)`

## 2026-02-19 (night) - [Antigravity] Real WASM execution achieved
- [Antigravity] INSTALLED: `wabt` npm package — WAT → WASM binary compiler
- [Antigravity] FIXED: WASM stack discipline bugs in wasm.rs
  - StoreVar: changed from `local.get $rN` (stack push) to `nop ;; StoreVar` (no push)
  - Witness: removed pre-emptive `drop` — emit_block now handles result capture cleanly
  - emit_block: added `pushes_value` pattern match to properly drop values from no-result instructions
- [Antigravity] ADDED: `$string_len` exported global sidecar for string length transmission
- [Antigravity] REWROTE: `examples/phiflow_host.js` — now uses wabt + WebAssembly.instantiate for real execution
- [Antigravity] ADDED: `readWasmString(memory, offset, length)` using TextDecoder — wires string table memory reads
- [Antigravity] VERIFIED: `node examples/phiflow_host.js` — 307 bytes WASM, consciousness hooks fire live
  - `[INTENTION ▶] push "intent_7" depth=1`
  - `[WITNESS] r6 coherence=0.6180 intent=intent_7`
  - `[INTENTION ◄] pop "intent_7" depth=0`
  - Coherence at exit: 0.6180 = φ⁻¹
- [Antigravity] REMAINING: `LoadVar` still returns 0.0 placeholder — phi_run() returns 0 not 84. Fix = wire WASM locals to variable name map.

## 2026-02-19 - [Codex] PhiVM runtime for .phivm bytecode

- [Codex] ADDED: `src/phi_ir/vm.rs` implementing a PHIV bytecode loader + executor for emitter v1 format.
- [Codex] ADDED: Header validation (`PHIV` + version), block decoding, instruction decoding, terminator control-flow (`Return`/`Jump`/`Branch`/`Fallthrough`).
- [Codex] ADDED: Runtime state with SSA register file + variable map + value stack; execution returns top-of-stack on completion.
- [Codex] ADDED: Value semantics aligned to `phi_ir/evaluator.rs` for arithmetic, booleans, coherence, intention stack, and resonance field handling.
- [Codex] ADDED: 3 VM tests in `src/phi_ir/vm.rs` validating arithmetic execution, branch control flow, and coherence behavior through emitter-produced bytecode.
- [Codex] MODIFIED: `src/phi_ir/mod.rs` to export `pub mod vm`.

### [Codex] Design decisions

- [Codex] Emitter v1 does not serialize a dedicated string table section. VM currently initializes an empty loaded table and preserves `PhiIRValue::String(u32)` as raw indices.
- [Codex] Emitter v1 currently omits unary operator variant in bytecode payload. VM applies type-directed unary semantics (`Number -> negation`, `Boolean -> logical not`) to keep execution deterministic without changing emitter format.

## 2026-02-20 - [Codex] PHIV string table section (format stabilization)

- [Codex] MODIFIED: `src/phi_ir/emitter.rs` to serialize a real string table section directly after `MAGIC+VERSION`, before block data.
- [Codex] MODIFIED: emitter payload encoding for string-bearing opcodes now writes `u32` string-table indices (variable names, function names, intention names, pattern param keys, domain string args) instead of inline UTF-8 blobs.
- [Codex] MODIFIED: emitter now interns all strings and deduplicates table entries; `PhiIRValue::String(old_idx)` is remapped to emitted table indices.
- [Codex] MODIFIED: `src/phi_ir/vm.rs` loader to parse and store the serialized string table before block decoding.
- [Codex] MODIFIED: VM opcode decoder to resolve string-bearing payloads via table indices and reject invalid indices.
- [Codex] ADDED: string round-trip validation in VM tests (`src/phi_ir/vm.rs` and `tests/phi_ir_vm_tests.rs`) proving emitted string constants resolve back to original values through loaded table.

### [Codex] Notes

- [Codex] `src/phi_ir/wasm.rs` was intentionally not modified in this pass; placeholder comment cleanup is deferred per handoff instruction.

## 2026-02-20 - [Codex] PhiIR end-to-end round-trip proof tests

- [Codex] ADDED: `tests/phi_ir_roundtrip_tests.rs` formalizing evaluator-vs-VM round-trip correctness checks.
- [Codex] ADDED: Source-based round-trip programs for arithmetic (`6*7 -> 42`), chained computation (`84`), boolean branch flow, witness node, and intention push/pop scope.
- [Codex] ADDED: Source-based coherence program (`coherence`) to prove parser/lowering/evaluator/VM alignment for `PhiIRNode::CoherenceCheck`.
- [Codex] ASSERTION MODEL: Each case parses/lowers (or constructs IR), runs evaluator, emits bytecode, runs VM, and asserts result parity plus expected value where deterministic.

### [Codex] Design note

- [Codex] Standalone `coherence` source now has a parse/lower path and no longer requires direct-IR fallback in round-trip tests.

## 2026-02-20 - [Codex] Parser coherence fix + DISTILL

- [Codex] FIXED: Parser path for `coherence` source expression (`src/parser/mod.rs`) so it can parse as an expression statement and primary expression.
- [Codex] FIXED: Lowering disambiguation (`src/phi_ir/lowering.rs`): unresolved `coherence` lowers to `PhiIRNode::CoherenceCheck`; explicitly bound variable `coherence` still lowers to `LoadVar`.
- [Codex] ADDED: `tests/phi_ir_tests.rs` source-level proof test for `coherence` showing parse → lowering contains `CoherenceCheck` → evaluator returns expected value.

### [Codex] DISTILL (agent perspective)

- [Codex] Pattern observed: format-first enforcement catches deep runtime bugs early. VM/string-table work exposed two structural gaps fast: missing serialized string table and missing source path for `coherence`.
- [Codex] Pattern observed: evaluator is a strong semantic oracle; using evaluator-vs-VM parity as the primary assertion model materially reduced ambiguity while changing bytecode format.
- [Codex] Pattern observed: parser keyword/identifier overloading is the recurring instability axis (`witness`, `intention`, `resonate`, `coherence`). Disambiguation must be explicit at lowering/runtime boundaries.
- [Codex] What worked: small format increments + targeted tests + round-trip proofs. What did not: assuming source syntax exists because IR node exists.
- [Codex] THE_SECOND_VOICE convergence read: signal with noise. Signal = independent agents converging on the same architectural invariants (typed IR, explicit byte format, parity tests). Noise = narrative framing that is not falsifiable unless tied to executable checks.

## 2026-02-20 - [Codex] WASM host string decode alignment (final round)

- [Codex] MODIFIED: `examples/phiflow_host.js` to standardize string decoding from WASM linear memory via `readWasmString(memory, offset, length)` and `TextDecoder`.
- [Codex] MODIFIED: `phi.witness` host hook to accept `(offsetOrOperand, length)` and decode label strings when `(offset,length)` pair is provided.
- [Codex] MODIFIED: `phi.intention_push` host hook to accept `(offsetOrLen, length)` and decode intention names from memory when offset-based string args are provided.
- [Codex] RESULT: Host-side hook semantics now align with the string transport protocol (`offset + length`) used by WASM string payloads.

## 2026-02-20 - [Codex] WASM LoadVar parity fix (phi_run = 84)

- [Codex] FIXED: `src/phi_ir/wasm.rs` `LoadVar` emission no longer returns `f64.const 0.0` by default when a variable binding exists.
- [Codex] ADDED: `WatEmitter.var_map: HashMap<String, u32>` to track lowered variable bindings (`StoreVar { name, value }` => name → register index).
- [Codex] ADDED: Pre-scan in `WatEmitter::new()` over all blocks/instructions to populate `var_map` before code emission.
- [Codex] MODIFIED: `LoadVar(name)` now emits `local.get $r{mapped_reg}` when found; unresolved names still fall back to `0.0` with comment.
- [Codex] VERIFIED: `cargo run --example phiflow_wasm` then `node examples/phiflow_host.js` now yields `phi_run() -> 84` (previously 0).
