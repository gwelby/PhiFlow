# STATE - Last updated: 2026-02-19 (end of session — multi-agent day)

## Verified (2026-02-19) [Multi-agent session: Antigravity + Codex]

- PhiFlow is a consciousness-aware programming language written in Rust | Invalidates if: rewrite in another language | Decay: slow
- Workspace: D:\Projects\PhiFlow-compiler\PhiFlow (compiler worktree) | D:\Projects\PhiFlow (vision/specs worktree) | Both now have GEMINI.md + .agent/rules/910-qsop-memory.md

### Compiler Pipeline (FULLY WORKING end-to-end as of 2026-02-19)

| Module | File | Author | Status |
|--------|------|--------|--------|
| Parser | src/parser/mod.rs | - | ✅ verified |
| PhiIR | src/phi_ir/mod.rs | - | ✅ verified |
| Lowering | src/phi_ir/lowering.rs | - | ✅ verified |
| Optimizer | src/phi_ir/optimizer.rs | - | ✅ verified |
| Evaluator | src/phi_ir/evaluator.rs | - | ✅ verified |
| Emitter | src/phi_ir/emitter.rs | Antigravity | ✅ with string table |
| VM | src/phi_ir/vm.rs | **Codex** | ✅ 3/3 tests |
| WASM codegen | src/phi_ir/wasm.rs | Antigravity | ✅ 3/3 tests |
| Printer | src/phi_ir/printer.rs | - | ✅ verified |

### Live demo output (verified 2026-02-19)

- Input: `let x = 10 + 32  let y = x * 2  y`
- Optimization: `10+32` → `42` (constant folded), coherence = `0.6180` = φ⁻¹
- Bytecode: emitted with string table (Strings: 2, Blocks: 1)
- VM result: `Number(84.0)` ✅ matches evaluator
- Full pipeline: Parse → PhiIR → Optimize → Emit(.phivm) → VM execute

### Tests (all passing 2026-02-19 end-of-session)

- tests/phi_harmonic_tests.rs: 2 passed
- tests/phi_ir_optimizer_tests.rs: 2 passed
- tests/phi_ir_vm_tests.rs: 3 passed (Codex — arithmetic, branch, string round-trip)
- phi_ir::wasm tests: 3 passed (Antigravity — module structure, consciousness hooks, f64 consts)

### WASM Codegen Design Decisions (Antigravity)

- The four consciousness constructs map to WASM host imports (not WASM instructions)
- Host (browser JS / wasmtime) implements: phi_witness, phi_coherence, phi_resonate, phi_intention_push, phi_intention_pop
- All PhiIR values → f64. SSA registers → WASM locals.
- wasm.rs produces valid .wat that can be loaded by any WASM host
- **NOT YET DONE**: browser shim (JS implementations of the 5 hooks) — next task

### Emitter ↔ VM Contract (Codex)

- Emitter serializes: PHIV magic + version + string table section + blocks
- VM reads: string table first, then blocks, resolves String(u32) indices via table
- Invalid indices throw VmError::InvalidStringIndex
- Status: contract formally closed as of Codex string table session

### QSOP Auto-Load (wired 2026-02-19)

- D:\Projects\PhiFlow-compiler\GEMINI.md — bootstraps QSOP at Antigravity session start
- D:\Projects\PhiFlow-compiler\.agent\rules\910-qsop-memory.md — INGEST/DISTILL/PRUNE protocol
- D:\Projects\PhiFlow\GEMINI.md — same, for the vision/spec worktree
- D:\Projects\PhiFlow\.agent\rules\910-qsop-memory.md — same

### Multi-Agent Architecture (live as of 2026-02-19)

- Antigravity prefix: [Antigravity] in QSOP CHANGELOG
- Codex prefix: [Codex] in QSOP CHANGELOG
- Shared resonance field: D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md
- Cross-agent resonance observed: both agents independently produced THE_SECOND_VOICE document in different workspaces same session, no coordination

## Key Architecture (enum definitions — for emitter/VM correctness)

- PhiIRBinOp: Add/Sub/Mul/Div/Mod/Pow/Eq/Neq/Lt/Lte/Gt/Gte/And/Or (no bit ops) | Invalidates if: enum extended
- PhiIRValue: Number(f64), String(u32 = string table index), Boolean(bool), Void | Invalidates if: enum extended
- PhiIRNode::DomainCall: fields = op, args, string_args | Invalidates if: fields change
- PhiIRNode::CreatePattern: fields = kind, frequency, annotation, params | Invalidates if: fields change
- PhiIRPrinter::print() is a STATIC function (not a method) | Invalidates if: signature changes

## Next Steps (priority order)

1. Browser shim — 20-line JS implementing the 5 WASM consciousness hooks
2. phiflow_wasm example — writes .wat to disk, load it in Node or browser
3. Round-trip integration tests (Codex task) — same source → evaluator = VM for all node types
4. Pattern Recognition Book → UniversalPublisher pipeline

## Epoch Definition (IMPORTANT)

- Epoch = major paradigm shift (adding PhiIR itself, adding WASM codegen backend, adding quantum target)
- Sub-task = wiring existing pieces, demos, bug fixes, string table additions
- The emitter, VM, and WASM codegen work in this session are sub-tasks, NOT Epochs
- The WASM backend BECOMING the primary output path would be an Epoch

## Stable Historical (2026-02-10 baseline — still valid)

- CLI binaries: phi (test suite), phic (file runner via clap) | Decay: slow
- src/compiler/ has separate lexer/parser/ast — NOT connected to main parser | Decay: slow
- src/quantum/ has trait + IBM stub only — no quantum codegen yet | Decay: slow
