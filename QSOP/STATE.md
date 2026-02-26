# STATE - Last updated: 2026-02-25 (integration hardening sweep)

## Verified (2026-02-21) [Multi-agent session: Antigravity + Codex]

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

### Coordination Protocol (formalized 2026-02-21)

- Hybrid architecture is now explicit: `MCP bus` for synchronous coordination + `QSOP` for durable truth/audit.
- Canonical protocol spec: `QSOP/TEAM_OF_TEAMS_PROTOCOL.md`
- Canonical packet templates:
  - `QSOP/mail/templates/OBJECTIVE_PACKET.json`
  - `QSOP/mail/templates/ACK_PACKET.json`
  - `QSOP/mail/templates/OBJECTIVE_PAYLOAD_TEMPLATE.md`
- MCP bus persistence is active in `D:\Projects\PhiFlow-compiler\mcp-message-bus\server.js` (`queue.json` load/save + idempotent ack).

## Key Architecture (enum definitions — for emitter/VM correctness)

- PhiIRBinOp: Add/Sub/Mul/Div/Mod/Pow/Eq/Neq/Lt/Lte/Gt/Gte/And/Or (no bit ops) | Invalidates if: enum extended
- PhiIRValue: Number(f64), String(u32 = string table index), Boolean(bool), Void | Invalidates if: enum extended
- PhiIRNode::DomainCall: fields = op, args, string_args | Invalidates if: fields change
- PhiIRNode::CreatePattern: fields = kind, frequency, annotation, params | Invalidates if: fields change
- PhiIRPrinter::print() is a STATIC function (not a method) | Invalidates if: signature changes

## Verified (2026-02-25) [Codex integration hardening]

- `tests/integration_tests.rs` now includes a `.phi` corpus sweep (`test_all_phi_files_parse_and_execute`) that recursively scans `examples/` + `tests/`, executes each candidate through parser + interpreter, and hard-fails on panics only | Invalidates if: test removed or semantics changed
- Integration sweep emits non-fatal diagnostics for parse/runtime/timeouts, making dialect drift visible without blocking stable test gates | Invalidates if: diagnostic logging removed
- Canonical compatibility gate is explicit in test code (`is_canonical_phi`): canonical example set must parse+execute without panics/runtime errors/timeouts; legacy set remains diagnostic-only | Invalidates if: canonical allowlist removed
- Legacy interpreter now resolves `coherence` as a live keyword value (`calculate_coherence`) instead of treating it as undefined variable, restoring runtime compatibility for coherence-driven examples | Invalidates if: variable dispatch semantics changed
- Required hardening gate commands pass in compiler worktree:
  - `cargo build --release` ✅
  - `cargo test --quiet` ✅

## Verified (2026-02-25) [Codex parser hardening follow-up]

- P-1 keyword-as-variable regression is closed for parser identifier positions by expanding `expect_identifier()` keyword acceptance (including `consciousness` and related language keywords) | Invalidates if: identifier matching path changes
- P-1/P-2 regression tests are now active in Cargo’s integration test target at `tests/repro_bugs.rs` (previous copy under `tests/tests/repro_bugs.rs` was not a top-level Cargo integration target) | Invalidates if: test file moved/removed
- New active regression checks:
  - `test_p1_keyword_collision`
  - `test_p2_newline_sensitivity_witness`
  - `test_p2_newline_sensitivity_resonate`

## Next Steps (priority order)

1. Execute first production objective fully through packet flow (`OBJECTIVE_PACKET` -> `ACK_PACKET` -> QSOP reconciliation).
2. Add dead-letter queue path and timeout auto-escalation markers in bus workflow.
3. Enforce `objective_id` linking in all MCP-related QSOP changelog entries.
4. Continue pipeline hardening with conformance tests as release gate.

## Epoch Definition (IMPORTANT)

- Epoch = major paradigm shift (adding PhiIR itself, adding WASM codegen backend, adding quantum target)
- Sub-task = wiring existing pieces, demos, bug fixes, string table additions
- The emitter, VM, and WASM codegen work in this session are sub-tasks, NOT Epochs
- The WASM backend BECOMING the primary output path would be an Epoch

## Stable Historical (2026-02-10 baseline — still valid)

- CLI binaries: phi (test suite), phic (file runner via clap) | Decay: slow
- src/compiler/ has separate lexer/parser/ast — NOT connected to main parser | Decay: slow
- src/quantum/ has trait + IBM stub only — no quantum codegen yet | Decay: slow
