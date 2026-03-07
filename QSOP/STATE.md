# STATE - Last updated: 2026-03-06 (Cross-Worktree Reality Check + Council Activation)

## Verified (2026-03-06) [Cross-worktree reality check]

- `D:\Projects\PhiFlow\PhiFlow` (`master` inner crate) is currently the most stable directly runnable lane:
  - `cargo test --quiet` passed on 2026-03-06 with warnings only
  - This supersedes the older note that `tests/performance_tests.rs` was still failing on `master`
- `D:\Projects\PhiFlow-compiler\PhiFlow` (`compiler` inner crate) remains the most advanced runtime lane, but it is not currently green end-to-end:
  - `cargo test --quiet --lib --tests` now fails at `tests/phi_ir_conformance_tests.rs::conformance_witness` with evaluator/WASM mismatch (`lhs=0`, `rhs=NaN`)
  - Full `cargo test --quiet` fails earlier while compiling multiple examples that import `phiflow`, and also reports missing `rlib` forms for several dependencies
  - This means the older claim that compiler-lane `cargo test --quiet` passes is stale
- Worktree hygiene snapshot (2026-03-06):
  - `master` is dirty in docs/protocol/bridge files (`CLAUDE_ARCHITECTURE_REVIEW.md`, `PHIFLOW_WORKSPACE_INDEX.md`, `bridges/web/*`, `examples/browser_shim.js`, etc.)
  - `compiler` is dirty in runtime/test/QSOP files and is an active integration lane rather than merge-ready state
  - `cleanup` and `language` are clean worktrees with only minimal branch-specific delta from the shared history
- Branch divergence snapshot (2026-03-06):
  - `master...compiler` = 11 / 12 unique commits
  - `master...cleanup` = 23 / 1 unique commits
  - `master...language` = 23 / 1 unique commits
- Operational truth as of 2026-03-06:
  - Stable execution/demo lane: `master`
  - Advanced but repair-needed runtime lane: `compiler`
  - Available capacity lanes for structured work: `cleanup`, `language`

## Council Activation (2026-03-06)

- `Codex` -> `D:\Projects\PhiFlow-compiler`
  - Mission: restore end-to-end green runtime status
  - Done condition: `cargo test --quiet` passes again, starting with `phi_ir_conformance_tests::conformance_witness` and example-target compile failures
- `Lumi` -> `D:\Projects\PhiFlow-lang`
  - Mission: own browser/WASM-facing language evolution only after ABI review against compiler truth
  - Done condition: new syntax/features land with docs/examples/tests, and browser-host assumptions match the runtime contracts Codex closes
- `Qwen` -> `D:\Projects\PhiFlow-cleanup`
  - Mission: reduce outer-repo entropy and leave a zero-search map of what is real vs dead
  - Done condition: `TRIAGE.md` + `STRUCT.md` are complete and the sprawl outside `PhiFlow/` is categorized for keep/archive/remove
- `Witness / docs lane` -> `D:\Projects\PhiFlow`
  - Mission: keep `QSOP/STATE.md`, `KNOW.md`, and `VISION.md` aligned with executable truth before merge activity
  - Done condition: all status docs reflect current branch reality and local dirty changelog work is reconciled intentionally, not implicitly

## Verified (2026-02-28) [Antigravity Phase 5: MCP Bus Guardrails]

- `phi_mcp` now enforces configurable execution guardrails via `McpConfig`:
  - `max_execution_steps` (default: 10,000) via `EvalError::StepLimitExceeded` — clean error, no crash
  - `timeout_ms` (default: 5,000) via `tokio::time::timeout` on all three eval paths in `tools.rs`
  - Both configurable at runtime via `PHI_MAX_STEPS`, `PHI_TIMEOUT_MS`, `MCP_QUEUE_PATH` env vars
- `McpHostProvider` now implements `broadcast` / `listen` with atomic file I/O against Codex's `queue.json` (tmp→rename) | Invalidates if: queue path or Codex persistence format changes
- Cross-agent round-trip verified: `tests/cross_agent_roundtrip.js --simulate` passed full send→persist→ack→changelog cycle in <2s
- `BusMessage` struct in `state.rs` is now the canonical packet type matching Codex's queue schema
- Verification gates passed:
  - `cargo check --bin phi_mcp` → clean compile
  - `node tests/mcp_guardrails_test.js` → `StepLimitExceeded(50)` caught in <500ms
  - `node tests/cross_agent_roundtrip.js --simulate` → full round-trip logged to CHANGELOG
  - `node tests/dlq_test.js` → `ttl_s` timeouts successfully trigger auto-escalation to DLQ and write `UNRECONCILED` to CHANGELOG in <5s

## Verified (2026-02-27) [Codex Phase 4 closeout]

- Yield/resume state now has an explicit serializable contract in `src/phi_ir/vm_state.rs`:
  - `VmState` for snapshot persistence
  - `VmWitnessEvent` for witness log payloads | Invalidates if: state schema changes
- Evaluator yield flow now uses the serializable contract through aliases (`FrozenEvalState` -> `VmState`, `WitnessEvent` -> `VmWitnessEvent`) while preserving compatibility for existing call sites | Invalidates if: aliasing or evaluator state fields change
- `PhiIRValue` now derives serde traits so VM/evaluator state can be serialized to/from JSON safely | Invalidates if: value enum changes without serde compatibility
- State round-trip regression is now explicit in `tests/phi_ir_evaluator_tests.rs::test_frozen_eval_state_roundtrips_through_json` | Invalidates if: test removed
- MCP has a true transport-level stdio E2E test at `tests/mcp_stdio_e2e_tests.rs` that drives `phi_mcp` over JSON-RPC (`initialize` -> `spawn` -> `read` -> `resume` -> `read`) | Invalidates if: MCP protocol router changes
- Reality-hook coherence mapping now blends CPU, memory, thermal, and network inputs in `src/sensors.rs` with fallback weighting on unsupported hosts | Invalidates if: sensor fusion logic changes
- Phase 4 draft scripts now exist:
  - `examples/sync_rule.phi` (QDrive sync flow)
  - `examples/companion_loop.phi` (P1 companion loop flow)
- Verification gates passed in this session:
  - `cargo test --test phi_ir_evaluator_tests --test mcp_integration_tests --test mcp_stdio_e2e_tests --test concurrent_streams_tests -- --nocapture`
  - `cargo run --release --bin phic -- examples/sync_rule.phi`
  - `cargo run --release --bin phic -- examples/companion_loop.phi`
  - `cargo test wasm_host -- --nocapture`
- Verification caveat:
  - One earlier run showed transient toolchain/resource instability (`E0463` and linker-format noise), but rerun in the same session passed.

## Verified (2026-02-26) [Codex Phase 3: WASM Universal Bridge]

- Native Rust WASM host bridge now exists at `src/wasm_host.rs` and runs PhiFlow-generated WAT using `wasmtime` runtime bindings | Invalidates if: module removed or runtime backend swapped
- Bridge supports configurable hook callbacks via `WasmHostHooks` for coherence, witness, resonate, intention push/pop lifecycle | Invalidates if: hook contract changes
- Bridge returns structured execution artifacts (`WasmRunResult`, `WasmHostSnapshot`, `WasmWitnessEvent`) to make WASM runs observable without Node/browser-only tooling | Invalidates if: result contract changes
- Library exports `pub mod wasm_host` for direct integration into bridge/server layers | Invalidates if: module export removed
- Dependency baseline now includes `wasmtime` + `wat` for native WAT parse and WASM execution | Invalidates if: dependency set changed
- Verification gates passed:
  - `cargo test wasm_host -- --nocapture`
  - `cargo build --release && cargo test`

## Verified (2026-02-26) [Codex Phase 2: MCP convergence bus hardening]

- MCP runtime state now includes a process-wide shared resonance map (`McpState.shared_resonance`) used by spawned and resumed evaluator instances | Invalidates if: MCP state contract changes
- MCP tool execution now wires `Evaluator::with_shared_resonance(...)`, enabling cross-stream resonance visibility through `read_resonance_field` | Invalidates if: MCP tool wiring is reverted
- MCP server binary now handles protocol-level `initialize` and `ping` requests in addition to `tools/list` and `tools/call` | Invalidates if: request router changes
- MCP integration tests now use status polling instead of fixed sleeps, reducing async timing fragility in CI/local runs | Invalidates if: tests return to fixed-delay synchronization
- New regression coverage confirms cross-stream aggregation:
  - `tests/mcp_integration_tests.rs::test_mcp_shared_resonance_visible_across_streams`
  - `tests/mcp_integration_tests.rs::test_mcp_spawn_and_read` (yield + observed value)
  - `src/bin/phi_mcp.rs::initialize_returns_tools_capability`
- Verification gates passed after Phase 2 changes:
  - `cargo test --test mcp_integration_tests --bin phi_mcp -- --nocapture`
  - `cargo build --release && cargo test`

## Verified (2026-02-26) [Codex Phase 1: Core VM Disentanglement hardening]

- `CallbackHostProvider` now supports intention lifecycle callbacks (`with_intention_push`, `with_intention_pop`) in addition to coherence/resonate/witness hooks, bringing closure-based providers to trait parity with `PhiHostProvider` | Invalidates if: host callback API changes
- Evaluator witness yield flow now invokes `host.on_witness(...)` exactly once per witness instruction in yield-capable execution path (`run_or_yield`), removing duplicate side effects during MCP-hosted runs | Invalidates if: witness dispatch path is refactored
- Yielded witness snapshots now preserve `observed_value` from witness target operands instead of dropping it to `None` | Invalidates if: witness snapshot schema changes
- Execution result naming now exposes `VmExecResult` with `EvalExecResult` kept as compatibility alias, so existing MCP/integration call sites remain stable while Phase 1 terminology converges | Invalidates if: execution result enum contract changes
- Verification gates passed after Phase 1 hardening:
  - `cargo test --test phi_ir_evaluator_tests --test mcp_integration_tests -- --nocapture`
  - `cargo build --release && cargo test`

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
