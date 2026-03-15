# CHANGELOG

## 2026-02-27 - [Codex] Phase 4 closeout patch set: serializable state, MCP stdio E2E, reality hooks

- ADDED: `src/phi_ir/vm_state.rs`
  - New serializable execution snapshot contract:
    - `VmState` (yield/resume state)
    - `VmWitnessEvent` (witness-log entry payload)
- UPDATED: `src/phi_ir/evaluator.rs`
  - `FrozenEvalState` now aliases serializable `VmState`.
  - `WitnessEvent` now aliases serializable `VmWitnessEvent`.
  - Yield/resume path remains backward-compatible while enabling state serialization.
- UPDATED: `src/phi_ir/mod.rs`
  - Exported `pub mod vm_state`.
  - `PhiIRValue` now derives `serde::Serialize` and `serde::Deserialize` to support persisted VM/evaluator state.
- UPDATED: `tests/phi_ir_evaluator_tests.rs`
  - Added `test_frozen_eval_state_roundtrips_through_json` to validate JSON serialize/deserialize and successful resume.
- ADDED: `tests/mcp_stdio_e2e_tests.rs`
  - True MCP transport-level E2E over stdio:
    - spawns `phi_mcp` binary,
    - performs `initialize`,
    - runs `spawn_phi_stream` -> `read_resonance_field` (yielded) -> `resume_phi_stream` -> `read_resonance_field` (completed).
- UPDATED: `src/sensors.rs`
  - Coherence mapping now blends:
    - CPU stability,
    - memory stability,
    - thermal stability (via `sysinfo::Components`),
    - network stability (via packet/error/traffic signals from `sysinfo::Networks`).
  - Includes graceful fallback weighting when thermal/network signals are unavailable.
- ADDED examples:
  - `examples/sync_rule.phi` (QDrive sync intent flow)
  - `examples/companion_loop.phi` (P1 companion witness/resonate loop)
- VERIFIED:
  - `cargo test --test phi_ir_evaluator_tests --test mcp_integration_tests --test mcp_stdio_e2e_tests --test concurrent_streams_tests -- --nocapture` ✅
  - `cargo run --release --bin phic -- examples/sync_rule.phi` ✅
  - `cargo run --release --bin phic -- examples/companion_loop.phi` ✅
  - `cargo test wasm_host -- --nocapture` ✅
- NOTE:
  - One earlier run in this session showed transient toolchain/resource instability (`E0463` and linker-format noise), but immediate rerun and final verification passed.

## 2026-02-26 - [Codex] Phase 3 realm execution: WASM Universal Bridge (`src/wasm_host.rs`)

- ADDED: `src/wasm_host.rs`
  - New native Rust WASM host bridge using `wasmtime` + `wat`.
  - Exposes source/WAT execution APIs:
    - `compile_source_to_wat(source)`
    - `run_source_with_host(source, hooks)`
    - `run_wat_with_host(wat_source, hooks)`
  - Implements host hook wiring for imported PhiFlow WASM consciousness hooks:
    - `phi.witness(i32) -> f64`
    - `phi.resonate(f64)`
    - `phi.coherence() -> f64`
    - `phi.intention_push(i32)`
    - `phi.intention_pop()`
  - Adds bridge-side contracts:
    - `WasmHostHooks` (custom coherence + lifecycle callbacks)
    - `WasmWitnessEvent`
    - `WasmHostSnapshot`
    - `WasmRunResult`
    - `WasmHostError`
- UPDATED: `Cargo.toml`
  - Added dependencies: `wasmtime`, `wat`.
- UPDATED: `src/lib.rs`
  - Exported `pub mod wasm_host`.
- ADDED tests in `src/wasm_host.rs`:
  - `wasm_host_uses_custom_coherence_provider`
  - `wasm_host_records_witness_and_resonate_events`
- VERIFIED:
  - `cargo test wasm_host -- --nocapture` ✅
  - `cargo build --release && cargo test` ✅

## 2026-02-26 - [Codex] Phase 2 realm execution: MCP convergence bus hardening

- UPDATED: `src/mcp_server/state.rs`
  - Added `shared_resonance: Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>` to `McpState`.
  - `McpState::new()` now initializes a process-wide shared resonance field for all spawned/resumed streams.
- UPDATED: `src/mcp_server/tools.rs`
  - `spawn_phi_stream` and `resume_phi_stream` now wire evaluators with `.with_shared_resonance(...)`.
  - `read_resonance_field` now reports the shared resonance snapshot (cross-stream visibility) rather than stream-local-only state.
  - Refactored tool helpers to reduce timing fragility in test interaction.
- UPDATED: `src/bin/phi_mcp.rs`
  - Added MCP protocol handshake support for `initialize`.
  - Added `ping` response path.
  - Added unit test `initialize_returns_tools_capability`.
- UPDATED: `tests/mcp_integration_tests.rs`
  - Replaced fixed-sleep checks with polling helpers (`wait_for_status`) for deterministic async behavior.
  - Added `test_mcp_shared_resonance_visible_across_streams` proving cross-stream resonance aggregation.
  - Tightened witness assertion to verify yielded `observed_value`.
- VERIFIED:
  - `cargo test --test mcp_integration_tests --bin phi_mcp -- --nocapture` ✅
  - `cargo build --release && cargo test` ✅

## 2026-02-26 - [Codex] Phase 1 realm hardening: host callbacks + witness yield correctness

- UPDATED: `src/host.rs`
  - `CallbackHostProvider` now supports full host hook coverage:
    - `with_intention_push(...)`
    - `with_intention_pop(...)`
  - This closes trait-level parity with `PhiHostProvider` and removes callback-only gaps for intention lifecycle observation.
- UPDATED: `src/phi_ir/evaluator.rs`
  - Added `VmExecResult` enum and kept `EvalExecResult` as backward-compatible alias.
  - Reworked witness execution path to eliminate duplicate `on_witness` host callback invocations.
  - Yielded witness snapshots now preserve `observed_value` from witness target operands.
  - `CoherenceCheck` now resolves through host contract (`resolve_coherence()`), preserving provider override semantics.
- UPDATED: `tests/phi_ir_evaluator_tests.rs`
  - Added `test_witness_callback_called_once_per_instruction`.
  - Added `test_witness_yield_preserves_observed_value_snapshot`.
  - Added `test_callback_host_receives_intention_push_and_pop`.
- VERIFIED:
  - `cargo test --test phi_ir_evaluator_tests --test mcp_integration_tests -- --nocapture` ✅
  - `cargo build --release && cargo test` ✅

## 2026-02-25 - [Codex] OBJ-20260225-001 agent protocol publication lane

- ADDED: `AGENT_PROTOCOL.json`
  - Machine-readable protocol contract for the five hooks:
    - `phi_witness`
    - `phi_resonate`
    - `phi_coherence`
    - `phi_intention_push`
    - `phi_intention_pop`
  - Includes canonical coherence formula and explicit `lambda = 0.618033988749895`.
  - Includes resonance field model, witness event schema, self-verification program, and canonical semantics reference.
- UPDATED: `README.md`
  - Added examples-table entry:
    - `agent_handshake.phi` — self-verifying protocol handshake for hook implementations.
- UPDATED: GitHub topics for discoverability (`gwelby/PhiFlow`):
  - `consciousness`, `webassembly`, `agent-protocol`, `phi`, `streaming`, `rust`
- VERIFIED:
  - `python -m json.tool AGENT_PROTOCOL.json` ✅
  - `gh api repos/gwelby/PhiFlow -q .topics` -> `["agent-protocol","consciousness","phi","rust","streaming","webassembly"]` ✅
  - `cargo test` ✅ (full suite passed)

## 2026-02-25 - [Codex] Canonical gate + coherence runtime compatibility

- UPDATED: `src/interpreter/mod.rs`
  - `PhiExpression::Variable("coherence")` now resolves to `calculate_coherence()` in legacy interpreter mode.
  - Fix closes runtime incompatibility for coherence-driven legacy examples (notably `examples/p1_demo.phi` and `examples/universalprocessor.phi`).
- UPDATED: `tests/integration_tests.rs`
  - Added explicit canonical allowlist (`is_canonical_phi`) and strict assertions for canonical parse+execute compatibility.
  - Retained non-fatal diagnostics for legacy/experimental files to keep drift visible without destabilizing CI.
  - Reduced non-canonical timeout budget to 5s for faster sweep feedback; canonical remains 30s.
- VERIFIED:
  - `cargo test --test integration_tests test_all_phi_files_parse_and_execute -- --nocapture` ✅
  - `cargo test --quiet` ✅
- Current sweep signal:
  - Canonical set: strict pass
  - Legacy drift remains parse-diagnostic only (12 files)

## 2026-02-25 - [Codex] Compiler hardening sweep gate for `.phi` corpus

- UPDATED: `tests/integration_tests.rs`
  - Added recursive `.phi` corpus collector across `examples/` and `tests/`.
  - ... (rest of the file)
