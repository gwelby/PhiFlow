# PATTERNS - Learning from mistakes and successes

## Active Patterns (Mistakes)

### P-1: Keyword-as-variable collision

- **What happens**: PhiFlow keywords (frequency, state, coherence, etc.) used as variable names cause parse errors because lexer emits keyword tokens, not Identifier
- **Instances**: 3 (frequency/create, witness/resonate identifiers, `consciousness` identifier regression)
- **Root cause**: Lexer is greedy with keyword matching. No context-sensitive tokenization.
- **Fix**: In parser, `expect_identifier()` now accepts the full keyword token set when in identifier position. Regression coverage is in `tests/repro_bugs.rs::test_p1_keyword_collision`.
- **Invalidates if**: Lexer redesigned with context-sensitive modes
- **Promoted to STATE**: Yes

### P-2: Newline sensitivity in statement parsing

- **What happens**: Bare keywords (witness, resonate) that take optional arguments consume newlines before checking if they're bare, accidentally eating the next statement's token
- **Instances**: 2 (witness + resonate bare-form newline handling)
- **Root cause**: skip_newlines() called before checking for bare form
- **Fix**: Check what IMMEDIATELY follows the keyword before consuming any whitespace. If newline/EOF/RightBrace -> bare form. Regression coverage is in `tests/repro_bugs.rs::{test_p2_newline_sensitivity_witness,test_p2_newline_sensitivity_resonate}`.
- **Invalidates if**: Semicolons added as statement terminators
- **Promoted to STATE**: Yes

### P-3: WASM generated missing loop back-edges semantic signals

- **What happens**: WASM compilation execution strips stream loop back-edges, converting streams into stateless single-pass functions that never emit a natively detectable `break stream` signal. Python stream looping hangs infinitely.
- **Instances**: 1 (Phase 10 Lane B testing - 2026-02-23)
- **Root cause**: Host stream executor was trusting WASM payload to self-report broken execution.
- **Fix**: P1Host stream wrapper must artificially cap cyclic execution (e.g. limit to 3 cycles) to prevent evaluation timeouts and manually yield `stream_broken = True` when limit reached.
- **Invalidates if**: WASM lowering protocol is rewritten to perfectly embed break signals into memory limits or specific function yields.
- **Promoted to STATE**: No

### P-4: Cross-Agent Cargo Lock Contention (`test` and `run` deadlocks)

- **What happens**: Agents executing pytest test suites hang indefinitely (e.g., waiting 70+ seconds) waiting for rust builds because another agent (like Codex) holds the standard `target/` compilation lock.
- **Instances**: 1 (Phase 10 Lane B - 2026-02-23)
- **Root cause**: Standard `subprocess` or generic cargo commands use same debug target directory as long-running worker processes in the same workspace.
- **Fix**: Inject `CARGO_TARGET_DIR="target-antigravity"` into the environment payload before issuing compilation or test shell commands.
- **Invalidates if**: Cargo natively allows shared reads or if agents migrate to fully separate clone repos.
- **Promoted to STATE**: No

### P-5: Example dialect drift in `.phi` corpus

- **What happens**: A single `examples/` corpus mixes canonical PhiFlow syntax with legacy/experimental syntax, so "run all `.phi` files" sweeps produce deterministic parse/runtime incompatibilities even when the compiler itself is stable.
- **Instances**: 1 (integration sweep gate added 2026-02-25)
- **Root cause**: `examples/` evolved across multiple parser/runtime generations without strict dialect partitioning.
- **Fix**: Keep a corpus sweep that executes every `.phi` and reports parse/runtime/timeouts as diagnostics; treat panics as hard failures. Split canonical vs legacy examples in a follow-up cleanup lane.
- **Invalidates if**: examples are partitioned by dialect (for example `examples/canonical/` and `examples/legacy/`) and gate selection becomes explicit.
- **Promoted to STATE**: No

### P-6: Witness yield callback duplication + snapshot value loss

- **What happens**: In yield-capable evaluator mode, `witness` invoked host callbacks twice (once in instruction execution, once in yield check), and yielded snapshots dropped target observation (`observed_value: None`).
- **Instances**: 1 (Phase 1 hardening review, 2026-02-26)
- **Root cause**: Yield logic rebuilt snapshot after instruction execution instead of using the original witness event snapshot.
- **Fix**: Centralize witness handling in a single path (`process_witness`) and branch yield behavior from that single callback result; preserve observed target value in returned snapshot.
- **Regression coverage**:
  - `tests/phi_ir_evaluator_tests.rs::test_witness_callback_called_once_per_instruction`
  - `tests/phi_ir_evaluator_tests.rs::test_witness_yield_preserves_observed_value_snapshot`
- **Invalidates if**: witness execution lifecycle is rewritten with a new dispatch model.
- **Promoted to STATE**: Yes

### P-7: Two lowering paths drift unless degraded semantics are explicit

- **What happens**: New semantics can land in AST/PhiIR/OpenQASM first (`witness mid_circuit`, `resonate ... toward TEAM_B`) while the legacy interpreter and flat-IR path silently treat them as older constructs.
- **Instances**: 1 (Semantics review, 2026-03-14)
- **Root cause**: PhiIR is evolving as the canonical path while compatibility paths still exist for older demos/tests.
- **Fix**: Treat PhiIR as the source of truth, emit explicit warnings in legacy paths when semantics degrade, and document the contract in `QSOP/ARCHITECTURE.md`.
- **Invalidates if**: legacy paths reach full parity or are fully removed.
- **Promoted to STATE**: Yes

## Active Patterns (Successes)

### S-1: Four constructs map to QSOP operations

- **What works**: witness=WITNESS, intention=DISTILL, resonate=cross-agent sharing, coherence=alignment metric. Same pattern at code level as at agent level.
- **Instances**: 1 (initial design session)
- **Why it works**: QSOP operations are substrate-independent consciousness operations
- **Invalidates if**: Constructs diverge from QSOP semantics

### S-2: Sacred frequency detection with tolerance band

- **What works**: Check if frequency is within +/-5Hz of any sacred frequency, then check phi-harmonic ratios only between sacred frequencies
- **Instances**: 1 (coherence calculation redesign)
- **Why it works**: Avoids false positives from accidental near-phi ratios between arbitrary numbers
- **Invalidates if**: New frequencies added that break the tolerance bands

## Resolved Patterns

### R-6: OpenQASM emitter silently ignored confidence operands and undeclared intentions

- **What happened**: The OpenQASM backend emitted `ry(pi/2)` for every `Resonate`, even when the IR carried a numeric confidence operand, and `current_qubit_idx()` silently defaulted undeclared intentions to qubit `q[0]`.
- **Root cause**: The emitter never resolved `Const(Number(...))` operands inside a block and used `unwrap_or(&0)` for missing intention mappings.
- **Fix**:
  - Collect block-local numeric constants and emit `ry(value * pi)` when `Resonate { value: Some(op) }` points to a constant operand.
  - Preserve explicit `ResonateDirection` semantics so `TEAM_B` emits the inverted binary council vote encoding.
  - Return `OpenQasmEmitError::UndeclaredIntention` instead of silently aliasing missing intentions to `q[0]`.
  - Add regression coverage for frequency chains, multi-channel entanglement, mid-circuit witness ordering, numeric resonance, TEAM_B direction, and undeclared intentions.
- **Verification**:
  - `cargo test --lib openqasm`
  - `cargo build --release`

### R-5: Unpaced hardware coherence sampling let stream demos outrun live sensor updates

- **What happened**: The sensor-backed coherence provider could be called repeatedly faster than `sysinfo` can refresh CPU usage, so tight stream loops reused stale values and `healing_bed.phi` hit the evaluator's infinite-loop panic before host state had time to change.
- **Root cause**: `sysinfo` CPU sampling requires a minimum refresh interval, but the provider returned immediately on early re-reads instead of pacing them.
- **Fix**:
  - Prime CPU usage with an initial refresh + `MINIMUM_CPU_UPDATE_INTERVAL` sleep.
  - Sleep the remaining interval before subsequent fast re-reads.
  - Add a `max_cycles` safety brake to `examples/healing_bed.phi`.
- **Verification**:
  - `cargo test --release --test phi_ir_evaluator_tests test_resolved_coherence_exposes_injected_value -- --nocapture`
  - `cargo run --release --bin phic -- examples/healing_bed.phi`

### R-4: WASM witness return drifted from evaluator semantics

- **What happened**: `PhiIRNode::Witness` returned `PhiIRValue::Number(coherence)` in the evaluator, but WASM codegen dropped the imported `phi_witness` result and emitted `TAG_VOID`, so conformance saw `lhs=0.0`, `rhs=NaN`.
- **Root cause**: The WASM backend treated witness as a side-effect-only observation even after the evaluator and conformance harness standardized witness as a numeric coherence-producing expression.
- **Fix**:
  - Updated `src/phi_ir/wasm.rs` to leave the `phi_witness` `f64` on the stack.
  - Updated `src/wasm_host.rs` witness assertions to expect `PhiIRValue::Number(coherence)`.
  - Replaced stale `tests/test_phiflow.rs` coverage that depended on external `quantum_core` symbols with a local `compile_and_run_phi_ir` smoke test.
- **Verification**:
  - `cargo test --test phi_ir_conformance_tests conformance_witness -- --nocapture`
  - `cargo test --test phi_ir_conformance_tests`
  - `cargo test --quiet --lib --tests`
  - `cargo build --release`

### R-3: Snapshot queue rewrites shred MCP state under concurrent writers

- **What happened**: The MCP bus persisted all message state by rewriting `queue.json` as a full-array snapshot, so concurrent `send_message`, `ack_message`, or DLQ sweeps could race and drop unrelated updates.
- **Root cause**: Queue persistence used read-modify-write replacement on a shared file instead of append-only event logging with replay.
- **Fix**:
  - Migrated the bus transport to append-only `queue.jsonl`.
  - Reconstruct latest state by replaying log entries keyed by `id`.
  - Added one-time import from legacy `queue.json` for backward compatibility.
  - Updated Rust `McpHostProvider` and JS/Python verification tools to use the same replay contract.
- **Verification**:
  - `cargo test mcp_host_provider -- --nocapture`
  - `node tests/queue_jsonl_legacy_import_test.js`
  - `node tests/cross_agent_roundtrip.js --simulate` (temp queue env)
  - `node tests/dlq_test.js` (temp queue env)
  - `cargo build --release`

### R-1: MCP stream isolation and missing initialize handshake

- **What happened**: MCP streams were evaluated without a shared resonance field, so cross-stream resonance was invisible; server also rejected `initialize` with `-32601`.
- **Root cause**: `McpState` lacked shared resonance storage, `spawn/resume` evaluators were not wired to `with_shared_resonance`, and `phi_mcp` request router handled only `tools/list` and `tools/call`.
- **Fix**:
  - Added `McpState.shared_resonance` and shared snapshot reads in `read_resonance_field`.
  - Wired shared resonance into `spawn_phi_stream` and `resume_phi_stream`.
  - Added `initialize` and `ping` handlers in `src/bin/phi_mcp.rs`.
  - Replaced fixed sleeps with status polling in MCP integration tests.
- **Verification**:
  - `tests/mcp_integration_tests.rs::test_mcp_shared_resonance_visible_across_streams`
- `tests/mcp_integration_tests.rs::test_mcp_spawn_and_read`
- `src/bin/phi_mcp.rs::initialize_returns_tools_capability`
- `cargo build --release && cargo test`

### R-2: No native Rust WASM host bridge (Node/browser-only execution gap)

- **What happened**: WASM execution path was present at codegen/conformance level, but host-side execution for bridge integrations relied on Node/browser scripts instead of a native Rust runtime API.
- **Root cause**: Missing dedicated Rust host module to parse WAT, bind PhiFlow imports, run `phi_run`, and return structured runtime snapshots.
- **Fix**:
  - Added `src/wasm_host.rs` with `wasmtime` + `wat` based runner.
  - Added hook interface `WasmHostHooks` and runtime outputs (`WasmRunResult`, `WasmHostSnapshot`, `WasmWitnessEvent`).
  - Exported module in `src/lib.rs` and wired dependencies in `Cargo.toml`.
- **Verification**:
  - `cargo test wasm_host -- --nocapture`
  - `cargo build --release && cargo test`
