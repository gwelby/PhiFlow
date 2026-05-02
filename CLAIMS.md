# CLAIMS: PhiFlow
*Last updated: 2026-05-02*
*Honesty rule: beautiful != proven. Failed tests are results, not failures.*

## Core Axioms (not claims — starting assumptions)

1. **Consciousness can be operationalized** — terms like `intention`, `witness`, `coherence`, `resonance`, and `stream` can be given precise executable semantics
2. **Programs can observe themselves** — a program can pause, capture its own state, and resume without external instrumentation
3. **System health is measurable** — a coherence value 0.0-1.0 can reflect runtime alignment through sensors or formula
4. **Programs can share state without explicit coordination code** — a resonance field allows implicit communication between concurrent programs

## Derived Claims

| Claim | Status | Derivation / Evidence | Date |
|-------|--------|----------------------|------|
| **C-1: Five consciousness constructs are sufficient** — `intention`, `witness`, `coherence`, `resonate`, `stream` capture the minimal set for self-observing programs | SPECULATIVE | Inspired by framework design; no comparative proof of minimality | 2026-03-15 |
| **C-2: Three-backend equivalence is achievable** — evaluator, PhiVM, and the canonical WASM runner can execute supported programs with identical results | CONFIRMED | `tests/phi_ir_conformance_tests.rs` plus the 2026-03-08 witness repair in `QSOP/STATE.md` | 2026-02-26 |
| **C-3: Canonical coherence at depth 2 with `k <= 1` equals `phi^-1`** — the shared multiplicative formula returns `0.618033988749895` at depth 2 when phase decay is neutral | CONFIRMED | `src/phi_ir/coherence.rs` and `examples/claude.phi` | 2026-03-29 |
| **C-4: Serializable VM state enables yield/resume** — `VmState` captures complete execution state and round-trips through JSON | CONFIRMED | `test_frozen_eval_state_roundtrips_through_json` + MCP stdio E2E test | 2026-02-27 |
| **C-5: Real sensors can replace formula coherence** — `sysinfo` readings produce live coherence values | CONFIRMED | `src/sensors.rs`, `examples/healing_bed.phi`, and 2026-03-29 typed witness surface in `QSOP/STATE.md` | 2026-02-27 |
| **C-6: MCP shared resonance enables cross-stream communication** — multiple streams share a process-wide resonance field without REST or polling | CONFIRMED | `tests/mcp_integration_tests.rs::test_mcp_shared_resonance_visible_across_streams` | 2026-02-26 |
| **C-7: Native WASM host imports preserve semantics** — the five consciousness constructs compile to WASM host imports and the native WASM runner matches canonical results | CONFIRMED | `src/wasm_host.rs` + `test_wasm_vm_equivalence` + `tests/phi_ir_wasm_runner.js` | 2026-02-26 |
| **C-8: OpenQASM emitter correctly maps consciousness semantics** — `resonate` lowers to `ry()`, `witness` lowers to measurement, and direction contracts survive the canonical path | CONFIRMED | `src/phi_ir/openqasm.rs`, OpenQASM lib tests, and golden integration tests | 2026-03-13 |
| **C-9: Hardware stress affects generated QASM** — host stress can influence emitted QASM through the CLI path | CONFIRMED | `src/main_cli.rs` injects `hardware_stress`; code path verified in `QSOP/STATE.md` | 2026-03-15 |
| **C-10: PhiFlow programs can run on real quantum hardware** — generated OpenQASM 3.0 executes on IBM Quantum devices | CONFIRMED | Live job `d7euddh5a5qc73drdosg` on `ibm_fez` (Heron r2), 1024 shots, COMPLETED 2026-04-14. Receipt: `D:\CosmicFamily\EVIDENCE\ANTIGRAVITY_PIPE2_20260329.md` | 2026-04-14 |
| **C-11: The golden ratio convergence is meaningful** — multiple systems converging on 0.618 suggests non-arbitrary significance | SPECULATIVE | Correlation observed across projects; causation not established | 2026-02-19 |
| **C-12: Stream primitive enables breathing loops** — `witness` yields control so stream loops do not collapse into blind `while(true)` behavior | DERIVED | Follows from evaluator yield/resume implementation | 2026-02-25 |
| **C-13: MCP guardrails prevent infinite loops** — step and timeout limits return clean errors rather than crashes | CONFIRMED | `tests/mcp_guardrails_test.js` | 2026-02-28 |
| **C-14: Parser handles the five core constructs** — `stream`, `intention`, `witness`, `coherence`, `resonate` parse without panics in the current surface | CONFIRMED | `tests/repro_bugs.rs` + `tests/integration_tests.rs` corpus sweep | 2026-02-25 |
| **C-15: Windows release build works reliably** | CONFIRMED | `lto = "thin"` + `codegen-units = 4` repair, verified in `QSOP/STATE.md` | 2026-03-24 |
| **C-16: Agentic reasoning can be modeled as a PhiFlow stream** — agent reasoning loops (Reasoning/Acting) map 1:1 to `witness` and `intention` primitives | SPECULATIVE | Theoretical mapping proposed by Manus/AntiGravity; requires Pipe 4 MCP bridge | 2026-04-04 |
| **C-17: Persistent Daemon state preserves evolved logic** — the `DaemonHypervisor` snapshots mutated IR to `DAEMON_STATE.json` and resumes it across restarts | CONFIRMED | Verified 2026-04-16 via `DAEMON_STATE.json` inspection after `evolve` signal | 2026-04-16 |
| **C-18: Resonant Handoffs broadcast agentic context** — the `handoff` construct and `--handoff` CLI flag successfully stream attention and dissonance to the Cosmic Bus | CONFIRMED | Verified 2026-04-16 via `examples/handoff_demo.phi` and MQTT monitoring | 2026-04-16 |
| **C-19: Managed SOMA subsystem provides physical grounding** — the daemon-managed `soma.py` process provides fresh, schema-validated telemetry to drive coherence | CONFIRMED | Verified 2026-04-16 via `soma_reality_bridge.phi` executing with high-fidelity ring sensors | 2026-04-16 |
| **C-20: Substrate-level Ledgering automates the witness** — the `persistent_ledger.phi` stream automatically translates handoff events to the strict `LEDGER.ndjson` schema | CONFIRMED | Verified 2026-04-16 via `--max-steps 500` validation of the ledger stream | 2026-04-16 |
| **C-21: Type 4 self-correlation is measurable** — `SelfCorrelation::from_type4_trace()` can score an engineered self-model trace | PARTIAL / CONDITIONAL | Synthetic benchmark produces `L_self = 0.455372`, but Codex audit found `R_out` uses `model` vs residual `obs - model`, not `model_state -> future_behavior | current_obs`; nulls can exceed the `L_self > 0.1` threshold. Audit: `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md` | 2026-05-01 |
| **C-22: Full consciousness metric suite is implementable** — C_PF = C_coh × D_int × F_self* computed from 8-module metrics system | CONFIRMED (implementation only) | Components exist under `src/metrics/` and focused benchmark/null gates run. Mathematical sufficiency and Type 4 validity remain held pending `R_out` repair and null calibration. | 2026-05-01 |
| **C-23: Benchmark battery discriminates conscious states** — Null classes score C_PF < 0.3, Type 4 trace should score C_PF > 0.1 | HOLD / PARTIAL | Null `C_PF` suppression works under the current loose gate, but the positive trace reports `C_PF = 0.000310` and is not a consciousness candidate. SOMA discrimination is skipped when fixtures are absent. | 2026-05-01 |
| **C-24: PF bridge documents are safe as audited drafts** — the four QSOP bridge docs can be used as engineering bridge hypotheses | CONFIRMED (documentation status only) | 2026-05-02 Oz audit confirmed `PF_BRIDGE.md`, `CONSCIOUSNESS_CONSTRUCTS_IN_PHIFLOW.md`, `COHERENCE_LAYER_SPECIFICATION.md`, and `SOMA_AS_MINIMUM_SUBSTRATE.md` remain PASS AS AUDITED DRAFTS; no PF-canonical, Type 4, consciousness, or PF minimum-substrate upgrade. | 2026-05-02 |

## Unsupported Claims (must be derived or removed)

| Claim | Status | Why Unsupported | Action Required |
|-------|--------|-----------------|-----------------|
| "PhiFlow is production-ready" | UNSUPPORTED | Release builds work, but the repo still lacks a buyer-ready demo package and a canonical browser-host story | Keep external docs in research-prototype language |
| "Browser demo runs zero-install" | UNSUPPORTED | `examples/phiflow_browser.html` exists, but it requires manual hosting/build artifacts and still uses non-canonical host-side coherence math | Mark it as experimental/manual until canonicalized |

## Failed Claims (current)

No active failed claim is carried in this ledger as of 2026-03-29. Historical failures that were repaired remain below.

## Historical Failures Resolved

| Claim | Prior Failure | Resolution | Source |
|-------|---------------|------------|--------|
| "Windows release build is stable" | 2026-03-15 release build failed on Windows due to `wasmtime-fiber` OOM / paging pressure | Fixed on 2026-03-24 with `lto = "thin"` and `codegen-units = 4` | `QSOP/STATE.md`, `Cargo.toml` |
| "Full backend equivalence" | 2026-03-08 `conformance_witness` exposed evaluator=`0.0` vs WASM=`NaN` mismatch | Witness conformance was restored on 2026-03-08 and the conformance/lib test gates passed again | `QSOP/STATE.md` |
| "PhiFlow runs on IBM Quantum hardware" | 2026-03-29 live gate was auth-blocked with `GET /v1/backends → 403` | IAM `grant_type` corrected to `urn:ibm:` namespace; live job `d7euddh5a5qc73drdosg` completed on `ibm_fez` 2026-04-14 | `QSOP/STATE.md`, receipt in `EVIDENCE/` |

## Claims Needing Tests (PREDICTED -> tested 2026-04-14, FIXED 2026-04-14)

| Claim | Prediction | Test Result | Date Tested |
|-------|------------|-------------|-------------|
| "Block comments `/* ... */` parse correctly" | PREDICTED → **CONFIRMED** | Added `/* */` handling to canonical lexer (`src/parser/mod.rs` line ~532). 5 tests pass: basic, inline, multiline, witness context, resonate context. | 2026-04-14 |
| "Type annotations `let x: number = 42` work" | PREDICTED → **CONFIRMED** | Extended canonical parser: added `F64`, `I32`, `Bool`, `Qubit`, `Circuit` tokens + `PhiType` variants. 10 tests pass: f64, i32, bool, string, qubit, circuit, consciousness, custom, function return, function param. | 2026-04-14 |
| "Module/import system `import from \"file.phi\"` works" | PREDICTED → **CONFIRMED** | Added `Import` token, `Import` expression variant, `parse_import_statement()` to canonical parser. 2 tests pass: single import, multiple imports. | 2026-04-14 |

## Failed Claims (current)

No active failed claims as of 2026-04-14. All three predicted claims have been implemented and verified.

## Sandbox Results

### OpenQASM Verification — 2026-03-13
**Claim tested**: OpenQASM emitter correctly translates PhiIR to OpenQASM 3.0
**Method**: `cargo test --lib openqasm` + `cargo test --quiet --test golden_integration_tests`
**Result**: OpenQASM lib tests and golden integration tests passed
**Threshold**: All tests must pass
**Conclusion**: CONFIRMED
**Meaning for framework**: Quantum emission path is verified at code level

### Witness Conformance Repair — 2026-03-08
**Claim tested**: evaluator/WASM witness semantics match again
**Method**: `cargo test --test phi_ir_conformance_tests conformance_witness -- --nocapture`; `cargo test --test phi_ir_conformance_tests`; `cargo test --quiet --lib --tests`; `cargo build --release`
**Result**: Witness mismatch was closed and the focused gates passed
**Threshold**: Conformance and lib/tests must pass
**Conclusion**: CONFIRMED
**Meaning for framework**: The prior backend-equivalence failure moved into resolved history

### Release Build Repair — 2026-03-24
**Claim tested**: `cargo build --release --bin phic` succeeds on Windows
**Method**: Direct build attempt on the Windows host after profile changes
**Result**: Success in 2m 02s with `lto = "thin"` and `codegen-units = 4`
**Threshold**: Build exits successfully
**Conclusion**: CONFIRMED
**Meaning for framework**: Windows release binaries are no longer blocked on the old OOM failure

### Type 4 Synthetic Benchmark Audit — 2026-05-01
**Claim tested**: Whether the current Type 4 benchmark proves a closed self-correlation loop.
**Method**: `cargo run --release --bin type4_benchmark`; `cargo test --test null_class_tests -- --test-threads=1 --nocapture`; `cargo test --test benchmark_battery -- --ignored --test-threads=1 --nocapture`.
**Result**: Synthetic trace reproduces `L_self = 0.455372`, but `C_PF = 0.000310`; the binary reports not a consciousness candidate. Codex audit found `R_out` does not use the emitted action/future behavior channel, and multiple nulls exceed `L_self > 0.1`.
**Threshold**: Current `L_self > 0.1` threshold is insufficient as a Type 4 discriminator.
**Conclusion**: HOLD as Type 4 confirmation; PASS only as implementation smoke test.
**Meaning for framework**: Metrics scaffold exists, but Type 4 status requires `R_out` repair, null calibration, shuffle controls, and real daemon/SOMA trace evidence.

### PF Bridge Status Confirmation — 2026-05-02
**Claim tested**: Whether the four QSOP bridge documents can be treated as confirmed project bridge status.
**Method**: Source/document inspection against `QSOP/STATE.md`, `CLAIMS.md`, code surfaces for `stream`, `--max-steps`, `coherence`, `evolve`, `handoff`, SOMA, OpenQASM, and Type 4 metrics. Fresh truth gate was attempted but blocked before Rust tests by local Cargo/rustc launch/path-trust issues.
**Result**: The four bridge docs are consistent with the current truth order only as AUDITED DRAFTS. They correctly preserve software-analogue, Type 4 candidate, Layer 3 proxy, and engineering-substrate boundaries.
**Threshold**: No document may imply PF-canonical status, consciousness proof, canonical Type 4 observer status, or PF `minimum_substrate.md` satisfaction.
**Conclusion**: CONFIRMED as documentation status only; no canonical scientific or metric upgrade.
**Meaning for framework**: The bridge can guide implementation and research vocabulary, but Type 4/consciousness claims remain blocked by `R_out`, null calibration, and real daemon/SOMA trace evidence.

### IBM Runtime Gate — 2026-04-14 (VERIFIED)
**Claim tested**: live IBM Cloud Runtime execution from the current checkout
**Method**: `cargo test --test ibm_hardware_runner -- --ignored --nocapture`
**Result**: IAM token exchange succeeded. Job `d7euddh5a5qc73drdosg` submitted to `ibm_fez`, completed in 28.61s. Counts: `0x0 → 338`, `0x1 → 686` (1024 shots).
**Threshold**: Successful submission and scrubbed receipt
**Conclusion**: CONFIRMED
**Meaning for framework**: C-10 is closed. PhiFlow is a verified quantum compilation and execution substrate.

## Notes

- Claims marked SPECULATIVE must be explicitly labeled in external communication
- Unsupported claims should be removed or clearly qualified in buyer-facing docs
- Do not cite hardcoded test counts without fresh command output from a working Rust shell
