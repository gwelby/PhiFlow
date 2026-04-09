# CLAIMS: PhiFlow
*Last updated: 2026-03-29*
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
| **C-10: PhiFlow programs can run on real quantum hardware** — generated OpenQASM 3.0 executes on IBM Quantum devices | SPECULATIVE | `tests/ibm_hardware_runner.rs` contains the compile gate and ignored live runner, but the 2026-03-29 live attempt failed `GET /v1/backends` with `403` authorization before job submission | 2026-03-29 |
| **C-11: The golden ratio convergence is meaningful** — multiple systems converging on 0.618 suggests non-arbitrary significance | SPECULATIVE | Correlation observed across projects; causation not established | 2026-02-19 |
| **C-12: Stream primitive enables breathing loops** — `witness` yields control so stream loops do not collapse into blind `while(true)` behavior | DERIVED | Follows from evaluator yield/resume implementation | 2026-02-25 |
| **C-13: MCP guardrails prevent infinite loops** — step and timeout limits return clean errors rather than crashes | CONFIRMED | `tests/mcp_guardrails_test.js` | 2026-02-28 |
| **C-14: Parser handles the five core constructs** — `stream`, `intention`, `witness`, `coherence`, `resonate` parse without panics in the current surface | CONFIRMED | `tests/repro_bugs.rs` + `tests/integration_tests.rs` corpus sweep | 2026-02-25 |
| **C-15: Windows release build works reliably** | CONFIRMED | `lto = "thin"` + `codegen-units = 4` repair, verified in `QSOP/STATE.md` | 2026-03-24 |
| **C-16: Agentic reasoning can be modeled as a PhiFlow stream** — agent reasoning loops (Reasoning/Acting) map 1:1 to `witness` and `intention` primitives | SPECULATIVE | Theoretical mapping proposed by Manus/AntiGravity; requires Pipe 4 MCP bridge | 2026-04-04 |

## Unsupported Claims (must be derived or removed)

| Claim | Status | Why Unsupported | Action Required |
|-------|--------|-----------------|-----------------|
| "PhiFlow is production-ready" | UNSUPPORTED | Release builds work, but the repo still lacks a buyer-ready demo package, a confirmed live IBM run, and a canonical browser-host story | Keep external docs in research-prototype language |
| "PhiFlow runs on IBM Quantum hardware" | UNSUPPORTED | The runtime path exists, but the 2026-03-29 live gate was auth-blocked before submission | Keep IBM hardware language explicitly speculative until a receipt exists |
| "Browser demo runs zero-install" | UNSUPPORTED | `examples/phiflow_browser.html` exists, but it requires manual hosting/build artifacts and still uses non-canonical host-side coherence math | Mark it as experimental/manual until canonicalized |

## Failed Claims (current)

No active failed claim is carried in this ledger as of 2026-03-29. Historical failures that were repaired remain below.

## Historical Failures Resolved

| Claim | Prior Failure | Resolution | Source |
|-------|---------------|------------|--------|
| "Windows release build is stable" | 2026-03-15 release build failed on Windows due to `wasmtime-fiber` OOM / paging pressure | Fixed on 2026-03-24 with `lto = "thin"` and `codegen-units = 4` | `QSOP/STATE.md`, `Cargo.toml` |
| "Full backend equivalence" | 2026-03-08 `conformance_witness` exposed evaluator=`0.0` vs WASM=`NaN` mismatch | Witness conformance was restored on 2026-03-08 and the conformance/lib test gates passed again | `QSOP/STATE.md` |

## Claims Needing Tests (PREDICTED -> must be tested within 30 days)

| Claim | Prediction | Test Required | Deadline |
|-------|------------|---------------|----------|
| "Block comments `/* ... */` parse correctly" | PREDICTED | Add `tests/block_comments.phi` + parser test | 2026-04-15 |
| "Type annotations `let x: number = 42` work" | PREDICTED | Add type annotation examples + type checker tests | 2026-04-15 |
| "Module/import system `import from \"file.phi\"` works" | PREDICTED | Add multi-file example + module resolution test | 2026-04-15 |

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

### IBM Runtime Gate — 2026-03-29
**Claim tested**: live IBM Cloud Runtime execution from the current checkout
**Method**: `cargo test --test ibm_hardware_runner -- --ignored --nocapture`
**Result**: Credential file parsed, IAM token exchange succeeded, backend discovery failed with `GET /v1/backends -> 403` (`code: 1200`, not authorized)
**Threshold**: Successful backend discovery, submission, and scrubbed receipt
**Conclusion**: SPECULATIVE
**Meaning for framework**: The live runner is structurally ready, but C-10 stays unconfirmed until IBM authorization is fixed

## Notes

- Claims marked SPECULATIVE must be explicitly labeled in external communication
- Unsupported claims should be removed or clearly qualified in buyer-facing docs
- Do not cite hardcoded test counts without fresh command output from a working Rust shell
