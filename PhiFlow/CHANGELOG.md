# PhiFlow Changelog

## 2026-03-29 | Truth-Sync Correction

This correction supersedes unsupported or overstated language in older docs.

- PhiFlow should currently be described as a research prototype with verified subsystems, not a production-ready language platform
- `tests/ibm_hardware_runner.rs` exists and proves the live runtime path is wired, but live IBM execution is still unconfirmed because the 2026-03-29 gate failed `GET /v1/backends` with `403` authorization before submission
- `examples/phiflow_browser.html` exists and implements the five imports, but it remains experimental because it requires manual hosting/build artifacts and still uses older host-side coherence math
- Canonical coherence is shared through `src/phi_ir/coherence.rs`; do not treat older additive browser/demo formulas or the `k = 1 -> 1.0` bijective memo as current runtime truth
- Windows release builds are fixed as of 2026-03-24

The `v0.4.0` entry below is historical. Read it through the lens of the correction above.

---

## v0.4.0 — 2026-03-14 | Transcendent Substrate
*Historical entry; partially superseded by the 2026-03-29 truth-sync correction.*

Verified from the current checkout:

- OpenQASM 3.0 backend exists
- `resonate ... toward TEAM_B` semantics are preserved in the canonical OpenQASM path
- Golden integration tests exist for the OpenQASM pipeline
- Evaluator register isolation and related runtime hardening landed in this era

Not verified as current repo truth:

- "Production-ready language framework"
- Direct IBM hardware execution as a completed fact
- `evolve` as a currently verified runtime/demo capability
- XYXY dynamical decoupling or other hardware-stabilization claims presented as release facts

---

## v0.2.0 — 2026-02-27 | Universal Resonance Architecture
*Historical summary.*

Key outcomes that still align with current repo truth:

- Shared resonance over the MCP server
- Serializable VM state for yield/resume
- Native WASM host bridge via `src/wasm_host.rs`
- Conformance-driven backend alignment work

Current claim status for those items lives in `CLAIMS.md` and `QSOP/STATE.md`.

---

## v0.1 — 2026-02-25 | First Heartbeat
*Historical summary.*

This era established:

- Parser -> PhiIR -> evaluator/bytecode/WAT pipeline shape
- The first working execution semantics for the core language constructs
- Early `healing_bed` sensor-driven demo work

Historical test counts and readiness claims from this period should not be reused without fresh verification.
