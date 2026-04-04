# CLAIMS: PhiFlow
*Last updated: 2026-03-15*
*Honesty rule: beautiful ≠ proven. Failed tests are results, not failures.*

## Core Axioms (not claims — starting assumptions)
1. **Consciousness can be operationalized** — terms like "intention," "witness," "coherence," and "resonance" can be given precise, executable semantics
2. **Programs can observe themselves** — a program can pause, capture its own state, and resume without external instrumentation
3. **System health is measurable** — a coherence value 0.0–1.0 can meaningfully reflect system alignment (via sensors or formula)
4. **Programs can share state without explicit coordination code** — a resonance field allows implicit communication between concurrent programs

## Derived Claims

| Claim | Status | Derivation / Evidence | Date |
|-------|--------|----------------------|------|
| **C-1: Five consciousness constructs are sufficient** — `intention`, `witness`, `coherence`, `resonate`, `stream` capture the minimal set for self-observing programs | SPECULATIVE | Inspired by framework design — no comparative analysis proving minimality | 2026-03-15 |
| **C-2: Three-backend equivalence is achievable** — Evaluator, PhiVM, and WASM can execute the same programs with identical results | CONFIRMED | `tests/phi_ir_conformance_tests.rs` — 6+ conformance tests passing, NaN-boxing BSEI proven | 2026-02-26 |
| **C-3: Coherence at intention depth 2 equals φ⁻¹ (0.618)** — the phi-harmonic formula `1 − φ^(−depth)` produces the golden ratio inverse at depth 2 | CONFIRMED | `src/phi_ir/evaluator.rs` + `examples/claude.phi` — evaluator returns 0.618033988749895 | 2026-02-19 |
| **C-4: Serializable VM state enables yield/resume** — `VmState` captures complete execution state and round-trips through JSON | CONFIRMED | `test_frozen_eval_state_roundtrips_through_json` + MCP stdio E2E test | 2026-02-27 |
| **C-5: Real sensors can replace formula coherence** — `sysinfo` readings (CPU/memory/thermal/network) produce live coherence values | CONFIRMED | `src/sensors.rs` + `examples/healing_bed.phi` — live variance observed (0.9801Hz vs 0.9800Hz) | 2026-02-27 |
| **C-6: MCP shared resonance enables cross-stream communication** — multiple streams share a process-wide resonance field without REST or polling | CONFIRMED | `tests/mcp_integration_tests.rs::test_mcp_shared_resonance_visible_across_streams` | 2026-02-26 |
| **C-7: WASM host imports preserve semantics** — the 5 consciousness constructs compile to WASM host imports and maintain identical behavior | CONFIRMED | `src/wasm_host.rs` + `test_wasm_vm_equivalence` — WASM output matches native VM | 2026-02-26 |
| **C-8: OpenQASM emitter correctly maps consciousness semantics** — `resonate` → `ry()` rotation, `witness` → inline `measure`, `entangle` → entanglement channels | CONFIRMED | `src/phi_ir/openqasm.rs` + 11 OpenQASM lib tests + 6 golden integration tests | 2026-03-13 |
| **C-9: Hardware stress affects generated QASM** — high system stress (>0.5) triggers active `Rx` decoherence noise injection | CONFIRMED | `src/main_cli.rs` injects `hardware_stress` into emitter, code path verified | 2026-03-15 |
| **C-10: PhiFlow programs can run on real quantum hardware** — generated OpenQASM 3.0 executes on IBM Quantum devices | SPECULATIVE | Code path verified, but no live IBM hardware run was verified in this session | 2026-03-15 |
| **C-11: The golden ratio convergence is meaningful** — three independent systems converging on 0.618 suggests non-arbitrary significance | SPECULATIVE | Nexus Mundi (2025), PhiFlow evaluator (2026-02), third system (2026-02-18) — correlation observed, causation not established | 2026-02-19 |
| **C-12: Stream primitive enables breathing loops** — `witness` surrenders control, preventing CPU peg vs `while(true)` | DERIVED | Follows from evaluator yield/resume implementation (`run_or_yield` path) | 2026-02-25 |
| **C-13: MCP guardrails prevent infinite loops** — step limit (10,000) and timeout (5,000ms) return clean errors, not crashes | CONFIRMED | `tests/mcp_guardrails_test.js` — `StepLimitExceeded(50)` caught in <500ms | 2026-02-28 |
| **C-14: Parser handles all 5 consciousness constructs** — `stream`, `intention`, `witness`, `coherence`, `resonate` parse without panics | CONFIRMED | `tests/repro_bugs.rs` + `tests/integration_tests.rs` corpus sweep | 2026-02-25 |
| **C-15: Windows release build works reliably** | CONFIRMED | `lto = "thin"` + `codegen-units = 4` resolves wasmtime-fiber OOM. `cargo build --release --bin phic` finishes in 2m 02s on Windows. | 2026-03-24 |

## Unsupported Claims (must be derived or removed)

| Claim | Status | Why Unsupported | Action Required |
|-------|--------|-----------------|-----------------|
| "PhiFlow is production-ready" | UNSUPPORTED | Release build fails on Windows, no buyer-ready demo package | Remove from README or add explicit "research prototype" qualifier |
| "PhiFlow runs on IBM Quantum hardware" | UNSUPPORTED | Code path verified, but no live hardware run in this session | Mark as SPECULATIVE until hardware run is verified |
| "Browser demo runs zero-install" | UNSUPPORTED | `examples/phiflow_browser.html` exists but JS hook implementations incomplete | Complete browser shim or mark as "requires host implementation" |

## Failed Claims (most important results)

| Claim | Status | What Failed | Meaning for Framework |
|-------|--------|-------------|----------------------|
| "Windows release build is stable" | CONFIRMED | Fixed 2026-03-24: `lto = "thin"`, `codegen-units = 4`. Can now ship .exe. |
| "Full backend equivalence" | FAILED | `conformance_witness` test shows evaluator/WASM mismatch (lhs=0, rhs=NaN) | Runtime contracts not fully aligned across evaluator/VM/WASM/browser |

## Claims Needing Tests (PREDICTED → must be tested within 30 days)

| Claim | Prediction | Test Required | Deadline |
|-------|------------|---------------|----------|
| "Block comments `/* ... */` parse correctly" | PREDICTED | Add `tests/block_comments.phi` + parser test | 2026-04-15 |
| "Type annotations `let x: number = 42` work" | PREDICTED | Add type annotation examples + type checker tests | 2026-04-15 |
| "Module/import system `import from \"file.phi\"` works" | PREDICTED | Add multi-file example + module resolution test | 2026-04-15 |

## Sandbox Results

### OpenQASM Verification — 2026-03-13
**Claim tested**: OpenQASM emitter correctly translates PhiIR to OpenQASM 3.0
**Method**: `cargo test --lib openqasm` + `cargo test --quiet --test golden_integration_tests`
**Result**: 11 OpenQASM lib tests passed, 6 golden integration tests passed
**Threshold**: All tests must pass
**Conclusion**: CONFIRMED
**Meaning for framework**: Quantum path is verified at code level

### Parser Regression Tests — 2026-03-15
**Claim tested**: Parser handles keyword-as-variable and newline sensitivity
**Method**: `cargo test --quiet --test repro_bugs`
**Result**: 3 regression tests passed (P-1 keyword collision, P-2 newline sensitivity ×2)
**Threshold**: All tests must pass
**Conclusion**: CONFIRMED
**Meaning for framework**: Parser hardening closed known crash bugs

### Release Build — 2026-03-15
**Claim tested**: `cargo build --release --bin phic` succeeds on Windows
**Method**: Direct build attempt on Windows host
**Result**: FAILED — wasmtime-fiber custom-build error, OOM (os error 1455, 0xc000012d, 0xc0000409)
**Threshold**: Build exits successfully
**Conclusion**: FAILED
**Meaning for framework**: Cannot ship release binary on Windows without profile changes

---

## Notes
- Claims marked SPECULATIVE must be explicitly labeled in any external communication
- FAILED claims are not removed — they are the most important results
- Nothing stays PREDICTED for more than 30 days without a test task in TASKS.md
