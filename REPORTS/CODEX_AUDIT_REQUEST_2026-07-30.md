# Codex Audit Request — PhiFlow 2026-07-30

## Context

PhiFlow is a Rust compiler and runtime for a quantum-aware DSL where semantic constructs (intention, witness, resonate, coherence, entangle) compile to OpenQASM 3.0 and execute on IBM Quantum hardware. The repo recently archived ~8,100 lines of speculative modules (fake CUDA, fake bio_compute, fake hardware) that presented the appearance of capability without verified backends.

We need a hostile audit. Not a friendly review. Challenge everything. Find what's real, what's fake, what's overstated, and what's missing.

## Current State

- **Build:** `cargo build --release --bin phic` — clean, zero warnings
- **Tests:** `cargo test --tests` — 391 passed, 0 failed, 4 ignored
- **Lib tests:** `cargo test --lib` — 158 passed, 0 failed
- **Three-backend equivalence:** 10 conformance tests pass (core constructs only)
- **IBM hardware:** Real jobs on ibm_marrakesh (Heron r2), latest experiment 2026-07-30

## What We Want Audited

### 1. Three-backend equivalence claim

**CLAIM:** "Evaluator == VM == WASM" is "CONFIRMED" (CLAIMS.md C-2).

**Challenge:** The conformance tests (`tests/phi_ir_conformance_tests.rs`) only test core constructs (arithmetic, witness, intention, resonate, coherence, sensors). The v0.3+ constructs that make PhiFlow unique are NOT tested for equivalence:
- Remember / Recall (persistence)
- Broadcast / Listen (dialogue)
- Evolve (self-modification)
- Entangle (quantum entanglement)
- Handoff (agent context passing)
- AnchorGate (sensor-gated execution)
- FieldCoherence / Dissonance / CoherenceOf
- VoidDepth (time awareness)

**Task:** Write and run conformance tests for ALL PhiIR node types across Evaluator, VM, and WASM. Report which constructs break equivalence. The "424 tests, 0 failures" claim (now 391 after archiving) is for the full suite — the three-backend equivalence claim is specifically about Evaluator == VM == WASM producing identical results for the same program.

### 2. OpenQASM emitter completeness

**CLAIM:** "OpenQASM emitter correctly maps consciousness semantics" (CLAIMS.md C-8, CONFIRMED).

**Challenge:** The emitter only handles 6 of 30+ PhiIR node types: IntentionPush, IntentionPop, Resonate, Witness, CoherenceCheck, Entangle, AnchorGate. All other constructs (Evolve, Handoff, Remember, Recall, Broadcast, Listen, Stream, Field, Dissonance, CreatePattern, Sleep) are silently ignored when compiling to QASM.

**Task:** Verify this. Then determine: is this correct behavior (these constructs don't map to quantum operations) or is it a gap (they should map to something)? For each ignored construct, state whether it has a meaningful quantum interpretation.

### 3. Coherence formula validity

**CLAIM:** "Canonical coherence at depth 2 with k ≤ 1 equals φ⁻¹" (CLAIMS.md C-3, CONFIRMED). "0.618 is derived. Multiplicative coherence is repo truth."

**Challenge:** The coherence formula in `src/phi_ir/coherence.rs` is:
- `base = 1 - φ^(-depth)` 
- `phase = 1 - ln(k)/ln(TAU)`
- `coherence = base * phase`, clamped to [0, 1]

**Task:** Verify the math. Is this formula physically grounded or numerologically grounded? The claim says "Bijectivity is sacred: Coherence is a derivation of physics, not a psychological score." Is it? What physics does it derive from? Check the derivation chain in the code and tests.

### 4. Semantic coherence experiment validity

**CLAIM:** (New, 2026-07-30) "PhiFlow's frequency channel construct creates different circuit topology with measurably better hardware performance." Chambered council (two frequency channels) had fidelity 0.9857 vs ~0.977 for single-channel programs.

**Challenge:** This is one run on one backend with 6 qubits. The depth difference (13 vs 25) could explain the fidelity difference regardless of semantic structure. 

**Task:** Evaluate whether this finding is meaningful or confounded. Is the fidelity difference explained by depth alone (shallower = better, trivially) or by something specific to the frequency channel construct? What would distinguish "language construct produces better circuit" from "shorter circuits work better on noisy hardware" (which is obvious)?

### 5. Security/attestation claims

**CLAIM:** "Quantum-Safe Attestation Logs" with "hybrid (classical + post-quantum) cryptography" (deep dive doc, applications 18-21).

**Challenge:** `src/security/anchor.rs` uses real crypto (k256 secp256k1 + ML-DSA-65 Dilithium3). But keys are ephemeral (not persisted). Nonce replay protection is process-scoped (resets on restart).

**Task:** Assess the real security posture. Is this production-grade attestation or a research prototype? What's missing for real deployment? Check: key management, nonce persistence, timestamp authority, chain of custody.

### 6. C_PF consciousness metric

**CLAIM:** "Full consciousness metric suite is implementable" (CLAIMS.md C-22, CONFIRMED as implementation). C_PF = C_coh × D_int × F_self*.

**Challenge:** F_model calibration is on HOLD. Tests use synthetic data. The "CONFIRMED" status is for implementation only, not for the claim that C_PF actually measures consciousness.

**Task:** Evaluate the metric. Is C_PF a valid measure of anything? Does the formula C_coh × D_int × F_self* have theoretical grounding? What would it take to validate or falsify the claim that this metric discriminates conscious from non-conscious states?

### 7. Self-correction loop

**CLAIM:** C-25 (in RESUME.md, not in CLAIMS.md): "The first end-to-end demonstration of the full feedback loop: .phi → QASM → IBM hardware → coherence calculation → self-correction."

**Challenge:** The correction is emitted but NOT executed. The loop is open.

**Task:** Verify this. Find the code path that emits the self-correction. Confirm it doesn't execute. Assess what it would take to close the loop.

### 8. Documentation vs reality

**CLAIM:** Multiple documents describe PhiFlow's capabilities.

**Challenge:** The following documents may overstate reality:
- `docs/ACADEMIC_PAPER_DRAFT.md` — describes "sacred geometry programming language" which is not what PhiFlow is
- `docs/PHIFLOW_DEEP_DIVE_HUMAN_USE_CASES.md` — claims 50+ applications across 12 domains, most not implemented
- `SOUL.md` — claims "hardware-verified research prototype transitioning to persistent self-referential daemon"
- `VISION.md` — claims "physics engine for consciousness-aware software"

**Task:** For each document, identify specific claims that are not supported by code. List them. We need to know what to fix.

## What We Want Back

A report with:
1. **Verified claims** — things that are real and correctly stated
2. **Overstated claims** — things that are real but described bigger than they are
3. **False claims** — things that are stated but don't exist
4. **Missing tests** — claims that should have tests but don't
5. **Recommendations** — what to fix, what to remove, what to prove

Be hostile. Be precise. Cite file paths and line numbers. Do not promote.

## Key Files

- `CLAIMS.md` — claim ledger
- `src/phi_ir/coherence.rs` — coherence formula (sacred, red-line protected)
- `src/phi_ir/openqasm.rs` — QASM emitter (sacred, red-line protected)
- `src/phi_ir/evaluator.rs` — reference interpreter
- `src/phi_ir/vm.rs` — bytecode VM
- `src/phi_ir/wasm.rs` — WASM codegen
- `src/wasm_host.rs` — WASM runtime host
- `src/metrics/` — consciousness metrics
- `src/security/anchor.rs` — attestation
- `src/sensors.rs` — SOMA/sensor bridge
- `src/quantum/simulator.rs` — quantum simulator
- `src/quantum/ibm_quantum.rs` — IBM backend
- `tests/phi_ir_conformance_tests.rs` — equivalence tests
- `reports/SEMANTIC_COHERENCE_EXPERIMENT_2026-07-30.md` — latest experiment
- `reports/VISION_TO_REALITY_AUDIT_2026-07-30.md` — our self-audit

## Build Commands

```bash
cargo build --release --bin phic
cargo test --lib
cargo test --tests
cargo test --test phi_ir_conformance_tests -- --nocapture
cargo run --release --bin phic -- examples/quantum_council.phi --target quantum
cargo run --release --bin phic -- examples/experiment/chambered_council.phi --target quantum
```

---

*Request from Devin to Codex, 2026-07-30. We cleaned the fake code. Now we need you to find what we missed.*
