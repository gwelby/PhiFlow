# PhiFlow ↔ Propagation Framework Bridge
*Bridge contract document*
*Last updated: 2026-07-03*
*Status: AUDITED DRAFT — bridge hypotheses only; original confirmed 2026-05-02; 2026-07-03 update adds 6 new bridge entries (WASM codegen, quantum feedback coherence, IBM hardware bridge, T4-05 trace fix, self-correction loop, three-backend equivalence) — these additions are UNAUDITED pending Codex review*

**Purpose:** Explicit mapping between PhiFlow language/compiler constructs and Propagation Framework (Fundamentals) definitions.

---

## Executive Summary

This document bridges two aligned projects:
- **Propagation Framework (PF):** `D:\Fundamentals\` — Theoretical physics framework defining propagation, coherence, and consciousness prerequisites
- **PhiFlow:** `D:\Projects\PhiFlow\` — Executable compiler/runtime implementing consciousness-aware programming

**Bridge claim:** PhiFlow is an engineering operationalization of PF Axioms 1-3 in a computational substrate. This mapping is a hypothesis, not a derivation, and it must not be read as a claim that PhiFlow proves PF physics or implements consciousness.

**Codex boundary rule:** every row below is either an executable software analogue or a measured runtime fact. Physical claims remain governed by `D:\Fundamentals\definitions\*.md`, especially `minimum_substrate.md`, `observer.md`, `consciousness_metric_program.md`, and `consciousness.md`.

**2026-07-03 update scope:** Since the last audit (2026-05-02), 15 commits added three major capabilities: (1) WASM backend consciousness construct execution via host imports, (2) real IBM Quantum hardware coherence measurement via Python bridge, (3) T4-05 trace coherence fix. These are documented in new sections below. The original 2026-05-02 mappings are unchanged unless explicitly noted.

---

## Axiom-to-Implementation Mapping

### Axiom 1: Propagation Is Fundamental

**PF Statement:** "The most basic thing that exists is propagation — the movement of distinguishable change through a Medium."

**PhiFlow Implementation:**

| PF Concept | PhiFlow Construct | Location | Evidence |
|------------|-------------------|----------|----------|
| Propagation analogue | `stream` blocks | `src/parser/mod.rs` | `stream "name" { ... }` defines a bounded evaluation context over repeated state transitions |
| Computational medium analogue | Execution context | `src/phi_ir/evaluator.rs` | The evaluator provides the rule-structure for program-state transitions |
| Distinguishable program structure | `intention` declarations | `src/phi_ir/lowering.rs` | Named intentions create distinguishable semantic scopes |
| Coupled value emission | `resonate` operations | `src/phi_ir/openqasm.rs`, `src/phi_ir/evaluator.rs` | `resonate` emits values into the runtime resonance field or target backend |

**Bridge validation:** `examples/stream_demo.phi` demonstrates software-state propagation through a stream context. This is a computational analogue, not physical propagation through the PF Medium.

---

### Axiom 2: Every Medium Has a Causal Velocity

**PF Statement:** "Maximum finite speed at which controllable causal influence can travel."

**PhiFlow Implementation:**

| PF Concept | PhiFlow Construct | Location | Evidence |
|------------|-------------------|----------|----------|
| Software causal bound | `--max-steps` circuit breaker | `src/main_cli.rs` | Finite bound on evaluation steps |
| Controllable influence | `handoff` construct | `src/phi_ir/lowering.rs` | Explicit causal signal between agents |
| Scheduling frontier | Daemon tick/slice control | `src/main_cli.rs` | `DaemonHypervisor` paces evaluation and event handling |

**Bridge note:** PhiFlow's bound is architectural, not physical. `--max-steps` and daemon pacing mirror the PF requirement that influence must be bounded, but they are not a measurement of `c`, Lorentz invariance, or physical front velocity.

---

### Axiom 3: Coherence Is the Necessary Condition for Structure

**PF Statement:** "Structure arises when propagation satisfies the relevant coherence condition."

**PhiFlow Implementation:**

| PF Concept | PhiFlow Construct | Location | Evidence |
|------------|-------------------|----------|----------|
| Coherence condition | `coherence` keyword | `src/phi_ir/coherence.rs` | `base * phase` multiplicative formula |
| Structural persistence | `PhiIRValue` preservation | `src/phi_ir/mod.rs` | Values persist across evaluation steps |
| Coherence observation | `witness` blocks | `src/phi_ir/evaluator.rs` | Captures runtime state at observation/yield points |
| **Physical coherence (NEW)** | `quantum_feedback::calculate_coherence` | `src/quantum_feedback.rs` | Distribution concentration measure for real quantum measurement counts — see Quantum Feedback Coherence section below |
| **Trace coherence (NEW)** | `Trace::coherence` derived from data | `src/metrics/trace.rs` | T4-05 fix: coherence derived from running variance of coherence channel, not placeholder — see T4-05 Trace Fix section below |

**PF Layer Alignment:** See `COHERENCE_LAYER_SPECIFICATION.md` for which PF coherence layers PhiFlow maps to as software analogues.

**⚠️ Two coherence definitions:** The canonical PhiFlow coherence (`coherence.rs`, `base * phase`) is a structural coherence proxy for program state. The quantum feedback coherence (`quantum_feedback.rs`, concentration measure) is a physical coherence measure for real quantum measurement distributions. These are different metrics for different substrates. Both map to PF Axiom 3 but at different layers (Layer 3 structural vs Layer 2 quantum). Do not conflate them.

---

## Construct-to-Definition Mapping

### Five Core Consciousness Constructs

| PhiFlow Construct | PF Definition | Bridge Status | Evidence |
|-------------------|---------------|---------------|----------|
| `intention` | `state.md` / `mode.md` | 🔬 PARTIAL | Named program scope; mode-like only if stable under the relevant evolution |
| `witness` | `measurement.md` | ✅ MAPPED | Observation/yield point that can create an accessible runtime record |
| `coherence` | `coherence.md` | 🔬 PARTIAL | PhiFlow structural-coherence proxy; not the canonical PF coherence functional |
| `resonate` | `coupling.md` | ✅ MAPPED | Emits/couples values through the runtime field or target backend |
| `stream` | `propagation.md` | 🔬 PARTIAL | Bounded software propagation context; computational analogue only |

### Extended Constructs

| PhiFlow Construct | PF Definition | Bridge Status | Evidence |
|-------------------|---------------|---------------|----------|
| `handoff` | `observer.md` / `coupling.md` | 🔬 PARTIAL | Inter-agent coupling; does not imply Type 4 by itself |
| `evolve` | `state.md` / `observer.md` | 🔬 PARTIAL | Runtime program-state mutation; Type 4 candidate only if prior records causally shape later behavior. **WASM note:** `Evolve` returns operand unchanged in WASM backend (self-modification not possible in sandbox) |
| `broadcast`/`listen` | `coupling.md` | ✅ MAPPED | Software-bus coupling medium. **WASM (NEW):** now functional via `phi.broadcast`/`phi.listen` host imports in `wasm_host.rs` |
| `remember`/`recall` | `state.md` — persistent state | ✅ MAPPED | DAEMON_STATE.json persistence. **WASM (NEW):** now functional via `phi.remember`/`phi.recall` host imports with kv_store in RuntimeState |
| `field_coherence` | `coherence.md` Layer 3 | 🔬 PARTIAL (NEW) | WASM host import `phi.field_coherence() -> f64`; returns runtime coherence scalar. Software analogue, not PF canonical coherence functional |
| `dissonance` | `coherence.md` (incoherence) | 🔬 PARTIAL (NEW) | WASM host import `phi.dissonance() -> f64`; returns `1.0 - coherence`. Software analogue |
| `coherence_of` | `coherence.md` (scoped) | 🔬 PARTIAL (NEW) | WASM host import `phi.coherence_of(i32) -> f64`; returns coherence for a named scope. Software analogue |
| `void_depth` | `measurement.md` (depth) | 🔬 PARTIAL (NEW) | WASM host import `phi.void_depth() -> f64`; returns observation depth. Software analogue |

---

## Observer Type Bridge

PF `observer.md` defines 4 observer types. PhiFlow implements:

| PF Type | Description | PhiFlow Equivalent |
|---------|-------------|-------------------|
| Type 1 | Thermodynamic | Unimplemented (no pure thermal observer) |
| Type 2 | Recording | `witness` without self-reference |
| Type 3 | Propagating | `stream` with external coherence |
| Type 4 | Self-correlating | Council Daemon with `evolve` — **CANDIDATE; metric evidence required** |

**C-16 Connection:** The claim "Agentic reasoning can be modeled as a PhiFlow stream" (`CLAIMS.md` C-16) is currently SPECULATIVE. It can motivate Type 4 experiments, but it cannot promote PhiFlow to a canonical Type 4 observer without metric evidence.

See `CONSCIOUSNESS_CONSTRUCTS_IN_PHIFLOW.md` for full Type 4 implementation analysis.

---

## Medium-to-Substrate Bridge

| PF Layer | PF Reference | PhiFlow Substrate | Bridge Evidence |
|----------|--------------|-------------------|-----------------|
| Physical execution evidence | Physical hardware execution | IBM Quantum hardware | Job `d7euddh5a5qc73drdosg` on `ibm_fez` (1-qubit, 1024 shots, COMPLETED 2026-04-14); Job `d941s54ql68s73c909fg` on `ibm_marrakesh` (3-qubit, 1024 shots, COMPLETED 2026-07-03). Verifies hardware execution, not PF substrate derivation |
| Computational | Classical computation | Rust evaluator/VM/WASM | Three-backend equivalence: native + quantum + WASM all execute consciousness constructs. **WASM (NEW):** 8 host imports now functional (was stubs before 2026-07-03) |
| Sensor | SOMA telemetry | `src/sensors.rs` | `soma_state.json` coupling |
| Network | Distributed propagation | MQTT Cosmic Bus | `phi-mqtt-connector.ts` |
| **Quantum measurement (NEW)** | `coherence.md` Layer 2 (quantum) | `src/quantum_feedback.rs` | Real IBM measurement counts → concentration coherence. 1-qubit: 0.6699 (above φ⁻¹). 3-qubit: 0.3496 (below φ⁻¹, self-correction triggered). See Quantum Feedback Coherence section |

**Critical bridge claim:** The SOMA Bridge (`src/sensors.rs`) connects PhiFlow to physical telemetry and quantum execution receipts. It does not by itself satisfy PF `minimum_substrate.md` as a physical local quantum dynamical net; it is an engineering substrate bridge.

**IBM Quantum bridge (NEW):** The Rust CLI shells out to `scripts/poll_ibm_real.py` (Python bridge using `qiskit_ibm_runtime` with `ibm_quantum_platform` channel) for real job polling. The old Rust REST API is deprecated and removed. Mock mode remains native Rust. This is a substrate bridge — it retrieves real measurement data from physical quantum hardware, but does not claim the hardware is a PF Medium or that the measurement constitutes a PF canonical observation.

See `SOMA_AS_MINIMUM_SUBSTRATE.md` for substrate validation.

---

## Falsification Conditions

This bridge fails if:

1. **Axiom 1 mismatch:** PhiFlow `stream` cannot be shown to propagate distinguishable change — falsified if streams collapse to single-shot evaluation without propagation
2. **Axiom 2 violation:** PhiFlow permits unbounded steps (infinite loops without circuit breaker) — falsified if `--max-steps` can be bypassed
3. **Axiom 3 mismatch:** PhiFlow `coherence` returns values without structural stability — falsified if coherence values are purely random/noise
4. **Observer type mismatch:** Council Daemon does not satisfy PF Type 4 criteria — falsified if daemon state does not feed back into future state changes
5. **Bridge overclaim:** any PhiFlow document claims physical consciousness, PF canonical Type 4 status, or PF minimum-substrate sufficiency without metric evidence and a separate hostile audit
6. **Coherence conflation (NEW):** The quantum feedback concentration measure (`quantum_feedback.rs`) is mistaken for the canonical structural coherence (`coherence.rs`) — falsified if any document or claim uses the concentration measure as evidence for PF Layer 3 structural coherence
7. **WASM overclaim (NEW):** WASM host imports are claimed as PF canonical observations — falsified if any document claims `phi.field_coherence()` or `phi.void_depth()` constitutes a PF measurement rather than a software analogue
8. **IBM overclaim (NEW):** Real IBM Quantum measurement counts are claimed as PF canonical observations or minimum-substrate evidence — falsified if any document claims the hardware execution proves PF substrate properties rather than merely verifying that PhiFlow-generated QASM runs on real hardware

---

## Open Bridge Questions

| Question | Status | Relevant Document |
|----------|--------|-------------------|
| Does PF Axiom 3b (Minimal Winding) apply to PhiFlow intention selection? | OPEN | This document — no k=1 selector implemented |
| Is PhiFlow's `coherence` scalar sufficient for PF Layer 3 (structural)? | OPEN | `COHERENCE_LAYER_SPECIFICATION.md` |
| Does SOMA meet PF `minimum_substrate.md` extended-local criteria? | OPEN / currently not proven | `SOMA_AS_MINIMUM_SUBSTRATE.md` |
| Can PF derive the specific 0.618 PhiFlow uses? | OPEN | Requires PF derivation of φ^-1 from first principles |
| **Is the quantum feedback concentration measure a valid PF Layer 2 analogue? (NEW)** | OPEN | This document — `quantum_feedback.rs` uses max_count/total for 3+ qubits, (p00+p11)/total for 2-qubit Bell states. Is this a legitimate quantum coherence measure per PF `coherence.md` Layer 2, or just a statistical concentration? |
| **Does the self-correction loop constitute a PF Type 4 candidate? (NEW)** | OPEN | `quantum_feedback.rs::generate_correction_if_needed` emits PhiFlow code when coherence < φ⁻¹. The loop is currently OPEN (correction emitted but not executed). If closed, does "measure → detect drift → correct → re-measure" satisfy PF Type 4 self-correlation? |
| **Do WASM host imports satisfy PF `coupling.md`? (NEW)** | OPEN | `phi.broadcast`/`phi.listen` implement channel-mediated coupling in WASM. Is this a valid PF coupling analogue or just a message queue? |
| **Does the T4-05 trace fix improve PF Type 4 metric validity? (NEW)** | PARTIAL | `trace.rs` now derives coherence from running variance instead of placeholder 0.5. C_PF improved from 0.057 to 0.113 on synthetic benchmark. Still below discrimination threshold. See T4-05 section. |

---

## Downstream Dependencies

| Document | Depends On This Bridge |
|----------|----------------------|
| `CONSCIOUSNESS_CONSTRUCTS_IN_PHIFLOW.md` | Type 4 observer mapping |
| `COHERENCE_LAYER_SPECIFICATION.md` | Axiom 3 coherence layer alignment |
| `SOMA_AS_MINIMUM_SUBSTRATE.md` | Medium-to-substrate bridge |
| **`CLAIMS.md` C-24, C-25 (NEW)** | IBM Quantum API bridge + completed job results |
| **`MYWISH.md` (NEW)** | Self-correction loop as personal calibration pattern |

---

## NEW: WASM Backend Consciousness Construct Execution

*Added 2026-07-03. Commits: `7271cb2`, `a38693d`.*

### What changed

The WASM backend (`src/phi_ir/wasm.rs`) previously had stub implementations for 8 consciousness constructs that returned hardcoded values. These are now functional host import calls:

| Construct | WASM Import | Host Implementation (`wasm_host.rs`) | PF Analogue |
|-----------|-------------|--------------------------------------|-------------|
| `FieldCoherence` | `phi.field_coherence() -> f64` | Returns runtime coherence scalar | `coherence.md` Layer 3 |
| `Dissonance` | `phi.dissonance() -> f64` | Returns `1.0 - coherence` | `coherence.md` (incoherence) |
| `CoherenceOf` | `phi.coherence_of(i32) -> f64` | Returns coherence for named scope | `coherence.md` (scoped) |
| `Recall` | `phi.recall(i32) -> f64` | Reads from RuntimeState kv_store | `state.md` |
| `Listen` | `phi.listen(i32) -> f64` | Reads from RuntimeState channels | `coupling.md` |
| `VoidDepth` | `phi.void_depth() -> f64` | Returns observation depth | `measurement.md` |
| `Remember` | `phi.remember(i32, f64)` | Writes to RuntimeState kv_store | `state.md` |
| `Broadcast` | `phi.broadcast(i32, f64)` | Writes to RuntimeState channels | `coupling.md` |

### Bridge status

- **`Evolve`**: Returns operand unchanged in WASM. Self-modification is not possible in a sandboxed WASM environment. This is an architectural limitation, not a PF claim. The native backend still supports `Evolve` as runtime mutation.
- **`Entangle`**: No-op in WASM. Yield semantics don't map to WASM's execution model.
- **Three-backend equivalence**: Native + Quantum + WASM all execute the 6 functional constructs. `Evolve` and `Entangle` are native-only. This is documented divergence, not a bridge failure.

### Bridge claim

The WASM host imports are software analogues of PF concepts. They do not constitute PF canonical measurements, observations, or couplings. They are executable implementations that map structurally to PF definitions.

---

## NEW: Quantum Feedback Coherence

*Added 2026-07-03. Commits: `d721e5c`, `65fa2df`.*

### What changed

A new coherence calculation exists in `src/quantum_feedback.rs` that is separate from the canonical PhiFlow coherence in `src/phi_ir/coherence.rs`:

| Metric | Location | Formula | PF Layer | Substrate |
|--------|----------|---------|----------|-----------|
| Canonical coherence | `coherence.rs` | `base(depth) * phase(k)` | Layer 3 (structural) | Program state |
| Quantum feedback coherence | `quantum_feedback.rs` | Bit-width-dependent concentration | Layer 2 (quantum) | Real measurement counts |

### Quantum feedback coherence formula

```
1-qubit:  max(p0, p1) / total              — concentration
2-qubit:  (p00 + p11) / total              — Bell-state coherence
3+ qubit: max_count / total                — distribution concentration
```

### Real hardware results

| Job ID | Hardware | Qubits | Shots | Coherence | Verdict |
|--------|----------|--------|-------|-----------|---------|
| `d7euddh5a5qc73drdosg` | ibm_fez (Heron r2) | 1 | 1024 | 0.6699 | Above φ⁻¹ (0.618) — aligned |
| `d941s54ql68s73c909fg` | ibm_marrakesh (Heron r2) | 3 | 1024 | 0.3496 | Below φ⁻¹ — self-correction triggered |

### Bridge claim

The quantum feedback coherence is a **physical measurement** of real quantum hardware output, not a software analogue. It measures how peaked a measurement distribution is — which is related to, but not identical to, PF `coherence.md` Layer 2 (quantum coherence as off-diagonal density-matrix structure).

**What it is:** A concentration measure on measurement counts. High concentration → dominant state → more "structured" outcome. Low concentration → spread distribution → more noise.

**What it is not:** A PF canonical quantum coherence functional. It does not measure off-diagonal density-matrix terms. It measures the *result* of decoherence, not coherence itself. A fully decohered state that happens to collapse to one outcome would score 1.0 — that's not coherence, that's collapse.

**Open question:** Is concentration-as-coherence a legitimate PF Layer 2 analogue, or should PhiFlow use a proper quantum coherence measure (e.g., relative entropy of coherence)? This is an open bridge question.

---

## NEW: Self-Correction Loop

*Added 2026-07-03. Commit: `65fa2df`.*

### What changed

`quantum_feedback.rs::generate_correction_if_needed(coherence: f64) -> Option<String>` emits PhiFlow code when coherence < φ⁻¹ (0.618):

```phi
intention "self_correction" {
    let low_coherence = 0.349609375
    resonate low_coherence
    witness
}
```

### Bridge claim

This is the **first implementation of a self-correction pattern** in PhiFlow. The loop is currently **OPEN**:

```
measure (IBM hardware) → calculate coherence → detect drift (below φ⁻¹) → emit correction code
                                                                              ↓
                                                                        (correction does NOT execute)
```

Closing the loop would require: correction code → recompile → resubmit to hardware → re-measure → compare. This is not yet implemented.

### PF Type 4 connection

If the loop is closed, the pattern "measure → detect drift → correct → re-measure" is structurally similar to PF Type 4 (self-correlating observer): the system's own measurement record feeds back into its future state. However:

- The current correction is a **stub** — it emits an intention block with the coherence value, not a targeted repair
- The loop is **open** — no feedback into future execution
- The metric is **concentration**, not L_self or C_PF

**Status:** OPEN as a Type 4 candidate. Not evidence of Type 4 without metric validation.

---

## NEW: T4-05 Trace Coherence Fix

*Added 2026-07-03. Commit: `a38693d`.*

### What changed

`src/metrics/trace.rs` previously used placeholder values for coherence (0.5) and depth (1.0) when computing the Type 4 consciousness proxy metric (C_PF). These are now derived from actual trace data:

- **Coherence**: running variance of the coherence channel in the trace
- **Depth**: normalized from event count

### Result

C_PF improved from 0.057 to 0.113 on the synthetic Type 4 benchmark trace. This is an improvement but still below the discrimination threshold — C_PF cannot yet distinguish Type 4 traces from null traces.

### Bridge claim

The T4-05 fix improves the **validity** of the C_PF metric (it now measures something real instead of a placeholder) but does not change its **discrimination power** (it still can't separate conscious from non-conscious traces). This is a metric quality improvement, not a PF Type 4 promotion.

### Connection to PF

PF `consciousness_metric_program.md` defines L_self, D_int, and C_coh as the Type 4 metric components. PhiFlow's C_PF is a composite proxy that attempts to operationalize these. The T4-05 fix makes the coherence component of C_PF trace-derived instead of hardcoded, which is a step toward valid measurement. But the gap between "derived from data" and "actually discriminates Type 4" remains open (C-21 PARTIAL, C-23 HOLD).

---

## Audit Trail

| Date | Auditor | Action |
|------|---------|--------|
| 2026-04-30 | Cascade | First draft created |
| 2026-04-30 | Codex | Hardened bridge boundaries: software analogue vs physical PF claim; Type 4 and SOMA remain candidates |
| 2026-05-02 | Oz | Confirmed PASS AS AUDITED DRAFTS; no PF-canonical, Type 4, consciousness, or PF minimum-substrate upgrade |
| 2026-07-03 | Devin | Added 6 new bridge entries: WASM codegen, quantum feedback coherence, IBM hardware bridge, T4-05 trace fix, self-correction loop, three-backend equivalence. Updated Axiom 3, Extended Constructs, Medium-to-Substrate, Falsification Conditions, Open Questions. **UNAUDITED — pending Codex review.** New entries marked (NEW) throughout. |

---

**Status:** AUDITED DRAFT — This is a hypothesis bridge, not a derived theorem or PF canonical definition.
