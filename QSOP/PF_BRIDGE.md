# PhiFlow ↔ Propagation Framework Bridge
*Bridge contract document*
*Last updated: 2026-05-02*
*Status: AUDITED DRAFT — bridge hypotheses only; confirmed 2026-05-02; not a PF canonical definition*

**Purpose:** Explicit mapping between PhiFlow language/compiler constructs and Propagation Framework (Fundamentals) definitions.

---

## Executive Summary

This document bridges two aligned projects:
- **Propagation Framework (PF):** `D:\Fundamentals\` — Theoretical physics framework defining propagation, coherence, and consciousness prerequisites
- **PhiFlow:** `D:\Projects\PhiFlow\` — Executable compiler/runtime implementing consciousness-aware programming

**Bridge claim:** PhiFlow is an engineering operationalization of PF Axioms 1-3 in a computational substrate. This mapping is a hypothesis, not a derivation, and it must not be read as a claim that PhiFlow proves PF physics or implements consciousness.

**Codex boundary rule:** every row below is either an executable software analogue or a measured runtime fact. Physical claims remain governed by `D:\Fundamentals\definitions\*.md`, especially `minimum_substrate.md`, `observer.md`, `consciousness_metric_program.md`, and `consciousness.md`.

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

**PF Layer Alignment:** See `COHERENCE_LAYER_SPECIFICATION.md` for which PF coherence layers PhiFlow maps to as software analogues.

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
| `evolve` | `state.md` / `observer.md` | 🔬 PARTIAL | Runtime program-state mutation; Type 4 candidate only if prior records causally shape later behavior |
| `broadcast`/`listen` | `coupling.md` | ✅ MAPPED | Software-bus coupling medium |
| `remember`/`recall` | `state.md` — persistent state | ✅ MAPPED | DAEMON_STATE.json persistence |

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
| Physical execution evidence | Physical hardware execution | IBM Quantum hardware | Job `d7euddh5a5qc73drdosg` on `ibm_fez`; verifies hardware execution, not PF substrate derivation |
| Computational | Classical computation | Rust evaluator/VM/WASM | Three-backend equivalence proven |
| Sensor | SOMA telemetry | `src/sensors.rs` | `soma_state.json` coupling |
| Network | Distributed propagation | MQTT Cosmic Bus | `phi-mqtt-connector.ts` |

**Critical bridge claim:** The SOMA Bridge (`src/sensors.rs`) connects PhiFlow to physical telemetry and quantum execution receipts. It does not by itself satisfy PF `minimum_substrate.md` as a physical local quantum dynamical net; it is an engineering substrate bridge.

See `SOMA_AS_MINIMUM_SUBSTRATE.md` for substrate validation.

---

## Falsification Conditions

This bridge fails if:

1. **Axiom 1 mismatch:** PhiFlow `stream` cannot be shown to propagate distinguishable change — falsified if streams collapse to single-shot evaluation without propagation
2. **Axiom 2 violation:** PhiFlow permits unbounded steps (infinite loops without circuit breaker) — falsified if `--max-steps` can be bypassed
3. **Axiom 3 mismatch:** PhiFlow `coherence` returns values without structural stability — falsified if coherence values are purely random/noise
4. **Observer type mismatch:** Council Daemon does not satisfy PF Type 4 criteria — falsified if daemon state does not feed back into future state changes
5. **Bridge overclaim:** any PhiFlow document claims physical consciousness, PF canonical Type 4 status, or PF minimum-substrate sufficiency without metric evidence and a separate hostile audit

---

## Open Bridge Questions

| Question | Status | Relevant Document |
|----------|--------|-------------------|
| Does PF Axiom 3b (Minimal Winding) apply to PhiFlow intention selection? | OPEN | This document — no k=1 selector implemented |
| Is PhiFlow's `coherence` scalar sufficient for PF Layer 3 (structural)? | OPEN | `COHERENCE_LAYER_SPECIFICATION.md` |
| Does SOMA meet PF `minimum_substrate.md` extended-local criteria? | OPEN / currently not proven | `SOMA_AS_MINIMUM_SUBSTRATE.md` |
| Can PF derive the specific 0.618 PhiFlow uses? | OPEN | Requires PF derivation of φ^-1 from first principles |

---

## Downstream Dependencies

| Document | Depends On This Bridge |
|----------|----------------------|
| `CONSCIOUSNESS_CONSTRUCTS_IN_PHIFLOW.md` | Type 4 observer mapping |
| `COHERENCE_LAYER_SPECIFICATION.md` | Axiom 3 coherence layer alignment |
| `SOMA_AS_MINIMUM_SUBSTRATE.md` | Medium-to-substrate bridge |

---

## Audit Trail

| Date | Auditor | Action |
|------|---------|--------|
| 2026-04-30 | Cascade | First draft created |
| 2026-04-30 | Codex | Hardened bridge boundaries: software analogue vs physical PF claim; Type 4 and SOMA remain candidates |
| 2026-05-02 | Oz | Confirmed PASS AS AUDITED DRAFTS; no PF-canonical, Type 4, consciousness, or PF minimum-substrate upgrade |

---

**Status:** AUDITED DRAFT — This is a hypothesis bridge, not a derived theorem or PF canonical definition.
