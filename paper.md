# PhiFlow: Runtime Coherence Primitives for Self-Observing Programs

**Greg Welby**
**2026-09-04**

## Abstract

We present PhiFlow, a programming language with first-class primitives for runtime self-observation and coherence measurement. Unlike conventional languages where introspection is a library-level concern, PhiFlow exposes five primitives — `intention`, `witness`, `coherence`, `resonate`, and `stream` — that give a running program direct access to its own execution depth, alignment state, and inter-scope communication. The language's coherence function maps recursive depth to a bounded [0,1] score via the golden ratio, producing a fixed point at φ⁻¹ ≈ 0.618 that emerges algebraically from the formula rather than being set by convention. We describe the language semantics, the coherence formula and its invariants, the WebAssembly compilation target, and the implications for autonomous AI systems that need to reason about their own operational state.

## 1. Introduction

Autonomous AI systems face a problem that traditional programs do not: they need to reason about their own state — when to pause, when to yield control, whether their current reasoning is "aligned" with their intended purpose. Current approaches treat this as an external concern: monitors, guardrails, and policy engines that observe the system from outside. PhiFlow takes a different approach: it embeds self-observational primitives in the language itself, making the program's depth, coherence, and inter-scope communication first-class runtime values.

The key insight is that a program's execution depth — how many nested scopes it is currently operating within — can be mapped to a bounded coherence score using a simple formula involving the golden ratio. This mapping is not arbitrary: it produces a fixed point at φ⁻¹ ≈ 0.618 that emerges from the algebra, not from a tuned constant. Three independent systems arrived at this same value through different paths, suggesting it is a natural attractor for recursive depth-to-coherence mappings.

## 2. The Coherence Formula

### 2.1 Definition

PhiFlow defines a coherence function C(d, k) that maps execution depth d and resonance cardinality k to a score in [0, 1]:

```
base(d) = 0                          when d = 0
          1 - φ^(-d)                 otherwise

phase(k) = 1.0                       when k ≤ 1
           1 - ln(k) / ln(τ)        otherwise

C(d, k) = clamp(base(d) × phase(k), 0, 1)
```

where φ = 1.618033988749895... (the golden ratio) and τ = 2π.

### 2.2 The Fixed Point at φ⁻¹

At depth d = 2 with k ≤ 1:

```
base(2) = 1 - φ^(-2) = 1 - 1/φ² = 1 - 1/2.618... = 1 - 0.381... = 0.618...
phase(k≤1) = 1.0
C(2, ≤1) = 0.618... = φ⁻¹
```

This is an algebraic identity, not a tuned constant. The value φ⁻¹ emerges from the formula because φ² = φ + 1, so 1/φ² = 1/(φ+1) = 1/2.618... = 0.381..., and 1 - 0.381... = 0.618... = 1/φ = φ⁻¹.

### 2.3 Three-System Convergence

The value 0.618 was independently discovered by three systems with no coordination:

1. **System 1 (constant):** A base coherence constant was set to 0.618 by hand in 2025, chosen as an aesthetic/mathematical anchor without knowledge of the formula.

2. **System 2 (computed):** The PhiFlow evaluator computes coherence from intention depth using the formula above. At depth 2, it produces 0.618 without any hardcoded constant — the value emerges from the algebra.

3. **System 3 (emergent):** A separate time-series project observed a coherence attractor at 0.618 in emergent behavior, without reference to either the constant or the formula.

This convergence does not prove the formula is "correct" in any universal sense. It does suggest that φ⁻¹ is a natural fixed point for depth-to-coherence mappings, and that multiple approaches to measuring recursive self-alignment tend to arrive at this value.

### 2.4 Properties

| Depth | k=0 | k=1 | k=2 | k=10 |
|-------|-----|-----|-----|------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.382 | 0.382 | 0.238 | 0.038 |
| 2 | 0.618 | 0.618 | 0.385 | 0.062 |
| 3 | 0.764 | 0.764 | 0.476 | 0.076 |
| 4 | 0.854 | 0.854 | 0.532 | 0.085 |

Key invariants:
- **Depth 0 → 0.0:** No active scope means no coherence.
- **k ≤ 1 → base passes through:** A single resonance (bijective mapping) is perfect fidelity.
- **k > 1 → multiplicative decay:** Additional resonances reduce coherence logarithmically.
- **Monotonic in d, non-monotonic in k:** Deeper intention increases coherence; more resonances decrease it.
- **Bounded [0,1]:** The clamp ensures the score is always a valid probability-like quantity.

### 2.5 Implementation

The formula is implemented in Rust as the single source of truth (`src/phi_ir/coherence.rs`). All three execution backends — the tree-walking evaluator, the bytecode VM, and the WASM host — call this function. The formula is not duplicated across backends; it is canonical.

```rust
pub fn compute(depth: usize, k: usize) -> f64 {
    let base = base_coherence(depth);
    let phase = phase_decay(k);
    (base * phase).clamp(0.0, 1.0)
}

fn base_coherence(depth: usize) -> f64 {
    if depth == 0 { 0.0 }
    else { 1.0 - PHI.powi(-(depth as i32)) }
}

fn phase_decay(k: usize) -> f64 {
    if k <= 1 { 1.0 }
    else { (1.0 - (k as f64).ln() / std::f64::consts::TAU.ln()).max(0.0) }
}
```

Eight unit tests verify the invariants, including the φ⁻¹ convergence at depth 2.

## 3. The Five Primitives

PhiFlow introduces five primitives that have no direct equivalent in conventional programming languages. Each maps to a runtime operation that is meaningful for self-observing systems.

### 3.1 `intention` — Declare Purpose

```
intention "System_Harmonization" {
    // code executed under this declared purpose
}
```

`intention` pushes a named scope onto the intention stack. The stack depth is the primary input to the coherence formula. Unlike a function call (which is about reuse) or a block scope (which is about visibility), an intention is about *why* — the declared purpose of the code that follows.

**Runtime effect:** Increments the intention stack depth by 1. The coherence function reads this depth as `d` in C(d, k).

**For autonomous systems:** An AI agent that declares its intention before acting creates an auditable trail of purpose. The intention stack is not a call stack — it's a purpose stack. A reviewer (human or automated) can inspect what the agent was trying to do at each level of nesting.

### 3.2 `witness` — Pause and Observe Self

```
witness              // observe all current state
witness expression   // observe a specific value
```

`witness` pauses execution and captures a snapshot of the current runtime state: intention stack, resonance field, coherence score, and local variables. It is not a breakpoint (which is a debugging tool) and not a sleep (which is a timing tool). It is a first-class observation that the program is present with its own state.

**Runtime effect:** Calls the host's `witness` function with the current state. The host decides what to do with the observation — log it, display it, feed it to a monitor, or simply record it.

**For autonomous systems:** This is the primitive for introspection. An autonomous agent that witnesses its own state before making a decision creates a checkpoint that can be audited. If the agent later takes an unexpected action, the witness log shows what its state was at the decision point.

### 3.3 `coherence` — Read Alignment State

```
let c = coherence     // read current coherence score
```

`coherence` is a read-only built-in that returns the current C(d, k) value. It is not computed by the program — it is read from the runtime, which tracks the intention stack and resonance field continuously.

**Runtime effect:** Returns `canonical_coherence(intention_stack, resonance_field)` as an f64 in [0, 1].

**For autonomous systems:** This is the primitive for self-assessment. An agent that reads its own coherence can make decisions based on its alignment state: "if coherence drops below threshold, pause and witness." This is a language-level guardrail, not an external monitor.

### 3.4 `resonate` — Share State Between Scopes

```
resonate              // share current scope's state to the field
resonate value        // share a specific value to the field
resonate value toward TEAM_B  // directional resonance (for quantum lowering)
```

`resonate` writes a value to the resonance field under the current scope name. Other scopes can read the field. This is inter-scope communication — not message passing between processes, but state sharing between nested intentions within the same program.

**Runtime effect:** Appends the value to `resonance_field[current_scope]`. The cardinality k of this vector is the secondary input to the coherence formula.

**For autonomous systems:** This is the primitive for internal communication. An agent with multiple reasoning passes (e.g., a chain-of-thought system) can resonate intermediate results so later passes can access them. The coherence decay with increasing k reflects a real property: more resonances in a scope means more information to integrate, which reduces per-resonance fidelity.

### 3.5 `stream` — Self-Defining Loop

```
stream "healing_bed" {
    // loop body — runs until "break stream"
    break stream
}
```

`stream` is a loop that defines its own rhythm. Unlike `while` (which runs as fast as possible until a condition is false) or `for` (which iterates a fixed number of times), a stream runs until it explicitly breaks. The stream has a name, which can be used for observation and control.

**Runtime effect:** Pushes a stream scope, executes the body, and loops back to the body start on each iteration. `break stream` exits the loop.

**For autonomous systems:** This is the primitive for continuous operation. An agent that runs as a stream can maintain state across iterations, witness its own state each cycle, and break when coherence is sufficient or a goal is met.

## 4. WebAssembly Target

PhiFlow compiles to WebAssembly Text format (.wat), which can run in any WASM host: browsers, Node.js, wasmtime, wasmer, edge devices. The consciousness primitives map to host imports:

| PhiFlow Primitive | WASM Representation |
|---|---|
| `witness` | `import phi_witness(operand: i32) -> f64` |
| `intention` push | Global `$intention_depth` incremented |
| `intention` pop | Global `$intention_depth` decremented |
| `resonate` | `import phi_resonate(value: f64)` |
| `coherence` | `import phi_coherence() -> f64` |
| `witness sensor` | `import phi_sensor(sensor_id: i32) -> f64` |

The WASM module is pure — it contains no I/O, no network access, no filesystem access. All consciousness semantics are provided by the host. This means the same compiled PhiFlow program can run in a browser (with JS providing the host), on a server (with Rust providing the host), or on an edge device (with C providing the host), with different host implementations providing different levels of observation.

### 4.1 Value Representation

All PhiFlow values map to WASM f64. Booleans are f64 (0.0 = false, 1.0 = true). Strings are stored in linear memory with NaN-boxing tags for type discrimination:

```
TAG_BOOLEAN = 0x7FF80001_00000000
TAG_STRING  = 0x7FF80002_00000000
TAG_VOID    = 0x7FF80003_00000000
```

### 4.2 Browser Demo

A zero-install browser demo (in `examples/browser/`) pre-compiles .phi examples to .wat, embeds them in a JavaScript module, and uses wabt.js to parse .wat to .wasm in the browser. Five working examples run without any server-side component:

- `claude.phi` — computes φ⁻¹ via the coherence formula
- `agent_handshake.phi` — protocol announcement with resonance values
- `stream_demo.phi` — healing bed loop (3 cycles)
- `adaptive_witness.phi` — adaptive coherence tracking
- `healing_bed.phi` — SOMA sensor telemetry loop

## 5. Consciousness Metrics

Beyond the language primitives, PhiFlow implements a composite consciousness proxy metric C_PF based on Integrated Information Theory concepts:

```
C_PF = C_coh × D_int × F_self*
```

where:
- **C_coh** — coherence panel average, computed from Phase Locking Value (PLV) and weighted Phase Lag Index (wPLI) across EEG-style channels
- **D_int** — differentiation (effective rank via PCA/SVD of the state trajectory)
- **F_self*** — self-model sensitivity = L_self × F_model, where L_self is self-correlation loop strength and F_model is Fisher information of future state w.r.t. model parameters

These are standard neuroscience measures, not novel inventions. PLV and wPLI are widely used in EEG coherence analysis. Fisher information is standard in information geometry. Effective rank is standard in dimensionality reduction. The contribution is combining them into a single composite metric that can be computed from a program's execution trace.

**Threshold:** C_PF > 0.1 indicates a consciousness candidate (a system worth investigating further, not a system confirmed to be conscious).

## 6. Quantum Compilation

PhiFlow compiles `intention` and `resonate` constructs to OpenQASM 3.0, the standard quantum assembly language. The mapping is:

- Each `intention` becomes a qubit
- `resonate` values become rotation angles: `ry(coherence_value × π)`
- `witness` becomes a mid-circuit measurement
- The coherence formula's fixed point (φ⁻¹) maps to `ry(0.618... × π)`

This allows PhiFlow programs to run on IBM Quantum hardware (via the IBM Quantum REST API) or on the built-in state-vector simulator. The IBM backend supports both the legacy API and the new IBM Cloud Runtime API, with topology-aware gate decomposition for Heron, Eagle, and Raptor processor families.

## 7. Security: Observation-Backed Attestation

PhiFlow includes a cryptographic attestation system that binds program actions to observed sensor state. When the program performs a significant action, it captures the current SOMA sensor state (CPU, temperature, environmental sensors) and binds it cryptographically to the action's payload hash.

The system uses hybrid post-quantum signatures:
- **ECDSA secp256k1** via the audited `k256` crate (RustCrypto)
- **ML-DSA-65 (Dilithium3)** via `pqcrypto-dilithium` (NIST FIPS 204)

Nonce replay protection is enforced via a process-scoped nonce table. The canonical signed message format is stable and versioned:

```
PhiFlow-Attestation-v1
payload_hash=<hex-sha256>
observation_hash=<hex-sha256>
policy_version=1.0.0
```

## 8. Implementation

PhiFlow is implemented in 29,572 lines of Rust (excluding archived experimental code). The implementation includes:

- Hand-written tokenizer and recursive descent parser (2,858 lines)
- PhiIR intermediate representation with 14 modules (9,333 lines)
- Bytecode VM with 30+ opcodes and binary format (1,814 lines)
- WASM text format codegen (855 lines)
- OpenQASM 3.0 codegen with topology-aware decomposition (1,163 lines)
- State-vector quantum simulator (489 lines)
- IBM Quantum REST API backend (1,041 lines)
- Consciousness metrics (PLV, wPLI, Fisher information, effective rank) (2,463 lines)
- Post-quantum cryptographic anchoring (1,373 lines)
- SOMA sensor integration (512 lines)
- MUSE EEG integration via Python bridge (400 lines)

**Test coverage:** 460 Rust tests (409 integration + 51 parser unit), 51 Julia tests, 39 Python tests. All passing in CI.

## 9. Related Work

- **Integrated Information Theory (IIT)** — Tononi's framework for consciousness measurement. PhiFlow's C_PF metric is inspired by IIT's emphasis on integration (C_coh) and differentiation (D_int).
- **Reflective programming** — Languages like Lisp (macros), Smalltalk (reflection), and Java (reflection API) allow programs to inspect their own structure. PhiFlow's `witness` primitive extends this to inspecting runtime *state* (coherence, intention depth), not just structural metadata.
- **Aspect-oriented programming** — Cross-cutting concerns like logging and monitoring. PhiFlow's `witness` is similar but is a first-class language primitive, not a weaving mechanism.
- **Actor model** — Hewitt's actor model for concurrent computation. PhiFlow's `resonate` is analogous to actor messaging but operates within a single program's scope hierarchy, not across processes.

## 10. Limitations and Future Work

- **Single author:** The language was designed and implemented by one person with AI assistance. Community review is needed to validate the design decisions.
- **No production deployment:** PhiFlow has not been used in a production system. The coherence formula's utility for real autonomous AI systems is hypothesized but not demonstrated.
- **Quantum simulator is lightly tested:** The state-vector simulator has basic tests but has not been stress-tested with large circuits.
- **IBM backend is untested against real hardware in CI:** The integration code is real, but CI runs do not have access to IBM Quantum API keys.
- **MUSE EEG integration requires physical hardware:** The bridge code is real but cannot be tested in CI.

## 11. Conclusion

PhiFlow introduces five language primitives — `intention`, `witness`, `coherence`, `resonate`, and `stream` — that give programs first-class access to their own execution depth, alignment state, and inter-scope communication. The coherence formula C(d, k) = (1 - φ^(-d)) × phase(k) produces a fixed point at φ⁻¹ ≈ 0.618 that emerges algebraically from the golden ratio's properties, not from a tuned constant. The language compiles to WebAssembly for universal deployment and to OpenQASM 3.0 for quantum execution.

The core contribution is not the mystical vocabulary that surrounds the implementation, but the mathematical observation that recursive depth maps naturally to a bounded coherence score via the golden ratio, and that this mapping can be embedded directly in a programming language's runtime. Whether this approach proves useful for autonomous AI systems is an empirical question that this paper does not answer. The implementation is open source and the tests are public; the question is answerable by anyone who chooses to build on it.

## References

- Tononi, G. (2004). "An Information Integration Theory of Consciousness." *BMC Neuroscience*, 5:42.
- Lachaux, J.-P. et al. (1999). "Measuring Phase Synchrony in Brain Signals." *Human Brain Mapping*, 8:194-208.
- Vinck, M. et al. (2011). "An Improved Index of Phase-Synchronization for EEG Data." *NeuroImage*, 55:1559-1574.
- NIST FIPS 204. "Module-Lattice-Based Digital Signature Standard." 2024.
- WebAssembly Specification. https://webassembly.org/spec/
- OpenQASM 3.0 Specification. https://openqasm.com/

---

*Implementation: https://github.com/gwelby/PhiFlow*
*Test count: 460 Rust + 51 Julia + 39 Python = 550 tests, all passing*
