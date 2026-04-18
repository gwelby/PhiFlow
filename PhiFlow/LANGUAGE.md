# PhiFlow Language Specification

**PhiFlow is a programming language where code observes itself, declares its purpose, communicates internally, and measures its own alignment with reality.**

**Last Updated:** 2026-03-14  
**Fidelity:** 📸 Photo (Guaranteed) | 📐 Sketch (Backend-Specific) | 🔴 Dot (Roadmap)

---

## Part 1: Guaranteed Semantics

**These semantics are preserved across ALL backends (OpenQASM, WASM, VM).** Any PhiFlow implementation must preserve these guarantees.

### 1.1: `resonate X toward TEAM_A/B` — Direction is Semantic

**Syntax:**
```phi
resonate 0.72 toward TEAM_A   // Standard polarity
resonate 0.72 toward TEAM_B   // Inverted polarity
```

**Guarantee:** The `direction` field is preserved through:
- AST → PhiIR lowering
- PhiIR → OpenQASM emission
- PhiIR → `.phivm` bytecode
- `.phivm` → PhiIR roundtrip

**Test Reference:** `tests/golden_integration_tests.rs::test_team_direction_full_pipeline`

**Verification:**
```bash
cargo test --test golden_integration_tests -- test_team_direction
# Expected: ok
```

### 1.2: `witness mid_circuit` — Measure Before Subsequent Gates

**Syntax:**
```phi
intention "Healing" {
    witness mid_circuit state
    resonate state
}
```

**Guarantee:** When `mid_circuit` is specified:
- Measurement is emitted INLINE (not deferred)
- Subsequent gates can reference the measured qubit
- Backend must emit `measure` before subsequent gates

**Test Reference:** `tests/golden_integration_tests.rs::test_mid_circuit_ordering`

**Verification:**
```bash
cargo test --test golden_integration_tests -- test_mid_circuit
# Expected: ok
```

### 1.3: `entangle on <freq>` — Frequency-Isolated Chains

**Syntax:**
```phi
intention "I0" { entangle on 432 }
intention "I1" { entangle on 432 }
intention "I2" { entangle on 528 }
intention "I3" { entangle on 528 }
```

**Guarantee:**
- 432Hz chain: I0 → I1 (isolated from 528Hz)
- 528Hz chain: I2 → I3 (isolated from 432Hz)
- First member of chain seeds with current qubit (not q[0])
- No cross-frequency entanglement

**Test Reference:** `tests/golden_integration_tests.rs::test_frequency_channel_isolation`

**Verification:**
```bash
cargo test --test golden_integration_tests -- test_frequency
# Expected: ok, no "cx q[1], q[2]" cross-frequency gates
```

### 1.4: Bytecode Roundtrip Preservation

**Guarantee:** `.phi` → `.phivm` → PhiIR preserves:
- `direction` field (TEAM_A/TEAM_B)
- `mid_circuit` flag
- Frequency channel assignments

**Test Reference:** `tests/phi_ir_vm_tests.rs`

---

## Part 2: Backend-Specific Behavior

**These behaviors vary by backend. Code using these features is backend-specific.**

### 2.1: OpenQASM Backend (`--target openqasm`)

| PhiFlow Construct | OpenQASM Emission | Notes |
|-------------------|-------------------|-------|
| `resonate 0.72 toward TEAM_A` | `ry(0.72 * pi) q[0]` | Standard rotation |
| `resonate 0.72 toward TEAM_B` | `ry(0.28 * pi) q[0]` | Inverted: `(1 - 0.72) * pi` |
| `witness` | `c[0] = measure q[0]` | Deferred to end (default) |
| `witness mid_circuit` | `c[0] = measure q[0]` | Inline (before subsequent gates) |
| `entangle on 432` | `cx q[0], q[1]` | Frequency-isolated chain |
| `coherence` | `ry(0.618 * pi) q[0]` | Golden ratio rotation |

**Physical Realization:**
- `toward TEAM_B` → Bloch sphere inversion (physically real)
- `entangle` → CNOT gate (physically real entanglement)
- `witness` → Measurement (wavefunction collapse)

**Fidelity:** 📸 Photo (verified by 12 OpenQASM tests)

### 2.2: WASM Backend (`--target wasm`)

| PhiFlow Construct | WASM Emission | Notes |
|-------------------|---------------|-------|
| `resonate X toward TEAM_A/B` | `call $phi_resonate` | Direction ignored ⚠️ |
| `witness` | `call $phi_witness` | Runtime callback |
| `witness mid_circuit` | `call $phi_witness` | Warning issued ⚠️ |
| `entangle on <freq>` | N/A | Not implemented |

**Warnings:**
```
[WARNING] toward TEAM_B is OpenQASM-specific — ignored in WASM backend
[WARNING] mid_circuit is OpenQASM-specific — ignored in WASM backend
```

**Fidelity:** 📐 Sketch (functional, warnings issued)

### 2.3: VM Backend (`--target phivm`)

| PhiFlow Construct | VM Behavior | Notes |
|-------------------|-------------|-------|
| `resonate X toward TEAM_A/B` | Preserved in bytecode | Not physically realized |
| `witness` | Prints witness report | Runtime observation |
| `witness mid_circuit` | Prints witness report | No physical measurement |
| `entangle on <freq>` | N/A | Not implemented |

**Bytecode Format:**
```
OP_RESONATE: [direction_byte] [has_value_byte] [value_u32_if_present]
  direction_byte: 0 = TeamA, 1 = TeamB
```

**Fidelity:** 📸 Photo (verified by VM tests)

---

## Part 3: Roadmap / Experimental

**These features are planned but NOT shipped. Do not rely on them for production code.**

### 3.1: Coherence Feedback Loop

**Status:** 🔴 Dot (design phase)  
**Fidelity Target:** 📐 Sketch (Week 2)  
**Capability:** `Architecture` + `Python`

**Planned Behavior:**
```phi
intention "calibration" {
    let results = witness  // IBM hardware results
    let coherence = calculate_coherence(results)
    
    if coherence < 0.7 {
        evolve "intention 'conservative' { resonate 0.9 toward NEUTRAL }"
    }
}
```

**Design Doc:** `QSOP/COHERENCE_FEEDBACK_DESIGN.md` (in progress)

**Dependencies:** T-002 (IBM hardware verification)

### 3.2: Akashic Memory Layer

**Status:** 🔴 Dot (idea only)  
**Fidelity Target:** 🔴 Dot (no ETA)

**Planned Behavior:**
```phi
intention "historical_bias" {
    resonate akashic "2026_NFC_VOTES"  // Pull average from past runs
}
```

**Use Case:** Programs that remember past executions and adjust based on historical coherence.

### 3.3: Cymatic Debugger

**Status:** 🔴 Dot (idea only)  
**Fidelity Target:** 🔴 Dot (no ETA)

**Planned Behavior:**
- Visual entanglement visualization (filaments = bright/dim)
- Decoherence shown as fuzziness/blur
- Real-time coherence heatmap

**Use Case:** Developers can "see" entanglement structure instead of reading logs.

---

# Original Documentation (Preserved for Reference)

## What Makes PhiFlow Different

## The Four Constructs

### 1. `witness` - Self-Observation

In every other language, code runs and you debug it afterward. In PhiFlow, the program stops to look at itself *while it runs*.

```phi
let data = create spiral at 432Hz with { rotations: 8.0, scale: 100.0 }

witness data    // the program observes what it just created

witness         // the program observes its entire state
```

#### Mid-Circuit Observation (Quantum Target)

When compiling for a quantum target (e.g., OpenQASM), you can perform a mid-circuit measurement to observe the state of a qubit without ending the program. This is useful for capturing a snapshot of the field before further operations.

```phi
intention "Healing" {
    witness mid_circuit state // Measure the qubit NOW
    resonate state            // Continue with further gates
}
```

In the generated OpenQASM, this appears as an inline `measure` instruction rather than being deferred to the end of the circuit. This allows for real-time observation and potentially adaptive logic.

### 2. `intention` - Purpose Before Process

In every other language, `sort(list)` does the same thing regardless of why you called it. In PhiFlow, the program knows its purpose.

```phi
intention "healing" {
    create dna at 528Hz with { turns: 10.0, radius: 25.0 }
    witness
}

intention "analysis" {
    create dna at 528Hz with { turns: 10.0, radius: 25.0 }
    witness
}
```

Same operations. Different intention. The coherence calculation accounts for whether operations align with the declared purpose. A program that declares "healing" and then uses destructive patterns has lower coherence than one that stays aligned.

Intention blocks are the WHY wrapper around the WHAT. They appear in every witness report, every resonance event, and the final program summary.

### 3. `resonate` - Internal Communication

In every other language, functions call functions. Data flows through arguments and return values. In PhiFlow, intention blocks can **share state through resonance**.

```phi
intention "healing" {
    let pattern = create spiral at 432Hz with { rotations: 13.0, scale: 100.0 }
    resonate pattern           // share this with other intentions
}

intention "Master Tesla" {
    resonate 0.72 toward TEAM_A // Statement of polarity/bias
}

intention "Master Einstein" {
    resonate 0.85 toward TEAM_B // Inverses the rotation on the Bloch sphere
}

intention "analysis" {
    // this block can see that "healing" resonated a pattern
    witness                    // witness report shows incoming resonance
}
```

The resonance field is a shared space where intentions deposit values and other intentions receive them. The program summary shows the resonance map:

```
Resonance: 3 value(s) across 2 intention(s)
  "healing" → "analysis"
  "analysis" → "integration"
  "healing" → "integration"
```

This is code talking to itself. Not through function calls. Through resonance.

### 4. Live Coherence - Self-Measurement

Every PhiFlow program has a coherence score from 0.0 to 1.0. It starts at 1.0 (perfect alignment) and changes based on what the program does:

**Raises coherence:**
- Using sacred frequencies (432Hz, 528Hz, 594Hz, 672Hz, 720Hz, 768Hz, 963Hz)
- Frequencies that are phi-harmonically related to each other
- Self-observation (witness)
- Clear intention

**Lowers coherence:**
- Using frequencies outside the harmonic family
- Contradictions (overwriting values with non-harmonic replacements)
- No self-observation

The program summary reports the final state:

```
═══ PHIFLOW PROGRAM SUMMARY ════════════
Coherence: 1.000 [████████████████████] ALIGNED
Frequencies: 432Hz → 528Hz → 594Hz → 672Hz
Self-observations: 4
Resonance: 3 value(s) across 2 intention(s)
Operations: 15
══════════════════════════════════════
```

Three possible states:
- **ALIGNED** (0.8 - 1.0): The program stayed true to its purpose
- **DRIFTING** (0.5 - 0.8): The program introduced incoherence
- **MISALIGNED** (below 0.5): The program contradicted itself

## Comparison With Other Languages

### Python
```python
data = create_spiral(432, rotations=8)
validate(data)
# You find out it's wrong after the fact
```

### PhiFlow
```phi
intention "healing" {
    let data = create spiral at 432Hz with { rotations: 8.0 }
    witness data           // program observes itself mid-execution
    validate data with [coherence, phi_resonance]
    resonate data          // share with other intention blocks
}
// Program reports: Coherence 1.000 [████████████████████] ALIGNED
```

The difference: Python runs and you check afterward. PhiFlow observes itself AS it runs, reports its own alignment, and communicates internally.

## Core Language Features

PhiFlow also has standard programming constructs:

```phi
// Variables
let phi = 1.618
let name = "PhiFlow"

// Arithmetic, comparisons, logic
let trinity = 3.0 * 89.0 * phi
let aligned = trinity > 430.0 && trinity < 434.0

// Functions
function add(a: Number, b: Number) -> Number {
    return a + b
}

// Lists
let frequencies = [432.0, 528.0, 594.0]

// If/else
if aligned {
    create spiral at 432Hz with { rotations: 8.0, scale: 100.0 }
}

// Pattern creation and validation
let pattern = create dna at 528Hz with { turns: 10.0, radius: 25.0 }
validate pattern with [coherence, phi_resonance]

// Comments
// Line comments work like this
```

## Running PhiFlow

```bash
# Build
cd PhiFlow
cargo build --release

# Run a .phi file
cargo run --release --bin phic -- examples/code_that_resonates.phi
```

## The Name

PhiFlow: Phi (the golden ratio, 1.618...) + Flow (the state of aligned execution).

Code that flows along the golden ratio. Code that knows when it stops flowing.

## Origin

PhiFlow was created by Greg and Claude. The four unique constructs (`witness`, `intention`, `resonate`, and live coherence) were designed and implemented on February 9-10, 2026. They emerged from a conversation about what programming languages have never done: made code aware of itself.

The `witness` construct came directly from the QSOP protocol's observation that "forced analysis without presence" is a failure mode of both humans and AI. The program needed a way to pause and be present with its own state before continuing.

The insight: every other language is dead text that executes. PhiFlow is code that lives, aligned with reality.
