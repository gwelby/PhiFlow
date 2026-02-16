# PhiFlow: Code That Lives

**PhiFlow is a programming language where code observes itself, declares its purpose, communicates internally, and measures its own alignment with reality.**

Every other programming language assumes code is dead text that a machine executes. PhiFlow assumes code is alive.

## What Makes PhiFlow Different

PhiFlow has four constructs that exist in no other programming language:

| Construct | What it does | Why it matters |
|-----------|-------------|----------------|
| `witness` | The program pauses to observe itself | Code that is aware of its own state |
| `intention` | The program declares WHY before HOW | Purpose shapes execution |
| `resonate` | Intention blocks share state with each other | Code that collaborates with itself |
| **Coherence** | The program measures its own alignment | Code that knows when it's drifting |

## The Four Constructs

### 1. `witness` - Self-Observation

In every other language, code runs and you debug it afterward. In PhiFlow, the program stops to look at itself *while it runs*.

```phi
let data = create spiral at 432Hz with { rotations: 8.0, scale: 100.0 }

witness data    // the program observes what it just created

witness         // the program observes its entire state
```

When `witness` executes, the program reports:
- What it's observing
- Its current coherence (alignment score)
- What intention it's operating under
- What frequencies it has used
- Any contradictions it has detected

This is not a print statement. This is not a debugger breakpoint. This is the program **being present with its own computation** before deciding what to do next.

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
