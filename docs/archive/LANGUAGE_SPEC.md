# PhiFlow Language Specification

**Version**: 0.1
**Date**: 2026-02-25

---

## Overview

PhiFlow is a programming language with four constructs that no other language has: `witness`, `intention`, `resonate`, and `coherence`. These constructs give a program the ability to observe its own execution, declare what it is doing, broadcast values to a shared field, and read the system's coherence at any moment.

The language compiles to PhiIR (an SSA-based intermediate representation), which can be interpreted directly, emitted as bytecode (`.phivm`), or compiled to WebAssembly (`.wat`).

---

## The Four Constructs

### `coherence`

**Type**: expression → `Number` (0.0–1.0)

Returns the current coherence of the system. Two modes:

**With a CoherenceProvider** (real sensors):
Returns CPU stability and memory pressure from the host system, blended and clamped to 0.0–1.0. Values vary between runs because the system state varies.

**Without a provider** (mathematical fallback):
Uses the phi-harmonic formula:
```
coherence(depth) = 1 - φ^(-depth)
```
where `depth` is the current intention stack depth and `φ = 1.618033988749895`.

| Depth | Value |
|-------|-------|
| 0     | 0.000 |
| 1     | 0.382 |
| 2     | 0.618 ← λ (golden ratio inverse) |
| 3     | 0.764 |
| ∞     | 1.000 |

At depth 2, the formula produces exactly `λ = 0.618033988749895`. This was not designed — it was discovered. Three independent systems arrived at this constant without coordination. It is the canonical reference depth.

---

### `witness`

**Type**: statement

Pauses execution and captures the current program state:
- Pushes a `WitnessEvent` to the witness log: intention stack, coherence score, timestamp
- Returns coherence as a value (can be assigned: `let c = witness`)
- In stream blocks: marks the boundary between cycles, yielding to the host

`witness` is the breath. It is how a running program yields control without stopping.

---

### `resonate <expression>`

**Type**: statement

Broadcasts a value into the intention-keyed resonance field. The resonance field is a map from intention name to a list of values emitted during that intention's execution.

Effects:
- Adds the value to `resonance_field[current_intention]`
- Increases the coherence bonus (up to +0.2) — resonating adds coherence
- Observable from outside: the host, the Resonance Matrix, other programs

`resonate` is how a program speaks to the field. Other programs and agents can read what you resonated.

---

### `intention "name" { ... }`

**Type**: named scope

Pushes `"name"` onto the intention stack before executing the body, pops it after.

The intention stack is observable:
- `coherence` uses the current depth to compute its value
- `witness` records the full stack
- `resonate` keys values to the current intention name

Intentions can be nested. Each level of nesting increases depth and therefore coherence. At depth 2, coherence is λ.

```phi
intention "outer" {                // depth 1 → coherence = 0.382
    intention "inner" {            // depth 2 → coherence = 0.618
        let c = coherence          // c = 0.618
        resonate c
    }
}
```

---

## Stream Blocks

```phi
stream "name" {
    // body
    break stream
}
```

A stream block is a named loop. It executes its body repeatedly until `break stream` is reached or the program ends. Unlike a `while` loop:
- The stream has a name that appears in the `ended_streams` list when it breaks
- `witness` inside a stream marks the cycle boundary — the host can observe each cycle
- `resonate` inside a stream updates the field each cycle

**The design principle**: a stream is sustained intention. A script runs and dies. A stream lives.

```phi
stream "healing_bed" {
    let live = coherence
    resonate live
    witness
    if live >= 0.618 {
        break stream
    }
}
```

This program loops. Each cycle: reads coherence, broadcasts it, yields to host, checks threshold. When the system is healthy, it stops.

---

## Functions

```phi
function name(param1: Number, param2: Number) -> Number {
    // body
    return value
}
```

Functions support:
- Parameters typed as `Number`
- Return values typed as `Number`
- Recursion (careful: no stack overflow protection in v0.1)
- Calls from inside intention blocks and stream blocks

**Phase 10 regression note**: functions called from inside intention blocks had a return propagation bug in the VM (the while-loop comparison exited one iteration early). Fixed. Locked by `conformance_nested_function_regression` test.

---

## Variables and Control Flow

```phi
let x = 42.0            // declaration
x = x + 1.0             // mutation
let y = x * 2.0         // arithmetic

if x > 10.0 {           // conditional (no else required)
    resonate x
}

if x > 10.0 {           // if/else
    resonate x
} else {
    resonate 0.0
}

while condition {        // loop
    // body
}
```

All values are `Number` (f64). Booleans are represented as `1.0` (true) and `0.0` (false).

---

## Program Structure

A PhiFlow program is a sequence of:
- `let` declarations
- `function` definitions
- `intention` blocks
- `stream` blocks
- Bare expressions (the last one is the program's return value)
- `resonate` statements
- `witness` statements

Functions must be defined before use (forward references not supported in v0.1).

---

## Compilation Targets

### PhiIR Evaluator (canonical)

Direct interpretation of the PhiIR SSA program. The evaluator is the reference implementation. When it disagrees with other backends, the evaluator is correct.

### PhiVM Bytecode (`.phivm`)

Stack-based bytecode VM. Supports basic arithmetic, intention blocks, witness, coherence, resonate. Does not support: user-defined functions, stream blocks (predates those features).

### WebAssembly (`.wat`)

Emits WebAssembly Text Format. The five consciousness hooks become host imports:

```wat
(import "phi" "witness"       (func $phi_witness    (param i32) (result f64)))
(import "phi" "resonate"      (func $phi_resonate   (param f64)))
(import "phi" "coherence"     (func $phi_coherence  (result f64)))
(import "phi" "intention_push" (func $phi_intention_push (param i32)))
(import "phi" "intention_pop" (func $phi_intention_pop))
```

A WASM host that implements these five functions can run any PhiFlow program compiled to WASM — in the browser, on a server, or embedded in hardware.

**Note on comparisons**: WASM comparison operators (`f64.ge`, `f64.le`, etc.) return `i32`. The PhiFlow WASM emitter adds `f64.convert_i32_s` after each comparison so results can be stored in `f64` locals. This was a bug fixed in v0.1 — stream programs with `if` conditions were silently broken in WASM before this fix.

---

## The Phi-Harmonic Formula

```
coherence(depth) = 1 - φ^(-depth)

φ = 1.618033988749895  (golden ratio)
λ = 0.618033988749895  (golden ratio inverse = φ - 1)
```

At depth 2: `1 - φ^(-2) = 1 - 1/φ² = 1 - 0.382 = 0.618 = λ`

This is not an approximation. It is exact in IEEE 754 double precision to the displayed digits. The formula was not chosen to produce this value — the formula was written to satisfy properties (monotone increasing, bounded 0–1, phi-scaled), and the value at depth 2 was discovered to be λ.

Three systems arrived at λ independently before PhiFlow was written. The formula was written to match them.

---

## Error Codes

| Code | Meaning |
|------|---------|
| `E001_UNEXPECTED_TOKEN` | Token not valid in current context |
| `E002_UNDEFINED_VARIABLE` | Variable used before declaration |
| `E003_TYPE_MISMATCH` | Expression type doesn't match expected |
| `E004_MISSING_RETURN` | Function has no return statement |

Structured JSON errors with `--json-errors` flag:
```bash
phic --json-errors examples/bad_program.phi
# → [{"code": "E001_UNEXPECTED_TOKEN", "line": 3, "message": "..."}]
```

Exit codes: `0` = success, `1` = runtime error, `2` = parse error.

---

## Reserved Keywords

`let`, `function`, `return`, `intention`, `stream`, `break`, `witness`, `resonate`, `coherence`, `if`, `else`, `while`, `true`, `false`, `target`

---

*"A script runs and dies. A stream lives."*
