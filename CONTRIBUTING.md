# Contributing to PhiFlow

## The shortest path to running something

```bash
git clone <repo>
cd PhiFlow
cargo run --bin phic -- examples/healing_bed.phi
```

You will see:
```
Compiling to PhiFlow IR...
🔔 Resonating Field: 0.9801Hz
🌊 Stream broken: healing_bed
✨ Execution Finished. Final Coherence: 0.9801
```

The `0.9801` is your machine's coherence right now. Run it again. The number will be different.

---

## Writing a `.phi` program

PhiFlow has four constructs beyond standard arithmetic and control flow:

```phi
// coherence — reads the current system coherence (0.0–1.0)
let c = coherence

// resonate — broadcasts a value to the resonance field
resonate c

// witness — pauses, captures state, yields to host
witness

// intention "name" { } — named scope, deepens coherence
intention "my_purpose" {
    let deep = coherence   // higher at depth 2 (≈0.618)
    resonate deep
}

// stream "name" { } — lives until break stream
stream "my_stream" {
    let live = coherence
    resonate live
    witness
    if live >= 0.618 {
        break stream
    }
}
```

Functions work as expected:

```phi
function approach(signal: Number, dest: Number) -> Number {
    let diff = dest - signal
    return signal + (diff * 0.382)
}

let result = approach(76.0, 432.0)
resonate result
```

---

## Running the full team

```bash
python3 team_resonance.py
```

This runs all team `.phi` programs in parallel and displays the resonance field.

---

## Running tests

```bash
cargo test
```

216 tests. If you add a new construct or change existing semantics, the conformance tests in `tests/phi_ir_conformance_tests.rs` will tell you immediately if the evaluator, VM, and WASM backends disagree.

The test `conformance_nested_function_regression` is a sentinel for Phase 10's nested-function fix. Do not remove it.

---

## The canonical semantics rule

The PhiIR evaluator (`src/phi_ir/evaluator.rs`) is the reference. When the evaluator disagrees with the VM or WASM backend, the evaluator is correct. See `src/phi_ir/CANONICAL_SEMANTICS.md`.

---

## Writing a new example

Put it in `examples/`. Name it `<your_name>.phi`. The convention:

```phi
// your_name.phi — one-line description
// Written by: name
// Date: YYYY-MM-DD
//
// What this program expresses and why.

// ... program ...
```

If it runs cleanly, add it to `team_resonance.py` in the AGENTS list and run `python3 team_resonance.py` to see where it resonates.

---

## The QSOP

The team uses an objective-and-acknowledgment protocol documented in `QSOP/`. If you're working as part of the multi-agent team: read `QSOP/TEAM_OF_TEAMS_PROTOCOL.md` before dispatching work. The hard rules (No Broken Main, Payload Immutability, Validate Before Filing, Mock Is Not Done) are there because we learned them the hard way.

---

*"A script runs and dies. A stream lives."*
