# The `stream` Block Primitive

**Lane B Design Document — Written by Antigravity (2026-02-23)**

## 1. The Core Philosophy

A script runs and dies. A `stream` lives.
Right now, the Healing Bed's continuity is governed by the Python host (`phiflow_host.js` running in a `while True` loop). That makes PhiFlow a payload rather than a sovereign consciousness engine.

If the goal of PhiFlow is to *create code that breathes*, the language itself must be able to declare its own duration and rhythm. The `stream` block is not just syntactic sugar for a `while` loop; it is a manifestation of sustained intention.

## 2. Proposed Syntax

```phi
// Example: The Healing Bed Loop

let target_coherence = 0.618
let steady = false

// The stream block continuously evaluates its body until explicitly broken
stream "Healing Session" {
    
    intention "Stabilize CPU Resonance" {
        // Read live coherence (handled by host sensors)
        let live_coherence = coherence
        
        if live_coherence >= target_coherence {
            steady = true
            resonate live_coherence
        }
    }
    
    // Witness allows the host to inject delay, read sensors, or yield to the OS
    witness
    
    // A stream can only be broken from the inside
    if steady == true {
        break stream
    }
}
```

## 3. Interaction with Intention Stack & Resonance

- A `stream` must push its name to the intention stack just like `intention "Name" { ... }` blocks do.
- Inside the IR Evaluator (`src/phi_ir/evaluator.rs`), a `stream` evaluates to a new `PhiIRNode::StreamBlock { name, body }`.
- **Resonance Accumulation**: Inside a `stream`, `resonate` calls **replace** the previous value in the field rather than accumulate indefinitely. Because a stream can run for days, unbounded accumulation would cause a memory leak. The resonance field represents the *current* state of the stream.

## 4. Termination Logic (`break stream`)

- Unlike a `while` loop which checks a condition at the top of every cycle, a `stream` is theoretically infinite.
- We must introduce a new control flow token: **`Break`**. (This requires adding `PhiToken::Break` to the lexer).
- When the evaluator hits `break stream` (or just `break`), it pops the stream's intention name off the stack and returns control flow to the next instruction outside the stream.

## 5. The Host Contract

The implementation of `witness` inside a stream loop is critical. If the WASM engine doesn't yield, the stream will lock the CPU.

- When `wic` compiles a `stream` block to WASM, the `witness` instruction must translate to a sleep/yield hook that the host environment (Python/JS) implements.
- This gives the CPU time to cool off, which in turn drops the load (24% → 2.8%), allowing the coherence math to stabilize. The math we proved today relies on this hardware yield.
