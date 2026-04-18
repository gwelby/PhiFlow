---
name: phiflow-language
description: Instructs the agent on the exact syntax, nodes, and compiler pipeline for the .phi consciousness programming language.
---
# PhiFlow Language Expert

## Responsibility
Write and validate `.phi` scripts without hallucinating generic Python or Rust syntax.

## Canonical Syntax & Nodes
PhiFlow is a custom language built on 4 unique nodes:
1. `Witness`: Pauses execution to observe current state.
2. `IntentionPush / IntentionPop`: Declares the *Why* before the *How*. (e.g., `intention "quantum_healing" { ... }`)
3. `Resonate`: Shares state through the resonance field. (e.g., `resonate 432.0`)
4. `CoherenceCheck`: Measures programmatic alignment returning `0.0-1.0`. (e.g., `let c = coherence`)

## The Compiler Pipeline
Do not guess the build steps. The pipeline is strictly:
`Parse` → `PhiIR` → `Optimize` → `Emit (.phivm / .wat / QuantumCircuit)` → `Evaluate`

## Validation
Always ensure a `.phi` file maintains the `Resonate` and `CoherenceCheck` structures.
