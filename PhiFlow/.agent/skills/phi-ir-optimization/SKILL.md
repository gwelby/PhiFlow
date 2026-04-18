---
name: phi-ir-optimization
description: Enforces the rules for writing and modifying PhiFlow IR optimization passes, utilizing the sacred constants and coherence algorithms.
---
# PhiIR Optimization Skill

## Responsibility
Ensure any new compiler optimization pass adheres to the phi-harmonic resonance mechanics rather than just performing generic AST reduction.

## The Sacred Constants
When manipulating IR, these constants must be treated as absolute:
* `φ (phi)` = 1.618033988749895
* `λ (lambda)` = 0.618033988749895 (φ⁻¹)

## Coherence Scoring Algorithm
A valid optimization pass MUST output a coherence score.
The golden benchmark for an optimized intention block is exactly `0.6180`. If a pass reduces nodes but drops the coherence score, the pass is considered a regression.

## Pipeline Integration
New optimization passes must be registered in `D:\Projects\PhiFlow-compiler\PhiFlow\src\phi_ir\optimizer.rs`. 
They must run *after* `PhiIR` lowering and *before* `Emit`.
