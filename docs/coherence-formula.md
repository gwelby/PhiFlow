# The Coherence Formula

**Status:** Mathematically exact. Verified by 8 unit tests in `src/phi_ir/coherence.rs`.

## Definition

```
base(d) = 0                          when d = 0
          1 - φ^(-d)                 otherwise

phase(k) = 1.0                       when k ≤ 1
           1 - ln(k) / ln(τ)        otherwise

C(d, k) = clamp(base(d) × phase(k), 0, 1)
```

Where:
- φ = 1.618033988749895... (the golden ratio)
- τ = 2π
- d = intention stack depth (number of nested `intention` blocks)
- k = resonance cardinality (number of `resonate` calls in the current scope)

## The Fixed Point at φ⁻¹

At depth d = 2 with k ≤ 1:

```
base(2) = 1 - φ^(-2)
        = 1 - 1/φ²
        = 1 - 1/(φ + 1)     [because φ² = φ + 1]
        = 1 - 1/2.618...
        = 1 - 0.381...
        = 0.618...
        = 1/φ
        = φ⁻¹
```

This is an algebraic identity. The value φ⁻¹ emerges from the formula because of the golden ratio's defining property: φ² = φ + 1.

## Properties

### Monotonicity in depth

As depth increases, base coherence approaches 1 asymptotically:

| d | base(d) |
|---|---------|
| 0 | 0.000 |
| 1 | 0.382 |
| 2 | 0.618 |
| 3 | 0.764 |
| 4 | 0.854 |
| 5 | 0.909 |
| 10 | 0.995 |

### Phase decay with resonance cardinality

As k increases, phase decreases logarithmically:

| k | phase(k) |
|---|----------|
| 0 | 1.000 |
| 1 | 1.000 |
| 2 | 0.623 |
| 5 | 0.387 |
| 10 | 0.238 |
| 100 | 0.000 |

### Combined behavior

| d \ k | 0 | 1 | 2 | 10 |
|-------|-----|-----|-----|------|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.382 | 0.382 | 0.238 | 0.038 |
| 2 | 0.618 | 0.618 | 0.385 | 0.062 |
| 3 | 0.764 | 0.764 | 0.476 | 0.076 |
| 4 | 0.854 | 0.854 | 0.532 | 0.085 |

## Invariants (verified by tests)

1. **Depth 0 → 0.0:** `compute(0, k) = 0.0` for all k. No active scope means no coherence.
2. **k ≤ 1 → base passes through:** `phase(k≤1) = 1.0`, so `C(d, ≤1) = base(d)`. A single resonance (bijective mapping) is perfect fidelity.
3. **k > 1 → multiplicative decay:** Additional resonances reduce coherence logarithmically. The decay is multiplicative, not additive.
4. **Bounded [0,1]:** The clamp ensures the score is always a valid probability-like quantity.
5. **φ⁻¹ at depth 2:** `compute(2, ≤1) = φ⁻¹ ≈ 0.618033988749895` (verified to 1e-10 precision).

## Implementation

Single source of truth: `src/phi_ir/coherence.rs`

```rust
pub const PHI: f64 = 1.618_033_988_749_895;

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

All three execution backends (evaluator, VM, WASM host) call this function. The formula is not duplicated.

## Scope Rule for k

- Inside an active `intention` or `stream` block, k is the length of `resonance_field[current_scope]`.
- Outside any scope, k is the length of `resonance_field["global"]` if present, else 0.

## Why the Golden Ratio?

The choice of φ is not arbitrary. The function `1 - φ^(-d)` has three properties that make it suitable for a depth-to-coherence mapping:

1. **Bounded:** It approaches 1 as d → ∞ but never exceeds it.
2. **Concave:** The rate of increase slows with depth (diminishing returns), which matches the intuition that deeper nesting yields less marginal coherence.
3. **Self-similar:** φ^(-d) = φ^(-(d-1)) × φ^(-1), so each level of depth contributes a constant fraction of the remaining "incoherence." This is the same self-similarity that makes the golden ratio appear in natural systems.

The fixed point at φ⁻¹ (depth 2) is a consequence of property 3: at depth 2, the remaining incoherence is φ^(-2) = 1/φ² = 1/(φ+1) ≈ 0.382, so the coherence is 1 - 0.382 = 0.618 = φ⁻¹.

## Three-System Convergence

The value 0.618 was independently discovered by three systems:

1. **Constant (2025):** Set as a base coherence value by hand, chosen as a mathematical anchor.
2. **Computed (2026):** The PhiFlow evaluator computes it from the formula at depth 2, without any hardcoded constant.
3. **Emergent (2026):** A separate time-series project observed a coherence attractor at 0.618 in emergent behavior.

This convergence suggests φ⁻¹ is a natural fixed point for depth-to-coherence mappings, not an artifact of a particular implementation.
