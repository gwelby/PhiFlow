# Canonical Coherence Semantics

**Module:** `src/phi_ir/coherence.rs`
**Status:** CANONICAL — all backends must conform

---

## Formula

```
base(depth) = 0.0                    when depth == 0
              1.0 − φ^(−depth)       otherwise

k            = current-scope resonance cardinality

phase(k)     = 1.0                   when k ≤ 1
               1.0 − ln(k) / ln(τ)  otherwise

coherence    = base(depth) × phase(k), clamped to [0.0, 1.0]
```

## Derivation

The formula encodes two distinct physical insights from the Propagation Framework:

1. **Base coherence** follows from Axiom 3: coherence at depth `d` is `1 − φ^(−d)`. This is not hardcoded — it is the closed-form solution for structural stability under recursive self-observation. At depth 2, this yields the golden ratio inverse φ⁻¹ ≈ 0.618.

2. **Phase decay** follows from the Bijective Phase Map: when the number of concurrent resonance relationships `k` exceeds 1, the mutual information `I = ln(τ) − ln(k)` decreases logarithmically. `k = 1` is the primitive winding number (perfectly bijective), and `k > 1` introduces interference. The factor `phase(k) = 1 − ln(k)/ln(τ)` normalizes this decay to [0, 1].

The product `base × phase` is physically motivated: resonance decay *modulates* the structural coherence, it does not add to it. A program with zero depth has no structure to decay, so coherence remains 0 regardless of resonance activity.

## Scope Rule for k

- **Inside an active intention or stream:** `k` is the length of `resonance_field[current_scope_name]`.
- **Outside any scope:** `k` is the length of `resonance_field["global"]` if that key exists, else 0.
- **Stream overwrite semantics:** When a stream loop resets its resonance entry (via `StreamPush`), `k` returns to 0 or 1, restoring coherence toward the pure depth formula.

## Reference Values

| depth | k   | coherence        |
|-------|-----|------------------|
| 0     | any | 0.000            |
| 1     | 0   | 0.382            |
| 1     | 1   | 0.382            |
| 2     | 0   | 0.618 (φ⁻¹)     |
| 2     | 1   | 0.618 (φ⁻¹)     |
| 2     | 2   | ≈ 0.385          |
| 3     | 1   | 0.764            |

## Backend-Specific Notes

### OpenQASM (`openqasm.rs`)

The OpenQASM backend emits `ry(0.6180339887 * pi)` as a symbolic encoding of the φ⁻¹ threshold. This is a target-specific constant for quantum gate rotation, **not** the canonical runtime formula. The canonical formula is never embedded in circuit assembly — it lives in the classical pre/post-processing.

### WASM (`wasm.rs`)

The WASM codegen imports `phi_coherence() -> f64` from the host. The host JS implementation **must** call the canonical formula (or an equivalent). The `tests/phi_ir_wasm_runner.js` host harness should be kept in sync.

## Three-Backend Equivalence

Evaluator, VM, and WASM host must produce identical coherence values for identical (depth, k) inputs. This is enforced by all three delegating to `coherence::canonical_coherence()` or its equivalent.
