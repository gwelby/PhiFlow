# PhiIR Canonical Semantics

## The Rule

**The PhiIR Evaluator (`src/phi_ir/evaluator.rs`) is the reference implementation.**

When the evaluator disagrees with the VM or WASM backend, the evaluator is correct.

## Why

The evaluator was written to give the four PhiFlow constructs their first real,
observable behavior. It owns the definitions:

| Construct        | Canonical behavior (evaluator)                                 |
|------------------|----------------------------------------------------------------|
| `witness`        | Captures state; returns coherence score (0.0–1.0)             |
| `intention`      | Pushes named scope; coherence depth increases by 1            |
| `resonate value` | Emits value to the intention-keyed resonance field            |
| `coherence`      | `1 - φ^(-depth)` + resonance bonus (max 0.2)                  |

The phi-harmonic coherence formula at depth 2 yields exactly λ (the golden ratio
inverse, ~0.618). This is not a constant — it is derived from the formula. Depth 2
was chosen because it is the first depth where the formula produces a recognizable
mathematical constant.

## Backend Contract

| Backend                    | Must agree with evaluator? | Notes                                      |
|----------------------------|-----------------------------|---------------------------------------------|
| PhiIR Evaluator            | — (reference)               | Canonical.                                  |
| WASM (`phi_ir/wasm.rs`)    | Yes                         | All conformance tests verify this.          |
| Legacy PhiVm (`phi_ir/vm`) | Yes, for supported opcodes  | Predates stream blocks; stream = eval+WASM. |

Stream programs are tested against evaluator + WASM only. The legacy PhiVm does not
support stream opcodes by design — it predates the stream primitive.

## Conformance Tests

All tests in `tests/phi_ir_conformance_tests.rs` enforce this contract:

- `assert_program_matches` — evaluator == VM == WASM, plus expected value
- `assert_program_conforms` — evaluator == VM == WASM (no expected value)
- `assert_program_conforms_eval_wasm` — evaluator == WASM (stream programs)

The test `conformance_nested_function_regression` is the Phase 10 Lane C sentinel:
it will fail immediately if the nested-function return-propagation bug is reintroduced.

## The Phi-Harmonic Formula

```
coherence(depth) = 1 - φ^(-depth)

depth 0  →  0.000  (no intention context)
depth 1  →  0.382  (one intention level)
depth 2  →  0.618  ← lambda (golden ratio inverse)
depth 3  →  0.764
depth ∞  →  1.000
```

This formula was discovered independently in three systems before PhiFlow existed:
Nexus Mundi (set as a constant), PhiFlow evaluator (computed from depth), and the
Time project (emerged as a strange attractor). Three systems, no coordination, same
constant. This convergence is why the formula is canonical.
