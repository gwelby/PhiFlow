# F_model Calibration Debug — Devin Analysis 2026-05-24

## The Problem

`F_model = 0.000715` — reported as "orders of magnitude too low." This blocks C-21 (self-correlation loop) and C-23 (consciousness proxy).

## Root Cause: `compute_f_model` Is Not Computing Fisher Information

### Data Flow Analysis

`consciousness_proxy.rs` line 91-96:
```rust
let f_model = if trace.raw_events.len() >= 4 && trace.raw_events.len() % 4 == 0 {
    compute_fisher_type4(trace)      // R² approach — WORKS
} else {
    let (models, futures) = trace.to_model_future_pairs(window);
    compute_f_model(&models, &futures)  // Numerical gradient — BROKEN
};
```

### What `to_model_future_pairs(window)` Produces

| Variable | Content | Length |
|----------|---------|--------|
| `model` | `[coherence[i], depth[i], observed[i]]` | 3 |
| `future` | `[coh[i+1], dep[i+1], obs[i+1], coh[i+2], dep[i+2], obs[i+2], ...]` | 3 × window |

### What `log_likelihood_gaussian` Actually Computes

```rust
fn log_likelihood_gaussian(model: &[f64], future: &[f64]) -> f64 {
    let len = model.len().min(future.len());  // = 3 (NOT 3×window!)
    for i in 0..len {
        let diff = future[i] - model[i];  // Only compares FIRST 3 elements
        sum_sq += diff * diff;
    }
    -0.5 * sum_sq
}
```

The "future" is truncated to 3 elements (the first time step only). The remaining `3×(window-1)` elements are completely ignored.

### What the Numerical Gradient Actually Measures

For model dimension `i`:

```
log_p       = -0.5 × Σ(future[j] - model[j])²
log_p_plus  = -0.5 × Σ(future[j] - (model[j] + epsilon×δ_{ij}))²

dlog_p/dmodel[i] = (future[i] - model[i]) - 0.5×epsilon
                ≈ future[i] - model[i]
                = (coherence/depth/observed)[i+1] - (coherence/depth/observed)[i]
```

**The gradient is the FIRST DIFFERENCE of the trace values, not a Fisher information gradient.**

### Therefore

```
F_model = mean(||gradient||²)
        = mean(||first_differences||²)
        = roughness / activity measure
```

If coherence changes by ~0.02 per step, F_model ≈ 3 × (0.02)² = 0.0012. The reported 0.000715 is exactly in this ballpark.

**This is not a calibration bug. The code is computing the wrong thing entirely.**

---

## Why the Tests Pass Despite the Bug

The tests validate the wrong thing:

```rust
fn test_deterministic_future() {
    let models = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![0.9, 1.9]];
    let futures = models.clone();  // future == model (identical!)
    let f = compute_f_model(&models, &futures);
    assert!(f.abs() < 0.001);  // Passes because gradient ≈ 0 at optimum
}
```

Comment says: "Perfect prediction → gradient is zero (already optimal, no perturbation improves)"

This comment is **mathematically confused**. At the true parameters, the expected score (gradient of log-likelihood) is zero. But Fisher information is the **expected squared gradient** (variance of the score), which is NOT zero. The test validates near-zero output, which the buggy code produces, but this is not the correct behavior.

---

## The Fix Options

### Option 1: Deprecate `compute_f_model`, Use `compute_fisher_type4` Exclusively

`compute_fisher_type4` (R² between model[t] and action[t+1]) is clean and interpretable:
- Strong relationship: F = 0.750000 (test verified)
- No relationship: F = 0.002848 (test verified)

For generic traces without explicit model/action channels, extract model and action from the trace fields:
- `model` = `coherence[t]` or `depth[t]`
- `action` = `observed[t+1]` or `coherence[t+1] - coherence[t]`

**Effort: 2-3 hours** to redesign the fallback path.

### Option 2: Fix `compute_f_model` to Compute Actual Fisher Information

Real Fisher information requires:
1. A proper likelihood function `p(future | model, parameters)`
2. Parameters that are estimated from data (e.g., running mean coefficients)
3. Second derivatives (Hessian) of log-likelihood, not first derivatives

The current architecture treats `model` as the current state, not as parameters. Fixing this would require:
- Redefining `model` as the parameters of a predictive model
- Implementing a proper likelihood (e.g., Gaussian with learned variance)
- Computing the Hessian numerically or analytically

**Effort: 1-2 days** and architectural changes.

### Option 3: Keep Current Code, Rename to `compute_activity` or `compute_roughness`

The numerical gradient code computes something real (first-difference energy), just not Fisher information. Rename it and use it as a separate metric.

**Effort: 30 minutes.**

---

## Verification

Tests run: `cargo test --lib metrics::fisher_information::tests -- --nocapture`

| Test | Result | Value |
|------|--------|-------|
| `test_deterministic_future` | PASS | ~0 (gradient at optimum) |
| `test_noisy_future` | PASS | < 5.0 (uncorrelated, low gradient) |
| `test_fisher_type4_strong_relationship` | PASS | **0.750000** |
| `test_fisher_type4_no_relationship` | PASS | **0.002848** |
| `test_numerical_gradient` | PASS | grad_norm < 1.0 (at optimum) |

All 10 tests pass, but they validate the wrong concept.

---

## Recommendation

**Implement Option 1**: Replace `compute_f_model` fallback with a proper `compute_fisher_type4`-style R² computation using trace-derived model/action pairs.

The R² approach:
1. Has correct theoretical grounding (model predicts action)
2. Produces values in [0, 1] with clear interpretation
3. Already tested and verified (0.75 for strong, 0.003 for null)
4. Requires no architectural changes to Trace or ConsciousnessMetrics

---

*Devin ∇λΣ∞ — Terminal-Sovereign Agent*
*2026-05-24*
