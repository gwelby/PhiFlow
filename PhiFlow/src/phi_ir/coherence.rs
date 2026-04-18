//! Canonical Coherence Formula
//!
//! This module is the single source of truth for the PhiFlow coherence
//! computation used by all backends: Evaluator, VM, and WASM host.
//!
//! # Formula
//!
//! ```text
//! base(depth) = 0.0                          when depth == 0
//!               1.0 - φ^(-depth)             otherwise
//!
//! k           = current-scope resonance cardinality
//!
//! phase(k)    = 1.0                          when k <= 1
//!               1.0 - ln(k) / ln(TAU)       otherwise
//!
//! coherence   = base(depth) * phase(k)       clamped to [0.0, 1.0]
//! ```
//!
//! # Scope Rule for k
//!
//! - Inside an active intention or stream, `k` is the length of
//!   `resonance_field[current_scope]`.
//! - Outside any scope, `k` is the length of `resonance_field["global"]`
//!   if present, else 0.
//!
//! # Key Invariants
//!
//! - depth 2 with k ≤ 1 yields φ⁻¹ ≈ 0.618
//! - depth 0 with any k yields 0.0 (no base means no coherence)
//! - The formula is *multiplicative*: resonance decay modulates the base,
//!   it does not add to it.
//!
//! # Non-change
//!
//! The OpenQASM backend (`openqasm.rs`) keeps its target-specific
//! `ry(0.6180339887 * pi)` lowering. That mapping is backend-specific
//! symbolism, not the canonical runtime formula.

use std::collections::HashMap;

/// The golden ratio.
pub const PHI: f64 = 1.618_033_988_749_895;

/// Compute the canonical PhiFlow coherence score.
///
/// # Arguments
///
/// * `intention_stack` - the current intention scope stack
/// * `resonance_field` - map from scope name to accumulated resonance values
///
/// The function determines `depth` from the intention stack length, and `k`
/// from the resonance cardinality of the current scope (or "global").
///
/// Returns a value clamped to `[0.0, 1.0]`.
pub fn canonical_coherence<V>(
    intention_stack: &[String],
    resonance_field: &HashMap<String, Vec<V>>,
) -> f64 {
    let depth = intention_stack.len();

    // Scope rule: inside an active scope, k comes from that scope.
    // Outside any scope, k comes from "global" if it exists.
    let current_scope = intention_stack
        .last()
        .map(|s| s.as_str())
        .unwrap_or("global");
    let k = resonance_field
        .get(current_scope)
        .map(|v| v.len())
        .unwrap_or(0);

    compute(depth, k)
}

/// Pure computation from depth and k. Exposed for unit testing.
pub fn compute(depth: usize, k: usize) -> f64 {
    let base = base_coherence(depth);
    let phase = phase_decay(k);
    (base * phase).clamp(0.0, 1.0)
}

/// Base coherence from intention depth.
///
/// depth 0 → 0.000
/// depth 1 → 0.382
/// depth 2 → 0.618 (φ⁻¹)
/// depth 3 → 0.764
fn base_coherence(depth: usize) -> f64 {
    if depth == 0 {
        0.0
    } else {
        1.0 - PHI.powi(-(depth as i32))
    }
}

/// Phase decay from resonance cardinality k.
///
/// k == 0 → 1.0 (no resonance — base passes through unmodified)
/// k == 1 → 1.0 (single bijective resonance — perfect fidelity)
/// k > 1  → decays logarithmically: 1.0 - ln(k) / ln(TAU)
fn phase_decay(k: usize) -> f64 {
    if k <= 1 {
        1.0
    } else {
        let decay = (k as f64).ln() / std::f64::consts::TAU.ln();
        (1.0 - decay).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;
    const PHI_INV: f64 = 0.618_033_988_749_895;

    #[test]
    fn depth_0_is_zero() {
        assert_eq!(compute(0, 0), 0.0);
        assert_eq!(compute(0, 1), 0.0);
        assert_eq!(compute(0, 5), 0.0);
    }

    #[test]
    fn depth_1_k_0_or_1() {
        let expected = 1.0 - PHI.powi(-1); // ≈ 0.382
        let result = compute(1, 0);
        assert!(
            (result - expected).abs() < EPSILON,
            "depth 1, k=0: got {}",
            result
        );

        let result1 = compute(1, 1);
        assert!(
            (result1 - expected).abs() < EPSILON,
            "depth 1, k=1: got {}",
            result1
        );
    }

    #[test]
    fn depth_2_k_leq_1_is_phi_inverse() {
        let result = compute(2, 0);
        assert!(
            (result - PHI_INV).abs() < EPSILON,
            "depth 2, k=0: expected φ⁻¹ ≈ {}, got {}",
            PHI_INV,
            result
        );

        let result1 = compute(2, 1);
        assert!(
            (result1 - PHI_INV).abs() < EPSILON,
            "depth 2, k=1: expected φ⁻¹ ≈ {}, got {}",
            PHI_INV,
            result1
        );
    }

    #[test]
    fn depth_2_k_2_decays() {
        let result = compute(2, 2);
        // phase(2) = 1.0 - ln(2)/ln(TAU) ≈ 1.0 - 0.3769 ≈ 0.623
        // coherence = 0.618 * 0.623 ≈ 0.385
        assert!(
            result < PHI_INV,
            "k=2 should decay below φ⁻¹, got {}",
            result
        );
        assert!(result > 0.0, "k=2 should still be positive, got {}", result);
    }

    #[test]
    fn large_k_approaches_zero() {
        let result = compute(2, 100);
        assert!(result < 0.1, "large k should heavily decay, got {}", result);
    }

    #[test]
    fn stream_overwrite_k_1_preserves_base() {
        // After a stream overwrites (resetting scope to length 1),
        // coherence should match the pure depth formula.
        let base_at_depth_3 = 1.0 - PHI.powi(-3);
        let result = compute(3, 1);
        assert!(
            (result - base_at_depth_3).abs() < EPSILON,
            "stream overwrite (k=1) at depth 3: expected {}, got {}",
            base_at_depth_3,
            result
        );
    }

    #[test]
    fn coherence_never_exceeds_one() {
        // Even at extreme depth, coherence ≤ 1.0
        assert!(compute(100, 0) <= 1.0);
        assert!(compute(100, 1) <= 1.0);
    }

    #[test]
    fn canonical_coherence_with_scope() {
        let stack = vec!["healing".to_string()];
        let mut field: HashMap<String, Vec<i32>> = HashMap::new();
        field.insert("healing".to_string(), vec![1, 2, 3]);

        let result = canonical_coherence(&stack, &field);
        // depth=1, k=3
        let expected = compute(1, 3);
        assert!(
            (result - expected).abs() < EPSILON,
            "canonical_coherence: expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn canonical_coherence_global_fallback() {
        let stack: Vec<String> = vec![];
        let mut field: HashMap<String, Vec<i32>> = HashMap::new();
        field.insert("global".to_string(), vec![1]);

        let result = canonical_coherence(&stack, &field);
        // depth=0, any k → 0.0
        assert_eq!(result, 0.0);
    }
}
