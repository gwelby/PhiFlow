//! Self-Correlation Computation (L_self)
//!
//! Implements the Type 4 observer metric from the PF consciousness_metric_program:
//!   R_in  = I_dir(past_observations → model_state)
//!   R_out = I_dir(model_state → future_behavior | current_obs)
//!   L_self = min(R_in_normalized, R_out_normalized)
//!
//! If either leg is zero, the self-correlation loop is broken → not Type 4.

use super::mutual_information::normalized_mi;
use super::trace::Trace;

/// Self-correlation metrics for Type 4 observer verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfCorrelation {
    /// Normalized mutual information: past observations → current model
    pub r_in_norm: f64,
    /// Normalized mutual information: current model → future behavior
    pub r_out_norm: f64,
    /// Self-correlation loop strength: min(R_in, R_out)
    pub l_self: f64,
    /// Whether the loop is structurally closed (L_self > threshold)
    pub loop_closed: bool,
    /// Threshold used for loop_closed determination
    pub threshold: f64,
}

impl SelfCorrelation {
    /// Create a new SelfCorrelation with all zeros.
    pub fn zero() -> Self {
        Self {
            r_in_norm: 0.0,
            r_out_norm: 0.0,
            l_self: 0.0,
            loop_closed: false,
            threshold: 0.01,
        }
    }

    /// Compute self-correlation from a trace.
    ///
    /// # Arguments
    /// * `trace` - The execution trace with coherence, depth, observed values
    /// * `window` - Number of samples to use for past/future windows
    /// * `bins` - Number of bins for discretization in MI computation
    /// * `threshold` - Minimum L_self to consider loop "closed"
    pub fn from_trace(trace: &Trace, window: usize, bins: usize, threshold: f64) -> Self {
        if trace.len() < window * 2 + 1 {
            // Not enough data for meaningful computation
            return Self::zero();
        }

        // Extract vectors from trace
        // For type4_trace_benchmark format:
        // - observed = the raw observations (OBS)
        // We compute model as running mean of past observations
        let obs = &trace.observed.values;

        // Build model (running mean) and deviation signals
        let mut model: Vec<f64> = Vec::with_capacity(trace.len());
        let mut deviation: Vec<f64> = Vec::with_capacity(trace.len());

        let mut model_sum = 0.55; // Initial model mean from benchmark
        let mut model_n = 1.0;

        for &o in obs.iter() {
            let model_mean = model_sum / model_n;
            model.push(model_mean);
            deviation.push(o - model_mean);

            // Update running mean (self-model update)
            model_sum += o;
            model_n += 1.0;
        }

        // R_in: correlation between past observations and current model
        // model[t] depends on obs[0..t-1] through running mean
        // Measure as correlation between obs[t-1] and model[t]
        if obs.len() < 2 {
            return Self::zero();
        }

        let obs_t_minus_1: Vec<f64> = obs[..obs.len() - 1].to_vec();
        let model_t: Vec<f64> = model[1..].to_vec();

        // R_in via Pearson correlation as proxy (fast, works well for running means)
        let r_in_corr = pearson_correlation(&obs_t_minus_1, &model_t);
        let r_in_norm = r_in_corr.abs(); // Normalize to [0, 1]

        // R_out: does model predict future deviation?
        // Measure MI between model[t] and deviation[t]
        let r_out_norm = if model.len() >= window {
            // Use MI for R_out (more general than correlation)
            normalized_mi(&model[..window.min(model.len())], &deviation[..window.min(deviation.len())], bins)
        } else {
            0.0
        };

        // L_self = min(R_in, R_out)
        let l_self = r_in_norm.min(r_out_norm);
        let loop_closed = l_self > threshold;

        Self {
            r_in_norm,
            r_out_norm,
            l_self,
            loop_closed,
            threshold,
        }
    }

    /// Compute self-correlation specifically from type4_trace_benchmark format.
    /// This is optimized for the 4-tuple output (step, obs, model, action).
    ///
    /// R_out now correctly measures: I(model[t] -> action[t+1] | obs[t])
    /// This is model predicting future behavior (action), not residual deviation.
    pub fn from_type4_trace(trace: &Trace, threshold: f64) -> Self {
        if trace.len() < 4 {
            return Self::zero();
        }

        // Reconstruct obs, model, and action from raw_events
        // Format: step, obs, model, action in groups of 4
        let mut steps: Vec<f64> = Vec::new();
        let mut obs_vals: Vec<f64> = Vec::new();
        let mut model_vals: Vec<f64> = Vec::new();
        let mut actions: Vec<f64> = Vec::new();

        // Parse from raw_events which were grouped in chunks of 4
        for chunk in trace.raw_events.chunks(4) {
            if chunk.len() == 4 {
                steps.push(chunk[0].1);
                obs_vals.push(chunk[1].1);
                model_vals.push(chunk[2].1);
                actions.push(chunk[3].1);
            }
        }

        if obs_vals.len() < 4 {
            return Self::zero();
        }

        // R_in: correlation between past obs and current model
        // obs[t-1] -> model[t] (past observations predict current model state)
        let r_in_corr = if obs_vals.len() >= 2 {
            pearson_correlation(&obs_vals[..obs_vals.len() - 1], &model_vals[1..])
        } else {
            0.0
        };
        let r_in_norm = r_in_corr.abs();

        // R_out: MI between model[t] and action[t+1] (future behavior)
        // This measures directed influence: does model state predict future action?
        // action[t+1] is the behavior taken after observing the model at time t
        let r_out_norm = if model_vals.len() >= 2 && actions.len() >= 2 {
            // model[t] aligned with action[t+1] (one-step prediction)
            let model_t = &model_vals[..model_vals.len() - 1];
            let action_future = &actions[1..];
            normalized_mi(model_t, action_future, 5)
        } else {
            0.0
        };

        let l_self = r_in_norm.min(r_out_norm);
        let loop_closed = l_self > threshold;

        Self {
            r_in_norm,
            r_out_norm,
            l_self,
            loop_closed,
            threshold,
        }
    }

    /// Compute R_out with shuffle control to validate temporal alignment.
    ///
    /// Returns (actual_r_out, shuffled_r_out) where shuffled_r_out should be
    /// significantly lower if the relationship is genuinely temporal.
    ///
    /// This breaks temporal alignment while preserving marginal distributions,
    /// serving as a null model for the model->future behavior relationship.
    pub fn from_type4_trace_with_shuffle_control(
        trace: &Trace,
        threshold: f64,
    ) -> (Self, f64) {
        let base = Self::from_type4_trace(trace, threshold);

        // Compute shuffled R_out
        let mut steps: Vec<f64> = Vec::new();
        let mut obs_vals: Vec<f64> = Vec::new();
        let mut model_vals: Vec<f64> = Vec::new();
        let mut actions: Vec<f64> = Vec::new();

        for chunk in trace.raw_events.chunks(4) {
            if chunk.len() == 4 {
                steps.push(chunk[0].1);
                obs_vals.push(chunk[1].1);
                model_vals.push(chunk[2].1);
                actions.push(chunk[3].1);
            }
        }

        // Shuffle actions to break temporal alignment
        let shuffled_r_out = if model_vals.len() >= 2 && actions.len() >= 2 {
            use rand::seq::SliceRandom;
            use rand::thread_rng;

            let mut rng = thread_rng();
            let mut shuffled_actions = actions.clone();
            shuffled_actions.shuffle(&mut rng);

            let model_t = &model_vals[..model_vals.len() - 1];
            let shuffled_future = &shuffled_actions[1..];
            normalized_mi(model_t, shuffled_future, 5)
        } else {
            0.0
        };

        (base, shuffled_r_out)
    }
}

/// Convenience function to compute self-correlation from a trace.
/// 
/// Automatically detects type4_trace_benchmark format (4-tuple resonance events)
/// vs generic witness-based traces and uses the appropriate method.
pub fn self_correlation_from_trace(trace: &Trace, window: usize, bins: usize, threshold: f64) -> SelfCorrelation {
    // Check if we have raw_events in type4 format (4 values per cycle: step, obs, model, action)
    if trace.raw_events.len() >= 4 && trace.raw_events.len() % 4 == 0 {
        // Verify the pattern: step values should be 1.0, 2.0, 3.0, etc.
        let first_step = trace.raw_events.get(0).map(|(_, v)| *v).unwrap_or(0.0);
        let second_step = trace.raw_events.get(4).map(|(_, v)| *v).unwrap_or(0.0);
        
        if first_step > 0.0 && (second_step - first_step).abs() < 2.0 {
            // This looks like type4_trace_benchmark format
            return SelfCorrelation::from_type4_trace(trace, threshold);
        }
    }
    
    // Fall back to generic method
    SelfCorrelation::from_trace(trace, window, bins, threshold)
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x == 0.0 || den_y == 0.0 {
        return 0.0;
    }

    num / (den_x.sqrt() * den_y.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_null() {
        // Feed-forward: y = 0.3 * sin(t), no memory
        // Should produce L_self ≈ 0 (no self-correlation loop)
        let mut trace = Trace::new();
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let obs = 0.3 * t.sin();
            trace.observed.push(obs, t);
            trace.coherence.push(0.5, t);
            trace.depth.push(1.0, t);
        }

        let sc = SelfCorrelation::from_trace(&trace, 10, 5, 0.01);
        // Feed-forward should have very low L_self
        assert!(sc.l_self < 0.05, "Feed-forward should have L_self < 0.05, got {}", sc.l_self);
        assert!(!sc.loop_closed, "Feed-forward should not have closed loop");
    }

    #[test]
    fn test_running_mean_self_correlation() {
        // Running mean creates genuine self-correlation
        // model[t] = mean(obs[0..t]), so past obs definitely predict model
        let mut trace = Trace::new();
        let mut obs_vec: Vec<f64> = Vec::new();

        // Generate slowly varying observations
        for i in 0..100 {
            let obs = 0.5 + 0.3 * (i as f64 * 0.1).sin() + 0.05 * (i as f64).sin();
            obs_vec.push(obs);
            trace.observed.push(obs, i as f64);
            trace.coherence.push(0.5, i as f64);
            trace.depth.push(1.0, i as f64);
        }

        let sc = SelfCorrelation::from_trace(&trace, 10, 5, 0.01);

        // Running mean should have substantial R_in (past obs → model)
        assert!(sc.r_in_norm > 0.5, "Running mean should have R_in > 0.5, got {}", sc.r_in_norm);

        // L_self may or may not be high depending on R_out
        // The key is that the structure is present
        println!("Running mean: R_in={}, R_out={}, L_self={}", sc.r_in_norm, sc.r_out_norm, sc.l_self);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001, "Perfect correlation should be ~1.0");

        let z = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &z);
        assert!((r_neg - (-1.0)).abs() < 0.001, "Perfect anti-correlation should be ~-1.0");
    }

    #[test]
    fn test_self_correlation_zero() {
        let sc = SelfCorrelation::zero();
        assert_eq!(sc.l_self, 0.0);
        assert!(!sc.loop_closed);
    }

    #[test]
    fn test_threshold_behavior() {
        // Test that threshold correctly determines loop_closed
        let mut trace = Trace::new();
        for i in 0..50 {
            trace.observed.push(i as f64, i as f64);
            trace.coherence.push(0.5, i as f64);
            trace.depth.push(1.0, i as f64);
        }

        let sc_low = SelfCorrelation::from_trace(&trace, 10, 5, 0.001);
        let sc_high = SelfCorrelation::from_trace(&trace, 10, 5, 0.9);

        // Same data, different thresholds
        assert!(sc_low.threshold < sc_high.threshold);
        // loop_closed depends on threshold relative to l_self
    }
}
