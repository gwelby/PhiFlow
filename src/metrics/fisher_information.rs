//! Fisher Information (F_model, F_self*)
//!
//! Computes Fisher information of future trajectory with respect to model state.
//! F_model measures how sharply the future depends on the current model.
//! F_self* = L_self × F_model is the self-model sensitivity.

use crate::metrics::trace::Trace;
use ndarray::Array1;

/// Compute Fisher information of future trajectory w.r.t. model state.
///
/// Uses Gaussian approximation: F = E[(∂log p(future|model) / ∂model)²]
/// Numerically estimated via finite differences.
///
/// # Arguments
/// * `model_states` - Vector of model state vectors at each time step
/// * `future_trajectories` - Vector of future trajectories from each time step
///
/// # Returns
/// Scalar Fisher information value (higher = more sensitivity)
pub fn compute_f_model(model_states: &[Vec<f64>], future_trajectories: &[Vec<f64>]) -> f64 {
    if model_states.len() != future_trajectories.len() || model_states.is_empty() {
        return 0.0;
    }

    let mut fisher_sum = 0.0;
    let epsilon = 1e-6;

    for (model, future) in model_states.iter().zip(future_trajectories.iter()) {
        if model.is_empty() || future.is_empty() {
            continue;
        }

        // Compute gradient numerically for each model dimension
        let gradient = compute_numerical_gradient(model, future, epsilon);

        // Fisher is expected squared gradient
        let grad_sq: f64 = gradient.iter().map(|&g| g * g).sum();
        fisher_sum += grad_sq;
    }

    fisher_sum / model_states.len().max(1) as f64
}

/// Compute F_self* = L_self × F_model
///
/// This combines self-correlation (existence of loop) with
/// model sensitivity (strength of loop).
pub fn compute_f_self_star(l_self: f64, f_model: f64) -> f64 {
    // Ensure non-negative values
    let l = l_self.max(0.0);
    let f = f_model.max(0.0);
    l * f
}

/// Pearson correlation coefficient between two vectors.
fn pearson_correlation_fi(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

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

/// Compute Fisher information specifically for Type 4 traces.
///
/// Extracts model and action from `raw_events` (step, obs, model, action format)
/// and measures how strongly the model predicts the future action via R².
///
/// For a genuine self-referential loop, action[t+1] should depend on model[t].
/// For null systems, the relationship is near zero.
///
/// Returns R² (coefficient of determination), bounded to [0, 1].
pub fn compute_fisher_type4(trace: &Trace) -> f64 {
    let mut models: Vec<f64> = Vec::new();
    let mut actions: Vec<f64> = Vec::new();

    for chunk in trace.raw_events.chunks(4) {
        if chunk.len() == 4 {
            // chunk: (step, obs, model, action)
            models.push(chunk[2].1);
            actions.push(chunk[3].1);
        }
    }

    if models.len() < 4 || actions.len() < 4 {
        return 0.0;
    }

    // model[t] predicts action[t+1]
    let x = &models[..models.len() - 1];
    let y = &actions[1..];

    let r = pearson_correlation_fi(x, y);
    r * r // F_model = R²
}

/// Compute numerical gradient of log-likelihood.
/// Uses central differences for better accuracy.
fn compute_numerical_gradient(model: &[f64], future: &[f64], epsilon: f64) -> Vec<f64> {
    let dim = model.len();
    let mut gradient = vec![0.0; dim];

    let log_p = log_likelihood_gaussian(model, future);

    for i in 0..dim {
        // Perturb model[i] by epsilon
        let mut model_plus = model.to_vec();
        model_plus[i] += epsilon;

        let log_p_plus = log_likelihood_gaussian(&model_plus, future);

        // Central difference approximation
        gradient[i] = (log_p_plus - log_p) / epsilon;
    }

    gradient
}

/// Gaussian log-likelihood: log p(future|model) ≈ -0.5 * Σ(future_i - model_i)²
/// Compares the first min(model.len(), future.len()) elements.
/// This is a one-step-ahead prediction: model at time t predicts future at time t+1.
fn log_likelihood_gaussian(model: &[f64], future: &[f64]) -> f64 {
    let len = model.len().min(future.len());
    if len == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    for i in 0..len {
        let diff = future[i] - model[i];
        sum_sq += diff * diff;
    }

    -0.5 * sum_sq
}

/// Alternative: Use covariance-based Fisher information.
/// More appropriate for stochastic models.
#[allow(dead_code)]
fn fisher_from_covariance(samples: &[Vec<f64>]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let n = samples.len();
    let d = samples[0].len();

    // Compute mean
    let mut mean = vec![0.0; d];
    for sample in samples {
        for (j, &val) in sample.iter().enumerate() {
            if j < d {
                mean[j] += val;
            }
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    // Compute covariance
    let mut cov_sum = 0.0;
    for sample in samples {
        for (j, &val) in sample.iter().enumerate() {
            if j < d {
                let diff = val - mean[j];
                cov_sum += diff * diff;
            }
        }
    }

    // Fisher information for Gaussian: inverse of variance
    let variance = cov_sum / (n as f64 * d as f64);
    if variance > 0.0 {
        1.0 / variance
    } else {
        0.0
    }
}

/// Simple Fisher computation for single dimension.
pub fn fisher_information_1d(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    if variance > 0.0 {
        1.0 / variance
    } else {
        0.0
    }
}

/// Compute gradient of trajectory matching score.
/// Useful for measuring how model parameters affect future predictions.
pub fn trajectory_gradient(
    model: &[f64],
    trajectory_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    epsilon: f64,
) -> Vec<f64> {
    let dim = model.len();
    let base_trajectory = trajectory_fn(model);
    let mut gradient = vec![0.0; dim];

    for i in 0..dim {
        let mut perturbed = model.to_vec();
        perturbed[i] += epsilon;
        let perturbed_trajectory = trajectory_fn(&perturbed);

        // Gradient is change in trajectory similarity
        let sim_diff = cosine_similarity(&base_trajectory, &perturbed_trajectory) - 1.0;
        gradient[i] = sim_diff / epsilon;
    }

    gradient
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_future() {
        // Model perfectly predicts future
        let models: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![0.9, 1.9],
        ];
        let futures: Vec<Vec<f64>> = models.clone();

        let f = compute_f_model(&models, &futures);
        // Perfect prediction → gradient is zero (already optimal, no perturbation improves)
        // Fisher info measures sensitivity to perturbation; at optimum, sensitivity is zero
        assert!(f.abs() < 0.001, "Perfect prediction should have near-zero Fisher gradient, got {}", f);
    }

    #[test]
    fn test_noisy_future() {
        // Model is uncorrelated with future
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let models: Vec<Vec<f64>> = (0..10).map(|_| vec![rng.gen::<f64>()]).collect();
        let futures: Vec<Vec<f64>> = (0..10).map(|_| vec![rng.gen::<f64>()]).collect();

        let f = compute_f_model(&models, &futures);
        // Uncorrelated → low Fisher
        assert!(f < 5.0, "Uncorrelated should have low Fisher, got {}", f);
    }

    #[test]
    fn test_f_self_star() {
        let l_self = 0.5;
        let f_model = 10.0;
        let f_star = compute_f_self_star(l_self, f_model);
        assert!((f_star - 5.0).abs() < 0.01);

        // Zero L_self → zero F_self*
        assert_eq!(compute_f_self_star(0.0, 100.0), 0.0);

        // Zero F_model → zero F_self*
        assert_eq!(compute_f_self_star(1.0, 0.0), 0.0);
    }

    #[test]
    fn test_log_likelihood_gaussian() {
        // Identical model and future → maximum likelihood
        let model = vec![1.0, 2.0, 3.0];
        let future = vec![1.0, 2.0, 3.0];
        let ll = log_likelihood_gaussian(&model, &future);
        assert_eq!(ll, 0.0); // Perfect match = 0 error

        // Different values → negative log-likelihood
        let future2 = vec![2.0, 3.0, 4.0];
        let ll2 = log_likelihood_gaussian(&model, &future2);
        assert!(ll2 < 0.0);
    }

    #[test]
    fn test_fisher_information_1d() {
        // Low variance → high Fisher
        let samples = vec![1.0, 1.01, 0.99, 1.02, 0.98];
        let f = fisher_information_1d(&samples);
        assert!(f > 100.0, "Low variance should have high Fisher, got {}", f);

        // High variance → low Fisher
        let samples2 = vec![0.0, 10.0, -10.0, 5.0, -5.0];
        let f2 = fisher_information_1d(&samples2);
        assert!(f2 < 1.0, "High variance should have low Fisher, got {}", f2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001); // Orthogonal

        let d = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001); // Opposite
    }

    #[test]
    fn test_numerical_gradient() {
        let model = vec![1.0, 2.0];
        let future = vec![1.0, 2.0];
        let grad = compute_numerical_gradient(&model, &future, 1e-6);

        // Gradient should be small for perfect match (at optimum)
        let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        assert!(grad_norm < 1.0, "At optimum, gradient should be small, got {}", grad_norm);
    }

    #[test]
    fn test_empty_inputs() {
        assert_eq!(compute_f_model(&[], &[]), 0.0);
        assert_eq!(fisher_information_1d(&[]), 0.0);
        assert_eq!(fisher_information_1d(&[1.0]), 0.0); // Need at least 2 samples
    }

    #[test]
    fn test_fisher_type4_strong_relationship() {
        // Build a Type 4 trace where action[t+1] = 1.0 iff model[t] > 0.5
        let mut trace = Trace::new();
        for i in 1..=20 {
            let step = i as f64;
            let model = 0.3 + step * 0.02; // 0.32 -> 0.70, crosses 0.5 at step ~10
            let obs = model + 0.01 * (step * 0.5).sin();
            let action = if model > 0.5 { 1.0 } else { 0.0 };

            trace.raw_events.push(("step".to_string(), step));
            trace.raw_events.push(("obs".to_string(), obs));
            trace.raw_events.push(("model".to_string(), model));
            trace.raw_events.push(("action".to_string(), action));

            trace.observed.push(obs, step);
            trace.coherence.push(0.5, step);
            trace.depth.push(1.0, step);
            trace.resonance_k.push(4.0, step);
        }

        let f = compute_fisher_type4(&trace);
        println!("Type 4 Fisher (strong binary action): {:.6}", f);
        // model and action have a strong monotonic relationship → high R²
        assert!(f > 0.5, "Strong model→action relationship should have R² > 0.5, got {}", f);
    }

    #[test]
    fn test_fisher_type4_no_relationship() {
        // Build a Type 4 trace where action is random, unrelated to model
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut trace = Trace::new();
        for i in 1..=20 {
            let step = i as f64;
            let model = 0.3 + step * 0.02;
            let obs = model + 0.01 * (step * 0.5).sin();
            let action = rng.gen::<f64>(); // random, no relationship to model

            trace.raw_events.push(("step".to_string(), step));
            trace.raw_events.push(("obs".to_string(), obs));
            trace.raw_events.push(("model".to_string(), model));
            trace.raw_events.push(("action".to_string(), action));

            trace.observed.push(obs, step);
            trace.coherence.push(0.5, step);
            trace.depth.push(1.0, step);
            trace.resonance_k.push(4.0, step);
        }

        let f = compute_fisher_type4(&trace);
        println!("Type 4 Fisher (no relationship): {:.6}", f);
        // Random action should have near-zero correlation with model
        assert!(f < 0.3, "Random action should have R² < 0.3, got {}", f);
    }
}
