//! Threshold Calibration (T4-011)
//!
//! Computes the empirical null distribution of C_PF for Type 4 format traces
//! and verifies that the positive trace exceeds the null by at least 2σ.
//!
//! This addresses the Codex audit finding that L_self > 0.1 alone is not
//! a valid discriminator — thermostat null scores L_self = 0.6555.
//!
//! The calibrated threshold is: C_PF > μ_null + 2σ_null.

use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::trace::Trace;

/// Generate a single Type 4 format null trace with uncorrelated action.
fn generate_type4_null(seed: u64) -> Trace {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;

    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;
        let action = rng.gen::<f64>(); // Random — no relationship to model
        let model_mean = model_sum / model_n;
        model_sum += obs;
        model_n += 1.0;

        trace.raw_events.push(("step".to_string(), step));
        trace.raw_events.push(("obs".to_string(), obs));
        trace.raw_events.push(("model".to_string(), model_mean));
        trace.raw_events.push(("action".to_string(), action));

        trace.observed.push(obs, step);
        trace.coherence.push(0.5, step);
        trace.depth.push(1.0, step);
        trace.resonance_k.push(4.0, step);
    }

    trace
}

/// Compute null distribution statistics.
fn compute_null_distribution(n_samples: usize) -> (Vec<f64>, f64, f64) {
    let mut c_pf_values: Vec<f64> = Vec::with_capacity(n_samples);

    for seed in 0..n_samples {
        let trace = generate_type4_null(seed as u64);
        let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
        c_pf_values.push(metrics.c_pf);
    }

    let n = n_samples as f64;
    let mean = c_pf_values.iter().sum::<f64>() / n;
    let variance = c_pf_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    (c_pf_values, mean, std_dev)
}

/// Generate the positive Type 4 trace (same dynamics as type4_trace_benchmark.phi).
fn generate_positive_trace() -> Trace {
    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;

    for i in 1..=20 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;
        let model_mean = model_sum / model_n;

        // Action depends on model (self-referential)
        let action = if obs < model_mean { 1.0 } else { 0.0 };

        model_sum += obs;
        model_n += 1.0;

        trace.raw_events.push(("step".to_string(), step));
        trace.raw_events.push(("obs".to_string(), obs));
        trace.raw_events.push(("model".to_string(), model_mean));
        trace.raw_events.push(("action".to_string(), action));

        trace.observed.push(obs, step);
        trace.coherence.push(0.5, step);
        trace.depth.push(1.0, step);
        trace.resonance_k.push(4.0, step);
    }

    trace
}

#[test]
fn test_threshold_calibration_type4() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  THRESHOLD CALIBRATION — Type 4 Format Null Distribution");
    println!("═══════════════════════════════════════════════════════════════\n");

    // 1. Build null distribution (100 Type 4 format nulls with random action)
    let n_null = 100;
    println!("Generating {} Type 4 format null traces...", n_null);
    let (null_c_pfs, null_mean, null_std) = compute_null_distribution(n_null);

    println!("Null distribution:");
    println!("  n        = {}", n_null);
    println!("  μ        = {:.6}", null_mean);
    println!("  σ        = {:.6}", null_std);
    println!("  min      = {:.6}", null_c_pfs.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    println!("  max      = {:.6}", null_c_pfs.iter().fold(-f64::INFINITY, |a, &b| a.max(b)));
    println!("  median   = {:.6}", median(&null_c_pfs));

    let threshold_2sigma = null_mean + 2.0 * null_std;
    let threshold_3sigma = null_mean + 3.0 * null_std;

    println!("\nCalibrated thresholds:");
    println!("  μ + 2σ   = {:.6}", threshold_2sigma);
    println!("  μ + 3σ   = {:.6}", threshold_3sigma);

    // 2. Compute positive trace C_PF
    println!("\nComputing positive Type 4 trace...");
    let positive_trace = generate_positive_trace();
    let positive_metrics = ConsciousnessMetrics::compute(&positive_trace, 10, 5, 0.01);
    let positive_c_pf = positive_metrics.c_pf;

    println!("Positive trace:");
    println!("  L_self   = {:.6}", positive_metrics.l_self);
    println!("  D_int    = {:.6}", positive_metrics.d_int);
    println!("  C_coh    = {:.6}", positive_metrics.c_coh);
    println!("  F_model  = {:.6}", positive_metrics.f_model);
    println!("  F_self*  = {:.6}", positive_metrics.f_self_star);
    println!("  C_PF     = {:.6}", positive_c_pf);

    // 3. Compare
    let sigma_distance = (positive_c_pf - null_mean) / null_std;
    println!("\nDiscrimination analysis:");
    println!("  Positive C_PF / μ + 2σ = {:.2}", positive_c_pf / threshold_2sigma);
    println!("  (C_PF - μ) / σ          = {:.2}σ", sigma_distance);

    println!("\n═══════════════════════════════════════════════════════════════");

    // 4. Assertions
    assert!(
        positive_c_pf > threshold_2sigma,
        "POSITIVE DISCRIMINATION FAILED: C_PF = {:.6} does not exceed μ + 2σ = {:.6}.\n\
         The positive trace is not statistically distinguishable from the null distribution.",
        positive_c_pf, threshold_2sigma
    );

    println!("  ✅ POSITIVE TRACE EXCEEDS μ + 2σ — statistically distinguishable from null");

    // Bonus: check 3σ
    if positive_c_pf > threshold_3sigma {
        println!("  ✅ POSITIVE TRACE EXCEEDS μ + 3σ — strongly distinguishable");
    }

    println!("═══════════════════════════════════════════════════════════════");
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}
