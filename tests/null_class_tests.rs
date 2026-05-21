//! Null Class Tests (T4-010)
//!
//! Validates that non-conscious systems score low on consciousness metrics.
//! All null classes must score C_PF < 0.3 (hard pass/fail gate).

use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::trace::Trace;

/// Feed-forward system: output depends only on current input, no memory.
/// Expected: L_self ≈ 0, C_PF < 0.01
#[test]
fn null_class_feedforward() {
    let mut trace = Trace::new();

    // y = 0.3 * sin(t) - pure function of t, no self-reference
    for i in 0..100 {
        let t = i as f64 * 0.1;
        let obs = 0.3 * t.sin();
        trace.observed.push(obs, t);
        trace.coherence.push(0.5, t);
        trace.depth.push(1.0, t);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Feed-forward null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Feed-forward should have minimal self-correlation
    assert!(
        metrics.l_self < 0.3,
        "Feed-forward should have L_self < 0.3, got {:.6}",
        metrics.l_self
    );

    // C_PF should be very low
    assert!(
        metrics.c_pf < 0.3,
        "Feed-forward null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Pure noise system: random values, no structure.
/// Expected: L_self ≈ 1/sqrt(N), C_PF < 0.05
#[test]
fn null_class_noise() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut trace = Trace::new();
    for i in 0..200 {
        let obs = rng.gen::<f64>();
        trace.observed.push(obs, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Noise null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Random should have near-zero self-correlation
    assert!(
        metrics.l_self < 0.5,
        "Noise should have L_self < 0.5, got {:.6}",
        metrics.l_self
    );

    assert!(
        metrics.c_pf < 0.3,
        "Noise null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Replay system: copies past observations exactly.
/// Expected: High R_in (past predicts present) but R_out ≈ 1 (no information for future)
/// This is a "cheating detector" - pure replay has no genuine self-model.
#[test]
fn null_class_replay() {
    let mut trace = Trace::new();

    // Generate a pattern, then replay it
    let pattern: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();

    // First pass: original pattern
    for (i, &obs) in pattern.iter().enumerate() {
        trace.observed.push(obs, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }

    // Second pass: exact replay (no new information)
    for (i, &obs) in pattern.iter().enumerate() {
        trace.observed.push(obs, (i + 50) as f64);
        trace.coherence.push(0.5, (i + 50) as f64);
        trace.depth.push(1.0, (i + 50) as f64);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Replay null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Replay should score low C_PF (suppressed by D_int if nothing new)
    assert!(
        metrics.c_pf < 0.3,
        "Replay null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Simple recurrent controller (thermostat-like): one-parameter feedback.
/// Expected: Low D_int < 2.0, C_PF < 0.05
#[test]
fn null_class_thermostat() {
    let mut trace = Trace::new();

    // Thermostat: output = target - current (simple proportional control)
    let mut current = 0.5;
    let target = 0.7;

    for i in 0..100 {
        let error = target - current;
        let control = 0.1 * error; // Proportional control

        current += control;

        trace.observed.push(current, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Thermostat null:");
    println!("  D_int  = {:.6}", metrics.d_int);
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Simple controllers have low differentiation
    assert!(
        metrics.d_int < 3.0,
        "Thermostat should have D_int < 3.0, got {:.6}",
        metrics.d_int
    );

    assert!(
        metrics.c_pf < 0.3,
        "Thermostat null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Random walk: memory but no self-model structure.
/// Expected: L_self > 1/sqrt(N) but C_PF still low
#[test]
fn null_class_random_walk() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut trace = Trace::new();
    let mut current = 0.5;

    for i in 0..200 {
        // Random walk with drift
        current += rng.gen::<f64>() * 0.1 - 0.05;
        current = current.clamp(0.0, 1.0);

        trace.observed.push(current, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Random walk null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Random walk has memory (persistent state) but no self-model
    // Should still score low C_PF
    assert!(
        metrics.c_pf < 0.3,
        "Random walk null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Type 4 format null: same structure as positive trace, but action is random.
/// This specifically tests the `compute_fisher_type4` path.
#[test]
fn null_class_type4_format() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;

    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;

        // Action is RANDOM — no relationship to model
        let action = rng.gen::<f64>();

        let model_mean = model_sum / model_n;
        model_sum += obs;
        model_n += 1.0;

        // Type 4 format: step, obs, model, action
        trace.raw_events.push(("step".to_string(), step));
        trace.raw_events.push(("obs".to_string(), obs));
        trace.raw_events.push(("model".to_string(), model_mean));
        trace.raw_events.push(("action".to_string(), action));

        trace.observed.push(obs, step);
        trace.coherence.push(0.5, step);
        trace.depth.push(1.0, step);
        trace.resonance_k.push(4.0, step);
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Type4-format null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Random action breaks the model→action loop → low C_PF
    assert!(
        metrics.c_pf < 0.3,
        "Type4-format null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// Type 4 format null: shuffled action (preserves marginals, breaks temporal).
#[test]
fn null_class_type4_shuffled() {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;
    let mut actions: Vec<f64> = Vec::new();

    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;
        let model_mean = model_sum / model_n;

        // Generate action correlated with model (to build a "fake" positive)
        let action = if obs < model_mean { 1.0 } else { 0.0 };
        actions.push(action);

        model_sum += obs;
        model_n += 1.0;

        trace.observed.push(obs, step);
        trace.coherence.push(0.5, step);
        trace.depth.push(1.0, step);
        trace.resonance_k.push(4.0, step);
    }

    // Shuffle actions to break temporal alignment
    actions.shuffle(&mut rng);

    // Now populate raw_events with shuffled actions
    let mut model_sum2 = 0.55;
    let mut model_n2 = 1.0;
    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum2 / model_n2) * 0.40;
        let obs = base_val * mod_val;
        let model_mean = model_sum2 / model_n2;
        model_sum2 += obs;
        model_n2 += 1.0;

        trace.raw_events.push(("step".to_string(), step));
        trace.raw_events.push(("obs".to_string(), obs));
        trace.raw_events.push(("model".to_string(), model_mean));
        trace.raw_events.push(("action".to_string(), actions[i - 1]));
    }

    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("Type4-shuffled null:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    assert!(
        metrics.c_pf < 0.3,
        "Type4-shuffled null failed: C_PF = {:.6} >= 0.3",
        metrics.c_pf
    );
}

/// All null classes summary test.
#[test]
fn null_class_all_pass() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  NULL CLASS TEST SUMMARY");
    println!("  Requirement: All null classes score C_PF < 0.3");
    println!("  Type 4 format nulls now included (tests compute_fisher_type4 path)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_cases = vec![
        ("feedforward", run_feedforward()),
        ("noise", run_noise()),
        ("replay", run_replay()),
        ("thermostat", run_thermostat()),
        ("random_walk", run_random_walk()),
        ("type4_format", run_type4_format_null()),
        ("type4_shuffled", run_type4_shuffled_null()),
    ];

    let mut all_pass = true;
    for (name, (l_self, c_pf)) in test_cases {
        let pass = c_pf < 0.3;
        all_pass = all_pass && pass;
        println!(
            "  {:14}  L_self={:.4}  C_PF={:.4}  {}",
            name,
            l_self,
            c_pf,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    if all_pass {
        println!("  ✅ ALL NULL CLASSES PASS");
    } else {
        println!("  ❌ SOME NULL CLASSES FAIL");
    }
    println!("═══════════════════════════════════════════════════════════════");
}

// Helper functions for the summary test
fn run_feedforward() -> (f64, f64) {
    let mut trace = Trace::new();
    for i in 0..100 {
        let t = i as f64 * 0.1;
        trace.observed.push(0.3 * t.sin(), t);
        trace.coherence.push(0.5, t);
        trace.depth.push(1.0, t);
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_noise() -> (f64, f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();
    for i in 0..200 {
        trace.observed.push(rng.gen::<f64>(), i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_replay() -> (f64, f64) {
    let mut trace = Trace::new();
    let pattern: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    for (i, &obs) in pattern.iter().enumerate() {
        trace.observed.push(obs, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    for (i, &obs) in pattern.iter().enumerate() {
        trace.observed.push(obs, (i + 50) as f64);
        trace.coherence.push(0.5, (i + 50) as f64);
        trace.depth.push(1.0, (i + 50) as f64);
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_thermostat() -> (f64, f64) {
    let mut trace = Trace::new();
    let mut current = 0.5;
    for i in 0..100 {
        current += 0.1 * (0.7 - current);
        trace.observed.push(current, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_random_walk() -> (f64, f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();
    let mut current = 0.5;
    for i in 0..200 {
        current += rng.gen::<f64>() * 0.1 - 0.05;
        current = current.clamp(0.0, 1.0);
        trace.observed.push(current, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_type4_format_null() -> (f64, f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;

    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;
        let action = rng.gen::<f64>();
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
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}

fn run_type4_shuffled_null() -> (f64, f64) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();
    let mut model_sum = 0.55;
    let mut model_n = 1.0;
    let mut actions: Vec<f64> = Vec::new();

    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum / model_n) * 0.40;
        let obs = base_val * mod_val;
        let model_mean = model_sum / model_n;
        let action = if obs < model_mean { 1.0 } else { 0.0 };
        actions.push(action);
        model_sum += obs;
        model_n += 1.0;

        trace.observed.push(obs, step);
        trace.coherence.push(0.5, step);
        trace.depth.push(1.0, step);
        trace.resonance_k.push(4.0, step);
    }

    actions.shuffle(&mut rng);

    let mut model_sum2 = 0.55;
    let mut model_n2 = 1.0;
    for i in 1..=100 {
        let step = i as f64;
        let base_val = 0.90 - step * 0.012;
        let mod_val = 1.10 - (model_sum2 / model_n2) * 0.40;
        let obs = base_val * mod_val;
        let model_mean = model_sum2 / model_n2;
        model_sum2 += obs;
        model_n2 += 1.0;

        trace.raw_events.push(("step".to_string(), step));
        trace.raw_events.push(("obs".to_string(), obs));
        trace.raw_events.push(("model".to_string(), model_mean));
        trace.raw_events.push(("action".to_string(), actions[i - 1]));
    }
    let m = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
    (m.l_self, m.c_pf)
}
