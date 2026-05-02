//! State Discrimination Tests (T4-011)
//!
//! Tests that distinguish conscious vs. unconscious states.
//! Requires SOMA fixtures or synthetic proxies.
//!
//! Environment: PHIFLOW_SOMA_FIXTURES=<path_to_fixtures>
//! If not set, uses synthetic proxies as pre-flight.

use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::trace::Trace;

/// Wakeful state: high L_self, high D_int, high C_coh.
/// Expected: L_self > 0.3, C_PF > 0.1
#[test]
#[ignore = "Requires SOMA fixtures or synthetic wake proxy"]
fn state_wakeful() {
    let trace = load_or_synthesize_fixture("wakeful");
    let metrics = ConsciousnessMetrics::compute(&trace, 100, 10, 0.01);

    println!("Wakeful state:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  D_int  = {:.6}", metrics.d_int);
    println!("  C_coh  = {:.6}", metrics.c_coh);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    assert!(
        metrics.l_self > 0.3,
        "Wakeful should have L_self > 0.3, got {:.6}",
        metrics.l_self
    );
    assert!(
        metrics.c_pf > 0.1,
        "Wakeful should have C_PF > 0.1, got {:.6}",
        metrics.c_pf
    );
}

/// Deep sleep state: low L_self, low D_int, low C_coh.
/// Expected: L_self < 0.2, C_PF < 0.05
#[test]
#[ignore = "Requires SOMA fixtures or synthetic sleep proxy"]
fn state_deep_sleep() {
    let trace = load_or_synthesize_fixture("deep_sleep");
    let metrics = ConsciousnessMetrics::compute(&trace, 100, 10, 0.01);

    println!("Deep sleep state:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  D_int  = {:.6}", metrics.d_int);
    println!("  C_coh  = {:.6}", metrics.c_coh);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    assert!(
        metrics.l_self < 0.2,
        "Deep sleep should have L_self < 0.2, got {:.6}",
        metrics.l_self
    );
    assert!(
        metrics.c_pf < 0.05,
        "Deep sleep should have C_PF < 0.05, got {:.6}",
        metrics.c_pf
    );
}

/// Anesthesia: very low coherence, minimal self-model.
/// Expected: L_self ≈ 1/sqrt(N), C_PF < 0.01
#[test]
#[ignore = "Requires SOMA fixtures or synthetic anesthesia proxy"]
fn state_anesthesia() {
    let trace = load_or_synthesize_fixture("anesthesia");
    let metrics = ConsciousnessMetrics::compute(&trace, 100, 10, 0.01);

    println!("Anesthesia state:");
    println!("  L_self = {:.6}", metrics.l_self);
    println!("  C_PF   = {:.6}", metrics.c_pf);

    // Anesthesia should be nearly indistinguishable from noise
    assert!(
        metrics.c_pf < 0.01,
        "Anesthesia should have C_PF < 0.01, got {:.6}",
        metrics.c_pf
    );
}

/// Discrimination test: wake vs sleep should be separable.
#[test]
#[ignore = "Requires SOMA fixtures"]
fn discrimination_wake_vs_sleep() {
    let wake_trace = load_fixture("wakeful");
    let sleep_trace = load_fixture("deep_sleep");

    let wake_metrics = ConsciousnessMetrics::compute(&wake_trace, 100, 10, 0.01);
    let sleep_metrics = ConsciousnessMetrics::compute(&sleep_trace, 100, 10, 0.01);

    println!("Wake vs Sleep discrimination:");
    println!("  Wake:  L_self={:.4} C_PF={:.4}", wake_metrics.l_self, wake_metrics.c_pf);
    println!("  Sleep: L_self={:.4} C_PF={:.4}", sleep_metrics.l_self, sleep_metrics.c_pf);

    // Wake should have higher C_PF than sleep
    assert!(
        wake_metrics.c_pf > sleep_metrics.c_pf * 2.0,
        "Wake C_PF should be > 2x Sleep C_PF"
    );
}

/// Synthetic wake proxy: high-entropy structured data.
fn synthetic_wake_proxy() -> Trace {
    let mut trace = Trace::new();

    // Wakeful: multiple interacting frequencies (structured but complex)
    for i in 0..1000 {
        let t = i as f64 * 0.1;
        let obs = (t).sin() * 0.3
            + (t * 1.618).sin() * 0.2
            + (t * 2.414).sin() * 0.15
            + (i as f64 % 10.0) * 0.01; // Some modulation

        trace.observed.push(obs, t);
        trace.coherence.push(0.6 + 0.2 * (t).sin(), t); // Modulated coherence
        trace.depth.push(2.0 + (t * 0.1).sin(), t);    // Varying depth
    }

    trace
}

/// Synthetic sleep proxy: low-entropy, periodic but simple.
fn synthetic_sleep_proxy() -> Trace {
    let mut trace = Trace::new();

    // Deep sleep: simple periodic with low variability
    for i in 0..1000 {
        let t = i as f64 * 0.1;
        let obs = (t * 0.5).sin() * 0.1 + 0.5; // Low amplitude, simple

        trace.observed.push(obs, t);
        trace.coherence.push(0.4, t); // Low, constant coherence
        trace.depth.push(1.0, t);     // Minimal depth
    }

    trace
}

/// Synthetic anesthesia proxy: white noise, no structure.
fn synthetic_anesthesia_proxy() -> Trace {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();

    for i in 0..1000 {
        let t = i as f64 * 0.1;
        let obs = rng.gen::<f64>() * 0.5 + 0.25; // White noise

        trace.observed.push(obs, t);
        trace.coherence.push(0.3 + rng.gen::<f64>() * 0.1, t);
        trace.depth.push(0.5, t);
    }

    trace
}

/// Load fixture from SOMA or synthesize.
fn load_or_synthesize_fixture(name: &str) -> Trace {
    match std::env::var("PHIFLOW_SOMA_FIXTURES") {
        Ok(path) => load_fixture_from_path(&path, name),
        Err(_) => {
            println!("⚠️  PHIFLOW_SOMA_FIXTURES not set, using synthetic {} proxy", name);
            match name {
                "wakeful" => synthetic_wake_proxy(),
                "deep_sleep" => synthetic_sleep_proxy(),
                "anesthesia" => synthetic_anesthesia_proxy(),
                _ => panic!("Unknown fixture: {}", name),
            }
        }
    }
}

/// Load fixture from path (expects <path>/<name>.json format).
fn load_fixture_from_path(base_path: &str, name: &str) -> Trace {
    use std::path::Path;

    let path = Path::new(base_path).join(format!("{}.json", name));
    if !path.exists() {
        panic!("Fixture not found: {}", path.display());
    }

    // Parse JSON fixture into Trace
    // Format: { "coherence": [...], "observed": [...], ... }
    let content = std::fs::read_to_string(&path).expect("Failed to read fixture");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse fixture");

    let mut trace = Trace::new();

    if let Some(arr) = json["observed"].as_array() {
        for (i, val) in arr.iter().enumerate() {
            if let Some(v) = val.as_f64() {
                trace.observed.push(v, i as f64);
            }
        }
    }

    if let Some(arr) = json["coherence"].as_array() {
        for (i, val) in arr.iter().enumerate() {
            if let Some(v) = val.as_f64() {
                trace.coherence.push(v, i as f64);
            }
        }
    }

    if let Some(arr) = json["depth"].as_array() {
        for (i, val) in arr.iter().enumerate() {
            if let Some(v) = val.as_f64() {
                trace.depth.push(v, i as f64);
            }
        }
    }

    trace
}

/// Load fixture (always from SOMA, no fallback).
fn load_fixture(name: &str) -> Trace {
    let path = std::env::var("PHIFLOW_SOMA_FIXTURES")
        .expect("PHIFLOW_SOMA_FIXTURES must be set for this test");
    load_fixture_from_path(&path, name)
}
