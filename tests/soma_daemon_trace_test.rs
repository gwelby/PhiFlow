//! SOMA Daemon-Coupled Trace Test (T4 round 2)
//!
//! Scores a trace produced by examples/type4_soma_daemon.phi running under
//! `phic --daemon` with live SOMA — the program's own mutable agent state is
//! the self-model; no host-side proxy. Preregistered criteria and the
//! distinction from the 2026-10-07 collector FAIL are in
//! QSOP/EVIDENCE/type4_daemon_prereg_2026-10-09.md.
//!
//! Trace path: env SOMA_DAEMON_TRACE (default tests/fixtures/soma_daemon_trace.txt).
//! The collector baseline fixture (soma_live_trace.txt) is left untouched.

use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::self_correlation::SelfCorrelation;
use phiflow::metrics::trace::Trace;
use std::path::PathBuf;

/// T4-R2: daemon-coupled SOMA trace — program-carried self-model.
#[test]
fn test_soma_daemon_trace_type4() {
    let fixture_path = std::env::var("SOMA_DAEMON_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/fixtures/soma_daemon_trace.txt"));

    if !fixture_path.exists() {
        println!("SKIPPED: daemon-coupled trace fixture not found.");
        println!("  Run: phic --daemon --measure examples/type4_soma_daemon.phi");
        return;
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  T4-R2: Daemon-Coupled SOMA Trace — Program-Carried Model");
    println!("═══════════════════════════════════════════════════════════════\n");

    let trace = Trace::from_trace_file(&fixture_path).expect("Failed to parse trace file");

    println!("Loaded trace: {} cycles", trace.len());
    println!("  Observed range: [{:.4}, {:.4}]",
        trace.observed.values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        trace.observed.values.iter().fold(-f64::INFINITY, |a, &b| a.max(b)));

    // Same canonical path as the 2026-10-07 collector FAIL
    let self_corr = SelfCorrelation::from_type4_trace(&trace, 0.01);
    let metrics = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);

    println!("\nSelf-Correlation:");
    println!("  R_in  = {:.6}", self_corr.r_in_norm);
    println!("  R_out = {:.6}", self_corr.r_out_norm);
    println!("  L_self = {:.6}", self_corr.l_self);

    println!("\nConsciousness Metrics:");
    println!("  D_int   = {:.6}", metrics.d_int);
    println!("  C_coh   = {:.6}", metrics.c_coh);
    println!("  F_model = {:.6}", metrics.f_model);
    println!("  F_self* = {:.6}", metrics.f_self_star);
    println!("  C_PF    = {:.6}", metrics.c_pf);

    // Preregistered verdict: same threshold as the failed baseline
    let calibrated_threshold = 0.0021;
    println!("\n═══════════════════════════════════════════════════════════════");
    if self_corr.l_self > 0.1 {
        println!("  ✅ T4-R2 CLOSED — daemon-coupled trace L_self = {:.4} > 0.1", self_corr.l_self);
    } else {
        println!("  ❌ T4-R2 OPEN — L_self = {:.4} <= 0.1 (second real-trace negative)", self_corr.l_self);
    }
    println!("     C_PF = {:.4} vs calibrated null {:.4}", metrics.c_pf, calibrated_threshold);
    println!("═══════════════════════════════════════════════════════════════");
}
