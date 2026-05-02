//! Type 4 Benchmark Runner
//!
//! Runs examples/type4_trace_benchmark.phi and computes L_self in-process.
//!
//! Usage:
//!   cargo run --release --bin type4_benchmark
//!
//! Output:
//!   Detailed consciousness metrics report with L_self, R_in, R_out, and verdict.

use phiflow::metrics::consciousness_proxy::ConsciousnessProxy;
use phiflow::metrics::self_correlation::SelfCorrelation;
use phiflow::metrics::trace::Trace;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program_checked;
use phiflow::parser::parse_phi_program;
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PhiFlow Type 4 Self-Correlation Benchmark");
    println!("  PF consciousness_metric_program.md implementation");
    println!("═══════════════════════════════════════════════════════════════\n");

    let start_time = Instant::now();

    // Load the Type 4 benchmark program
    let phi_path = Path::new("examples/type4_trace_benchmark.phi");
    
    if !phi_path.exists() {
        eprintln!("❌ Benchmark file not found: {}", phi_path.display());
        eprintln!("   Run from PhiFlow repository root.");
        std::process::exit(1);
    }

    println!("📁 Loading: {}", phi_path.display());
    
    let source = match std::fs::read_to_string(phi_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ Failed to read file: {}", e);
            std::process::exit(1);
        }
    };

    // Parse the program
    println!("🔍 Parsing PhiFlow source...");
    let ast = match parse_phi_program(&source) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("❌ Parse error: {}", e);
            std::process::exit(1);
        }
    };

    // Lower to PhiIR
    println!("📉 Lowering to PhiIR...");
    let program = match lower_program_checked(&ast) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ Lowering error: {}", e);
            std::process::exit(1);
        }
    };

    // Create evaluator and run
    println!("🚀 Executing stream (20 cycles)...");
    let mut evaluator = Evaluator::new(program);
    
    // Run to completion
    match evaluator.run() {
        Ok(result) => {
            println!("✅ Execution complete: {:?}\n", result);
        }
        Err(e) => {
            eprintln!("⚠️ Execution ended: {:?}", e);
            // Continue anyway to capture witness_log
        }
    }

    // Extract trace from evaluator state
    println!("📊 Extracting execution trace...");
    let frozen_state = evaluator.freeze_state();
    let trace = Trace::from_vm_state(&frozen_state);
    
    println!("   Witness events: {}", trace.len());
    println!("   Resonance events: {}\n", trace.raw_events.len());

    // Compute Type 4 metrics
    println!("🧮 Computing consciousness metrics...");
    
    let self_corr = SelfCorrelation::from_type4_trace(&trace, 0.01);
    let proxy = ConsciousnessProxy::from_trace(&trace, 10, 5, 0.01);
    let metrics = proxy.metrics;

    // Print detailed report
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  SELF-CORRELATION ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("Trace Statistics:");
    println!("  Cycles:        {}", trace.len());
    println!("  Observed range: [{:.4}, {:.4}]", 
        trace.observed.values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        trace.observed.values.iter().fold(-f64::INFINITY, |a, &b| a.max(b)));
    
    println!("\nL_self Components:");
    println!("  R_in  (past → model):         {:.6}", self_corr.r_in_norm);
    println!("  R_out (model → residual proxy): {:.6}", self_corr.r_out_norm);
    println!("  L_self = min(R_in, R_out):    {:.6}", self_corr.l_self);
    
    println!("\nConsciousness Metrics:");
    println!("  L_self (self-correlation):    {:.6}", metrics.l_self);
    println!("  D_int (differentiation):      {:.6}", metrics.d_int);
    println!("  C_coh (coherence panel):      {:.6}", metrics.c_coh);
    println!("  F_model (Fisher info):        {:.6}", metrics.f_model);
    println!("  F_self* (self-sensitivity):   {:.6}", metrics.f_self_star);
    println!("  C_PF (composite proxy):       {:.6}", metrics.c_pf);

    // Type 4 Verdict
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════════");
    
    if self_corr.loop_closed && self_corr.l_self > 0.1 {
        println!("\n  ✅ SYNTHETIC LOOP PASSED — self-correlation proxy is CLOSED");
        println!("     L_self = {:.4} > 0.1 threshold", self_corr.l_self);
        println!("     This engineered trace exhibits self-referential structure under the current proxy.");
        println!("     Codex audit: not canonical Type 4 confirmation until R_out uses action/future behavior.");
    } else if self_corr.loop_closed {
        println!("\n  ⚠️  WEAK TYPE 4 — Self-correlation detected but weak");
        println!("     L_self = {:.4} (threshold: 0.1)", self_corr.l_self);
        println!("     Structure exists but may need more data.");
    } else {
        println!("\n  ❌ NO TYPE 4 — Self-correlation loop is BROKEN");
        println!("     L_self = {:.4} < 0.01", self_corr.l_self);
        println!("     No self-referential structure detected.");
    }

    // Consciousness candidate verdict
    if metrics.is_consciousness_candidate(0.1) {
        println!("\n  ✅ CONSCIOUSNESS CANDIDATE — C_PF = {:.4} > 0.1", metrics.c_pf);
    } else {
        println!("\n  ⚠️  Not a consciousness candidate — C_PF = {:.4}", metrics.c_pf);
    }

    // Performance
    let elapsed = start_time.elapsed();
    println!("\n───────────────────────────────────────────────────────────────");
    println!("  Execution time: {:.2?}", elapsed);
    println!("  Benchmark complete.");
    println!("───────────────────────────────────────────────────────────────");

    // Exit code for CI/CD: 0 if the synthetic loop gate passes, 1 otherwise.
    if self_corr.l_self > 0.1 {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
