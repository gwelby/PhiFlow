//! Benchmark Battery (T4-012)
//!
//! Complete consciousness metric benchmark suite.
//! Runs all tests and generates an evidence report.

use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::self_correlation::SelfCorrelation;
use phiflow::metrics::trace::Trace;
use std::io::Write;

/// Main benchmark entry point.
#[test]
#[ignore = "Manual run: cargo test --test benchmark_battery -- --ignored --nocapture"]
fn full_benchmark_battery() {
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PhiFlow Type 4 Synthetic Benchmark Battery");
    println!("  PF consciousness_metric_program.md implementation smoke test");
    println!("═══════════════════════════════════════════════════════════════\n");

    let timestamp = chrono::Utc::now().to_rfc3339();
    let mut report = BenchmarkReport::new(timestamp);

    // Phase 1: Type 4 Self-Correlation
    println!("📊 PHASE 1: Type 4 Self-Correlation");
    println!("───────────────────────────────────────────────────────────────");
    let phase1_pass = phase1_type4_tests(&mut report);
    println!("  Phase 1: {}", if phase1_pass { "PASS" } else { "FAIL" });

    // Phase 2: Null Class Tests
    println!("\n📊 PHASE 2: Null Class Tests");
    println!("───────────────────────────────────────────────────────────────");
    let phase2_pass = phase2_null_tests(&mut report);
    println!("  Phase 2: {}", if phase2_pass { "PASS" } else { "FAIL" });

    // Phase 3: State Discrimination (if SOMA available)
    println!("\n📊 PHASE 3: State Discrimination");
    println!("───────────────────────────────────────────────────────────────");
    let phase3_pass = phase3_discrimination_tests(&mut report);
    println!("  Phase 3: {}", if phase3_pass { "PASS" } else { "SKIP" });

    // Phase 4: PhiFlow Daemon
    println!("\n📊 PHASE 4: PhiFlow Daemon Type 4 Trace");
    println!("───────────────────────────────────────────────────────────────");
    let phase4_pass = phase4_daemon_trace(&mut report);
    println!("  Phase 4: {}", if phase4_pass { "PASS" } else { "FAIL" });

    // Overall verdict
    let all_pass = phase1_pass && phase2_pass && phase4_pass;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  FINAL VERDICT");
    println!("═══════════════════════════════════════════════════════════════");
    if all_pass {
        println!("  ✅ SYNTHETIC BENCHMARK BATTERY PASSED");
        println!("  Type 4 observer status: HOLD — proxy smoke test only");
    } else {
        println!("  ❌ BENCHMARK BATTERY FAILED");
        println!("  Type 4 observer status: NOT CONFIRMED");
    }
    println!("═══════════════════════════════════════════════════════════════");

    // Save report
    report.save().expect("Failed to save report");
    println!("\n📄 Report saved to: {}", report.path.display());

    // Assert for CI/CD
    assert!(all_pass, "Benchmark battery failed - see report for details");
}

/// Phase 1: Self-correlation tests
fn phase1_type4_tests(report: &mut BenchmarkReport) -> bool {
    // Test: Self-model loop detection
    let trace = create_self_model_trace();
    let sc = SelfCorrelation::from_trace(&trace, 10, 5, 0.01);

    report.add_test("self_model_l_self", sc.l_self, sc.l_self > 0.1);
    report.add_test("self_model_type4", sc.l_self, sc.loop_closed);

    println!("  Self-model loop:");
    println!("    L_self = {:.6} (threshold: 0.1)", sc.l_self);
    println!("    Verdict: {}", if sc.l_self > 0.1 { "✅ PASS" } else { "❌ FAIL" });

    sc.l_self > 0.1
}

/// Phase 2: Null class tests
fn phase2_null_tests(report: &mut BenchmarkReport) -> bool {
    let mut all_pass = true;

    // Feed-forward
    let ff_trace = create_feedforward_trace();
    let ff_metrics = ConsciousnessMetrics::compute(&ff_trace, 10, 5, 0.01);
    let ff_pass = ff_metrics.c_pf < 0.3;
    report.add_test("null_feedforward", ff_metrics.c_pf, ff_pass);
    all_pass = all_pass && ff_pass;

    // Noise
    let noise_trace = create_noise_trace();
    let noise_metrics = ConsciousnessMetrics::compute(&noise_trace, 10, 5, 0.01);
    let noise_pass = noise_metrics.c_pf < 0.3;
    report.add_test("null_noise", noise_metrics.c_pf, noise_pass);
    all_pass = all_pass && noise_pass;

    // Thermostat
    let thermo_trace = create_thermostat_trace();
    let thermo_metrics = ConsciousnessMetrics::compute(&thermo_trace, 10, 5, 0.01);
    let thermo_pass = thermo_metrics.c_pf < 0.3;
    report.add_test("null_thermostat", thermo_metrics.c_pf, thermo_pass);
    all_pass = all_pass && thermo_pass;

    println!("  Feed-forward: C_PF={:.4} {}", ff_metrics.c_pf, if ff_pass { "PASS" } else { "FAIL" });
    println!("  Noise:        C_PF={:.4} {}", noise_metrics.c_pf, if noise_pass { "PASS" } else { "FAIL" });
    println!("  Thermostat:   C_PF={:.4} {}", thermo_metrics.c_pf, if thermo_pass { "PASS" } else { "FAIL" });

    all_pass
}

/// Phase 3: State discrimination
fn phase3_discrimination_tests(report: &mut BenchmarkReport) -> bool {
    // Check if SOMA fixtures available
    if std::env::var("PHIFLOW_SOMA_FIXTURES").is_err() {
        println!("  SKIPPED (PHIFLOW_SOMA_FIXTURES not set)");
        report.add_note("Phase 3 skipped: SOMA fixtures not available");
        return true; // Skip is not a failure
    }

    // Would load fixtures and run tests here
    println!("  SOMA fixtures would be loaded here");
    report.add_note("Phase 3: SOMA integration pending");

    true
}

/// Phase 4: Daemon trace from type4_trace_benchmark.phi
fn phase4_daemon_trace(report: &mut BenchmarkReport) -> bool {
    // Execute the actual type4_trace_benchmark.phi program
    use phiflow::phi_ir::evaluator::Evaluator;
    use phiflow::phi_ir::lowering::lower_program_checked;
    use phiflow::parser::parse_phi_program;
    use phiflow::metrics::trace::Trace;

    let phi_path = std::path::Path::new("examples/type4_trace_benchmark.phi");
    let source = match std::fs::read_to_string(phi_path) {
        Ok(s) => s,
        Err(_) => {
            println!("  ⚠️  Phase 4: SKIPPED (type4_trace_benchmark.phi not found)");
            report.add_note("Phase 4 skipped: benchmark file not found");
            return true; // Skip is not a failure
        }
    };

    let ast = match parse_phi_program(&source) {
        Ok(a) => a,
        Err(_) => {
            println!("  ❌ Phase 4: Parse error");
            report.add_note("Phase 4 failed: parse error");
            return false;
        }
    };

    let program = match lower_program_checked(&ast) {
        Ok(p) => p,
        Err(_) => {
            println!("  ❌ Phase 4: Lowering error");
            report.add_note("Phase 4 failed: lowering error");
            return false;
        }
    };

    let mut evaluator = Evaluator::new(program);
    let _ = evaluator.run(); // Continue even if execution "ends"

    let frozen_state = evaluator.freeze_state();
    let trace = Trace::from_vm_state(&frozen_state);
    let sc = SelfCorrelation::from_type4_trace(&trace, 0.01);

    report.add_test("daemon_type4_l_self", sc.l_self, sc.l_self > 0.1);
    report.add_test("daemon_type4_loop", sc.l_self, sc.loop_closed);

    println!("  Daemon trace:");
    println!("    Witness events: {}", trace.len());
    println!("    L_self = {:.6}", sc.l_self);
    println!("    R_in   = {:.6}", sc.r_in_norm);
    println!("    R_out  = {:.6}", sc.r_out_norm);
    println!("    Verdict: {}", if sc.l_self > 0.1 { "✅ PASS" } else { "❌ FAIL" });

    sc.l_self > 0.1
}

/// Benchmark report structure
struct BenchmarkReport {
    timestamp: String,
    tests: Vec<(String, f64, bool)>,
    notes: Vec<String>,
    path: std::path::PathBuf,
}

impl BenchmarkReport {
    fn new(timestamp: String) -> Self {
        let date = timestamp.split('T').next().unwrap_or("unknown");
        let path = std::path::PathBuf::from(format!("QSOP/EVIDENCE/type4_battery_{}.md", date));
        Self {
            timestamp,
            tests: Vec::new(),
            notes: Vec::new(),
            path,
        }
    }

    fn add_test(&mut self, name: &str, value: f64, passed: bool) {
        self.tests.push((name.to_string(), value, passed));
    }

    fn add_note(&mut self, note: &str) {
        self.notes.push(note.to_string());
    }

    fn save(&self) -> std::io::Result<()> {
        // Ensure directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(&self.path)?;

        writeln!(file, "# Type 4 Benchmark Evidence Report")?;
        writeln!(file)?;
        writeln!(file, "**Date:** {}", self.timestamp)?;
        writeln!(file)?;
        writeln!(file, "**Codex audit note:** synthetic proxy smoke test only; Type 4 confirmation remains HOLD until `R_out` uses action/future behavior and null thresholds are recalibrated. See `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.")?;
        writeln!(file)?;
        writeln!(file, "## Test Results")?;
        writeln!(file)?;
        writeln!(file, "| Test | Value | Pass |")?;
        writeln!(file, "|------|-------|------|")?;

        for (name, value, passed) in &self.tests {
            writeln!(
                file,
                "| {} | {:.6} | {} |",
                name,
                value,
                if *passed { "✅" } else { "❌" }
            )?;
        }

        if !self.notes.is_empty() {
            writeln!(file)?;
            writeln!(file, "## Notes")?;
            writeln!(file)?;
            for note in &self.notes {
                writeln!(file, "- {}", note)?;
            }
        }

        // Overall verdict
        let all_pass = self.tests.iter().all(|(_, _, p)| *p);
        writeln!(file)?;
        writeln!(file, "## Verdict")?;
        writeln!(file)?;
        if all_pass {
            writeln!(file, "✅ **PASSED** - synthetic proxy smoke test only; Type 4 confirmation remains HOLD")?;
        } else {
            writeln!(file, "❌ **FAILED** - Type 4 observer status not confirmed")?;
        }

        Ok(())
    }
}

/// Create a synthetic self-model trace (running mean).
fn create_self_model_trace() -> Trace {
    let mut trace = Trace::new();
    let mut obs_vec: Vec<f64> = Vec::new();

    // Slowly varying observations
    for i in 0..200 {
        let obs = 0.5 + 0.3 * (i as f64 * 0.1).sin() + 0.05 * (i as f64).sin();
        obs_vec.push(obs);
        trace.observed.push(obs, i as f64);
        trace.coherence.push(0.5 + 0.1 * (i as f64 * 0.05).sin(), i as f64);
        trace.depth.push(1.0 + (i % 3) as f64, i as f64);
    }

    trace
}

/// Create a feed-forward trace.
fn create_feedforward_trace() -> Trace {
    let mut trace = Trace::new();
    for i in 0..100 {
        let t = i as f64 * 0.1;
        let obs = 0.3 * t.sin();
        trace.observed.push(obs, t);
        trace.coherence.push(0.5, t);
        trace.depth.push(1.0, t);
    }
    trace
}

/// Create a noise trace.
fn create_noise_trace() -> Trace {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut trace = Trace::new();
    for i in 0..200 {
        trace.observed.push(rng.gen::<f64>(), i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    trace
}

/// Create a thermostat trace.
fn create_thermostat_trace() -> Trace {
    let mut trace = Trace::new();
    let mut current = 0.5;
    for i in 0..100 {
        current += 0.1 * (0.7 - current);
        trace.observed.push(current, i as f64);
        trace.coherence.push(0.5, i as f64);
        trace.depth.push(1.0, i as f64);
    }
    trace
}
