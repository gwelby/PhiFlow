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
    println!("  Phase 3: {}", if phase3_pass { "PASS" } else { "FAIL" });

    // Phase 4: PhiFlow Daemon
    println!("\n📊 PHASE 4: PhiFlow Daemon Type 4 Trace");
    println!("───────────────────────────────────────────────────────────────");
    let phase4_pass = phase4_daemon_trace(&mut report);
    println!("  Phase 4: {}", if phase4_pass { "PASS" } else { "FAIL" });

    // Overall verdict — T4-04 fix: Phase 3 is now required
    let all_pass = phase1_pass && phase2_pass && phase3_pass && phase4_pass;

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
    assert!(
        all_pass,
        "Benchmark battery failed - see report for details"
    );
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
    println!(
        "    Verdict: {}",
        if sc.l_self > 0.1 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

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

    println!(
        "  Feed-forward: C_PF={:.4} {}",
        ff_metrics.c_pf,
        if ff_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Noise:        C_PF={:.4} {}",
        noise_metrics.c_pf,
        if noise_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Thermostat:   C_PF={:.4} {}",
        thermo_metrics.c_pf,
        if thermo_pass { "PASS" } else { "FAIL" }
    );

    all_pass
}

/// Phase 3: State discrimination
///
/// Loads wakeful and deep_sleep fixtures from `PHIFLOW_SOMA_FIXTURES` and
/// verifies that C_PF discriminates the two states (wake C_PF > 2x sleep C_PF).
/// Also checks individual state thresholds.
///
/// If `PHIFLOW_SOMA_FIXTURES` is not set, Phase 3 FAILS — skip is not a pass
/// (Codex guardrail, 2026-06-17).
fn phase3_discrimination_tests(report: &mut BenchmarkReport) -> bool {
    let fixture_path = match std::env::var("PHIFLOW_SOMA_FIXTURES") {
        Ok(p) => p,
        Err(_) => {
            println!("  FAIL (PHIFLOW_SOMA_FIXTURES not set)");
            report.add_test("phase3_soma_fixtures_available", 0.0, false);
            report.add_note("Phase 3 failed: SOMA fixtures not available — skip is not a pass");
            return false;
        }
    };

    let mut all_pass = true;

    // Load wakeful fixture
    let wake_trace = match load_fixture(&fixture_path, "wakeful") {
        Ok(t) => t,
        Err(e) => {
            println!("  FAIL (wakeful fixture: {})", e);
            report.add_test("phase3_wakeful_fixture_loaded", 0.0, false);
            report.add_note(&format!("Phase 3 failed: wakeful fixture load error: {}", e));
            return false;
        }
    };
    report.add_test("phase3_wakeful_fixture_loaded", 1.0, true);

    // Load deep_sleep fixture
    let sleep_trace = match load_fixture(&fixture_path, "deep_sleep") {
        Ok(t) => t,
        Err(e) => {
            println!("  FAIL (deep_sleep fixture: {})", e);
            report.add_test("phase3_deep_sleep_fixture_loaded", 0.0, false);
            report.add_note(&format!("Phase 3 failed: deep_sleep fixture load error: {}", e));
            return false;
        }
    };
    report.add_test("phase3_deep_sleep_fixture_loaded", 1.0, true);

    // Compute metrics for both states
    let wake_metrics = ConsciousnessMetrics::compute(&wake_trace, 100, 10, 0.01);
    let sleep_metrics = ConsciousnessMetrics::compute(&sleep_trace, 100, 10, 0.01);

    println!(
        "  Wakeful:  L_self={:.4} D_int={:.4} C_coh={:.4} C_PF={:.4}",
        wake_metrics.l_self, wake_metrics.d_int, wake_metrics.c_coh, wake_metrics.c_pf
    );
    println!(
        "  Sleep:    L_self={:.4} D_int={:.4} C_coh={:.4} C_PF={:.4}",
        sleep_metrics.l_self, sleep_metrics.d_int, sleep_metrics.c_coh, sleep_metrics.c_pf
    );

    // Individual state thresholds
    let wake_l_self_pass = wake_metrics.l_self > 0.3;
    report.add_test("phase3_wake_l_self_gt_0.3", wake_metrics.l_self, wake_l_self_pass);
    all_pass = all_pass && wake_l_self_pass;

    let wake_cpf_pass = wake_metrics.c_pf > 0.1;
    report.add_test("phase3_wake_cpf_gt_0.1", wake_metrics.c_pf, wake_cpf_pass);
    all_pass = all_pass && wake_cpf_pass;

    let sleep_l_self_pass = sleep_metrics.l_self < 0.2;
    report.add_test("phase3_sleep_l_self_lt_0.2", sleep_metrics.l_self, sleep_l_self_pass);
    all_pass = all_pass && sleep_l_self_pass;

    let sleep_cpf_pass = sleep_metrics.c_pf < 0.05;
    report.add_test("phase3_sleep_cpf_lt_0.05", sleep_metrics.c_pf, sleep_cpf_pass);
    all_pass = all_pass && sleep_cpf_pass;

    // Discrimination: wake C_PF should be > 2x sleep C_PF
    let discrimination_ratio = if sleep_metrics.c_pf > 1e-9 {
        wake_metrics.c_pf / sleep_metrics.c_pf
    } else {
        f64::INFINITY
    };
    let discrim_pass = discrimination_ratio > 2.0;
    report.add_test("phase3_discrimination_wake_gt_2x_sleep", discrimination_ratio, discrim_pass);
    all_pass = all_pass && discrim_pass;

    println!(
        "  Discrimination ratio (wake/sleep C_PF): {:.2} {}",
        discrimination_ratio,
        if discrim_pass { "PASS" } else { "FAIL" }
    );

    all_pass
}

/// Load a fixture from `<base_path>/<name>.json` and convert to a Trace.
///
/// Expected JSON format:
/// ```json
/// { "observed": [...], "coherence": [...], "depth": [...],
///   "model": [...], "action": [...] }
/// ```
///
/// If `model` or `action` arrays are missing or mismatched in length, they
/// are derived dynamically from `observed` using a running-mean model.
fn load_fixture(base_path: &str, name: &str) -> Result<Trace, String> {
    let path = std::path::Path::new(base_path).join(format!("{}.json", name));
    if !path.exists() {
        return Err(format!("fixture not found: {}", path.display()));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("read error: {}", e))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("parse error: {}", e))?;

    let mut trace = Trace::new();

    let mut observed: Vec<f64> = Vec::new();
    if let Some(arr) = json["observed"].as_array() {
        for (i, val) in arr.iter().enumerate() {
            if let Some(v) = val.as_f64() {
                trace.observed.push(v, i as f64);
                observed.push(v);
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

    // Load or derive model and action arrays
    let mut models: Vec<f64> = Vec::new();
    if let Some(arr) = json["model"].as_array() {
        for val in arr.iter() {
            if let Some(v) = val.as_f64() {
                models.push(v);
            }
        }
    }

    let mut actions: Vec<f64> = Vec::new();
    if let Some(arr) = json["action"].as_array() {
        for val in arr.iter() {
            if let Some(v) = val.as_f64() {
                actions.push(v);
            }
        }
    }

    // Derive model/action if missing or mismatched
    if models.len() != observed.len() || actions.len() != observed.len() {
        models.clear();
        actions.clear();
        let mut model_sum = 0.55;
        let mut model_n = 1.0;
        for &obs in &observed {
            let model_mean = model_sum / model_n;
            let action = if obs < model_mean { 1.0 } else { 0.0 };
            models.push(model_mean);
            actions.push(action);
            model_sum += obs;
            model_n += 1.0;
        }
    }

    // Populate raw_events in type4 format: (step, obs, model, action) per cycle
    for i in 0..observed.len() {
        trace.raw_events.push(("step".to_string(), i as f64));
        trace.raw_events.push(("obs".to_string(), observed[i]));
        trace.raw_events.push(("model".to_string(), models[i]));
        trace.raw_events.push(("action".to_string(), actions[i]));
    }

    Ok(trace)
}

/// Phase 4: Daemon trace from type4_trace_benchmark.phi
fn phase4_daemon_trace(report: &mut BenchmarkReport) -> bool {
    // Execute the actual type4_trace_benchmark.phi program
    use phiflow::metrics::trace::Trace;
    use phiflow::parser::parse_phi_program;
    use phiflow::phi_ir::evaluator::Evaluator;
    use phiflow::phi_ir::lowering::lower_program_checked;

    let phi_path = std::path::Path::new("examples/type4_trace_benchmark.phi");
    let source = match std::fs::read_to_string(phi_path) {
        Ok(s) => s,
        Err(_) => {
            println!("  ❌ Phase 4: FAILED (type4_trace_benchmark.phi not found)");
            report.add_test("daemon_type4_trace_file_available", 0.0, false);
            report.add_note("Phase 4 failed: benchmark file not found — skip is not a pass");
            return false;
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
    println!(
        "    Verdict: {}",
        if sc.l_self > 0.1 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

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
            writeln!(
                file,
                "✅ **PASSED** - synthetic proxy smoke test only; Type 4 confirmation remains HOLD"
            )?;
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
        trace
            .coherence
            .push(0.5 + 0.1 * (i as f64 * 0.05).sin(), i as f64);
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
