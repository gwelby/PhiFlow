use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    emitter,
    evaluator::Evaluator,
    lowering::lower_program,
    optimizer::{OptimizationLevel, Optimizer},
    vm::PhiVm,
    wasm::emit_wat,
    PhiIRProgram, PhiIRValue,
};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn assert_values_close(lhs: &PhiIRValue, rhs: &PhiIRValue, context: &str) {
    match (lhs, rhs) {
        (PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
            assert!(
                (a - b).abs() < 1e-9,
                "{}: numeric mismatch (lhs={}, rhs={})",
                context,
                a,
                b
            );
        }
        _ => assert_eq!(lhs, rhs, "{}: value mismatch", context),
    }
}

fn assert_number_close(lhs: f64, rhs: f64, context: &str) {
    assert!(
        (lhs - rhs).abs() < 1e-9,
        "{}: numeric mismatch (lhs={}, rhs={})",
        context,
        lhs,
        rhs
    );
}

fn write_temp_file(stem: &str, ext: &str, contents: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "phiflow_{}_{}_{}_{}.{}",
        stem,
        std::process::id(),
        std::thread::current().name().unwrap_or("thread"),
        now,
        ext
    ));
    fs::write(&path, contents).expect("failed to write temp file");
    path
}

fn run_wat_with_node(wat: &str) -> f64 {
    let wat_path = write_temp_file("conformance", "wat", wat);
    let runner_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("phi_ir_wasm_runner.js");

    let output = Command::new("node")
        .arg(&runner_path)
        .arg(&wat_path)
        .output()
        .expect("failed to run node; ensure Node.js is installed");

    let _ = fs::remove_file(&wat_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("node wasm runner failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim()
        .parse::<f64>()
        .expect("node runner did not print a numeric result")
}

/// Run evaluator + WASM only (skip legacy PhiVm).
/// Use for programs that contain stream blocks: the legacy PhiVm predates stream
/// opcodes and `InvalidOpcode` is expected on those paths.
/// The PhiIR evaluator is the canonical reference; WASM must agree with it.
fn run_eval_and_wasm(source: &str) -> (PhiIRValue, f64) {
    let expressions = parse_phi_program(source).expect("parse failed");
    let mut program: PhiIRProgram = lower_program(&expressions);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut evaluator = Evaluator::new(&program);
    let eval_result = evaluator.run().expect("evaluator failed");

    let wat = emit_wat(&program);
    let wasm_result = run_wat_with_node(&wat);

    (eval_result, wasm_result)
}

fn assert_program_conforms_eval_wasm(source: &str, label: &str) {
    let (eval_result, wasm_result) = run_eval_and_wasm(source);
    let eval_number = match eval_result {
        PhiIRValue::Number(n) => n,
        other => panic!("{} expected numeric result, got {:?}", label, other),
    };
    assert_number_close(eval_number, wasm_result, &format!("{} evaluator-wasm", label));
}

fn run_all_paths(source: &str) -> (PhiIRValue, PhiIRValue, f64) {
    let expressions = parse_phi_program(source).expect("parse failed");
    let mut program: PhiIRProgram = lower_program(&expressions);

    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut evaluator = Evaluator::new(&program);
    let eval_result = evaluator.run().expect("evaluator failed");

    let bytes = emitter::emit(&program);
    let vm_result = PhiVm::run_bytes(&bytes).expect("vm failed");

    let wat = emit_wat(&program);
    let wasm_result = run_wat_with_node(&wat);

    (eval_result, vm_result, wasm_result)
}

fn assert_program_matches(source: &str, expected: PhiIRValue, label: &str) {
    let (eval_result, vm_result, wasm_result) = run_all_paths(source);

    assert_values_close(&eval_result, &vm_result, &format!("{} evaluator-vm", label));
    assert_values_close(
        &eval_result,
        &expected,
        &format!("{} evaluator-expected", label),
    );

    let eval_number = match eval_result {
        PhiIRValue::Number(n) => n,
        other => panic!("{} expected numeric result, got {:?}", label, other),
    };
    assert_number_close(eval_number, wasm_result, &format!("{} evaluator-wasm", label));
}

fn assert_program_conforms(source: &str, label: &str) {
    let (eval_result, vm_result, wasm_result) = run_all_paths(source);

    assert_values_close(
        &eval_result,
        &vm_result,
        &format!("{} evaluator-vm", label),
    );

    let eval_number = match eval_result {
        PhiIRValue::Number(n) => n,
        other => panic!("{} expected numeric result, got {:?}", label, other),
    };
    assert_number_close(eval_number, wasm_result, &format!("{} evaluator-wasm", label));
}

#[test]
fn conformance_arithmetic_42() {
    assert_program_matches(
        r#"
        let x = 6 * 7
        x
        "#,
        PhiIRValue::Number(42.0),
        "arithmetic_42",
    );
}

#[test]
fn conformance_chained_84() {
    assert_program_matches(
        r#"
        let x = 10 + 32
        let y = x * 2
        y
        "#,
        PhiIRValue::Number(84.0),
        "chained_84",
    );
}

#[test]
fn conformance_witness() {
    assert_program_matches("witness", PhiIRValue::Number(0.0), "witness");
}

#[test]
fn conformance_intention_scope_return() {
    assert_program_matches(
        r#"
        intention "Healing" {
            42
        }
        "#,
        PhiIRValue::Number(42.0),
        "intention_scope_return",
    );
}

#[test]
fn conformance_coherence_check() {
    assert_program_matches(
        "coherence",
        PhiIRValue::Number(0.0),
        "coherence_check",
    );
}

#[test]
fn conformance_resonate_then_coherence() {
    let phi_inv = 1.0 - 1.618033988749895_f64.powi(-1);
    assert_program_matches(
        r#"
        intention "Channel" {
            resonate 10
            coherence
        }
        "#,
        PhiIRValue::Number(phi_inv + 0.05),
        "resonate_then_coherence",
    );
}

/// Verify that shared example fixtures compile and run on both evaluator and WASM
/// without error. Programs that primarily emit via the resonance field (e.g. claude.phi,
/// stream programs) don't have a meaningful return-value to compare, so this test
/// checks parse-lower-optimize-run success rather than value equality.
fn assert_program_runs_on_eval_and_wasm(source: &str, label: &str) {
    let expressions = parse_phi_program(source)
        .unwrap_or_else(|e| panic!("{}: parse failed: {:?}", label, e));
    let mut program: PhiIRProgram = lower_program(&expressions);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    // Evaluator: must run without error
    let mut evaluator = Evaluator::new(&program);
    evaluator
        .run()
        .unwrap_or_else(|e| panic!("{}: evaluator failed: {:?}", label, e));

    // WASM: must emit valid WAT that Node.js can instantiate and run
    let wat = emit_wat(&program);
    run_wat_with_node(&wat); // panics with node error message on invalid WAT
}

/// Shared fixture conformance: all team example programs must compile and run
/// on both the canonical evaluator and the WASM backend without error.
///
/// The legacy PhiVm is not tested here: example programs use user-defined
/// functions and stream blocks which predate the legacy VM's feature set.
#[test]
fn conformance_shared_fixture_examples() {
    let fixtures = [
        ("examples/claude.phi", include_str!("../examples/claude.phi")),
        (
            "examples/stream_demo.phi",
            include_str!("../examples/stream_demo.phi"),
        ),
        (
            "examples/adaptive_witness.phi",
            include_str!("../examples/adaptive_witness.phi"),
        ),
    ];
    for (label, source) in fixtures {
        assert_program_runs_on_eval_and_wasm(source, label);
    }
}

/// Regression: Phase 10 Lane C — function return values inside intention blocks
/// must agree between evaluator and WASM.
///
/// Pre-fix: evaluator returned phi^(-2) = 0.38197 (one loop iteration, not two).
/// Post-fix: returns 0.618... (correct — VM while-loop comparison fixed).
///
/// Only evaluator + WASM: the legacy PhiVm does not support user-defined functions.
#[test]
fn conformance_nested_function_regression() {
    let phi: f64 = 1.618033988749895;
    let expected = 1.0 - phi.powi(-2); // lambda = 0.6180339887498949

    let source = r#"
        function compute_lambda(phi_val: Number) -> Number {
            let result = 1.0
            let i = 0.0
            while i < 2.0 {
                result = result * phi_val
                i = i + 1.0
            }
            return 1.0 - (1.0 / result)
        }

        intention "nested_call_regression" {
            let phi_val = 1.618033988749895
            compute_lambda(phi_val)
        }
        "#;

    let (eval_result, wasm_result) = run_eval_and_wasm(source);

    let eval_n = match eval_result {
        PhiIRValue::Number(n) => n,
        other => panic!("nested_function_regression: evaluator returned {:?}", other),
    };
    assert_number_close(eval_n, expected, "nested_function_regression evaluator-expected");
    assert_number_close(eval_n, wasm_result, "nested_function_regression evaluator-wasm");
}

#[test]
fn test_wasm_claude_formula_returns_618() {
    // Phase 10 Lane C fail-first: WASM backend must agree with evaluator path.
    let source = include_str!("../examples/claude.phi");
    let expressions = parse_phi_program(source).expect("parse failed");
    let mut program = lower_program(&expressions);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut evaluator = Evaluator::new(&program);
    evaluator.run().expect("evaluator failed");

    let eval_number = match evaluator.resonated_values("LAMBDA_convergence").last() {
        Some(PhiIRValue::Number(n)) => *n,
        other => panic!(
            "expected resonated numeric lambda in evaluator path, got {:?}",
            other
        ),
    };

    let wat = emit_wat(&program);
    let wasm_result = run_wat_with_node(&wat);

    assert!(
        (eval_number - 0.618).abs() < 0.001,
        "evaluator path expected ~0.618, got {}",
        eval_number
    );
    assert!(
        (wasm_result - 0.618).abs() < 0.001,
        "WASM path returned {} not 0.618",
        wasm_result
    );
}
