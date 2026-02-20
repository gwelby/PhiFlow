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
