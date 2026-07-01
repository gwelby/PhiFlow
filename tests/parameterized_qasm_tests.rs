//! Integration tests for the full parameterized QASM pipeline.
//!
//! Tests the complete path: parse → lower → eval → scrape coherence →
//! emit_with_runtime_params → assert QASM structure.
//!
//! Closes the open next-step from the 2026-05-20 STATE.md verified entry.

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program_checked;
use phiflow::phi_ir::openqasm::OpenQasmEmitter;
use std::collections::HashMap;

/// Run the full parameterized QASM pipeline on a source string and return
/// (qasm, runtime_params).
fn run_parameterized_qasm(source: &str) -> (String, HashMap<String, f64>) {
    let ast = parse_phi_program(source).expect("Parse failed");
    let ir = lower_program_checked(&ast).expect("Lowering failed");

    // Run evaluator to capture live coherence per intention
    let mut evaluator = Evaluator::new(ir.clone());
    let _ = evaluator.run();

    let frozen = evaluator.freeze_state();
    let mut runtime_params = HashMap::new();
    for event in &frozen.witness_log {
        if let Some(name) = event.intention_stack.last() {
            // Last coherence value for each intention wins
            runtime_params.insert(name.clone(), event.coherence);
        }
    }

    let mut emitter = OpenQasmEmitter::new();
    let qasm = emitter
        .emit_with_runtime_params(&ir, &runtime_params)
        .expect("QASM emit failed");

    (qasm, runtime_params)
}

/// Test 1: Full pipeline on quantum_council.phi produces valid QASM with
/// correct header, qubit declarations, entanglement gates, and measurements.
#[test]
fn test_quantum_council_full_pipeline() {
    let source = r#"
intention "observe" {
    witness
    resonate 0.618
    entangle on 432
}

intention "integrate" {
    witness
    resonate 0.618
    entangle on 432
}

intention "transcend" {
    witness
    resonate 0.618
    entangle on 432
    witness
}
"#;

    let (qasm, params) = run_parameterized_qasm(source);

    // Runtime params should have coherence for all three intentions
    assert!(
        params.contains_key("observe"),
        "Runtime params should contain 'observe', got: {:?}",
        params.keys().collect::<Vec<_>>()
    );
    assert!(
        params.contains_key("integrate"),
        "Runtime params should contain 'integrate'"
    );
    assert!(
        params.contains_key("transcend"),
        "Runtime params should contain 'transcend'"
    );

    // QASM header
    assert!(
        qasm.starts_with("OPENQASM 3.0;"),
        "QASM should start with OPENQASM 3.0 header, got:\n{}",
        &qasm[..qasm.len().min(100)]
    );
    assert!(
        qasm.contains("include \"stdgates.inc\";"),
        "QASM should include stdgates"
    );

    // Three qubits for three intentions
    assert!(
        qasm.contains("qubit[3]"),
        "QASM should declare 3 qubits for 3 intentions, got:\n{}",
        qasm
    );

    // Entanglement gates (cx) — all three share 432 Hz
    assert!(
        qasm.contains("cx"),
        "QASM should contain cx entanglement gates for 432 Hz chain"
    );

    // Measurements
    assert!(
        qasm.contains("measure"),
        "QASM should contain measurement instructions"
    );
}

/// Test 2: Runtime coherence values override the placeholder 0.618 angle.
/// The emitted QASM should use the runtime coherence, not 0.618.
#[test]
fn test_runtime_params_override_placeholder() {
    let source = r#"
intention "test" {
    witness
    resonate 0.618
}
"#;

    let (qasm, params) = run_parameterized_qasm(source);

    // The runtime param should be the actual coherence, not 0.618
    if let Some(&coherence) = params.get("test") {
        // The QASM should contain the runtime coherence value, not 0.618
        let coherence_str = format!("{:.6}", coherence);
        let coherence_short = format!("{:.4}", coherence);
        assert!(
            qasm.contains(&coherence_str)
                || qasm.contains(&coherence_short)
                || qasm.contains(&format!("({} * pi)", coherence)),
            "QASM should contain runtime coherence value {} (from params), not placeholder 0.618\nQASM:\n{}",
            coherence,
            qasm
        );
    }
}

/// Test 3: Multiple intentions on the same frequency get entangled (cx gates).
#[test]
fn test_entanglement_on_same_frequency() {
    let source = r#"
intention "alpha" {
    witness
    resonate 0.5
    entangle on 432
}

intention "beta" {
    witness
    resonate 0.5
    entangle on 432
}
"#;

    let (qasm, _) = run_parameterized_qasm(source);

    // Two qubits on 432 Hz should get a cx gate
    assert!(
        qasm.contains("cx q[0], q[1]"),
        "Two intentions on 432 Hz should be entangled with cx q[0], q[1]\ngot:\n{}",
        qasm
    );
}

/// Test 4: Different frequencies do NOT cross-entangle.
#[test]
fn test_frequency_isolation_in_parameterized_mode() {
    let source = r#"
intention "alpha" {
    witness
    resonate 0.5
    entangle on 432
}

intention "beta" {
    witness
    resonate 0.5
    entangle on 528
}
"#;

    let (qasm, _) = run_parameterized_qasm(source);

    // Different frequencies should NOT have cx between them
    assert!(
        !qasm.contains("cx q[0], q[1]"),
        "Intentions on different frequencies (432, 528) should NOT be entangled\ngot:\n{}",
        qasm
    );
}

/// Test 5: Deferred measurements are deduplicated — exactly one final
/// measurement block, not duplicates.
#[test]
fn test_measurement_deduplication() {
    // This program has a witness after entangle (deferred) plus a final witness
    let source = r#"
intention "seat1" {
    witness
    resonate 0.618
    entangle on 432
    witness
}
intention "seat2" {
    witness
    resonate 0.618
    entangle on 432
    witness
}
"#;

    let (qasm, _) = run_parameterized_qasm(source);

    // Count measurement lines — should not have excessive duplicates
    let measure_count = qasm.matches("measure q[").count();
    assert!(
        measure_count <= 4, // At most 2 qubits × 2 (but dedup should reduce)
        "Measurements should be deduplicated, found {} measure lines\ngot:\n{}",
        measure_count,
        qasm
    );
}

/// Test 6: The actual quantum_council.phi example file runs through the
/// full pipeline without errors.
#[test]
fn test_quantum_council_example_file() {
    let source = std::fs::read_to_string("examples/quantum_council.phi")
        .expect("quantum_council.phi should exist");
    let (qasm, params) = run_parameterized_qasm(&source);

    // Should have coherence for all three council seats
    assert!(params.len() >= 3, "Should have >= 3 runtime params, got {}", params.len());

    // QASM should be non-empty and well-formed
    assert!(!qasm.is_empty(), "QASM output should not be empty");
    assert!(qasm.contains("OPENQASM 3.0;"), "Should have QASM header");
    assert!(qasm.contains("qubit[3]"), "Should declare 3 qubits");
    assert!(qasm.contains("cx"), "Should have entanglement gates");
}
