use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::openqasm::OpenQasmEmitter;

fn compile_to_qasm(source: &str) -> String {
    let program = parse_phi_program(source).expect("Parse failed");
    let ir = lower_program(&program);
    let mut emitter = OpenQasmEmitter::new();
    emitter.emit(&ir).expect("OpenQASM emit failed")
}

fn compile_to_qasm_optimized(source: &str) -> String {
    let program = parse_phi_program(source).expect("Parse failed");
    let ir = lower_program(&program);
    let mut emitter = OpenQasmEmitter::new();
    emitter.optimize_depth = true;
    emitter.emit(&ir).expect("OpenQASM emit failed")
}

/// Test 1: TEAM_B direction emits the complement angle (1 - θ).
#[test]
fn test_team_direction_openqasm() {
    let source = r#"
        intention "Tension" {
            resonate 0.72 toward TEAM_B
        }
    "#;

    let qasm = compile_to_qasm(source);
    // TEAM_B direction inverts: 1.0 - 0.72 = 0.28, emitted as ry(pi - (0.72 * pi))
    assert!(
        qasm.contains("pi - (0.72 * pi)"),
        "TEAM_B should invert angle: expected 'pi - (0.72 * pi)'\ngot:\n{qasm}"
    );
}

/// Test 2: mid_circuit witness measures inline (before later gates).
#[test]
fn test_mid_circuit_ordering() {
    let source = r#"
        intention "MeasureFirst" {
            witness mid_circuit
            resonate 0.5
        }
    "#;

    let qasm = compile_to_qasm(source);
    let measure_idx = qasm
        .find("measure q[0]")
        .expect("Should contain 'measure q[0]'");
    let resonate_idx = qasm
        .find("ry(0.5 * pi)")
        .expect("Should contain 'ry(0.5 * pi)'");
    assert!(
        measure_idx < resonate_idx,
        "mid_circuit witness must measure before the later resonate gate\ngot:\n{qasm}"
    );
}

/// Test 3: Entangle on different frequencies does NOT cross-entangle.
#[test]
fn test_frequency_channel_isolation() {
    let source = r#"
        intention "A" {
            entangle on 432
        }
        intention "B" {
            entangle on 528
        }
        intention "C" {
            entangle on 432
        }
    "#;

    let qasm = compile_to_qasm(source);
    // A=q[0], B=q[1], C=q[2]; 432 chain: q[0]↔q[2], 528 chain: just q[1] alone
    assert!(
        qasm.contains("cx q[0], q[2]"),
        "Qubits A and C share freq 432 — should be CX'd\ngot:\n{qasm}"
    );
    assert!(
        !qasm.contains("cx q[0], q[1]"),
        "A and B are on different frequencies — must NOT be CX'd\ngot:\n{qasm}"
    );
    assert!(
        !qasm.contains("cx q[1], q[2]"),
        "B and C are on different frequencies — must NOT be CX'd\ngot:\n{qasm}"
    );
}

/// Test 4: A top-level `witness` (deferred) does not add a measure before the resonate gate.
#[test]
fn test_deferred_witness_no_collision() {
    let source = r#"
        intention "GlobalMeasure" {
            resonate 0.1
            witness
        }
    "#;

    let qasm = compile_to_qasm(source);
    // resonate should be emitted before the deferred measure
    let resonate_idx = qasm
        .find("ry(0.1 * pi)")
        .expect("resonate gate should be present");
    let measure_idx = qasm
        .find("measure q[0]")
        .expect("final measure should be present");
    assert!(
        resonate_idx < measure_idx,
        "resonate gate must come before the deferred measure\ngot:\n{qasm}"
    );
}

/// Test 5: --optimize-depth produces different (shallower) CNOT ordering than linear.
#[test]
fn test_optimize_depth_flag() {
    let source = r#"
        intention "A" { entangle on 432 }
        intention "B" { entangle on 432 }
        intention "C" { entangle on 432 }
        intention "D" { entangle on 432 }
    "#;

    let linear = compile_to_qasm(source);
    let tree = compile_to_qasm_optimized(source);

    // The two topologies must differ in their CX arrangement.
    assert_ne!(
        linear, tree,
        "optimized (tree) topology should differ from linear"
    );
}

/// Test 6: Verify warning comment for post-collapsed gating.
#[test]
fn test_post_collapse_warning() {
    let source = r#"
        intention "Wait" {
            witness mid_circuit
            resonate 0.5
            coherence
        }
    "#;

    let qasm = compile_to_qasm(source);
    
    assert!(
        qasm.contains("// WARNING: Gating post-collapsed qubit q[0]"),
        "Should contain warning for resonate after mid-circuit measurement\ngot:\n{qasm}"
    );
    assert!(
        qasm.contains("// WARNING: Coherence post-collapsed qubit q[0]"),
        "Should contain warning for coherence after mid-circuit measurement\ngot:\n{qasm}"
    );
}

