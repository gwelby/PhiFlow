use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::openqasm::OpenQasmEmitter;

#[test]
fn test_team_direction_full_pipeline() {
    let source = "
intention \"Tesla\" {
    resonate 0.72 toward TEAM_A
}
intention \"Einstein\" {
    resonate 0.72 toward TEAM_B
}
witness
";
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);
    let mut emitter = OpenQasmEmitter::new();
    let qasm = emitter.emit(&program).expect("emit failed");

    assert!(qasm.contains("ry(0.72 * pi) q[0]"), "Expected ry(0.72 * pi) for TEAM_A, but got:\n{}", qasm);
    assert!(qasm.contains("ry(0.28 * pi) q[1]"), "Expected ry(0.28 * pi) for TEAM_B, but got:\n{}", qasm);
}

#[test]
fn test_mid_circuit_ordering() {
    let source = "
intention \"Healing\" {
    witness
    resonate 0.5
}
";
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);
    let mut emitter = OpenQasmEmitter::new();
    let qasm = emitter.emit(&program).expect("emit failed");

    let measure_idx = qasm.find("measure").expect("measure not found");
    let resonate_idx = qasm.find("ry(0.5 * pi)").expect("resonate not found");

    assert!(measure_idx < resonate_idx, "Witness (measure) must appear BEFORE resonate gate in mid-circuit context");
}

#[test]
fn test_frequency_channel_isolation() {
    let source = "
intention \"I0\" {
    entangle on 432
}
intention \"I1\" {
    entangle on 432
}
intention \"I2\" {
    entangle on 528
}
intention \"I3\" {
    entangle on 528
}
";
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);
    let mut emitter = OpenQasmEmitter::new();
    let qasm = emitter.emit(&program).expect("emit failed");

    // 432Hz chain: I0 (q[0]) → I1 (q[1])
    assert!(qasm.contains("cx q[0], q[1]; // Entangle via 432Hz"));
    
    // 528Hz chain: I2 (q[2]) is first member (no CNOT), I3 (q[3]) entangles to I2
    assert!(qasm.contains("cx q[2], q[3]; // Entangle via 528Hz"));
    
    // Verify no cross-frequency contamination
    assert!(!qasm.contains("cx q[1], q[2]"), "Incorrect cross-frequency entanglement detected:\n{}", qasm);
    assert!(!qasm.contains("cx q[0], q[2]"), "528Hz chain should not entangle to 432Hz qubits:\n{}", qasm);
}

#[test]
fn test_optimize_depth_cli_flag() {
    let source = "
intention \"I0\" { entangle on 432 }
intention \"I1\" { entangle on 432 }
intention \"I2\" { entangle on 432 }
intention \"I3\" { entangle on 432 }
";
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);

    // Linear (default)
    let mut emitter_linear = OpenQasmEmitter::new();
    let qasm_linear = emitter_linear.emit(&program).expect("emit failed");

    // Tree (optimized)
    let mut emitter_tree = OpenQasmEmitter::new();
    emitter_tree.optimize_depth = true;
    let qasm_tree = emitter_tree.emit(&program).expect("emit failed");

    // Linear: I0 → I1 → I2 → I3 (chain)
    assert!(qasm_linear.contains("cx q[0], q[1]"));
    assert!(qasm_linear.contains("cx q[1], q[2]"));
    assert!(qasm_linear.contains("cx q[2], q[3]"));

    // Tree: I0 → I1, I0 → I2, I1 → I3 (balanced)
    assert!(qasm_tree.contains("cx q[0], q[1]"));
    assert!(qasm_tree.contains("cx q[0], q[2]"));
    assert!(qasm_tree.contains("cx q[1], q[3]"));
}

#[test]
fn test_deferred_witness_no_collision() {
    let source = "
intention \"A\" { resonate 0.1 }
intention \"B\" { resonate 0.2 }
witness
witness
";
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);
    let mut emitter = OpenQasmEmitter::new();
    let qasm = emitter.emit(&program).expect("emit failed");

    // Count occurrences of measure
    let measure_count = qasm.matches("measure").count();
    // 2 qubits * 2 witness calls = 4 measurements if mid-circuit (though the emitter currently measures all qubits for each witness call)
    assert!(measure_count >= 4, "Expected at least 4 measurement operations, but found: {}", measure_count);
}

#[test]
fn test_coherence_feedback_loop_evolve() {
    use phiflow::compile_and_run_phi_ir;
    use phiflow::phi_ir::PhiIRValue;

    // A program that checks its own coherence and evolves if it's low.
    // In the evaluator, coherence starts at 0 and grows with intention depth.
    // At depth 1 (Tesla), coherence is 1 - 1.618^-1 = 0.382
    // At depth 2 (nested), coherence is 1 - 1.618^-2 = 0.618
    let source = "
let result = 0
intention \"Tesla\" {
    if coherence < 0.5 {
        evolve \"result = 1\"
    } else {
        evolve \"result = 2\"
    }
}
result
";
    let res = compile_and_run_phi_ir(source).expect("execution failed");
    
    // At depth 1, coherence (0.382) < 0.5, so result should be 1
    match res {
        PhiIRValue::Number(n) => assert_eq!(n, 1.0, "Expected evolution to set result to 1.0 (low coherence path)"),
        _ => panic! ("Expected number result"),
    }

    let source_high = "
let result = 0
intention \"Tesla\" {
    intention \"Nested\" {
        if coherence > 0.6 {
            evolve \"result = 2\"
        } else {
            evolve \"result = 1\"
        }
    }
}
result
";
    let res_high = compile_and_run_phi_ir(source_high).expect("execution failed");
    
    // At depth 2, coherence (0.618) > 0.6, so result should be 2
    match res_high {
        PhiIRValue::Number(n) => assert_eq!(n, 2.0, "Expected evolution to set result to 2.0 (high coherence path)"),
        _ => panic! ("Expected number result"),
    }
}
