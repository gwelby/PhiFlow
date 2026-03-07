use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::quantum_codegen::compile_ir_to_quantum;
use phiflow::quantum::QuantumGate;

#[test]
fn test_quantum_codegen_resonate_to_phi_harmonic() {
    let source = r#"
        let x = 0.618
        resonate x
    "#;

    let program = parse_phi_program(source).expect("Failed to parse program");
    let ir = lower_program(&program);
    let circuit = compile_ir_to_quantum(&ir);

    assert!(
        circuit.qubits >= 1,
        "Expected at least 1 qubit in the circuit"
    );

    let mut found = false;
    for gate in &circuit.gates {
        if let QuantumGate::PhiHarmonic(_, _) = gate {
            found = true;
            break;
        }
    }

    assert!(
        found,
        "Expected a PhiHarmonic gate to be generated from resonate"
    );
}

#[test]
fn test_quantum_codegen_coherence_to_sacred_frequency() {
    let source = r#"
        let c = coherence
    "#;

    let program = parse_phi_program(source).expect("Failed to parse program");
    let ir = lower_program(&program);
    let circuit = compile_ir_to_quantum(&ir);

    let mut found = false;
    for gate in &circuit.gates {
        if let QuantumGate::SacredFrequency(_, _) = gate {
            found = true;
            break;
        }
    }

    assert!(
        found,
        "Expected a SacredFrequency gate to be generated from coherence"
    );
}

#[test]
fn test_quantum_codegen_intention_scope() {
    let source = r#"
        intention "quantum_healing" {
            resonate 432.0
        }
    "#;

    let program = parse_phi_program(source).expect("Failed to parse program");
    let ir = lower_program(&program);
    let circuit = compile_ir_to_quantum(&ir);

    // An intention block might wrap the operations in a specific quantum context or metadata
    assert!(
        circuit.metadata.contains_key("intentions"),
        "Expected intention metadata to be recorded in the circuit"
    );
}
