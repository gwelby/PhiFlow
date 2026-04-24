use phiflow::compile_to_openqasm;
use phiflow::compile_to_openqasm_with_options;
use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::quantum_interaction::{
    analyze_quantum_overlay, QuantumOverlayError, QuantumOverlayPlan,
};
use phiflow::phi_ir::topology_transpiler::{
    choose_ladder_corridor, RoutingStrategy, TopologyTranspileConfig,
};
use phiflow::quantum::backend_topology::{
    normalize_edge, BackendTopologyProfile, EdgeCalibration, NativeTwoQGate, ProcessorFamily,
    QubitCalibration,
};
use phiflow::OpenQasmCompileOptions;
use std::collections::{HashMap, HashSet};

fn ladder_profile(depth: usize) -> BackendTopologyProfile {
    let mut coupling_map = Vec::new();
    let mut qubits = HashMap::new();
    let mut edges = HashMap::new();

    for layer in 0..depth {
        let left = layer * 2;
        let right = left + 1;
        coupling_map.push((left, right));
        edges.insert(normalize_edge(left, right), EdgeCalibration::default());
        qubits.insert(left, QubitCalibration::default());
        qubits.insert(right, QubitCalibration::default());

        if layer + 1 < depth {
            let next_left = (layer + 1) * 2;
            let next_right = next_left + 1;
            coupling_map.push((left, next_left));
            coupling_map.push((right, next_right));
            edges.insert(normalize_edge(left, next_left), EdgeCalibration::default());
            edges.insert(normalize_edge(right, next_right), EdgeCalibration::default());
        }
    }

    BackendTopologyProfile {
        backend_name: "ibm_fez".to_string(),
        family: ProcessorFamily::Heron,
        num_qubits: depth * 2,
        coupling_map,
        native_two_qubit_gate: NativeTwoQGate::Cz,
        qubits,
        edges,
    }
}

fn topology_options(profile: BackendTopologyProfile) -> OpenQasmCompileOptions {
    let native_two_qubit_gate = profile.native_two_qubit_gate;
    OpenQasmCompileOptions {
        optimize_depth: false,
        topology: Some(TopologyTranspileConfig {
            backend_name: profile.backend_name.clone(),
            strategy: RoutingStrategy::CalibrationWeightedShortestPath,
            native_two_qubit_gate,
        }),
        live_backend_profile: Some(profile),
    }
}

#[test]
fn test_resonate_channel_alias_syntax_is_accepted() {
    let source = r#"
intention "logic" {
    resonate 1.0 as "logic/truth"
}
"#;
    parse_phi_program(source).expect("channel alias syntax should parse");
}

#[test]
fn test_quantum_overlay_detects_contradiction_ladder() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let program = lower_program(&parse_phi_program(source).expect("parse failed"));

    let overlay = analyze_quantum_overlay(&program).expect("overlay analysis failed");
    match overlay {
        QuantumOverlayPlan::ContradictionLadder(plan) => {
            assert_eq!(plan.depth, 20);
            assert_eq!(plan.left_lane.len(), 20);
            assert_eq!(plan.right_lane.len(), 20);
            assert!(plan.witness_target.is_some());
        }
        other => panic!("expected contradiction ladder, got {other:?}"),
    }
}

#[test]
fn test_quantum_overlay_rejects_non_quantumizable_coherence_call() {
    let source = r#"
stream "conflict" {
    let l1 = coherence(1.0, sensor("cpu_usage"))
    let f1 = coherence(0.0, 1.0)
    let final_state = coherence(l1, f1)
    witness final_state
}
"#;
    let program = lower_program(&parse_phi_program(source).expect("parse failed"));

    let err = analyze_quantum_overlay(&program).expect_err("overlay should reject sensor mixing");
    assert!(matches!(
        err,
        QuantumOverlayError::MixedQuantumAndNonQuantumOperand(_)
    ));
}

#[test]
fn test_choose_ladder_corridor_prefers_swap_free_adjacent_path() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let program = lower_program(&parse_phi_program(source).expect("parse failed"));
    let overlay = analyze_quantum_overlay(&program).expect("overlay analysis failed");
    let QuantumOverlayPlan::ContradictionLadder(plan) = overlay else {
        panic!("expected contradiction ladder");
    };

    let profile = ladder_profile(20);
    let config = TopologyTranspileConfig {
        backend_name: "ibm_fez".to_string(),
        strategy: RoutingStrategy::CalibrationWeightedShortestPath,
        native_two_qubit_gate: NativeTwoQGate::Cz,
    };
    let corridor = choose_ladder_corridor(&plan, &profile, &config).expect("corridor required");

    assert_eq!(corridor.left_path.len(), 20);
    assert_eq!(corridor.right_path.len(), 20);
    for (left, right) in &corridor.rung_edges {
        assert!(profile.has_edge(*left, *right));
    }
}

#[test]
fn test_choose_ladder_corridor_uses_readout_tiebreaker_when_available() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let program = lower_program(&parse_phi_program(source).expect("parse failed"));
    let overlay = analyze_quantum_overlay(&program).expect("overlay analysis failed");
    let QuantumOverlayPlan::ContradictionLadder(plan) = overlay else {
        panic!("expected contradiction ladder");
    };

    let mut profile = ladder_profile(20);
    profile
        .qubits
        .entry(38)
        .or_default()
        .readout_error = Some(0.25);
    profile
        .qubits
        .entry(39)
        .or_default()
        .readout_error = Some(0.01);

    let config = TopologyTranspileConfig {
        backend_name: "ibm_fez".to_string(),
        strategy: RoutingStrategy::CalibrationWeightedShortestPath,
        native_two_qubit_gate: NativeTwoQGate::Cz,
    };
    let corridor = choose_ladder_corridor(&plan, &profile, &config).expect("corridor required");

    assert_eq!(corridor.witness_qubit, 39);
}

#[test]
fn test_topology_aware_openqasm_emits_only_valid_cz_edges() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let profile = ladder_profile(20);
    let valid_edges = profile
        .normalized_coupling_map()
        .into_iter()
        .collect::<HashSet<_>>();
    let qasm = compile_to_openqasm_with_options(source, &topology_options(profile))
        .expect("topology-aware compilation failed");

    for line in qasm.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("cz q[") {
            let mut numbers = rest
                .split(|c: char| !c.is_ascii_digit())
                .filter(|part| !part.is_empty())
                .filter_map(|part| part.parse::<usize>().ok());
            let a = numbers.next().expect("control qubit");
            let b = numbers.next().expect("target qubit");
            assert!(
                valid_edges.contains(&normalize_edge(a, b)),
                "invalid physical edge in line: {trimmed}"
            );
        }
    }
}

#[test]
fn test_topology_aware_openqasm_never_emits_cx_for_heron() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let qasm = compile_to_openqasm_with_options(source, &topology_options(ladder_profile(20)))
        .expect("topology-aware compilation failed");

    assert!(qasm.contains("cz q["));
    assert!(!qasm.contains("cx q["));
}

#[test]
fn test_topology_aware_openqasm_comments_include_physical_mapping() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let qasm = compile_to_openqasm_with_options(source, &topology_options(ladder_profile(20)))
        .expect("topology-aware compilation failed");

    assert!(qasm.contains("Physical left rail"));
    assert!(qasm.contains("Layer 1 physical rung"));
}

#[test]
fn test_legacy_openqasm_path_unchanged_without_topology_config() {
    let source = r#"
intention "A" { entangle on 432 }
intention "B" { entangle on 432 }
"#;
    let qasm = compile_to_openqasm(source, false).expect("legacy compilation failed");
    println!("DUMPING QASM: {}", qasm);
    assert!(qasm.contains("cx q[0], q[1]"));
    assert!(!qasm.contains("cz q[0], q[1]"));
}

#[test]
fn test_cognitive_dissonance_compiles_with_live_ibm_fez_profile() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let qasm = compile_to_openqasm_with_options(source, &topology_options(ladder_profile(20)))
        .expect("topology-aware compilation failed");

    assert!(qasm.contains("Topology-aware backend: ibm_fez"));
}

#[test]
fn test_cognitive_dissonance_requires_no_swaps_when_corridor_exists() {
    let source = include_str!("../examples/cognitive_dissonance.phi");
    let qasm = compile_to_openqasm_with_options(source, &topology_options(ladder_profile(20)))
        .expect("topology-aware compilation failed");

    assert!(!qasm.to_ascii_lowercase().contains("swap"));
}

#[test]
fn test_ibm_smoke_still_compiles_after_topology_layer() {
    let source = include_str!("../examples/ibm_smoke.phi");
    let qasm = compile_to_openqasm(source, false).expect("ibm_smoke should still compile");

    assert!(qasm.contains("OPENQASM 3.0;"));
    assert!(qasm.contains("ry(0.6180339887 * pi)"));
}

#[test]
fn test_existing_freq_chain_examples_preserve_behavior() {
    let source = r#"
intention "A" { entangle on 432 }
intention "B" { entangle on 432 }
intention "C" { entangle on 432 }
"#;
    let qasm = compile_to_openqasm(source, false).expect("legacy entanglement example should compile");

    assert!(qasm.contains("cx q["));
}
