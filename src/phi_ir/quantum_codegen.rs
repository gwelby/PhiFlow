use crate::phi_ir::quantum_interaction::{
    analyze_quantum_overlay, ContradictionLadderPlan, QuantumOverlayPlan,
};
use crate::phi_ir::{PhiIRNode, PhiIRProgram};
use crate::quantum::{QuantumCircuit, QuantumGate};
use std::collections::HashMap;

/// Compiles a sequence of PhiIRNodes into a QuantumCircuit.
/// This translates high-level consciousness and math operations into quantum gate equivalents.
pub fn compile_ir_to_quantum(ir: &PhiIRProgram) -> QuantumCircuit {
    if let Ok(overlay) = analyze_quantum_overlay(ir) {
        match overlay {
            QuantumOverlayPlan::ContradictionLadder(plan) => {
                return compile_contradiction_ladder(&plan);
            }
            QuantumOverlayPlan::LegacyFreqChains(plan) => {
                return compile_legacy_freq_chains(plan);
            }
        }
    }

    let mut circuit = QuantumCircuit {
        qubits: 0,
        gates: Vec::new(),
        measurements: Vec::new(),
        metadata: HashMap::new(),
    };

    // We'll track intentions to wrap them around operations
    let mut active_intentions = Vec::new();

    for block in &ir.blocks {
        for instruction in &block.instructions {
            match &instruction.node {
                PhiIRNode::Resonate {
                    value: _,
                    frequency_relationship,
                    ..
                } => {

                    // In tests, "resonate 0.618" could pass the ratio here.
                    // We map this into a PhiHarmonic gate on qubit 0.
                    let freq = frequency_relationship.unwrap_or(0.0);

                    if circuit.qubits == 0 {
                        circuit.qubits = 1;
                    }

                    circuit.gates.push(QuantumGate::PhiHarmonic(0, freq));
                }
                PhiIRNode::CoherenceCheck => {
                    // We translate coherence into a SacredFrequency gate
                    if circuit.qubits == 0 {
                        circuit.qubits = 1;
                    }
                    circuit.gates.push(QuantumGate::SacredFrequency(0, 432));
                }
                PhiIRNode::IntentionPush {
                    name,
                    frequency_hint: _,
                } => {
                    active_intentions.push(name.clone());

                    let intentions_val = serde_json::Value::Array(
                        active_intentions
                            .iter()
                            .map(|s| serde_json::Value::String(s.clone()))
                            .collect(),
                    );
                    circuit
                        .metadata
                        .insert("intentions".to_string(), intentions_val);
                }
                PhiIRNode::IntentionPop => {
                    active_intentions.pop();

                    if active_intentions.is_empty() {
                        circuit
                            .metadata
                            .insert("intentions".to_string(), serde_json::Value::Array(vec![]));
                    } else {
                        let intentions_val = serde_json::Value::Array(
                            active_intentions
                                .iter()
                                .map(|s| serde_json::Value::String(s.clone()))
                                .collect(),
                        );
                        circuit
                            .metadata
                            .insert("intentions".to_string(), intentions_val);
                    }
                }
                _ => {
                    // Skip classical math/variable nodes for now
                }
            }
        }
    }

    circuit
}

fn compile_contradiction_ladder(plan: &ContradictionLadderPlan) -> QuantumCircuit {
    let depth = plan.depth as u32;
    let left_offset = 0;
    let right_offset = depth;
    let mut circuit = QuantumCircuit {
        qubits: depth * 2,
        gates: Vec::new(),
        measurements: vec![depth.saturating_sub(1)],
        metadata: HashMap::new(),
    };

    circuit.metadata.insert(
        "overlay".to_string(),
        serde_json::Value::String("contradiction_ladder".to_string()),
    );
    circuit.metadata.insert(
        "depth".to_string(),
        serde_json::Value::Number(serde_json::Number::from(plan.depth)),
    );

    for layer in 0..depth {
        let left = left_offset + layer;
        let right = right_offset + layer;
        if layer == 0 {
            circuit.gates.push(QuantumGate::RY(left, std::f64::consts::PI));
            circuit.gates.push(QuantumGate::RY(right, 0.0));
        } else {
            circuit
                .gates
                .push(QuantumGate::RY(left, 0.6180339887 * std::f64::consts::PI));
            circuit
                .gates
                .push(QuantumGate::RY(right, 0.6180339887 * std::f64::consts::PI));
        }
        circuit.gates.push(QuantumGate::CZ(left, right));

        if layer + 1 < depth {
            circuit.gates.push(QuantumGate::CZ(left, left + 1));
            circuit.gates.push(QuantumGate::CZ(right, right + 1));
        }
    }

    if depth > 0 {
        circuit
            .gates
            .push(QuantumGate::CZ(depth - 1, right_offset + depth - 1));
    }

    circuit
}

fn compile_legacy_freq_chains(
    plan: crate::phi_ir::quantum_interaction::LegacyFreqChainPlan,
) -> QuantumCircuit {
    let mut circuit = QuantumCircuit {
        qubits: plan
            .frequencies
            .iter()
            .map(|chain| chain.operands.len() as u32)
            .max()
            .unwrap_or(0),
        gates: Vec::new(),
        measurements: Vec::new(),
        metadata: HashMap::new(),
    };

    circuit.metadata.insert(
        "overlay".to_string(),
        serde_json::Value::String("legacy_freq_chains".to_string()),
    );

    for chain in plan.frequencies {
        for idx in 0..chain.operands.len().saturating_sub(1) {
            circuit.gates.push(QuantumGate::CZ(idx as u32, idx as u32 + 1));
        }
        if !chain.operands.is_empty() {
            circuit.measurements.push((chain.operands.len() - 1) as u32);
        }
    }

    circuit
}
