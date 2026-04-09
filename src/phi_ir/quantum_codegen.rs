use crate::phi_ir::{PhiIRNode, PhiIRProgram};
use crate::quantum::{QuantumCircuit, QuantumGate};
use std::collections::HashMap;

/// Compiles a sequence of PhiIRNodes into a QuantumCircuit.
/// This translates high-level consciousness and math operations into quantum gate equivalents.
pub fn compile_ir_to_quantum(ir: &PhiIRProgram) -> QuantumCircuit {
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
