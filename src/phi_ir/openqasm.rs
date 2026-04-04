//! OpenQASM 3.0 Backend for PhiFlow
//!
//! This module compiles PhiFlow's PhiIR to OpenQASM 3.0 bytecode for execution
//! on IBM Quantum hardware (ibm_brisbane, ibm_fez, etc.).
//!
//! # Semantic Mapping
//!
//! | PhiFlow Construct | OpenQASM 3.0 | IBM Gate | Description |
//! |-------------------|--------------|----------|-------------|
//! | `resonate 0.72` | `ry(0.72 * pi)` | `ry` | Amplitude encoding: confidence → rotation angle |
//! | `resonate 0.72 toward TEAM_B` | `ry(0.28 * pi)` | `ry` | Inverted: `ry((1 - θ) * pi)` |
//! | `entangle on 432` | `cx q[i], q[j]` | `cx` | CNOT entanglement on 432Hz frequency chain |
//! | `witness` | `measure q -> c` | `measure` | Collapse all qubits to classical bits |
//! | `witness mid_circuit` | `measure` (inline) | `measure` | Mid-circuit measurement, gates can follow |
//! | `coherence` | `ry(0.618 * pi)` | `ry` | Golden ratio rotation (φ^(-1) ≈ 0.618) |
//!
//! # Frequency Chains
//!
//! Multiple intentions that `entangle on 432` form a chain:
//! - Linear topology (default): `cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];` — depth N-1
//! - Tree topology (`--optimize-depth`): `cx q[0],q[1]; cx q[0],q[2]; cx q[1],q[3];` — depth log₂(N)
//!
//! # Example
//!
//! PhiFlow source:
//! ```phi
//! intention "TEAM_A" {
//!     resonate 0.72 toward TEAM_A
//! }
//! intention "TEAM_B" {
//!     resonate 0.72 toward TEAM_B
//! }
//! witness
//! ```
//!
//! Compiles to:
//! ```qasm
//! OPENQASM 3.0;
//! include "stdgates.inc";
//! qubit[2] q;
//! bit[2] c;
//! ry(0.72 * pi) q[0]; // TEAM_A
//! ry(0.28 * pi) q[1]; // TEAM_B (inverted)
//! c[0] = measure q[0];
//! c[1] = measure q[1];
//! ```

use crate::phi_ir::{CollapsePolicy, Operand, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, TeamDirection};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenQasmEmitError {
    UndeclaredIntention(String),
}

impl fmt::Display for OpenQasmEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenQasmEmitError::UndeclaredIntention(name) => {
                write!(f, "intention `{name}` was used without being declared")
            }
        }
    }
}

impl Error for OpenQasmEmitError {}

pub struct OpenQasmEmitter {
    pub source: String,
    qubit_mapping: HashMap<String, usize>,
    active_intentions: Vec<String>,
    freq_chains: HashMap<u32, Vec<usize>>,
    num_qubits: usize,
    pub optimize_depth: bool,
}

impl OpenQasmEmitter {
    pub fn new() -> Self {
        OpenQasmEmitter {
            source: String::new(),
            qubit_mapping: HashMap::new(),
            active_intentions: Vec::new(),
            freq_chains: HashMap::new(),
            num_qubits: 1,
            optimize_depth: false,
        }
    }

    pub fn emit(&mut self, ir: &PhiIRProgram) -> Result<String, OpenQasmEmitError> {
        self.reset();
        self.source.push_str("OPENQASM 3.0;\n");
        self.source.push_str("include \"stdgates.inc\";\n\n");

        let num_qubits = if ir.intentions_declared.is_empty() {
            1
        } else {
            ir.intentions_declared.len()
        };
        self.num_qubits = num_qubits;

        self.source.push_str(&format!("qubit[{}] q;\n", num_qubits));
        self.source.push_str(&format!("bit[{}] c;\n\n", num_qubits));

        if ir.intentions_declared.is_empty() {
            self.qubit_mapping.insert("default".to_string(), 0);
        } else {
            for (i, name) in ir.intentions_declared.iter().enumerate() {
                self.qubit_mapping.insert(name.clone(), i);
            }
        }

        for block in &ir.blocks {
            let number_constants = self.collect_number_constants(block);
            self.source.push_str(&format!("// Block {}\n", block.label));
            for inst in &block.instructions {
                match &inst.node {
                    PhiIRNode::IntentionPush { name, .. } => {
                        if !self.qubit_mapping.contains_key(name) {
                            return Err(OpenQasmEmitError::UndeclaredIntention(name.clone()));
                        }
                        self.active_intentions.push(name.clone());
                        self.source.push_str(&format!("// Intention: {}\n", name));
                    }
                    PhiIRNode::IntentionPop => {
                        self.active_intentions.pop();
                    }
                    PhiIRNode::Resonate { value, direction, .. } => {
                        let target_q = self.current_qubit_idx()?;
                        let theta = self.resonate_theta(*value, &number_constants, *direction);
                        self.source
                            .push_str(&format!("    ry({theta}) q[{target_q}]; // Resonate\n"));
                    }
                    PhiIRNode::Witness {
                        collapse_policy, ..
                    } => {
                        match collapse_policy {
                            CollapsePolicy::MidCircuit
                            | CollapsePolicy::Deferred
                            | CollapsePolicy::NonDestructive => {
                                if self.num_qubits > 1 {
                                    for i in 0..self.num_qubits {
                                        self.source.push_str(&format!(
                                            "    c[{i}] = measure q[{i}]; // Witness q{i}\n"
                                        ));
                                    }
                                } else {
                                    let target_q = self.current_qubit_idx()?;
                                    self.source.push_str(&format!(
                                        "    c[{target_q}] = measure q[{target_q}]; // Witness\n"
                                    ));
                                }
                            }
                        }
                    }
                    PhiIRNode::CoherenceCheck => {
                        let target_q = self.current_qubit_idx()?;
                        self.source.push_str(&format!(
                            "    ry(0.6180339887 * pi) q[{}]; // Coherence\n",
                            target_q
                        ));
                    }
                    PhiIRNode::Entangle(freq) => {
                        let f = freq.round() as u32;
                        let q2 = self.current_qubit_idx()?;
                        
                        // Entangle on frequency channel: create or extend chain
                        let chain = self.freq_chains.entry(f).or_insert_with(Vec::new);
                        
                        // First member of chain: just register, no CNOT yet
                        if chain.is_empty() {
                            chain.push(q2);
                        } else {
                            // Subsequent members: entangle to previous chain member
                            let parent_idx = if self.optimize_depth {
                                (chain.len() - 1) / 2
                            } else {
                                chain.len() - 1
                            };
                            
                            let q1 = chain[parent_idx];
                            if q1 != q2 && !chain.contains(&q2) {
                                self.source.push_str(&format!("    cx q[{}], q[{}]; // Entangle via {}Hz\n", q1, q2, f));
                                chain.push(q2);
                            }
                        }
                    }
                    _ => {
                        // Classical math/registers are largely skipped in this pure-quantum representation
                    }
                }
            }
        }

        Ok(self.source.clone())
    }

    fn reset(&mut self) {
        self.source.clear();
        self.qubit_mapping.clear();
        self.active_intentions.clear();
        self.freq_chains.clear();
        self.num_qubits = 1;
    }

    fn collect_number_constants(&self, block: &PhiIRBlock) -> HashMap<Operand, f64> {
        let mut constants = HashMap::new();
        for inst in &block.instructions {
            if let (Some(result), PhiIRNode::Const(PhiIRValue::Number(value))) =
                (inst.result, &inst.node)
            {
                constants.insert(result, *value);
            }
        }
        constants
    }

    fn resonate_theta(&self, value: Option<Operand>, number_constants: &HashMap<Operand, f64>, direction: TeamDirection) -> String {
        match value.and_then(|op| number_constants.get(&op).copied()) {
            Some(confidence) => {
                let multiplier = match direction {
                    TeamDirection::TeamA => confidence,
                    TeamDirection::TeamB => 1.0 - confidence,
                };
                format!("{} * pi", format_multiplier(multiplier))
            }
            None => "pi/2".to_string(),
        }
    }

    fn current_qubit_idx(&self) -> Result<usize, OpenQasmEmitError> {
        if let Some(name) = self.active_intentions.last() {
            self.qubit_mapping
                .get(name)
                .copied()
                .ok_or_else(|| OpenQasmEmitError::UndeclaredIntention(name.clone()))
        } else {
            Ok(0)
        }
    }
}

fn format_multiplier(value: f64) -> String {
    let mut formatted = format!("{value:.12}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi_ir::PhiInstruction;

    /// Full pipeline: parse source -> lower to PhiIR -> emit OpenQASM.
    /// This catches bugs that manual IR construction tests can't — specifically
    /// whether the parser and lowerer correctly thread the confidence value through
    /// to the Resonate node so the emitter can use it.
    #[test]
    fn test_full_pipeline_resonate_value() {
        use crate::parser::parse_phi_program;
        use crate::phi_ir::lowering::lower_program;
        use crate::phi_ir::printer::PhiIRPrinter;

        let source = "intention \"Tesla\" {\n    resonate 0.72\n}\nwitness\n";
        let exprs = parse_phi_program(source).expect("parse failed");
        let program = lower_program(&exprs);

        // Dump IR for diagnosis
        let ir_dump = PhiIRPrinter::print(&program);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&program).expect("emit failed");

        assert!(
            code.contains("ry(0.72 * pi)"),
            "Expected ry(0.72 * pi) but got pi/2.\nIR dump:\n{}\nQASM:\n{}",
            ir_dump,
            code
        );
    }

    #[test]
    fn test_openqasm_emission_basic() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["Eagles".to_string(), "Chiefs".to_string()];

        let mut block = PhiIRBlock {
            id: 0,
            label: "entry".to_string(),
            instructions: Vec::new(),
            terminator: PhiIRNode::Return(0),
        };

        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::IntentionPush {
                name: "Eagles".to_string(),
                frequency_hint: None,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::CoherenceCheck,
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::IntentionPush {
                name: "Chiefs".to_string(),
                frequency_hint: None,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: None,
                frequency_relationship: None,
                direction: TeamDirection::TeamA,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(0.0),
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Witness {
                target: None,
                collapse_policy: CollapsePolicy::MidCircuit,
            },
        });

        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("OpenQASM emission should succeed");

        assert!(code.contains("OPENQASM 3.0;"));
        assert!(code.contains("qubit[2] q;"));
        assert!(code.contains("bit[2] c;"));
        assert!(code.contains("ry(0.618")); // coherence on Eagles (q[0])
        assert!(code.contains("ry(pi/2) q[1]")); // resonate on Chiefs (q[1])
        // Entangle(0.0) on Chiefs seeds the 0Hz chain, no CNOT emitted (first member)
        assert!(code.contains("c[1] = measure q[1]")); // witness on Chiefs
    }

    fn create_test_ir(intentions: Vec<&str>) -> PhiIRProgram {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = intentions.into_iter().map(|s| s.to_string()).collect();

        let mut block = PhiIRBlock {
            id: 0,
            label: "entry".to_string(),
            instructions: Vec::new(),
            terminator: PhiIRNode::Return(0),
        };

        for (i, name) in ir.intentions_declared.iter().enumerate() {
            block.instructions.push(PhiInstruction {
                result: None,
                node: PhiIRNode::IntentionPush {
                    name: name.clone(),
                    frequency_hint: None,
                },
            });
            if i > 0 {
                // Entangle starting from the 2nd intention
                block.instructions.push(PhiInstruction {
                    result: None,
                    node: PhiIRNode::Entangle(432.0),
                });
            }
        }
        ir.blocks.push(block);
        ir
    }

    #[test]
    fn test_openqasm_linear_topology() {
        let ir = create_test_ir(vec!["I0", "I1", "I2", "I3"]);
        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("OpenQASM emission should succeed");

        // I0 has no entangle, I1 seeds 432Hz chain, I2→I1, I3→I2
        assert!(code.contains("cx q[1], q[2]"));
        assert!(code.contains("cx q[2], q[3]"));
    }

    #[test]
    fn test_openqasm_tree_topology() {
        let ir = create_test_ir(vec!["I0", "I1", "I2", "I3"]);
        let mut emitter = OpenQasmEmitter::new();
        emitter.optimize_depth = true;
        let code = emitter.emit(&ir).expect("OpenQASM emission should succeed");

        // Tree topology: I1 is root, I2→I1, I3→I1 (balanced)
        assert!(code.contains("cx q[1], q[2]"));
        assert!(code.contains("cx q[1], q[3]"));
    }

    fn push_intention(block: &mut PhiIRBlock, name: &str) {
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::IntentionPush {
                name: name.to_string(),
                frequency_hint: None,
            },
        });
    }

    fn push_number(block: &mut PhiIRBlock, result: Operand, value: f64) {
        block.instructions.push(PhiInstruction {
            result: Some(result),
            node: PhiIRNode::Const(PhiIRValue::Number(value)),
        });
    }

    fn new_block(label: &str) -> PhiIRBlock {
        PhiIRBlock {
            id: 0,
            label: label.to_string(),
            instructions: Vec::new(),
            terminator: PhiIRNode::Return(0),
        }
    }

    #[test]
    fn test_openqasm_frequency_chain_entanglement() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["I0", "I1", "I2", "I3"]
            .into_iter()
            .map(str::to_string)
            .collect();

        let mut block = new_block("entry");
        for (i, name) in ir.intentions_declared.iter().enumerate() {
            push_intention(&mut block, name);
            if i > 0 {
                block.instructions.push(PhiInstruction {
                    result: None,
                    node: PhiIRNode::Entangle(432.0),
                });
            }
        }
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("432Hz chain should emit");

        // I0 has no entangle, I1 seeds 432Hz chain, I2→I1, I3→I2
        assert!(code.contains("cx q[1], q[2]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[2], q[3]; // Entangle via 432Hz"));
    }

    #[test]
    fn test_openqasm_resonate_confidence_values() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["TEAM_A".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "TEAM_A");
        push_number(&mut block, 0, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(0),
                frequency_relationship: None,
                direction: TeamDirection::TeamA,
            },
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("numeric resonate should emit");

        assert!(code.contains("ry(0.72 * pi) q[0]; // Resonate"));
        assert!(!code.contains("ry(pi/2) q[0]; // Resonate"));
    }

    #[test]
    fn test_openqasm_team_direction() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["TEAM_A".to_string(), "TEAM_B".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "TEAM_A");
        push_number(&mut block, 0, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(0),
                frequency_relationship: None,
                direction: TeamDirection::TeamA,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::IntentionPop,
        });
        push_intention(&mut block, "TEAM_B");
        push_number(&mut block, 1, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(1),
                frequency_relationship: None,
                direction: TeamDirection::TeamB,
            },
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("team direction mapping should emit");

        assert!(code.contains("ry(0.72 * pi) q[0]; // Resonate"));
        assert!(code.contains("ry(0.28 * pi) q[1]; // Resonate"));
    }

    #[test]
    fn test_openqasm_witness_mid_circuit() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["I0".to_string(), "I1".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "I0");
        push_intention(&mut block, "I1");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Witness {
                target: None,
                collapse_policy: CollapsePolicy::MidCircuit,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::CoherenceCheck,
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("mid-circuit witness should emit");

        let measure_idx = code
            .find("c[0] = measure q[0]; // Witness q0")
            .expect("witness should measure before later gates");
        let coherence_idx = code
            .find("ry(0.6180339887 * pi) q[1]; // Coherence")
            .expect("coherence gate should still be emitted");

        assert!(measure_idx < coherence_idx);
    }

    #[test]
    fn test_openqasm_undeclared_intention_error() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["Declared".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "Ghost");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::CoherenceCheck,
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let error = emitter.emit(&ir).expect_err("undeclared intention should fail");

        assert_eq!(
            error,
            OpenQasmEmitError::UndeclaredIntention("Ghost".to_string())
        );
    }

    #[test]
    fn test_openqasm_multiple_frequency_channels() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["I0", "I1", "I2", "I3", "I4"]
            .into_iter()
            .map(str::to_string)
            .collect();

        let mut block = new_block("entry");
        push_intention(&mut block, "I0");
        push_intention(&mut block, "I1");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(432.0),
        });
        push_intention(&mut block, "I2");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(432.0),
        });
        push_intention(&mut block, "I3");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(528.0),
        });
        push_intention(&mut block, "I4");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(528.0),
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("multiple channels should emit");

        // 432Hz: I0 no entangle, I1 seeds, I2→I1
        // 528Hz: I3 seeds, I4→I3
        assert!(code.contains("cx q[1], q[2]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[3], q[4]; // Entangle via 528Hz"));
        assert!(!code.contains("cx q[2], q[3]"));
    }
}
