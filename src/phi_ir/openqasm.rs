use crate::phi_ir::{
    CollapsePolicy, Operand, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue,
    ResonateDirection,
};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenQasmEmitError {
    UndeclaredIntention(String),
    MissingQubit(String),
}

impl fmt::Display for OpenQasmEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenQasmEmitError::UndeclaredIntention(name) => {
                write!(f, "intention `{name}` was used without being declared")
            }
            OpenQasmEmitError::MissingQubit(message) => f.write_str(message),
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
    /// Deferred measurements emitted after the last gate (Final / NonDestructive policy)
    deferred_measures: Vec<String>,
    /// Qubits that have been collapsed by a mid-circuit measurement
    collapsed_qubits: HashSet<usize>,
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
            deferred_measures: Vec::new(),
            collapsed_qubits: HashSet::new(),
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

        if !ir.intentions_declared.is_empty() {
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
                    PhiIRNode::Resonate {
                        value, direction, ..
                    } => {
                        let target_q = self
                            .current_qubit_idx()
                            .map_err(OpenQasmEmitError::MissingQubit)?;
                        if self.collapsed_qubits.contains(&target_q) {
                            eprintln!("WARNING: Resonate instruction applied to qubit [{}] AFTER it was witnessed mid-circuit. Qubit state is collapsed.", target_q);
                            self.source.push_str(&format!("    // WARNING: Gating post-collapsed qubit q[{}]\n", target_q));
                        }
                        let theta = self.resonate_theta(*value, *direction, &number_constants);
                        self.source
                            .push_str(&format!("    ry({theta}) q[{target_q}]; // Resonate\n"));
                    }
                    PhiIRNode::Witness {
                        collapse_policy, ..
                    } => {
                        match collapse_policy {
                            CollapsePolicy::MidCircuit => {
                                // Mid-circuit: emit measurement immediately so later gates
                                // can be conditioned on the result.
                                if self.num_qubits > 1 {
                                    for i in 0..self.num_qubits {
                                        self.source.push_str(&format!(
                                            "    c[{i}] = measure q[{i}]; // MidCircuit Witness q{i}\n"
                                        ));
                                        self.collapsed_qubits.insert(i);
                                    }
                                } else {
                                    let target_q = self
                                        .single_declared_qubit_idx()
                                        .map_err(OpenQasmEmitError::MissingQubit)?;
                                    self.source.push_str(&format!(
                                        "    c[{target_q}] = measure q[{target_q}]; // MidCircuit Witness\n"
                                    ));
                                    self.collapsed_qubits.insert(target_q);
                                }
                            }
                            CollapsePolicy::Final | CollapsePolicy::NonDestructive => {
                                // Final: record intent and flush at end of circuit.
                                if self.num_qubits > 1 {
                                    for i in 0..self.num_qubits {
                                        self.deferred_measures.push(format!(
                                            "    c[{i}] = measure q[{i}]; // Final Witness q{i}"
                                        ));
                                    }
                                } else {
                                    let target_q = self
                                        .single_declared_qubit_idx()
                                        .map_err(OpenQasmEmitError::MissingQubit)?;
                                    self.deferred_measures.push(format!(
                                        "    c[{target_q}] = measure q[{target_q}]; // Final Witness"
                                    ));
                                }
                            }
                        }
                    }
                    PhiIRNode::CoherenceCheck => {
                        let target_q = self
                            .current_qubit_idx()
                            .map_err(OpenQasmEmitError::MissingQubit)?;
                        if self.collapsed_qubits.contains(&target_q) {
                            eprintln!("WARNING: CoherenceCheck applied to qubit [{}] AFTER it was witnessed mid-circuit. Qubit state is collapsed.", target_q);
                            self.source.push_str(&format!("    // WARNING: Coherence post-collapsed qubit q[{}]\n", target_q));
                        }
                        self.source.push_str(&format!(
                            "    ry(0.6180339887 * pi) q[{}]; // Coherence\n",
                            target_q
                        ));
                    }
                    PhiIRNode::Entangle(freq) => {
                        let f = freq.round() as u32;
                        let q2 = self
                            .current_qubit_idx()
                            .map_err(OpenQasmEmitError::MissingQubit)?;
                        // Seed the chain with the first qubit that actually fires an Entangle
                        // on this frequency.  Using q2 (rather than a hard-coded qubit 0)
                        // correctly isolates unrelated frequency channels.
                        let chain = self.freq_chains.entry(f).or_insert_with(|| vec![q2]);

                        if !chain.contains(&q2) {
                            if self.collapsed_qubits.contains(&q2) {
                                eprintln!("WARNING: Entangle instruction applied to qubit [{}] AFTER it was witnessed mid-circuit. Qubit state is collapsed.", q2);
                                self.source.push_str(&format!("    // WARNING: Entangle post-collapsed qubit q[{}]\n", q2));
                            }
                            let parent_idx = if self.optimize_depth {
                                (chain.len() - 1) / 2
                            } else {
                                chain.len() - 1
                            };
                            let q1 = chain[parent_idx];
                            self.source.push_str(&format!("    cx q[{}], q[{}]; // Entangle via {}Hz\n", q1, q2, f));
                            chain.push(q2);
                        }
                    }
                    _ => {
                        // Classical math/registers are largely skipped in this pure-quantum representation
                    }
                }
            }
        }

        // Flush deferred (Final/NonDestructive) measurements at end of circuit
        if !self.deferred_measures.is_empty() {
            self.source.push_str("\n    // --- Final Witness measurements (end-of-circuit) ---\n");
            let deferred = self.deferred_measures.drain(..).collect::<Vec<_>>();
            for line in deferred {
                self.source.push_str(&format!("{line}\n"));
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
        self.deferred_measures.clear();
        self.collapsed_qubits.clear();
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

    fn resonate_theta(
        &self,
        value: Option<Operand>,
        direction: ResonateDirection,
        number_constants: &HashMap<Operand, f64>,
    ) -> String {
        match value.and_then(|op| number_constants.get(&op).copied()) {
            Some(confidence) => {
                let theta = format!("{} * pi", format_multiplier(confidence));
                match direction {
                    ResonateDirection::TeamA => theta,
                    ResonateDirection::TeamB => format!("pi - ({theta})"),
                }
            }
            None => "pi/2".to_string(),
        }
    }

    fn current_qubit_idx(&self) -> Result<usize, String> {
        let name = self.active_intentions.last().ok_or_else(|| {
            "no active intention is bound to a qubit; declare an intention before emitting qubit-specific OpenQASM operations"
                .to_string()
        })?;

        self.qubit_mapping
            .get(name)
            .copied()
            .ok_or_else(|| format!("intention `{name}` was used without being declared"))
    }

    fn single_declared_qubit_idx(&self) -> Result<usize, String> {
        if !self.active_intentions.is_empty() {
            return self.current_qubit_idx();
        }

        if self.qubit_mapping.len() == 1 {
            return self
                .qubit_mapping
                .values()
                .copied()
                .next()
                .ok_or_else(|| "exactly one qubit was expected, but none were declared".to_string());
        }

        Err(
            "no active intention is bound to a qubit; witness must target an active intention or a single declared qubit"
                .to_string(),
        )
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
    fn test_full_pipeline_team_b_direction() {
        use crate::parser::parse_phi_program;
        use crate::phi_ir::lowering::lower_program;

        let source = "intention \"Tesla\" {\n    resonate 0.72 toward TEAM_B\n}\n";
        let exprs = parse_phi_program(source).expect("parse failed");
        let program = lower_program(&exprs);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&program).expect("emit failed");

        assert!(code.contains("ry(pi - (0.72 * pi)) q[0]; // Resonate"));
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
        // Eagles seeds the 0Hz chain as root (q[0]) by firing Entangle first.
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(0.0),
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
                direction: ResonateDirection::TeamA,
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
        assert!(code.contains("cx q[0], q[1]")); // entangle Eagles and Chiefs
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

        for name in ir.intentions_declared.iter() {
            block.instructions.push(PhiInstruction {
                result: None,
                node: PhiIRNode::IntentionPush {
                    name: name.clone(),
                    frequency_hint: None,
                },
            });
            // Every intention fires Entangle so that the first one seeds the chain
            // as the root, producing a well-defined linear/tree topology.
            block.instructions.push(PhiInstruction {
                result: None,
                node: PhiIRNode::Entangle(432.0),
            });
        }
        ir.blocks.push(block);
        ir
    }

    #[test]
    fn test_openqasm_linear_topology() {
        let ir = create_test_ir(vec!["I0", "I1", "I2", "I3"]);
        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("OpenQASM emission should succeed");

        assert!(code.contains("cx q[0], q[1]"));
        assert!(code.contains("cx q[1], q[2]"));
        assert!(code.contains("cx q[2], q[3]"));
    }

    #[test]
    fn test_openqasm_tree_topology() {
        let ir = create_test_ir(vec!["I0", "I1", "I2", "I3"]);
        let mut emitter = OpenQasmEmitter::new();
        emitter.optimize_depth = true;
        let code = emitter.emit(&ir).expect("OpenQASM emission should succeed");

        assert!(code.contains("cx q[0], q[1]"));
        assert!(code.contains("cx q[0], q[2]"));
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
        // All four intentions declare the same 432Hz frequency.  I0 fires first
        // and becomes the chain root (q[0]); subsequent qubits chain linearly.
        for name in ir.intentions_declared.iter() {
            push_intention(&mut block, name);
            block.instructions.push(PhiInstruction {
                result: None,
                node: PhiIRNode::Entangle(432.0),
            });
        }
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("432Hz chain should emit");

        assert!(code.contains("cx q[0], q[1]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[1], q[2]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[2], q[3]; // Entangle via 432Hz"));
    }

    #[test]
    fn test_openqasm_resonate_confidence_values() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["Tesla".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "Tesla");
        push_number(&mut block, 0, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(0),
                frequency_relationship: None,
                direction: ResonateDirection::TeamA,
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
        ir.intentions_declared = vec!["Tesla".to_string(), "Einstein".to_string()];

        let mut block = new_block("entry");
        push_intention(&mut block, "Tesla");
        push_number(&mut block, 0, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(0),
                frequency_relationship: None,
                direction: ResonateDirection::TeamA,
            },
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::IntentionPop,
        });
        push_intention(&mut block, "Einstein");
        push_number(&mut block, 1, 0.72);
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Resonate {
                value: Some(1),
                frequency_relationship: None,
                direction: ResonateDirection::TeamB,
            },
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let code = emitter.emit(&ir).expect("team direction mapping should emit");

        assert!(code.contains("ry(0.72 * pi) q[0]; // Resonate"));
        assert!(code.contains("ry(pi - (0.72 * pi)) q[1]; // Resonate"));
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
        println!("{}", code);

        let measure_idx = code
            .find("c[0] = measure q[0]; // MidCircuit Witness q0")
            .unwrap_or_else(|| panic!("witness should measure before later gates\ncode:\n{}", code));
        let coherence_idx = code
            .find("ry(0.6180339887 * pi) q[1]; // Coherence")
            .unwrap_or_else(|| panic!("coherence gate should still be emitted\ncode:\n{}", code));

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
    fn test_openqasm_missing_qubit_error() {
        let mut ir = PhiIRProgram::new();
        ir.intentions_declared = vec!["Tesla".to_string()];

        let mut block = new_block("entry");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::CoherenceCheck,
        });
        ir.blocks.push(block);

        let mut emitter = OpenQasmEmitter::new();
        let error = emitter
            .emit(&ir)
            .expect_err("coherence without an active intention should fail");

        assert_eq!(
            error,
            OpenQasmEmitError::MissingQubit(
                "no active intention is bound to a qubit; declare an intention before emitting qubit-specific OpenQASM operations"
                    .to_string()
            )
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
        // I0 seeds both the 432Hz and 528Hz chains as q[0] root.
        push_intention(&mut block, "I0");
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(432.0),
        });
        block.instructions.push(PhiInstruction {
            result: None,
            node: PhiIRNode::Entangle(528.0),
        });
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

        assert!(code.contains("cx q[0], q[1]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[1], q[2]; // Entangle via 432Hz"));
        assert!(code.contains("cx q[0], q[3]; // Entangle via 528Hz"));
        assert!(code.contains("cx q[3], q[4]; // Entangle via 528Hz"));
        assert!(!code.contains("cx q[2], q[3]"));
    }
}
