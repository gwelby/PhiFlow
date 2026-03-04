use phiflow::phi_ir::{
    emitter,
    vm::PhiVm,
    PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, PhiInstruction,
};

fn single_block_program(instructions: Vec<PhiInstruction>, terminator: PhiIRNode) -> PhiIRProgram {
    PhiIRProgram {
        blocks: vec![PhiIRBlock {
            id: 0,
            label: "entry".to_string(),
            instructions,
            terminator,
        }],
        entry: 0,
        string_table: Vec::new(),
        frequencies_declared: Vec::new(),
        intentions_declared: Vec::new(),
    }
}

#[test]
fn vm_runs_emitted_arithmetic_program() {
    let program = single_block_program(
        vec![
            PhiInstruction {
                result: Some(0),
                node: PhiIRNode::Const(PhiIRValue::Number(10.0)),
            },
            PhiInstruction {
                result: Some(1),
                node: PhiIRNode::Const(PhiIRValue::Number(32.0)),
            },
            PhiInstruction {
                result: Some(2),
                node: PhiIRNode::BinOp {
                    op: PhiIRBinOp::Add,
                    left: 0,
                    right: 1,
                },
            },
        ],
        PhiIRNode::Return(2),
    );

    let bytes = emitter::emit(&program);
    let result = PhiVm::run_bytes(&bytes).expect("VM should execute bytecode");
    assert_eq!(result, PhiIRValue::Number(42.0));
}

#[test]
fn vm_runs_emitted_branch_program() {
    let program = PhiIRProgram {
        blocks: vec![
            PhiIRBlock {
                id: 0,
                label: "entry".to_string(),
                instructions: vec![PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Boolean(true)),
                }],
                terminator: PhiIRNode::Branch {
                    condition: 0,
                    then_block: 1,
                    else_block: 2,
                },
            },
            PhiIRBlock {
                id: 1,
                label: "then".to_string(),
                instructions: vec![PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::Const(PhiIRValue::Number(7.0)),
                }],
                terminator: PhiIRNode::Return(1),
            },
            PhiIRBlock {
                id: 2,
                label: "else".to_string(),
                instructions: vec![PhiInstruction {
                    result: Some(2),
                    node: PhiIRNode::Const(PhiIRValue::Number(9.0)),
                }],
                terminator: PhiIRNode::Return(2),
            },
        ],
        entry: 0,
        string_table: Vec::new(),
        frequencies_declared: Vec::new(),
        intentions_declared: Vec::new(),
    };

    let bytes = emitter::emit(&program);
    let result = PhiVm::run_bytes(&bytes).expect("VM should execute branch bytecode");
    assert_eq!(result, PhiIRValue::Number(7.0));
}

#[test]
fn vm_round_trips_emitted_string_values() {
    let program = PhiIRProgram {
        blocks: vec![PhiIRBlock {
            id: 0,
            label: "entry".to_string(),
            instructions: vec![PhiInstruction {
                result: Some(0),
                node: PhiIRNode::Const(PhiIRValue::String(0)),
            }],
            terminator: PhiIRNode::Return(0),
        }],
        entry: 0,
        string_table: vec!["resonance".to_string()],
        frequencies_declared: Vec::new(),
        intentions_declared: Vec::new(),
    };

    let bytes = emitter::emit(&program);
    let mut vm = PhiVm::from_bytes(&bytes).expect("VM should load bytecode");
    let result = vm.run().expect("VM should execute bytecode");

    match result {
        PhiIRValue::String(index) => {
            assert_eq!(
                vm.string_table().get(index as usize),
                Some(&"resonance".to_string())
            );
        }
        other => panic!("expected string result, got {:?}", other),
    }
}
