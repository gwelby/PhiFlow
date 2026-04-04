use phiflow::phi_ir::{
    emitter, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, PhiInstruction,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn write_temp_bytecode(stem: &str, bytes: &[u8]) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "phivm_runner_{}_{}_{}.phivm",
        stem,
        std::process::id(),
        now
    ));
    fs::write(&path, bytes).expect("failed to write temp bytecode");
    path
}

fn run_phivm(extra_args: &[&str], input: &Path) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_phivm"));
    command.args(extra_args).arg(input);
    command.output().expect("failed to run phivm binary")
}

#[test]
fn phivm_executes_bytecode_file() {
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

    let path = write_temp_bytecode("arithmetic", &emitter::emit(&program));
    let output = run_phivm(&[], &path);
    let _ = fs::remove_file(&path);

    assert!(
        output.status.success(),
        "phivm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "42\n");
}

#[test]
fn phivm_resolves_string_results() {
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

    let path = write_temp_bytecode("string", &emitter::emit(&program));
    let output = run_phivm(&[], &path);
    let _ = fs::remove_file(&path);

    assert!(
        output.status.success(),
        "phivm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "\"resonance\"\n");
}

#[test]
fn phivm_can_disassemble_before_execution() {
    let program = single_block_program(
        vec![PhiInstruction {
            result: Some(0),
            node: PhiIRNode::Const(PhiIRValue::Boolean(true)),
        }],
        PhiIRNode::Return(0),
    );

    let path = write_temp_bytecode("disassemble", &emitter::emit(&program));
    let output = run_phivm(&["--disassemble"], &path);
    let _ = fs::remove_file(&path);

    assert!(
        output.status.success(),
        "phivm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PhiVM bytecode v1"));
    assert!(stdout.contains("Blocks: 1"));
    assert!(stdout.ends_with("true\n"));
}
