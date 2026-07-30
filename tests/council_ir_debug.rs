//! Debug test: inspect PhiIR from quantum_council.phi

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program_checked;

#[test]
fn debug_council_ir() {
    let source = std::fs::read_to_string("examples/quantum_council.phi")
        .expect("council file not found");

    let ast = parse_phi_program(&source).expect("parse failed");
    let program = lower_program_checked(&ast).expect("lower failed");

    println!("\nIntentions declared: {:?}", program.intentions_declared);
    for block in &program.blocks {
        println!("\nBlock: {}", block.label);
        for inst in &block.instructions {
            println!("  {:?}", inst.node);
        }
    }
}
