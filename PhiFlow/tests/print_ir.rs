use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::{Optimizer, OptimizationLevel};
use std::fs;

fn main() {
    let source = fs::read_to_string("examples/claude.phi").expect("failed to read claude.phi");
    let exprs = parse_phi_program(&source).expect("parse failed");
    let mut program = lower_program(&exprs);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);
    println!("{:#?}", program);
}
