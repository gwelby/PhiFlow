fn main() {
    let source = std::fs::read_to_string("examples/evolving_organism.phi").unwrap();
    let exprs = phiflow::parser::parse_phi_program(&source).unwrap();
    let prog = phiflow::phi_ir::lowering::lower_program(&exprs);
    println!("{}", phiflow::phi_ir::printer::PhiIRPrinter::print(&prog));
}
