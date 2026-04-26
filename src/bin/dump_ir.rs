fn main() {
    let source = r#"
intention "A" { entangle on 432 }
intention "B" { entangle on 432 }
"#;
    let exprs = phiflow::parser::parse_phi_program(source).unwrap();
    println!("AST: {:#?}", exprs);
    let ir = phiflow::phi_ir::lowering::lower_program_checked(&exprs).unwrap();
    println!("IR: {:#?}", ir);
}
