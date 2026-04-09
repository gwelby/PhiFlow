use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::parser::parse_phi_program;

fn main() {
    let source = r#"
let limit = 0.5
intention "Redundant_Chaos" {
    stream "chaos" {
        resonate 0.8
        let current = coherence
        if current < limit {
            break stream
        }
    }
    witness
}
"#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    match evaluator.run() {
        Ok(val) => println!("Success: {:?}", val),
        Err(e) => println!("Error: {:?}", e),
    }
}
