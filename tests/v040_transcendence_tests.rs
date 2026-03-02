use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::evaluator::{Evaluator, VmExecResult};
use phiflow::parser::parse_phi_program;
use phiflow::host::{CallbackHostProvider, WitnessAction};

#[test]
fn test_entangle_yields_with_frequency() {
    let source = r#"
        entangle on 432.0Hz
        42.0
    "#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);

    let mut eval = Evaluator::new(&program);
    let res = eval.run_or_yield().unwrap();

    if let VmExecResult::Entangled { frequency, frozen_state } = res {
        assert_eq!(frequency, 432.0);
        let res2 = eval.resume(frozen_state).unwrap();
        if let VmExecResult::Complete(val) = res2 {
            assert_eq!(val.as_number(), Some(42.0));
        } else {
            panic!("Expected completion after entangle");
        }
    } else {
        panic!("Expected entangle yield");
    }
}

#[test]
fn test_evolve_blocks() {
    let source = r#"
        let x = 1.0
        evolve "x = x + 1.0"
        x
    "#;
    let exprs = phiflow::parser::parse_phi_program(source).unwrap();
    let program = phiflow::phi_ir::lowering::lower_program(&exprs);
    for block in &program.blocks {
        println!("Block {}", block.id);
        for instr in &block.instructions {
            println!("  {:?} = {:?}", instr.result, instr.node);
        }
        println!("  Terminator: {:?}", block.terminator);
    }
    
    let mut eval = Evaluator::new(&program);
    let result = eval.run().unwrap();
    println!("Final Result: {:?}", result);
}

#[test]
fn test_evolving_organism_blocks() {
    let source = std::fs::read_to_string("examples/evolving_organism.phi").unwrap();
    let exprs = phiflow::parser::parse_phi_program(&source).unwrap();
    let program = phiflow::phi_ir::lowering::lower_program(&exprs);
    for block in &program.blocks {
        println!("Block {}", block.id);
        println!("  Terminator: {:?}", block.terminator);
    }
}

#[test]
fn test_evolve_boolean_return() {
    let source = r#"
        let x = 0.0
        let should_break = evolve "let healed = true 
 healed"
        if should_break == true {
            x = 1.0
        }
        x
    "#;
    let exprs = phiflow::parser::parse_phi_program(source).unwrap();
    let program = phiflow::phi_ir::lowering::lower_program(&exprs);
    let mut eval = Evaluator::new(&program);
    let result = eval.run().unwrap();
    assert_eq!(result.as_number(), Some(1.0));
}
