use phiflow::parser::PhiExpression;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::Optimizer;
use phiflow::phi_ir::{PhiIRNode, PhiIRValue};

#[test]
fn test_constant_folding_basic() {
    // 2 + 3
    let exprs = vec![PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::Number(2.0)),
        operator: phiflow::parser::BinaryOperator::Add,
        right: Box::new(PhiExpression::Number(3.0)),
    }];

    let mut program = lower_program(&exprs);

    println!("Before Optimization:");
    println!(
        "{}",
        phiflow::phi_ir::printer::PhiIRPrinter::print(&program)
    );

    Optimizer::new(phiflow::phi_ir::optimizer::OptimizationLevel::Basic).optimize(&mut program);

    println!("After Optimization:");
    println!(
        "{}",
        phiflow::phi_ir::printer::PhiIRPrinter::print(&program)
    );

    // Expected:
    // The BinOp should be replaced by Const(5.0)
    // The original Const(2) and Const(3) might be dead and removed if DCE works.

    let block = &program.blocks[0];
    // We expect the terminator to be Fallthrough or Return.
    // The last instruction should be Const(5.0)
    // Let's check instructions.

    let instructions = &block.instructions;

    // Check if we have a Const(5.0)
    let has_five = instructions.iter().any(|i| {
        if let PhiIRNode::Const(PhiIRValue::Number(n)) = &i.node {
            (n - 5.0).abs() < f64::EPSILON
        } else {
            false
        }
    });

    assert!(has_five, "Constant folding failed: 5.0 not found");
}

#[test]
fn test_dead_code_elimination() {
    // let x = 2 + 3; (unused)
    // let y = 10;
    // return y;

    let exprs = vec![
        PhiExpression::LetBinding {
            name: "x".to_string(),
            value: Box::new(PhiExpression::BinaryOp {
                left: Box::new(PhiExpression::Number(2.0)),
                operator: phiflow::parser::BinaryOperator::Add,
                right: Box::new(PhiExpression::Number(3.0)),
            }),
            phi_type: None,
        },
        PhiExpression::LetBinding {
            name: "y".to_string(),
            value: Box::new(PhiExpression::Number(10.0)),
            phi_type: None,
        },
        PhiExpression::Variable("y".to_string()),
    ];

    let mut program = lower_program(&exprs);
    Optimizer::new(phiflow::phi_ir::optimizer::OptimizationLevel::Basic).optimize(&mut program);

    println!(
        "{}",
        phiflow::phi_ir::printer::PhiIRPrinter::print(&program)
    );

    let _block = &program.blocks[0];

    // "x" calculation (2+3) should be folded to 5, then stored.
    // StoreVar is side-effect, so it stays.
    // The Const(2) and Const(3) inputs should be DCE'd if possible.
    // Wait, `StoreVar` keeps `x` alive.
    // And `x` variable might be used?
    // In this specific IR, `StoreVar` takes a value.
    // If the value is a Const, it stays.
    // Optimization doesn't remove `StoreVar` because it's "impure".

    // To test pure DCE, we need an expression result that is NOT stored and NOT returned.
    // e.g. just `2 + 3;` as a statement.
    // In Lowering, `Block(exprs)` lowers all.
    // If we have `2 + 3` in the middle, its result is unused.
    //
    // We expect:
    // 1. Const(10) -> Used by return.
    // 2. That's it.
    // `lower_program` terminates with `Return(last_op)`.
    // `lower_program` calls `lower_expr` for each.
    // If the last expression was `10.0`, then `Return(last_op)` uses it.
    // Anything else should be dead.

    let exprs_dce = vec![
        PhiExpression::BinaryOp {
            // Unused calculation (Operand 0..2)
            left: Box::new(PhiExpression::Number(2.0)),
            operator: phiflow::parser::BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(3.0)),
        },
        PhiExpression::Number(10.0), // Return this (Operand 3)
    ];

    let mut prog = lower_program(&exprs_dce);

    Optimizer::new(phiflow::phi_ir::optimizer::OptimizationLevel::Basic).optimize(&mut prog);

    let _block = &prog.blocks[0];

    // Check that `2+3` is gone.
    // 2+3 involves: Const(2), Const(3), BinOp.
    // They should all be Nop.

    for instr in &prog.blocks[0].instructions {
        if let PhiIRNode::Const(PhiIRValue::Number(n)) = &instr.node {
            if *n == 5.0 {
                panic!("Found Const(5.0) which should have been DCE'd!");
            }
        }
        if let PhiIRNode::BinOp { .. } = &instr.node {
            panic!("Found BinOp which should have been DCE'd (or folded then DCE'd)!");
        }
    }
}
