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

    let block = &program.blocks[0];

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

    let exprs_dce = vec![
        PhiExpression::BinaryOp {
            // Unused calculation
            left: Box::new(PhiExpression::Number(2.0)),
            operator: phiflow::parser::BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(3.0)),
        },
        PhiExpression::Number(10.0), // Return this
    ];

    let mut prog_dce = lower_program(&exprs_dce);
    Optimizer::new(phiflow::phi_ir::optimizer::OptimizationLevel::Basic).optimize(&mut prog_dce);

    let block = &prog_dce.blocks[0];

    // The instruction corresponding to `2+3` (BinOp) should be Nop (or Const(5) then Nop).
    // The inputs Const(2) and Const(3) should be Nop.

    let active_instructions = block
        .instructions
        .iter()
        .filter(|i| !matches!(i.node, PhiIRNode::Nop))
        .count();

    // We expect:
    // 1. Const(10) -> Used by return/fallthrough?
    // 2. That's it?
    // `lower_program` terminates with `Return(0)`? No, `Return(last_op)`?
    // Check `lowering.rs`: `lower_block` returns last result.
    // `lower_program` calls `lower_expr` for each.
    // It calls `ctx.terminate(Return(0))` if not terminated.
    // And `Return(0)` uses Operand 0?
    // If Operand 0 was the result of `2+3`, then it IS used!
    // Ah, `lower_program` logic: usage of 0 is just dummy.
    // But if `2+3` defines Operand 0.
    // And `Return(0)` uses it.
    // Then it's NOT dead.

    // I need to check `lower_program` implementation details in `lowering.rs`.
    // Line 138: `ctx.terminate(PhiIRNode::Return(0));`
    // Operand(0) is the FIRST instruction.
    // So the first instruction is ALWAYS kept alive by the default terminator!
    // This is a bug/feature of the test harness lowering.

    // Fix: We should return the LAST operand, or `Void`.
    // But the test case `2+3` is first.
    // I should create a dummy first instruction so `2+3` is not 0.

    let exprs_fixed = vec![
        PhiExpression::Number(999.0), // Op 0 (kept alive by Return(0))
        PhiExpression::BinaryOp {
            // Op 1 (Unused!)
            left: Box::new(PhiExpression::Number(2.0)),
            operator: phiflow::parser::BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(3.0)),
        },
        PhiExpression::Number(10.0),
    ];

    let mut prog = lower_program(&exprs_fixed);
    // Manually ensure terminator doesn't point to the `2+3` result (which would be op index ~3 after consts).
    // `lower_program` hardcodes `Return(0)`.
    // So Ops > 0 should be DCE'able if unused.

    Optimizer::new(phiflow::phi_ir::optimizer::OptimizationLevel::Basic).optimize(&mut prog);

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
