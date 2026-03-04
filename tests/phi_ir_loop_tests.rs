//! Tests for PhiIR Control Flow Lowering (Loops)

use phiflow::parser::{BinaryOperator, PhiExpression};
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::PhiIRValue;

fn run_ast(exprs: Vec<PhiExpression>) -> PhiIRValue {
    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    evaluator.run().expect("Evaluation failed")
}

#[test]
fn test_lowering_while_loop_decrements() {
    // let x = 3.0;
    // while x > 0.0 {
    //     x = x - 1.0;
    // }
    // x
    let exprs = vec![
        // let x = 3.0
        PhiExpression::LetBinding {
            name: "x".to_string(),
            phi_type: None,
            value: Box::new(PhiExpression::Number(3.0)),
        },
        // while x > 0.0
        PhiExpression::WhileLoop {
            condition: Box::new(PhiExpression::BinaryOp {
                left: Box::new(PhiExpression::Variable("x".to_string())),
                operator: BinaryOperator::Greater,
                right: Box::new(PhiExpression::Number(0.0)),
            }),
            body: Box::new(PhiExpression::LetBinding {
                name: "x".to_string(),
                phi_type: None,
                value: Box::new(PhiExpression::BinaryOp {
                    left: Box::new(PhiExpression::Variable("x".to_string())),
                    operator: BinaryOperator::Subtract,
                    right: Box::new(PhiExpression::Number(1.0)),
                }),
            }),
        },
        // return x
        PhiExpression::Variable("x".to_string()),
    ];

    let result = run_ast(exprs);
    assert_eq!(result, PhiIRValue::Number(0.0));
}

#[test]
fn test_lowering_while_loop_false_start() {
    // let x = 10.0;
    // while x < 5.0 {
    //     x = 0.0;
    // }
    // x
    let exprs = vec![
        PhiExpression::LetBinding {
            name: "x".to_string(),
            phi_type: None,
            value: Box::new(PhiExpression::Number(10.0)),
        },
        PhiExpression::WhileLoop {
            condition: Box::new(PhiExpression::BinaryOp {
                left: Box::new(PhiExpression::Variable("x".to_string())),
                operator: BinaryOperator::Less,
                right: Box::new(PhiExpression::Number(5.0)),
            }),
            body: Box::new(PhiExpression::LetBinding {
                name: "x".to_string(),
                phi_type: None,
                value: Box::new(PhiExpression::Number(0.0)),
            }),
        },
        PhiExpression::Variable("x".to_string()),
    ];

    let result = run_ast(exprs);
    assert_eq!(result, PhiIRValue::Number(10.0));
}
