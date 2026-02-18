use phiflow::parser::PhiExpression;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::PhiIRValue;

#[test]
fn test_evaluator_basic_arithmetic() {
    // 10 + 20
    let exprs = vec![PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::Number(10.0)),
        operator: phiflow::parser::BinaryOperator::Add,
        right: Box::new(PhiExpression::Number(20.0)),
    }];

    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    let result = evaluator.run().expect("Evaluation failed");

    if let PhiIRValue::Number(n) = result {
        assert!(
            (n - 30.0).abs() < f64::EPSILON,
            "Expected 30.0, got {:?}",
            n
        );
    } else {
        panic!("Expected Number, got {:?}", result);
    }
}

#[test]
fn test_evaluator_multiple_instructions() {
    // (10 + 20) * 2
    let exprs = vec![PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::BinaryOp {
            left: Box::new(PhiExpression::Number(10.0)),
            operator: phiflow::parser::BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(20.0)),
        }),
        operator: phiflow::parser::BinaryOperator::Multiply,
        right: Box::new(PhiExpression::Number(2.0)),
    }];

    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    let result = evaluator.run().expect("Evaluation failed");

    if let PhiIRValue::Number(n) = result {
        assert!(
            (n - 60.0).abs() < f64::EPSILON,
            "Expected 60.0, got {:?}",
            n
        );
    } else {
        panic!("Expected Number, got {:?}", result);
    }
}
