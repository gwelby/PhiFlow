use phiflow::parser::{BinaryOperator, PhiExpression};
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::printer::PhiIRPrinter;

#[test]
fn test_lower_number() {
    let expr = PhiExpression::Number(42.0);
    let program = lower_program(&[expr]);

    let output = PhiIRPrinter::print(&program);
    println!("{}", output);

    assert!(output.contains("Const Number(42.0)"));
}

#[test]
fn test_lower_binary_op() {
    let expr = PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::Number(10.0)),
        operator: BinaryOperator::Add,
        right: Box::new(PhiExpression::Number(32.0)),
    };

    let program = lower_program(&[expr]);
    let output = PhiIRPrinter::print(&program);
    println!("{}", output);

    assert!(output.contains("BinOp Add"));
    assert!(output.contains("Const Number(10.0)"));
    assert!(output.contains("Const Number(32.0)"));
}

#[test]
fn test_lower_witness() {
    let expr = PhiExpression::Witness {
        expression: None,
        body: None,
    };

    let program = lower_program(&[expr]);
    let output = PhiIRPrinter::print(&program);

    assert!(output.contains("Witness target=ALL policy=Deferred"));
}

#[test]
fn test_lower_intention() {
    let expr = PhiExpression::IntentionBlock {
        intention: "Heal".to_string(),
        body: Box::new(PhiExpression::Number(1.0)),
    };

    let program = lower_program(&[expr]);
    let output = PhiIRPrinter::print(&program);

    assert!(output.contains("IntentionPush \"Heal\""));
    assert!(output.contains("IntentionPop"));
}

#[test]
fn test_lower_create_pattern() {
    let expr = PhiExpression::CreatePattern {
        pattern_type: "Flower".to_string(),
        frequency: 432.0,
        parameters: std::collections::HashMap::new(),
    };

    let program = lower_program(&[expr]);
    let output = PhiIRPrinter::print(&program);

    // Check for CreatePattern
    assert!(output.contains("CreatePattern Flower @"));
    assert!(output.contains("Number(432.0)"));
}
