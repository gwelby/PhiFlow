use phiflow::parser::{parse_phi_program, PhiExpression};
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::PhiIRValue;

const NESTED_SOURCE: &str = r#"
function square(n: Number) -> Number {
    return n * n
}

function one_minus_inv_square(x: Number) -> Number {
    let sq = square(x)
    return 1.0 - (1.0 / sq)
}

intention "lambda_test" {
    let phi = 1.618
    let result = one_minus_inv_square(phi)
    resonate result
}
"#;

const SIMPLE_SOURCE: &str = r#"
function give_42() -> Number {
    return 42.0
}
intention "test" {
    let x = give_42()
    resonate x
}
"#;

const CLAUDE_FORMULA_SOURCE: &str = r#"
function phi_power(base: Number, exponent: Number) -> Number {
    let result = 1.0
    let i = 0.0
    while i < exponent {
        result = result * base
        i = i + 1.0
    }
    return result
}

function coherence_formula(depth: Number) -> Number {
    let phi = 1.618033988749895
    let denom = phi_power(phi, depth)
    return 1.0 - (1.0 / denom)
}

intention "LAMBDA_convergence" {
    let depth = 2.0
    let lambda = coherence_formula(depth)
    resonate lambda
}
"#;

fn run_and_get_var(source: &str, intention_name: &str) -> PhiIRValue {
    let ast = parse_phi_program(source).expect("parse failed");
    let ir = lower_program(&ast);
    let mut evaluator = Evaluator::new(&ir);
    evaluator.run().expect("eval fails");

    let val = evaluator.resonated_values(intention_name).last().cloned();
    val.unwrap_or_else(|| panic!("expected value resonated to '{intention_name}'"))
}

#[test]
fn test_nested_function_call_in_intention_block_returns_correct_value() {
    let result = run_and_get_var(NESTED_SOURCE, "lambda_test");
    let value = match result {
        PhiIRValue::Number(n) => n,
        other => panic!("expected Number for result, got {:?}", other),
    };

    assert!(
        (value - 0.618).abs() < 0.001,
        "Expected ~0.618 but got {}",
        value
    );
}

#[test]
fn test_function_call_result_assigned_in_intention_block() {
    let x = run_and_get_var(SIMPLE_SOURCE, "test");
    assert_eq!(x, PhiIRValue::Number(42.0));
}

#[test]
fn test_ast_nested_call_in_intention_block_shape() {
    let ast = parse_phi_program(NESTED_SOURCE).expect("parse failed");

    let intention = ast
        .iter()
        .find_map(|expr| {
            if let PhiExpression::IntentionBlock { body, .. } = expr {
                Some(body.as_ref())
            } else {
                None
            }
        })
        .expect("intention block not found");

    let block = match intention {
        PhiExpression::Block(exprs) => exprs,
        other => panic!("expected block body in intention, got {:?}", other),
    };

    let let_result = block
        .iter()
        .find_map(|expr| {
            if let PhiExpression::LetBinding { name, value, .. } = expr {
                if name == "result" {
                    Some(value.as_ref())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .expect("let result binding not found");

    match let_result {
        PhiExpression::FunctionCall { name, arguments } => {
            assert_eq!(name, "one_minus_inv_square");
            assert_eq!(arguments.len(), 1);
            assert!(matches!(arguments[0], PhiExpression::Variable(ref v) if v == "phi"));
        }
        other => panic!(
            "expected FunctionCall in let result binding, got {:?}",
            other
        ),
    }
}

#[test]
fn test_claude_formula_resonates_to_618() {
    let lambda = run_and_get_var(CLAUDE_FORMULA_SOURCE, "LAMBDA_convergence");
    let value = match lambda {
        PhiIRValue::Number(n) => n,
        other => panic!("expected Number for lambda, got {:?}", other),
    };
    assert!(
        (value - 0.618).abs() < 0.001,
        "Expected ~0.618 but got {}",
        value
    );
}
