use phiflow::parser::{PhiExpression, PhiLexer, PhiParser};

fn parse_source(input: &str) -> Vec<PhiExpression> {
    let mut lexer = PhiLexer::new(input);
    let tokens = lexer
        .tokenize()
        .unwrap_or_else(|e| panic!("lexing failed: {e}"));
    let mut parser = PhiParser::new(tokens);
    parser
        .parse()
        .unwrap_or_else(|e| panic!("parsing failed: {e}"))
}

#[test]
fn test_p1_keyword_collision() {
    // P-1: keywords should still parse when used as variable names.
    let input = r#"
        let witness = 10
        let consciousness = 20
        let frequency = 30
        let intention = 40
        let resonate = 50
    "#;

    let expressions = parse_source(input);
    assert_eq!(expressions.len(), 5);
}

#[test]
fn test_p2_newline_sensitivity_witness() {
    // P-2: bare witness should not consume the next statement.
    let input = r#"
        witness
        let x = 10
    "#;
    let expressions = parse_source(input);

    assert_eq!(
        expressions.len(),
        2,
        "expected 2 expressions (Witness, Let), found {}",
        expressions.len()
    );

    match &expressions[0] {
        PhiExpression::Witness { expression, body } => {
            assert!(expression.is_none(), "witness should be bare");
            assert!(body.is_none(), "witness should have no body");
        }
        other => panic!("first expression should be Witness, got {:?}", other),
    }
}

#[test]
fn test_p2_newline_sensitivity_resonate() {
    // P-2 companion check: bare resonate should also not consume the next statement.
    let input = r#"
        resonate
        let x = 10
    "#;
    let expressions = parse_source(input);

    assert_eq!(
        expressions.len(),
        2,
        "expected 2 expressions (Resonate, Let), found {}",
        expressions.len()
    );

    match &expressions[0] {
        PhiExpression::Resonate { expression } => {
            assert!(expression.is_none(), "resonate should be bare");
        }
        other => panic!("first expression should be Resonate, got {:?}", other),
    }
}
