#[cfg(test)]
mod tests {
    use phiflow::parser::{PhiExpression, PhiLexer, PhiParser};

    #[test]
    fn test_p1_keyword_collision() {
        // P-1: Keywords should be allowed as variable names
        let input = "
            let witness = 10
            let consciousness = 20
            let frequency = 30
            let intention = 40
        ";
        let mut lexer = PhiLexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = PhiParser::new(tokens);
        let result = parser.parse();

        assert!(
            result.is_ok(),
            "Failed to parse keywords as variables: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_p2_newline_sensitivity() {
        // P-2: Bare witness should not eat the next line
        let input = "
            witness
            let x = 10
        ";
        let mut lexer = PhiLexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = PhiParser::new(tokens);
        let expressions = parser
            .parse()
            .expect("Failed to parse witness with newline");

        assert_eq!(
            expressions.len(),
            2,
            "Expected 2 expressions (Witness, Let), found {}",
            expressions.len()
        );

        match &expressions[0] {
            PhiExpression::Witness { expression, body } => {
                assert!(expression.is_none(), "Witness should be bare");
                assert!(body.is_none(), "Witness should have no body");
            }
            _ => panic!("First expression should be Witness"),
        }
    }
}
