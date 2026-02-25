use phiflow::parser::{PhiLexer, PhiParser, PhiToken};
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;

#[test]
fn test_lexer_break_stream() {
    let source = "break stream";
    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize().expect("Lexing failed");

    assert_eq!(tokens[0], PhiToken::Break);
    assert_eq!(tokens[1], PhiToken::Stream);
}

#[test]
fn test_stream_execution_and_resonance_overwrite() {
    let source = r#"
    let x = 0.0
    stream "test_loop" {
        x = x + 1.0
        resonate x
        if x > 2.5 {
            break stream
        }
    }
    "#;

    // This will fail to compile or parse until Lane B is implemented fully.
    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = PhiParser::new(tokens);
    let ast = parser.parse().unwrap();
    let ir = lower_program(&ast);

    let mut evaluator = Evaluator::new(&ir);
    evaluator.run().unwrap();

    // The final resonance value in 'test_loop' field should be 3.0
    let resonated = evaluator.resonated_values("test_loop");
    assert_eq!(
        resonated.last().unwrap(),
        &phiflow::phi_ir::PhiIRValue::Number(3.0)
    );
}
