use phiflow::parser::{PhiLexer, PhiParser};
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::PhiIRValue;

/// Prove that a variable declared OUTSIDE a stream block is visible INSIDE it.
/// The stream should loop N times while threshold is unreachable, not bail on cycle 1.
#[test]
fn test_stream_scope_outer_variable_visible_inside() {
    // x starts at 0.0, increments by 1.0 each cycle.
    // threshold = 3.0 is declared OUTSIDE the stream.
    // The loop should run exactly 3 times (x=1, x=2, x=3), breaking when x >= threshold.
    let source = r#"
let x = 0.0
let threshold = 3.0
stream "counter" {
    x = x + 1.0
    resonate x
    if x >= threshold {
        break stream
    }
}
"#;

    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = PhiParser::new(tokens);
    let ast = parser.parse().unwrap();
    let ir = lower_program(&ast);

    let mut evaluator = Evaluator::new(&ir);
    evaluator.run().unwrap();

    // The stream should have run 3 cycles (x=1, x=2, x=3 before break)
    let resonated = evaluator.resonated_values("counter");

    // With correct scoping: final resonance = 3.0 (3rd cycle overwrites)
    // With broken scoping: threshold = Void = 0.0, so x >= threshold on cycle 1 and resonance = 1.0
    assert_eq!(
        resonated.last().unwrap(),
        &PhiIRValue::Number(3.0),
        "Expected 3 cycles (outer threshold=3.0 visible inside stream), got: {:?}",
        resonated
    );
}

/// Confirm threshold loads as a Number, not Void, when declared outside a stream context
#[test]
fn test_outer_variable_loads_correctly_across_blocks() {
    let source = r#"
let outer_val = 42.0
stream "test" {
    let inner = outer_val
    resonate inner
    break stream
}
"#;

    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = PhiParser::new(tokens);
    let ast = parser.parse().unwrap();
    let ir = lower_program(&ast);

    let mut evaluator = Evaluator::new(&ir);
    evaluator.run().unwrap();

    let resonated = evaluator.resonated_values("test");
    assert_eq!(
        resonated.last().unwrap(),
        &PhiIRValue::Number(42.0),
        "outer_val=42.0 must be visible inside stream block, got: {:?}",
        resonated
    );
}
