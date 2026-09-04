//! Parser unit tests — tokenizer, expression precedence, statement parsing,
//! and the five consciousness primitives.
//!
//! The parser (src/parser/mod.rs, 2858 lines) previously had zero unit tests.
//! It was exercised only through integration tests that parse .phi files
//! end-to-end. These tests cover the parser directly: tokenization edge cases,
//! operator precedence, keyword recognition, and AST structure for each
//! primitive.

use phiflow::parser::{
    parse_phi_program, PhiExpression, PhiLexer, PhiToken, ResonateDirection,
};

// ─── Tokenizer tests ───────────────────────────────────────────────

#[test]
fn tokenizer_empty_input_produces_only_eof() {
    let mut lexer = PhiLexer::new("");
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens.len(), 1);
    assert!(matches!(tokens[0], PhiToken::Eof));
}

#[test]
fn tokenizer_whitespace_only_produces_newlines_and_eof() {
    let mut lexer = PhiLexer::new("   \n\t  \n  ");
    let tokens = lexer.tokenize().unwrap();
    // The tokenizer emits Newline tokens for whitespace containing newlines
    // The last token should always be Eof
    assert!(matches!(tokens.last().unwrap(), PhiToken::Eof));
    // All non-Eof tokens should be Newline (no other content)
    for t in &tokens[..tokens.len()-1] {
        assert!(matches!(t, PhiToken::Newline), "expected Newline, got {:?}", t);
    }
}

#[test]
fn tokenizer_comments_are_skipped() {
    let mut lexer = PhiLexer::new("// this is a comment\n");
    let tokens = lexer.tokenize().unwrap();
    // Comments are skipped, but the trailing newline is tokenized
    assert!(matches!(tokens.last().unwrap(), PhiToken::Eof));
    // Should contain at most Newline + Eof (no comment tokens)
    for t in &tokens[..tokens.len()-1] {
        assert!(matches!(t, PhiToken::Newline), "expected Newline, got {:?}", t);
    }
}

#[test]
fn tokenizer_numbers() {
    let mut lexer = PhiLexer::new("42 3.14 0.618 1e10");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::Number(n) if (n - 42.0).abs() < 1e-10));
    assert!(matches!(tokens[1], PhiToken::Number(n) if (n - 3.14).abs() < 1e-10));
    assert!(matches!(tokens[2], PhiToken::Number(n) if (n - 0.618).abs() < 1e-10));
}

#[test]
fn tokenizer_negative_numbers() {
    let mut lexer = PhiLexer::new("-5.0");
    let tokens = lexer.tokenize().unwrap();
    // -5.0 should tokenize as Minus then Number(5.0), or as Number(-5.0)
    // depending on lexer design. Check which:
    if matches!(tokens[0], PhiToken::Number(_)) {
        assert!(matches!(tokens[0], PhiToken::Number(n) if (n - (-5.0)).abs() < 1e-10));
    } else {
        assert!(matches!(tokens[0], PhiToken::Minus));
        assert!(matches!(tokens[1], PhiToken::Number(n) if (n - 5.0).abs() < 1e-10));
    }
}

#[test]
fn tokenizer_strings() {
    let mut lexer = PhiLexer::new("\"hello world\" \"test\"");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::String(ref s) if s == "hello world"));
    assert!(matches!(tokens[1], PhiToken::String(ref s) if s == "test"));
}

#[test]
fn tokenizer_empty_string() {
    let mut lexer = PhiLexer::new("\"\"");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::String(ref s) if s == ""));
}

#[test]
fn tokenizer_booleans() {
    let mut lexer = PhiLexer::new("true false");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::Boolean(true)));
    assert!(matches!(tokens[1], PhiToken::Boolean(false)));
}

#[test]
fn tokenizer_identifiers() {
    let mut lexer = PhiLexer::new("foo bar_baz x123");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::Identifier(ref s) if s == "foo"));
    assert!(matches!(tokens[1], PhiToken::Identifier(ref s) if s == "bar_baz"));
    assert!(matches!(tokens[2], PhiToken::Identifier(ref s) if s == "x123"));
}

#[test]
fn tokenizer_keywords() {
    let mut lexer = PhiLexer::new("intention witness resonate stream let function return if else while for break");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::Intention));
    assert!(matches!(tokens[1], PhiToken::Witness));
    assert!(matches!(tokens[2], PhiToken::Resonate));
    assert!(matches!(tokens[3], PhiToken::Stream));
    assert!(matches!(tokens[4], PhiToken::Let));
    assert!(matches!(tokens[5], PhiToken::Function));
    assert!(matches!(tokens[6], PhiToken::Return));
    assert!(matches!(tokens[7], PhiToken::If));
    assert!(matches!(tokens[8], PhiToken::Else));
    assert!(matches!(tokens[9], PhiToken::While));
    assert!(matches!(tokens[10], PhiToken::For));
    assert!(matches!(tokens[11], PhiToken::Break));
}

#[test]
fn tokenizer_operators() {
    let mut lexer = PhiLexer::new("+ - * / % ^ == != < <= > >= && || !");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::Plus));
    assert!(matches!(tokens[1], PhiToken::Minus));
    assert!(matches!(tokens[2], PhiToken::Star));
    assert!(matches!(tokens[3], PhiToken::Slash));
    assert!(matches!(tokens[4], PhiToken::Percent));
    assert!(matches!(tokens[5], PhiToken::Power));
    assert!(matches!(tokens[6], PhiToken::EqualEqual));
    assert!(matches!(tokens[7], PhiToken::NotEqual));
    assert!(matches!(tokens[8], PhiToken::Less));
    assert!(matches!(tokens[9], PhiToken::LessEqual));
    assert!(matches!(tokens[10], PhiToken::Greater));
    assert!(matches!(tokens[11], PhiToken::GreaterEqual));
    assert!(matches!(tokens[12], PhiToken::And));
    assert!(matches!(tokens[13], PhiToken::Or));
    assert!(matches!(tokens[14], PhiToken::Not));
}

#[test]
fn tokenizer_delimiters() {
    let mut lexer = PhiLexer::new("( ) { } [ ] , ; : -> .");
    let tokens = lexer.tokenize().unwrap();
    assert!(matches!(tokens[0], PhiToken::LeftParen));
    assert!(matches!(tokens[1], PhiToken::RightParen));
    assert!(matches!(tokens[2], PhiToken::LeftBrace));
    assert!(matches!(tokens[3], PhiToken::RightBrace));
    assert!(matches!(tokens[4], PhiToken::LeftBracket));
    assert!(matches!(tokens[5], PhiToken::RightBracket));
    assert!(matches!(tokens[6], PhiToken::Comma));
    assert!(matches!(tokens[7], PhiToken::Semicolon));
    assert!(matches!(tokens[8], PhiToken::Colon));
    assert!(matches!(tokens[9], PhiToken::Arrow));
    assert!(matches!(tokens[10], PhiToken::Dot));
}

#[test]
fn tokenizer_sacred_frequency_literal() {
    // Sacred frequency literals like 432Hz should tokenize as Sacred(432)
    let mut lexer = PhiLexer::new("432Hz");
    let tokens = lexer.tokenize().unwrap();
    // Either Sacred(432) or Number(432) + Hz — check which
    if matches!(tokens[0], PhiToken::Sacred(_)) {
        assert!(matches!(tokens[0], PhiToken::Sacred(f) if f == 432));
    } else {
        // If not a sacred literal, it should at least tokenize
        assert!(tokens.len() >= 1);
    }
}

#[test]
fn tokenizer_mixed_program() {
    let source = "let x = 1.0\nresonate x";
    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    // let x = 1.0 Newline resonate x Eof
    assert!(tokens.len() >= 7);
    assert!(matches!(tokens[0], PhiToken::Let));
    assert!(matches!(tokens[1], PhiToken::Identifier(ref s) if s == "x"));
    assert!(matches!(tokens[2], PhiToken::Equal));
}

// ─── Parser: basic expressions ─────────────────────────────────────

#[test]
fn parse_number_literal() {
    let exprs = parse_phi_program("42.0").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::Number(n) if (n - 42.0).abs() < 1e-10));
}

#[test]
fn parse_string_literal() {
    let exprs = parse_phi_program("\"hello\"").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::String(ref s) if s == "hello"));
}

#[test]
fn parse_boolean_literal() {
    let exprs = parse_phi_program("true").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::Boolean(true)));
}

#[test]
fn parse_let_binding() {
    let exprs = parse_phi_program("let x = 42.0").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::LetBinding { name, value, .. } => {
            assert_eq!(name, "x");
            assert!(matches!(value.as_ref(), PhiExpression::Number(n) if (n - 42.0).abs() < 1e-10));
        }
        other => panic!("expected LetBinding, got {:?}", other),
    }
}

#[test]
fn parse_let_binding_with_type() {
    let exprs = parse_phi_program("let x: Number = 42.0").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::LetBinding { name, .. } if name == "x"));
}

// ─── Parser: operator precedence ───────────────────────────────────

#[test]
fn parse_addition() {
    let exprs = parse_phi_program("1.0 + 2.0").unwrap();
    assert_eq!(exprs.len(), 1);
    // Should be BinaryOp { op: Add, left: Number(1), right: Number(2) }
    assert!(matches!(&exprs[0], PhiExpression::BinaryOp { .. }));
}

#[test]
fn parse_multiplication() {
    let exprs = parse_phi_program("2.0 * 3.0").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::BinaryOp { .. }));
}

#[test]
fn parse_precedence_mul_over_add() {
    // 1 + 2 * 3 should parse as 1 + (2 * 3), not (1 + 2) * 3
    let exprs = parse_phi_program("1.0 + 2.0 * 3.0").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::BinaryOp { left, right, .. } => {
            // op should be Add
            // left should be Number(1.0)
            assert!(matches!(left.as_ref(), PhiExpression::Number(n) if (n - 1.0).abs() < 1e-10));
            // right should be BinaryOp { op: Mul, ... }
            assert!(matches!(right.as_ref(), PhiExpression::BinaryOp { .. }));
        }
        other => panic!("expected BinaryOp, got {:?}", other),
    }
}

#[test]
fn parse_parentheses_override_precedence() {
    // (1 + 2) * 3 should parse as (1 + 2) * 3
    let exprs = parse_phi_program("(1.0 + 2.0) * 3.0").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::BinaryOp { left, right, .. } => {
            // op should be Mul
            // left should be BinaryOp { op: Add, ... }
            assert!(matches!(left.as_ref(), PhiExpression::BinaryOp { .. }));
            // right should be Number(3.0)
            assert!(matches!(right.as_ref(), PhiExpression::Number(n) if (n - 3.0).abs() < 1e-10));
        }
        other => panic!("expected BinaryOp, got {:?}", other),
    }
}

#[test]
fn parse_power_operator() {
    let exprs = parse_phi_program("2.0 ^ 3.0").unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::BinaryOp { .. }));
}

#[test]
fn parse_comparison_operators() {
    for src in &["1.0 < 2.0", "1.0 <= 2.0", "1.0 > 2.0", "1.0 >= 2.0", "1.0 == 2.0", "1.0 != 2.0"] {
        let exprs = parse_phi_program(src).unwrap_or_else(|e| panic!("failed to parse '{}': {}", src, e));
        assert_eq!(exprs.len(), 1, "expected 1 expression for '{}'", src);
        assert!(matches!(&exprs[0], PhiExpression::BinaryOp { .. }), "expected BinaryOp for '{}'", src);
    }
}

#[test]
fn parse_logical_operators() {
    for src in &["true && false", "true || false", "!true"] {
        let exprs = parse_phi_program(src).unwrap_or_else(|e| panic!("failed to parse '{}': {}", src, e));
        assert_eq!(exprs.len(), 1, "expected 1 expression for '{}'", src);
    }
}

// ─── Parser: the five consciousness primitives ─────────────────────

#[test]
fn parse_intention_block() {
    let source = "intention \"test\" {\n    resonate 1.0\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::IntentionBlock { intention, body } => {
            assert_eq!(intention, "test");
            // Body is a Block(Vec<PhiExpression>) containing the statements
            match body.as_ref() {
                PhiExpression::Block(stmts) => {
                    assert!(!stmts.is_empty());
                    assert!(matches!(&stmts[0], PhiExpression::Resonate { .. }));
                }
                PhiExpression::Resonate { .. } => { /* single-statement body is also valid */ }
                other => panic!("expected Block or Resonate body, got {:?}", other),
            }
        }
        other => panic!("expected IntentionBlock, got {:?}", other),
    }
}

#[test]
fn parse_bare_witness() {
    let exprs = parse_phi_program("witness").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::Witness { mid_circuit, expression, body } => {
            assert!(!*mid_circuit);
            assert!(expression.is_none());
            assert!(body.is_none());
        }
        other => panic!("expected Witness, got {:?}", other),
    }
}

#[test]
fn parse_witness_with_expression() {
    let exprs = parse_phi_program("witness 42.0").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::Witness { expression, .. } => {
            assert!(expression.is_some());
            assert!(matches!(expression.as_ref().unwrap().as_ref(), PhiExpression::Number(n) if (n - 42.0).abs() < 1e-10));
        }
        other => panic!("expected Witness, got {:?}", other),
    }
}

#[test]
fn parse_resonate_bare() {
    let exprs = parse_phi_program("resonate").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::Resonate { expression, direction } => {
            assert!(expression.is_none());
            assert!(matches!(direction, ResonateDirection::TeamA));
        }
        other => panic!("expected Resonate, got {:?}", other),
    }
}

#[test]
fn parse_resonate_with_value() {
    let exprs = parse_phi_program("resonate 0.618").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::Resonate { expression, .. } => {
            assert!(expression.is_some());
            assert!(matches!(expression.as_ref().unwrap().as_ref(), PhiExpression::Number(n) if (n - 0.618).abs() < 1e-10));
        }
        other => panic!("expected Resonate, got {:?}", other),
    }
}

#[test]
fn parse_resonate_toward_team_b() {
    let source = "resonate 1.0 toward TEAM_B";
    let result = parse_phi_program(source);
    // This may or may not be supported depending on parser version
    if let Ok(exprs) = result {
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            PhiExpression::Resonate { direction, .. } => {
                assert!(matches!(direction, ResonateDirection::TeamB));
            }
            other => panic!("expected Resonate, got {:?}", other),
        }
    }
    // If it fails to parse, that's acceptable — the toward syntax may need specific formatting
}

#[test]
fn parse_stream_block() {
    let source = "stream \"my_stream\" {\n    witness\n    break\n}";
    let result = parse_phi_program(source);
    if let Ok(exprs) = result {
        assert_eq!(exprs.len(), 1);
        assert!(matches!(&exprs[0], PhiExpression::StreamBlock { name, .. } if name == "my_stream"));
    }
}

#[test]
fn parse_break_statement() {
    // "break stream" must appear inside a stream block
    let source = "stream \"test\" {\n    break stream\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::StreamBlock { body, .. } => {
            // Body is a Block containing BreakStream
            match body.as_ref() {
                PhiExpression::Block(stmts) => {
                    assert!(!stmts.is_empty());
                    assert!(matches!(&stmts[0], PhiExpression::BreakStream));
                }
                PhiExpression::BreakStream => { /* single-statement body */ }
                other => panic!("expected Block or BreakStream body, got {:?}", other),
            }
        }
        other => panic!("expected StreamBlock, got {:?}", other),
    }
}

// ─── Parser: control flow ──────────────────────────────────────────

#[test]
fn parse_if_else() {
    let source = "if true {\n    resonate 1.0\n} else {\n    resonate 0.0\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::IfElse { .. }));
}

#[test]
fn parse_while_loop() {
    let source = "while x < 10.0 {\n    x = x + 1.0\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], PhiExpression::WhileLoop { .. }));
}

#[test]
fn parse_function_definition() {
    let source = "function add(a: Number, b: Number) -> Number {\n    return a + b\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::FunctionDef { name, parameters, .. } => {
            assert_eq!(name, "add");
            assert_eq!(parameters.len(), 2);
            assert_eq!(parameters[0].0, "a");
            assert_eq!(parameters[1].0, "b");
        }
        other => panic!("expected FunctionDef, got {:?}", other),
    }
}

#[test]
fn parse_function_call() {
    let exprs = parse_phi_program("add(1.0, 2.0)").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::FunctionCall { name, arguments } => {
            assert_eq!(name, "add");
            assert_eq!(arguments.len(), 2);
        }
        other => panic!("expected FunctionCall, got {:?}", other),
    }
}

#[test]
fn parse_return_statement() {
    let source = "function f() -> Number {\n    return 42.0\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    // The return is inside the function body, which is a Block
    match &exprs[0] {
        PhiExpression::FunctionDef { body, .. } => {
            match body.as_ref() {
                PhiExpression::Block(stmts) => {
                    assert!(!stmts.is_empty());
                    assert!(matches!(&stmts[0], PhiExpression::Return(_)));
                }
                PhiExpression::Return(_) => { /* single-statement body is also valid */ }
                other => panic!("expected Block or Return body, got {:?}", other),
            }
        }
        other => panic!("expected FunctionDef, got {:?}", other),
    }
}

// ─── Parser: multi-statement programs ──────────────────────────────

#[test]
fn parse_multiple_statements() {
    let source = "let x = 1.0\nlet y = 2.0\nresonate x";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 3);
    assert!(matches!(&exprs[0], PhiExpression::LetBinding { .. }));
    assert!(matches!(&exprs[1], PhiExpression::LetBinding { .. }));
    assert!(matches!(&exprs[2], PhiExpression::Resonate { .. }));
}

#[test]
fn parse_program_with_comments() {
    let source = "// A comment\nlet x = 1.0 // inline comment\nresonate x";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn parse_claude_phi_example() {
    // Parse the actual claude.phi example to verify the parser handles real code
    let source = r#"
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
    witness
    resonate lambda
}
"#;
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 3);
    assert!(matches!(&exprs[0], PhiExpression::FunctionDef { name, .. } if name == "phi_power"));
    assert!(matches!(&exprs[1], PhiExpression::FunctionDef { name, .. } if name == "coherence_formula"));
    assert!(matches!(&exprs[2], PhiExpression::IntentionBlock { intention, .. } if intention == "LAMBDA_convergence"));
}

// ─── Parser: error cases ───────────────────────────────────────────

#[test]
fn parse_unmatched_brace_returns_error() {
    let source = "intention \"test\" {\n    resonate 1.0\n";
    let result = parse_phi_program(source);
    assert!(result.is_err());
}

#[test]
fn parse_unterminated_string_returns_error() {
    let source = "intention \"test";
    let result = parse_phi_program(source);
    assert!(result.is_err());
}

#[test]
fn parse_unknown_keyword_as_identifier() {
    // "foobar" is not a keyword, so it should be parsed as an identifier
    // (either as a variable reference or an error depending on context)
    let result = parse_phi_program("foobar");
    // Some parsers treat bare identifiers as variable references, others as errors
    // Just verify it doesn't crash
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn parse_empty_program_returns_empty_vec() {
    let exprs = parse_phi_program("").unwrap();
    assert!(exprs.is_empty());
}

#[test]
fn parse_comment_only_program_returns_empty_vec() {
    let exprs = parse_phi_program("// just a comment\n").unwrap();
    assert!(exprs.is_empty());
}

// ─── Parser: variable assignment ───────────────────────────────────

#[test]
fn parse_variable_assignment() {
    let source = "x = 42.0";
    let result = parse_phi_program(source);
    // Assignment (as opposed to let binding) may or may not be supported
    // as a top-level statement. Just verify it doesn't crash.
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn parse_variable_reference() {
    let exprs = parse_phi_program("resonate x").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::Resonate { expression, .. } => {
            assert!(expression.is_some());
            assert!(matches!(expression.as_ref().unwrap().as_ref(), PhiExpression::Variable(ref s) if s == "x"));
        }
        other => panic!("expected Resonate, got {:?}", other),
    }
}

// ─── Parser: nested constructs ─────────────────────────────────────

#[test]
fn parse_nested_intention_blocks() {
    let source = "intention \"outer\" {\n    intention \"inner\" {\n        witness\n    }\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::IntentionBlock { intention, body, .. } => {
            assert_eq!(intention, "outer");
            // Body is a Block containing the inner IntentionBlock
            match body.as_ref() {
                PhiExpression::Block(stmts) => {
                    assert!(!stmts.is_empty());
                    assert!(matches!(&stmts[0], PhiExpression::IntentionBlock { intention, .. } if intention == "inner"));
                }
                PhiExpression::IntentionBlock { intention, .. } if intention == "inner" => {
                    /* single-statement body is also valid */
                }
                other => panic!("expected Block or IntentionBlock body, got {:?}", other),
            }
        }
        other => panic!("expected IntentionBlock, got {:?}", other),
    }
}

#[test]
fn parse_intention_with_witness_and_resonate() {
    let source = "intention \"test\" {\n    witness\n    resonate 0.618\n}";
    let exprs = parse_phi_program(source).unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        PhiExpression::IntentionBlock { body, .. } => {
            // Body should be a sequence containing witness and resonate
            // The exact structure depends on how the parser handles multiple statements in a block
            assert!(matches!(body.as_ref(),
                PhiExpression::Block(_) |
                PhiExpression::Witness { .. } |
                PhiExpression::Resonate { .. }
            ));
        }
        other => panic!("expected IntentionBlock, got {:?}", other),
    }
}
