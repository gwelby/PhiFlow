// Predicted claims test suite — 2026-04-15
// Tests for: block comments, type annotations, import system

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

// ─── Claim: Block comments parse correctly ───

#[test]
fn test_block_comment_basic() {
    let input = r#"/* this is a comment */ let x = 42"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse past block comment");
}

#[test]
fn test_block_comment_inline() {
    let input = r#"let x = /* comment */ 42"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse inline block comment");
    match &expressions[0] {
        PhiExpression::LetBinding { name, .. } => {
            assert_eq!(name, "x");
        }
        other => panic!("expected LetBinding, got {:?}", other),
    }
}

#[test]
fn test_block_comment_multiline() {
    let input = r#"
/*
  This is a
  multi-line block comment
  that should be ignored
*/
let x = 10
let y = 20
"#;
    let expressions = parse_source(input);
    assert_eq!(expressions.len(), 2, "should parse two let statements after multiline comment, got {}", expressions.len());
}

#[test]
fn test_block_comment_doesnt_corrupt_tokens() {
    let input = r#"witness /* observe this */ x"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "block comment should not corrupt witness token");
}

#[test]
fn test_block_comment_around_resonate() {
    let input = r#"/* setup */ resonate 0.5"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "block comment before resonate should parse");
}

// ─── Claim: Type annotations work ───

#[test]
fn test_type_annotation_f64() {
    let input = r#"let x: f64 = 3.14"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation f64");
    match &expressions[0] {
        PhiExpression::LetBinding { phi_type, .. } => {
            assert!(phi_type.is_some(), "should have type annotation");
        }
        other => panic!("expected LetBinding, got {:?}", other),
    }
}

#[test]
fn test_type_annotation_i32() {
    let input = r#"let count: i32 = 42"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation i32");
}

#[test]
fn test_type_annotation_bool() {
    let input = r#"let flag: bool = true"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation bool");
}

#[test]
fn test_type_annotation_string() {
    let input = r#"let msg: string = "hello""#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation string");
}

#[test]
fn test_type_annotation_qubit() {
    let input = r#"let q: qubit = 0"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation qubit");
}

#[test]
fn test_type_annotation_circuit() {
    let input = r#"let circ: circuit = empty"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation circuit");
}

#[test]
fn test_type_annotation_consciousness() {
    let input = r#"let state: consciousness = init"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse type annotation consciousness");
}

#[test]
fn test_type_annotation_custom() {
    let input = r#"let x: MyType = value"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse custom type annotation");
}

#[test]
fn test_type_annotation_function_return() {
    let input = r#"
function compute() -> f64 {
    return 1.0
}
"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse function return type annotation");
}

#[test]
fn test_type_annotation_function_param() {
    let input = r#"
function greet(name: string) -> string {
    return name
}
"#;
    let expressions = parse_source(input);
    assert!(!expressions.is_empty(), "should parse function parameter type annotation");
}

#[test]
fn test_let_without_type_annotation() {
    // Verify that type annotations are optional
    let input = r#"let x = 42"#;
    let expressions = parse_source(input);
    match &expressions[0] {
        PhiExpression::LetBinding { phi_type, .. } => {
            assert!(phi_type.is_none(), "type annotation should be optional");
        }
        other => panic!("expected LetBinding, got {:?}", other),
    }
}

// ─── Claim: Import system works ───

#[test]
fn test_import_from_string_syntax() {
    // Test whether "import from \"file.phi\"" syntax is supported
    let input = r#"import from "helpers.phi""#;
    let result = std::panic::catch_unwind(|| parse_source(input));
    assert!(
        result.is_ok(),
        "import from syntax should parse without panic — if this fails, import is NOT supported"
    );
}

#[test]
fn test_import_multiple_files() {
    let input = r#"
import from "math.phi"
import from "utils.phi"
let x = 1
"#;
    let result = std::panic::catch_unwind(|| parse_source(input));
    assert!(
        result.is_ok(),
        "multiple imports should parse — if this fails, import is NOT supported"
    );
}
