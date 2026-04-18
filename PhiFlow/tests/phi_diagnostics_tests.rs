use phiflow::parser::parse_phi_program_with_diagnostics;

#[test]
fn test_e001_unexpected_token_in_intention() {
    let src = r#"
intention "x" {
    state something
}
"#;
    let err = parse_phi_program_with_diagnostics(src).unwrap_err();
    assert_eq!(err.error_code, "E001_UNEXPECTED_TOKEN", "Got: {:?}", err);
    assert!(!err.hint.is_empty(), "hint must be non-empty");
    assert!(!err.example_fix.is_empty(), "example_fix must be non-empty");
}

#[test]
fn test_e002_unexpected_eof_unclosed_intention() {
    let src = r#"intention "x" {"#;
    let err = parse_phi_program_with_diagnostics(src).unwrap_err();
    assert_eq!(err.error_code, "E002_UNEXPECTED_EOF", "Got: {:?}", err);
}

#[test]
fn test_e003_expected_token_missing_colon_in_param() {
    let src = r#"
function f(x Number) -> Number {
    return x
}
"#;
    let err = parse_phi_program_with_diagnostics(src).unwrap_err();
    assert_eq!(err.error_code, "E003_EXPECTED_TOKEN", "Got: {:?}", err);
    assert!(err.expected.is_some(), "E003 must populate expected field");
}

#[test]
fn test_e004_unexpected_char() {
    let src = "let x = 5 @ 3";
    let err = parse_phi_program_with_diagnostics(src).unwrap_err();
    assert_eq!(err.error_code, "E004_UNEXPECTED_CHAR", "Got: {:?}", err);
    assert!(err.line > 0, "E004 must have a line number (got 0)");
}

#[test]
fn test_display_format_contains_error_code() {
    let src = r#"intention "x" {"#;
    let err = parse_phi_program_with_diagnostics(src).unwrap_err();
    let display = format!("{}", err);
    assert!(
        display.contains("E00"),
        "Display must contain error code; got: {}",
        display
    );
    assert!(
        display.contains("fix:"),
        "Display must contain 'fix:'; got: {}",
        display
    );
}
