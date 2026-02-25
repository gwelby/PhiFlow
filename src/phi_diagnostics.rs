use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhiDiagnostic {
    /// Short machine-readable code, e.g. "E001_UNEXPECTED_TOKEN"
    pub error_code: String,
    /// 1-based line number. 0 = unknown.
    pub line: usize,
    /// 1-based column number. 0 = unknown.
    pub column: usize,
    /// What the parser actually found (token name or char).
    pub found: String,
    /// What the parser expected (None if no single expectation).
    pub expected: Option<String>,
    /// One-sentence human hint. Must not repeat `found`/`expected` verbatim.
    pub hint: String,
    /// Minimal valid .phi snippet that fixes the problem.
    pub example_fix: String,
}

impl std::fmt::Display for PhiDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] line {}:{} — found `{}` {}— {}\n  fix: {}",
            self.error_code,
            self.line,
            self.column,
            self.found,
            self.expected
                .as_ref()
                .map(|e| format!("(expected `{}`) ", e))
                .unwrap_or_default(),
            self.hint,
            self.example_fix,
        )
    }
}
