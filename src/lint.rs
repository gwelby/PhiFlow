//! Static warnings for PhiFlow programs. Warnings never stop a program; `phic`
//! prints them to stderr before running.
//!
//! Current checks:
//! - **stream-reset** (QSOP/PATTERNS.md P-3): a stream body re-runs from its
//!   first line every tick, so a variable set to a constant at the top of the
//!   stream body and then updated from its own value (`count = count + 1`)
//!   resets on every tick. A stop condition like `if count > 100 { break stream }`
//!   then never fires. Updates inside a `for`/`while` loop in the stream body
//!   are per-tick accumulation and are not flagged.

use crate::parser::PhiExpression;
use std::collections::HashSet;

/// One static warning.
#[derive(Debug, Clone, PartialEq)]
pub struct LintWarning {
    pub check: &'static str,
    pub message: String,
}

impl std::fmt::Display for LintWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "warning[{}]: {}", self.check, self.message)
    }
}

/// Run every check over a parsed program.
pub fn lint_program(program: &[PhiExpression]) -> Vec<LintWarning> {
    let mut out = Vec::new();
    for expr in program {
        visit(expr, &mut out);
    }
    out
}

fn visit(expr: &PhiExpression, out: &mut Vec<LintWarning>) {
    if let PhiExpression::StreamBlock { name, body } = expr {
        check_stream_reset(name, body, out);
    }
    for child in children(expr) {
        visit(child, out);
    }
}

/// Structural children (statement bodies), not every sub-expression.
fn children(expr: &PhiExpression) -> Vec<&PhiExpression> {
    match expr {
        PhiExpression::Block(stmts) => stmts.iter().collect(),
        PhiExpression::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            let mut v = vec![then_branch.as_ref()];
            if let Some(e) = else_branch {
                v.push(e.as_ref());
            }
            v
        }
        PhiExpression::ForLoop { body, .. }
        | PhiExpression::WhileLoop { body, .. }
        | PhiExpression::IntentionBlock { body, .. }
        | PhiExpression::StreamBlock { body, .. }
        | PhiExpression::FunctionDef { body, .. }
        | PhiExpression::AgentBlock { body, .. } => vec![body.as_ref()],
        PhiExpression::Witness { body: Some(b), .. } => vec![b.as_ref()],
        _ => Vec::new(),
    }
}

fn top_level(body: &PhiExpression) -> Vec<&PhiExpression> {
    match body {
        PhiExpression::Block(stmts) => stmts.iter().collect(),
        other => vec![other],
    }
}

fn is_constant(expr: &PhiExpression) -> bool {
    match expr {
        PhiExpression::Number(_) | PhiExpression::Boolean(_) | PhiExpression::String(_) => true,
        PhiExpression::UnaryOp { operand, .. } => is_constant(operand),
        _ => false,
    }
}

fn references(expr: &PhiExpression, name: &str) -> bool {
    match expr {
        PhiExpression::Variable(v) => v == name,
        PhiExpression::BinaryOp { left, right, .. } => {
            references(left, name) || references(right, name)
        }
        PhiExpression::UnaryOp { operand, .. } => references(operand, name),
        PhiExpression::FunctionCall { arguments, .. } => {
            arguments.iter().any(|a| references(a, name))
        }
        PhiExpression::List(items) => items.iter().any(|a| references(a, name)),
        PhiExpression::ListAccess { list, index } => {
            references(list, name) || references(index, name)
        }
        _ => false,
    }
}

fn check_stream_reset(stream: &str, body: &PhiExpression, out: &mut Vec<LintWarning>) {
    let mut const_init: Vec<&str> = Vec::new();
    for stmt in top_level(body) {
        if let PhiExpression::LetBinding { name, value, .. } = stmt {
            if is_constant(value) && !const_init.contains(&name.as_str()) {
                const_init.push(name);
            }
        }
    }
    if const_init.is_empty() {
        return;
    }
    let mut flagged: HashSet<String> = HashSet::new();
    find_self_updates(body, &const_init, &mut flagged);
    for name in const_init {
        if flagged.contains(name) {
            out.push(LintWarning {
                check: "stream-reset",
                message: format!(
                    "in stream \"{stream}\", `{name}` is set to a constant at the top of the \
                     stream body and later updated from its own value. The stream body re-runs \
                     from its first line every tick, so `{name}` resets each tick and conditions \
                     on it may never fire. Move the initial binding above the stream \
                     (QSOP/PATTERNS.md P-3)."
                ),
            });
        }
    }
}

/// Self-updates of a tracked name, skipping per-tick loops and nested
/// streams/functions (they have their own scope rules).
fn find_self_updates(expr: &PhiExpression, tracked: &[&str], flagged: &mut HashSet<String>) {
    match expr {
        PhiExpression::LetBinding { name, value, .. } => {
            if tracked.contains(&name.as_str()) && references(value, name) {
                flagged.insert(name.clone());
            }
        }
        PhiExpression::ForLoop { .. }
        | PhiExpression::WhileLoop { .. }
        | PhiExpression::StreamBlock { .. }
        | PhiExpression::FunctionDef { .. } => {}
        PhiExpression::Block(stmts) => {
            for s in stmts {
                find_self_updates(s, tracked, flagged);
            }
        }
        other => {
            for c in children(other) {
                find_self_updates(c, tracked, flagged);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_phi_program;

    fn lint(src: &str) -> Vec<LintWarning> {
        let ast = parse_phi_program(src).expect("parses");
        lint_program(&ast)
    }

    #[test]
    fn flags_counter_reset_inside_stream() {
        let w = lint(
            "stream \"s\" {\n    let count = 0.0\n    count = count + 1.0\n    if count > 3.0 {\n        break stream\n    }\n}\n",
        );
        assert_eq!(w.len(), 1, "{:?}", w);
        assert_eq!(w[0].check, "stream-reset");
        assert!(w[0].message.contains("`count`"));
    }

    #[test]
    fn counter_above_stream_is_clean() {
        let w = lint(
            "let count = 0.0\nstream \"s\" {\n    count = count + 1.0\n    if count > 3.0 {\n        break stream\n    }\n}\n",
        );
        assert!(w.is_empty(), "{:?}", w);
    }

    #[test]
    fn per_tick_loop_accumulator_is_clean() {
        let w = lint(
            "stream \"s\" {\n    let acc = 0.0\n    let i = 0.0\n    while i < 3.0 {\n        acc = acc + 1.0\n        i = i + 1.0\n    }\n    break stream\n}\n",
        );
        assert!(w.is_empty(), "{:?}", w);
    }
}
