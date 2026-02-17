//! IR Pretty Printer
//!
//! Formats PhiFlow IR instructions for human-readable debug output.
//! Useful for inspecting the lowering pass and verifying correctness.

use super::{IrProgram, Opcode};
use std::fmt::Write;

/// Format an entire IR program as a human-readable string.
pub fn print_program(program: &IrProgram) -> String {
    let mut output = String::new();

    writeln!(output, "╔══════════════════════════════════════════╗").unwrap();
    writeln!(output, "║       PhiFlow IR Program Listing         ║").unwrap();
    writeln!(output, "╠══════════════════════════════════════════╣").unwrap();
    writeln!(
        output,
        "║ Total instructions: {:>20} ║",
        program.total_instructions()
    )
    .unwrap();
    writeln!(
        output,
        "║ Functions defined:  {:>20} ║",
        program.functions.len()
    )
    .unwrap();
    writeln!(output, "║ Labels allocated:   {:>20} ║", program.next_label).unwrap();
    writeln!(output, "╚══════════════════════════════════════════╝").unwrap();
    writeln!(output).unwrap();

    // Main program body
    writeln!(output, "── MAIN ──────────────────────────────────").unwrap();
    for (i, op) in program.instructions.iter().enumerate() {
        writeln!(output, "  {:>4}: {}", i, format_opcode(op)).unwrap();
    }
    writeln!(output).unwrap();

    // Function bodies
    for (name, func) in &program.functions {
        writeln!(output, "── FN {} ({}) ──", name, func.params.join(", ")).unwrap();
        for (i, op) in func.body.iter().enumerate() {
            writeln!(output, "  {:>4}: {}", i, format_opcode(op)).unwrap();
        }
        writeln!(output).unwrap();
    }

    output
}

/// Format a single opcode as a human-readable string.
pub fn format_opcode(op: &Opcode) -> String {
    match op {
        // Literals
        Opcode::PushNumber(n) => format!("PUSH_NUM    {}", n),
        Opcode::PushString(s) => format!("PUSH_STR    \"{}\"", s),
        Opcode::PushBool(b) => format!("PUSH_BOOL   {}", b),
        Opcode::PushVoid => "PUSH_VOID".to_string(),

        // Variables
        Opcode::Store(name) => format!("STORE       ${}", name),
        Opcode::Load(name) => format!("LOAD        ${}", name),

        // Arithmetic
        Opcode::Add => "ADD".to_string(),
        Opcode::Sub => "SUB".to_string(),
        Opcode::Mul => "MUL".to_string(),
        Opcode::Div => "DIV".to_string(),
        Opcode::Mod => "MOD".to_string(),
        Opcode::Pow => "POW".to_string(),

        // Comparison
        Opcode::Eq => "EQ".to_string(),
        Opcode::Ne => "NE".to_string(),
        Opcode::Lt => "LT".to_string(),
        Opcode::Le => "LE".to_string(),
        Opcode::Gt => "GT".to_string(),
        Opcode::Ge => "GE".to_string(),

        // Logical
        Opcode::And => "AND".to_string(),
        Opcode::Or => "OR".to_string(),
        Opcode::Not => "NOT".to_string(),
        Opcode::Neg => "NEG".to_string(),

        // Control flow
        Opcode::Jump(label) => format!("JUMP        L{}", label),
        Opcode::JumpIfTrue(label) => format!("JUMP_TRUE   L{}", label),
        Opcode::JumpIfFalse(label) => format!("JUMP_FALSE  L{}", label),
        Opcode::LabelMark(label) => format!("L{}:", label),

        // Functions
        Opcode::DefineFunction {
            name,
            params,
            body_label,
        } => {
            format!(
                "DEF_FN      {}({}) → L{}",
                name,
                params.join(", "),
                body_label
            )
        }
        Opcode::Call { name, arg_count } => {
            format!("CALL        {}({})", name, arg_count)
        }
        Opcode::Return => "RETURN".to_string(),

        // Lists
        Opcode::MakeList(n) => format!("MAKE_LIST   [{}]", n),
        Opcode::ListAccess => "LIST_GET".to_string(),

        // I/O
        Opcode::Print => "PRINT".to_string(),
        Opcode::Pop => "POP".to_string(),

        // ═══ CONSCIOUSNESS OPCODES ═══
        Opcode::Witness {
            has_expression,
            has_body,
        } => {
            let expr = if *has_expression { "expr" } else { "all" };
            let body = if *has_body { "+body" } else { "" };
            format!("◇ WITNESS   ({}){}  ← program observes itself", expr, body)
        }
        Opcode::WitnessEnd => "◇ WITNESS_END".to_string(),

        Opcode::IntentionPush(intention) => {
            format!("◈ INTENT_PUSH \"{}\"  ← declare purpose", intention)
        }
        Opcode::IntentionPop => "◈ INTENT_POP".to_string(),

        Opcode::Resonate { has_expression } => {
            let what = if *has_expression {
                "value"
            } else {
                "coherence"
            };
            format!("◎ RESONATE   ({})  ← share to field", what)
        }

        Opcode::Coherence => "◉ COHERENCE  ← measure alignment".to_string(),
        Opcode::FrequencyCheck => "♫ FREQ_CHECK ← validate sacred frequency".to_string(),

        // Patterns
        Opcode::CreatePattern {
            pattern_type,
            frequency,
        } => {
            format!("CREATE_PAT  {} @{}Hz", pattern_type, frequency)
        }
        Opcode::ValidatePattern { metrics } => {
            format!("VALIDATE    [{}]", metrics.join(", "))
        }

        // Loops
        Opcode::ForLoopInit {
            variable,
            end_label,
        } => {
            format!("FOR_INIT    ${} → L{}", variable, end_label)
        }
        Opcode::ForLoopNext {
            variable,
            body_label,
            end_label,
        } => {
            format!(
                "FOR_NEXT    ${} body=L{} end=L{}",
                variable, body_label, end_label
            )
        }

        // Stubs
        Opcode::Stub {
            node_type,
            description,
        } => {
            format!("⚠ STUB       {} ({})", node_type, description)
        }

        Opcode::Halt => "HALT".to_string(),
    }
}
