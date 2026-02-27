//! AST → IR Lowering Pass
//!
//! Converts PhiExpression AST nodes into a flat sequence of IR opcodes.
//! This is Phase 1 lowering — covers core language constructs and
//! consciousness-aware operations. Hardware/quantum/bio nodes are
//! lowered to Stub opcodes for future backends.

use super::{FunctionDef, IrProgram, Label, Opcode};
use crate::parser::{BinaryOperator, PhiExpression, PhiValue, UnaryOperator};
use std::collections::HashMap;

/// The lowering context, tracking state during AST → IR conversion.
pub struct Lowering {
    program: IrProgram,
    active_stream_end: Option<Label>,
}

impl Lowering {
    pub fn new() -> Self {
        Lowering {
            program: IrProgram::new(),
            active_stream_end: None,
        }
    }

    /// Lower a complete program (a list of top-level AST expressions).
    pub fn lower_program(mut self, expressions: &[PhiExpression]) -> IrProgram {
        for expr in expressions {
            self.lower_to_main(expr);
        }
        self.program.emit(Opcode::Halt);
        self.program
    }

    /// Lower an expression, emitting opcodes to the main program body.
    fn lower_to_main(&mut self, expr: &PhiExpression) {
        let mut ops = Vec::new();
        self.lower_expression(expr, &mut ops);
        for op in ops {
            self.program.emit(op);
        }
    }

    /// Lower a single expression, appending opcodes to the given buffer.
    fn lower_expression(&mut self, expr: &PhiExpression, buffer: &mut Vec<Opcode>) {
        match expr {
            // ─── Literals ───────────────────────────────────
            PhiExpression::Number(n) => {
                self.emit(buffer, Opcode::PushNumber(*n));
            }

            PhiExpression::String(s) => {
                self.emit(buffer, Opcode::PushString(s.clone()));
            }

            PhiExpression::Boolean(b) => {
                self.emit(buffer, Opcode::PushBool(*b));
            }

            // ─── Variable binding and access ────────────────
            PhiExpression::LetBinding { name, value, .. } => {
                // Lower the value expression first (pushes result onto stack)
                self.lower_expression(value, buffer);
                // Store the result into the named variable
                self.emit(buffer, Opcode::Store(name.clone()));
            }

            PhiExpression::Variable(name) => {
                self.emit(buffer, Opcode::Load(name.clone()));
            }

            // ─── Binary operations ──────────────────────────
            PhiExpression::BinaryOp {
                left,
                operator,
                right,
            } => {
                // Push left operand, then right, then the operator
                self.lower_expression(left, buffer);
                self.lower_expression(right, buffer);
                let op = match operator {
                    BinaryOperator::Add => Opcode::Add,
                    BinaryOperator::Subtract => Opcode::Sub,
                    BinaryOperator::Multiply => Opcode::Mul,
                    BinaryOperator::Divide => Opcode::Div,
                    BinaryOperator::Modulo => Opcode::Mod,
                    BinaryOperator::Power => Opcode::Pow,
                    BinaryOperator::Equal => Opcode::Eq,
                    BinaryOperator::NotEqual => Opcode::Ne,
                    BinaryOperator::Less => Opcode::Lt,
                    BinaryOperator::LessEqual => Opcode::Le,
                    BinaryOperator::Greater => Opcode::Gt,
                    BinaryOperator::GreaterEqual => Opcode::Ge,
                    BinaryOperator::And => Opcode::And,
                    BinaryOperator::Or => Opcode::Or,
                };
                self.emit(buffer, op);
            }

            // ─── Unary operations ───────────────────────────
            PhiExpression::UnaryOp { operator, operand } => {
                self.lower_expression(operand, buffer);
                let op = match operator {
                    UnaryOperator::Negate => Opcode::Neg,
                    UnaryOperator::Not => Opcode::Not,
                };
                self.emit(buffer, op);
            }

            // ─── Block (sequence of expressions) ────────────
            PhiExpression::Block(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    self.lower_expression(e, buffer);
                    // Pop intermediate results, keep last
                    if i < exprs.len() - 1 {
                        self.emit(buffer, Opcode::Pop);
                    }
                }
                if exprs.is_empty() {
                    self.emit(buffer, Opcode::PushVoid);
                }
            }

            // ─── If/Else ────────────────────────────────────
            PhiExpression::IfElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let else_label = self.program.fresh_label();
                let end_label = self.program.fresh_label();

                // Evaluate condition
                self.lower_expression(condition, buffer);

                if else_branch.is_some() {
                    // Jump to else if false
                    self.emit(buffer, Opcode::JumpIfFalse(else_label));
                    // Then branch
                    self.lower_expression(then_branch, buffer);
                    self.emit(buffer, Opcode::Jump(end_label));
                    // Else branch
                    self.emit(buffer, Opcode::LabelMark(else_label));
                    self.lower_expression(else_branch.as_ref().unwrap(), buffer);
                    self.emit(buffer, Opcode::LabelMark(end_label));
                } else {
                    // No else: jump past then if false
                    self.emit(buffer, Opcode::JumpIfFalse(end_label));
                    self.lower_expression(then_branch, buffer);
                    self.emit(buffer, Opcode::Pop); // discard then result
                    self.emit(buffer, Opcode::LabelMark(end_label));
                }
            }

            // ─── While loop ─────────────────────────────────
            PhiExpression::WhileLoop { condition, body } => {
                let loop_start = self.program.fresh_label();
                let loop_end = self.program.fresh_label();

                self.emit(buffer, Opcode::LabelMark(loop_start));
                self.lower_expression(condition, buffer);
                self.emit(buffer, Opcode::JumpIfFalse(loop_end));
                self.lower_expression(body, buffer);
                self.emit(buffer, Opcode::Pop); // discard body result
                self.emit(buffer, Opcode::Jump(loop_start));
                self.emit(buffer, Opcode::LabelMark(loop_end));
                self.emit(buffer, Opcode::PushVoid); // loops produce void
            }

            // ─── For loop ───────────────────────────────────
            PhiExpression::ForLoop {
                variable,
                iterable,
                body,
            } => {
                let loop_start = self.program.fresh_label();
                let loop_end = self.program.fresh_label();

                // Evaluate the iterable (pushes a list onto stack)
                self.lower_expression(iterable, buffer);

                // Initialize: set up the for-loop state
                self.emit(
                    buffer,
                    Opcode::ForLoopInit {
                        variable: variable.clone(),
                        end_label: loop_end,
                    },
                );

                self.emit(buffer, Opcode::LabelMark(loop_start));

                // Check next iteration
                self.emit(
                    buffer,
                    Opcode::ForLoopNext {
                        variable: variable.clone(),
                        body_label: loop_start,
                        end_label: loop_end,
                    },
                );

                // Loop body
                self.lower_expression(body, buffer);
                self.emit(buffer, Opcode::Pop); // discard body result
                self.emit(buffer, Opcode::Jump(loop_start));

                self.emit(buffer, Opcode::LabelMark(loop_end));
                self.emit(buffer, Opcode::PushVoid);
            }

            // ─── Return ─────────────────────────────────────
            PhiExpression::Return(expr) => {
                self.lower_expression(expr, buffer);
                self.emit(buffer, Opcode::Return);
            }

            // ─── Lists ──────────────────────────────────────
            PhiExpression::List(items) => {
                let count = items.len();
                for item in items {
                    self.lower_expression(item, buffer);
                }
                self.emit(buffer, Opcode::MakeList(count));
            }

            PhiExpression::ListAccess { list, index } => {
                self.lower_expression(list, buffer);
                self.lower_expression(index, buffer);
                self.emit(buffer, Opcode::ListAccess);
            }

            // ─── Function definition ────────────────────────
            PhiExpression::FunctionDef {
                name,
                parameters,
                body,
                ..
            } => {
                let params: Vec<String> = parameters.iter().map(|(n, _)| n.clone()).collect();

                // Lower the function body into a separate instruction buffer
                let mut func_buffer = Vec::new();
                self.lower_expression(body, &mut func_buffer);
                func_buffer.push(Opcode::Return);

                let body_label = self.program.fresh_label();

                // Register the function
                self.program.functions.insert(
                    name.clone(),
                    FunctionDef {
                        name: name.clone(),
                        params: params.clone(),
                        body: func_buffer,
                    },
                );

                // Emit the definition marker in the main stream
                self.emit(
                    buffer,
                    Opcode::DefineFunction {
                        name: name.clone(),
                        params,
                        body_label,
                    },
                );
            }

            // ─── Function call ──────────────────────────────
            PhiExpression::FunctionCall { name, arguments } => {
                let arg_count = arguments.len();

                // Handle built-in print
                if name == "print" || name == "println" {
                    for arg in arguments {
                        self.lower_expression(arg, buffer);
                        self.emit(buffer, Opcode::Print);
                    }
                    self.emit(buffer, Opcode::PushVoid);
                    return;
                }

                // Push arguments left-to-right
                for arg in arguments {
                    self.lower_expression(arg, buffer);
                }
                self.emit(
                    buffer,
                    Opcode::Call {
                        name: name.clone(),
                        arg_count,
                    },
                );
            }

            // ═════════════════════════════════════════════════
            // CONSCIOUSNESS CONSTRUCTS — The soul of PhiFlow
            // ═════════════════════════════════════════════════
            PhiExpression::Witness { expression, body } => {
                let has_expression = expression.is_some();
                let has_body = body.is_some();

                // If witnessing a specific expression, lower it first
                if let Some(expr) = expression {
                    self.lower_expression(expr, buffer);
                }

                self.emit(
                    buffer,
                    Opcode::Witness {
                        has_expression,
                        has_body,
                    },
                );

                // If there's a body to execute after witnessing
                if let Some(body_expr) = body {
                    self.lower_expression(body_expr, buffer);
                    self.emit(buffer, Opcode::WitnessEnd);
                }
            }

            PhiExpression::IntentionBlock { intention, body } => {
                self.emit(buffer, Opcode::IntentionPush(intention.clone()));
                self.lower_expression(body, buffer);
                self.emit(buffer, Opcode::IntentionPop);
            }

            PhiExpression::Resonate { expression } => {
                let has_expression = expression.is_some();
                if let Some(expr) = expression {
                    self.lower_expression(expr, buffer);
                }
                self.emit(buffer, Opcode::Resonate { has_expression });
            }

            // ─── Pattern creation ───────────────────────────
            PhiExpression::CreatePattern {
                pattern_type,
                frequency,
                ..
            } => {
                self.emit(
                    buffer,
                    Opcode::CreatePattern {
                        pattern_type: pattern_type.clone(),
                        frequency: *frequency,
                    },
                );
            }

            // ─── Consciousness validation ───────────────────
            PhiExpression::ConsciousnessValidation { pattern, metrics } => {
                self.lower_expression(pattern, buffer);
                self.emit(
                    buffer,
                    Opcode::ValidatePattern {
                        metrics: metrics.clone(),
                    },
                );
            }

            // ═════════════════════════════════════════════════
            // STUBS — Future hardware/quantum/bio backends
            // These are lowered as Stub opcodes to preserve info
            // ═════════════════════════════════════════════════
            PhiExpression::ConsciousnessState {
                state,
                coherence,
                frequency,
            } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "ConsciousnessState".to_string(),
                        description: format!(
                            "state={}, coherence={}, freq={}",
                            state, coherence, frequency
                        ),
                    },
                );
            }

            PhiExpression::FrequencyPattern {
                base_frequency,
                harmonics,
                phi_scaling,
            } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "FrequencyPattern".to_string(),
                        description: format!(
                            "base={}Hz, harmonics={}, phi={}",
                            base_frequency,
                            harmonics.len(),
                            phi_scaling
                        ),
                    },
                );
            }

            PhiExpression::QuantumField {
                field_type,
                dimensions,
                coherence_target,
            } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "QuantumField".to_string(),
                        description: format!(
                            "type={}, dims={:?}, target={}",
                            field_type, dimensions, coherence_target
                        ),
                    },
                );
            }

            PhiExpression::BiologicalInterface {
                target,
                transduction_method,
                frequency,
            } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "BiologicalInterface".to_string(),
                        description: format!(
                            "target={}, method={}, freq={}",
                            target, transduction_method, frequency
                        ),
                    },
                );
            }

            PhiExpression::HardwareSync { device_type, .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "HardwareSync".to_string(),
                        description: format!("device={}", device_type),
                    },
                );
            }

            PhiExpression::ConsciousnessFlow { .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "ConsciousnessFlow".to_string(),
                        description: "branching on consciousness state".to_string(),
                    },
                );
            }

            PhiExpression::EmergencyProtocol { .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "EmergencyProtocol".to_string(),
                        description: "emergency interrupt handler".to_string(),
                    },
                );
            }

            PhiExpression::AudioSynthesis { audio_type, .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "AudioSynthesis".to_string(),
                        description: format!("type={}", audio_type),
                    },
                );
            }

            PhiExpression::ConsciousnessMonitor { .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "ConsciousnessMonitor".to_string(),
                        description: "real-time coherence monitoring".to_string(),
                    },
                );
            }

            PhiExpression::PatternTransform { transform_type, .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "PatternTransform".to_string(),
                        description: format!("transform={}", transform_type),
                    },
                );
            }

            PhiExpression::PatternCombine { combine_type, .. } => {
                self.emit(
                    buffer,
                    Opcode::Stub {
                        node_type: "PatternCombine".to_string(),
                        description: format!("combine={}", combine_type),
                    },
                );
            }

            PhiExpression::StreamBlock { name, body } => {
                let loop_start = self.program.fresh_label();
                let loop_end = self.program.fresh_label();

                // Setup the stream environment
                self.emit(
                    buffer,
                    Opcode::StreamInit {
                        name: name.clone(),
                        end_label: loop_end,
                    },
                );

                self.emit(buffer, Opcode::LabelMark(loop_start));

                // Process stream cycle / termination criteria
                self.emit(
                    buffer,
                    Opcode::StreamNext {
                        name: name.clone(),
                        body_label: loop_start,
                        end_label: loop_end,
                    },
                );

                // Body of the stream loop
                // Push the end label onto the break stack conceptually, but handled via AST logic for break statement routing
                // For simplicity, we assume `break stream` is resolved dynamically via a stack/VM lookup during evaluation,
                // or we can pass a context label down. In this lowering phase, we'll store the currently active stream context
                // in the `Lowering` struct if we needed to inject `end_label` into `BreakStream`.

                // We temporarily overwrite an active stream end label in our struct context so breaks can bind.
                let prev_end = self.active_stream_end;
                self.active_stream_end = Some(loop_end);

                self.lower_expression(body, buffer);

                self.active_stream_end = prev_end;

                self.emit(buffer, Opcode::Pop); // discard body result
                self.emit(buffer, Opcode::Jump(loop_start));

                self.emit(buffer, Opcode::LabelMark(loop_end));
                self.emit(buffer, Opcode::PushVoid);
            }

            PhiExpression::BreakStream => {
                if let Some(end_label) = self.active_stream_end {
                    self.emit(buffer, Opcode::StreamBreak { end_label });
                } else {
                    // Compilation error conceptually, but emit halt or stub for now
                    self.emit(
                        buffer,
                        Opcode::Stub {
                            node_type: "BreakStream Error".to_string(),
                            description: "break stream outside of stream block".to_string(),
                        },
                    );
                }
            }
        }
    }

    /// Emit an opcode to the buffer.
    fn emit(&mut self, buffer: &mut Vec<Opcode>, op: Opcode) {
        buffer.push(op);
    }
}

impl Default for Lowering {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: lower a parsed program to IR in one call.
pub fn lower(expressions: &[PhiExpression]) -> IrProgram {
    Lowering::new().lower_program(expressions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_number() {
        let ast = vec![PhiExpression::Number(42.0)];
        let ir = lower(&ast);
        assert_eq!(ir.instructions.len(), 2); // PushNumber + Halt
        assert_eq!(ir.instructions[0], Opcode::PushNumber(42.0));
        assert_eq!(ir.instructions[1], Opcode::Halt);
    }

    #[test]
    fn test_lower_let_binding() {
        let ast = vec![PhiExpression::LetBinding {
            name: "x".to_string(),
            value: Box::new(PhiExpression::Number(432.0)),
            phi_type: None,
        }];
        let ir = lower(&ast);
        assert_eq!(ir.instructions[0], Opcode::PushNumber(432.0));
        assert_eq!(ir.instructions[1], Opcode::Store("x".to_string()));
        assert_eq!(ir.instructions[2], Opcode::Halt);
    }

    #[test]
    fn test_lower_binary_add() {
        let ast = vec![PhiExpression::BinaryOp {
            left: Box::new(PhiExpression::Number(432.0)),
            operator: BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(528.0)),
        }];
        let ir = lower(&ast);
        assert_eq!(ir.instructions[0], Opcode::PushNumber(432.0));
        assert_eq!(ir.instructions[1], Opcode::PushNumber(528.0));
        assert_eq!(ir.instructions[2], Opcode::Add);
    }

    #[test]
    fn test_lower_witness() {
        let ast = vec![PhiExpression::Witness {
            expression: Some(Box::new(PhiExpression::Number(528.0))),
            body: None,
        }];
        let ir = lower(&ast);
        assert_eq!(ir.instructions[0], Opcode::PushNumber(528.0));
        assert_eq!(
            ir.instructions[1],
            Opcode::Witness {
                has_expression: true,
                has_body: false,
            }
        );
    }

    #[test]
    fn test_lower_intention_block() {
        let ast = vec![PhiExpression::IntentionBlock {
            intention: "healing".to_string(),
            body: Box::new(PhiExpression::Number(432.0)),
        }];
        let ir = lower(&ast);
        assert_eq!(
            ir.instructions[0],
            Opcode::IntentionPush("healing".to_string())
        );
        assert_eq!(ir.instructions[1], Opcode::PushNumber(432.0));
        assert_eq!(ir.instructions[2], Opcode::IntentionPop);
    }

    #[test]
    fn test_lower_resonate() {
        let ast = vec![PhiExpression::Resonate {
            expression: Some(Box::new(PhiExpression::Number(528.0))),
        }];
        let ir = lower(&ast);
        assert_eq!(ir.instructions[0], Opcode::PushNumber(528.0));
        assert_eq!(
            ir.instructions[1],
            Opcode::Resonate {
                has_expression: true
            }
        );
    }

    #[test]
    fn test_lower_if_else() {
        let ast = vec![PhiExpression::IfElse {
            condition: Box::new(PhiExpression::Boolean(true)),
            then_branch: Box::new(PhiExpression::Number(1.0)),
            else_branch: Some(Box::new(PhiExpression::Number(2.0))),
        }];
        let ir = lower(&ast);
        // Should have: PushBool, JumpIfFalse, PushNumber(1), Jump, LabelMark, PushNumber(2), LabelMark, Halt
        assert!(ir.instructions.len() >= 7);
        assert_eq!(ir.instructions[0], Opcode::PushBool(true));
    }
}
