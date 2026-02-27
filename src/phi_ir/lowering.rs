//! AST to PhiIR Lowering
//!
//! Converts `PhiExpression` (AST) into `PhiIRProgram` (SSA-like IR).
//!
//! This module is responsible for:
//! 1. Traversing the AST.
//! 2. Generating SSA variables (Operands).
//! 3. Constructing BasicBlocks.
//! 4. Handling Control Flow (Branch, Jump).
//! 5. Maintaining scope for variables.

use crate::parser::{BinaryOperator, PhiExpression, UnaryOperator};
use crate::phi_ir::{
    BlockId, CollapsePolicy, DomainOp, Operand, PatternKind, PhiIRBinOp, PhiIRBlock, PhiIRNode,
    PhiIRProgram, PhiIRUnOp, PhiIRValue, PhiInstruction, SacredFrequency, Param,
};
use std::collections::HashMap;

/// Result of lowering an expression: either a Value (Operand) or None (Void/Statement).
#[derive(Debug, Clone, Copy, PartialEq)]
enum LowerResult {
    Value(Operand),
    None,
}

struct LoweringContext {
    program: PhiIRProgram,
    current_block: BlockId,
    /// Counter for generating instruction indices (Operands)
    next_operand: u32,
    /// Counter for generating BlockIds
    next_block_id: u32,
    /// Map of variable name to the most recent Operand holding its value.
    /// In a real SSA pass, this would handle Phi nodes.
    /// For this linear lowering, we'll simple-load declarations.
    /// Actually, since we have `LoadVar` / `StoreVar` nodes in `PhiIR`,
    /// we can let the backend handle register allocation/Phi nodes.
    /// So we don't strictly *need* to map vars to operands here,
    /// unless we are optimizing out `LoadVar`.
    /// We will emit `LoadVar` / `StoreVar` for simplicity and "True IR" behavior.
    scope: HashMap<String, Operand>,
    /// Track nested stream exit blocks for `break stream`
    stream_exits: Vec<BlockId>,
}

impl LoweringContext {
    fn new() -> Self {
        let entry = 0;
        let mut program = PhiIRProgram::new();

        // Create entry block
        program.blocks.push(PhiIRBlock {
            id: entry,
            label: "entry".to_string(),
            instructions: Vec::new(),
            terminator: PhiIRNode::Fallthrough, // Temporary
        });

        LoweringContext {
            program,
            current_block: entry,
            next_operand: 0,
            next_block_id: 1, // 0 is entry
            scope: HashMap::new(),
            stream_exits: Vec::new(),
        }
    }

    fn new_block(&mut self, label: &str) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.program.blocks.push(PhiIRBlock {
            id,
            label: label.to_string(),
            instructions: Vec::new(),
            terminator: PhiIRNode::Fallthrough,
        });
        id
    }

    fn set_current_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    fn emit(&mut self, node: PhiIRNode) -> Operand {
        // Determine if this node produces a value (SSA def)
        let produces_value = matches!(
            node,
            PhiIRNode::Const(_)
                | PhiIRNode::LoadVar(_)
                | PhiIRNode::BinOp { .. }
                | PhiIRNode::UnaryOp { .. }
                | PhiIRNode::Call { .. }
                | PhiIRNode::ListNew(_)
                | PhiIRNode::ListGet { .. }
                | PhiIRNode::CreatePattern { .. }
                | PhiIRNode::DomainCall { .. }
                | PhiIRNode::Witness { .. }     // returns coherence score (0.0–1.0)
                | PhiIRNode::CoherenceCheck // returns coherence score (0.0–1.0)
        );

        let (result_op, ret_op) = if produces_value {
            let op = self.next_operand;
            self.next_operand += 1;
            (Some(op), op)
        } else {
            (None, 0) // Dummy operand for non-value nodes
        };

        if let Some(block) = self
            .program
            .blocks
            .iter_mut()
            .find(|b| b.id == self.current_block)
        {
            block.instructions.push(PhiInstruction {
                result: result_op,
                node,
            });
        } else {
            panic!("Current block {} not found", self.current_block);
        }

        ret_op
    }

    fn terminate(&mut self, node: PhiIRNode) {
        if let Some(block) = self
            .program
            .blocks
            .iter_mut()
            .find(|b| b.id == self.current_block)
        {
            block.terminator = node;
        }
    }

    // Helper to get block by ID to check if it's already terminated (not Fallthrough)
    fn is_terminated(&self, block_id: BlockId) -> bool {
        if let Some(block) = self.program.blocks.iter().find(|b| b.id == block_id) {
            !matches!(block.terminator, PhiIRNode::Fallthrough)
        } else {
            false
        }
    }
}

pub fn lower_program(expressions: &[PhiExpression]) -> PhiIRProgram {
    let mut ctx = LoweringContext::new();

    let mut last_result = LowerResult::None;

    for expr in expressions {
        last_result = lower_expr(&mut ctx, expr);
    }

    // Ensure final block terminates
    if !ctx.is_terminated(ctx.current_block) {
        let ret_val = match last_result {
            LowerResult::Value(op) => op,
            LowerResult::None => 0,
        };
        ctx.terminate(PhiIRNode::Return(ret_val));
        // Or just Fallthrough if it's the end?
        // Let's use Return(Void) equivalent.
        // We'll emit a Const Void and Return it if needed.
        // For now, let's leave as Fallthrough implies end of program?
        // IR_DESIGN says "Terminator".
    }

    ctx.program
}

fn lower_expr(ctx: &mut LoweringContext, expr: &PhiExpression) -> LowerResult {
    match expr {
        PhiExpression::Number(n) => {
            let op = ctx.emit(PhiIRNode::Const(PhiIRValue::Number(*n)));
            LowerResult::Value(op)
        }
        PhiExpression::String(s) => {
            // Emulate string intern or just create constant
            // In real lower, we'd add to string_table
            // For now, let's use string index 0 placeholder or improve `PhiIRValue`
            // `PhiIRValue::String` takes u32 index.
            // Let's verify PhiIRValue definition in mod.rs.
            // It is `String(u32)`.
            // We need to add to string_table.
            ctx.program.string_table.push(s.clone());
            let idx = (ctx.program.string_table.len() - 1) as u32;
            let op = ctx.emit(PhiIRNode::Const(PhiIRValue::String(idx)));
            LowerResult::Value(op)
        }
        PhiExpression::Boolean(b) => {
            let op = ctx.emit(PhiIRNode::Const(PhiIRValue::Boolean(*b)));
            LowerResult::Value(op)
        }

        // Variables
        PhiExpression::Variable(name) => {
            // Disambiguation rule:
            // - `coherence` with no in-scope binding is the language keyword -> CoherenceCheck.
            // - If user explicitly bound a variable named `coherence`, treat it like any variable.
            let op = if name == "coherence" && !ctx.scope.contains_key(name) {
                ctx.emit(PhiIRNode::CoherenceCheck)
            } else {
                ctx.emit(PhiIRNode::LoadVar(name.clone()))
            };
            LowerResult::Value(op)
        }
        PhiExpression::LetBinding { name, value, .. } => {
            let val = match lower_expr(ctx, value) {
                LowerResult::Value(v) => v,
                LowerResult::None => {
                    // emit Void?
                    ctx.emit(PhiIRNode::Const(PhiIRValue::Void))
                }
            };
            ctx.emit(PhiIRNode::StoreVar {
                name: name.clone(),
                value: val,
            });
            ctx.scope.insert(name.clone(), val);
            LowerResult::None
        }

        // Function definitions and calls
        PhiExpression::FunctionDef {
            name,
            parameters,
            body,
            ..
        } => {
            let body_block = ctx.new_block(&format!("fn_{}_entry", name));
            let params: Vec<Param> = parameters
                .iter()
                .map(|(n, _)| Param { name: n.clone() })
                .collect();

            // Register function metadata in the current flow.
            ctx.emit(PhiIRNode::FuncDef {
                name: name.clone(),
                params: params.clone(),
                body: body_block,
            });

            // Lower function body in its own block.
            let saved_block = ctx.current_block;
            ctx.set_current_block(body_block);
            let body_result = lower_expr(ctx, body);
            if !ctx.is_terminated(ctx.current_block) {
                let ret = unwrap_val(ctx, body_result);
                ctx.terminate(PhiIRNode::Return(ret));
            }
            ctx.set_current_block(saved_block);

            LowerResult::None
        }
        PhiExpression::FunctionCall { name, arguments } => {
            let mut args = Vec::with_capacity(arguments.len());
            for arg in arguments {
                let lowered = lower_expr(ctx, arg);
                args.push(unwrap_val(ctx, lowered));
            }
            let op = ctx.emit(PhiIRNode::Call {
                name: name.clone(),
                args,
            });
            LowerResult::Value(op)
        }

        // Binary Ops
        PhiExpression::BinaryOp {
            left,
            operator,
            right,
        } => {
            let res_left = lower_expr(ctx, left);
            let lhs = unwrap_val(ctx, res_left);
            let res_right = lower_expr(ctx, right);
            let rhs = unwrap_val(ctx, res_right);

            let phi_op = match operator {
                BinaryOperator::Add => PhiIRBinOp::Add,
                BinaryOperator::Subtract => PhiIRBinOp::Sub,
                BinaryOperator::Multiply => PhiIRBinOp::Mul,
                BinaryOperator::Divide => PhiIRBinOp::Div,
                BinaryOperator::Modulo => PhiIRBinOp::Mod,
                BinaryOperator::Power => PhiIRBinOp::Pow,
                BinaryOperator::Equal => PhiIRBinOp::Eq,
                BinaryOperator::NotEqual => PhiIRBinOp::Neq,
                BinaryOperator::Less => PhiIRBinOp::Lt,
                BinaryOperator::LessEqual => PhiIRBinOp::Lte,
                BinaryOperator::Greater => PhiIRBinOp::Gt,
                BinaryOperator::GreaterEqual => PhiIRBinOp::Gte,
                BinaryOperator::And => PhiIRBinOp::And,
                BinaryOperator::Or => PhiIRBinOp::Or,
            };

            let res = ctx.emit(PhiIRNode::BinOp {
                op: phi_op,
                left: lhs,
                right: rhs,
            });
            LowerResult::Value(res)
        }

        // Unary Ops - Missing in parser AST? Checked lowering.rs, it has UnaryOp
        PhiExpression::UnaryOp { operator, operand } => {
            let res = lower_expr(ctx, operand);
            let val = unwrap_val(ctx, res);
            let phi_op = match operator {
                UnaryOperator::Negate => PhiIRUnOp::Neg,
                UnaryOperator::Not => PhiIRUnOp::Not,
            };
            let res = ctx.emit(PhiIRNode::UnaryOp {
                op: phi_op,
                operand: val,
            });
            LowerResult::Value(res)
        }

        // Consciousness
        PhiExpression::Witness { expression, body } => {
            let target = if let Some(e) = expression {
                let res = lower_expr(ctx, e);
                Some(unwrap_val(ctx, res))
            } else {
                None
            };

            // Emit Witness
            // Note: IR Design says Witness returns a value (coherence)?
            // PhiIRNode definition has no return value explicit in enum, but `emit` returns Operand.
            // Implies Witness produces a value (Coherence Score).
            let op = ctx.emit(PhiIRNode::Witness {
                target,
                collapse_policy: CollapsePolicy::Deferred,
            });

            if let Some(b) = body {
                // How to scope the body?
                // In PhiIR, Witness is an instruction.
                // Ideally, we'd wrap body in Intention? Or just execute it.
                // A "Witness Block" isn't explicitly a scope in PhiIR aside from Intention.
                // We'll just lower the body.
                lower_expr(ctx, b);
            }

            LowerResult::Value(op)
        }

        PhiExpression::IntentionBlock { intention, body } => {
            // Push Intention
            ctx.program.intentions_declared.push(intention.clone());
            ctx.emit(PhiIRNode::IntentionPush {
                name: intention.clone(),
                frequency_hint: None,
            });

            // Body
            let res = lower_expr(ctx, body);

            // Pop Intention
            ctx.emit(PhiIRNode::IntentionPop);

            res
        }

        PhiExpression::StreamBlock { name, body } => {
            let header_block = ctx.new_block("stream_header");
            let body_block = ctx.new_block("stream_body");
            let exit_block = ctx.new_block("stream_exit");

            // Register stream name so reporters can find resonated values
            ctx.program.intentions_declared.push(name.clone());

            // Enter Stream
            ctx.emit(PhiIRNode::StreamPush(name.clone()));
            ctx.terminate(PhiIRNode::Jump(header_block));

            // Header (just links to body in an infinite loop)
            ctx.set_current_block(header_block);
            ctx.terminate(PhiIRNode::Jump(body_block));

            // Body
            ctx.set_current_block(body_block);
            ctx.stream_exits.push(exit_block);
            lower_expr(ctx, body);
            ctx.stream_exits.pop();

            // If the body didn't manually terminate (e.g. by BreakStream), loop back
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(header_block));
            }

            // Exit block
            ctx.set_current_block(exit_block);
            ctx.emit(PhiIRNode::StreamPop);

            LowerResult::None
        }

        PhiExpression::BreakStream => {
            if let Some(&exit_block) = ctx.stream_exits.last() {
                ctx.terminate(PhiIRNode::Jump(exit_block));
                // We must create a new unreachable block so subsequent parsing continues logically without panicking
                let unreachable = ctx.new_block("unreachable_after_break");
                ctx.set_current_block(unreachable);
            }
            LowerResult::None
        }

        PhiExpression::Resonate { expression } => {
            let val = if let Some(e) = expression {
                let res = lower_expr(ctx, e);
                Some(unwrap_val(ctx, res))
            } else {
                None
            };

            // Resonate produces... Void? Or boolean success?
            // Let's say Void for now in AST usage.
            ctx.emit(PhiIRNode::Resonate {
                value: val,
                frequency_relationship: None,
            });
            LowerResult::None
        }

        PhiExpression::CreatePattern {
            pattern_type,
            frequency,
            ..
        } => {
            let freq_op = ctx.emit(PhiIRNode::Const(PhiIRValue::Number(*frequency)));

            let kind = match pattern_type.as_str() {
                "Flower" => PatternKind::Flower,
                "Spiral" => PatternKind::Spiral,
                "Toroid" => PatternKind::Toroid,
                _ => PatternKind::Field,
            };

            let op = ctx.emit(PhiIRNode::CreatePattern {
                kind,
                frequency: freq_op,
                annotation: SacredFrequency::Arbitrary(*frequency),
                params: vec![],
            });
            LowerResult::Value(op)
        }

        PhiExpression::ConsciousnessValidation { pattern, metrics } => {
            // Validates pattern
            // Wait, `ValidatePattern` node exists in PhiIR?
            // `DomainCall` usually.
            // But we added `Validate` to `DomainOp`.
            // `PhiIRNode` does not have explicit `ValidatePattern`,
            // except `DomainCall` OR we might have added it?
            // Let's check my `mod.rs` creation...
            // Step 2183: `PhiIRNode` HAS `ValidatePattern`? No.
            // It has `CreatePattern`.
            // It has `DomainCall`.
            // It has `DomainOp::Validate`.
            // Ah! `mod.rs` has `ValidatePattern { metrics: Vec<String> }` in *IR_DESIGN*,
            // but `src/ir/mod.rs` (Stack VM) has it.
            // `src/phi_ir/mod.rs` (Step 2183) ...
            // Line 209: `ValidatePattern { metrics: Vec<String> }`?
            // Let me check Step 2183 content...
            // Line 266: `DomainOp::Validate`.
            // Line 211: `ValidatePattern { metrics: Vec<String> }`? NO.
            // Reviewing Step 2183...
            // "CreatePattern" is there.
            // "DomainCall" is there.
            // No top-level `ValidatePattern` node in `PhiIRNode`.
            // So I should use `DomainCall`.

            let res = lower_expr(ctx, pattern);
            let pat_op = unwrap_val(ctx, res);

            let op = ctx.emit(PhiIRNode::DomainCall {
                op: DomainOp::Validate,
                args: vec![pat_op],
                string_args: metrics.clone(),
            });

            LowerResult::Value(op)
        }

        // Control Flow
        PhiExpression::IfElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let res = lower_expr(ctx, condition);
            let cond_op = unwrap_val(ctx, res);

            let start_block = ctx.current_block;
            let then_block_id = ctx.new_block("then");
            let else_block_id = ctx.new_block("else");
            let merge_block_id = ctx.new_block("merge");

            // Terminate start block with Branch
            ctx.set_current_block(start_block);
            ctx.terminate(PhiIRNode::Branch {
                condition: cond_op,
                then_block: then_block_id,
                else_block: else_block_id,
            });

            // THEN Block
            ctx.set_current_block(then_block_id);
            lower_expr(ctx, then_branch);
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(merge_block_id));
            }

            // ELSE Block
            ctx.set_current_block(else_block_id);
            if let Some(else_expr) = else_branch {
                lower_expr(ctx, else_expr);
            }
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(merge_block_id));
            }

            // Resume in Merge Block
            ctx.set_current_block(merge_block_id);
            LowerResult::None // If expressions merge values, we need Phi nodes. Skipping for now.
        }

        PhiExpression::WhileLoop { condition, body } => {
            let header_block = ctx.new_block("while_header");
            let body_block = ctx.new_block("while_body");
            let exit_block = ctx.new_block("while_exit");

            // Jump to header from current
            ctx.terminate(PhiIRNode::Jump(header_block));

            // Header: check condition
            ctx.set_current_block(header_block);
            let res = lower_expr(ctx, condition);
            let cond_op = unwrap_val(ctx, res);
            ctx.terminate(PhiIRNode::Branch {
                condition: cond_op,
                then_block: body_block,
                else_block: exit_block,
            });

            // Body: execute statements, then jump back to header
            ctx.set_current_block(body_block);
            lower_expr(ctx, body);
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(header_block));
            }

            // Continue in exit block
            ctx.set_current_block(exit_block);
            LowerResult::None
        }

        PhiExpression::Return(expr) => {
            let lowered = lower_expr(ctx, expr);
            let value = unwrap_val(ctx, lowered);
            ctx.terminate(PhiIRNode::Return(value));
            // Continue lowering into a fresh unreachable block.
            let unreachable = ctx.new_block("unreachable_after_return");
            ctx.set_current_block(unreachable);
            LowerResult::None
        }

        PhiExpression::ForLoop {
            variable,
            iterable,
            body,
        } => {
            // For loops need desugaring.
            // Simplified lowering for Range (start..end) if iterable is a specific structure
            // For now, we'll mark it as unimplemented or attempt a basic desugar if we assume iterable is a range.
            // Since we lack a dedicated Range expression, we'll emit a comment/placeholder.
            // TODO: generic iterator support.
            let header_block = ctx.new_block("for_header"); // Placeholder to link flow
            ctx.terminate(PhiIRNode::Jump(header_block));
            ctx.set_current_block(header_block);
            // ... implementation pending iterable definition ...
            LowerResult::None
        }

        PhiExpression::Block(exprs) => {
            let mut last = LowerResult::None;
            for e in exprs {
                last = lower_expr(ctx, e);
            }
            last
        }

        // Handle others or fallback to Void
        _ => LowerResult::None,
    }
}

/// Helper to ensure we get an operand, emitting Void if None
fn unwrap_val(ctx: &mut LoweringContext, res: LowerResult) -> Operand {
    match res {
        LowerResult::Value(op) => op,
        LowerResult::None => ctx.emit(PhiIRNode::Const(PhiIRValue::Void)),
    }
}
