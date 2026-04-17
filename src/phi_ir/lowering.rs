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

use crate::parser::{
    BinaryOperator, PhiExpression, ResonateDirection as AstResonateDirection, UnaryOperator,
};
use crate::phi_ir::{
    BlockId, CollapsePolicy, DomainOp, Operand, Param, PatternKind, PhiIRBinOp, PhiIRBlock,
    PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue, PhiInstruction,
    ResonateDirection as IrResonateDirection, SacredFrequency, SensorKind,
};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Result of lowering an expression: either a Value (Operand) or None (Void/Statement).
#[derive(Debug, Clone, Copy, PartialEq)]
enum LowerResult {
    Value(Operand),
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    InvalidSensorWitnessSyntax,
    UnknownSensor(String),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::InvalidSensorWitnessSyntax => {
                write!(
                    f,
                    "sensor witness must use witness sensor(\"cpu_usage|cpu_temp|memory_usage\")"
                )
            }
            LoweringError::UnknownSensor(name) => {
                write!(f, "unknown sensor `{name}`")
            }
        }
    }
}

impl Error for LoweringError {}

struct LoweringContext {
    program: PhiIRProgram,
    current_block: BlockId,
    /// Counter for generating instruction indices (Operands)
    next_operand: u32,
    /// Counter for generating BlockIds
    next_block_id: u32,
    /// Map of variable name to the most recent Operand holding its value.
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
                | PhiIRNode::WitnessSensor { .. } // FIX: returns sensor value
                | PhiIRNode::CoherenceCheck     // returns coherence score (0.0–1.0)
                | PhiIRNode::Recall(_)
                | PhiIRNode::Listen(_)
                | PhiIRNode::VoidDepth
                | PhiIRNode::Evolve(_)
                | PhiIRNode::Entangle(_)
                | PhiIRNode::FieldCoherence
                | PhiIRNode::Dissonance
                | PhiIRNode::CoherenceOf(_)
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

pub fn lower_program_checked(expressions: &[PhiExpression]) -> Result<PhiIRProgram, LoweringError> {
    for expr in expressions {
        validate_sensor_witness(expr)?;
    }

    Ok(lower_program_unchecked(expressions))
}

pub fn lower_program(expressions: &[PhiExpression]) -> PhiIRProgram {
    lower_program_checked(expressions).expect("PhiIR lowering failed")
}

fn lower_program_unchecked(expressions: &[PhiExpression]) -> PhiIRProgram {
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
    }

    ctx.program
}

fn validate_sensor_witness(expr: &PhiExpression) -> Result<(), LoweringError> {
    match expr {
        PhiExpression::Witness {
            expression, body, ..
        } => {
            if let Some(inner) = expression {
                validate_sensor_witness(inner)?;
            }
            if let Some(body) = body {
                validate_sensor_witness(body)?;
            }
            Ok(())
        }
        PhiExpression::FunctionCall { name, arguments } => {
            if name == "sensor" {
                if arguments.len() != 1 {
                    return Err(LoweringError::InvalidSensorWitnessSyntax);
                }
                let sensor_name = match &arguments[0] {
                    PhiExpression::String(name) => name,
                    _ => return Err(LoweringError::InvalidSensorWitnessSyntax),
                };
                SensorKind::from_name(sensor_name)
                    .ok_or_else(|| LoweringError::UnknownSensor(sensor_name.clone()))?;
            }
            for argument in arguments {
                validate_sensor_witness(argument)?;
            }
            Ok(())
        }
        PhiExpression::FunctionDef { body, .. }

        | PhiExpression::Evolve(body)
        | PhiExpression::Return(body)
        | PhiExpression::PatternTransform { pattern: body, .. }
        | PhiExpression::HardwareSync {
            consciousness_mapping: body,
            ..
        }
        | PhiExpression::Remember { value: body, .. }
        | PhiExpression::Broadcast { value: body, .. }
        | PhiExpression::IntentionBlock { body, .. }
        | PhiExpression::StreamBlock { body, .. }
        | PhiExpression::AgentBlock { body, .. }
        | PhiExpression::Handoff { context: body, .. }
        | PhiExpression::AudioSynthesis { pattern: body, .. } => validate_sensor_witness(body),
        PhiExpression::ConsciousnessMonitor {
            expression,
            callback,
        } => {
            validate_sensor_witness(expression)?;
            validate_sensor_witness(callback)
        }
        PhiExpression::List(arguments)
        | PhiExpression::Block(arguments)
        | PhiExpression::PatternCombine {
            patterns: arguments,
            ..
        } => {
            for argument in arguments {
                validate_sensor_witness(argument)?;
            }
            Ok(())
        }
        PhiExpression::LetBinding { value, .. }
        | PhiExpression::ConsciousnessValidation { pattern: value, .. } => {
            validate_sensor_witness(value)
        }
        PhiExpression::IfElse {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_sensor_witness(condition)?;
            validate_sensor_witness(then_branch)?;
            if let Some(else_branch) = else_branch {
                validate_sensor_witness(else_branch)?;
            }
            Ok(())
        }
        PhiExpression::ForLoop { iterable, body, .. } => {
            validate_sensor_witness(iterable)?;
            validate_sensor_witness(body)
        }
        PhiExpression::WhileLoop { condition, body } => {
            validate_sensor_witness(condition)?;
            validate_sensor_witness(body)
        }
        PhiExpression::BinaryOp { left, right, .. } => {
            validate_sensor_witness(left)?;
            validate_sensor_witness(right)
        }
        PhiExpression::UnaryOp { operand, .. } => validate_sensor_witness(operand),
        PhiExpression::ListAccess { list, index } => {
            validate_sensor_witness(list)?;
            validate_sensor_witness(index)
        }
        PhiExpression::ConsciousnessFlow {
            condition,
            branches,
        } => {
            validate_sensor_witness(condition)?;
            for (_, branch) in branches {
                validate_sensor_witness(branch)?;
            }
            Ok(())
        }
        PhiExpression::EmergencyProtocol {
            trigger,
            immediate_action,
            ..
        } => {
            validate_sensor_witness(trigger)?;
            validate_sensor_witness(immediate_action)
        }
        PhiExpression::CreatePattern { .. }
        | PhiExpression::Variable(_)
        | PhiExpression::BreakStream
        | PhiExpression::Resonate { .. }
        | PhiExpression::Recall(_)
        | PhiExpression::Listen(_)
        | PhiExpression::VoidDepth
        | PhiExpression::ConsciousnessState { .. }
        | PhiExpression::FrequencyPattern { .. }
        | PhiExpression::QuantumField { .. }
        | PhiExpression::BiologicalInterface { .. }
        | PhiExpression::Number(_)
        | PhiExpression::String(_)
        | PhiExpression::Boolean(_)
        | PhiExpression::Entangle(_)
        | PhiExpression::Import { .. } => Ok(()),
    }
}

fn lower_expr(ctx: &mut LoweringContext, expr: &PhiExpression) -> LowerResult {
    match expr {
        PhiExpression::Number(n) => {
            let op = ctx.emit(PhiIRNode::Const(PhiIRValue::Number(*n)));
            LowerResult::Value(op)
        }
        PhiExpression::String(s) => {
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
            let op = if name == "coherence" && !ctx.scope.contains_key(name) {
                ctx.emit(PhiIRNode::CoherenceCheck)
            } else if name == "field" && !ctx.scope.contains_key(name) {
                ctx.emit(PhiIRNode::FieldCoherence)
            } else if name == "dissonance" && !ctx.scope.contains_key(name) {
                ctx.emit(PhiIRNode::Dissonance)
            } else {
                ctx.emit(PhiIRNode::LoadVar(name.clone()))
            };
            LowerResult::Value(op)
        }
        PhiExpression::LetBinding { name, value, .. } => {
            let val = match lower_expr(ctx, value) {
                LowerResult::Value(v) => v,
                LowerResult::None => ctx.emit(PhiIRNode::Const(PhiIRValue::Void)),
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

            ctx.emit(PhiIRNode::FuncDef {
                name: name.clone(),
                params: params.clone(),
                body: body_block,
            });

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
            if name == "coherence_of" && arguments.len() == 1 {
                if let PhiExpression::String(target) = &arguments[0] {
                    let op = ctx.emit(PhiIRNode::CoherenceOf(target.clone()));
                    return LowerResult::Value(op);
                }
            }

            if name == "sensor" && arguments.len() == 1 {
                if let PhiExpression::String(s) = &arguments[0] {
                    let sensor = SensorKind::from_name(s).expect("validated");
                    let op = ctx.emit(PhiIRNode::WitnessSensor { sensor });
                    return LowerResult::Value(op);
                }
            }

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
        PhiExpression::Witness {
            mid_circuit,
            expression,
            body,
        } => {
            let mut target = None;
            let mut is_sensor = false;
            let mut sensor_name = String::new();

            if let Some(e) = expression {
                if let PhiExpression::FunctionCall { name, arguments } = &**e {
                    if name == "sensor" && arguments.len() == 1 {
                        if let PhiExpression::String(s) = &arguments[0] {
                            is_sensor = true;
                            sensor_name = s.clone();
                        }
                    }
                }

                if !is_sensor {
                    let res = lower_expr(ctx, e);
                    target = Some(unwrap_val(ctx, res));
                }
            }

            let op = if is_sensor {
                let sensor = SensorKind::from_name(&sensor_name)
                    .expect("sensor witness should be validated before lowering");
                ctx.emit(PhiIRNode::WitnessSensor { sensor })
            } else {
                let policy = if *mid_circuit {
                    CollapsePolicy::MidCircuit
                } else {
                    CollapsePolicy::Final
                };

                ctx.emit(PhiIRNode::Witness {
                    target,
                    collapse_policy: policy,
                })
            };

            if let Some(b) = body {
                lower_expr(ctx, b);
            }

            LowerResult::Value(op)
        }

        PhiExpression::IntentionBlock { intention, body } => {
            ctx.program.intentions_declared.push(intention.clone());
            ctx.emit(PhiIRNode::IntentionPush {
                name: intention.clone(),
                frequency_hint: None,
            });

            let res = lower_expr(ctx, body);
            ctx.emit(PhiIRNode::IntentionPop);
            res
        }

        PhiExpression::StreamBlock { name, body } => {
            let header_block = ctx.new_block("stream_header");
            let body_block = ctx.new_block("stream_body");
            let exit_block = ctx.new_block("stream_exit");

            ctx.program.intentions_declared.push(name.clone());

            ctx.emit(PhiIRNode::StreamPush(name.clone(), None));
            ctx.terminate(PhiIRNode::Jump(header_block));

            ctx.set_current_block(header_block);
            ctx.terminate(PhiIRNode::Jump(body_block));

            ctx.set_current_block(body_block);
            ctx.stream_exits.push(exit_block);
            lower_expr(ctx, body);
            ctx.stream_exits.pop();

            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(header_block));
            }

            ctx.set_current_block(exit_block);
            ctx.emit(PhiIRNode::StreamPop);

            LowerResult::None
        }

        PhiExpression::BreakStream => {
            if let Some(&exit_block) = ctx.stream_exits.last() {
                ctx.terminate(PhiIRNode::Jump(exit_block));
                let unreachable = ctx.new_block("unreachable_after_break");
                ctx.set_current_block(unreachable);
            }
            LowerResult::None
        }

        PhiExpression::Resonate {
            expression,
            direction,
        } => {
            let val = if let Some(e) = expression {
                let res = lower_expr(ctx, e);
                Some(unwrap_val(ctx, res))
            } else {
                None
            };

            ctx.emit(PhiIRNode::Resonate {
                value: val,
                frequency_relationship: None,
                direction: match *direction {
                    AstResonateDirection::TeamA => IrResonateDirection::TeamA,
                    AstResonateDirection::TeamB => IrResonateDirection::TeamB,
                },
            });
            LowerResult::None
        }

        // v0.3.0 Persistence & Dialogue
        PhiExpression::Remember { key, value } => {
            let res = lower_expr(ctx, value);
            let val_op = unwrap_val(ctx, res);
            ctx.emit(PhiIRNode::Remember {
                key: key.clone(),
                value: val_op,
            });
            LowerResult::None
        }
        PhiExpression::Recall(key) => {
            let op = ctx.emit(PhiIRNode::Recall(key.clone()));
            LowerResult::Value(op)
        }
        PhiExpression::Broadcast { channel, value } => {
            let res = lower_expr(ctx, value);
            let val_op = unwrap_val(ctx, res);
            ctx.emit(PhiIRNode::Broadcast {
                channel: channel.clone(),
                value: val_op,
            });
            LowerResult::None
        }
        PhiExpression::Listen(channel) => {
            let op = ctx.emit(PhiIRNode::Listen(channel.clone()));
            LowerResult::Value(op)
        }

        // Agent Identity
        PhiExpression::AgentBlock {
            name,
            version,
            body,
        } => {
            ctx.emit(PhiIRNode::AgentDecl {
                name: name.clone(),
                version: version.clone(),
            });
            lower_expr(ctx, body)
        }

        // Time-awareness
        PhiExpression::VoidDepth => {
            let op = ctx.emit(PhiIRNode::VoidDepth);
            LowerResult::Value(op)
        }

        // v0.4.0 Transcendent Capabilities
        PhiExpression::Evolve(expr) => {
            let res = lower_expr(ctx, expr);
            let op = unwrap_val(ctx, res);
            let res_op = ctx.emit(PhiIRNode::Evolve(op));
            LowerResult::Value(res_op)
        }
        PhiExpression::Entangle(freq) => {
            let op = ctx.emit(PhiIRNode::Entangle(*freq));
            LowerResult::Value(op)
        }
        PhiExpression::Handoff {
            target_agent,
            task_id,
            context,
        } => {
            let res = lower_expr(ctx, context);
            let ctx_op = unwrap_val(ctx, res);
            let op = ctx.emit(PhiIRNode::Handoff {
                target_agent: target_agent.clone(),
                task_id: task_id.clone(),
                context_op: ctx_op,
            });
            LowerResult::Value(op)
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
            let res = lower_expr(ctx, pattern);
            let pat_op = unwrap_val(ctx, res);

            let op = ctx.emit(PhiIRNode::DomainCall {
                op: DomainOp::Validate,
                args: vec![pat_op],
                string_args: metrics.clone(),
            });

            LowerResult::Value(op)
        }

        PhiExpression::Block(exprs) => {
            let mut last = LowerResult::None;
            for e in exprs {
                last = lower_expr(ctx, e);
            }
            last
        }

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

            ctx.set_current_block(start_block);
            ctx.terminate(PhiIRNode::Branch {
                condition: cond_op,
                then_block: then_block_id,
                else_block: else_block_id,
            });

            ctx.set_current_block(then_block_id);
            lower_expr(ctx, then_branch);
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(merge_block_id));
            }

            ctx.set_current_block(else_block_id);
            if let Some(else_expr) = else_branch {
                lower_expr(ctx, else_expr);
            }
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(merge_block_id));
            }

            ctx.set_current_block(merge_block_id);
            LowerResult::None
        }

        PhiExpression::WhileLoop { condition, body } => {
            let header_block = ctx.new_block("while_header");
            let body_block = ctx.new_block("while_body");
            let exit_block = ctx.new_block("while_exit");

            ctx.terminate(PhiIRNode::Jump(header_block));

            ctx.set_current_block(header_block);
            let res = lower_expr(ctx, condition);
            let cond_op = unwrap_val(ctx, res);
            ctx.terminate(PhiIRNode::Branch {
                condition: cond_op,
                then_block: body_block,
                else_block: exit_block,
            });

            ctx.set_current_block(body_block);
            lower_expr(ctx, body);
            if !ctx.is_terminated(ctx.current_block) {
                ctx.terminate(PhiIRNode::Jump(header_block));
            }

            ctx.set_current_block(exit_block);
            LowerResult::None
        }

        PhiExpression::Return(expr) => {
            let lowered = lower_expr(ctx, expr);
            let value = unwrap_val(ctx, lowered);
            ctx.terminate(PhiIRNode::Return(value));
            let unreachable = ctx.new_block("unreachable_after_return");
            ctx.set_current_block(unreachable);
            LowerResult::None
        }

        _ => LowerResult::None,
    }
}

fn unwrap_val(ctx: &mut LoweringContext, res: LowerResult) -> Operand {
    match res {
        LowerResult::Value(op) => op,
        LowerResult::None => ctx.emit(PhiIRNode::Const(PhiIRValue::Void)),
    }
}
