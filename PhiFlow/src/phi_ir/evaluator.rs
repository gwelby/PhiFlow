//! PhiIR Direct Evaluator
//!
//! Interprets PhiIRProgram directly.

use crate::phi_ir::{
    BlockId, Operand, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue,
    PhiInstruction,
};
use std::collections::HashMap;

#[derive(Debug)]
pub enum EvalError {
    BlockNotFound(BlockId),
    OperandNotFound(Operand),
    DivisionByZero,
    InvalidOperation(String),
    Unimplemented(String),
}

type EvalResult<T> = Result<T, EvalError>;

pub struct Evaluator<'a> {
    program: &'a PhiIRProgram,
    /// Storage for SSA values (Operands).
    /// Since operands are u32, we could use a Vec, but sparse storage (HashMap)
    /// might be safer if operands are not contiguous (though they usually are).
    /// Let's use a Vec and resize if needed, or HashMap for flexibility.
    /// Operands are globally unique in the program? No, they are unique per function/program context.
    /// In current lowering, they are unique per program (LowingContext::next_operand).
    registers: HashMap<Operand, PhiIRValue>,

    current_block: BlockId,
    instruction_ptr: usize,
}

impl<'a> Evaluator<'a> {
    pub fn new(program: &'a PhiIRProgram) -> Self {
        Self {
            program,
            registers: HashMap::new(),
            current_block: program.blocks.first().map(|b| b.id).unwrap_or(0),
            instruction_ptr: 0,
        }
    }

    pub fn run(&mut self) -> EvalResult<PhiIRValue> {
        loop {
            // Get current block
            let block = self.get_block(self.current_block)?;

            // Execute instructions
            if self.instruction_ptr < block.instructions.len() {
                let instr = &block.instructions[self.instruction_ptr];
                self.instruction_ptr += 1;
                self.execute_instruction(instr)?;
            } else {
                // Execute terminator
                return self.execute_terminator(&block.terminator);
            }
        }
    }

    fn get_block(&self, id: BlockId) -> EvalResult<&'a PhiIRBlock> {
        self.program
            .blocks
            .iter()
            .find(|b| b.id == id)
            .ok_or(EvalError::BlockNotFound(id))
    }

    fn execute_instruction(&mut self, instr: &PhiInstruction) -> EvalResult<()> {
        let value = match &instr.node {
            PhiIRNode::Const(v) => Some(v.clone()),
            PhiIRNode::BinOp { op, left, right } => Some(self.eval_binop(op, *left, *right)?),
            PhiIRNode::UnaryOp { op, operand } => Some(self.eval_unop(op, *operand)?),
            PhiIRNode::LoadVar(_) => {
                // For now, variables are not deeply integrated in this simple evaluator
                // beyond what SSA provides. If LoadVar meant "load from memory", we'd need a memory map.
                // In lowering, LoadVar is just an instruction.
                // We'll treat it as returning Void or Unimplemented for now given strictly SSA nature?
                // Actually, `LoweringContext` didn't emit Store/Load pairs for registers.
                // It emitted `LoadVar` instruction.
                // We need a separate variable storage if we want to support variables across modifications.
                // But PhiIR is SSA-like.
                // Let's assume for this MV P that we only fallback to simple evaluation.
                // Real SSA handles variables via Phis.
                // Our `LoadVar` looks up in a `Scope`.
                // Let's implement a scope map.
                return Err(EvalError::Unimplemented(
                    "LoadVar not yet supported in simple evaluator".to_string(),
                ));
            }
            PhiIRNode::StoreVar { .. } => {
                return Err(EvalError::Unimplemented(
                    "StoreVar not yet supported in simple evaluator".to_string(),
                ));
            }
            PhiIRNode::Nop => None,
            // ... other nodes
            _ => {
                return Err(EvalError::Unimplemented(format!(
                    "Instruction {:?} not implemented",
                    instr.node
                )))
            }
        };

        if let Some(val) = value {
            if let Some(res) = instr.result {
                self.registers.insert(res, val);
            }
        }

        Ok(())
    }

    fn execute_terminator(&mut self, node: &PhiIRNode) -> EvalResult<PhiIRValue> {
        match node {
            PhiIRNode::Return(op) => {
                let val = self.get_operand(*op)?;
                Ok(val.clone())
            }
            PhiIRNode::Fallthrough => {
                // If it's the last block, return Void.
                // Or find next block? PhiIR design implies Fallthrough goes to next physical block.
                // Let's assume simple sequential blocks if not linked.
                // But strictly speaking, Fallthrough should point to next block ID?
                // Our `PhiIRBlock` doesn't strictly enforce ordering but `blocks` list does.
                // We can find index of current block and go to next.
                let current_idx = self
                    .program
                    .blocks
                    .iter()
                    .position(|b| b.id == self.current_block)
                    .unwrap();
                if current_idx + 1 < self.program.blocks.len() {
                    self.current_block = self.program.blocks[current_idx + 1].id;
                    self.instruction_ptr = 0;
                    // Continue loop, but we need to return something or indicate continue.
                    // The run loop handles this by updating state and looping.
                    // But `execute_terminator` returns `Result<PhiIRValue>`.
                    // We need a way to signal "Continue".
                    // Let's change `execute_terminator` to return `EvalResult<Option<PhiIRValue>>`.
                    // If None, continue.
                    Err(EvalError::Unimplemented(
                        "Fallthrough logic to be refined".to_string(),
                    ))
                } else {
                    Ok(PhiIRValue::Void) // End of program
                }
            }
            _ => Err(EvalError::Unimplemented(format!(
                "Terminator {:?} not implemented",
                node
            ))),
        }
    }

    fn get_operand(&self, op: Operand) -> EvalResult<&PhiIRValue> {
        self.registers
            .get(&op)
            .ok_or(EvalError::OperandNotFound(op))
    }

    fn eval_binop(&self, op: &PhiIRBinOp, left: Operand, right: Operand) -> EvalResult<PhiIRValue> {
        let l = self.get_operand(left)?;
        let r = self.get_operand(right)?;

        match (l, r) {
            (PhiIRValue::Number(lhs), PhiIRValue::Number(rhs)) => match op {
                PhiIRBinOp::Add => Ok(PhiIRValue::Number(lhs + rhs)),
                PhiIRBinOp::Sub => Ok(PhiIRValue::Number(lhs - rhs)),
                PhiIRBinOp::Mul => Ok(PhiIRValue::Number(lhs * rhs)),
                PhiIRBinOp::Div => {
                    if *rhs == 0.0 {
                        Err(EvalError::DivisionByZero)
                    } else {
                        Ok(PhiIRValue::Number(lhs / rhs))
                    }
                }
                _ => Err(EvalError::Unimplemented(format!(
                    "BinOp {:?} for numbers",
                    op
                ))),
            },
            _ => Err(EvalError::InvalidOperation(
                "Type mismatch or unsupported types".to_string(),
            )),
        }
    }

    fn eval_unop(&self, op: &PhiIRUnOp, operand: Operand) -> EvalResult<PhiIRValue> {
        let val = self.get_operand(operand)?;
        match val {
            PhiIRValue::Number(n) => match op {
                PhiIRUnOp::Neg => Ok(PhiIRValue::Number(-n)),
                _ => Err(EvalError::Unimplemented(format!(
                    "UnOp {:?} for numbers",
                    op
                ))),
            },
            _ => Err(EvalError::InvalidOperation(
                "Type mismatch or unsupported types".to_string(),
            )),
        }
    }
}
