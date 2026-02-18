//! PhiIR Optimizer
//!
//! Implements optimization passes for PhiIR:
//! 1. Constant Folding: Evaluates constant expressions at compile time.
//! 2. Dead Code Elimination: Removes unused pure instructions.

use crate::phi_ir::{
    BlockId, Operand, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue,
    PhiInstruction,
};
use std::collections::{HashMap, HashSet};

pub struct Optimizer;

impl Optimizer {
    /// Runs all optimization passes until convergence or max iterations.
    pub fn optimize(program: &mut PhiIRProgram) {
        let mut changed = true;
        let mut pass = 0;
        const MAX_PASSES: usize = 10;

        while changed && pass < MAX_PASSES {
            changed = false;
            // println!("Optimization Pass {}", pass + 1);

            // 1. Constant Folding
            if Self::constant_folding(program) {
                changed = true;
            }

            // 2. Dead Code Elimination
            if Self::dead_code_elimination(program) {
                changed = true;
            }

            pass += 1;
        }
    }

    /// Constant Folding Pass
    /// Returns true if any changes were made.
    fn constant_folding(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;

        // Build a map of definitions for quick lookup
        // We only care about Const definitions for folding
        // Map: Operand -> PhiIRValue
        let mut const_values: HashMap<Operand, PhiIRValue> = HashMap::new();

        // Populate initial constants
        for block in &program.blocks {
            for instr in &block.instructions {
                if let Some(res) = instr.result {
                    if let PhiIRNode::Const(val) = &instr.node {
                        const_values.insert(res, val.clone());
                    }
                }
            }
        }

        // Iterate and fold
        // We must iterate all blocks.
        // We iterate instructions strictly.
        for block in &mut program.blocks {
            for instr in &mut block.instructions {
                // If this instruction is already a Const, skip
                if matches!(instr.node, PhiIRNode::Const(_)) {
                    continue;
                }

                // Try to fold
                if let Some(folded) = Self::try_fold(&instr.node, &const_values) {
                    // Update instruction to be Const
                    instr.node = PhiIRNode::Const(folded.clone());
                    // Update map for subsequent instructions in this pass (if logical)
                    if let Some(res) = instr.result {
                        const_values.insert(res, folded);
                    }
                    changed = true;
                }
            }
        }

        changed
    }

    fn try_fold(node: &PhiIRNode, consts: &HashMap<Operand, PhiIRValue>) -> Option<PhiIRValue> {
        match node {
            PhiIRNode::BinOp { op, left, right } => {
                let lhs = consts.get(left)?;
                let rhs = consts.get(right)?;

                match (lhs, rhs) {
                    (PhiIRValue::Number(l), PhiIRValue::Number(r)) => match op {
                        PhiIRBinOp::Add => Some(PhiIRValue::Number(l + r)),
                        PhiIRBinOp::Sub => Some(PhiIRValue::Number(l - r)),
                        PhiIRBinOp::Mul => Some(PhiIRValue::Number(l * r)),
                        PhiIRBinOp::Div => Some(PhiIRValue::Number(l / r)),
                        PhiIRBinOp::Mod => Some(PhiIRValue::Number(l % r)),
                        PhiIRBinOp::Pow => Some(PhiIRValue::Number(l.powf(*r))),
                        PhiIRBinOp::Eq => Some(PhiIRValue::Boolean((l - r).abs() < f64::EPSILON)),
                        PhiIRBinOp::Neq => Some(PhiIRValue::Boolean((l - r).abs() >= f64::EPSILON)),
                        PhiIRBinOp::Lt => Some(PhiIRValue::Boolean(l < r)),
                        PhiIRBinOp::Lte => Some(PhiIRValue::Boolean(l <= r)),
                        PhiIRBinOp::Gt => Some(PhiIRValue::Boolean(l > r)),
                        PhiIRBinOp::Gte => Some(PhiIRValue::Boolean(l >= r)),
                        _ => None,
                    },
                    (PhiIRValue::Boolean(l), PhiIRValue::Boolean(r)) => match op {
                        PhiIRBinOp::And => Some(PhiIRValue::Boolean(*l && *r)),
                        PhiIRBinOp::Or => Some(PhiIRValue::Boolean(*l || *r)),
                        PhiIRBinOp::Eq => Some(PhiIRValue::Boolean(l == r)),
                        PhiIRBinOp::Neq => Some(PhiIRValue::Boolean(l != r)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            PhiIRNode::UnaryOp { op, operand } => {
                let val = consts.get(operand)?;
                match val {
                    PhiIRValue::Number(n) => match op {
                        PhiIRUnOp::Neg => Some(PhiIRValue::Number(-n)),
                        _ => None,
                    },
                    PhiIRValue::Boolean(b) => match op {
                        PhiIRUnOp::Not => Some(PhiIRValue::Boolean(!b)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Dead Code Elimination Pass
    /// Returns true if any changes were made.
    fn dead_code_elimination(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;

        // 1. Identify all used operands
        // Used in:
        // - Instruction inputs (BinOp, Call, etc.)
        // - Terminators (Branch condition, Return val)
        // - Side-effect instructions inputs (implicitly, e.g. StoreVar value)
        let mut used_operands: HashSet<Operand> = HashSet::new();

        for block in &program.blocks {
            // Check instructions
            for instr in &block.instructions {
                Self::collect_used_operands(&instr.node, &mut used_operands);
            }
            // Check terminator
            Self::collect_used_operands(&block.terminator, &mut used_operands);
        }

        // 2. Eliminate unused pure instructions
        for block in &mut program.blocks {
            for instr in &mut block.instructions {
                // If it produces a result that is unused...
                if let Some(res) = instr.result {
                    if !used_operands.contains(&res) {
                        // Check if instruction is pure (no side effects)
                        if Self::is_pure(&instr.node) {
                            if !matches!(instr.node, PhiIRNode::Nop) {
                                instr.node = PhiIRNode::Nop;
                                changed = true;
                            }
                        }
                    }
                }
                // If it produces NO result (None), it's likely side-effect only and kept.
                // Or if it DOES produce a result but it's used, we keep it.
            }
        }

        changed
    }

    fn collect_used_operands(node: &PhiIRNode, used: &mut HashSet<Operand>) {
        match node {
            PhiIRNode::BinOp { left, right, .. } => {
                used.insert(*left);
                used.insert(*right);
            }
            PhiIRNode::UnaryOp { operand, .. } => {
                used.insert(*operand);
            }
            PhiIRNode::StoreVar { value, .. } => {
                used.insert(*value);
            }
            PhiIRNode::Call { args, .. }
            | PhiIRNode::ListNew(args)
            | PhiIRNode::DomainCall { args, .. } => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            PhiIRNode::ListGet { list, index } => {
                used.insert(*list);
                used.insert(*index);
            }
            PhiIRNode::Return(val) => {
                used.insert(*val);
            }
            PhiIRNode::Branch { condition, .. } => {
                used.insert(*condition);
            }
            PhiIRNode::Witness { target, .. } => {
                if let Some(t) = target {
                    used.insert(*t);
                }
            }
            PhiIRNode::Resonate { value, .. } => {
                if let Some(v) = value {
                    used.insert(*v);
                }
            }
            PhiIRNode::CreatePattern {
                frequency, params, ..
            } => {
                used.insert(*frequency);
                for (_, arg) in params {
                    used.insert(*arg);
                }
            }
            // Others don't use operands or are Nop/Const/LoadVar
            _ => {}
        }
    }

    fn is_pure(node: &PhiIRNode) -> bool {
        match node {
            // Pure computations
            PhiIRNode::Const(_)
            | PhiIRNode::LoadVar(_) // Reading variable is pure in local scope (no mutation of var itself, just read)
            | PhiIRNode::BinOp { .. }
            | PhiIRNode::UnaryOp { .. }
            | PhiIRNode::ListNew(_)
            | PhiIRNode::ListGet { .. }
            | PhiIRNode::CreatePattern { .. } // Creating a pattern object is pure, unless it renders immediately? Assume pure obj creation.
            | PhiIRNode::FuncDef { .. } // Functional definition is pure
             => true,

            // Side effects or control flow
            PhiIRNode::StoreVar { .. } // Mutates state
            | PhiIRNode::Call { .. } // Unknown side effects
            | PhiIRNode::DomainCall { .. } // Unknown
            | PhiIRNode::Witness { .. } // Side effect: observing
            | PhiIRNode::IntentionPush { .. }
            | PhiIRNode::IntentionPop
            | PhiIRNode::Resonate { .. }
            | PhiIRNode::CoherenceCheck
            | PhiIRNode::Return(_)
            | PhiIRNode::Branch { .. }
            | PhiIRNode::Jump(_)
            | PhiIRNode::Fallthrough
            | PhiIRNode::Nop
            => false,
        }
    }
}
