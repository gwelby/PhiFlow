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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Basic,       // Constant Folding, DCE
    Aggressive,  // Inlining, etc.
    PhiHarmonic, // The Golden Ratio, Dreaming, Self-Healing
}

/// Measures the "Harmonic Health" of the IR.
/// Ideally, the ratio of control flow to computation should approach Phi (1.618).
pub struct CoherenceMonitor {
    pub instruction_count: usize,
    pub control_flow_count: usize,
    pub arithmetic_count: usize,
    pub coherence_score: f64,
}

impl CoherenceMonitor {
    pub fn new() -> Self {
        Self {
            instruction_count: 0,
            control_flow_count: 0,
            arithmetic_count: 0,
            coherence_score: 1.0,
        }
    }

    pub fn analyze(&mut self, program: &PhiIRProgram) {
        self.instruction_count = 0;
        self.control_flow_count = 0;
        self.arithmetic_count = 0;

        for block in &program.blocks {
            for instr in &block.instructions {
                self.instruction_count += 1;
                match instr.node {
                    PhiIRNode::Branch { .. } | PhiIRNode::Jump(_) | PhiIRNode::Return(_) => {
                        self.control_flow_count += 1;
                    }
                    PhiIRNode::BinOp { .. } | PhiIRNode::UnaryOp { .. } => {
                        self.arithmetic_count += 1;
                    }
                    _ => {}
                }
            }
            // Terminators count as well
            self.instruction_count += 1;
            match block.terminator {
                PhiIRNode::Branch { .. } | PhiIRNode::Jump(_) | PhiIRNode::Return(_) => {
                    self.control_flow_count += 1;
                }
                _ => {}
            }
        }

        self.compute_score();
    }

    fn compute_score(&mut self) {
        const PHI: f64 = 1.6180339887;

        if self.control_flow_count == 0 {
            self.coherence_score = 0.5;
            return;
        }

        let ratio = self.arithmetic_count as f64 / self.control_flow_count as f64;
        let deviation = (ratio - PHI).abs();

        self.coherence_score = 1.0 / (1.0 + deviation);
    }
}

impl Default for CoherenceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Optimizer {
    pub level: OptimizationLevel,
    pub monitor: CoherenceMonitor,
}

impl Optimizer {
    pub fn new(level: OptimizationLevel) -> Self {
        Self {
            level,
            monitor: CoherenceMonitor::new(),
        }
    }

    /// Runs all optimization passes until convergence or max iterations.
    pub fn optimize(&mut self, program: &mut PhiIRProgram) {
        let mut changed = true;
        let mut pass = 0;
        const MAX_PASSES: usize = 10;

        while changed && pass < MAX_PASSES {
            changed = false;

            if Self::constant_folding(program) {
                changed = true;
            }

            if Self::dead_code_elimination(program) {
                changed = true;
            }

            pass += 1;
        }

        if self.level == OptimizationLevel::PhiHarmonic {
            if Self::unroll_loops(program) {
            }
        }

        if self.level == OptimizationLevel::PhiHarmonic {
            self.monitor.analyze(program);
            self.stabilize(program);
        }
    }

    fn constant_folding(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;
        let mut const_values: HashMap<Operand, PhiIRValue> = HashMap::new();

        for block in &program.blocks {
            for instr in &block.instructions {
                if let Some(res) = instr.result {
                    if let PhiIRNode::Const(val) = &instr.node {
                        const_values.insert(res, val.clone());
                    }
                }
            }
        }

        for block in &mut program.blocks {
            for instr in &mut block.instructions {
                if matches!(instr.node, PhiIRNode::Const(_)) {
                    continue;
                }

                if let Some(folded) = Self::try_fold(&instr.node, &const_values) {
                    instr.node = PhiIRNode::Const(folded.clone());
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

    fn dead_code_elimination(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;
        let mut used_operands: HashSet<Operand> = HashSet::new();

        for block in &program.blocks {
            for instr in &block.instructions {
                Self::collect_used_operands(&instr.node, &mut used_operands);
            }
            Self::collect_used_operands(&block.terminator, &mut used_operands);
        }

        for block in &mut program.blocks {
            for instr in &mut block.instructions {
                if let Some(res) = instr.result {
                    if !used_operands.contains(&res) {
                        if Self::is_pure(&instr.node) && !matches!(instr.node, PhiIRNode::Nop) {
                            instr.node = PhiIRNode::Nop;
                            changed = true;
                        }
                    }
                }
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
            PhiIRNode::Witness {
                target: Some(t), ..
            } => {
                used.insert(*t);
            }
            PhiIRNode::Witness { target: None, .. } => {}
            PhiIRNode::Resonate {
                value: Some(v), ..
            } => {
                used.insert(*v);
            }
            PhiIRNode::Resonate { value: None, .. } => {}
            PhiIRNode::CreatePattern {
                frequency, params, ..
            } => {
                used.insert(*frequency);
                for (_, arg) in params {
                    used.insert(*arg);
                }
            }
            PhiIRNode::Sleep { duration } => {
                used.insert(*duration);
            }
            PhiIRNode::Remember { value, .. } => {
                used.insert(*value);
            }
            PhiIRNode::Broadcast { value, .. } => {
                used.insert(*value);
            }
            _ => {}
        }
    }

    fn is_pure(node: &PhiIRNode) -> bool {
        match node {
            PhiIRNode::Const(_)
            | PhiIRNode::LoadVar(_)
            | PhiIRNode::BinOp { .. }
            | PhiIRNode::UnaryOp { .. }
            | PhiIRNode::ListNew(_)
            | PhiIRNode::ListGet { .. }
            | PhiIRNode::CreatePattern { .. }
            | PhiIRNode::FuncDef { .. }
             => true,

            PhiIRNode::StoreVar { .. }
            | PhiIRNode::Call { .. }
            | PhiIRNode::DomainCall { .. }
            | PhiIRNode::Witness { .. }
            | PhiIRNode::IntentionPush { .. }
            | PhiIRNode::IntentionPop
            | PhiIRNode::Resonate { .. }
            | PhiIRNode::CoherenceCheck
            | PhiIRNode::Sleep { .. }
            | PhiIRNode::StreamPush(_)
            | PhiIRNode::StreamPop
            | PhiIRNode::Return(_)
            | PhiIRNode::Branch { .. }
            | PhiIRNode::Jump(_)
            | PhiIRNode::Fallthrough
            | PhiIRNode::Nop
            | PhiIRNode::Remember { .. }
            | PhiIRNode::Recall(_)
            | PhiIRNode::Broadcast { .. }
            | PhiIRNode::Listen(_)
            | PhiIRNode::AgentDecl { .. }
            | PhiIRNode::VoidDepth
            | PhiIRNode::Evolve(_)
            | PhiIRNode::Entangle(_)
            => false,
        }
    }

    fn unroll_loops(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;
        let mut loops_to_unroll = Vec::new();

        for block in &program.blocks {
            if let PhiIRNode::Branch {
                then_block,
                ..
            } = block.terminator
            {
                let header_id = block.id;
                let body_id = then_block;

                let body_jumps_back = program
                    .blocks
                    .iter()
                    .find(|b| b.id == body_id)
                    .is_some_and(
                        |b| matches!(b.terminator, PhiIRNode::Jump(target) if target == header_id),
                    );

                if body_jumps_back {
                    loops_to_unroll.push((header_id, body_id));
                }
            }
        }

        let mut next_block_id = program.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;
        let mut next_operand = Self::find_max_operand(program) + 1;

        for (header_id, body_id) in loops_to_unroll {
            let (check1_id, mut check1_block) = Self::clone_block(
                program,
                header_id,
                next_block_id,
                &mut next_operand,
                "_unroll_1",
            );
            next_block_id += 1;
            let (body1_id, mut body1_block) = Self::clone_block(
                program,
                body_id,
                next_block_id,
                &mut next_operand,
                "_unroll_1",
            );
            next_block_id += 1;

            let (check2_id, mut check2_block) = Self::clone_block(
                program,
                header_id,
                next_block_id,
                &mut next_operand,
                "_unroll_2",
            );
            next_block_id += 1;
            let (body2_id, mut body2_block) = Self::clone_block(
                program,
                body_id,
                next_block_id,
                &mut next_operand,
                "_unroll_2",
            );
            next_block_id += 1;

            if let Some(body_block) = program.blocks.iter_mut().find(|b| b.id == body_id) {
                body_block.terminator = PhiIRNode::Jump(check1_id);
            }

            if let PhiIRNode::Branch { then_block, .. } = &mut check1_block.terminator {
                *then_block = body1_id;
            }

            body1_block.terminator = PhiIRNode::Jump(check2_id);

            if let PhiIRNode::Branch { then_block, .. } = &mut check2_block.terminator {
                *then_block = body2_id;
            }

            program.blocks.push(check1_block);
            program.blocks.push(body1_block);
            program.blocks.push(check2_block);
            program.blocks.push(body2_block);

            changed = true;
        }

        changed
    }

    fn stabilize(&self, program: &mut PhiIRProgram) {
        if self.monitor.coherence_score < 0.618 {
            let mut next_operand = Self::find_max_operand(program) + 1;

            for block in &mut program.blocks {
                if matches!(block.terminator, PhiIRNode::Jump(_)) {
                    let duration_op = next_operand;
                    next_operand += 1;

                    let const_instr = PhiInstruction {
                        result: Some(duration_op),
                        node: PhiIRNode::Const(PhiIRValue::Number(16.0)),
                    };

                    let sleep_instr = PhiInstruction {
                        result: None,
                        node: PhiIRNode::Sleep {
                            duration: duration_op,
                        },
                    };

                    block.instructions.push(const_instr);
                    block.instructions.push(sleep_instr);
                }
            }
        }
    }

    fn find_max_operand(program: &PhiIRProgram) -> u32 {
        let mut max = 0;
        for block in &program.blocks {
            for instr in &block.instructions {
                if let Some(op) = instr.result {
                    if op > max {
                        max = op;
                    }
                }
            }
        }
        max
    }

    fn clone_block(
        program: &PhiIRProgram,
        source_id: BlockId,
        new_id: BlockId,
        next_operand: &mut u32,
        suffix: &str,
    ) -> (BlockId, PhiIRBlock) {
        let source = program.blocks.iter().find(|b| b.id == source_id).unwrap();
        let mut new_block = source.clone();
        new_block.id = new_id;
        new_block.label = format!("{}{}", source.label, suffix);

        let mut operand_map = HashMap::new();

        for instr in &mut new_block.instructions {
            if let Some(old_op) = instr.result {
                let new_op = *next_operand;
                *next_operand += 1;
                operand_map.insert(old_op, new_op);
                instr.result = Some(new_op);
            }
        }

        for instr in &mut new_block.instructions {
            Self::remap_operands(&mut instr.node, &operand_map);
        }
        Self::remap_operands(&mut new_block.terminator, &operand_map);

        (new_id, new_block)
    }

    fn remap_operands(node: &mut PhiIRNode, map: &HashMap<Operand, Operand>) {
        let mut map_op = |op: &mut Operand| {
            if let Some(&new_op) = map.get(op) {
                *op = new_op;
            }
        };

        match node {
            PhiIRNode::BinOp { left, right, .. } => {
                map_op(left);
                map_op(right);
            }
            PhiIRNode::UnaryOp { operand, .. } => {
                map_op(operand);
            }
            PhiIRNode::StoreVar { value, .. } => {
                map_op(value);
            }
            PhiIRNode::Call { args, .. }
            | PhiIRNode::ListNew(args)
            | PhiIRNode::DomainCall { args, .. } => {
                for arg in args {
                    map_op(arg);
                }
            }
            PhiIRNode::ListGet { list, index } => {
                map_op(list);
                map_op(index);
            }
            PhiIRNode::Return(val) => {
                map_op(val);
            }
            PhiIRNode::Branch { condition, .. } => {
                map_op(condition);
            }
            PhiIRNode::Witness {
                target: Some(t), ..
            } => {
                map_op(t);
            }
            PhiIRNode::Witness { target: None, .. } => {}
            PhiIRNode::Resonate {
                value: Some(v), ..
            } => {
                map_op(v);
            }
            PhiIRNode::Resonate { value: None, .. } => {}
            PhiIRNode::CreatePattern {
                frequency, params, ..
            } => {
                map_op(frequency);
                for (_, arg) in params {
                    map_op(arg);
                }
            }
            PhiIRNode::Sleep { duration } => {
                map_op(duration);
            }
            PhiIRNode::Remember { value, .. } => {
                map_op(value);
            }
            PhiIRNode::Broadcast { value, .. } => {
                map_op(value);
            }
            _ => {}
        }
    }
}
