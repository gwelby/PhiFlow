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
        // A "Living" code should have structure.
        // Arbitrary metric: Ratio of Arithmetic / Control Flow should be close to Phi?
        // Or strictly: We just measure complexity.
        // Let's use a simple placeholder:
        // Coherence = 1.0 / (1.0 + |(Arithmetic / Control) - Phi|)
        // If Control is 0, score is 0.5 (imperfect but safe).

        const PHI: f64 = 1.6180339887;

        if self.control_flow_count == 0 {
            self.coherence_score = 0.5;
            return;
        }

        let ratio = self.arithmetic_count as f64 / self.control_flow_count as f64;
        let deviation = (ratio - PHI).abs();

        // Higher deviation = Lower score.
        // Max score = 1.0 (perfect Phi alignment).
        self.coherence_score = 1.0 / (1.0 + deviation);
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

        // 3. Phi-Harmonic Loop Unrolling (Fibonacci)
        // Runs once after standard optimizations converge (or max passes).
        if self.level == OptimizationLevel::PhiHarmonic {
            // println!("Phi-Harmonic Unrolling...");
            if Self::unroll_loops(program) {
                // If unrolled, run clean-up passes (DCE) again?
                // Ideally yes, but let's stick to the unrolling first.
            }
        }

        // 4. Coherence Check (Phi-Harmonic)
        if self.level == OptimizationLevel::PhiHarmonic {
            self.monitor.analyze(program);
            // println!("Phi-Harmonic Coherence Score: {}", self.monitor.coherence_score);
            self.stabilize(program);
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
            PhiIRNode::Sleep { duration } => {
                used.insert(*duration);
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
            | PhiIRNode::Sleep { .. }
            | PhiIRNode::Return(_)
            | PhiIRNode::Branch { .. }
            | PhiIRNode::Jump(_)
            | PhiIRNode::Fallthrough
            | PhiIRNode::Nop
            => false,
        }
    }

    /// Loop Unrolling Pass (Phi-Harmonic)
    /// Unrolls loops by a Fibonacci factor (3 for now).
    fn unroll_loops(program: &mut PhiIRProgram) -> bool {
        let mut changed = false;
        let mut loops_to_unroll = Vec::new();

        // 1. Identify Loops (Simple header-body pattern from lowering)
        // Look for: Header -> Branch(cond, Body, Exit)
        //           Body   -> Jump(Header)
        for block in &program.blocks {
            if let PhiIRNode::Branch {
                condition: _,
                then_block,
                else_block: _,
            } = block.terminator
            {
                let header_id = block.id;
                let body_id = then_block;

                // Check if body jumps back to header
                let body_jumps_back = program.blocks.iter().find(|b| b.id == body_id).map_or(
                    false,
                    |b| matches!(b.terminator, PhiIRNode::Jump(target) if target == header_id),
                );

                if body_jumps_back {
                    loops_to_unroll.push((header_id, body_id));
                }
            }
        }

        // 2. Unroll
        // Factor 3 (Fibonacci): Header -> Body1 -> Check1 -> Body2 -> Check2 -> Body3 -> Header
        // Original: Header -> Body -> Header

        let mut next_block_id = program.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;
        let mut next_operand = Self::find_max_operand(program) + 1;

        for (header_id, body_id) in loops_to_unroll {
            // Factor 3 means we need 2 clones of (Check+Body) inserted.

            // Clone 1
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

            // Clone 2
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

            // Wiring

            // Update Original Body -> Jump(Check1)
            if let Some(body_block) = program.blocks.iter_mut().find(|b| b.id == body_id) {
                body_block.terminator = PhiIRNode::Jump(check1_id);
            }

            // Update Check1 -> Branch(cond, Body1, Exit)
            if let PhiIRNode::Branch { then_block, .. } = &mut check1_block.terminator {
                *then_block = body1_id;
            }

            // Update Body1 -> Jump(Check2)
            body1_block.terminator = PhiIRNode::Jump(check2_id);

            // Update Check2 -> Branch(cond, Body2, Exit)
            if let PhiIRNode::Branch { then_block, .. } = &mut check2_block.terminator {
                *then_block = body2_id;
            }

            // Update Body2 -> Jump(Header) - Preserved from clone

            // Add new blocks to program
            program.blocks.push(check1_block);
            program.blocks.push(body1_block);
            program.blocks.push(check2_block);
            program.blocks.push(body2_block);

            changed = true;
        }

        changed
    }

    /// Stabilize the flow if coherence is too low.
    /// Injects 'Sleep' instructions to slow down chaotic loops.
    fn stabilize(&self, program: &mut PhiIRProgram) {
        // Threshold: 1/Phi approx 0.618
        if self.monitor.coherence_score < 0.618 {
            // println!("[Stabilize] Coherence low ({:.3}). Stabilizing...", self.monitor.coherence_score);

            // We need a fresh operand for the Sleep duration constant.
            let mut next_operand = Self::find_max_operand(program) + 1;

            // We'll inject Sleep(16ms) into blocks that end with a Jump (likely loop back-edges).
            for block in &mut program.blocks {
                if matches!(block.terminator, PhiIRNode::Jump(_)) {
                    // 1. Create constant 16.0
                    let duration_op = next_operand;
                    next_operand += 1;

                    let const_instr = PhiInstruction {
                        result: Some(duration_op),
                        node: PhiIRNode::Const(PhiIRValue::Number(16.0)),
                    };

                    // 2. Create Sleep instruction
                    let sleep_instr = PhiInstruction {
                        result: None,
                        node: PhiIRNode::Sleep {
                            duration: duration_op,
                        },
                    };

                    // 3. Insert before terminator
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

        // Remap definitions
        for instr in &mut new_block.instructions {
            if let Some(old_op) = instr.result {
                let new_op = *next_operand;
                *next_operand += 1;
                operand_map.insert(old_op, new_op);
                instr.result = Some(new_op);
            }
        }

        // Remap usages (inputs) using the map
        for instr in &mut new_block.instructions {
            Self::remap_operands(&mut instr.node, &operand_map);
        }
        // Remap terminator usages
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
            PhiIRNode::Witness { target, .. } => {
                if let Some(t) = target {
                    map_op(t);
                }
            }
            PhiIRNode::Resonate { value, .. } => {
                if let Some(v) = value {
                    map_op(v);
                }
            }
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
            _ => {}
        }
    }
}
