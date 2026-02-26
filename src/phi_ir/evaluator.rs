//! PhiIR Direct Evaluator
//!
//! Interprets a `PhiIRProgram` directly, giving the four unique PhiFlow constructs
//! real, observable behavior:
//!
//! - `Witness`           → Captures program state; returns coherence score (0.0–1.0)
//! - `IntentionPush/Pop` → Maintains a live intention stack; scopes execution purpose
//! - `Resonate`          → Shares values through an intention-keyed resonance field
//! - `CoherenceCheck`    → Phi-harmonic coherence: depth 2 yields the golden ratio (0.618)
//!
//! Coherence formula: `1 - φ^(-depth)` + resonance bonus (max 0.2)
//!   depth 0 → 0.000 | depth 1 → 0.382 | depth 2 → 0.618 | depth 3 → 0.764

use crate::phi_ir::{
    BlockId, Operand, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue,
    PhiInstruction,
};
use std::collections::HashMap;

const PHI: f64 = 1.618033988749895;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum EvalError {
    BlockNotFound(BlockId),
    OperandNotFound(Operand),
    DivisionByZero,
    InvalidOperation(String),
    Unimplemented(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::BlockNotFound(id) => write!(f, "Block {} not found", id),
            EvalError::OperandNotFound(op) => write!(f, "Operand {} not found", op),
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::InvalidOperation(s) => write!(f, "Invalid operation: {}", s),
            EvalError::Unimplemented(s) => write!(f, "Unimplemented: {}", s),
        }
    }
}

type EvalResult<T> = Result<T, EvalError>;

// ---------------------------------------------------------------------------
// Witness event — observable output of `witness` construct
// ---------------------------------------------------------------------------

/// A snapshot of program state recorded each time `witness` executes.
#[derive(Debug, Clone)]
pub struct WitnessEvent {
    /// Active intention stack at the moment of observation (innermost last).
    pub intention_stack: Vec<String>,
    /// Phi-harmonic coherence score: 0.0 (no purpose) → 1.0 (fully aligned).
    pub coherence: f64,
    /// Number of SSA registers live at this point.
    pub register_count: usize,
    /// Total values shared through the resonance field across all intentions.
    pub resonance_count: usize,
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

pub struct Evaluator<'a> {
    program: &'a PhiIRProgram,
    functions: HashMap<String, FunctionMeta>,
    coherence_provider: Option<Box<dyn Fn() -> f64 + Send + Sync + 'a>>,

    /// SSA registers: instruction index (Operand) → computed value.
    registers: HashMap<Operand, PhiIRValue>,

    /// Named variable store, for `LoadVar` / `StoreVar` (mutable bindings).
    variables: HashMap<String, PhiIRValue>,

    /// Active intention stack. Each `IntentionPush` pushes a name; `IntentionPop` removes it.
    intention_stack: Vec<String>,

    /// Tracks active stream loop names to determine if resonance overwrites or appends.
    active_streams: Vec<String>,

    /// Resonance field: intention name → values shared via `resonate`.
    /// The special key "global" is used when no intention is active.
    resonance_field: HashMap<String, Vec<PhiIRValue>>,
    /// Ordered resonance events (scope, value) for CLI/diagnostic output.
    resonance_events: Vec<(String, PhiIRValue)>,
    /// Stream names that exited via StreamPop.
    ended_streams: Vec<String>,

    /// Every `Witness` execution appends an event here.
    pub witness_log: Vec<WitnessEvent>,

    current_block: BlockId,
    instruction_ptr: usize,
}

#[derive(Debug, Clone)]
struct FunctionMeta {
    params: Vec<String>,
    body: BlockId,
}

impl<'a> Evaluator<'a> {
    pub fn new(program: &'a PhiIRProgram) -> Self {
        let mut functions = HashMap::new();
        for block in &program.blocks {
            for instr in &block.instructions {
                if let PhiIRNode::FuncDef { name, params, body } = &instr.node {
                    functions.insert(
                        name.clone(),
                        FunctionMeta {
                            params: params.iter().map(|p| p.name.clone()).collect(),
                            body: *body,
                        },
                    );
                }
            }
        }

        Self {
            program,
            functions,
            coherence_provider: None,
            registers: HashMap::new(),
            variables: HashMap::new(),
            intention_stack: Vec::new(),
            active_streams: Vec::new(),
            resonance_field: HashMap::new(),
            resonance_events: Vec::new(),
            ended_streams: Vec::new(),
            witness_log: Vec::new(),
            current_block: program.blocks.first().map(|b| b.id).unwrap_or(0),
            instruction_ptr: 0,
        }
    }

    pub fn with_coherence_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn() -> f64 + Send + Sync + 'a,
    {
        self.coherence_provider = Some(Box::new(provider));
        self
    }

    /// Run the program to completion. Returns the final value.
    pub fn run(&mut self) -> EvalResult<PhiIRValue> {
        loop {
            let block_id = self.current_block;
            let block = self.get_block(block_id)?;
            let instr_count = block.instructions.len();

            if self.instruction_ptr < instr_count {
                let instr = block.instructions[self.instruction_ptr].clone();
                self.instruction_ptr += 1;
                self.execute_instruction(&instr)?;
            } else {
                let terminator = block.terminator.clone();
                if let Some(value) = self.execute_terminator(&terminator)? {
                    return Ok(value);
                }
            }
        }
    }

    /// Expose current coherence so callers can inspect without running `witness`.
    pub fn coherence(&self) -> f64 {
        self.compute_coherence()
    }

    /// Expose a read-only view of values resonated under a given intention.
    pub fn resonated_values(&self, intention: &str) -> &[PhiIRValue] {
        self.resonance_field
            .get(intention)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Ordered resonance events produced during execution.
    pub fn resonance_events(&self) -> &[(String, PhiIRValue)] {
        self.resonance_events.as_slice()
    }

    /// Stream names that completed execution.
    pub fn ended_streams(&self) -> &[String] {
        self.ended_streams.as_slice()
    }

    pub fn resonance_field(&self) -> &HashMap<String, Vec<PhiIRValue>> {
        &self.resonance_field
    }

    // -----------------------------------------------------------------------
    // Block lookup
    // -----------------------------------------------------------------------

    fn get_block(&self, id: BlockId) -> EvalResult<&'a PhiIRBlock> {
        self.program
            .blocks
            .iter()
            .find(|b| b.id == id)
            .ok_or(EvalError::BlockNotFound(id))
    }

    // -----------------------------------------------------------------------
    // Instruction execution
    // -----------------------------------------------------------------------

    fn execute_instruction(&mut self, instr: &PhiInstruction) -> EvalResult<()> {
        let value: Option<PhiIRValue> = match &instr.node {
            // --- Primitives ---
            PhiIRNode::Nop => None,
            PhiIRNode::Const(v) => Some(v.clone()),

            // --- Variables ---
            PhiIRNode::LoadVar(name) => {
                let val = self
                    .variables
                    .get(name)
                    .cloned()
                    .unwrap_or(PhiIRValue::Void);
                Some(val)
            }
            PhiIRNode::StoreVar { name, value } => {
                let val = self.get_reg(*value)?.clone();
                self.variables.insert(name.clone(), val);
                None
            }

            // --- Arithmetic & Logic ---
            PhiIRNode::BinOp { op, left, right } => Some(self.eval_binop(op, *left, *right)?),
            PhiIRNode::UnaryOp { op, operand } => Some(self.eval_unop(op, *operand)?),
            PhiIRNode::Call { name, args } => {
                let mut arg_values = Vec::with_capacity(args.len());
                for arg in args {
                    arg_values.push(self.get_reg(*arg)?.clone());
                }
                Some(self.execute_function(name, arg_values)?)
            }
            PhiIRNode::FuncDef { .. } => None,

            // --- The Four Unique PhiFlow Constructs ---
            // `witness` pauses to observe program state.
            // Returns coherence score as a Number (0.0–1.0).
            // Records a WitnessEvent in witness_log.
            PhiIRNode::Witness { target, .. } => {
                // Optionally observe a specific operand (for future use by backends).
                let _observed = target.and_then(|op| self.get_reg(op).ok().cloned());

                let coherence = self.compute_coherence();
                let resonance_count: usize = self.resonance_field.values().map(|v| v.len()).sum();

                self.witness_log.push(WitnessEvent {
                    intention_stack: self.intention_stack.clone(),
                    coherence,
                    register_count: self.registers.len(),
                    resonance_count,
                });

                Some(PhiIRValue::Number(coherence))
            }

            // `intention "Name" { ... }` enters a purposeful scope.
            // Pushes intention name and initialises its resonance field slot.
            PhiIRNode::IntentionPush { name, .. } => {
                self.intention_stack.push(name.clone());
                self.resonance_field.entry(name.clone()).or_default();
                None
            }

            // Exit intention scope. Pops the innermost intention name.
            PhiIRNode::IntentionPop => {
                self.intention_stack.pop();
                None
            }

            PhiIRNode::StreamPush(name) => {
                self.intention_stack.push(name.clone());
                self.active_streams.push(name.clone());
                // Clear the resonance field for the beginning of the stream execution
                self.resonance_field.insert(name.clone(), Vec::new());
                None
            }

            PhiIRNode::StreamPop => {
                self.intention_stack.pop();
                if let Some(stream_name) = self.active_streams.pop() {
                    self.ended_streams.push(stream_name);
                }
                None
            }

            // `resonate value` shares a value through the resonance field.
            // Filed under the current intention name, or "global" if none active.
            PhiIRNode::Resonate { value, .. } => {
                let key = self
                    .intention_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "global".to_string());

                if let Some(op) = value {
                    if let Ok(val) = self.get_reg(*op) {
                        let val = val.clone();
                        self.resonance_events.push((key.clone(), val.clone()));
                        if self.active_streams.contains(&key) {
                            // Stream constructs are continuous presents, they overwrite their cycle value.
                            self.resonance_field.insert(key, vec![val]);
                        } else {
                            // Standard intention blocks append their state.
                            self.resonance_field.entry(key).or_default().push(val);
                        }
                    }
                }
                None
            }

            // `coherence` evaluates and returns current phi-harmonic coherence.
            PhiIRNode::CoherenceCheck => Some(PhiIRValue::Number(self.resolve_coherence())),

            // --- Domain calls: no-op in base evaluator (backend-specific) ---
            PhiIRNode::DomainCall { .. } => None,
            PhiIRNode::CreatePattern { .. } => None,

            // --- Phi-Harmonic stabilization ---
            // `Sleep { duration }` is a harmonic pause for self-healing.
            // In the evaluator, we record the intent but don't actually sleep
            // (tests would hang). A real runtime backend can override this.
            PhiIRNode::Sleep { .. } => None,

            // Control flow nodes are terminators — handled in execute_terminator
            PhiIRNode::Branch { .. }
            | PhiIRNode::Jump(_)
            | PhiIRNode::Return(_)
            | PhiIRNode::Fallthrough => None,

            other => {
                return Err(EvalError::Unimplemented(format!(
                    "Instruction {:?} not implemented in base evaluator",
                    other
                )))
            }
        };

        // Store result in SSA register if this instruction produces a value.
        if let (Some(val), Some(reg)) = (value, instr.result) {
            self.registers.insert(reg, val);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Terminator execution — returns Some(value) to halt, None to continue
    // -----------------------------------------------------------------------

    fn execute_terminator(&mut self, node: &PhiIRNode) -> EvalResult<Option<PhiIRValue>> {
        match node {
            PhiIRNode::Return(op) => {
                let val = self.get_reg(*op)?.clone();
                Ok(Some(val))
            }

            PhiIRNode::Jump(target) => {
                self.current_block = *target;
                self.instruction_ptr = 0;
                Ok(None)
            }

            PhiIRNode::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.get_reg(*condition)?;
                let target = match cond {
                    PhiIRValue::Boolean(true) => *then_block,
                    PhiIRValue::Boolean(false) => *else_block,
                    PhiIRValue::Number(n) => {
                        if *n != 0.0 {
                            *then_block
                        } else {
                            *else_block
                        }
                    }
                    _ => *else_block,
                };
                self.current_block = target;
                self.instruction_ptr = 0;
                Ok(None)
            }

            PhiIRNode::Fallthrough => {
                // Advance to the next physical block in program order.
                let current_idx = self
                    .program
                    .blocks
                    .iter()
                    .position(|b| b.id == self.current_block)
                    .unwrap_or(0);

                if current_idx + 1 < self.program.blocks.len() {
                    self.current_block = self.program.blocks[current_idx + 1].id;
                    self.instruction_ptr = 0;
                    Ok(None)
                } else {
                    Ok(Some(PhiIRValue::Void))
                }
            }

            other => Err(EvalError::Unimplemented(format!(
                "Terminator {:?} not implemented",
                other
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Phi-harmonic coherence
    // -----------------------------------------------------------------------

    /// Compute the phi-harmonic coherence score (0.0–1.0).
    ///
    /// Intention depth drives coherence via `1 - φ^(-depth)`:
    ///   depth 0 → 0.000  (no declared purpose)
    ///   depth 1 → 0.382  (one intention active)
    ///   depth 2 → 0.618  (golden ratio — two aligned intentions)
    ///   depth 3 → 0.764
    ///   depth 4 → 0.854
    ///
    /// Resonance connections add up to 0.200 more (0.05 per shared value).
    fn compute_coherence(&self) -> f64 {
        let depth = self.intention_stack.len();
        let resonance_count: usize = self.resonance_field.values().map(|v| v.len()).sum();

        if depth == 0 && resonance_count == 0 {
            return 0.0;
        }

        let intention_coherence = if depth > 0 {
            1.0 - PHI.powi(-(depth as i32))
        } else {
            0.0
        };

        let resonance_bonus = (resonance_count as f64 * 0.05).min(0.2);

        (intention_coherence + resonance_bonus).min(1.0)
    }

    fn resolve_coherence(&self) -> f64 {
        if let Some(provider) = &self.coherence_provider {
            provider()
        } else {
            self.compute_coherence()
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn get_reg(&self, op: Operand) -> EvalResult<&PhiIRValue> {
        self.registers
            .get(&op)
            .ok_or(EvalError::OperandNotFound(op))
    }

    fn eval_binop(&self, op: &PhiIRBinOp, left: Operand, right: Operand) -> EvalResult<PhiIRValue> {
        let l = self.get_reg(left)?;
        let r = self.get_reg(right)?;

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
                PhiIRBinOp::Mod => Ok(PhiIRValue::Number(lhs % rhs)),
                PhiIRBinOp::Pow => Ok(PhiIRValue::Number(lhs.powf(*rhs))),
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean((lhs - rhs).abs() < f64::EPSILON)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean((lhs - rhs).abs() >= f64::EPSILON)),
                PhiIRBinOp::Lt => Ok(PhiIRValue::Boolean(lhs < rhs)),
                PhiIRBinOp::Lte => Ok(PhiIRValue::Boolean(lhs <= rhs)),
                PhiIRBinOp::Gt => Ok(PhiIRValue::Boolean(lhs > rhs)),
                PhiIRBinOp::Gte => Ok(PhiIRValue::Boolean(lhs >= rhs)),
                _ => Err(EvalError::Unimplemented(format!(
                    "BinOp {:?} not supported for Numbers",
                    op
                ))),
            },
            (PhiIRValue::Boolean(l), PhiIRValue::Boolean(r)) => match op {
                PhiIRBinOp::And => Ok(PhiIRValue::Boolean(*l && *r)),
                PhiIRBinOp::Or => Ok(PhiIRValue::Boolean(*l || *r)),
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(l == r)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(l != r)),
                _ => Err(EvalError::InvalidOperation(
                    "Unsupported boolean binary op".to_string(),
                )),
            },
            _ => Err(EvalError::InvalidOperation(
                "Type mismatch in binary operation".to_string(),
            )),
        }
    }

    fn eval_unop(&self, op: &PhiIRUnOp, operand: Operand) -> EvalResult<PhiIRValue> {
        let val = self.get_reg(operand)?;
        match val {
            PhiIRValue::Number(n) => match op {
                PhiIRUnOp::Neg => Ok(PhiIRValue::Number(-n)),
                PhiIRUnOp::Not => Ok(PhiIRValue::Boolean(*n == 0.0)),
            },
            PhiIRValue::Boolean(b) => match op {
                PhiIRUnOp::Not => Ok(PhiIRValue::Boolean(!b)),
                _ => Err(EvalError::InvalidOperation(
                    "Neg not supported on Boolean".to_string(),
                )),
            },
            _ => Err(EvalError::InvalidOperation(
                "Unary op on unsupported type".to_string(),
            )),
        }
    }

    fn execute_function(&mut self, name: &str, args: Vec<PhiIRValue>) -> EvalResult<PhiIRValue> {
        let meta = self
            .functions
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::InvalidOperation(format!("Undefined function: {}", name)))?;

        let saved_block = self.current_block;
        let saved_ip = self.instruction_ptr;
        let saved_variables = std::mem::take(&mut self.variables);

        self.variables = HashMap::new();
        for (idx, param_name) in meta.params.iter().enumerate() {
            let value = args.get(idx).cloned().unwrap_or(PhiIRValue::Void);
            self.variables.insert(param_name.clone(), value);
        }

        self.current_block = meta.body;
        self.instruction_ptr = 0;

        let result = loop {
            let block_id = self.current_block;
            let block = self.get_block(block_id)?.clone();

            if self.instruction_ptr < block.instructions.len() {
                let instr = block.instructions[self.instruction_ptr].clone();
                self.instruction_ptr += 1;
                self.execute_instruction(&instr)?;
                continue;
            }

            let terminator = block.terminator.clone();
            if let Some(value) = self.execute_terminator(&terminator)? {
                break value;
            }
        };

        self.variables = saved_variables;
        self.current_block = saved_block;
        self.instruction_ptr = saved_ip;

        Ok(result)
    }
}
