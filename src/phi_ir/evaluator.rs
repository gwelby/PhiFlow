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

use crate::host::{DefaultHostProvider, PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::parser::parse_phi_program;
use crate::phi_ir::{
    lowering::lower_program,
    vm_state::{VmState, VmWitnessEvent},
    BlockId, Operand, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue,
    PhiInstruction,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
    SynthesisError(String),
    StepLimitExceeded(usize),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::BlockNotFound(id) => write!(f, "Block {} not found", id),
            EvalError::OperandNotFound(op) => write!(f, "Operand {} not found", op),
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::InvalidOperation(s) => write!(f, "Invalid operation: {}", s),
            EvalError::Unimplemented(s) => write!(f, "Unimplemented: {}", s),
            EvalError::SynthesisError(s) => write!(f, "Synthesis error: {}", s),
            EvalError::StepLimitExceeded(limit) => {
                write!(f, "Execution step limit exceeded: {} steps", limit)
            }
        }
    }
}

type EvalResult<T> = Result<T, EvalError>;

// ---------------------------------------------------------------------------
// Execution result — supports yield/resume for host-controlled witness
// ---------------------------------------------------------------------------

/// The result of running an evaluator to completion or yield.
#[derive(Debug, Clone)]
pub enum VmExecResult {
    /// Program completed normally with a final value.
    Complete(PhiIRValue),
    /// Program yielded at a `witness` statement. Contains the frozen state
    /// needed to resume, plus the witness snapshot that triggered the yield.
    Yielded {
        snapshot: WitnessSnapshot,
        frozen_state: FrozenEvalState,
    },
    /// Program yielded to synchronize with other entangled streams.
    Entangled {
        frequency: f64,
        frozen_state: FrozenEvalState,
    },
}

/// Backward-compatible alias for older call sites.
pub type EvalExecResult = VmExecResult;

/// Serializable evaluator state used for yield/resume.
pub type FrozenEvalState = VmState;

// ---------------------------------------------------------------------------
// Witness event — observable output of `witness` construct
// ---------------------------------------------------------------------------

/// Snapshot entry for each witness event.
pub type WitnessEvent = VmWitnessEvent;

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

pub struct Evaluator<'a> {
    program: PhiIRProgram,
    functions: HashMap<String, FunctionMeta>,
    host: Box<dyn PhiHostProvider + 'a>,

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
    /// Shared resonance field across multiple evaluators
    shared_resonance: Option<Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>>,
    /// Ordered resonance events (scope, value) for CLI/diagnostic output.
    resonance_events: Vec<(String, PhiIRValue)>,
    /// Stream names that exited via StreamPop.
    ended_streams: Vec<String>,

    /// Every `Witness` execution appends an event here.
    pub witness_log: Vec<WitnessEvent>,

    current_block: BlockId,
    instruction_ptr: usize,

    // --- v0.3.0 Metadata ---
    pub agent_name: Option<String>,
    pub agent_version: Option<String>,
    pub yield_timestamp: Option<f64>,

    // --- Guardrails ---
    pub max_steps: Option<usize>,
    pub step_count: usize,
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

        let mut variables = HashMap::new();
        variables.insert("PHI".to_string(), PhiIRValue::Number(PHI));

        Self {
            program: program.clone(),
            functions,
            host: Box::new(DefaultHostProvider),
            registers: HashMap::new(),
            variables,
            intention_stack: Vec::new(),
            active_streams: Vec::new(),
            resonance_field: HashMap::new(),
            shared_resonance: None,
            resonance_events: Vec::new(),
            ended_streams: Vec::new(),
            witness_log: Vec::new(),
            current_block: program.entry,
            instruction_ptr: 0,
            agent_name: None,
            agent_version: None,
            yield_timestamp: None,
            max_steps: None,
            step_count: 0,
        }
    }

    /// Set a custom host provider. This replaces the default host.
    pub fn with_host(mut self, host: Box<dyn PhiHostProvider + 'a>) -> Self {
        self.host = host;
        self
    }

    /// Link this evaluator to a globally shared resonance field.
    pub fn with_shared_resonance(
        mut self,
        shared: Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>,
    ) -> Self {
        self.shared_resonance = Some(shared);
        self
    }

    /// Set an execution step limit to prevent infinite loops.
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Backwards-compatible: set a coherence provider closure.
    pub fn with_coherence_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn() -> f64 + Send + Sync + 'static,
    {
        use crate::host::CallbackHostProvider;
        self.host =
            Box::new(CallbackHostProvider::new().with_coherence(move |_internal| provider()));
        self
    }

    /// Run the program to completion. Returns the final value.
    pub fn run(&mut self) -> EvalResult<PhiIRValue> {
        match self.run_or_yield()? {
            VmExecResult::Complete(value) => Ok(value),
            VmExecResult::Yielded { .. } => Ok(PhiIRValue::Number(self.compute_coherence())),
            VmExecResult::Entangled { .. } => Ok(PhiIRValue::Number(self.compute_coherence())),
        }
    }

    /// Run the program, but may return `Yielded` if a `witness` triggers
    /// a host-requested yield. The caller can inspect the frozen state
    /// and call `resume()` to continue.
    pub fn run_or_yield(&mut self) -> EvalResult<VmExecResult> {
        let mut loop_counter = 0;
        loop {
            if let Some(max) = self.max_steps {
                if self.step_count > max {
                    return Err(EvalError::StepLimitExceeded(max));
                }
            }
            self.step_count += 1;

            loop_counter += 1;
            if loop_counter > 50000 && self.max_steps.is_none() {
                panic!(
                    "Infinite loop detected in evaluator. current_block: {}, ip: {}",
                    self.current_block, self.instruction_ptr
                );
            }
            let block_id = self.current_block;
            let block = self.get_block(block_id)?;
            let instr_count = block.instructions.len();

            if self.instruction_ptr < instr_count {
                let instr = block.instructions[self.instruction_ptr].clone();
                self.instruction_ptr += 1;
                if let Some(yield_result) = self.execute_instruction_with_yield(&instr)? {
                    return Ok(yield_result);
                }
            } else {
                let terminator = block.terminator.clone();
                if let Some(value) = self.execute_terminator(&terminator)? {
                    return Ok(VmExecResult::Complete(value));
                }
            }
        }
    }

    /// Resume execution after a yield. Restores frozen state and continues.
    pub fn resume(&mut self, state: FrozenEvalState) -> EvalResult<VmExecResult> {
        self.registers = state.registers;
        self.variables = state.variables;
        self.intention_stack = state.intention_stack;
        self.active_streams = state.active_streams;
        self.resonance_field = state.resonance_field;
        self.resonance_events = state.resonance_events;
        self.ended_streams = state.ended_streams;
        self.witness_log = state.witness_log;
        self.current_block = state.current_block;
        self.instruction_ptr = state.instruction_ptr;
        self.yield_timestamp = state.yield_timestamp;
        self.agent_name = state.agent_name;
        self.agent_version = state.agent_version;
        self.run_or_yield()
    }

    /// Capture the current evaluator state as a frozen snapshot.
    fn freeze_state(&self) -> FrozenEvalState {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        FrozenEvalState {
            registers: self.registers.clone(),
            variables: self.variables.clone(),
            intention_stack: self.intention_stack.clone(),
            active_streams: self.active_streams.clone(),
            resonance_field: self.resonance_field.clone(),
            resonance_events: self.resonance_events.clone(),
            ended_streams: self.ended_streams.clone(),
            witness_log: self.witness_log.clone(),
            current_block: self.current_block,
            instruction_ptr: self.instruction_ptr,
            yield_timestamp: Some(now),
            agent_name: self.agent_name.clone(),
            agent_version: self.agent_version.clone(),
        }
    }

    /// Expose current coherence so callers can inspect without running `witness`.
    pub fn coherence(&self) -> f64 {
        self.compute_coherence()
    }

    /// Expose the host-resolved coherence, including injected hardware metrics.
    pub fn resolved_coherence(&self) -> f64 {
        self.resolve_coherence()
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

    fn get_block(&self, id: BlockId) -> EvalResult<PhiIRBlock> {
        self.program
            .blocks
            .iter()
            .find(|b| b.id == id)
            .cloned()
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
            PhiIRNode::Witness { target, .. } => {
                let (coherence, _snapshot, _action) = self.process_witness(*target)?;
                Some(PhiIRValue::Number(coherence))
            }

            PhiIRNode::IntentionPush { name, .. } => {
                self.intention_stack.push(name.clone());
                self.resonance_field.entry(name.clone()).or_default();
                self.host.on_intention_push(name);
                None
            }

            PhiIRNode::IntentionPop => {
                let popped = self.intention_stack.pop().unwrap_or_default();
                self.host.on_intention_pop(&popped);
                None
            }

            PhiIRNode::StreamPush(name) => {
                self.intention_stack.push(name.clone());
                self.active_streams.push(name.clone());
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

            PhiIRNode::Resonate { value, direction: _, .. } => {
                let key = self
                    .intention_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "global".to_string());

                if let Some(op) = value {
                    if let Ok(val) = self.get_reg(*op) {
                        let val = val.clone();
                        let val_str = self.value_to_string(&val);
                        self.resonance_events.push((key.clone(), val.clone()));
                        if self.active_streams.contains(&key) {
                            self.resonance_field.insert(key.clone(), vec![val.clone()]);
                            if let Some(shared) = &self.shared_resonance {
                                let mut guard = shared.lock().unwrap();
                                guard.insert(key.clone(), vec![val.clone()]);
                            }
                        } else {
                            self.resonance_field
                                .entry(key.clone())
                                .or_default()
                                .push(val.clone());
                            if let Some(shared) = &self.shared_resonance {
                                let mut guard = shared.lock().unwrap();
                                guard.entry(key.clone()).or_default().push(val.clone());
                            }
                        }
                        self.host.on_resonate(&key, &val_str);
                    }
                }
                None
            }

            PhiIRNode::CoherenceCheck => Some(PhiIRValue::Number(self.resolve_coherence())),

            // --- v0.3.0 Persistence & Dialogue ---
            PhiIRNode::Remember { key, value } => {
                let val = self.get_reg(*value)?;
                let val_str = self.value_to_string(val);
                self.host.persist(key, &val_str);
                None
            }
            PhiIRNode::Recall(key) => {
                if let Some(val_str) = self.host.recall(key) {
                    Some(self.string_to_value(&val_str))
                } else {
                    Some(PhiIRValue::Void)
                }
            }
            PhiIRNode::Broadcast { channel, value } => {
                let val = self.get_reg(*value)?;
                let val_str = self.value_to_string(val);
                self.host.broadcast(channel, &val_str);
                None
            }
            PhiIRNode::Listen(channel) => {
                if let Some(val_str) = self.host.listen(channel) {
                    Some(self.string_to_value(&val_str))
                } else {
                    Some(PhiIRValue::Void)
                }
            }

            PhiIRNode::AgentDecl { name, version } => {
                self.agent_name = Some(name.clone());
                self.agent_version = Some(version.clone());
                None
            }

            PhiIRNode::VoidDepth => {
                if let Some(yield_ts) = self.yield_timestamp {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    Some(PhiIRValue::Number(now - yield_ts))
                } else {
                    Some(PhiIRValue::Number(0.0))
                }
            }

            // --- v0.4.0 Strategic Capabilities ---
            PhiIRNode::Evolve(op) => {
                let code_val = self.get_reg(*op)?;
                let code_str = self.value_to_string(code_val);

                // 1. Compile the evolved code
                let exprs = parse_phi_program(&code_str)
                    .map_err(|e| EvalError::SynthesisError(format!("Parse failed: {}", e)))?;
                let evolved_prog = lower_program(&exprs);

                // 2. Splice blocks into the current program
                // We need to offset BlockIds to avoid collisions.
                let max_id = self.program.blocks.iter().map(|b| b.id).max().unwrap_or(0);
                let id_offset = max_id + 1;

                for mut block in evolved_prog.blocks.clone() {
                    block.id += id_offset;
                    // Remap internal jumps/branches
                    self.remap_block_ids(&mut block.terminator, id_offset);
                    self.program.blocks.push(block);
                }

                // Log the mutation to the resonance field (Fossil Record)
                let msg = format!(
                    "Stream evolved logic at {:.3}s (offset={})",
                    self.coherence(),
                    id_offset
                );
                self.resonance_events
                    .push(("_evolution".to_string(), PhiIRValue::Void));
                self.resonance_field
                    .entry("_evolution".to_string())
                    .or_default();
                self.host.on_resonate("_evolution", &msg);

                // 3. Execute the evolved blocks as a nested context,
                // saving the current block/IP so we resume cleanly.
                let saved_block = self.current_block;
                let saved_ip = self.instruction_ptr;

                self.current_block = evolved_prog.entry + id_offset;
                self.instruction_ptr = 0;

                let evolved_result = loop {
                    let block_id = self.current_block;
                    let block = self.get_block(block_id)?.clone();

                    if self.instruction_ptr < block.instructions.len() {
                        let instr = block.instructions[self.instruction_ptr].clone();
                        self.instruction_ptr += 1;
                        self.execute_instruction(&instr)?;
                        continue;
                    }

                    let terminator = block.terminator.clone();
                    if let Some(val) = self.execute_terminator(&terminator)? {
                        break val;
                    }
                };

                // Restore control to the caller block
                self.current_block = saved_block;
                self.instruction_ptr = saved_ip;

                println!("Evolve returned: {:?}", evolved_result);

                Some(evolved_result)
            }

            PhiIRNode::Entangle(freq) => {
                // Handled by execute_instruction_with_yield
                None
            }

            // --- Domain calls: no-op in base evaluator ---
            PhiIRNode::DomainCall { .. } => Some(PhiIRValue::Void),
            PhiIRNode::CreatePattern { .. } => Some(PhiIRValue::Void),
            PhiIRNode::Sleep { .. } => Some(PhiIRValue::Void),

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

        if let (Some(val), Some(reg)) = (value, instr.result) {
            self.registers.insert(reg, val);
        }

        Ok(())
    }

    fn value_to_string(&self, val: &PhiIRValue) -> String {
        match val {
            PhiIRValue::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    format!("{:.1}", n)
                } else {
                    n.to_string()
                }
            }
            PhiIRValue::Boolean(b) => b.to_string(),
            PhiIRValue::Void => "void".to_string(),
            PhiIRValue::String(idx) => {
                // If it's a string from the program table, we can resolve it
                if let Some(s) = self.program.string_table.get(*idx as usize) {
                    s.clone()
                } else {
                    format!("_str_{}", idx)
                }
            }
        }
    }

    fn string_to_value(&self, s: &str) -> PhiIRValue {
        if let Ok(n) = s.parse::<f64>() {
            PhiIRValue::Number(n)
        } else if s == "true" {
            PhiIRValue::Boolean(true)
        } else if s == "false" {
            PhiIRValue::Boolean(false)
        } else if s == "void" {
            PhiIRValue::Void
        } else {
            // Treat unknown strings as Void for now, or could intern them if we had a mutable table
            PhiIRValue::Void
        }
    }

    fn resonance_count(&self) -> usize {
        self.resonance_field.values().map(|v| v.len()).sum()
    }

    fn process_witness(
        &mut self,
        target: Option<Operand>,
    ) -> EvalResult<(f64, WitnessSnapshot, WitnessAction)> {
        let observed = target.and_then(|op| self.get_reg(op).ok().cloned());
        let coherence = self.compute_coherence();
        let resonance_count = self.resonance_count();

        self.witness_log.push(WitnessEvent {
            intention_stack: self.intention_stack.clone(),
            coherence,
            register_count: self.registers.len(),
            resonance_count,
            agent_name: self.agent_name.clone(),
        });

        let snapshot = WitnessSnapshot {
            intention_stack: self.intention_stack.clone(),
            coherence,
            register_count: self.registers.len(),
            resonance_count,
            observed_value: observed.map(|v| self.value_to_string(&v)),
            agent_name: self.agent_name.clone(),
        };
        let action = self.host.on_witness(&snapshot);

        Ok((coherence, snapshot, action))
    }

    fn execute_instruction_with_yield(
        &mut self,
        instr: &PhiInstruction,
    ) -> EvalResult<Option<VmExecResult>> {
        if let PhiIRNode::Witness { target, .. } = &instr.node {
            let (coherence, snapshot, action) = self.process_witness(*target)?;

            if let Some(reg) = instr.result {
                self.registers.insert(reg, PhiIRValue::Number(coherence));
            }

            if action == WitnessAction::Yield {
                let frozen = self.freeze_state();
                return Ok(Some(VmExecResult::Yielded {
                    snapshot,
                    frozen_state: frozen,
                }));
            }
            return Ok(None);
        }

        if let PhiIRNode::Entangle(freq) = &instr.node {
            self.host.on_entangle(*freq);
            let frozen = self.freeze_state();
            return Ok(Some(VmExecResult::Entangled {
                frequency: *freq,
                frozen_state: frozen,
            }));
        }

        self.execute_instruction(instr)?;
        Ok(None)
    }

    fn remap_block_ids(&self, node: &mut PhiIRNode, offset: u32) {
        match node {
            PhiIRNode::Jump(target) => {
                *target += offset;
            }
            PhiIRNode::Branch {
                then_block,
                else_block,
                ..
            } => {
                *then_block += offset;
                *else_block += offset;
            }
            _ => {}
        }
    }

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

    fn compute_coherence(&self) -> f64 {
        let depth = self.intention_stack.len();
        let resonance_count = self.resonance_count();

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
        let internal = self.compute_coherence();
        self.host.get_coherence(internal)
    }

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
        let meta =
            self.functions.get(name).cloned().ok_or_else(|| {
                EvalError::InvalidOperation(format!("Undefined function: {}", name))
            })?;

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
