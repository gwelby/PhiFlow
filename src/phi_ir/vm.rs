//! PhiIR Bytecode VM
//!
//! Loads `.phivm` bytes emitted by `phi_ir::emitter` and executes them.

use crate::host::{DefaultHostProvider, PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::{BlockId, Operand, PhiIRBinOp, PhiIRValue, ResonateDirection, SensorKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const MAGIC: &[u8; 4] = b"PHIV";
const VERSION: u8 = 1;
const PHI: f64 = 1.618_033_988_749_895;

// --- Opcodes (must match emitter.rs) ---
const OP_NOP: u8 = 0x00;
const OP_CONST_NUM: u8 = 0x01;
const OP_CONST_STR: u8 = 0x02;
const OP_CONST_BOOL: u8 = 0x03;
const OP_CONST_VOID: u8 = 0x04;
const OP_LOAD_VAR: u8 = 0x10;
const OP_STORE_VAR: u8 = 0x11;
const OP_BINOP: u8 = 0x20;
const OP_UNOP: u8 = 0x21;
const OP_CALL: u8 = 0x22;
const OP_LIST_NEW: u8 = 0x23;
const OP_LIST_GET: u8 = 0x24;
const OP_FUNC_DEF: u8 = 0x25;
const OP_WITNESS: u8 = 0x30;
const OP_INTENTION_PUSH: u8 = 0x31;
const OP_INTENTION_POP: u8 = 0x32;
const OP_RESONATE: u8 = 0x33;
const OP_COHERENCE_CHECK: u8 = 0x34;
const OP_SLEEP: u8 = 0x35;
const OP_CREATE_PATTERN: u8 = 0x36;
const OP_WITNESS_SENSOR: u8 = 0x38;
const OP_FIELD: u8 = 0x39;
const OP_DISSONANCE: u8 = 0x3A;
const OP_COHERENCE_OF: u8 = 0x3B;
const OP_STREAM_PUSH: u8 = 0x3C;
const OP_STREAM_POP: u8 = 0x3D;
const OP_DOMAIN_CALL: u8 = 0x40;
const OP_REMEMBER: u8 = 0x50;
const OP_RECALL: u8 = 0x51;
const OP_BROADCAST: u8 = 0x52;
const OP_LISTEN: u8 = 0x53;
const OP_AGENT_DECL: u8 = 0x54;
const OP_VOID_DEPTH: u8 = 0x55;
const OP_EVOLVE: u8 = 0x60;
const OP_ENTANGLE: u8 = 0x61;
const OP_HANDOFF: u8 = 0x70;
const OP_RETURN: u8 = 0xE0;
const OP_JUMP: u8 = 0xE1;
const OP_BRANCH: u8 = 0xE2;
const OP_FALLTHROUGH: u8 = 0xE3;

#[derive(Debug)]
pub enum VmError {
    InvalidMagic,
    UnsupportedVersion(u8),
    InvalidOpcode(u8),
    InvalidBinOp(u8),
    InvalidResultFlag(u8),
    InvalidBoolFlag(u8),
    InvalidOptionalOperandFlag { opcode: u8, flag: u8 },
    InvalidStringIndex(u32),
    InvalidSensorId(i32),
    InvalidUtf8(std::str::Utf8Error),
    UnexpectedEof { needed: usize, remaining: usize },
    TrailingBytes(usize),
    BlockNotFound(BlockId),
    OperandNotFound(Operand),
    DivisionByZero,
    InvalidOperation(String),
    UnavailableSensor(SensorKind),
    StepLimitExceeded(usize),
    InvalidTerminator,
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::InvalidMagic => write!(f, "Invalid PHIV magic header"),
            VmError::UnsupportedVersion(v) => write!(f, "Unsupported PHIV version {}", v),
            VmError::InvalidOpcode(op) => write!(f, "Invalid opcode 0x{op:02X}"),
            VmError::InvalidBinOp(op) => write!(f, "Invalid binop byte 0x{op:02X}"),
            VmError::InvalidResultFlag(v) => write!(f, "Invalid result flag {}", v),
            VmError::InvalidBoolFlag(v) => write!(f, "Invalid bool flag {}", v),
            VmError::InvalidOptionalOperandFlag { opcode, flag } => write!(
                f,
                "Invalid optional operand flag {} for opcode 0x{opcode:02X}",
                flag
            ),
            VmError::InvalidStringIndex(i) => write!(f, "Invalid string table index {}", i),
            VmError::InvalidSensorId(i) => write!(f, "Invalid sensor ID {}", i),
            VmError::InvalidUtf8(e) => write!(f, "Invalid UTF-8 string payload: {}", e),
            VmError::UnexpectedEof { needed, remaining } => write!(
                f,
                "Unexpected EOF: needed {} bytes, remaining {}",
                needed, remaining
            ),
            VmError::TrailingBytes(n) => write!(f, "Trailing bytes after program decode: {}", n),
            VmError::BlockNotFound(id) => write!(f, "Block {} not found", id),
            VmError::OperandNotFound(op) => write!(f, "Operand {} not found", op),
            VmError::DivisionByZero => write!(f, "Division by zero"),
            VmError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            VmError::UnavailableSensor(sensor) => {
                write!(
                    f,
                    "Sensor `{}` is unavailable on this host",
                    sensor.as_name()
                )
            }
            VmError::StepLimitExceeded(max) => {
                write!(f, "Execution step limit exceeded ({} steps)", max)
            }
            VmError::InvalidTerminator => write!(f, "Invalid terminator opcode"),
        }
    }
}

type VmResult<T> = Result<T, VmError>;

// ---------------------------------------------------------------------------
// Frozen state — serializable snapshot for yield/resume
// ---------------------------------------------------------------------------

/// All mutable VM state, captured atomically when the program yields at `witness`.
/// Can be stored to disk, sent across a channel, or inspected by the host daemon.
#[derive(Debug, Clone)]
pub struct FrozenVmState {
    pub program: BytecodeProgram,
    pub registers: HashMap<Operand, PhiIRValue>,
    pub variables: HashMap<String, PhiIRValue>,
    pub intention_stack: Vec<String>,
    pub active_streams: Vec<String>,
    pub resonance_field: HashMap<String, Vec<PhiIRValue>>,
    pub current_block: BlockId,
    pub instruction_ptr: usize,
    pub coherence_history: Vec<(f64, f64)>,
    pub witness_log: Vec<WitnessSnapshot>,
    pub step_count: usize,
    pub yield_timestamp: f64,
}

// ---------------------------------------------------------------------------
// Execution result — supports yield/resume for host-controlled witness
// ---------------------------------------------------------------------------

/// Result of running the PhiVM to completion or a host-requested yield.
#[derive(Debug, Clone)]
pub enum VmExecResult {
    /// Program completed normally with a final value.
    Complete(PhiIRValue),
    /// Program yielded at a `witness` statement. The host can inspect
    /// the snapshot and call `resume()` to continue with the frozen state.
    Yielded {
        snapshot: WitnessSnapshot,
        frozen_state: FrozenVmState,
    },
}

#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    pub version: u8,
    pub string_table: Vec<String>,
    pub blocks: Vec<BytecodeBlock>,
}

#[derive(Debug, Clone)]
pub struct BytecodeBlock {
    pub id: BlockId,
    pub instructions: Vec<BytecodeInstruction>,
    pub terminator: BytecodeNode,
}

#[derive(Debug, Clone)]
pub struct BytecodeInstruction {
    pub result: Option<Operand>,
    pub node: BytecodeNode,
}

#[derive(Debug, Clone)]
pub enum BytecodeNode {
    Nop,
    Const(PhiIRValue),
    LoadVar(String),
    StoreVar {
        name: String,
        value: Operand,
    },
    BinOp {
        op: PhiIRBinOp,
        left: Operand,
        right: Operand,
    },
    UnaryOp {
        operand: Operand,
    },
    Call {
        name: String,
        args: Vec<Operand>,
    },
    ListNew(Vec<Operand>),
    ListGet {
        list: Operand,
        index: Operand,
    },
    FuncDef {
        name: String,
        body: BlockId,
    },
    Witness {
        target: Option<Operand>,
    },
    WitnessSensor {
        sensor: SensorKind,
    },
    IntentionPush {
        name: String,
    },
    IntentionPop,
    StreamPush {
        name: String,
        threshold: Option<f64>,
    },
    StreamPop,
    FieldCoherence,
    Dissonance,
    CoherenceOf(String),
    Resonate {
        value: Option<Operand>,
        direction: crate::phi_ir::ResonateDirection,
    },
    CoherenceCheck,
    Sleep {
        duration: Operand,
    },
    CreatePattern {
        frequency: Operand,
        params: Vec<(String, Operand)>,
    },
    DomainCall {
        args: Vec<Operand>,
        string_args: Vec<String>,
    },
    Remember {
        key: String,
        value: Operand,
    },
    Recall(String),
    Broadcast {
        channel: String,
        value: Operand,
    },
    Listen(String),
    AgentDecl {
        name: String,
        version: String,
    },
    VoidDepth,
    Evolve(Operand),
    Entangle(f64),
    Handoff {
        target_agent: String,
        task_id: String,
        context_op: Operand,
    },
    Return(Operand),
    Jump(BlockId),
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    Fallthrough,
}

/// PhiIR bytecode runtime.
pub struct PhiVm {
    program: BytecodeProgram,
    block_index: HashMap<BlockId, usize>,
    registers: HashMap<Operand, PhiIRValue>,
    variables: HashMap<String, PhiIRValue>,
    value_stack: Vec<PhiIRValue>,
    intention_stack: Vec<String>,
    active_streams: Vec<String>,
    resonance_field: HashMap<String, Vec<PhiIRValue>>,
    shared_resonance: Option<Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>>,
    current_block: BlockId,
    instruction_ptr: usize,
    host: Arc<dyn PhiHostProvider>,
    sensor_provider: Option<Arc<dyn Fn(SensorKind) -> Option<f64> + Send + Sync>>,
    /// History of coherence values for dissonance calculation.
    /// Each entry is (timestamp_seconds, coherence_value).
    coherence_history: Vec<(f64, f64)>,
    /// Ordered witness events for post-execution inspection.
    pub witness_log: Vec<WitnessSnapshot>,
    pub max_steps: Option<usize>,
    pub step_count: usize,
}

impl PhiVm {
    /// Load VM from `.phivm` bytes.
    pub fn from_bytes(bytes: &[u8]) -> VmResult<Self> {
        let program = parse_program(bytes)?;
        let mut block_index = HashMap::new();
        for (idx, block) in program.blocks.iter().enumerate() {
            block_index.insert(block.id, idx);
        }

        let current_block = program.blocks.first().map(|b| b.id).unwrap_or(0);

        Ok(Self {
            program,
            block_index,
            registers: HashMap::new(),
            variables: HashMap::new(),
            value_stack: Vec::new(),
            intention_stack: Vec::new(),
            active_streams: Vec::new(),
            resonance_field: HashMap::new(),
            shared_resonance: None,
            current_block,
            instruction_ptr: 0,
            host: Arc::new(DefaultHostProvider),
            sensor_provider: None,
            coherence_history: Vec::new(),
            witness_log: Vec::new(),
            max_steps: None,
            step_count: 0,
        })
    }

    /// Attach a custom host provider. Call before `run()`.
    pub fn with_host(mut self, host: Arc<dyn PhiHostProvider>) -> Self {
        self.host = host;
        self
    }

    /// Convenience entrypoint: parse bytes, run, and return final value.
    pub fn run_bytes(bytes: &[u8]) -> VmResult<PhiIRValue> {
        let mut vm = Self::from_bytes(bytes)?;
        vm.run()
    }

    pub fn run_bytes_with_sensor_provider<F>(bytes: &[u8], provider: F) -> VmResult<PhiIRValue>
    where
        F: Fn(SensorKind) -> Option<f64> + Send + Sync + 'static,
    {
        let mut vm = Self::from_bytes(bytes)?.with_sensor_provider(provider);
        vm.run()
    }

    pub fn with_sensor_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn(SensorKind) -> Option<f64> + Send + Sync + 'static,
    {
        self.sensor_provider = Some(Arc::new(provider));
        self
    }

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

    /// Return the loaded string table.
    pub fn string_table(&self) -> &[String] {
        &self.program.string_table
    }

    /// Return the decoded bytecode program.
    pub fn program(&self) -> &BytecodeProgram {
        &self.program
    }

    /// Return the current value stack.
    pub fn value_stack(&self) -> &[PhiIRValue] {
        &self.value_stack
    }

    /// Execute to completion and return the top-of-stack value.
    /// This is the backward-compatible shim used by all existing tests and callers.
    /// For daemon loops that need yield/resume, use `run_or_yield()` instead.
    pub fn run(&mut self) -> VmResult<PhiIRValue> {
        match self.run_or_yield()? {
            VmExecResult::Complete(val) => Ok(val),
            VmExecResult::Yielded { .. } => {
                Ok(self.value_stack.last().cloned().unwrap_or(PhiIRValue::Void))
            }
        }
    }

    /// Run the program, but may return `Yielded` if a `witness` triggers
    /// a host-requested yield. The caller can inspect the frozen state
    /// and call `resume()` to continue execution.
    pub fn run_or_yield(&mut self) -> VmResult<VmExecResult> {
        if self.program.blocks.is_empty() {
            return Ok(VmExecResult::Complete(PhiIRValue::Void));
        }

        loop {
            if let Some(max) = self.max_steps {
                if self.step_count > max {
                    return Err(VmError::StepLimitExceeded(max));
                }
            }
            self.step_count += 1;

            let block = self.get_block(self.current_block)?;
            let instr_count = block.instructions.len();

            if self.instruction_ptr < instr_count {
                let instr = block.instructions[self.instruction_ptr].clone();
                self.instruction_ptr += 1;
                // execute_instruction returns Some(snapshot) when a Witness yield is requested.
                if let Some(snapshot) = self.execute_instruction_yielding(&instr)? {
                    let frozen = self.freeze_state();
                    return Ok(VmExecResult::Yielded { snapshot, frozen_state: frozen });
                }
            } else {
                let terminator = block.terminator.clone();
                if let Some(val) = self.execute_terminator(&terminator)? {
                    return Ok(VmExecResult::Complete(
                        self.value_stack.last().cloned().unwrap_or(val),
                    ));
                }
            }
        }
    }

    /// Resume execution after a yield. Restores frozen state and continues
    /// via `run_or_yield()`, so the caller can handle further yields.
    pub fn resume(&mut self, state: FrozenVmState) -> VmResult<VmExecResult> {
        self.program = state.program;
        self.registers = state.registers;
        self.variables = state.variables;
        self.intention_stack = state.intention_stack;
        self.active_streams = state.active_streams;
        self.resonance_field = state.resonance_field;
        self.current_block = state.current_block;
        self.instruction_ptr = state.instruction_ptr;
        self.coherence_history = state.coherence_history;
        self.witness_log = state.witness_log;
        self.step_count = state.step_count;
        // Re-index the block map since program may have evolved.
        self.block_index.clear();
        for (idx, block) in self.program.blocks.iter().enumerate() {
            self.block_index.insert(block.id, idx);
        }
        self.run_or_yield()
    }

    /// Capture the current VM state as a frozen snapshot.
    pub fn freeze_state(&self) -> FrozenVmState {
        let yield_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        FrozenVmState {
            program: self.program.clone(),
            registers: self.registers.clone(),
            variables: self.variables.clone(),
            intention_stack: self.intention_stack.clone(),
            active_streams: self.active_streams.clone(),
            resonance_field: self.resonance_field.clone(),
            current_block: self.current_block,
            instruction_ptr: self.instruction_ptr,
            coherence_history: self.coherence_history.clone(),
            witness_log: self.witness_log.clone(),
            step_count: self.step_count,
            yield_timestamp,
        }
    }

    fn get_block(&self, id: BlockId) -> VmResult<&BytecodeBlock> {
        let idx = *self
            .block_index
            .get(&id)
            .ok_or(VmError::BlockNotFound(id))?;
        Ok(&self.program.blocks[idx])
    }

    fn get_reg(&self, op: Operand) -> VmResult<&PhiIRValue> {
        self.registers.get(&op).ok_or(VmError::OperandNotFound(op))
    }

    fn execute_instruction(&mut self, instr: &BytecodeInstruction) -> VmResult<()> {
        self.execute_instruction_yielding(instr).map(|_| ())
    }

    /// Like `execute_instruction` but returns `Some(snapshot)` when the
    /// host requests a yield at a `Witness` opcode. All other opcodes return `None`.
    fn execute_instruction_yielding(
        &mut self,
        instr: &BytecodeInstruction,
    ) -> VmResult<Option<WitnessSnapshot>> {
        let value: Option<PhiIRValue> = match &instr.node {
            BytecodeNode::Nop => None,
            BytecodeNode::Const(v) => Some(v.clone()),
            BytecodeNode::LoadVar(name) => Some(
                self.variables
                    .get(name)
                    .cloned()
                    .unwrap_or(PhiIRValue::Void),
            ),
            BytecodeNode::StoreVar { name, value } => {
                let val = self.get_reg(*value)?.clone();
                self.variables.insert(name.clone(), val);
                None
            }
            BytecodeNode::BinOp { op, left, right } => Some(self.eval_binop(op, *left, *right)?),
            BytecodeNode::UnaryOp { operand } => Some(self.eval_unop(*operand)?),
            BytecodeNode::Call { args, .. } => {
                for op in args {
                    let _ = self.get_reg(*op)?;
                }
                Some(PhiIRValue::Void)
            }
            BytecodeNode::ListNew(ops) => {
                for op in ops {
                    let _ = self.get_reg(*op)?;
                }
                Some(PhiIRValue::Void)
            }
            BytecodeNode::ListGet { list, index } => {
                let _ = self.get_reg(*list)?;
                let _ = self.get_reg(*index)?;
                Some(PhiIRValue::Void)
            }
            BytecodeNode::FuncDef { .. } => None,
            BytecodeNode::Witness { target } => {
                let observed = if let Some(op) = target {
                    let val = self.get_reg(*op)?;
                    Some(format!("{:?}", val))
                } else {
                    None
                };
                let coherence = self.compute_coherence();
                self.record_coherence_history(coherence);
                let snapshot = WitnessSnapshot {
                    intention_stack: self.intention_stack.clone(),
                    coherence,
                    register_count: self.registers.len(),
                    resonance_count: self.resonance_field.values().map(|v| v.len()).sum(),
                    observed_value: observed,
                    agent_name: None,
                };
                let action = self.host.on_witness(&snapshot);
                self.witness_log.push(snapshot.clone());
                // Propagate the yield action — the caller (run_or_yield) handles the
                // early return; we signal it by returning Some(snapshot).
                if action == WitnessAction::Yield {
                    // Store coherence on the value stack so run() shim still works.
                    if let Some(reg) = instr.result {
                        self.registers.insert(reg, PhiIRValue::Number(coherence));
                    }
                    self.value_stack.push(PhiIRValue::Number(coherence));
                    return Ok(Some(snapshot));
                }
                Some(PhiIRValue::Number(coherence))
            }
            BytecodeNode::WitnessSensor { sensor } => {
                Some(PhiIRValue::Number(self.resolve_sensor(*sensor)?))
            }
            BytecodeNode::IntentionPush { name } => {
                self.intention_stack.push(name.clone());
                self.resonance_field.entry(name.clone()).or_default();
                None
            }
            BytecodeNode::IntentionPop => {
                self.intention_stack.pop();
                None
            }
            BytecodeNode::StreamPush { name, threshold: _ } => {
                self.intention_stack.push(name.clone());
                self.active_streams.push(name.clone());
                self.resonance_field.insert(name.clone(), Vec::new());
                None
            }
            BytecodeNode::StreamPop => {
                self.intention_stack.pop();
                self.active_streams.pop();
                None
            }
            BytecodeNode::FieldCoherence => {
                if let Some(shared) = &self.shared_resonance {
                    let guard = shared.lock().unwrap();
                    let mut sum = 0.0;
                    let mut count = 0;
                    for vals in guard.values() {
                        for val in vals {
                            if let PhiIRValue::Number(n) = val {
                                sum += n;
                                count += 1;
                            }
                        }
                    }
                    let score = if count > 0 { sum / count as f64 } else { 0.0 };
                    Some(PhiIRValue::Number(score))
                } else {
                    Some(PhiIRValue::Number(self.compute_coherence()))
                }
            }
            BytecodeNode::Dissonance => {
                // Compute rate of coherence change over recent witness cycles
                let history = &self.coherence_history;
                if history.len() < 2 {
                    Some(PhiIRValue::Number(0.0))
                } else {
                    let last = history[history.len() - 1].1;
                    let prev = history[history.len() - 2].1;
                    let delta = last - prev;
                    // Normalize to -1.0 .. 1.0. A 0.1 change is significant.
                    let normalized = (delta * 10.0).clamp(-1.0, 1.0);
                    Some(PhiIRValue::Number(normalized))
                }
            }
            BytecodeNode::CoherenceOf(name) => {
                if let Some(shared) = &self.shared_resonance {
                    let guard = shared.lock().unwrap();
                    if let Some(vals) = guard.get(name) {
                        if let Some(PhiIRValue::Number(n)) = vals.last() {
                            Some(PhiIRValue::Number(*n))
                        } else {
                            Some(PhiIRValue::Void)
                        }
                    } else {
                        Some(PhiIRValue::Void)
                    }
                } else {
                    Some(PhiIRValue::Void)
                }
            }
            BytecodeNode::Resonate {
                value,
                direction: _,
            } => {
                if let Some(op) = value {
                    let val = self.get_reg(*op)?.clone();
                    let key = self
                        .intention_stack
                        .last()
                        .cloned()
                        .unwrap_or_else(|| "global".to_string());
                    if self.active_streams.contains(&key) {
                        self.resonance_field.insert(key.clone(), vec![val.clone()]);
                        if let Some(shared) = &self.shared_resonance {
                            let mut guard = shared.lock().unwrap();
                            guard.insert(key, vec![val]);
                        }
                    } else {
                        self.resonance_field
                            .entry(key.clone())
                            .or_default()
                            .push(val.clone());
                        if let Some(shared) = &self.shared_resonance {
                            let mut guard = shared.lock().unwrap();
                            guard.entry(key).or_default().push(val);
                        }
                    }
                }
                None
            }
            BytecodeNode::CoherenceCheck => Some(PhiIRValue::Number(self.compute_coherence())),
            BytecodeNode::Sleep { duration } => {
                let _ = self.get_reg(*duration)?;
                None
            }
            BytecodeNode::CreatePattern { frequency, params } => {
                let _ = self.get_reg(*frequency)?;
                for (_, op) in params {
                    let _ = self.get_reg(*op)?;
                }
                None
            }
            BytecodeNode::DomainCall { args, .. } => {
                for op in args {
                    let _ = self.get_reg(*op)?;
                }
                None
            }
            BytecodeNode::Remember { key, value } => {
                let val = self.get_reg(*value)?;
                let val_str = match val {
                    PhiIRValue::Number(n) => n.to_string(),
                    PhiIRValue::String(s) => s.clone(),
                    PhiIRValue::Boolean(b) => b.to_string(),
                    PhiIRValue::Void => "void".to_string(),
                };
                self.host.persist(key, &val_str);
                None
            }
            BytecodeNode::Recall(key) => {
                if let Some(val_str) = self.host.recall(key) {
                    // Try to parse as number first, then bool, then return as Void
                    if let Ok(n) = val_str.parse::<f64>() {
                        Some(PhiIRValue::Number(n))
                    } else if val_str == "true" {
                        Some(PhiIRValue::Boolean(true))
                    } else if val_str == "false" {
                        Some(PhiIRValue::Boolean(false))
                    } else {
                        // Store returned string in the string table
                        let idx = self.program.string_table.len() as u32;
                        self.program.string_table.push(val_str);
                        Some(PhiIRValue::String(idx))
                    }
                } else {
                    Some(PhiIRValue::Void)
                }
            }
            BytecodeNode::Broadcast { channel, value } => {
                let val = self.get_reg(*value)?;
                // Broadcast to MQTT for consistency
                let json_val = match val {
                    PhiIRValue::Number(n) => serde_json::json!(n),
                    PhiIRValue::String(s) => serde_json::json!(s),
                    PhiIRValue::Boolean(b) => serde_json::json!(b),
                    PhiIRValue::Void => serde_json::Value::Null,
                };
                let _ = crate::resonance_bus::emit_resonance(json_val, channel, "phivm");
                None
            }
            BytecodeNode::Listen(channel) => {
                if let Some(val_str) = self.host.listen(channel) {
                    if let Ok(n) = val_str.parse::<f64>() {
                        Some(PhiIRValue::Number(n))
                    } else if val_str == "true" {
                        Some(PhiIRValue::Boolean(true))
                    } else if val_str == "false" {
                        Some(PhiIRValue::Boolean(false))
                    } else {
                        let idx = self.program.string_table.len() as u32;
                        self.program.string_table.push(val_str);
                        Some(PhiIRValue::String(idx))
                    }
                } else {
                    Some(PhiIRValue::Void)
                }
            }
            BytecodeNode::AgentDecl { .. } => None,
            BytecodeNode::VoidDepth => Some(PhiIRValue::Number(0.0)),
            BytecodeNode::Evolve(_op) => {
                // VM doesn't support runtime IR splicing yet
                Some(PhiIRValue::Void)
            }
            BytecodeNode::Entangle(_freq) => None,
            BytecodeNode::Handoff { target_agent, task_id, context_op } => {
                let context_val = self.get_reg(*context_op)?;
                let context_json = match context_val {
                    PhiIRValue::Number(n) => serde_json::json!(n),
                    PhiIRValue::String(s) => serde_json::json!(s),
                    PhiIRValue::Boolean(b) => serde_json::json!(b),
                    PhiIRValue::Void => serde_json::Value::Null,
                };
                
                let coherence = self.compute_coherence();
                let dissonance = if self.coherence_history.len() < 2 {
                    0.0
                } else {
                    let last = self.coherence_history[self.coherence_history.len() - 1].1;
                    let prev = self.coherence_history[self.coherence_history.len() - 2].1;
                    last - prev
                };

                let handoff_payload = serde_json::json!({
                    "target": target_agent,
                    "task_id": task_id,
                    "attention": self.intention_stack.last().cloned().unwrap_or_else(|| "global".to_string()),
                    "context": context_json,
                    "coherence": coherence,
                    "dissonance": dissonance,
                });

                let _ = crate::resonance_bus::emit_resonance(handoff_payload, "_handoff", "phivm");
                None
            }
            BytecodeNode::Return(_)
            | BytecodeNode::Jump(_)
            | BytecodeNode::Branch { .. }
            | BytecodeNode::Fallthrough => None,
        };

        if let Some(value) = value {
            if let Some(reg) = instr.result {
                self.registers.insert(reg, value.clone());
            }
            self.value_stack.push(value);
        }

        Ok(None)
    }

    fn execute_terminator(&mut self, node: &BytecodeNode) -> VmResult<Option<PhiIRValue>> {
        match node {
            BytecodeNode::Return(op) => {
                let val = self.get_reg(*op)?.clone();
                self.value_stack.push(val.clone());
                Ok(Some(val))
            }
            BytecodeNode::Jump(target) => {
                self.current_block = *target;
                self.instruction_ptr = 0;
                Ok(None)
            }
            BytecodeNode::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.get_reg(*condition)?;
                let target = match cond {
                    PhiIRValue::Boolean(true) => *then_block,
                    PhiIRValue::Boolean(false) => *else_block,
                    PhiIRValue::Number(n) if *n != 0.0 => *then_block,
                    _ => *else_block,
                };
                self.current_block = target;
                self.instruction_ptr = 0;
                Ok(None)
            }
            BytecodeNode::Fallthrough => {
                let current_idx = *self
                    .block_index
                    .get(&self.current_block)
                    .ok_or(VmError::BlockNotFound(self.current_block))?;

                if current_idx + 1 < self.program.blocks.len() {
                    self.current_block = self.program.blocks[current_idx + 1].id;
                    self.instruction_ptr = 0;
                    Ok(None)
                } else {
                    Ok(Some(
                        self.value_stack.last().cloned().unwrap_or(PhiIRValue::Void),
                    ))
                }
            }
            _ => Err(VmError::InvalidTerminator),
        }
    }

    fn compute_coherence(&self) -> f64 {
        crate::phi_ir::coherence::canonical_coherence(&self.intention_stack, &self.resonance_field)
    }

    fn record_coherence_history(&mut self, coherence: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.coherence_history.push((timestamp, coherence));
        if self.coherence_history.len() > 100 {
            self.coherence_history
                .drain(..self.coherence_history.len() - 100);
        }
    }

    fn resolve_sensor(&self, sensor: SensorKind) -> VmResult<f64> {
        let value = self
            .sensor_provider
            .as_ref()
            .and_then(|provider| provider(sensor))
            .or_else(|| crate::sensors::read_sensor(sensor));

        match value {
            Some(v) => Ok(v),
            None => match sensor {
                SensorKind::SomaSchumann
                | SensorKind::Soma432
                | SensorKind::SomaPresence
                | SensorKind::SomaFanHz
                | SensorKind::SomaAc60
                | SensorKind::SomaPeakDbc => Ok(0.0),
                _ => Err(VmError::UnavailableSensor(sensor)),
            },
        }
    }

    fn eval_unop(&self, operand: Operand) -> VmResult<PhiIRValue> {
        let value = self.get_reg(operand)?;
        match value {
            // Emitter v1 does not serialize the unary operator variant.
            // We preserve useful behavior by using type-directed semantics.
            PhiIRValue::Number(n) => Ok(PhiIRValue::Number(-n)),
            PhiIRValue::Boolean(b) => Ok(PhiIRValue::Boolean(!b)),
            _ => Err(VmError::InvalidOperation(
                "Unary op on unsupported type".to_string(),
            )),
        }
    }

    fn eval_binop(&self, op: &PhiIRBinOp, left: Operand, right: Operand) -> VmResult<PhiIRValue> {
        let l = self.get_reg(left)?;
        let r = self.get_reg(right)?;

        match (l, r) {
            (PhiIRValue::Number(lhs), PhiIRValue::Number(rhs)) => match op {
                PhiIRBinOp::Add => Ok(PhiIRValue::Number(lhs + rhs)),
                PhiIRBinOp::Sub => Ok(PhiIRValue::Number(lhs - rhs)),
                PhiIRBinOp::Mul => Ok(PhiIRValue::Number(lhs * rhs)),
                PhiIRBinOp::Div => {
                    if *rhs == 0.0 {
                        Err(VmError::DivisionByZero)
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
                PhiIRBinOp::And | PhiIRBinOp::Or => Err(VmError::InvalidOperation(
                    "Logical op on Number".to_string(),
                )),
            },
            (PhiIRValue::Boolean(lhs), PhiIRValue::Boolean(rhs)) => match op {
                PhiIRBinOp::And => Ok(PhiIRValue::Boolean(*lhs && *rhs)),
                PhiIRBinOp::Or => Ok(PhiIRValue::Boolean(*lhs || *rhs)),
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(lhs == rhs)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(lhs != rhs)),
                _ => Err(VmError::InvalidOperation(
                    "Unsupported boolean binary op".to_string(),
                )),
            },
            _ => Err(VmError::InvalidOperation(
                "Type mismatch in binary operation".to_string(),
            )),
        }
    }
}

fn parse_program(bytes: &[u8]) -> VmResult<BytecodeProgram> {
    let mut reader = ByteReader::new(bytes);
    let magic = reader.read_exact(4)?;
    if magic != MAGIC {
        return Err(VmError::InvalidMagic);
    }

    let version = reader.read_u8()?;
    if version != VERSION {
        return Err(VmError::UnsupportedVersion(version));
    }

    let string_count = reader.read_u32()?;
    let mut string_table = Vec::with_capacity(string_count as usize);
    for _ in 0..string_count {
        string_table.push(reader.read_string()?);
    }

    let block_count = reader.read_u32()?;

    let mut blocks = Vec::with_capacity(block_count as usize);
    for _ in 0..block_count {
        let id = reader.read_u32()?;
        let instr_count = reader.read_u32()?;

        let mut instructions = Vec::with_capacity(instr_count as usize);
        for _ in 0..instr_count {
            let has_result = reader.read_u8()?;
            let result = match has_result {
                0 => None,
                1 => Some(reader.read_u32()?),
                v => return Err(VmError::InvalidResultFlag(v)),
            };
            let node = parse_node(&mut reader, &string_table)?;
            instructions.push(BytecodeInstruction { result, node });
        }

        let terminator = parse_node(&mut reader, &string_table)?;
        blocks.push(BytecodeBlock {
            id,
            instructions,
            terminator,
        });
    }

    let trailing = reader.remaining();
    if trailing > 0 {
        return Err(VmError::TrailingBytes(trailing));
    }

    Ok(BytecodeProgram {
        version,
        string_table,
        blocks,
    })
}

fn parse_node(reader: &mut ByteReader<'_>, string_table: &[String]) -> VmResult<BytecodeNode> {
    let opcode = reader.read_u8()?;
    let node = match opcode {
        OP_NOP => BytecodeNode::Nop,
        OP_CONST_NUM => BytecodeNode::Const(PhiIRValue::Number(reader.read_f64()?)),
        OP_CONST_STR => {
            let index = reader.read_u32()?;
            if index as usize >= string_table.len() {
                return Err(VmError::InvalidStringIndex(index));
            }
            BytecodeNode::Const(PhiIRValue::String(index))
        }
        OP_CONST_BOOL => {
            let flag = reader.read_u8()?;
            let value = match flag {
                0 => false,
                1 => true,
                other => return Err(VmError::InvalidBoolFlag(other)),
            };
            BytecodeNode::Const(PhiIRValue::Boolean(value))
        }
        OP_CONST_VOID => BytecodeNode::Const(PhiIRValue::Void),
        OP_LOAD_VAR => BytecodeNode::LoadVar(read_string_ref(reader, string_table)?),
        OP_STORE_VAR => BytecodeNode::StoreVar {
            name: read_string_ref(reader, string_table)?,
            value: reader.read_u32()?,
        },
        OP_BINOP => BytecodeNode::BinOp {
            op: parse_binop(reader.read_u8()?)?,
            left: reader.read_u32()?,
            right: reader.read_u32()?,
        },
        OP_UNOP => BytecodeNode::UnaryOp {
            operand: reader.read_u32()?,
        },
        OP_CALL => {
            let name = read_string_ref(reader, string_table)?;
            let argc = reader.read_u32()?;
            let mut args = Vec::with_capacity(argc as usize);
            for _ in 0..argc {
                args.push(reader.read_u32()?);
            }
            BytecodeNode::Call { name, args }
        }
        OP_LIST_NEW => {
            let count = reader.read_u32()?;
            let mut ops = Vec::with_capacity(count as usize);
            for _ in 0..count {
                ops.push(reader.read_u32()?);
            }
            BytecodeNode::ListNew(ops)
        }
        OP_LIST_GET => BytecodeNode::ListGet {
            list: reader.read_u32()?,
            index: reader.read_u32()?,
        },
        OP_FUNC_DEF => BytecodeNode::FuncDef {
            name: read_string_ref(reader, string_table)?,
            body: reader.read_u32()?,
        },
        OP_WITNESS => BytecodeNode::Witness {
            target: read_optional_operand(reader, OP_WITNESS)?,
        },
        OP_WITNESS_SENSOR => {
            let sensor_id = reader.read_u8()? as i32;
            let sensor =
                SensorKind::from_id(sensor_id).ok_or(VmError::InvalidSensorId(sensor_id))?;
            BytecodeNode::WitnessSensor { sensor }
        }
        OP_FIELD => BytecodeNode::FieldCoherence,
        OP_DISSONANCE => BytecodeNode::Dissonance,
        OP_COHERENCE_OF => BytecodeNode::CoherenceOf(read_string_ref(reader, string_table)?),
        OP_STREAM_PUSH => {
            let name = read_string_ref(reader, string_table)?;
            let has_threshold = reader.read_u8()?;
            let threshold = if has_threshold == 1 {
                Some(reader.read_f64()?)
            } else {
                None
            };
            BytecodeNode::StreamPush { name, threshold }
        }
        OP_STREAM_POP => BytecodeNode::StreamPop,
        OP_INTENTION_PUSH => BytecodeNode::IntentionPush {
            name: read_string_ref(reader, string_table)?,
        },
        OP_INTENTION_POP => BytecodeNode::IntentionPop,
        OP_RESONATE => {
            let direction_byte = reader.read_u8()?;
            let direction = if direction_byte == 0 {
                crate::phi_ir::ResonateDirection::TeamA
            } else {
                crate::phi_ir::ResonateDirection::TeamB
            };
            BytecodeNode::Resonate {
                value: read_optional_operand(reader, OP_RESONATE)?,
                direction,
            }
        }
        OP_COHERENCE_CHECK => BytecodeNode::CoherenceCheck,
        OP_SLEEP => BytecodeNode::Sleep {
            duration: reader.read_u32()?,
        },
        OP_CREATE_PATTERN => {
            let frequency = reader.read_u32()?;
            let param_count = reader.read_u32()?;
            let mut params = Vec::with_capacity(param_count as usize);
            for _ in 0..param_count {
                let key = read_string_ref(reader, string_table)?;
                let val = reader.read_u32()?;
                params.push((key, val));
            }
            BytecodeNode::CreatePattern { frequency, params }
        }
        OP_DOMAIN_CALL => {
            let argc = reader.read_u32()?;
            let mut args = Vec::with_capacity(argc as usize);
            for _ in 0..argc {
                args.push(reader.read_u32()?);
            }

            let strc = reader.read_u32()?;
            let mut string_args = Vec::with_capacity(strc as usize);
            for _ in 0..strc {
                string_args.push(read_string_ref(reader, string_table)?);
            }

            BytecodeNode::DomainCall { args, string_args }
        }
        OP_RETURN => BytecodeNode::Return(reader.read_u32()?),
        OP_JUMP => BytecodeNode::Jump(reader.read_u32()?),
        OP_BRANCH => BytecodeNode::Branch {
            condition: reader.read_u32()?,
            then_block: reader.read_u32()?,
            else_block: reader.read_u32()?,
        },
        OP_FALLTHROUGH => BytecodeNode::Fallthrough,
        OP_REMEMBER => BytecodeNode::Remember {
            key: read_string_ref(reader, string_table)?,
            value: reader.read_u32()?,
        },
        OP_RECALL => BytecodeNode::Recall(read_string_ref(reader, string_table)?),
        OP_BROADCAST => BytecodeNode::Broadcast {
            channel: read_string_ref(reader, string_table)?,
            value: reader.read_u32()?,
        },
        OP_LISTEN => BytecodeNode::Listen(read_string_ref(reader, string_table)?),
        OP_AGENT_DECL => BytecodeNode::AgentDecl {
            name: read_string_ref(reader, string_table)?,
            version: read_string_ref(reader, string_table)?,
        },
        OP_VOID_DEPTH => BytecodeNode::VoidDepth,
        OP_EVOLVE => BytecodeNode::Evolve(reader.read_u32()?),
        OP_ENTANGLE => BytecodeNode::Entangle(reader.read_f64()?),
        OP_HANDOFF => BytecodeNode::Handoff {
            target_agent: read_string_ref(reader, string_table)?,
            task_id: read_string_ref(reader, string_table)?,
            context_op: reader.read_u32()?,
        },
        _ => return Err(VmError::InvalidOpcode(opcode)),
    };

    Ok(node)
}

fn read_string_ref(reader: &mut ByteReader<'_>, string_table: &[String]) -> VmResult<String> {
    let index = reader.read_u32()?;
    string_table
        .get(index as usize)
        .cloned()
        .ok_or(VmError::InvalidStringIndex(index))
}

fn read_optional_operand(reader: &mut ByteReader<'_>, opcode: u8) -> VmResult<Option<Operand>> {
    let flag = reader.read_u8()?;
    match flag {
        0 => Ok(None),
        1 => Ok(Some(reader.read_u32()?)),
        other => Err(VmError::InvalidOptionalOperandFlag {
            opcode,
            flag: other,
        }),
    }
}

fn parse_binop(byte: u8) -> VmResult<PhiIRBinOp> {
    let op = match byte {
        0x00 => PhiIRBinOp::Add,
        0x01 => PhiIRBinOp::Sub,
        0x02 => PhiIRBinOp::Mul,
        0x03 => PhiIRBinOp::Div,
        0x04 => PhiIRBinOp::Mod,
        0x05 => PhiIRBinOp::Pow,
        0x06 => PhiIRBinOp::Eq,
        0x07 => PhiIRBinOp::Neq,
        0x08 => PhiIRBinOp::Lt,
        0x09 => PhiIRBinOp::Lte,
        0x0A => PhiIRBinOp::Gt,
        0x0B => PhiIRBinOp::Gte,
        0x0C => PhiIRBinOp::And,
        0x0D => PhiIRBinOp::Or,
        other => return Err(VmError::InvalidBinOp(other)),
    };
    Ok(op)
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }

    fn read_exact(&mut self, len: usize) -> VmResult<&'a [u8]> {
        if self.remaining() < len {
            return Err(VmError::UnexpectedEof {
                needed: len,
                remaining: self.remaining(),
            });
        }
        let start = self.cursor;
        let end = start + len;
        self.cursor = end;
        Ok(&self.bytes[start..end])
    }

    fn read_u8(&mut self) -> VmResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> VmResult<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> VmResult<f64> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> VmResult<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_exact(len)?;
        let value = std::str::from_utf8(bytes).map_err(VmError::InvalidUtf8)?;
        Ok(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{BytecodeNode, PhiVm, VmError};
    use super::{
        OP_COHERENCE_CHECK, OP_CONST_NUM, OP_FALLTHROUGH, OP_INTENTION_POP, OP_INTENTION_PUSH,
        OP_RESONATE, OP_RETURN, OP_WITNESS,
    };
    use crate::phi_ir::{emitter, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, PhiInstruction};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    fn emit_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_f64(out: &mut Vec<u8>, value: f64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn emit_string(out: &mut Vec<u8>, value: &str) {
        emit_u32(out, value.len() as u32);
        out.extend_from_slice(value.as_bytes());
    }

    fn build_native_consciousness_opcode_program() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PHIV");
        out.push(1); // version

        emit_u32(&mut out, 1); // string table count
        emit_string(&mut out, "healing");

        emit_u32(&mut out, 1); // block count
        emit_u32(&mut out, 0); // block id
        emit_u32(&mut out, 6); // instruction count

        // r0 = const 432.0
        out.push(1);
        emit_u32(&mut out, 0);
        out.push(OP_CONST_NUM);
        emit_f64(&mut out, 432.0);

        // intention_push "healing"
        out.push(0);
        out.push(OP_INTENTION_PUSH);
        emit_u32(&mut out, 0);

        // r1 = witness r0
        out.push(1);
        emit_u32(&mut out, 1);
        out.push(OP_WITNESS);
        out.push(1);
        emit_u32(&mut out, 0);

        // resonate r0
        out.push(0);
        out.push(OP_RESONATE);
        out.push(0); // TeamA
        out.push(1);
        emit_u32(&mut out, 0);

        // r2 = coherence_check
        out.push(1);
        emit_u32(&mut out, 2);
        out.push(OP_COHERENCE_CHECK);

        // intention_pop
        out.push(0);
        out.push(OP_INTENTION_POP);

        // return r2
        out.push(OP_RETURN);
        emit_u32(&mut out, 2);

        out
    }

    fn build_invalid_witness_flag_program(flag: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PHIV");
        out.push(1); // version

        emit_u32(&mut out, 0); // string table count
        emit_u32(&mut out, 1); // block count
        emit_u32(&mut out, 0); // block id
        emit_u32(&mut out, 1); // instruction count

        // witness with invalid optional-operand flag
        out.push(0);
        out.push(OP_WITNESS);
        out.push(flag);

        // fallthrough terminator
        out.push(OP_FALLTHROUGH);

        out
    }

    fn single_block_program(
        instructions: Vec<PhiInstruction>,
        terminator: PhiIRNode,
    ) -> PhiIRProgram {
        PhiIRProgram {
            blocks: vec![PhiIRBlock {
                id: 0,
                label: "entry".to_string(),
                instructions,
                terminator,
            }],
            entry: 0,
            string_table: Vec::new(),
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        }
    }

    #[test]
    fn vm_executes_basic_arithmetic() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(10.0)),
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::Const(PhiIRValue::Number(32.0)),
                },
                PhiInstruction {
                    result: Some(2),
                    node: PhiIRNode::BinOp {
                        op: crate::phi_ir::PhiIRBinOp::Add,
                        left: 0,
                        right: 1,
                    },
                },
            ],
            PhiIRNode::Return(2),
        );

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute bytecode");
        assert_eq!(result, PhiIRValue::Number(42.0));
    }

    #[test]
    fn vm_executes_branch_terminator() {
        let program = PhiIRProgram {
            blocks: vec![
                PhiIRBlock {
                    id: 0,
                    label: "entry".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(0),
                        node: PhiIRNode::Const(PhiIRValue::Boolean(true)),
                    }],
                    terminator: PhiIRNode::Branch {
                        condition: 0,
                        then_block: 1,
                        else_block: 2,
                    },
                },
                PhiIRBlock {
                    id: 1,
                    label: "then".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(1),
                        node: PhiIRNode::Const(PhiIRValue::Number(7.0)),
                    }],
                    terminator: PhiIRNode::Return(1),
                },
                PhiIRBlock {
                    id: 2,
                    label: "else".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(2),
                        node: PhiIRNode::Const(PhiIRValue::Number(9.0)),
                    }],
                    terminator: PhiIRNode::Return(2),
                },
            ],
            entry: 0,
            string_table: Vec::new(),
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        };

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute branch bytecode");
        assert_eq!(result, PhiIRValue::Number(7.0));
    }

    #[test]
    fn vm_coherence_tracks_intention_and_resonance() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(432.0)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::IntentionPush {
                        name: "healing".to_string(),
                        frequency_hint: None,
                    },
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(0),
                        frequency_relationship: None,
                        direction: crate::phi_ir::ResonateDirection::TeamA,
                    },
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::CoherenceCheck,
                },
            ],
            PhiIRNode::Return(1),
        );

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute coherence bytecode");
        match result {
            PhiIRValue::Number(n) => {
                // Canonical: base(depth=1) * phase(k=1) = 0.382 * 1.0
                let expected = 1.0 - super::PHI.powi(-1);
                assert!(
                    (n - expected).abs() < 1e-9,
                    "expected coherence near {}, got {}",
                    expected,
                    n
                );
            }
            other => panic!("expected Number coherence result, got {:?}", other),
        }
    }

    #[test]
    fn vm_round_trips_string_values_through_string_table() {
        let program = PhiIRProgram {
            blocks: vec![PhiIRBlock {
                id: 0,
                label: "entry".to_string(),
                instructions: vec![
                    PhiInstruction {
                        result: Some(0),
                        node: PhiIRNode::Const(PhiIRValue::String(1)),
                    },
                    PhiInstruction {
                        result: None,
                        node: PhiIRNode::StoreVar {
                            name: "message".to_string(),
                            value: 0,
                        },
                    },
                    PhiInstruction {
                        result: Some(1),
                        node: PhiIRNode::LoadVar("message".to_string()),
                    },
                ],
                terminator: PhiIRNode::Return(1),
            }],
            entry: 0,
            string_table: vec!["hello".to_string(), "hello".to_string()],
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        };

        let bytes = emitter::emit(&program);
        let mut vm = PhiVm::from_bytes(&bytes).expect("VM should load bytecode");

        assert_eq!(
            vm.string_table()
                .iter()
                .filter(|value| value.as_str() == "hello")
                .count(),
            1,
            "emitted string table should deduplicate values"
        );

        let result = vm.run().expect("VM should execute bytecode");
        match result {
            PhiIRValue::String(index) => {
                let value = vm
                    .string_table()
                    .get(index as usize)
                    .expect("string index should resolve in VM table");
                assert_eq!(value, "hello");
            }
            other => panic!("expected string result, got {:?}", other),
        }
    }

    #[test]
    fn vm_executes_native_consciousness_opcodes_from_raw_bytecode() {
        let bytes = build_native_consciousness_opcode_program();

        let mut vm = PhiVm::from_bytes(&bytes).expect("VM should decode manual bytecode");
        let result = vm.run().expect("VM should execute manual bytecode");

        // Canonical: base(depth=1) * phase(k=1) = 0.382 * 1.0
        let expected = 1.0 - super::PHI.powi(-1);
        let coherence = match result {
            PhiIRValue::Number(value) => value,
            other => panic!("expected Number result, got {:?}", other),
        };

        assert!(
            (coherence - expected).abs() < 1e-12,
            "coherence mismatch: expected {expected}, got {coherence}"
        );

        assert!(
            vm.intention_stack.is_empty(),
            "intention stack should be empty after IntentionPop"
        );
        assert_eq!(
            vm.resonance_field.get("healing").map(|values| values.len()),
            Some(1),
            "resonate should persist exactly one value in the healing channel"
        );

        let witness_value = vm
            .registers
            .get(&1)
            .expect("witness result register should be populated");
        match witness_value {
            PhiIRValue::Number(value) => {
                let expected_witness = 1.0 - super::PHI.powi(-1);
                assert!(
                    (*value - expected_witness).abs() < 1e-12,
                    "witness coherence mismatch: expected {expected_witness}, got {value}"
                );
            }
            other => panic!("expected witness register to hold Number, got {:?}", other),
        }

        let second_run = PhiVm::run_bytes(&bytes).expect("same bytecode should be deterministic");
        assert_eq!(
            result, second_run,
            "native opcode execution must be deterministic"
        );
    }

    #[test]
    fn vm_decodes_resonate_direction_from_emitted_bytecode() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(0.72)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(0),
                        frequency_relationship: None,
                        direction: crate::phi_ir::ResonateDirection::TeamB,
                    },
                },
            ],
            PhiIRNode::Fallthrough,
        );

        let bytes = emitter::emit(&program);
        let vm = PhiVm::from_bytes(&bytes).expect("VM should decode emitted bytecode");
        let block = vm
            .program()
            .blocks
            .first()
            .expect("decoded program should contain an entry block");
        let resonate = &block.instructions[1].node;

        match resonate {
            BytecodeNode::Resonate { value, direction } => {
                assert_eq!(*value, Some(0));
                assert_eq!(*direction, crate::phi_ir::ResonateDirection::TeamB);
            }
            other => panic!("expected decoded resonate node, got {:?}", other),
        }
    }

    #[test]
    fn vm_rejects_invalid_optional_operand_flag() {
        let bytes = build_invalid_witness_flag_program(2);
        let err = PhiVm::from_bytes(&bytes)
            .err()
            .expect("invalid witness flag should fail decoding");

        match err {
            VmError::InvalidOptionalOperandFlag { opcode, flag } => {
                assert_eq!(opcode, OP_WITNESS);
                assert_eq!(flag, 2);
            }
            other => panic!("expected InvalidOptionalOperandFlag, got {:?}", other),
        }
    }

    #[test]
    fn vm_stream_resonance_overwrites_active_stream_scope() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::StreamPush("pulse".to_string(), None),
                },
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(1.0)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(0),
                        frequency_relationship: None,
                        direction: crate::phi_ir::ResonateDirection::TeamA,
                    },
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::Const(PhiIRValue::Number(2.0)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(1),
                        frequency_relationship: None,
                        direction: crate::phi_ir::ResonateDirection::TeamA,
                    },
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::StreamPop,
                },
            ],
            PhiIRNode::Return(1),
        );

        let bytes = emitter::emit(&program);
        let mut vm = PhiVm::from_bytes(&bytes).expect("VM should decode emitted bytecode");
        let result = vm.run().expect("VM should execute stream bytecode");

        assert_eq!(result, PhiIRValue::Number(2.0));
        assert_eq!(
            vm.resonance_field.get("pulse"),
            Some(&vec![PhiIRValue::Number(2.0)]),
            "active streams should overwrite the previous resonated value",
        );
    }

    #[test]
    fn vm_field_coherence_reads_shared_resonance_average() {
        let program = single_block_program(
            vec![PhiInstruction {
                result: Some(0),
                node: PhiIRNode::FieldCoherence,
            }],
            PhiIRNode::Return(0),
        );

        let mut shared = HashMap::new();
        shared.insert(
            "alpha".to_string(),
            vec![PhiIRValue::Number(0.25), PhiIRValue::Number(0.75)],
        );
        shared.insert("beta".to_string(), vec![PhiIRValue::Number(0.5)]);

        let bytes = emitter::emit(&program);
        let mut vm = PhiVm::from_bytes(&bytes)
            .expect("VM should decode emitted bytecode")
            .with_shared_resonance(Arc::new(Mutex::new(shared)));
        let result = vm.run().expect("VM should execute field coherence opcode");

        assert_eq!(result, PhiIRValue::Number(0.5));
    }

    #[test]
    fn vm_coherence_of_reads_named_shared_stream() {
        let program = single_block_program(
            vec![PhiInstruction {
                result: Some(0),
                node: PhiIRNode::CoherenceOf("beta".to_string()),
            }],
            PhiIRNode::Return(0),
        );

        let mut shared = HashMap::new();
        shared.insert(
            "beta".to_string(),
            vec![PhiIRValue::Number(0.25), PhiIRValue::Number(0.75)],
        );

        let bytes = emitter::emit(&program);
        let mut vm = PhiVm::from_bytes(&bytes)
            .expect("VM should decode emitted bytecode")
            .with_shared_resonance(Arc::new(Mutex::new(shared)));
        let result = vm.run().expect("VM should execute coherence_of opcode");

        assert_eq!(result, PhiIRValue::Number(0.75));
    }

    #[test]
    fn vm_dissonance_uses_recent_witness_history() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Witness {
                        target: None,
                        collapse_policy: crate::phi_ir::CollapsePolicy::Final,
                    },
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::IntentionPush {
                        name: "healing".to_string(),
                        frequency_hint: None,
                    },
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::Const(PhiIRValue::Number(432.0)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(1),
                        frequency_relationship: None,
                        direction: crate::phi_ir::ResonateDirection::TeamA,
                    },
                },
                PhiInstruction {
                    result: Some(2),
                    node: PhiIRNode::Witness {
                        target: None,
                        collapse_policy: crate::phi_ir::CollapsePolicy::Final,
                    },
                },
                PhiInstruction {
                    result: Some(3),
                    node: PhiIRNode::Dissonance,
                },
            ],
            PhiIRNode::Return(3),
        );

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute dissonance opcode");

        assert_eq!(result, PhiIRValue::Number(1.0));
    }
}
