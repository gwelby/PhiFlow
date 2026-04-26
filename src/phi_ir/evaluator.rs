//! PhiIR Direct Evaluator
//!
//! Interprets a `PhiIRProgram` directly, giving the four unique PhiFlow constructs
//! real, observable behavior:
//!
//! - `Witness`           → Captures program state; returns coherence score (0.0–1.0)
//! - `IntentionPush/Pop` → Maintains a live intention stack; scopes execution purpose
//! - `Resonate`          → Shares values through an intention-keyed resonance field
//! - `CoherenceCheck`    → Canonical coherence: base(depth) × phase(k)
//!
//! Canonical runtime formula:
//!   base(depth) = 0 when depth == 0, else `1 - φ^(-depth)`
//!   phase(k)    = 1 when k <= 1, else `1 - ln(k)/ln(τ)`
//!   coherence   = clamp(base(depth) × phase(k), 0, 1)

use crate::host::{DefaultHostProvider, PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::parser::parse_phi_program;
use crate::phi_ir::{
    lowering::lower_program_checked,
    vm_state::{VmState, VmWitnessEvent},
    BlockId, Operand, PhiIRBinOp, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRUnOp, PhiIRValue,
    PhiInstruction, SensorKind,
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
    PolicyViolation(String),
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
            EvalError::PolicyViolation(s) => write!(f, "Policy violation: {}", s),
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
    pub registers: HashMap<Operand, PhiIRValue>,

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
    sensor_provider: Option<Arc<dyn Fn(SensorKind) -> Option<f64> + Send + Sync + 'a>>,

    /// Cumulative coherence penalty from frequent witnessing.
    pub measurement_coherence_penalty: f64,
    /// Maps active stream name to its required minimum coherence threshold.
    pub stream_thresholds: HashMap<String, f64>,

    /// Optional hardware-reality modifier (0.0–1.0).
    /// When present, the canonical phi-stack coherence is multiplied by this value
    /// to produce the actual execution coherence. This lets live sensor data (thermal
    /// stress, memory pressure, network degradation) create a reality penalty without
    /// replacing the phi-stack baseline entirely.
    ///
    /// `compute_coherence() = canonical_coherence * hardware_modifier()`
    ///
    /// At idle (low stress): modifier ≈ 0.9–1.0 → minimal impact
    /// Under load (high thermal/memory): modifier ≈ 0.3–0.6 → system self-throttles
    hardware_modifier: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
}

#[derive(Debug, Clone)]
struct FunctionMeta {
    params: Vec<String>,
    body: BlockId,
}

impl<'a> Evaluator<'a> {
    pub fn new(program: PhiIRProgram) -> Self {
        let mut variables = HashMap::new();
        variables.insert("PHI".to_string(), PhiIRValue::Number(PHI));

        let mut eval = Self {
            program: program.clone(),
            functions: HashMap::new(),
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
            sensor_provider: None,
            measurement_coherence_penalty: 0.0,
            stream_thresholds: HashMap::new(),
            hardware_modifier: None,
        };
        eval.rebuild_functions_map();
        eval
    }

    fn rebuild_functions_map(&mut self) {
        self.functions.clear();
        for block in &self.program.blocks {
            for instr in &block.instructions {
                if let PhiIRNode::FuncDef { name, params, body } = &instr.node {
                    self.functions.insert(
                        name.clone(),
                        FunctionMeta {
                            params: params.iter().map(|p| p.name.clone()).collect(),
                            body: *body,
                        },
                    );
                }
            }
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

    /// Inject new IR blocks into the running evaluator state.
    pub fn evolve(&mut self, evolved_prog: PhiIRProgram) {
        let id_offset = self.program.blocks.len() as BlockId;
        for mut block in evolved_prog.blocks.clone() {
            block.id += id_offset;
            self.program.blocks.push(block);
        }
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

    pub fn with_sensor_provider<F>(mut self, provider: F) -> Self
    where
        F: Fn(SensorKind) -> Option<f64> + Send + Sync + 'a,
    {
        self.sensor_provider = Some(Arc::new(provider));
        self
    }

    /// Set a hardware-reality modifier: a closure returning a 0.0–1.0 score that
    /// represents the current physical execution environment health.
    ///
    /// This modifier is applied multiplicatively to the internal phi-stack coherence:
    ///   `compute_coherence() = canonical_phi_coherence × hardware_modifier()`
    ///
    /// This is distinct from `with_coherence_provider()` (which replaces the score
    /// entirely for testing). The hardware modifier preserves the phi-stack
    /// baseline while letting real hardware conditions (thermal stress, memory
    /// pressure, network degradation) introduce a physical reality penalty.
    pub fn with_hardware_modifier<F>(mut self, modifier: F) -> Self
    where
        F: Fn() -> f64 + Send + Sync + 'static,
    {
        self.hardware_modifier = Some(Arc::new(modifier));
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
            if loop_counter > 100000 && self.max_steps.is_none() {
                return Err(EvalError::StepLimitExceeded(100000));
            }
            let block_id = self.current_block;
            let block = self.get_block(block_id)?;
            let instr_count = block.instructions.len();

            // Threshold enforcement: Check if any active stream has a threshold we've fallen below
            let current_coherence = self.compute_coherence();
            for stream in &self.active_streams {
                if let Some(threshold) = self.stream_thresholds.get(stream) {
                    if current_coherence < *threshold {
                        let snapshot = WitnessSnapshot {
                            intention_stack: self.intention_stack.clone(),
                            coherence: current_coherence,
                            register_count: self.registers.len(),
                            resonance_count: self.resonance_count(),
                            observed_value: Some(format!("Threshold yield for stream '{}': {:.3} < {:.3}", stream, current_coherence, threshold)),
                            agent_name: self.agent_name.clone(),
                        };
                        let frozen = self.freeze_state();
                        return Ok(VmExecResult::Yielded {
                            snapshot,
                            frozen_state: frozen,
                        });
                    }
                }
            }

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
        self.program = state.program;
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
        self.measurement_coherence_penalty = state.measurement_coherence_penalty;
        
        self.rebuild_functions_map();
        self.run_or_yield()
    }

    /// Capture the current evaluator state as a frozen snapshot.
    pub fn freeze_state(&self) -> FrozenEvalState {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        FrozenEvalState {
            program: self.program.clone(),
            registers: self.registers.clone(),
            variables: self.variables.clone(),
            intention_stack: self.intention_stack.clone(),
            active_streams: self.active_streams.clone(),
            resonance_field: self.resonance_field.clone(),
            // Cap history to prevent state-file bloat (Singularity Hardening)
            resonance_events: self.resonance_events.iter().skip(self.resonance_events.len().saturating_sub(100)).cloned().collect(),
            ended_streams: self.ended_streams.clone(),
            witness_log: self.witness_log.iter().skip(self.witness_log.len().saturating_sub(100)).cloned().collect(),
            current_block: self.current_block,
            instruction_ptr: self.instruction_ptr,
            yield_timestamp: Some(now),
            agent_name: self.agent_name.clone(),
            agent_version: self.agent_version.clone(),
            measurement_coherence_penalty: self.measurement_coherence_penalty,
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

    /// Extends the evaluator's runtime state by injecting a classical outcome or variable.
    pub fn inject_variable(&mut self, name: &str, value: PhiIRValue) {
        self.variables.insert(name.to_string(), value);
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

            PhiIRNode::WitnessSensor { sensor } => {
                let (value, _snapshot, _action) = self.process_sensor_witness(*sensor)?;
                Some(PhiIRValue::Number(value))
            }

            PhiIRNode::AnchorGate {
                target,
                min_presence,
                frequency,
                gate_fidelity,
            } => {
                use crate::phi_ir::SensorKind;
                use crate::security::anchor::AnchorError;
                const IBM_HERON_R2_GATE_FIDELITY_SPEC: f64 = 0.992;

                let soma_presence_raw = if let Some(ref prov) = self.sensor_provider {
                    prov(SensorKind::SomaPresence)
                } else {
                    crate::sensors::read_sensor(SensorKind::SomaPresence)
                };

                let soma_432_raw = if let Some(ref prov) = self.sensor_provider {
                    prov(SensorKind::Soma432)
                } else {
                    crate::sensors::read_sensor(SensorKind::Soma432)
                };

                match soma_presence_raw {
                    None => {
                        println!(
                            "[anchor: {}] SOMA absent or stale — ObserveOnly mode (no hardware blocking)",
                            target
                        );
                    }
                    Some(presence) => {
                        if presence < *min_presence {
                            let AnchorError::PolicyViolation(msg) = AnchorError::PolicyViolation(format!(
                                "anchor '{}': soma_presence {:.3} < required {:.3}",
                                target, presence, min_presence
                            )) else { unreachable!() };
                            return Err(EvalError::PolicyViolation(msg));
                        }
                        println!(
                            "[anchor: {}] presence check PASS ({:.3} >= {:.3})",
                            target, presence, min_presence
                        );
                    }
                }

                match soma_432_raw {
                    None => {
                        println!(
                            "[anchor: {}] soma_432 sensor absent — frequency check skipped (ObserveOnly)",
                            target
                        );
                    }
                    Some(freq_val) => {
                        let freq_diff = (freq_val - frequency).abs();
                        if freq_diff > 5.0 {
                            let AnchorError::PolicyViolation(msg) = AnchorError::PolicyViolation(format!(
                                "anchor '{}': soma_432 {:.2} Hz is {:.2} Hz away from required {:.2} Hz (tolerance ±5.0 Hz)",
                                target, freq_val, freq_diff, frequency
                            )) else { unreachable!() };
                            return Err(EvalError::PolicyViolation(msg));
                        }
                        println!(
                            "[anchor: {}] frequency check PASS (soma_432={:.2} Hz, target={:.2} Hz)",
                            target, freq_val, frequency
                        );
                    }
                }

                if *gate_fidelity > IBM_HERON_R2_GATE_FIDELITY_SPEC {
                    let AnchorError::PolicyViolation(msg) = AnchorError::PolicyViolation(format!(
                        "anchor '{}': gate_fidelity threshold {:.4} exceeds IBM Heron r2 spec baseline {:.4} [spec-based, not live-calibrated]",
                        target, gate_fidelity, IBM_HERON_R2_GATE_FIDELITY_SPEC
                    )) else { unreachable!() };
                    return Err(EvalError::PolicyViolation(msg));
                }
                println!(
                    "[anchor: {}] gate_fidelity check PASS (threshold={:.4}, spec_baseline={:.4}) [spec-based, not live-calibrated]",
                    target, gate_fidelity, IBM_HERON_R2_GATE_FIDELITY_SPEC
                );

                let coherence = self.compute_coherence();
                self.witness_log.push(WitnessEvent {
                    intention_stack: self.intention_stack.clone(),
                    coherence,
                    register_count: self.registers.len(),
                    resonance_count: self.resonance_count(),
                    agent_name: self.agent_name.clone(),
                });

                None
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

            PhiIRNode::StreamPush(name, threshold) => {
                self.intention_stack.push(name.clone());
                self.active_streams.push(name.clone());
                self.resonance_field.insert(name.clone(), Vec::new());
                if let Some(t) = threshold {
                    self.stream_thresholds.insert(name.clone(), *t);
                }
                None
            }

            PhiIRNode::StreamPop => {
                self.intention_stack.pop();
                if let Some(stream_name) = self.active_streams.pop() {
                    self.ended_streams.push(stream_name.clone());
                    self.stream_thresholds.remove(&stream_name);
                }
                None
            }

            PhiIRNode::FieldCoherence => {
                if let Some(shared) = &self.shared_resonance {
                    let guard = shared.lock().unwrap();
                    // Field coherence: average of recent resonations or average of active stream scores
                    // For now, let's just average the lengths or values if numeric
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

            PhiIRNode::Dissonance => {
                let history = &self.witness_log;
                if history.len() < 2 {
                    Some(PhiIRValue::Number(0.0))
                } else {
                    let last = history[history.len() - 1].coherence;
                    let prev = history[history.len() - 2].coherence;
                    let delta = last - prev;
                    // Normalize to -1.0 .. 1.0. A 0.1 change is significant.
                    let normalized = (delta * 10.0).clamp(-1.0, 1.0);
                    Some(PhiIRValue::Number(normalized))
                }
            }

            PhiIRNode::CoherenceOf(name) => {
                if let Some(shared) = &self.shared_resonance {
                    let guard = shared.lock().unwrap();
                    if let Some(vals) = guard.get(name) {
                        // Return the last resonated value if it's a number
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

            PhiIRNode::Resonate {
                value,
                direction: _,
                ..
            } => {
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

                        // --- Pipe 3: MQTT Resonance Bus ---
                        let json_val = match &val {
                            PhiIRValue::Number(n) => serde_json::json!(n),
                            PhiIRValue::String(s) => serde_json::json!(s),
                            PhiIRValue::Boolean(b) => serde_json::json!(b),
                            PhiIRValue::Void => serde_json::Value::Null,
                        };
                        let _ = crate::resonance_bus::emit_resonance(json_val, &key, "phiflow");
                        // ----------------------------------

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
                let evolved_prog = lower_program_checked(&exprs)
                    .map_err(|e| EvalError::SynthesisError(e.to_string()))?;

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

            PhiIRNode::Handoff {
                target_agent,
                task_id,
                context_op,
            } => {
                let context_val = self.get_reg(*context_op)?.clone();
                let context_json = match &context_val {
                    PhiIRValue::Number(n) => serde_json::json!(n),
                    PhiIRValue::String(s) => serde_json::json!(s),                    PhiIRValue::Boolean(b) => serde_json::json!(b),
                    PhiIRValue::Void => serde_json::Value::Null,
                };

                let coherence = self.coherence();
                let dissonance = if self.witness_log.len() < 2 {
                    0.0
                } else {
                    let last = self.witness_log[self.witness_log.len() - 1].coherence;
                    let prev = self.witness_log[self.witness_log.len() - 2].coherence;
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

                // Route through host (allows signing in SystemHostProvider)
                self.host.broadcast("_handoff", &handoff_payload.to_string());

                // Log locally
                self.resonance_events.push(("_handoff".to_string(), context_val.clone()));
                self.resonance_field
                    .entry("_handoff".to_string())
                    .or_default()
                    .push(context_val);

                None
            }

            // --- Domain calls: no-op in base evaluator ---
            PhiIRNode::DomainCall {
                op,
                args,
                string_args,
            } => {
                match op {
                    crate::phi_ir::DomainOp::QuantumField => {
                        // AntiGravity Pipe 2: IBM Quantum Feedback
                        // Emits QASM, Polls, and triggers Self-Correction
                        let job_id = string_args
                            .get(0)
                            .cloned()
                            .unwrap_or_else(|| "mock_job".to_string());
                        let api_key = std::env::var("IBM_QUANTUM_API_KEY")
                            .unwrap_or_else(|_| "MOCK_KEY".to_string());

                        if let Ok(counts) = crate::quantum_feedback::poll_ibm_job(&job_id, &api_key)
                        {
                            let coherence = crate::quantum_feedback::calculate_coherence(&counts);
                            if let Some(correction_source) =
                                crate::quantum_feedback::generate_correction_if_needed(coherence)
                            {
                                // Trigger evolve internally
                                let exprs = parse_phi_program(&correction_source).map_err(|e| {
                                    EvalError::SynthesisError(format!("Parse failed: {}", e))
                                })?;
                                let evolved_prog = lower_program_checked(&exprs)
                                    .map_err(|e| EvalError::SynthesisError(e.to_string()))?;

                                let max_id =
                                    self.program.blocks.iter().map(|b| b.id).max().unwrap_or(0);
                                let id_offset = max_id + 1;

                                for mut block in evolved_prog.blocks.clone() {
                                    block.id += id_offset;
                                    self.remap_block_ids(&mut block.terminator, id_offset);
                                    self.program.blocks.push(block);
                                }

                                let msg =
                                    format!("Quantum Feedback Evolved logic at {:.3}c", coherence);
                                self.resonance_events.push((
                                    "_quantum_evolution".to_string(),
                                    PhiIRValue::Number(coherence),
                                ));
                                self.resonance_field
                                    .entry("_quantum_evolution".to_string())
                                    .or_default();
                                self.host.on_resonate("_quantum_evolution", &msg);

                                let saved_block = self.current_block;
                                let saved_ip = self.instruction_ptr;
                                self.current_block = evolved_prog.entry + id_offset;
                                self.instruction_ptr = 0;

                                let _evolved_result = loop {
                                    let block_id = self.current_block;
                                    let block = self.get_block(block_id)?.clone();

                                    if self.instruction_ptr < block.instructions.len() {
                                        let instr =
                                            block.instructions[self.instruction_ptr].clone();
                                        self.instruction_ptr += 1;
                                        self.execute_instruction(&instr)?;
                                        continue;
                                    }

                                    let terminator = block.terminator.clone();
                                    if let Some(val) = self.execute_terminator(&terminator)? {
                                        break val;
                                    }
                                };

                                self.current_block = saved_block;
                                self.instruction_ptr = saved_ip;
                            }
                        }
                        Some(PhiIRValue::Void)
                    }
                    _ => Some(PhiIRValue::Void),
                }
            }
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
            PhiIRValue::String(s) => s.clone(),
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
        // Sample coherence BEFORE applying the observer cost penalty.
        // The disturbance from this observation affects the NEXT reading, not the current one.
        // (Canonical quantum semantics: measuring disturbs *future* state.)
        let observed = target.and_then(|op| self.get_reg(op).ok().cloned());
        let coherence = self.compute_coherence();

        // Now accrue the observer-cost penalty for subsequent measurements
        self.measurement_coherence_penalty += 0.01;
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

    fn process_sensor_witness(
        &mut self,
        sensor: SensorKind,
    ) -> EvalResult<(f64, WitnessSnapshot, WitnessAction)> {
        // Sample the sensor value, then accrue the observer-cost penalty
        let value = self.resolve_sensor(sensor)?;
        self.measurement_coherence_penalty += 0.01;
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
            observed_value: Some(format!("sensor({}): {}", sensor.as_name(), value)),
            agent_name: self.agent_name.clone(),
        };
        let action = self.host.on_witness(&snapshot);

        Ok((value, snapshot, action))
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

        if let PhiIRNode::WitnessSensor { sensor } = &instr.node {
            let (value, snapshot, action) = self.process_sensor_witness(*sensor)?;

            if let Some(reg) = instr.result {
                self.registers.insert(reg, PhiIRValue::Number(value));
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
        let raw = crate::phi_ir::coherence::canonical_coherence(&self.intention_stack, &self.resonance_field);
        let phi_coherence = (raw - self.measurement_coherence_penalty).max(0.0);

        // Apply hardware-reality modifier if wired.
        // This makes live sensor conditions (thermal stress, memory pressure, network
        // degradation) reduce execution coherence multiplicatively — the system
        // self-throttles exactly as a consciousness-aware runtime should.
        match &self.hardware_modifier {
            Some(modifier) => {
                let hw = modifier().clamp(0.0, 1.0);
                phi_coherence * hw
            }
            None => phi_coherence,
        }
    }

    fn resolve_coherence(&self) -> f64 {
        let internal = self.compute_coherence();
        self.host.get_coherence(internal)
    }

    fn resolve_sensor(&self, sensor: SensorKind) -> EvalResult<f64> {
        let value = self
            .sensor_provider
            .as_ref()
            .and_then(|provider| provider(sensor))
            .or_else(|| crate::sensors::read_sensor(sensor));

        // Host and SOMA sensors are optional environmental enrichment.
        // When a sensor is unavailable on this specific host or the SOMA bridge is offline,
        // degrade gracefully to 0.0 (no signal) rather than aborting the program.
        match value {
            Some(v) => Ok(v),
            None => Ok(0.0),
        }
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
            (PhiIRValue::Void, PhiIRValue::Void) => match op {
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(true)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(false)),
                _ => Err(EvalError::InvalidOperation("Unsupported Void binary op".to_string())),
            },
            (PhiIRValue::String(l_idx), PhiIRValue::String(r_idx)) => match op {
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(l_idx == r_idx)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(l_idx != r_idx)),
                _ => Err(EvalError::InvalidOperation("Unsupported String binary op".to_string())),
            },
            (a, b) => match op {
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(false)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(true)),
                _ => Err(EvalError::InvalidOperation(format!(
                    "Type mismatch in binary operation between {:?} and {:?}",
                    a, b
                ))),
            },
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
