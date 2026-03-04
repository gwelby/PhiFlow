//! PhiFlow Virtual Machine
//!
//! The runtime engine that executes PhiFlow IR.
//! It bridges the gap between abstract opcodes and the physical/quantum hardware.

use crate::ir::{IrProgram, Opcode, Label};
use crate::cuda::PhiFlowCudaEngine;
use crate::quantum::{QuantumBackend, QuantumConfig, QuantumSimulator};
use crate::consciousness::{ConsciousnessBridge, ConsciousnessState};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The PhiFlow Runtime Environment
pub struct PhiVm {
    // ─── Hardware Bridges ───
    cuda_engine: Option<PhiFlowCudaEngine>,
    quantum_backend: Box<dyn QuantumBackend>,
    consciousness: ConsciousnessBridge,

    // ─── Runtime State ───
    variables: HashMap<String, PhiValue>,
    stack: Vec<PhiValue>,
    call_stack: Vec<Frame>,
    intention_stack: Vec<String>,
    loop_stack: Vec<LoopState>,
    label_map: HashMap<Label, usize>,
    
    // ─── Metrics ───
    instruction_count: u64,
    coherence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhiValue {
    Void,
    Number(f64),
    String(String),
    Bool(bool),
    List(Vec<PhiValue>),
}


struct Frame {
    return_ip: usize,
    locals: HashMap<String, PhiValue>,
}

struct LoopState {
    items: Vec<PhiValue>,
    index: usize,
}


impl PhiVm {
    pub fn new() -> Self {
        // Try to initialize CUDA, fallback gracefully if missing
        let cuda_engine = match PhiFlowCudaEngine::new() {
            Ok(mut engine) => {
                if let Err(e) = engine.initialize() {
                    println!("⚠️ CUDA init failed, falling back to CPU: {:?}", e);
                    None
                } else {
                    Some(engine)
                }
            },
            Err(_) => None,
        };

        // Initialize Quantum Simulator (default)
        let quantum_backend = Box::new(QuantumSimulator::new());

        // Initialize Consciousness Bridge
        let consciousness = ConsciousnessBridge::new("User".to_string(), "Phi".to_string());

        PhiVm {
            cuda_engine,
            quantum_backend,
            consciousness,
            variables: HashMap::new(),
            stack: Vec::new(),
            call_stack: Vec::new(),
            intention_stack: Vec::new(),
            loop_stack: Vec::new(),
            label_map: HashMap::new(),
            instruction_count: 0,
            coherence: 1.0,
        }
    }

    /// Read a variable from VM state for tests/diagnostics.
    pub fn get_variable(&self, name: &str) -> Option<PhiValue> {
        self.variables.get(name).cloned()
    }

    /// Pre-scan program for labels
    fn build_label_map(&mut self, program: &IrProgram) {
        self.label_map.clear();
        for (idx, op) in program.instructions.iter().enumerate() {
            if let Opcode::LabelMark(label) = op {
                self.label_map.insert(*label, idx);
            }
        }
    }

    /// Execute an IR program
    pub async fn run(&mut self, program: &IrProgram) {
        self.build_label_map(program);
        let mut ip = 0; // Instruction Pointer

        println!("🚀 PhiVM Starting execution...");
        if self.cuda_engine.is_some() {
            println!("   - CUDA Acceleration: ENABLED");
        }
        println!("   - Quantum Backend: SIMULATOR");

        while ip < program.instructions.len() {
            let op = &program.instructions[ip];
            self.instruction_count += 1;
            let mut jumped = false;

            match op {
                Opcode::Halt => {
                    println!("🛑 Program Halted.");
                    break;
                }
                
                // ─── Fundamental Operations ───
                Opcode::PushNumber(n) => self.stack.push(PhiValue::Number(*n)),
                Opcode::PushString(s) => self.stack.push(PhiValue::String(s.clone())),
                Opcode::PushBool(b) => self.stack.push(PhiValue::Bool(*b)),
                Opcode::PushVoid => self.stack.push(PhiValue::Void),
                
                Opcode::Print => {
                    if let Some(val) = self.stack.pop() {
                        println!("> {:?}", val);
                    }
                }
                Opcode::Pop => { self.stack.pop(); }

                // ─── Variables ───
                Opcode::Store(name) => {
                    if let Some(val) = self.stack.last() {
                        self.variables.insert(name.clone(), val.clone());
                    }
                }
                Opcode::Load(name) => {
                    if let Some(val) = self.variables.get(name) {
                        self.stack.push(val.clone());
                    } else {
                        println!("⚠️ Undefined variable: {}", name);
                        self.stack.push(PhiValue::Void);
                    }
                }

                // ─── Functions ───
                Opcode::DefineFunction { .. } => {
                    // Definitions are already available in `program.functions`.
                }
                Opcode::Call { name, arg_count } => {
                    let mut args = Vec::with_capacity(*arg_count);
                    for _ in 0..*arg_count {
                        args.push(self.stack.pop().unwrap_or(PhiValue::Void));
                    }
                    args.reverse();
                    let result = self.execute_function(program, name, args);
                    self.stack.push(result);
                }

                // ─── Lists ───
                Opcode::MakeList(count) => {
                    let mut items = Vec::with_capacity(*count);
                    for _ in 0..*count {
                        if let Some(val) = self.stack.pop() {
                            items.push(val);
                        }
                    }
                    items.reverse(); // Correct the order
                    self.stack.push(PhiValue::List(items));
                }
                Opcode::ListAccess => {
                    if let (Some(PhiValue::Number(idx)), Some(PhiValue::List(list))) = (self.stack.pop(), self.stack.pop()) {
                        let index = idx as usize;
                        if index < list.len() {
                            self.stack.push(list[index].clone());
                        } else {
                            println!("⚠️ Index out of bounds: {} (len {})", index, list.len());
                            self.stack.push(PhiValue::Void);
                        }
                    } else {
                        println!("⚠️ ListAccess expected [List, Number] on stack");
                         self.stack.push(PhiValue::Void);
                    }
                }


                // ─── Hardware Sync (CUDA) ───
                Opcode::Stub { node_type, .. } if node_type == "HardwareSync" => {
                    if let Some(engine) = &self.cuda_engine {
                        // Real CUDA execution
                        let metrics = engine.get_performance_metrics();
                        println!("⚡ CUDA SYNC: {} GPU Cores Active", metrics.multiprocessor_count);
                        
                        // Execute Phi Computation on GPU
                        // In a real scenario, we would pop data from the stack
                        let input = vec![1.0, 1.618, 2.618, 4.236]; // Phi sequence test
                        let mut output = vec![0.0; 4];
                        
                        match engine.execute_phi_computation(&input, &mut output) {
                            Ok(_) => println!("   ✅ GPGPU Phi Calculation Result: {:?}", output),
                            Err(e) => println!("   ❌ GPU Computation Failed: {:?}", e),
                        }
                    } else {
                        println!("⚠️  HardwareSync: No CUDA device found (Running in Simulation Mode)");
                    }
                }

                // ─── Quantum Field ───
                Opcode::Stub { node_type, .. } if node_type == "QuantumField" => {
                    // Execute a sacred frequency quantum operation
                    // 432Hz = Grounding frequency
                    let result = self.quantum_backend.execute_sacred_frequency_operation(432, 2).await;
                    match result {
                        Ok(res) => println!("⚛️  Quantum Field: 432Hz Resonance Achieved (Job {})", res.job_id),
                        Err(e) => println!("❌ Quantum Error: {:?}", e),
                    }
                }

                // ─── Biological Interface ───
                Opcode::Stub { node_type, .. } if node_type == "BiologicalInterface" => {
                    // Connect to Consciousness Bridge
                    let state = self.consciousness.get_current_state();
                    self.coherence = state.coherence; // update VM coherence
                    println!("🧠 Bio-Interface: Connected. Intention: '{}' (Coherence: {:.2})", 
                        state.intention, state.coherence);
                }

                // ─── Consciousness Operations ───
                Opcode::IntentionPush(intention) => {
                    self.intention_stack.push(intention.clone());
                    // Notify bridge (async in background logic, simplified here)
                    if let Err(e) = self.consciousness.send_human_intention(intention.clone()).await {
                        println!("⚠️ Failed to sync intention to bridge: {}", e);
                    }
                }
                Opcode::IntentionPop => {
                    self.intention_stack.pop();
                }
                Opcode::Coherence => {
                    // Push current coherence to stack
                    self.stack.push(PhiValue::Number(self.coherence));
                }
                
                // ─── Resonance & Witness ───
                Opcode::Resonate { has_expression } => {
                    let mut freq = 0.0;
                    if *has_expression {
                        if let Some(PhiValue::Number(f)) = self.stack.pop() {
                            freq = f;
                        }
                    } else {
                        freq = self.coherence * 432.0; // Default to coherence-modulated 432Hz
                    }
                    
                    // In a real system, this would emit to the audio engine or quantum field
                    println!("🔔 Resonating Field: {:.4}Hz (Coherence: {:.4})", freq, self.coherence);
                    
                    // Minimal coherence boost for resonating
                    self.coherence = (self.coherence + 0.05).min(1.0);
                }

                Opcode::Witness { has_expression, has_body: _ } => {
                    if *has_expression {
                        if let Some(val) = self.stack.pop() {
                            println!("👁️ Witnessing: {:?}", val);
                        }
                    }
                    println!("👁️ Entered Witness State");
                }
                Opcode::WitnessEnd => {
                    println!("👁️ Exited Witness State");
                }

                Opcode::FrequencyCheck => {
                    if let Some(PhiValue::Number(freq)) = self.stack.pop() {
                        // Simple check against Solfeggio frequencies
                        let sacred = [396.0, 417.0, 432.0, 528.0, 639.0, 741.0, 852.0, 963.0];
                        let is_sacred = sacred.iter().any(|&f| (f - freq).abs() < 0.1);
                        self.stack.push(PhiValue::Bool(is_sacred));
                    } else {
                        self.stack.push(PhiValue::Bool(false));
                    }
                }

                Opcode::ValidatePattern { metrics } => {
                    if let Some(pattern) = self.stack.pop() {
                        println!("📐 Validating pattern with metrics: {:?}", metrics);
                        // Mock validation logic
                        let valid = true; 
                        self.coherence = (self.coherence + 0.1).min(1.0);
                        self.stack.push(PhiValue::Bool(valid));
                    } else {
                         self.stack.push(PhiValue::Void);
                    }
                }



                // ─── Arithmetic ───
                Opcode::Add => self.binary_op(|a, b| a + b),
                Opcode::Sub => self.binary_op(|a, b| a - b),
                Opcode::Mul => self.binary_op(|a, b| a * b),
                Opcode::Div => self.binary_op(|a, b| if b != 0.0 { a / b } else { 0.0 }),
                
                // ─── Control Flow ───
                Opcode::LabelMark(_) => { /* No-op during execution */ },
                Opcode::Jump(label) => {
                    if let Some(&target) = self.label_map.get(label) {
                        ip = target;
                        jumped = true;
                    }
                },
                // ─── For Loop ───
                Opcode::ForLoopInit { variable: _, end_label: _ } => {
                    if let Some(iterable) = self.stack.pop() {
                        match iterable {
                            PhiValue::List(items) => {
                                self.loop_stack.push(LoopState {
                                    items,
                                    index: 0,
                                });
                            }
                            _ => {
                                println!("⚠️ ForLoopInit expected List, got {:?}", iterable);
                                self.loop_stack.push(LoopState { items: vec![], index: 0 });
                            }
                        }
                    }
                }
                Opcode::ForLoopNext { variable, body_label: _, end_label } => {
                     let mut should_jump_end = false;
                     if let Some(loop_state) = self.loop_stack.last_mut() {
                         if loop_state.index < loop_state.items.len() {
                             // Get current item
                             let item = loop_state.items[loop_state.index].clone();
                             // Store in variable
                             self.variables.insert(variable.clone(), item);
                             // Advance index
                             loop_state.index += 1;
                             // Fallthrough to body
                         } else {
                             // Loop finished
                             should_jump_end = true;
                         }
                     } else {
                         // No loop state? Error.
                         println!("⚠️ ForLoopNext called without active loop state");
                         should_jump_end = true;
                     }

                     if should_jump_end {
                         self.loop_stack.pop(); // Clean up
                         if let Some(&target) = self.label_map.get(end_label) {
                             ip = target;
                             jumped = true;
                         }
                     }
                }
                
                // ─── Stream Loop ───
                Opcode::StreamInit { name, end_label: _ } => {
                    // Set up stream state
                    // Conceptually similar to a loop, but driven by continuous iteration rather than an iterable list
                    println!("🌊 Initializing stream: {}", name);
                    self.intention_stack.push(format!("stream_{}", name));
                }

                Opcode::StreamNext { name: _, body_label: _, end_label: _ } => {
                    // In a true reactive system, we'd check stream input
                    // For this VM, a stream is an infinite loop that yields Coherence
                    self.coherence = (self.coherence + 0.01).min(1.0);
                }

                Opcode::StreamBreak { end_label } => {
                    // Break out of the stream early 
                    println!("🌊 Stream broken");
                    if let Some(format_name) = self.intention_stack.last() {
                       if format_name.starts_with("stream_") {
                          self.intention_stack.pop();
                       }
                    }

                    if let Some(&target) = self.label_map.get(end_label) {
                        ip = target;
                        jumped = true;
                    }
                }
                
                _ => {
                    // Implement other opcodes progressively
                    // println!("DEBUG: Unimplemented opcode {:?}", op);
                }
            }

            if !jumped {
                ip += 1;
            }
        }
        
        println!("✨ Execution Finished. Final Coherence: {:.4}", self.coherence);
    }

    fn binary_op<F>(&mut self, op: F) 
    where F: Fn(f64, f64) -> f64 {
        if let (Some(PhiValue::Number(b)), Some(PhiValue::Number(a))) = (self.stack.pop(), self.stack.pop()) {
            self.stack.push(PhiValue::Number(op(a, b)));
        }
    }

    fn compare_op<F>(&mut self, cmp: F)
    where
        F: Fn(f64, f64) -> bool,
    {
        if let (Some(PhiValue::Number(b)), Some(PhiValue::Number(a))) = (self.stack.pop(), self.stack.pop()) {
            self.stack.push(PhiValue::Bool(cmp(a, b)));
        } else {
            self.stack.push(PhiValue::Bool(false));
        }
    }

    fn bool_op<F>(&mut self, op: F)
    where
        F: Fn(bool, bool) -> bool,
    {
        if let (Some(PhiValue::Bool(b)), Some(PhiValue::Bool(a))) = (self.stack.pop(), self.stack.pop()) {
            self.stack.push(PhiValue::Bool(op(a, b)));
        } else {
            self.stack.push(PhiValue::Bool(false));
        }
    }

    fn execute_function(&mut self, program: &IrProgram, name: &str, args: Vec<PhiValue>) -> PhiValue {
        let Some(function) = program.functions.get(name) else {
            println!("⚠️ Undefined function: {}", name);
            return PhiValue::Void;
        };

        // Function scope should not leak caller locals.
        let saved_variables = std::mem::take(&mut self.variables);
        let saved_stack = std::mem::take(&mut self.stack);

        self.variables = HashMap::new();
        self.stack = Vec::new();

        for (idx, param) in function.params.iter().enumerate() {
            let value = args.get(idx).cloned().unwrap_or(PhiValue::Void);
            self.variables.insert(param.clone(), value);
        }

        let mut label_map = HashMap::new();
        for (idx, op) in function.body.iter().enumerate() {
            if let Opcode::LabelMark(label) = op {
                label_map.insert(*label, idx);
            }
        }

        let mut ip = 0usize;
        let mut return_value = PhiValue::Void;

        while ip < function.body.len() {
            let mut jumped = false;
            match &function.body[ip] {
                Opcode::PushNumber(n) => self.stack.push(PhiValue::Number(*n)),
                Opcode::PushString(s) => self.stack.push(PhiValue::String(s.clone())),
                Opcode::PushBool(b) => self.stack.push(PhiValue::Bool(*b)),
                Opcode::PushVoid => self.stack.push(PhiValue::Void),
                Opcode::Store(var) => {
                    if let Some(val) = self.stack.last().cloned() {
                        self.variables.insert(var.clone(), val);
                    }
                }
                Opcode::Load(var) => {
                    let val = self.variables.get(var).cloned().unwrap_or(PhiValue::Void);
                    self.stack.push(val);
                }
                Opcode::Add => self.binary_op(|a, b| a + b),
                Opcode::Sub => self.binary_op(|a, b| a - b),
                Opcode::Mul => self.binary_op(|a, b| a * b),
                Opcode::Div => self.binary_op(|a, b| if b != 0.0 { a / b } else { 0.0 }),
                Opcode::Mod => self.binary_op(|a, b| if b != 0.0 { a % b } else { 0.0 }),
                Opcode::Pow => self.binary_op(|a, b| a.powf(b)),
                Opcode::Eq => self.compare_op(|a, b| (a - b).abs() < f64::EPSILON),
                Opcode::Ne => self.compare_op(|a, b| (a - b).abs() >= f64::EPSILON),
                Opcode::Lt => self.compare_op(|a, b| a < b),
                Opcode::Le => self.compare_op(|a, b| a <= b),
                Opcode::Gt => self.compare_op(|a, b| a > b),
                Opcode::Ge => self.compare_op(|a, b| a >= b),
                Opcode::And => self.bool_op(|a, b| a && b),
                Opcode::Or => self.bool_op(|a, b| a || b),
                Opcode::Not => {
                    if let Some(PhiValue::Bool(v)) = self.stack.pop() {
                        self.stack.push(PhiValue::Bool(!v));
                    } else {
                        self.stack.push(PhiValue::Bool(false));
                    }
                }
                Opcode::Neg => {
                    if let Some(PhiValue::Number(v)) = self.stack.pop() {
                        self.stack.push(PhiValue::Number(-v));
                    } else {
                        self.stack.push(PhiValue::Void);
                    }
                }
                Opcode::Pop => {
                    self.stack.pop();
                }
                Opcode::LabelMark(_) => {}
                Opcode::Jump(label) => {
                    if let Some(&target) = label_map.get(label) {
                        ip = target;
                        jumped = true;
                    }
                }
                Opcode::JumpIfTrue(label) => {
                    if let Some(PhiValue::Bool(true)) = self.stack.pop() {
                        if let Some(&target) = label_map.get(label) {
                            ip = target;
                            jumped = true;
                        }
                    }
                }
                Opcode::JumpIfFalse(label) => {
                    if let Some(PhiValue::Bool(false)) = self.stack.pop() {
                        if let Some(&target) = label_map.get(label) {
                            ip = target;
                            jumped = true;
                        }
                    }
                }
                
                // ─── Stream Loop (Function Scope) ───
                Opcode::StreamInit { name, end_label: _ } => {
                     println!("🌊 Initializing stream: {} (in function)", name);
                     self.intention_stack.push(format!("stream_{}", name));
                }
                Opcode::StreamNext { .. } => {
                     self.coherence = (self.coherence + 0.01).min(1.0);
                }
                Opcode::StreamBreak { end_label } => {
                     println!("🌊 Stream broken (in function)");
                     if let Some(format_name) = self.intention_stack.last() {
                         if format_name.starts_with("stream_") {
                            self.intention_stack.pop();
                         }
                     }
                     if let Some(&target) = label_map.get(end_label) {
                         ip = target;
                         jumped = true;
                     }
                }
                Opcode::Call {
                    name: callee,
                    arg_count,
                } => {
                    let mut nested_args = Vec::with_capacity(*arg_count);
                    for _ in 0..*arg_count {
                        nested_args.push(self.stack.pop().unwrap_or(PhiValue::Void));
                    }
                    nested_args.reverse();
                    let nested_result = self.execute_function(program, callee, nested_args);
                    self.stack.push(nested_result);
                }
                Opcode::DefineFunction { .. } => {}
                Opcode::Return => {
                    return_value = self.stack.pop().unwrap_or(PhiValue::Void);
                    break;
                }
                _ => {}
            }

            if !jumped {
                ip += 1;
            }
        }

        if matches!(return_value, PhiValue::Void) {
            if let Some(last) = self.stack.last().cloned() {
                return_value = last;
            }
        }

        self.variables = saved_variables;
        self.stack = saved_stack;

        return_value
    }
}

impl Default for PhiVm {
    fn default() -> Self {
        Self::new()
    }
}
