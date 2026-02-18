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
                    println!("🔔 Resonating Field: {:.1}Hz (Coherence: {:.4})", freq, self.coherence);
                    
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

                Opcode::JumpIfTrue(label) => {
                    if let Some(PhiValue::Bool(true)) = self.stack.pop() {
                        if let Some(&target) = self.label_map.get(label) {
                            ip = target;
                            jumped = true;
                        }
                    }
                },
                Opcode::JumpIfFalse(label) => {
                    if let Some(PhiValue::Bool(false)) = self.stack.pop() {
                        if let Some(&target) = self.label_map.get(label) {
                            ip = target;
                            jumped = true;
                        }
                    }
                },
                
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
}
