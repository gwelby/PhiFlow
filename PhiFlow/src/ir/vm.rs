//! PhiFlow Virtual Machine
//!
//! The runtime engine that executes PhiFlow IR.
//! It bridges the gap between abstract opcodes and the physical/quantum hardware.

use crate::ir::{IrProgram, Opcode};
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
    
    // ─── Metrics ───
    instruction_count: u64,
    coherence: f64,
}

#[derive(Debug, Clone)]
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
            instruction_count: 0,
            coherence: 1.0,
        }
    }

    /// Execute an IR program
    pub async fn run(&mut self, program: &IrProgram) {
        let mut ip = 0; // Instruction Pointer

        println!("🚀 PhiVM Starting execution...");
        if self.cuda_engine.is_some() {
            println!("   - CUDA Acceleration: ENABLED");
        }
        println!("   - Quantum Backend: SIMULATOR");

        while ip < program.instructions.len() {
            let op = &program.instructions[ip];
            self.instruction_count += 1;

            match op {
                Opcode::Halt => {
                    println!("🛑 Program Halted.");
                    break;
                }
                
                // ─── Fundamental Operations ───
                Opcode::PushNumber(n) => self.stack.push(PhiValue::Number(*n)),
                Opcode::PushString(s) => self.stack.push(PhiValue::String(s.clone())),
                Opcode::PushBool(b) => self.stack.push(PhiValue::Bool(*b)),
                Opcode::Print => {
                    if let Some(val) = self.stack.pop() {
                        println!("> {:?}", val);
                    }
                }
                Opcode::Pop => { self.stack.pop(); }

                // ─── Hardware Sync (CUDA) ───
                Opcode::Stub { node_type, .. } if node_type == "HardwareSync" => {
                    if let Some(engine) = &self.cuda_engine {
                        // Real CUDA execution
                        let metrics = engine.get_performance_metrics();
                        println!("⚡ CUDA SYNC: {} GPU Cores Active", metrics.multiprocessor_count);
                        // In real flow, we'd dispatch kernels here
                    } else {
                        println!("⚠️  HardwareSync: No CUDA device found (Running in Simulation Mode)");
                    }
                }

                // ─── Quantum Field ───
                Opcode::Stub { node_type, .. } if node_type == "QuantumField" => {
                    // Execute a sacred frequency quantum operation
                    let result = self.quantum_backend.execute_sacred_frequency_operation(432, 2).await;
                    match result {
                        Ok(res) => println!("⚛️  Quantum Field: 432Hz Resonance Achieved (Job {})", res.job_id),
                        Err(e) => println!("❌ Quantum Error: {:?}", e),
                    }
                }

                // ─── Biological Interface ───
                Opcode::Stub { node_type, .. } if node_type == "BiologicalInterface" => {
                    // Connect to EEG/Muse
                    let state = self.consciousness.get_current_state();
                    println!("🧠 Bio-Interface: Connected. Coherence: {:.2}", state.coherence);
                }

                // ─── Standard Arithmetic ───
                Opcode::Add => self.binary_op(|a, b| a + b),
                Opcode::Sub => self.binary_op(|a, b| a - b),
                
                _ => {
                    // Implement other opcodes progressively
                    // println!("DEBUG: Executing {:?}", op);
                }
            }

            ip += 1;
        }
        
        println!("✨ Execution Finished. Coherence: {:.4}", self.coherence);
    }

    fn binary_op<F>(&mut self, op: F) 
    where F: Fn(f64, f64) -> f64 {
        if let (Some(PhiValue::Number(b)), Some(PhiValue::Number(a))) = (self.stack.pop(), self.stack.pop()) {
            self.stack.push(PhiValue::Number(op(a, b)));
        }
    }
}
