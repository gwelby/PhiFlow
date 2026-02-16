// PhiFlow Virtual Machine Module
// Exports interpreter and runtime components

pub mod interpreter;

pub use interpreter::{PhiFlowInterpreter, PhiFlowValue, RuntimeError, ConsciousnessMonitor, QuantumBackend};