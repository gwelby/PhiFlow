//! **DEPRECATED** — This module has been moved to `src/_archive/`.
//! It is superseded by the PhiIR pipeline (`src/phi_ir/`) and is no longer compiled.
//! Kept for historical reference only.
//!
// PhiFlow Virtual Machine Module
// Exports interpreter and runtime components

pub mod interpreter;

pub use interpreter::{
    ConsciousnessMonitor, PhiFlowInterpreter, PhiFlowValue, QuantumBackend, RuntimeError,
};
