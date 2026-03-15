// PhiFlow Core Library
// Consciousness-Enhanced Programming Language

// Core modules
pub mod phi_core;
pub mod visualization;
pub mod parser;
pub mod interpreter;

// Compiler modules
pub mod compiler;
pub mod vm;

// Sacred mathematics and consciousness modules
pub mod sacred;
pub mod consciousness;

// Quantum computing integration
pub mod quantum;

// Hardware integration
pub mod hardware;

// Biological computation
pub mod bio_compute;

// Re-export main types for convenience
pub use sacred::{PhiMemoryAllocator, SacredFrequency, SacredFrequencyGenerator};
pub use consciousness::{ConsciousnessMonitor, ConsciousnessState, EEGData};
pub use quantum::{QuantumGate, QuantumCircuit, QuantumResult};

// Re-export compiler and VM types
pub use compiler::{PhiFlowLexer, PhiFlowParser, Token, PhiFlowExpression as CompilerExpression};
pub use vm::{PhiFlowInterpreter, PhiFlowValue, RuntimeError};

// PhiFlow version
pub const VERSION: &str = "1.0.0";

// Sacred constants
pub const PHI: f64 = 1.618033988749895;
pub const LAMBDA: f64 = 0.618033988749895;