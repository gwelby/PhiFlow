// PhiFlow Core Library
// Consciousness-Enhanced Programming Language

// Future-facing scaffolding: sacred/consciousness/visualization modules
// contain living design vocabulary not yet wired to the new PhiIR backend.
// Suppress noise until they are activated.
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    non_upper_case_globals
)]

// Core modules
pub mod interpreter;
pub mod ir;
pub mod parser;
pub mod phi_diagnostics;
pub mod phi_core;
pub mod phi_ir;
pub mod visualization;

// Compiler modules
pub mod compiler;
pub mod vm;
pub mod sensors;

// Sacred mathematics and consciousness modules
pub mod consciousness;
pub mod sacred;

// Quantum computing integration
pub mod quantum;

// Hardware integration
pub mod cuda;
pub mod hardware;

// Biological computation
pub mod bio_compute;

// Re-export main types for convenience
pub use consciousness::{ConsciousnessMonitor, ConsciousnessState, EEGData};
pub use quantum::{QuantumCircuit, QuantumGate, QuantumResult};
pub use sacred::{PhiMemoryAllocator, SacredFrequency, SacredFrequencyGenerator};
pub use phi_diagnostics::PhiDiagnostic;

// Re-export compiler and VM types
pub use compiler::{PhiFlowExpression as CompilerExpression, PhiFlowLexer, PhiFlowParser, Token};
pub use vm::{PhiFlowInterpreter, PhiFlowValue, RuntimeError};

// PhiFlow version
pub const VERSION: &str = "1.0.0";

// Sacred constants
pub const PHI: f64 = 1.618033988749895;
pub const LAMBDA: f64 = 0.618033988749895;

/// Compile and run PhiFlow source code using the new PhiIR pipeline.
/// Returns the final result of the program.
pub fn compile_and_run_phi_ir(source: &str) -> Result<phi_ir::PhiIRValue, String> {
    // 1. Parse using the new parser (src/parser/mod.rs → PhiExpression)
    use parser::parse_phi_program;
    let expressions = parse_phi_program(source).map_err(|e| format!("Parse error: {}", e))?;

    // 2. Lower AST → PhiIR
    use phi_ir::lowering::lower_program;
    let mut program = lower_program(&expressions);

    // 3. Optimize
    use phi_ir::optimizer::{OptimizationLevel, Optimizer};
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    // 4. Evaluate
    use phi_ir::evaluator::Evaluator;
    let mut evaluator = Evaluator::new(&program);
    evaluator
        .run()
        .map_err(|e| format!("Runtime error: {:?}", e))
}
