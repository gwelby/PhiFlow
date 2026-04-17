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
pub mod host;
pub mod interpreter;
pub mod ir;
pub mod mcp_server; // <-- Added MCP
pub mod parser;
pub mod phi_core;
pub mod phi_diagnostics;
pub mod phi_ir;
pub mod visualization;

// Compiler modules
pub mod compiler;
pub mod resonance_bus;
pub mod sensors;
pub mod system_host;
pub mod vm;
pub mod wasm_host;

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
pub use host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
pub use system_host::SystemHostProvider;
pub use phi_diagnostics::PhiDiagnostic;
pub use phi_ir::TeamDirection;
pub use quantum::{QuantumCircuit, QuantumGate, QuantumResult};
pub use sacred::{PhiMemoryAllocator, SacredFrequency, SacredFrequencyGenerator};

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
/// Also emits any `resonate` events to the resonance bus (JSONL file).
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

    // 4. Evaluate with a host that writes resonate events to the resonance bus
    use host::CallbackHostProvider;
    use phi_ir::evaluator::Evaluator;

    let host = CallbackHostProvider::new().with_resonate(|intention, value_str| {
        // Parse the string back to a serde_json Value for the bus
        let json_val: serde_json::Value = value_str
            .parse::<f64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| serde_json::Value::String(value_str.to_string()));
        let _ = resonance_bus::emit_resonance(json_val, intention, "phiflow_evaluator");
    });

    let mut evaluator = Evaluator::new(&program).with_host(Box::new(host));
    evaluator
        .run()
        .map_err(|e| format!("Runtime error: {:?}", e))
}
