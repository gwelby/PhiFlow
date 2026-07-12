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
pub mod cascade_keys;
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
pub mod quantum_feedback;
pub mod resonance_bus;
pub mod sensors;
pub mod system_host;
pub mod security;
pub mod vm;
pub mod wasm_host;

// Type 4 Observer Metrics (PF consciousness_metric_program)
pub mod metrics;

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
pub use phi_diagnostics::PhiDiagnostic;
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

#[derive(Clone, Default)]
pub struct OpenQasmCompileOptions {
    pub optimize_depth: bool,
    pub topology: Option<phi_ir::topology_transpiler::TopologyTranspileConfig>,
    pub live_backend_profile: Option<quantum::backend_topology::BackendTopologyProfile>,
    pub anchor_signing_key: Option<std::sync::Arc<security::anchor::AnchorSigningKey>>,
}

/// Compile and run PhiFlow source code using the new PhiIR pipeline.
/// Returns the final result of the program.
pub fn compile_and_run_phi_ir(source: &str) -> Result<phi_ir::PhiIRValue, String> {
    // 1. Parse using the new parser (src/parser/mod.rs → PhiExpression)
    use parser::parse_phi_program;
    let expressions = parse_phi_program(source).map_err(|e| format!("Parse error: {}", e))?;

    // 2. Lower AST → PhiIR
    use phi_ir::lowering::lower_program_checked;
    let mut program =
        lower_program_checked(&expressions).map_err(|e| format!("Lowering error: {}", e))?;

    // 3. Optimize
    use phi_ir::optimizer::{OptimizationLevel, Optimizer};
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    // 4. Evaluate
    use phi_ir::evaluator::Evaluator;
    let mut evaluator = Evaluator::new(program);
    evaluator
        .run()
        .map_err(|e| format!("Runtime error: {:?}", e))
}

/// Compile PhiFlow source to OpenQASM 3.0 using the canonical PhiIR path.
pub fn compile_to_openqasm(source: &str, optimize_depth: bool) -> Result<String, String> {
    compile_to_openqasm_with_options(
        source,
        &OpenQasmCompileOptions {
            optimize_depth,
            ..OpenQasmCompileOptions::default()
        },
    )
}

/// Compile PhiFlow source to OpenQASM 3.0 with optional topology-aware routing.
pub fn compile_to_openqasm_with_options(
    source: &str,
    options: &OpenQasmCompileOptions,
) -> Result<String, String> {
    use parser::parse_phi_program;
    use phi_ir::lowering::lower_program_checked;
    use phi_ir::openqasm::OpenQasmEmitter;
    use phi_ir::optimizer::{OptimizationLevel, Optimizer};
    use phi_ir::quantum_interaction::analyze_quantum_overlay;

    let expressions = parse_phi_program(source).map_err(|e| format!("Parse error: {}", e))?;
    let mut program =
        lower_program_checked(&expressions).map_err(|e| format!("Lowering error: {}", e))?;

    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut emitter = OpenQasmEmitter::new();
    emitter.optimize_depth = options.optimize_depth;
    if let Some(ref key) = options.anchor_signing_key {
        emitter.anchor_fingerprint_ecdsa = Some(key.fingerprint());
        emitter.anchor_fingerprint_pq = Some(key.fingerprint_pq());
    }

    if let Some(topology) = &options.topology {
        let profile = options
            .live_backend_profile
            .as_ref()
            .ok_or_else(|| {
                "Topology-aware OpenQASM compilation requires a backend topology profile"
                    .to_string()
            })?;
        let overlay = analyze_quantum_overlay(&program)
            .map_err(|e| format!("Quantum overlay analysis error: {}", e))?;
        emitter
            .emit_with_topology(&program, &overlay, profile, topology)
            .map_err(|e| e.to_string())
    } else {
        emitter.emit(&program).map_err(|e| e.to_string())
    }
}
