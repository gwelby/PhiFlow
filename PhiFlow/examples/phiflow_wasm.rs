use phiflow::parser::parse_phi_program;
/// PhiFlow WASM Demo
///
/// Compiles a PhiFlow program to WebAssembly Text Format (.wat)
/// and writes it to `output.wat`.
///
/// The output can be loaded in any WASM host using the
/// browser shim at `examples/phiflow_host.js`.
///
/// Run: cargo run --example phiflow_wasm
use phiflow::phi_ir::emitter::emit;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use phiflow::phi_ir::wasm::emit_wat;
use std::fs;

fn main() {
    // --- Source program ---
    let source = r#"
        intention compute {
            let x = 10 + 32
            let y = x * 2
            witness x
            y
        }
    "#;

    println!("=== PhiFlow WASM Codegen Demo ===\n");
    println!("Source:\n{}", source.trim());

    // --- Compile ---
    let exprs = parse_phi_program(source).expect("parse failed");
    let mut program = lower_program(&exprs);

    let mut opt = Optimizer::new(OptimizationLevel::Basic);
    opt.optimize(&mut program);

    println!(
        "\nCoherence score: {:.4} (φ⁻¹ = 0.6180)",
        opt.monitor.coherence_score
    );

    // --- Emit .phivm bytecode ---
    let bytecode = emit(&program);
    fs::write("output.phivm", &bytecode).expect("could not write output.phivm");
    println!("\nBytecode: {} bytes → output.phivm", bytecode.len());

    // --- Emit .wat WebAssembly ---
    let wat = emit_wat(&program);
    fs::write("output.wat", &wat).expect("could not write output.wat");
    println!("WASM:     {} chars → output.wat", wat.len());

    println!("\n=== Generated .wat ===\n");
    println!("{}", wat);

    println!("=== Host ===");
    println!("Load output.wat with: node examples/phiflow_host.js");
    println!("Or open examples/phiflow_browser.html in any browser.");
}
