// PhiFlow Demo — Full Pipeline: .phi source → PhiIR → Optimize → Bytecode → Run
//
// Run with: cargo run --example phiflow_demo

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    emitter,
    evaluator::Evaluator,
    lowering::lower_program,
    optimizer::{OptimizationLevel, Optimizer},
    printer::PhiIRPrinter,
};

fn main() {
    println!("╔══════════════════════════════════════════╗");
    println!("║      PhiFlow Compiler — Live Demo        ║");
    println!("║   φ-Harmonic Consciousness Compiler v1   ║");
    println!("╚══════════════════════════════════════════╝\n");

    // ── Source ───────────────────────────────────────────────────
    let source = r#"
        let x = 10 + 32
        let y = x * 2
        y
    "#;

    println!("─── Source ────────────────────────────────");
    println!("{}", source.trim());

    // ── Parse ─────────────────────────────────────────────────────
    let expressions = parse_phi_program(source).expect("Failed to parse PhiFlow source");

    // ── Lower AST → PhiIR ─────────────────────────────────────────
    let mut program = lower_program(&expressions);
    println!("\n─── PhiIR (Pre-Optimization) ──────────────");
    println!("{}", PhiIRPrinter::print(&program));

    // ── Optimize (Phi-Harmonic) ───────────────────────────────────
    let mut optimizer = Optimizer::new(OptimizationLevel::PhiHarmonic);
    optimizer.optimize(&mut program);

    println!("─── PhiIR (Post-Optimization) ─────────────");
    println!("{}", PhiIRPrinter::print(&program));
    println!(
        "    Coherence Score: {:.4}  (target ≥ 0.618 = φ⁻¹)",
        optimizer.monitor.coherence_score
    );

    // ── Emit bytecode ─────────────────────────────────────────────
    let bytecode = emitter::emit(&program);
    println!("\n─── Bytecode (.phivm) ──────────────────────");
    println!("{}", emitter::disassemble(&bytecode));
    println!("    Raw bytes: {} total", bytecode.len());

    std::fs::write("output.phivm", &bytecode)
        .map(|_| println!("    Saved → output.phivm"))
        .unwrap_or_else(|e| println!("    (Could not write: {})", e));

    // ── Evaluate ──────────────────────────────────────────────────
    let mut evaluator = Evaluator::new(&program);
    match evaluator.run() {
        Ok(result) => {
            println!("\n─── Result ─────────────────────────────────");
            println!("    {:?}", result);
            println!("\n✓ Pipeline complete.\n");
        }
        Err(e) => {
            println!("\n✗ Runtime error: {:?}\n", e);
        }
    }
}
