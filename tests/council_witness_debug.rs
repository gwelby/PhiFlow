//! Debug test: inspect witness_log from quantum_council.phi

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program_checked;

#[test]
fn debug_council_witness_log() {
    let source = std::fs::read_to_string("examples/quantum_council.phi")
        .expect("council file not found");

    let ast = parse_phi_program(&source).expect("parse failed");
    let program = lower_program_checked(&ast).expect("lower failed");

    let mut evaluator = Evaluator::new(program);
    let _ = evaluator.run();

    let frozen = evaluator.freeze_state();

    println!("\nWitness log entries: {}", frozen.witness_log.len());
    for (i, event) in frozen.witness_log.iter().enumerate() {
        println!(
            "  [{}] intention_stack={:?} coherence={:.4} agent_name={:?}",
            i,
            event.intention_stack,
            event.coherence,
            event.agent_name
        );
    }
}
