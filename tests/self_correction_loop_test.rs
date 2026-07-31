//! C-25: Self-correction loop end-to-end test.
//!
//! Verifies the full loop: detect low coherence → generate correction →
//! execute correction through Evaluator → re-measure → verify improvement.
//!
//! The hardware feedback is simulated (mock counts), but the
//! detect → correct → execute → re-measure chain is real.

use phiflow::quantum_feedback::{
    calculate_coherence, generate_correction_if_needed, poll_ibm_job_mock,
    run_self_correction_loop, MockMode,
};
use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    evaluator::Evaluator,
    lowering::lower_program_checked,
    optimizer::{OptimizationLevel, Optimizer},
};

const PHI_INV: f64 = 0.618033988749895;

#[test]
fn test_decoherent_mock_produces_low_coherence() {
    let counts = poll_ibm_job_mock(MockMode::Decoherent);
    let coherence = calculate_coherence(&counts);
    assert!(
        coherence < PHI_INV,
        "Decoherent mock should produce coherence below φ⁻¹ (got {})",
        coherence
    );
    assert_eq!(coherence, 0.0, "Decoherent mock (01/10 split) should give exactly 0.0");
}

#[test]
fn test_coherent_mock_produces_high_coherence() {
    let counts = poll_ibm_job_mock(MockMode::Coherent);
    let coherence = calculate_coherence(&counts);
    assert!(
        coherence >= PHI_INV,
        "Coherent mock should produce coherence above φ⁻¹ (got {})",
        coherence
    );
    assert_eq!(coherence, 1.0, "Coherent mock (00/11 Bell split) should give exactly 1.0");
}

#[test]
fn test_correction_generated_for_low_coherence() {
    let correction = generate_correction_if_needed(0.0);
    assert!(correction.is_some(), "Should generate correction for coherence 0.0");

    let code = correction.unwrap();
    assert!(
        code.contains("intention"),
        "Correction should use intention construct"
    );
    assert!(
        code.contains("resonate"),
        "Correction should use resonate to stabilize"
    );
    assert!(
        code.contains("witness"),
        "Correction should witness to record the event"
    );
}

#[test]
fn test_no_correction_for_high_coherence() {
    let correction = generate_correction_if_needed(0.9);
    assert!(correction.is_none(), "Should not generate correction for high coherence");
}

#[test]
fn test_correction_code_parses_and_lowers() {
    let correction = generate_correction_if_needed(0.0).unwrap();
    let exprs = parse_phi_program(&correction).expect("Correction should parse");
    let prog = lower_program_checked(&exprs).expect("Correction should lower");
    assert!(
        !prog.blocks.is_empty(),
        "Correction should produce at least one IR block"
    );
}

#[test]
fn test_correction_executes_without_error() {
    let correction = generate_correction_if_needed(0.0).unwrap();
    let exprs = parse_phi_program(&correction).expect("Correction should parse");
    let mut prog = lower_program_checked(&exprs).expect("Correction should lower");
    let mut opt = Optimizer::new(OptimizationLevel::Basic);
    opt.optimize(&mut prog);
    let mut eval = Evaluator::new(prog);
    eval.run().expect("Correction should execute without error");
}

#[test]
fn test_self_correction_loop_closes() {
    // Run the full self-correction loop
    let result = run_self_correction_loop();

    // 1. Initial coherence should be below threshold (decoherent)
    assert!(
        result.initial_coherence < PHI_INV,
        "Initial coherence should be below φ⁻¹ (got {})",
        result.initial_coherence
    );

    // 2. Correction should have been generated
    assert!(
        result.correction_source.is_some(),
        "Correction source should be generated"
    );

    // 3. Correction should have executed successfully
    assert!(
        result.correction_executed,
        "Correction should execute without error"
    );

    // 4. Final coherence should be above threshold (coherent after correction)
    assert!(
        result.final_coherence >= PHI_INV,
        "Final coherence should be above φ⁻¹ (got {})",
        result.final_coherence
    );

    // 5. Coherence should have improved
    assert!(
        result.improved(),
        "Self-correction loop should improve coherence above threshold"
    );

    // 6. The improvement should be significant (from 0.0 to 1.0)
    assert!(
        result.delta() > 0.5,
        "Improvement should be significant (got {})",
        result.delta()
    );
}
