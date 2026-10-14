//! Step-limit contract: `--max-steps 0` documents "infinite" and now maps to
//! usize::MAX at the CLI (was: a silent 1e9 cap — a finite ceiling labeled
//! infinite). The evaluator's own 100k safety brake applies only when
//! max_steps is None — an explicit unlimited choice must bypass it.

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::evaluator::{EvalError, Evaluator};
use phiflow::phi_ir::lowering::lower_program;

// ~150k loop iterations ≈ >400k instruction dispatches — comfortably past
// the 100k None-brake, far below any per-invocation cap.
const LOOP_PAST_BRAKE: &str = r#"
let i = 0.0
while i < 150000.0 {
    i = i + 1.0
}
i
"#;

#[test]
fn test_explicit_unlimited_bypasses_100k_brake() {
    // What --max-steps 0 now sets: Some(usize::MAX). The run must complete —
    // before the fix this would have silently died at 1e9, and if we'd naively
    // used None it would die at 100k via the safety brake.
    let exprs = parse_phi_program(LOOP_PAST_BRAKE).expect("parse failed");
    let program = lower_program(&exprs);
    let mut eval = Evaluator::new(program);
    eval.max_steps = Some(usize::MAX);
    let result = eval.run().expect("usize::MAX cap must not trip any brake");
    assert_eq!(
        result,
        phiflow::phi_ir::PhiIRValue::Number(150000.0),
        "loop must run to completion"
    );
}

#[test]
fn test_none_still_brakes_at_100k() {
    // The evaluator's safety brake is preserved for truly-unset limits:
    // max_steps=None in a single run_or_yield invocation dies at 100k steps.
    // This guards genuinely runaway programs that never asked for unlimited.
    let exprs = parse_phi_program(LOOP_PAST_BRAKE).expect("parse failed");
    let program = lower_program(&exprs);
    let mut eval = Evaluator::new(program); // max_steps: None by default
    let err = eval.run().expect_err("None must brake at 100k");
    assert!(
        matches!(err, EvalError::StepLimitExceeded(_)),
        "expected StepLimitExceeded, got {err:?}"
    );
}

#[test]
fn test_bounded_cap_still_enforced() {
    // Sanity: an explicit finite cap still fires at its bound.
    let exprs = parse_phi_program(LOOP_PAST_BRAKE).expect("parse failed");
    let program = lower_program(&exprs);
    let mut eval = Evaluator::new(program);
    eval.max_steps = Some(5_000);
    let err = eval.run().expect_err("5000-step cap must fire");
    assert!(matches!(err, EvalError::StepLimitExceeded(5_000)));
}
